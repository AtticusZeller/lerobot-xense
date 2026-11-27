from __future__ import annotations

import json
import os
import time
from collections.abc import Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.collection_progress import CollectionPlan, PlanRound, PlanSession, PlanTarget
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.robot_utils import get_logger
from lerobot.utils.utils import log_say

logger = get_logger("collection_guide")
WINDOW = "Guided recording - Reset the environment"
TARGET = (255, 206, 75)
ALIGNED = (125, 245, 100)
UNALIGNED = (92, 125, 255)
TEXT = (245, 245, 245)
MAX_RESULT_AGE_S = 2.0


class AlignmentResult(Protocol):
    @property
    def status(self) -> str: ...

    @property
    def passed(self) -> bool: ...

    @property
    def observed_outline(self) -> list[list[float]] | None: ...

    @property
    def iou(self) -> float | None: ...


class AlignmentEvaluator(Protocol):
    def prepare_round(self, plan: CollectionPlan, targets: Sequence[PlanTarget]) -> object: ...

    def evaluate(self, rgb: NDArray[np.uint8], prepared: object) -> Sequence[AlignmentResult]: ...


class NativeRGBCamera(Protocol):
    def native_rgb_identity(self) -> tuple[str, tuple[int, int]]: ...


def dataset_revision(root: Path) -> str:
    trees = [p.stem for p in (root / ".cache/huggingface/trees").glob("*.json") if len(p.stem) == 40]
    return trees[0] if len(trees) == 1 else "local"


def validate_thresholds(path: Path | None) -> None:
    if path is None:
        return
    try:
        config = json.loads(path.read_bytes())
        if not isinstance(config, dict) or config.get("version") != 1:
            raise ValueError("expected alignment thresholds version 1")
        iou = config["min_iou"]
        if type(iou) not in (float, int) or not 0 < iou <= 1:
            raise ValueError("min_iou must be in (0, 1]")
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid collection alignment thresholds {path}: {exc}") from exc


def open_evaluator(thresholds: Path | None) -> AlignmentEvaluator:
    torch = import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    alignment = import_module("lerobot_check.alignment")
    return cast(AlignmentEvaluator, alignment.AlignmentEvaluator.from_files(thresholds, device="cuda"))


def require_gui() -> None:
    if not cv2.currentUIFramework():
        raise RuntimeError("Guided recording requires an OpenCV GUI backend (install GUI-enabled opencv-python)")
    if os.name == "posix" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("Guided recording requires a desktop display")


def head_key(plan: CollectionPlan) -> str:
    prefix = f"{OBS_IMAGES}."
    camera = plan["camera"]
    if not camera.startswith(prefix):
        raise ValueError(f"Invalid collection plan camera feature: {camera}")
    return camera.removeprefix(prefix)


def validate_dataset(plan: CollectionPlan, repo_id: str, root: Path) -> None:
    if (repo_id, dataset_revision(root)) != (plan["repo_id"], plan["revision"]):
        raise ValueError("Collection plan does not match the local dataset identity (repo_id or baseline revision)")
    if not (root / "meta/info.json").is_file():
        raise FileNotFoundError(f"Guided recording requires an existing local dataset: {root}")
    if not plan["camera_serial"]:
        raise ValueError("Collection plan is missing a calibrated camera serial")


def validate_dataset_contents(plan: CollectionPlan, dataset: LeRobotDataset) -> None:
    count = dataset.meta.total_episodes
    if count < plan["source_episode_count"] or any(
        r["episode_index"] is not None and r["episode_index"] >= count for r in plan["rounds"]
    ):
        raise ValueError("Collection plan does not match the local dataset episode count")
    width, height = plan["image_size"]
    feature = dataset.features.get(plan["camera"])
    if feature is None or feature["dtype"] not in {"image", "video"} or tuple(feature["shape"]) != (height, width, 3):
        raise ValueError(f"Collection plan camera feature {plan['camera']} does not match the local dataset")


def validate_camera(plan: CollectionPlan, camera: NativeRGBCamera | None) -> None:
    width, height = plan["image_size"]
    if camera is None or not hasattr(camera, "native_rgb_identity"):
        raise ValueError(f"Collection plan camera {head_key(plan)} cannot report a connected native RGB profile")
    serial, size = camera.native_rgb_identity()
    if serial != plan["camera_serial"] or list(size) != plan["image_size"]:
        raise ValueError(
            f"Collection plan camera/profile mismatch: expected {plan['camera_serial']} {width}x{height}, "
            f"got {serial} {size[0]}x{size[1]}"
        )


def save_collection_episode(
    dataset: LeRobotDataset, session: PlanSession, plan: CollectionPlan, round_id: int
) -> CollectionPlan:
    buffer = dataset.episode_buffer
    assert buffer is not None
    episode_index = buffer["episode_index"]
    dataset.save_episode()
    dataset.close_episode_parquet()
    logger.info(f"Saved episode {episode_index} for collection plan {plan['plan_id']}, round {round_id}")
    try:
        updated = session.mark_collected(round_id, episode_index)
    except BaseException:
        logger.error(
            f"Episode {episode_index} saved, but progress update failed for round {round_id} "
            f"in plan {plan['plan_id']} ({session.path}). "
            "Verify the saved episode before repairing progress."
        )
        raise
    logger.info(f"Collection round {round_id} saved as episode {episode_index}; progress written")
    return updated


def is_manual(target: PlanTarget) -> bool:
    return target["outline"] is None or target["object"] == "box_base"


def _arrow_end(anchor: list[float], end: list[float], width: int, height: int) -> tuple[int, int]:
    x, y = anchor
    dx, dy = end[0] - x, end[1] - y
    tx = ((width - 1 if dx > 0 else 0) - x) / dx if dx else 1.0
    ty = ((height - 1 if dy > 0 else 0) - y) / dy if dy else 1.0
    scale = max(0.0, min(1.0, tx, ty))
    return round(x + dx * scale), round(y + dy * scale)


def _polyline(canvas: NDArray[np.uint8], outline: list[list[float]], color: tuple[int, int, int]) -> None:
    points = np.rint(outline).astype(np.int32)
    cv2.polylines(canvas, [points], True, color, 2, cv2.LINE_AA)


def draw_guide(
    rgb: NDArray[np.uint8],
    round_: PlanRound,
    progress: str,
    results: Sequence[AlignmentResult],
    status: str,
    confirmed: bool,
) -> NDArray[np.uint8]:
    height, width = rgb.shape[:2]
    targets = round_["targets"]
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    canvas = np.full((height + 100 + 22 * len(targets), max(width, 640), 3), (31, 25, 18), dtype=np.uint8)
    canvas[42 : 42 + height, :width] = image
    image = canvas[42 : 42 + height, :width]
    cv2.putText(canvas, progress, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.57, TEXT, 1, cv2.LINE_AA)

    for number, target in enumerate(targets, 1):
        manual = is_manual(target)
        result = results[number - 1] if results else None
        color = ALIGNED if result and result.passed else UNALIGNED
        if (outline := target["outline"]) is not None:
            _polyline(image, outline, TARGET)
        if result and result.observed_outline is not None:
            _polyline(image, result.observed_outline, color)
        pixel = target["pixel"]
        anchor = (round(pixel[0]), round(pixel[1]))
        cv2.arrowedLine(
            image, anchor, _arrow_end(pixel, target["arrow_end_pixel"], width, height), TARGET, 2, cv2.LINE_AA
        )
        cv2.circle(image, anchor, 4, TARGET, -1, cv2.LINE_AA)
        label_pos = (min(anchor[0] + 8, width - 18), max(anchor[1] - 7, 20))
        cv2.putText(image, str(number), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, str(number), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, TARGET, 2, cv2.LINE_AA)
        description = "manual" if manual else result.status if result else status
        iou = f"{result.iou:.0%}" if result and result.iou is not None else "--"
        cv2.putText(
            canvas,
            f"{number}. {target['object']}  IoU {iou}  |  {description}",
            (12, height + 42 + 22 * number),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color if result and not manual else TEXT,
            1,
            cv2.LINE_AA,
        )

    all_aligned = bool(results) and all(result.passed for result in results)
    hint = (
        "Finishing reset..."
        if confirmed
        else "All aligned - Enter to confirm | Esc to stop"
        if all_aligned
        else "Enter to confirm manually | Esc to stop"
    )
    cv2.putText(
        canvas,
        hint,
        (12, canvas.shape[0] - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        ALIGNED if all_aligned else TEXT,
        1,
        cv2.LINE_AA,
    )
    return canvas


class CollectionGuide:
    def __init__(self, executor: ThreadPoolExecutor, thresholds: Path | None, play_sounds: bool):
        self.executor = executor
        self.thresholds = thresholds
        self.play_sounds = play_sounds
        self.evaluator: AlignmentEvaluator | None = None
        self.unavailable: str | None = None

    @contextmanager
    def reset(self, plan: CollectionPlan, round_: PlanRound) -> Generator[ResetGuide]:
        reset = ResetGuide(self, plan, round_)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        try:
            yield reset
        finally:
            try:
                if reset.future is not None:
                    wait((reset.future,))
            finally:
                try:
                    cv2.destroyWindow(WINDOW)
                except cv2.error:
                    if not reset.window_closed:
                        raise


class ResetGuide:
    def __init__(self, guide: CollectionGuide, plan: CollectionPlan, round_: PlanRound):
        self.guide = guide
        self.plan = plan
        self.round = round_
        self.key = head_key(plan)
        self.width, self.height = plan["image_size"]
        self.manual_only = all(is_manual(target) for target in round_["targets"])
        self.confirmed = False
        self.accepted = False
        self.window_closed = False
        self.announced = False
        self.prepared: object | None = None
        self.future: Future[tuple[float, Sequence[AlignmentResult]]] | None = None
        self.results: Sequence[AlignmentResult] = ()
        self.result_time = 0.0
        self.status = "waiting for alignment"
        rounds = plan["rounds"]
        counts = [sum(item["status"] == state for item in rounds) for state in ("collected", "skipped", "pending")]
        self.progress = (
            f"Round {round_['id']} | {counts[0]}/{len(rounds)} collected, {counts[1]} skipped, {counts[2]} pending"
        )

    def _disable_alignment(self, exc: Exception) -> None:
        self.guide.unavailable = f"manual: {str(exc).splitlines()[0]}" if str(exc) else f"manual: {type(exc).__name__}"
        logger.warn(f"Automatic collection alignment unavailable: {exc}. Confirm manually.")
        self.guide.evaluator = None

    def _evaluate(self, rgb: NDArray[np.uint8], captured_at: float) -> tuple[float, Sequence[AlignmentResult]]:
        if self.guide.evaluator is None:
            try:
                self.guide.evaluator = open_evaluator(self.guide.thresholds)
            except Exception as exc:
                self._disable_alignment(exc)
                return captured_at, ()
        if self.prepared is None:
            self.prepared = self.guide.evaluator.prepare_round(self.plan, self.round["targets"])
        try:
            return captured_at, self.guide.evaluator.evaluate(rgb, self.prepared)
        except Exception as exc:
            self._disable_alignment(exc)
            return captured_at, ()

    def tick(self, observation: dict, *, robot_moving: bool = False) -> bool:
        rgb = observation.get(self.key)
        if not isinstance(rgb, np.ndarray) or rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Expected native {self.width}x{self.height} RGB uint8 frame in {self.key}")
        now = time.monotonic()
        if self.future is not None and self.future.done():
            self.result_time, self.results = self.future.result()
            self.future = None
        if self.guide.unavailable:
            self.status = self.guide.unavailable[:70]
        results = self.results if now - self.result_time <= MAX_RESULT_AGE_S else ()
        if not self.announced and not self.confirmed and results and all(result.passed for result in results):
            self.announced = True
            try:
                log_say("All collection targets aligned", self.guide.play_sounds)
            except OSError as exc:
                logger.warn(f"Could not play collection alignment prompt: {exc}")
        cv2.imshow(WINDOW, draw_guide(rgb, self.round, self.progress, results, self.status, self.confirmed))
        key = cv2.waitKey(1)
        try:
            self.window_closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            self.window_closed = True
        if self.window_closed or key == 27:
            return False
        self.confirmed |= key in (10, 13)
        if not self.confirmed and not self.manual_only and not self.guide.unavailable and self.future is None:
            self.future = self.guide.executor.submit(self._evaluate, rgb.copy(), now)
        self.accepted = self.confirmed and self.future is None and not robot_moving
        return not self.accepted
