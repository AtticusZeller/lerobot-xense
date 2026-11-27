from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, TypedDict, cast
from uuid import UUID


class PlanTarget(TypedDict):
    kind: Literal["placement"]
    object: str
    index: int
    arm: str
    x_m: float
    y_m: float
    z_m: float
    yaw_deg: float
    yaw_symmetry_deg: float
    outline_plane_z_m: float | None
    pixel: list[float]
    arrow_end_pixel: list[float]
    outline: list[list[float]] | None
    reason: str | None


class PlanRound(TypedDict):
    id: int
    mask_bundle_sha256: str
    targets: list[PlanTarget]
    status: Literal["pending", "collected", "skipped"]
    episode_index: int | None


class PlanSettings(TypedDict):
    round_count: int
    objects_per_round: int
    seed: int
    min_revisit_distance_m: float
    region_source: str


class CollectionPlan(TypedDict):
    format: Literal["lerobot-collection-plan"]
    version: Literal[1]
    plan_id: str
    generation: int
    next_round_id: int
    repo_id: str
    revision: str
    source_episode_count: int
    camera: str
    image_size: list[int]
    camera_serial: str | None
    calibration_sha256: str
    mask_bundle_sha256: str
    settings: PlanSettings
    excluded_objects: dict[str, str]
    rounds: list[PlanRound]


class PlanBusyError(ValueError):
    pass


def _require(condition: object, field: str) -> None:
    if not condition:
        raise ValueError(f"Invalid collection plan: {field}")


def _int(value: object, minimum: int = 1) -> bool:
    return type(value) is int and minimum <= value <= 2**53 - 1


def _number(value: Any) -> bool:
    return type(value) in (int, float) and -float("inf") < value < float("inf")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _point(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(map(_number, value))


def _in_frame(point: list[float], size: list[int]) -> bool:
    return all(0 <= coordinate < bound for coordinate, bound in zip(point, size, strict=True))


def _validate_target(target: Any, size: list[int], multi: bool) -> None:
    _require(isinstance(target, dict) and target.get("kind") == "placement", "target kind")
    for key in ("object", "arm"):
        _require(_text(target.get(key)), f"target {key}")
    _require(_int(target.get("index")), "target index")
    for key in ("x_m", "y_m", "z_m", "yaw_deg", "yaw_symmetry_deg"):
        _require(_number(target.get(key)), f"target {key}")
    _require(0 < target["yaw_symmetry_deg"] <= 360, "target yaw symmetry")
    _require(0 <= target["yaw_deg"] < target["yaw_symmetry_deg"], "target folded yaw")
    _require(_point(target.get("pixel")) and _in_frame(target["pixel"], size), "target pixel")
    _require(_point(target.get("arrow_end_pixel")), "arrow endpoint")
    _require({"outline", "reason", "outline_plane_z_m"} <= target.keys(), "outline fields")
    outline, reason, plane = (target[k] for k in ("outline", "reason", "outline_plane_z_m"))
    if outline is None:
        _require(not multi and _text(reason) and plane is None, "unprojectable target")
    else:
        _require(
            isinstance(outline, list)
            and len(outline) >= 3
            and all(_point(p) and _in_frame(p, size) for p in outline)
            and reason is None
            and _number(plane),
            "target outline",
        )


def validate_plan(plan: Any) -> CollectionPlan:
    _require(isinstance(plan, dict), "expected an object")
    _require(
        plan.get("format") == "lerobot-collection-plan"
        and _int(plan.get("version"))
        and plan["version"] == 1,
        "unknown format or version",
    )
    try:
        _require(str(UUID(plan["plan_id"])) == plan["plan_id"], "plan_id")
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Invalid collection plan: plan_id") from exc
    for key in ("generation", "next_round_id"):
        _require(_int(plan.get(key)), key)
    for key in ("repo_id", "revision", "camera"):
        _require(_text(plan.get(key)), key)
    _require(_int(plan.get("source_episode_count"), 0), "source_episode_count")
    _require("camera_serial" in plan, "camera_serial")
    _require(plan["camera_serial"] is None or _text(plan["camera_serial"]), "camera_serial")
    for key in ("calibration_sha256", "mask_bundle_sha256"):
        _require(_sha(plan.get(key)), key)
    size = plan.get("image_size")
    _require(isinstance(size, list) and len(size) == 2 and all(map(_int, size)), "image_size")
    settings = plan.get("settings")
    _require(isinstance(settings, dict), "settings")
    for key in ("round_count", "objects_per_round"):
        _require(_int(settings.get(key)), f"settings.{key}")
    _require(_int(settings.get("seed"), 0), "settings.seed")
    distance = settings.get("min_revisit_distance_m")
    _require(_number(distance) and distance > 0, "settings.min_revisit_distance_m")
    _require(_text(settings.get("region_source")), "settings.region_source")
    excluded = plan.get("excluded_objects")
    _require(
        isinstance(excluded, dict) and all(_text(k) and _text(v) for k, v in excluded.items()),
        "excluded_objects",
    )
    _require(isinstance(plan.get("rounds"), list), "rounds")
    ids: set[int] = set()
    episodes: set[int] = set()
    for round_ in plan["rounds"]:
        _require(isinstance(round_, dict), "round")
        id_ = round_.get("id")
        _require(_int(id_) and id_ < plan["next_round_id"] and id_ not in ids, "round id")
        ids.add(id_)
        _require(_sha(round_.get("mask_bundle_sha256")), "round mask_bundle_sha256")
        status = round_.get("status")
        _require(status in ("pending", "collected", "skipped"), "round status")
        _require("episode_index" in round_, "round episode_index")
        episode = round_["episode_index"]
        if status == "collected":
            _require(
                _int(episode, plan["source_episode_count"]) and episode not in episodes,
                "collected episode_index (must be new and unique)",
            )
            episodes.add(episode)
        else:
            _require(episode is None, "non-collected episode_index")
        targets = round_.get("targets")
        _require(isinstance(targets, list) and targets, "round targets")
        if status == "pending":
            _require(len(targets) == settings["objects_per_round"], "pending round target count")
        for target in targets:
            _validate_target(target, size, multi=len(targets) > 1)
        _require(len({target["object"] for target in targets}) == len(targets), "duplicate object")
    return cast("CollectionPlan", plan)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    _require(len(result) == len(pairs), "duplicate JSON field")
    return result


def _finite_float(token: str) -> float:
    value = float(token)
    _require(math.isfinite(value), "non-finite JSON number")
    return value


def load_plan(path: Path) -> CollectionPlan:
    with path.open(encoding="utf-8") as stream:
        try:
            payload = json.load(
                stream,
                object_pairs_hook=_strict_object,
                parse_float=_finite_float,
                parse_constant=_finite_float,
            )
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError(f"Invalid collection plan JSON: {path}") from exc
        return validate_plan(payload)


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
            f.close()
            os.replace(temporary, path)
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def plan_session(path: Path) -> Generator[PlanSession]:
    import fcntl

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PlanBusyError(f"Collection plan is locked: {path}") from exc
        session = PlanSession(path)
        try:
            yield session
        finally:
            session._active = False


class PlanSession:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._active = True

    def _check_active(self) -> None:
        if not self._active:
            raise RuntimeError("Collection plan session is closed")

    def read(self) -> CollectionPlan:
        self._check_active()
        return load_plan(self.path)

    def write(self, plan: CollectionPlan) -> None:
        self._check_active()
        atomic_write(self.path, json_bytes(validate_plan(plan)))

    def mark_collected(self, round_id: int, episode_index: int) -> CollectionPlan:
        return self._mark(round_id, "collected", episode_index)

    def mark_skipped(self, round_id: int) -> CollectionPlan:
        return self._mark(round_id, "skipped", None)

    def _mark(
        self, round_id: int, status: Literal["collected", "skipped"], episode_index: int | None
    ) -> CollectionPlan:
        plan = self.read()
        round_ = next((r for r in plan["rounds"] if r["id"] == round_id), None)
        if round_ is None:
            raise ValueError(f"Unknown collection round {round_id}")
        if (round_["status"], round_["episode_index"]) == (status, episode_index):
            return plan
        if round_["status"] != "pending":
            raise ValueError(f"Collection round {round_id} is already {round_['status']}")
        round_["status"], round_["episode_index"] = status, episode_index
        self.write(plan)
        return plan


def next_pending(plan: CollectionPlan) -> PlanRound | None:
    return next((r for r in plan["rounds"] if r["status"] == "pending"), None)


def validate_capture(
    plan: CollectionPlan,
    *,
    repo_id: str,
    revision: str,
    episode_count: int,
    camera: str,
    serial: str,
    image_size: tuple[int, int],
) -> None:
    if (repo_id, revision) != (plan["repo_id"], plan["revision"]):
        raise ValueError("Collection plan does not match the local dataset identity")
    if episode_count < plan["source_episode_count"] or any(
        r["episode_index"] is not None and r["episode_index"] >= episode_count
        for r in plan["rounds"]
    ):
        raise ValueError("Collection plan does not match the local dataset episode count")
    if not plan["camera_serial"] or (camera, serial, list(image_size)) != (
        plan["camera"],
        plan["camera_serial"],
        plan["image_size"],
    ):
        raise ValueError("Collection plan does not match the live head camera/profile")


def _immutable_content(plan: CollectionPlan) -> CollectionPlan:
    content = deepcopy(plan)
    for round_ in content["rounds"]:
        round_["status"] = "pending"
        round_["episode_index"] = None
    return content


def import_progress(destination: Path, returned: Path) -> CollectionPlan:
    received = load_plan(returned)
    with plan_session(destination) as session:
        current = session.read()
        if _immutable_content(current) != _immutable_content(received):
            raise ValueError("Returned collection plan does not match this plan generation")
        for local, remote in zip(current["rounds"], received["rounds"], strict=True):
            if local["status"] == "pending":
                local["status"], local["episode_index"] = remote["status"], remote["episode_index"]
            elif remote["status"] != "pending" and (local["status"], local["episode_index"]) != (
                remote["status"],
                remote["episode_index"],
            ):
                raise ValueError(f"Conflicting progress for collection round {local['id']}")
        session.write(current)
        return current
