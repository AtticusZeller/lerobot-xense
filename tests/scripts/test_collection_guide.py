from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, current_thread
from time import monotonic
from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.mock_robot import MockRobotConfig
from lerobot.scripts import collection_guide as guide_mod, lerobot_record as record_mod
from lerobot.teleoperators.mock_teleop import MockTeleopConfig
from lerobot.utils.collection_progress import (
    CollectionPlan,
    PlanBusyError,
    PlanTarget,
    json_bytes,
    load_plan,
    plan_session,
)
from tests.scripts.conftest_teleop_doubles import FakeTeleop, fresh_events

SERIAL = "0000123"
WIDTH, HEIGHT = 64, 48
REPO_ID = "test/collection"
HEAD = "observation.images.head"
FRAME = np.full((HEIGHT, WIDTH, 3), 100, dtype=np.uint8)
TARGET: PlanTarget = {
    "kind": "placement",
    "object": "iphone",
    "index": 1,
    "arm": "left",
    "x_m": 0.0,
    "y_m": 0.0,
    "z_m": 0.0,
    "yaw_deg": 0.0,
    "yaw_symmetry_deg": 180,
    "pixel": [18, 18],
    "arrow_end_pixel": [32, 18],
    "outline": [[12, 12], [25, 12], [25, 25], [12, 25]],
    "outline_plane_z_m": 0.0,
    "reason": None,
}


def make_plan(n: int = 2) -> CollectionPlan:
    sha = "b" * 64
    return {
        "format": "lerobot-collection-plan",
        "version": 1,
        "plan_id": "00000000-0000-0000-0000-000000000001",
        "generation": 1,
        "next_round_id": n + 1,
        "repo_id": REPO_ID,
        "revision": "local",
        "source_episode_count": 1,
        "camera": HEAD,
        "camera_serial": SERIAL,
        "image_size": [WIDTH, HEIGHT],
        "calibration_sha256": sha,
        "mask_bundle_sha256": sha,
        "settings": {
            "round_count": n,
            "objects_per_round": 1,
            "seed": 0,
            "min_revisit_distance_m": 0.2,
            "region_source": "test",
        },
        "excluded_objects": {},
        "rounds": [
            {"id": i + 1, "mask_bundle_sha256": sha, "status": "pending", "episode_index": None, "targets": [TARGET]}
            for i in range(n)
        ],
    }


class FakeRobot:
    name = robot_type = "mock_robot"
    action_features = {"tcp.x": float}
    observation_features = {**action_features, "head": (HEIGHT, WIDTH, 3)}
    is_connected = False

    def __init__(self):
        self.cameras = {"head": SimpleNamespace(native_rgb_identity=lambda: (SERIAL, (WIDTH, HEIGHT)))}
        self.observed = 0
        self.actions = 0
        self.on_observe = lambda: None

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_observation(self):
        self.observed += 1
        self.on_observe()
        return {"tcp.x": 0.12, "head": FRAME}

    def send_action(self, action):
        self.actions += 1
        return action


class SessionTeleop(FakeTeleop):
    is_connected = False

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False


@pytest.fixture
def record_setup(tmp_path, monkeypatch):
    robot, teleop, events = FakeRobot(), SessionTeleop(), fresh_events()
    config = record_mod.RecordConfig(
        robot=MockRobotConfig(),
        teleop=MockTeleopConfig(),
        resume=True,
        play_sounds=False,
        collection_plan=tmp_path / "plan.json",
        dataset=record_mod.DatasetRecordConfig(
            repo_id=REPO_ID,
            root=tmp_path / "dataset",
            single_task="collect",
            fps=100,
            video=False,
            vcodec="h264",
            push_to_hub=False,
            episode_time_s=0.025,
            reset_time_s=0,
            num_episodes=4,
        ),
    )
    assert config.collection_plan is not None
    config.collection_plan.write_bytes(json_bytes(make_plan()))
    ds = LeRobotDataset.create(
        REPO_ID,
        config.dataset.fps,
        root=config.dataset.root,
        robot_type=robot.name,
        features={
            "action": {"dtype": "float32", "shape": (1,), "names": ["tcp.x"]},
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["tcp.x"]},
            HEAD: {"dtype": "image", "shape": FRAME.shape, "names": ["height", "width", "channels"]},
        },
        use_videos=False,
        vcodec="h264",
    )
    ds.add_frame(
        {
            "observation.state": np.array([0], np.float32),
            "action": np.array([0], np.float32),
            HEAD: FRAME,
            "task": "seed",
        }
    )
    ds.save_episode()
    ds.finalize()
    window = SimpleNamespace(open=False, opens=0, frames=0, key=lambda: 13)

    def show(*args):
        assert window.open
        window.frames += 1

    def open_window(*args):
        assert not window.open
        window.open = True
        window.opens += 1
        window.frames = 0

    def close_window(*args):
        assert window.open
        window.open = False

    def no_evaluator(*args):
        raise RuntimeError("CUDA unavailable for test")

    monkeypatch.setattr(record_mod, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(record_mod, "make_teleoperator_from_config", lambda _: teleop)
    monkeypatch.setattr(record_mod, "init_keyboard_listener", lambda **_: (None, events))
    monkeypatch.setattr(record_mod, "is_headless", lambda: False)
    monkeypatch.setattr(record_mod, "require_gui", lambda: None)
    monkeypatch.setattr(guide_mod, "open_evaluator", no_evaluator)
    monkeypatch.setattr(cv2, "namedWindow", open_window)
    monkeypatch.setattr(cv2, "destroyWindow", close_window)
    monkeypatch.setattr(cv2, "imshow", show)
    monkeypatch.setattr(cv2, "waitKey", lambda _: window.key())
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_: float(window.open))
    return config, robot, events, window


def assert_progress(path, expected):
    plan = load_plan(path)
    assert [(r["status"], r["episode_index"]) for r in plan["rounds"]] == expected
    for round_ in plan["rounds"]:
        round_.update(status="pending", episode_index=None)
    assert plan == make_plan()


def read_episode_indices(root):
    return [
        pq.read_table(f, columns=["episode_index"]).column("episode_index").to_pylist()
        for f in sorted(root.rglob("*.parquet"))
    ]


def test_resumed_rounds_retakes_and_live_readable_parquet(record_setup):
    cfg, robot, events, window = record_setup
    retaken = False
    seen = []

    def key():
        if window.frames == 1:
            seen.append(next(r["id"] for r in load_plan(cfg.collection_plan)["rounds"] if r["status"] == "pending"))
            with pytest.raises(PlanBusyError), plan_session(cfg.collection_plan):
                pass
            if window.opens == 3:
                assert read_episode_indices(cfg.dataset.root / "meta/episodes") == [[0], [1]]
                assert set(read_episode_indices(cfg.dataset.root / "data")[1]) == {1}
            events["exit_early"] = True
            return 83
        return 13

    def observe():
        nonlocal retaken
        if not window.open and not retaken:
            events.update(exit_early=True, rerecord_episode=True)
            retaken = True

    robot.on_observe, window.key = observe, key
    dataset = record_mod.record(cfg)
    assert seen == [1, 1, 2]
    assert robot.observed == robot.actions
    assert not window.open and not robot.is_connected
    assert dataset.num_episodes == 3
    assert_progress(cfg.collection_plan, [("collected", 1), ("collected", 2)])
    assert read_episode_indices(dataset.root / "meta/episodes") == [[0], [1], [2]]
    assert [set(rows) for rows in read_episode_indices(dataset.root / "data")] == [{0}, {1}, {2}]
    record_mod.record(cfg)
    assert window.opens == 3


@pytest.mark.parametrize("during_record", [False, True])
def test_escape_keeps_original_record_behavior(record_setup, during_record):
    cfg, robot, events, window = record_setup
    if during_record:

        def observe():
            if not window.open:
                events.update(exit_early=True, stop_recording=True)

        robot.on_observe = observe
    else:
        window.key = lambda: 27
    ds = record_mod.record(cfg)
    assert ds.num_episodes == (2 if during_record else 1)
    assert_progress(cfg.collection_plan, [("collected", 1) if during_record else ("pending", None), ("pending", None)])
    assert window.opens == 1 and not window.open


@pytest.mark.parametrize("stage", ["save_episode", "close_episode_parquet"])
def test_failed_save_or_footer_never_marks_progress(record_setup, monkeypatch, stage):
    cfg, robot, _, window = record_setup

    def fail(_):
        raise OSError("write failed")

    monkeypatch.setattr(LeRobotDataset, stage, fail)
    with pytest.raises(OSError, match="write failed"):
        record_mod.record(cfg)
    assert_progress(cfg.collection_plan, [("pending", None), ("pending", None)])
    assert not window.open and not robot.is_connected


@pytest.mark.parametrize(
    "kind", ["serial", "profile", "frame", "revision", "count", "exhausted_count", "malformed", "thresholds"]
)
def test_refuses_mismatches_without_writing(record_setup, kind):
    cfg, robot, _, window = record_setup
    plan = make_plan()
    if kind == "serial":
        robot.cameras["head"].native_rgb_identity = lambda: ("123", (WIDTH, HEIGHT))
    elif kind == "profile":
        robot.cameras["head"].native_rgb_identity = lambda: (SERIAL, (WIDTH + 1, HEIGHT))
    elif kind == "frame":
        robot.get_observation = lambda: {"tcp.x": 0.0, "head": FRAME[:, : WIDTH - 1]}
    elif kind == "revision":
        plan["revision"] = "a" * 40
    elif kind == "count":
        plan["source_episode_count"] = 2
    elif kind == "exhausted_count":
        for i, round_ in enumerate(plan["rounds"], 1):
            round_.update(status="collected", episode_index=i)
    elif kind == "thresholds":
        cfg.collection_alignment_thresholds = cfg.collection_plan.with_name("thresholds.json")
        cfg.collection_alignment_thresholds.write_text('{"version": 1, "min_iou": 2}')
    plan_bytes = b"{" if kind == "malformed" else json_bytes(plan)
    cfg.collection_plan.write_bytes(plan_bytes)
    with pytest.raises(ValueError):
        record_mod.record(cfg)
    assert cfg.collection_plan.read_bytes() == plan_bytes
    assert read_episode_indices(cfg.dataset.root / "meta/episodes") == [[0]]
    assert not window.open and not robot.is_connected


def test_local_only_does_not_download(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("attempted to talk to Hub")

    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.get_safe_version", fail)
    with pytest.raises(FileNotFoundError):
        LeRobotDataset(REPO_ID, root=tmp_path, local_files_only=True, vcodec="h264")


def test_unplanned_recording_still_aggregates_shards(record_setup):
    cfg, _, _, window = record_setup
    cfg.collection_plan = None
    cfg.dataset.num_episodes = 2
    ds = record_mod.record(cfg)
    assert not window.opens
    assert read_episode_indices(ds.root / "meta/episodes") == [[0], [1, 2]]
    assert [set(rows) for rows in read_episode_indices(ds.root / "data")] == [{0}, {1, 2}]


def test_confirm_drains_inference_without_blocking_gui(record_setup, monkeypatch):
    _, _, _, window = record_setup
    running, release = Event(), Event()
    inputs = []

    class Evaluator:
        def prepare_round(self, plan, targets):
            assert current_thread().name.startswith("test-alignment")
            return targets

        def evaluate(self, rgb, prepared):
            inputs.append(rgb)
            running.set()
            assert release.wait(3)
            return ()

    monkeypatch.setattr(guide_mod, "open_evaluator", lambda _: Evaluator())
    plan = make_plan(1)
    observation = {"head": FRAME.copy()}
    with (
        ThreadPoolExecutor(1, thread_name_prefix="test-alignment") as executor,
        guide_mod.CollectionGuide(executor, None, False).reset(plan, plan["rounds"][0]) as reset,
    ):
        window.key = lambda: -1
        assert reset.tick(observation)
        assert running.wait(3)
        window.key = lambda: 13
        try:
            assert reset.tick(observation)
            assert reset.confirmed and not reset.accepted
            for _ in range(3):
                assert reset.tick(observation)
            assert len(inputs) == 1 and not np.shares_memory(inputs[0], observation["head"])
        finally:
            release.set()
        assert reset.future is not None
        reset.future.result(timeout=3)
        assert reset.tick(observation, robot_moving=True)
        assert not reset.tick(observation)
        assert reset.accepted
    assert not window.open


def test_manual_box_round_never_starts_inference(record_setup, monkeypatch):
    _, _, _, window = record_setup
    plan = make_plan(1)
    plan["rounds"][0]["targets"] = [
        {**TARGET, "object": "box_base", "outline": None, "outline_plane_z_m": None, "reason": "no reliable outline"}
    ]
    window.key = lambda: -1

    def fail(*_args):
        raise AssertionError("manual-only reset submitted inference")

    with ThreadPoolExecutor(1) as executor:
        monkeypatch.setattr(executor, "submit", fail)
        with guide_mod.CollectionGuide(executor, None, True).reset(plan, plan["rounds"][0]) as reset:
            assert reset.tick({"head": FRAME})
            assert not reset.announced
            window.key = lambda: 13
            assert not reset.tick({"head": FRAME}) and reset.accepted


def test_alignment_only_announces_fresh_complete_result(record_setup, monkeypatch):
    _, _, _, window = record_setup
    captured = []
    monkeypatch.setattr(guide_mod, "log_say", lambda *args: captured.append(args))
    plan = make_plan(1)

    @dataclass
    class Match:
        passed: bool = True
        status: str = "aligned"
        iou: float = 1.0
        observed_outline: list[list[float]] | None = None

    with ThreadPoolExecutor(1) as executor:
        guide = guide_mod.CollectionGuide(executor, None, True)
        guide.unavailable = "test: no submission"
        with guide.reset(plan, plan["rounds"][0]) as reset:
            assert not reset.results
            reset.results = (Match(observed_outline=TARGET["outline"]),)
            window.key = lambda: -1
            assert reset.tick({"head": FRAME})
            assert not captured
            reset.result_time = monotonic()
            assert reset.tick({"head": FRAME})
            assert reset.tick({"head": FRAME})
            assert captured == [("All collection targets aligned", True)]
        with guide.reset(plan, plan["rounds"][0]) as reset:
            assert not reset.results
            assert reset.tick({"head": FRAME})
            assert len(captured) == 1
