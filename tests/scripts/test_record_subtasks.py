"""Per-frame subtask labels while recording: config, controller button, RT loop and saved dataset."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import draccus
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.scripts import lerobot_record as cli
from lerobot.teleoperators.bi_pico4.bi_pico4 import BiPico4
from tests.scripts.conftest_teleop_doubles import (
    FakeBiPico4Teleop,
    FakeDataset,
    FakeRobot,
    fresh_events,
    run_loop,
)

ROOT = Path(__file__).resolve().parents[2]
SUBTASKS = ["Open the egg carton.", "Take out the eggs one by one and place them in the egg tray."]


class SubtaskTeleop(FakeBiPico4Teleop):
    """Bi-Pico4 double whose subtask button fires (already edge-detected) on given frames."""

    def __init__(self, *, subtask_frames=()):
        super().__init__()
        self._subtask_frames = set(subtask_frames)

    def get_subtask_button(self) -> bool:
        return self._frame in self._subtask_frames


def bimanual_robot(iterations: int) -> FakeRobot:
    features = {f"{side}_{k}": float for side in ("left", "right") for k in FakeRobot(iterations=0).action_features}
    return FakeRobot(iterations=iterations, action_features=features, name="bi_elite_cs66_rt", bimanual=True)


def dataset_features(robot: FakeRobot) -> dict:
    return {
        **hw_to_dataset_features(robot.action_features, "action"),
        **hw_to_dataset_features(robot.action_features, "observation"),
    }


def record(robot, teleop, dataset, subtasks, single_task=None, iterations=None):
    with patch.object(cli, "_record_loop_sleep"), patch.object(cli, "logger"):
        run_loop(
            cli.flexiv_rizon4_rt_record_loop,
            iterations=iterations or robot._iterations,
            robot=robot,
            events=fresh_events(),
            fps=dataset.fps,
            dataset=dataset,
            teleop=teleop,
            control_time_s=float("inf"),
            single_task=single_task,
            subtasks=subtasks,
        )


def test_cursor_starts_on_first_subtask_and_stops_at_last():
    cursor = cli.SubtaskCursor(SUBTASKS)
    assert cursor.task == SUBTASKS[0]
    assert cursor.advance() is True
    assert cursor.task == SUBTASKS[1]
    assert cursor.advance() is False
    assert cursor.task == SUBTASKS[1]


def test_config_accepts_subtasks_without_single_task():
    cfg = cli.DatasetRecordConfig(repo_id="test/x", subtasks=SUBTASKS)
    assert cfg.single_task is None


@pytest.mark.parametrize("subtasks", [[], ["a", "a"]])
def test_config_rejects_empty_or_duplicate_subtasks(subtasks):
    with pytest.raises(ValueError, match="subtasks"):
        cli.DatasetRecordConfig(repo_id="test/x", subtasks=subtasks)


def test_config_still_requires_a_task_without_subtasks():
    with pytest.raises(ValueError, match="single_task"):
        cli.DatasetRecordConfig(repo_id="test/x")


@pytest.mark.parametrize(
    ("robot_type", "teleop_type"),
    [("bi_elite_cs66_rt", "pico4"), ("bi_elite_cs66_rt", None), ("so100_follower", "bi_pico4")],
)
def test_unsupported_rig_is_rejected_before_hardware(robot_type, teleop_type):
    with pytest.raises(ValueError, match="dataset.subtasks"):
        cli._check_subtask_recording_supported(robot_type, teleop_type)


def test_bi_elite_with_bi_pico4_is_supported():
    cli._check_subtask_recording_supported("bi_elite_cs66_rt", "bi_pico4")


def test_bi_pico4_subtask_button_fires_once_per_press_of_left_y():
    teleop = object.__new__(BiPico4)  # Skips __init__, which builds SDK-backed readers.
    teleop._is_connected = False
    teleop._left_pico4 = SimpleNamespace(_last_y_button=False)
    teleop._was_subtask_button_pressed = False
    presses = []
    for held in (False, True, True, True, False, True):
        teleop._left_pico4._last_y_button = held
        presses.append(teleop.get_subtask_button())
    assert presses == [False, True, False, False, False, True]


def test_rt_loop_labels_frames_from_the_press_onwards_and_stays_on_last():
    robot = bimanual_robot(iterations=8)
    dataset = FakeDataset(fps=30, features=dataset_features(robot))
    record(robot, SubtaskTeleop(subtask_frames={3, 6}), dataset, SUBTASKS)
    assert [frame["task"] for frame in dataset.frames] == [SUBTASKS[0]] * 3 + [SUBTASKS[1]] * 5


def test_rt_loop_without_subtasks_keeps_single_task():
    robot = bimanual_robot(iterations=3)
    dataset = FakeDataset(fps=30, features=dataset_features(robot))
    record(robot, SubtaskTeleop(subtask_frames={1}), dataset, None, single_task="whole task")
    assert [frame["task"] for frame in dataset.frames] == ["whole task"] * 3


def test_saved_episode_keeps_per_frame_tasks(tmp_path):
    robot = bimanual_robot(iterations=6)
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        "test/subtasks", fps=30, features=dataset_features(robot), root=root, use_videos=False
    )
    record(robot, SubtaskTeleop(subtask_frames={2}), dataset, SUBTASKS)
    dataset.save_episode()
    dataset.finalize()

    loaded = LeRobotDataset("test/subtasks", root=root)
    assert sorted(loaded.meta.tasks.index) == sorted(SUBTASKS)
    assert [loaded[i]["task"] for i in range(len(loaded))] == [SUBTASKS[0]] * 2 + [SUBTASKS[1]] * 4


def test_egg_carton_subtask_recipe_reads_subtasks_from_yaml():
    cfg = draccus.parse(
        cli.RecordConfig,
        config_path=ROOT / "recipes/record/bi_elite_cs66_rt/egg-carton-to-tray-subtask.yaml",
        args=[],
    )
    assert cfg.dataset.subtasks == SUBTASKS
    assert cfg.dataset.single_task is None
    cli._check_subtask_recording_supported(cfg.robot.type, cfg.teleop.type)


def test_record_rejects_unsupported_rig_before_connecting():
    robot = Mock()
    cfg = cli.RecordConfig(
        robot=SimpleNamespace(type="bi_elite_cs66_rt"),
        teleop=SimpleNamespace(type="pico4", ori_sensitivity=1.0),
        dataset=cli.DatasetRecordConfig(repo_id="test/x", subtasks=SUBTASKS, push_to_hub=False),
        play_sounds=False,
    )
    with (
        patch.object(cli, "make_robot_from_config", return_value=robot),
        patch.object(cli, "make_teleoperator_from_config", return_value=Mock()),
        patch.object(cli, "init_logging"),
        patch.object(cli, "logger"),
        pytest.raises(ValueError, match="dataset.subtasks"),
    ):
        cli.record(cfg)
    robot.connect.assert_not_called()
