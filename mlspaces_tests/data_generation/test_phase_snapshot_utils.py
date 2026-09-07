import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from molmo_spaces.data_generation import pipeline
from molmo_spaces.data_generation.phase_snapshot_utils import (
    PhaseSnapshotCollector,
    discard_camera_frames,
    save_phase_snapshot_episode,
)
from molmo_spaces.data_generation.pipeline import save_house_trajectories

PHASES = {
    "unknown": 0,
    "gripper-open": 1,
    "pregrasp": 2,
    "grasp": 3,
    "gripper-close": 4,
    "lift": 5,
    "preplace": 6,
    "place": 7,
    "retreat": 8,
    "go_home": 9,
}


def _observation(phase: str, frame_index: int) -> list[dict]:
    return [
        {
            "policy_phase": PHASES[phase],
            "wrist_camera": np.full((3, 4, 3), frame_index, dtype=np.uint8),
            "exo_camera_1": np.full((3, 4, 3), frame_index + 10, dtype=np.uint8),
            "qpos": np.array([frame_index, frame_index + 1], dtype=np.float32),
            "sensor_param_wrist_camera": {"intrinsic_cv": np.eye(3, dtype=np.float32)},
        }
    ]


def test_phase_collector_requires_every_phase_and_keeps_views_synchronized():
    collector = PhaseSnapshotCollector(
        phase_name_to_id=PHASES,
        camera_names=("wrist_camera", "exo_camera_1"),
        required_phases=("pregrasp", "grasp", "lift"),
        seed=7,
    )

    for frame_index, phase in enumerate(("pregrasp", "pregrasp", "grasp", "lift")):
        collector.observe(_observation(phase, frame_index), frame_index)

    assert collector.is_complete
    data = collector.finalize(episode_seed=123)
    assert data["phase_frame_counts"] == {"pregrasp": 2, "grasp": 1, "lift": 1}
    for phase, samples in data["snapshots"].items():
        assert len(samples) == 1, phase
        sample = samples[0]
        wrist_value = int(sample["images"]["wrist_camera"][0, 0, 0])
        exo_value = int(sample["images"]["exo_camera_1"][0, 0, 0])
        assert exo_value - wrist_value == 10
        assert sample["frame_index"] == wrist_value


def test_release_is_gripper_open_only_after_place():
    collector = PhaseSnapshotCollector(
        phase_name_to_id=PHASES,
        camera_names=("wrist_camera", "exo_camera_1"),
        required_phases=("place", "release"),
        seed=11,
    )

    assert collector.observe(_observation("gripper-open", 0), 0) == "initial-gripper-open"
    assert collector.missing_phases == ("place", "release")
    collector.observe(_observation("place", 1), 1)
    assert collector.observe(_observation("gripper-open", 2), 2) == "release"
    assert collector.is_complete


def test_discard_camera_frames_preserves_phase_and_calibration():
    observation = _observation("pregrasp", 4)
    observation[0]["wrist_camera_depth"] = np.ones((3, 4), dtype=np.float32)

    discard_camera_frames(observation, ("wrist_camera", "exo_camera_1"))

    assert "wrist_camera" not in observation[0]
    assert "exo_camera_1" not in observation[0]
    assert "wrist_camera_depth" not in observation[0]
    assert observation[0]["policy_phase"] == PHASES["pregrasp"]
    assert "sensor_param_wrist_camera" in observation[0]


def test_save_phase_snapshot_episode_writes_pngs_and_metadata(tmp_path):
    collector = PhaseSnapshotCollector(
        phase_name_to_id=PHASES,
        camera_names=("wrist_camera", "exo_camera_1"),
        required_phases=("pregrasp",),
        seed=3,
    )
    collector.observe(
        _observation("pregrasp", 6),
        6,
        snapshot_payload_factory=lambda: {
            "points": {
                "sampling_method": "kubric",
                "track_ids": np.arange(2, dtype=np.int32),
                "body_ids": np.array([1, 2], dtype=np.int32),
                "cameras": {
                    camera_name: {
                        "points_2d": np.array([[0.5, 0.5], [1.5, 1.5]]),
                        "visibility": np.ones(2, dtype=np.float32),
                    }
                    for camera_name in ("wrist_camera", "exo_camera_1")
                },
            }
        },
    )

    summary = save_phase_snapshot_episode(
        collector.finalize(episode_seed=99),
        output_dir=tmp_path,
        episode_index=0,
        scene_metadata={"object_name": "cup", "place_receptacle_name": "table"},
    )

    metadata_path = tmp_path / summary["metadata"]
    metadata = json.loads(metadata_path.read_text())
    assert summary["num_image_sets"] == 1
    assert metadata["scene"]["object_name"] == "cup"
    snapshot = metadata["snapshots"][0]
    assert snapshot["phase"] == "pregrasp"
    assert snapshot["frame_index"] == 6
    for relative_path in snapshot["images"].values():
        image_path = metadata_path.parent / relative_path
        assert image_path.is_file()
        assert Image.open(image_path).size == (4, 3)
    point_path = metadata_path.parent / snapshot["points"]["path"]
    with np.load(point_path) as point_data:
        np.testing.assert_array_equal(point_data["track_ids"], np.arange(2))
        assert "wrist_camera_points_2d" in point_data
        assert "exo_camera_1_visibility" in point_data


def test_snapshot_only_house_save_writes_manifest_without_trajectory(tmp_path):
    collector = PhaseSnapshotCollector(
        phase_name_to_id=PHASES,
        camera_names=("wrist_camera", "exo_camera_1"),
        required_phases=("pregrasp",),
        seed=5,
    )
    collector.observe(_observation("pregrasp", 2), 2)
    episode_info = {
        "history": {
            "observations": [],
            "obs_scene": {"object_name": "mug", "policy_phases": PHASES},
        },
        "sensor_suite": None,
        "phase_snapshot_data": collector.finalize(episode_seed=17),
    }
    config = SimpleNamespace(
        point_tracks_only=False,
        generate_phase_snapshots=True,
        phase_snapshots_only=True,
    )
    house_dir = tmp_path / "house_12"

    save_house_trajectories(
        logging.getLogger(__name__),
        [episode_info],
        house_dir,
        config,
        "_batch_1_of_1",
    )

    manifest_path = house_dir / "phase_snapshots_batch_1_of_1.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["num_episodes"] == 1
    assert manifest["episodes"][0]["num_image_sets"] == 1
    assert not (house_dir / "trajectories_batch_1_of_1.h5").exists()
    assert not list(house_dir.glob("*.mp4"))


def test_snapshot_batches_keep_distinct_images_and_metadata(tmp_path):
    config = SimpleNamespace(
        point_tracks_only=False,
        generate_phase_snapshots=True,
        phase_snapshots_only=True,
    )
    for batch in (1, 2):
        collector = PhaseSnapshotCollector(
            phase_name_to_id=PHASES,
            camera_names=("wrist_camera", "exo_camera_1"),
            required_phases=("pregrasp",),
        )
        collector.observe(_observation("pregrasp", batch), batch)
        episode = {
            "history": {"obs_scene": {}},
            "phase_snapshot_data": collector.finalize(episode_seed=batch),
        }
        save_house_trajectories(
            logging.getLogger(__name__),
            [episode],
            tmp_path,
            config,
            f"_batch_{batch}_of_2",
            batch_num=batch,
            total_batches=2,
        )

    for batch in (1, 2):
        manifest = json.loads((tmp_path / f"phase_snapshots_batch_{batch}_of_2.json").read_text())
        metadata_path = tmp_path / manifest["episodes"][0]["metadata"]
        metadata = json.loads(metadata_path.read_text())
        assert metadata["episode_seed"] == batch
        image_path = metadata_path.parent / metadata["snapshots"][0]["images"]["wrist_camera"]
        with Image.open(image_path) as image:
            np.testing.assert_array_equal(np.asarray(image), batch)


@pytest.mark.parametrize("require_missing_phase", [False, True])
def test_snapshot_rollout_synchronizes_payloads_and_rejects_missing_phases(
    monkeypatch, require_missing_phase
):
    phases = ("pregrasp", "pregrasp", "pregrasp", "place", "gripper-open")
    observations = []
    task = SimpleNamespace(frame=0, env=SimpleNamespace())

    def observation():
        obs = _observation(phases[task.frame], task.frame)
        observations.append(obs)
        return obs

    def step_chunk(actions, stop_on_success=False):
        task.frame += 1
        return observation(), 0, task.is_done(), False, [{}]

    def get_action_chunk(obs):
        # Camera arrays must remain available until the policy consumes them.
        assert int(obs[0]["wrist_camera"][0, 0, 0]) == task.frame
        return [0]

    def sample_points(env, cameras, width, height, **kwargs):
        assert env is task.env
        assert cameras == ("wrist_camera", "exo_camera_1")
        assert (width, height) == (4, 3)
        return {"frame": task.frame, "seed": kwargs["seed"]}

    task.reset = lambda: (observation(), {})
    task.step_chunk = step_chunk
    task.is_done = lambda: task.frame == len(phases) - 1
    task.judge_success = lambda: True
    policy = SimpleNamespace(get_all_phases=lambda: PHASES, get_action_chunk=get_action_chunk)
    defaults = {
        name: field.default
        for name, field in pipeline.MlSpacesExpConfig.model_fields.items()
        if name.startswith("phase_snapshot")
    }
    defaults.update(
        generate_phase_snapshots=True,
        phase_snapshots_only=True,
        phase_snapshot_generate_points=True,
        phase_snapshot_required_phases=("pregrasp", "place", "release")
        + (("lift",) if require_missing_phase else ()),
        camera_config=SimpleNamespace(
            cameras=[SimpleNamespace(name=name) for name in ("wrist_camera", "exo_camera_1")],
            img_resolution=(4, 3),
        ),
    )
    monkeypatch.setattr(pipeline, "sample_aligned_kubric_points_for_frame", sample_points)
    success = pipeline.ParallelRolloutRunner.run_single_rollout(
        episode_seed=17, task=task, policy=policy, exp_config=SimpleNamespace(**defaults)
    )

    assert success == (not require_missing_phase)
    saved = task._phase_snapshot_data
    assert saved["missing_phases"] == (["lift"] if require_missing_phase else [])
    for samples in saved["snapshots"].values():
        for sample in samples:
            frame = sample["frame_index"]
            assert sample["points"] == {"frame": frame, "seed": 17 + frame * 104729}
            np.testing.assert_array_equal(sample["images"]["wrist_camera"], frame)
    assert all(
        "wrist_camera" not in obs[0] and "exo_camera_1" not in obs[0] for obs in observations
    )
