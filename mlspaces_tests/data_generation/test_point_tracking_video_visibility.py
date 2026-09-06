"""CPU rollout tests for Kubric sampling and geometry/depth visibility in videos."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from molmo_spaces.data_generation import pipeline
from molmo_spaces.utils.point_tracking_utils import (
    RASTER_VISIBILITY_METHOD,
    save_point_tracks,
    select_kubric_candidate_indices,
    select_unambiguous_kubric_candidate_indices,
    track_points_for_frame,
)


class _MovingCamera:
    fov = 90.0

    def __init__(self, task, index):
        self.task = task
        self.index = index

    def get_pose(self):
        pose = np.eye(4)
        pose[0, 3] = self.index * 0.01 + (0.125 if self.task.frame == 5 else 0)
        return pose


class _VideoTask:
    """Real body transforms with tiny controlled rasters, no GPU/assets needed.

    The two geoms share one body, so checking a body ID instead of the exact
    geom would incorrectly label frame 2 visible. Render buffers are reused
    deliberately to catch missing copies in the history/candidate caches.
    """

    def __init__(self, camera_count):
        self.frame = 0
        self.model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <body name="object" pos="0 0 2">
              <freejoint/>
              <geom name="surface" type="box" size="0.1 0.1 0.1"/>
              <geom name="other_surface" type="sphere" size="0.05"/>
            </body>
          </worldbody>
        </mujoco>
        """)
        self.data = mujoco.MjData(self.model)
        self.surface = self.model.geom("surface").id
        self.other = self.model.geom("other_surface").id
        self.body = self.model.body("object").id
        self.cameras = {f"camera_{i}": _MovingCamera(self, i) for i in range(camera_count)}
        self.geom_buffer = np.zeros((8, 8, 3), dtype=np.int32)
        self.depth_buffer = np.zeros((8, 8), dtype=np.float32)
        self.segmentation_calls = []
        self.env = SimpleNamespace(
            mj_model=self.model,
            current_model=self.model,
            current_data=self.data,
            camera_manager=SimpleNamespace(registry=self.cameras),
            render_depth_frame=self.render_depth,
            render_segmentation_frame=self.render_segmentation,
        )

    def render_depth(self, camera):
        self.depth_buffer.fill(1.975 if self.frame == 1 else 2.0)
        return self.depth_buffer

    def render_segmentation(self, camera):
        self.segmentation_calls.append((camera, self.frame))
        self.geom_buffer[:, :, 0] = self.other if self.frame in (1, 2) else self.surface
        self.geom_buffer[:, :, 1] = mujoco.mjtObj.mjOBJ_GEOM.value
        self.geom_buffer[:, :, 2] = self.body
        return self.geom_buffer

    def reset(self):
        self.frame = 0
        self._update_body()
        return {}, {}

    def _update_body(self):
        self.data.qpos[0] = 8.0 if self.frame == 6 else 0.0
        mujoco.mj_forward(self.model, self.data)

    def is_done(self):
        return self.frame == 7

    def step_chunk(self, actions, stop_on_success=False):
        # Two policy actions still yield only one recorded video frame.
        assert len(actions) == 2
        self.frame += 1
        self._update_body()
        return {}, 0, self.is_done(), False, [{"success": self.is_done()}]

    def judge_success(self):
        return self.is_done()


def _run_video(
    aligned,
    camera_count,
    runner=None,
    kubric=True,
    query_interval=0,
    exclude_ambiguous=False,
    task=None,
):
    task = _VideoTask(camera_count) if task is None else task
    config = SimpleNamespace(
        generate_point_tracks=True,
        point_tracks_only=True,
        point_track_sampling="image",
        point_track_use_kubric_sampling=kubric,
        point_track_align_across_cameras=aligned,
        point_track_include_background=True,
        point_track_num_points=64,
        point_track_query_interval=query_interval,
        point_track_kubric_sampling_stride=4,
        point_track_kubric_max_sampled_fraction=1.0,
        camera_config=SimpleNamespace(
            cameras=[SimpleNamespace(name=name) for name in task.cameras],
            img_resolution=(8, 8),
        ),
    )
    policy = SimpleNamespace(get_action_chunk=lambda observation: [0, 0])
    if exclude_ambiguous is not None:
        config.point_track_exclude_raster_ambiguous = exclude_ambiguous
    if runner is None:
        runner = pipeline.ParallelRolloutRunner
    assert runner.run_single_rollout(
        episode_seed=20260826, task=task, policy=policy, exp_config=config
    )
    return task


@pytest.mark.parametrize("aligned,camera_count", [(False, 1), (False, 5), (True, 5)])
def test_kubric_video_occlusion_reappearance_and_saved_masks(aligned, camera_count, tmp_path):
    task = _run_video(aligned, camera_count)
    tracks = task._point_track_data
    assert len(tracks) == camera_count
    # Every video frame is scored, including frames outside the query grid.
    assert sorted(task.segmentation_calls) == sorted(
        (name, frame) for name in task.cameras for frame in range(8)
    )
    checked = 0
    for values in tracks.values():
        assert values["visibility_method"] == RASTER_VISIBILITY_METHOD
        assert values["trajs_2d"].shape == (8, 64, 2)
        assert values["geom_ids"].shape == (64,)
        # Source points sampled at t=0/4 have identical physical depth/location.
        selected = (values["geom_ids"] == task.surface) & np.isin(values["query_frames"], [0, 4])
        if not np.any(selected):
            continue
        checked += 1
        assert np.all(values["visibility"][[0, 3, 5, 7]][:, selected] == 1)
        # A 2.5 cm occluder passes the old 3 cm test but fails the new test.
        assert np.all(values["visibility"][1, selected] == 0)
        assert not np.any(values["raster_ambiguous"][1, selected])
        # Another geom at the same depth is ambiguous, even on the same body.
        assert np.all(values["visibility"][2, selected] == 0)
        assert np.all(values["raster_ambiguous"][2, selected])
        # Replayed body and camera poses must match their original frames.
        assert not np.any(values["in_frame"][6, selected])
        np.testing.assert_allclose(
            values["trajs_2d"][5, selected, 0],
            values["trajs_2d"][0, selected, 0] - 0.25,
            atol=1e-6,
        )

    assert checked > 0
    if aligned:
        reference = next(iter(tracks.values()))
        for values in tracks.values():
            for key in (
                "geom_ids",
                "track_ids",
                "body_ids",
                "query_frames",
                "query_source_cameras",
            ):
                np.testing.assert_array_equal(values[key], reference[key])
            np.testing.assert_array_equal(values["points_3d"], reference["points_3d"])

    # Exercise the actual rollout-to-NPZ handoff without invoking the RGB codec.
    pipeline.save_house_trajectories(
        logging.getLogger(__name__),
        [{"history": {"observations": []}, "point_track_data": tracks}],
        tmp_path,
        SimpleNamespace(point_tracks_only=True),
        "_batch_1_of_1",
    )
    for camera, values in tracks.items():
        with np.load(tmp_path / f"episode_00000000_{camera}_point_tracks.npz") as saved:
            assert str(saved["visibility_method"]) == RASTER_VISIBILITY_METHOD
            np.testing.assert_array_equal(saved["geom_ids"], values["geom_ids"])
            np.testing.assert_array_equal(saved["visibility_valid"], ~values["raster_ambiguous"])
            reasons = saved["visibility_reason_names"][saved["visibility_reason_codes"]]
            assert saved["visibility_reason_codes"].dtype == np.uint8
            np.testing.assert_array_equal(reasons == "visible", values["visibility"] > 0.5)
            np.testing.assert_array_equal(reasons == "raster_ambiguous", values["raster_ambiguous"])
            np.testing.assert_array_equal(reasons == "out_of_frame", ~values["in_frame"])
            assert float(saved["visibility_depth_relative_tolerance"]) == pytest.approx(0.01)
            assert float(saved["visibility_depth_absolute_tolerance_m"]) == pytest.approx(0.001)


@pytest.mark.parametrize("query_interval", [0, 2])
def test_legacy_image_video_mode_remains_available(query_interval):
    task = _run_video(False, 1, kubric=False, query_interval=query_interval)
    values = task._point_track_data["camera_0"]
    assert values["trajs_2d"].shape[0] == 8
    assert values["sampling_method"] == "image"
    assert "visibility_method" not in values
    assert "geom_ids" not in values


def test_compact_geom_raster_accepts_support_in_a_neighbor():
    # The rounded pixel sees an occluder, but a different pixel of the 2x2
    # neighborhood supports the exact point. A nearest-pixel test misses it.
    data = SimpleNamespace(
        xpos=np.array([[0, 0, 2]], dtype=float),
        xmat=np.eye(3).reshape(1, 9),
    )
    depth = np.full((8, 8), 1.0, dtype=np.float32)
    geom_map = np.full((8, 8, 1), 11, dtype=np.int32)
    depth[3, 3] = 2.0
    geom_map[3, 3, 0] = 7
    inputs = dict(
        data=data,
        local_coords=np.zeros((1, 3)),
        body_ids=np.array([0]),
        camera=None,
        img_width=8,
        img_height=8,
        depth_frame=depth,
        precomputed_w2c=np.eye(4),
        precomputed_intrinsics=np.array([[4, 0, 4], [0, 4, 4], [0, 0, 1]]),
    )
    assert track_points_for_frame(**inputs)[1][0] == 0
    assert (
        track_points_for_frame(
            **inputs,
            segmentation_frame=geom_map,
            geom_ids=np.array([7]),
        )[1][0]
        == 1
    )


@pytest.mark.parametrize(
    "problem", ["missing_geom", "missing_mask", "wrong_shape", "invalid_state"]
)
def test_raster_export_rejects_incomplete_or_inconsistent_masks(tmp_path, problem):
    options = dict(
        geom_ids=np.array([7]),
        visibility_method=RASTER_VISIBILITY_METHOD,
        in_frame=np.ones((2, 1), dtype=bool),
        raster_ambiguous=np.zeros((2, 1), dtype=bool),
    )
    if problem == "missing_geom":
        options.pop("geom_ids")
    elif problem == "missing_mask":
        options.pop("raster_ambiguous")
    elif problem == "wrong_shape":
        options["in_frame"] = np.ones((1, 2), dtype=bool)
    else:
        options["raster_ambiguous"][:] = True
    with pytest.raises(ValueError):
        save_point_tracks(
            tmp_path / "bad.npz",
            trajs_2d=np.zeros((2, 1, 2)),
            visibility=np.ones((2, 1)),
            points_3d_initial=None,
            points_3d=np.zeros((2, 1, 3)),
            body_ids=np.array([1]),
            intrinsics=np.eye(3),
            total_mesh_verts=None,
            **options,
        )


class _PartlyAmbiguousVideoTask(_VideoTask):
    def render_segmentation(self, camera):
        result = super().render_segmentation(camera)
        if self.frame == 2:
            result[:, :, 0] = self.surface
            # Only the last camera has ambiguity, in its lower-right region.
            # This is outside the aligned sampler's t=0/4 query grid.
            if camera == list(self.cameras)[-1]:
                result[4:, 3:, 0] = self.other
        return result


@pytest.mark.parametrize("aligned,camera_count", [(False, 1), (False, 5), (True, 5)])
def test_clean_video_default_replaces_whole_tracks_and_exports(aligned, camera_count, tmp_path):
    task = _run_video(
        aligned,
        camera_count,
        exclude_ambiguous=None,
        task=_PartlyAmbiguousVideoTask(camera_count),
    )
    tracks = task._point_track_data
    assert len(tracks) == camera_count
    for camera, values in tracks.items():
        assert values["exclude_raster_ambiguous"]
        assert values["trajs_2d"].shape == (8, 64, 2)
        assert not np.any(values["raster_ambiguous"])
        assert np.any((values["visibility"] == 0) & values["in_frame"])
        assert np.any(~values["in_frame"])
        stats = values["visibility_filter_stats"]
        assert (stats["initial_rejected_tracks"] > 0) == (stats["replacement_rounds"] > 0)
        expected_cameras = list(task.cameras) if aligned else [camera]
        np.testing.assert_array_equal(values["visibility_check_cameras"], expected_cameras)
        assert values["aligned_across_cameras"] == aligned
        if not aligned:
            assert np.all(values["query_source_cameras"] == camera)
        assert np.all(values["geom_ids"] == task.surface)
        # Shared tracks must be clean in other cameras too. Independent ones
        # only require valid observations in their own output video.
        task.frame = 2
        task._update_body()
        last_camera = list(task.cameras)[-1] if aligned else camera
        *_, diagnostics = track_points_for_frame(
            task.data,
            values["points_3d"][2] - task.data.xpos[task.body],
            values["body_ids"],
            task.cameras[last_camera],
            8,
            8,
            task.render_depth(last_camera),
            segmentation_frame=task.render_segmentation(last_camera),
            geom_ids=values["geom_ids"],
            return_diagnostics=True,
        )
        assert not diagnostics["raster_ambiguous"].any()
    if aligned:
        reference = next(iter(tracks.values()))
        for values in tracks.values():
            for key in (
                "track_ids",
                "geom_ids",
                "body_ids",
                "segment_ids",
                "query_frames",
                "query_source_cameras",
                "points_3d",
            ):
                np.testing.assert_array_equal(values[key], reference[key])

    pipeline.save_house_trajectories(
        logging.getLogger(__name__),
        [{"history": {"observations": []}, "point_track_data": tracks}],
        tmp_path,
        SimpleNamespace(point_tracks_only=True),
        "_batch_1_of_1",
    )
    for camera, values in tracks.items():
        with np.load(tmp_path / f"episode_00000000_{camera}_point_tracks.npz") as saved:
            assert saved["exclude_raster_ambiguous"]
            np.testing.assert_array_equal(
                saved["visibility_check_cameras"],
                values["visibility_check_cameras"],
            )
            assert bool(saved["aligned_across_cameras"]) == aligned
            assert saved["visibility_valid"].all()
            assert not saved["raster_ambiguous"].any()
            assert not (saved["visibility_reason_codes"] == 3).any()
            assert (saved["visibility_reason_codes"] == 2).any()
            assert saved["trajs_2d"].shape == (8, 64, 2)
            for key, count in values["visibility_filter_stats"].items():
                assert int(saved[f"visibility_filter_{key}"]) == count


@pytest.mark.parametrize("aligned,camera_count", [(False, 1), (False, 5), (True, 5)])
def test_clean_video_rejects_episode_when_all_candidates_ambiguous(aligned, camera_count):
    with pytest.raises(RuntimeError, match="No unambiguous|replacement rounds"):
        _run_video(aligned, camera_count, exclude_ambiguous=True)


@pytest.mark.parametrize("camera_count", [2, 5])
def test_independent_tracks_ignore_other_camera_ambiguity(camera_count):
    class NoOtherCameraAmbiguity(_PartlyAmbiguousVideoTask):
        def render_segmentation(self, camera):
            result = super().render_segmentation(camera)
            if self.frame == 2:
                result[:, :, 0] = self.surface
            return result

    bad_other = _run_video(
        False,
        camera_count,
        exclude_ambiguous=True,
        task=_PartlyAmbiguousVideoTask(camera_count),
    )
    good_other = _run_video(
        False,
        camera_count,
        exclude_ambiguous=True,
        task=NoOtherCameraAmbiguity(camera_count),
    )
    actual = bad_other._point_track_data["camera_0"]
    expected = good_other._point_track_data["camera_0"]
    for key in (
        "track_ids",
        "geom_ids",
        "body_ids",
        "query_frames",
        "query_points",
        "query_source_cameras",
        "points_3d",
        "trajs_2d",
        "visibility",
    ):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual["visibility_filter_stats"] == expected["visibility_filter_stats"]
    assert actual["visibility_filter_stats"]["initial_rejected_tracks"] == 0
    bad_other.frame = 2
    bad_other._update_body()
    last_camera = list(bad_other.cameras)[-1]
    *_, diagnostics = track_points_for_frame(
        bad_other.data,
        actual["points_3d"][2] - bad_other.data.xpos[bad_other.body],
        actual["body_ids"],
        bad_other.cameras[last_camera],
        8,
        8,
        bad_other.render_depth(last_camera),
        segmentation_frame=bad_other.render_segmentation(last_camera),
        geom_ids=actual["geom_ids"],
        return_diagnostics=True,
    )
    # These would have been rejected by the old all-camera check, despite
    # having entirely valid tracks in camera_0. They are now kept.
    assert diagnostics["raster_ambiguous"].any()
    assert not actual["raster_ambiguous"].any()


def test_clean_kubric_preserves_allocations_good_slots_and_seed():
    bodies = np.repeat([2, 4], 100)
    segments = np.repeat([0, 1], 100)
    initial = select_kubric_candidate_indices(bodies, 64, 10, 1.0, segments)
    checked = []

    def is_valid(indices):
        checked.extend(indices.tolist())
        return indices % 3 != 0

    selected, stats = select_unambiguous_kubric_candidate_indices(
        bodies,
        64,
        10,
        is_valid,
        1.0,
        segments,
    )
    assert len(selected) == 64
    assert np.all(selected % 3 != 0)
    np.testing.assert_array_equal(segments[selected], segments[initial])
    np.testing.assert_array_equal(selected[initial % 3 != 0], initial[initial % 3 != 0])
    assert stats["initial_rejected_tracks"] == np.count_nonzero(initial % 3 == 0)
    assert len(checked) == len(set(checked)) == stats["validated_candidates"]
    repeated, repeated_stats = select_unambiguous_kubric_candidate_indices(
        bodies,
        64,
        10,
        lambda indices: indices % 3 != 0,
        1.0,
        segments,
    )
    np.testing.assert_array_equal(selected, repeated)
    assert stats == repeated_stats
    unchanged, unchanged_stats = select_unambiguous_kubric_candidate_indices(
        bodies,
        64,
        10,
        lambda indices: np.ones(len(indices), dtype=bool),
        1.0,
        segments,
    )
    np.testing.assert_array_equal(unchanged, initial)
    assert unchanged_stats["replacement_rounds"] == 0


@pytest.mark.parametrize("problem", ["empty", "budget", "limit", "exhausted", "mask", "dtype"])
def test_clean_kubric_failure_is_explicit(problem):
    options = dict(
        body_ids=np.zeros(4, dtype=np.int32),
        max_points=16,
        seed=0,
        candidate_is_valid=lambda indices: np.zeros(len(indices), dtype=bool),
    )
    if problem == "empty":
        options["body_ids"] = np.zeros(0, dtype=np.int32)
    elif problem == "budget":
        options["max_points"] = 0
    elif problem == "limit":
        options["max_rejection_rounds"] = 0
    elif problem == "mask":
        options["candidate_is_valid"] = lambda indices: np.zeros((len(indices), 1), dtype=bool)
    elif problem == "dtype":
        options["candidate_is_valid"] = lambda indices: np.ones(len(indices), dtype=float)
    with pytest.raises((ValueError, RuntimeError)):
        select_unambiguous_kubric_candidate_indices(**options)


def test_clean_export_refuses_ambiguous_observations(tmp_path):
    with pytest.raises(ValueError, match="no ambiguity"):
        save_point_tracks(
            tmp_path / "bad.npz",
            trajs_2d=np.zeros((2, 1, 2)),
            visibility=np.zeros((2, 1)),
            points_3d_initial=None,
            points_3d=np.zeros((2, 1, 3)),
            body_ids=np.array([1]),
            intrinsics=np.eye(3),
            total_mesh_verts=None,
            geom_ids=np.array([7]),
            visibility_method=RASTER_VISIBILITY_METHOD,
            in_frame=np.ones((2, 1), dtype=bool),
            raster_ambiguous=np.ones((2, 1), dtype=bool),
            exclude_raster_ambiguous=True,
        )
    assert not (tmp_path / "bad.npz").exists()
