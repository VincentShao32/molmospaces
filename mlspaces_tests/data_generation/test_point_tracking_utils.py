from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.sensors import TCPPoseSensor, get_core_sensors
from molmo_spaces.utils.point_tracking_utils import (
    _fixed_phase_strided_mask,
    _random_phase_strided_mask,
    get_kubric_num_to_sample,
    get_kubric_segment_ids,
    get_object_body_ids,
    get_robot_body_ids,
    get_trackable_body_ids,
    sample_from_image,
    sample_kubric_candidates_from_image,
    save_point_tracks,
    select_kubric_candidate_indices,
    track_points_for_frame,
)


class IdentityCamera:
    fov = 90.0

    @staticmethod
    def get_pose() -> np.ndarray:
        return np.eye(4, dtype=np.float64)


def test_point_track_core_sensors_defer_tcp_pose_to_robot_sensor_suite():
    config = SimpleNamespace(
        point_tracks_only=True,
        camera_config=SimpleNamespace(
            cameras=(SimpleNamespace(name="camera", record_depth=False),),
            img_resolution=(64, 48),
        ),
    )

    suite = SensorSuite(get_core_sensors(config))
    suite.extend([TCPPoseSensor(uuid="tcp_pose")])

    assert set(suite.sensors) == {"camera", "qpos", "tcp_pose"}


def _tracking_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="object" pos="0 0 2">
              <freejoint/>
              <geom type="box" size="0.1 0.1 0.1"/>
              <body name="object_child">
                <geom type="sphere" size="0.05"/>
              </body>
            </body>
            <body name="background" pos="0 0 3">
              <geom type="box" size="0.2 0.2 0.2"/>
            </body>
            <body name="robot_0/link" pos="0 0 1">
              <geom type="capsule" size="0.05 0.2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_body_classification_includes_freejoint_descendants_and_robot_namespace():
    model, _ = _tracking_model()
    object_id = model.body("object").id
    child_id = model.body("object_child").id
    robot_id = model.body("robot_0/link").id

    assert get_object_body_ids(model) == {object_id, child_id}
    assert get_robot_body_ids(model) == {robot_id}
    assert get_trackable_body_ids(model) == {object_id, child_id, robot_id}

    background_id = model.body("background").id
    segment_ids = get_kubric_segment_ids(
        model,
        np.array([object_id, child_id, robot_id, background_id, 0]),
        get_trackable_body_ids(model),
    )
    np.testing.assert_array_equal(segment_ids, [object_id, object_id, robot_id, 0, 0])


def test_strided_mask_is_seeded_and_uses_one_grid_phase():
    mask_a = _random_phase_strided_mask(7, 9, 3, np.random.RandomState(11))
    mask_b = _random_phase_strided_mask(7, 9, 3, np.random.RandomState(11))

    assert np.array_equal(mask_a, mask_b)
    ys, xs = np.where(mask_a)
    assert len(ys) > 0
    assert len(set(ys % 3)) == 1
    assert len(set(xs % 3)) == 1
    assert _random_phase_strided_mask(7, 9, 1, np.random.RandomState(11)) is None


def test_fixed_phase_strided_mask_reuses_requested_phase():
    mask = _fixed_phase_strided_mask(7, 9, 3, phase_y=2, phase_x=1)
    ys, xs = np.where(mask)

    assert set(ys % 3) == {2}
    assert set(xs % 3) == {1}


def test_kubric_budget_balances_segments_and_applies_fraction_cap():
    allocation = get_kubric_num_to_sample(
        np.array([10, 100, 1000]),
        max_sampled_fraction=0.1,
        tracks_to_sample=60,
    )

    np.testing.assert_array_equal(allocation, [1, 10, 49])


def test_kubric_candidate_extraction_uses_grid_and_pixel_centers():
    model, data = _tracking_model()
    object_id = model.body("object").id
    segmentation = np.full((4, 4, 3), -1, dtype=np.int32)
    segmentation[:, :, 0] = 0
    segmentation[:, :, 1] = mujoco.mjtObj.mjOBJ_GEOM.value
    segmentation[:, :, 2] = object_id
    depth = np.full((4, 4), 2.0, dtype=np.float32)

    local, body_ids, world, query = sample_kubric_candidates_from_image(
        model,
        data,
        IdentityCamera(),
        img_width=4,
        img_height=4,
        depth_frame=depth,
        seg_frame=segmentation,
        frame_index=3,
        sampling_stride=2,
        spatial_phase=(1, 0),
        object_body_ids={object_id},
    )

    assert local.shape == world.shape == (4, 3)
    np.testing.assert_array_equal(body_ids, np.full(4, object_id))
    np.testing.assert_array_equal(query[:, 0], np.full(4, 3))
    assert set(query[:, 1]) == {1.5, 3.5}
    assert set(query[:, 2]) == {0.5, 2.5}
    np.testing.assert_allclose(world[:, :2], query[:, [2, 1]] - 2.0)
    reprojection, visibility, _ = track_points_for_frame(
        data,
        local,
        body_ids,
        IdentityCamera(),
        img_width=4,
        img_height=4,
        depth_frame=depth,
    )
    np.testing.assert_allclose(reprojection, query[:, [2, 1]])
    np.testing.assert_array_equal(visibility, np.ones(4))


def test_kubric_candidate_selection_is_per_segment_and_aligned():
    body_ids = np.concatenate(
        [
            np.full(5, 1, dtype=np.int32),
            np.full(5, 2, dtype=np.int32),
            np.full(100, 3, dtype=np.int32),
        ]
    )
    segment_ids = np.concatenate([np.full(10, 7, dtype=np.int32), np.full(100, 8, dtype=np.int32)])
    selected_indices = select_kubric_candidate_indices(
        body_ids,
        max_points=11,
        seed=5,
        max_sampled_fraction=0.1,
        segment_ids=segment_ids,
    )

    assert len(selected_indices) == 11
    assert np.count_nonzero(selected_indices < 10) == 1
    assert np.count_nonzero(body_ids[selected_indices] == 3) == 10
    np.testing.assert_array_equal(
        selected_indices,
        select_kubric_candidate_indices(body_ids, 11, 5, 0.1, segment_ids),
    )


def test_image_sampling_reserves_background_budget():
    model, data = _tracking_model()
    object_id = model.body("object").id
    background_id = model.body("background").id

    segmentation = np.zeros((4, 4, 3), dtype=np.int32)
    segmentation[:2, :, 2] = object_id
    segmentation[2:, :, 2] = background_id
    depth = np.full((4, 4), 2.0, dtype=np.float32)

    local, body_ids, world, _ = sample_from_image(
        model,
        data,
        IdentityCamera(),
        img_width=4,
        img_height=4,
        depth_frame=depth,
        seg_frame=segmentation,
        max_points=8,
        seed=7,
        object_body_ids={object_id},
        background_body_ids={background_id},
        background_fraction=0.5,
    )

    assert local.shape == (8, 3)
    assert world.shape == (8, 3)
    assert np.count_nonzero(body_ids == object_id) == 4
    assert np.count_nonzero(body_ids == background_id) == 4


def test_tracking_projects_points_and_detects_occlusion():
    data = SimpleNamespace(
        xpos=np.array([[0, 0, 0], [0, 0, 2]], dtype=np.float64),
        xmat=np.tile(np.eye(3, dtype=np.float64).reshape(1, 9), (2, 1)),
    )
    local = np.array([[0, 0, 0], [0.5, 0, 0]], dtype=np.float32)
    body_ids = np.array([1, 1], dtype=np.int32)
    intrinsics = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float32)

    coords, visibility, world = track_points_for_frame(
        data,
        local,
        body_ids,
        camera=None,
        img_width=100,
        img_height=100,
        depth_frame=np.full((100, 100), 2.0, dtype=np.float32),
        precomputed_w2c=np.eye(4, dtype=np.float32),
        precomputed_intrinsics=intrinsics,
    )

    np.testing.assert_allclose(coords, [[50, 50], [75, 50]])
    np.testing.assert_allclose(world, [[0, 0, 2], [0.5, 0, 2]])
    np.testing.assert_array_equal(visibility, [1, 1])

    _, occluded, _ = track_points_for_frame(
        data,
        local,
        body_ids,
        camera=None,
        img_width=100,
        img_height=100,
        depth_frame=np.full((100, 100), 1.0, dtype=np.float32),
        precomputed_w2c=np.eye(4, dtype=np.float32),
        precomputed_intrinsics=intrinsics,
    )
    np.testing.assert_array_equal(occluded, [0, 0])


def test_tracking_uses_four_neighbor_exact_geom_raster_support():
    model, data = _tracking_model()
    object_id = model.body("object").id
    object_geom_id = int(model.body_geomadr[object_id])
    local = np.zeros((1, 3), dtype=np.float32)
    body_ids = np.array([object_id], dtype=np.int32)
    geom_ids = np.array([object_geom_id], dtype=np.int32)
    intrinsics = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]], dtype=np.float32)

    segmentation = np.full((100, 100, 3), -1, dtype=np.int32)
    segmentation[50, 50, 0] = object_geom_id
    segmentation[50, 50, 1] = mujoco.mjtObj.mjOBJ_GEOM.value
    segmentation[50, 50, 2] = object_id
    depth = np.full((100, 100), 4.0, dtype=np.float32)
    depth[50, 50] = 2.0

    _, visible, _, diagnostics = track_points_for_frame(
        data,
        local,
        body_ids,
        camera=None,
        img_width=100,
        img_height=100,
        depth_frame=depth,
        precomputed_w2c=np.eye(4, dtype=np.float32),
        precomputed_intrinsics=intrinsics,
        return_diagnostics=True,
        segmentation_frame=segmentation,
        geom_ids=geom_ids,
    )
    np.testing.assert_array_equal(visible, [1])
    assert diagnostics["visibility_reason"][0] == "visible"
    assert diagnostics["matching_geom_neighbor_count"][0] == 1
    assert not diagnostics["raster_ambiguous"][0]

    occluder_depth = np.full((100, 100), 1.9, dtype=np.float32)
    _, occluded, _, diagnostics = track_points_for_frame(
        data,
        local,
        body_ids,
        camera=None,
        img_width=100,
        img_height=100,
        depth_frame=occluder_depth,
        precomputed_w2c=np.eye(4, dtype=np.float32),
        precomputed_intrinsics=intrinsics,
        return_diagnostics=True,
        segmentation_frame=segmentation,
        geom_ids=geom_ids,
    )
    np.testing.assert_array_equal(occluded, [0])
    assert diagnostics["visibility_reason"][0] == "occluded_depth_confirmed"
    assert not diagnostics["raster_ambiguous"][0]

    no_matching_geom = np.full((100, 100, 3), -1, dtype=np.int32)
    _, ambiguous, _, diagnostics = track_points_for_frame(
        data,
        local,
        body_ids,
        camera=None,
        img_width=100,
        img_height=100,
        depth_frame=np.full((100, 100), 4.0, dtype=np.float32),
        precomputed_w2c=np.eye(4, dtype=np.float32),
        precomputed_intrinsics=intrinsics,
        return_diagnostics=True,
        segmentation_frame=no_matching_geom,
        geom_ids=geom_ids,
    )
    np.testing.assert_array_equal(ambiguous, [0])
    assert diagnostics["visibility_reason"][0] == "raster_ambiguous"
    assert diagnostics["raster_ambiguous"][0]


def test_shared_body_local_points_keep_identity_across_camera_projections():
    data = SimpleNamespace(
        xpos=np.array([[0, 0, 0], [0, 0, 2]], dtype=np.float64),
        xmat=np.tile(np.eye(3, dtype=np.float64).reshape(1, 9), (2, 1)),
    )
    local = np.array([[0, 0, 0], [0.5, 0, 0]], dtype=np.float32)
    body_ids = np.array([1, 1], dtype=np.int32)
    intrinsics = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float32)
    camera_b_w2c = np.eye(4, dtype=np.float32)
    camera_b_w2c[0, 3] = -0.25

    coords_a, _, world_a = track_points_for_frame(
        data,
        local,
        body_ids,
        camera=None,
        img_width=100,
        img_height=100,
        depth_frame=np.full((100, 100), 2.0, dtype=np.float32),
        precomputed_w2c=np.eye(4, dtype=np.float32),
        precomputed_intrinsics=intrinsics,
    )
    coords_b, _, world_b = track_points_for_frame(
        data,
        local,
        body_ids,
        camera=None,
        img_width=100,
        img_height=100,
        depth_frame=np.full((100, 100), 2.0, dtype=np.float32),
        precomputed_w2c=camera_b_w2c,
        precomputed_intrinsics=intrinsics,
    )

    np.testing.assert_allclose(world_a, world_b)
    np.testing.assert_allclose(coords_b[:, 0], coords_a[:, 0] - 12.5)
    np.testing.assert_allclose(coords_b[:, 1], coords_a[:, 1])


def test_save_point_tracks_preserves_optional_metadata(tmp_path):
    output = tmp_path / "tracks.npz"
    save_point_tracks(
        output,
        trajs_2d=np.zeros((2, 3, 2)),
        visibility=np.ones((2, 3)),
        points_3d_initial=np.zeros((3, 3)),
        points_3d=np.zeros((2, 3, 3)),
        body_ids=np.array([1, 2, 3]),
        intrinsics=np.eye(3),
        total_mesh_verts=12,
        query_frames=np.array([0, 1, 1]),
        query_points=np.array([[0, 1.5, 2.5], [1, 3.5, 4.5], [1, 5.5, 6.5]]),
        sampling_method="kubric",
        sampling_stride=4,
        sampling_phase=np.array([3, 1, 2]),
        max_sampled_fraction=0.1,
        segment_ids=np.array([1, 1, 2]),
        track_ids=np.array([0, 1, 2]),
        aligned_across_cameras=True,
        query_source_cameras=np.array(["wrist", "overhead", "wrist"]),
    )

    with np.load(output) as saved:
        assert saved["trajs_2d"].dtype == np.float32
        assert saved["body_ids"].dtype == np.int32
        np.testing.assert_array_equal(saved["query_frames"], [0, 1, 1])
        np.testing.assert_allclose(saved["query_points"][:, 1] % 1, 0.5)
        assert str(saved["sampling_method"]) == "kubric"
        assert int(saved["sampling_stride"]) == 4
        np.testing.assert_array_equal(saved["sampling_phase"], [3, 1, 2])
        assert float(saved["max_sampled_fraction"]) == pytest.approx(0.1)
        np.testing.assert_array_equal(saved["segment_ids"], [1, 1, 2])
        np.testing.assert_array_equal(saved["track_ids"], [0, 1, 2])
        assert bool(saved["aligned_across_cameras"])
        np.testing.assert_array_equal(saved["query_source_cameras"], ["wrist", "overhead", "wrist"])
        assert int(saved["num_sampled_from"]) == 12
