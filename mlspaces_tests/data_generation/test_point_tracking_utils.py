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
    candidate_mask_context_features,
    depth_penalized_size_features,
    get_body_subtree_ids,
    get_kubric_num_to_sample,
    get_kubric_segment_ids,
    get_kubric_segment_names,
    get_manipulation_target_body_ids,
    get_object_body_ids,
    get_robot_body_ids,
    get_trackable_body_ids,
    sample_aligned_kubric_points_for_frame,
    sample_from_image,
    sample_kubric_candidates_from_image,
    save_point_tracks,
    select_failure_targeted_candidate_indices,
    select_failure_targeted_cross_view_shortlist_indices,
    select_kubric_candidate_indices,
    soft_depth_sampling_weights,
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
    np.testing.assert_array_equal(
        get_kubric_segment_names(model, segment_ids),
        ["object", "object", "robot_0/link", "background", "background"],
    )


def test_manipulation_target_ids_include_pickup_and_gripper_subtrees():
    model, data = _tracking_model()
    object_id = model.body("object").id
    object_child_id = model.body("object_child").id
    gripper_id = model.body("robot_0/link").id
    robot_view = SimpleNamespace(
        get_gripper_movegroup_ids=lambda: ["gripper"],
        get_move_group=lambda _move_group_id: SimpleNamespace(root_body_id=gripper_id),
    )
    pickup_object = SimpleNamespace(body_ids=[object_id, object_child_id])
    object_manager = SimpleNamespace(
        get_object_by_name=lambda name: pickup_object if name == "object" else None
    )
    task = SimpleNamespace(
        config=SimpleNamespace(task_config=SimpleNamespace(pickup_obj_name="object")),
        env=SimpleNamespace(
            current_model=model,
            current_data=data,
            current_batch_index=0,
            current_robot=SimpleNamespace(robot_view=robot_view),
            object_managers=[object_manager],
        ),
    )

    body_ids, labels = get_manipulation_target_body_ids(task)

    assert body_ids == {object_id, object_child_id, gripper_id}
    assert get_body_subtree_ids(model, object_id) == {object_id, object_child_id}
    assert labels == {
        object_id: "manipulated_object",
        object_child_id: "manipulated_object",
        gripper_id: "gripper",
    }


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


def test_candidate_context_features_detect_small_segments_and_edges():
    segment_map = np.zeros((7, 7), dtype=np.int32)
    segment_map[2:5, 2:5] = 7
    query_points = np.array(
        [
            [0.0, 3.5, 3.5],
            [0.0, 2.5, 2.5],
        ],
        dtype=np.float32,
    )

    area, support, edge_distance = candidate_mask_context_features(
        segment_map,
        query_points,
        np.array([7, 7], dtype=np.int32),
        local_radius_px=1,
        max_edge_distance_px=3,
    )

    np.testing.assert_allclose(area, np.full(2, 9 / 49), rtol=1e-6)
    assert support[0] == pytest.approx(1.0)
    assert support[1] == pytest.approx(4 / 9)
    np.testing.assert_allclose(edge_distance, [2.0, 1.0])


def test_depth_penalty_suppresses_perspective_smallness_without_a_cutoff():
    adjusted_area, adjusted_support = depth_penalized_size_features(
        np.array([0.01, 0.01, 0.001], dtype=np.float32),
        np.array([0.5, 0.5, 0.1], dtype=np.float32),
        np.array([0.5, 4.0, 4.0], dtype=np.float32),
        reference_depth_m=1.0,
    )

    np.testing.assert_allclose(adjusted_area, [0.01, 0.16, 0.016])
    np.testing.assert_allclose(adjusted_support, [0.5, 1.0, 0.4])
    assert adjusted_area[1] > 0.02
    assert adjusted_area[2] <= 0.02
    np.testing.assert_allclose(
        soft_depth_sampling_weights(
            np.array([0.5, 4.0], dtype=np.float32),
            reference_depth_m=1.0,
            minimum_weight=0.10,
        ),
        [1.0, 0.15625],
    )


def test_failure_targeted_depth_weight_prefers_near_without_removing_far():
    n_candidates = 200
    segment_ids = np.ones(n_candidates, dtype=np.int32)
    selected, buckets, candidate_counts = select_failure_targeted_candidate_indices(
        segment_ids,
        np.full(n_candidates, 5.0, dtype=np.float32),
        np.full(n_candidates, 0.001, dtype=np.float32),
        np.ones(n_candidates, dtype=np.float32),
        np.ones((2, n_candidates), dtype=bool),
        np.ones((2, n_candidates), dtype=bool),
        np.full(n_candidates, 5.0, dtype=np.float32),
        max_points=100,
        seed=11,
        source_depth_m=np.concatenate(
            [
                np.full(100, 0.5, dtype=np.float32),
                np.full(100, 4.0, dtype=np.float32),
            ]
        ),
        depth_sampling_min_weight=0.10,
    )

    small_selected = selected[buckets == "small_thin"]
    assert np.count_nonzero(small_selected < 100) >= 15
    assert np.count_nonzero(small_selected >= 100) >= 1
    assert candidate_counts["small_thin"] == n_candidates


def test_failure_targeted_cross_view_shortlist_is_balanced_and_deterministic():
    n_candidates = 120
    segment_ids = np.tile(np.arange(1, 5, dtype=np.int32), 30)
    source_cameras = np.tile(np.repeat(np.asarray(["wrist", "exo"]), 4), 15)
    target_labels = np.where(segment_ids <= 2, "pickup", "gripper")
    edge_distance = np.full(n_candidates, 8.0, dtype=np.float32)
    area_fraction = np.full(n_candidates, 0.2, dtype=np.float32)
    local_support = np.ones(n_candidates, dtype=np.float32)
    edge_distance[np.arange(n_candidates) % 6 == 0] = 3.0
    area_fraction[np.arange(n_candidates) % 6 == 1] = 0.01
    priority = (edge_distance <= 4.0) | (area_fraction <= 0.02)

    selected = select_failure_targeted_cross_view_shortlist_indices(
        segment_ids,
        source_cameras,
        edge_distance,
        area_fraction,
        local_support,
        max_candidates=40,
        seed=17,
        target_labels=target_labels,
    )
    repeated = select_failure_targeted_cross_view_shortlist_indices(
        segment_ids,
        source_cameras,
        edge_distance,
        area_fraction,
        local_support,
        max_candidates=40,
        seed=17,
        target_labels=target_labels,
    )

    np.testing.assert_array_equal(selected, repeated)
    assert len(selected) == len(np.unique(selected)) == 40
    assert int(priority[selected].sum()) == 16
    assert set(source_cameras[selected]) == {"wrist", "exo"}
    assert set(segment_ids[selected]) == {1, 2, 3, 4}
    assert set(target_labels[selected]) == {"pickup", "gripper"}


def test_failure_targeted_cross_view_shortlist_can_prioritize_only_small_objects():
    n_candidates = 60
    segment_ids = np.ones(n_candidates, dtype=np.int32)
    source_cameras = np.full(n_candidates, "camera")
    edge_distance = np.full(n_candidates, 8.0, dtype=np.float32)
    edge_distance[:20] = 3.0
    area_fraction = np.full(n_candidates, 0.2, dtype=np.float32)
    area_fraction[20:36] = 0.01

    selected = select_failure_targeted_cross_view_shortlist_indices(
        segment_ids,
        source_cameras,
        edge_distance,
        area_fraction,
        np.ones(n_candidates, dtype=np.float32),
        max_candidates=20,
        seed=29,
        priority_fraction=0.8,
        prioritize_edges=False,
    )

    assert set(range(20, 36)).issubset(set(selected))


def test_failure_targeted_selection_honors_bucket_quotas_and_fills():
    n_candidates = 100
    segment_ids = np.ones(n_candidates, dtype=np.int32)
    edge_distance = np.full(n_candidates, 5.0, dtype=np.float32)
    edge_distance[:20] = 2.0
    area_fraction = np.full(n_candidates, 0.2, dtype=np.float32)
    area_fraction[20:40] = 0.01
    local_support = np.ones(n_candidates, dtype=np.float32)
    visibility = np.ones((2, n_candidates), dtype=bool)
    in_frame = np.ones((2, n_candidates), dtype=bool)
    visibility[1, 40:70] = False
    occluder_edge = np.full(n_candidates, 5.0, dtype=np.float32)
    occluder_edge[40:50] = 1.0

    selected, buckets, candidate_counts = select_failure_targeted_candidate_indices(
        segment_ids,
        edge_distance,
        area_fraction,
        local_support,
        visibility,
        in_frame,
        occluder_edge,
        max_points=40,
        seed=7,
    )

    assert len(selected) == len(np.unique(selected)) == 40
    selected_names, selected_counts = np.unique(buckets, return_counts=True)
    assert dict(zip(selected_names, selected_counts)) == {
        "baseline": 10,
        "cross_view_occlusion": 6,
        "object_edge": 8,
        "occlusion_edge": 8,
        "small_thin": 8,
    }
    assert candidate_counts == {
        "occlusion_edge": 10,
        "cross_view_occlusion": 30,
        "object_edge": 20,
        "small_thin": 20,
        "baseline": 100,
    }


def test_failure_targeted_selection_excludes_unsafe_source_edge_band():
    n_candidates = 50
    segment_ids = np.ones(n_candidates, dtype=np.int32)
    edge_distance = np.concatenate(
        [
            np.ones(10, dtype=np.float32),
            np.full(10, 2.0, dtype=np.float32),
            np.full(10, 3.0, dtype=np.float32),
            np.full(20, 5.0, dtype=np.float32),
        ]
    )
    visibility = np.ones((2, n_candidates), dtype=bool)
    visibility[1, :15] = False

    selected, buckets, candidate_counts = select_failure_targeted_candidate_indices(
        segment_ids,
        edge_distance,
        np.full(n_candidates, 0.2, dtype=np.float32),
        np.ones(n_candidates, dtype=np.float32),
        visibility,
        np.ones_like(visibility),
        np.full(n_candidates, 5.0, dtype=np.float32),
        max_points=45,
        seed=13,
        edge_distance_px=4.0,
        minimum_source_edge_distance_px=2.0,
    )

    assert len(selected) == 45
    assert np.all(edge_distance[selected] >= 2.0)
    assert np.all(edge_distance[selected[buckets == "object_edge"]] <= 4.0)
    assert candidate_counts["object_edge"] == 20
    assert candidate_counts["cross_view_occlusion"] == 5
    assert candidate_counts["baseline"] == 40


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


def _snapshot_visibility_env(*, all_ambiguous=False, occluded=False):
    """Two exact geoms on one logical object; deliberately reuse renderer buffers."""
    model, data = _tracking_model()
    body = model.body("object").id
    surface = model.body_geomadr[body]
    other_surface = model.body_geomadr[model.body("object_child").id]
    segmentation = np.empty((8, 16, 3), dtype=np.int32)
    depth = np.empty((8, 16), dtype=np.float32)

    def render_segmentation(camera_name):
        segmentation[:, :, 0] = surface
        segmentation[:, :, 1] = mujoco.mjtObj.mjOBJ_GEOM.value
        segmentation[:, :, 2] = body
        if camera_name == "other" and not occluded:
            segmentation[:, 0 if all_ambiguous else 8 :, 0] = other_surface
            segmentation[:, 0 if all_ambiguous else 8 :, 2] = model.body("object_child").id
        return segmentation

    def render_depth(camera_name):
        depth.fill(1.0 if occluded and camera_name == "other" else 2.0)
        return depth

    return SimpleNamespace(
        current_model=model,
        current_data=data,
        camera_manager=SimpleNamespace(
            registry={"query": IdentityCamera(), "other": IdentityCamera()}
        ),
        render_depth_frame=render_depth,
        render_segmentation_frame=render_segmentation,
    )


@pytest.mark.parametrize("sampling_mode", ["kubric", "failure_targeted"])
@pytest.mark.parametrize("exclude_ambiguous", [False, True])
def test_snapshot_sampling_filters_ambiguity_and_keeps_arrays_aligned(
    sampling_mode, exclude_ambiguous
):
    env = _snapshot_visibility_env()
    sampled = sample_aligned_kubric_points_for_frame(
        env,
        ("query", "other"),
        16,
        8,
        max_points=200,
        seed=17,
        sampling_stride=1,
        max_sampled_fraction=1.0,
        sampling_mode=sampling_mode,
        exclude_raster_ambiguous=exclude_ambiguous,
    )
    filtered = sampled["num_raster_ambiguous_candidates_filtered"]
    assert (filtered > 0) == exclude_ambiguous
    assert sampled["num_candidates"] == sampled["num_raw_candidates"] - filtered
    ambiguous = np.any([cam["raster_ambiguous"] for cam in sampled["cameras"].values()])
    assert ambiguous == (not exclude_ambiguous)

    # Reproject the exported body-local points independently of cached subsets.
    for name, camera in env.camera_manager.registry.items():
        xy, visible, world, diagnostics = track_points_for_frame(
            env.current_data,
            sampled["local_coords"],
            sampled["body_ids"],
            camera,
            16,
            8,
            env.render_depth_frame(name),
            return_diagnostics=True,
            segmentation_frame=env.render_segmentation_frame(name),
            geom_ids=sampled["geom_ids"],
        )
        exported = sampled["cameras"][name]
        np.testing.assert_array_equal(exported["points_2d"], xy)
        np.testing.assert_array_equal(exported["visibility"], visible)
        np.testing.assert_array_equal(sampled["points_3d"], world)
        np.testing.assert_array_equal(
            exported["visibility_reason"], diagnostics["visibility_reason"]
        )
        source = sampled["query_source_cameras"] == name
        np.testing.assert_allclose(sampled["query_points"][source][:, [2, 1]], xy[source])
        assert np.all(visible[source] == 1)


@pytest.mark.parametrize("sampling_mode", ["kubric", "failure_targeted"])
def test_snapshot_sampling_rejects_all_ambiguous_candidates(sampling_mode):
    with pytest.raises(RuntimeError, match="All multiview point candidates had ambiguous"):
        sample_aligned_kubric_points_for_frame(
            _snapshot_visibility_env(all_ambiguous=True),
            ("query", "other"),
            16,
            8,
            sampling_stride=1,
            sampling_mode=sampling_mode,
        )


@pytest.mark.parametrize("sampling_mode", ["kubric", "failure_targeted"])
def test_snapshot_filter_retains_depth_confirmed_occlusion(sampling_mode):
    sampled = sample_aligned_kubric_points_for_frame(
        _snapshot_visibility_env(occluded=True),
        ("query", "other"),
        16,
        8,
        sampling_stride=1,
        sampling_mode=sampling_mode,
    )
    assert sampled["num_raster_ambiguous_candidates_filtered"] > 0
    np.testing.assert_array_equal(sampled["query_source_cameras"], "query")
    np.testing.assert_array_equal(sampled["source_depth_m"], 2.0)
    np.testing.assert_array_equal(sampled["cameras"]["query"]["visibility_reason"], "visible")
    np.testing.assert_array_equal(
        sampled["cameras"]["other"]["visibility_reason"], "occluded_depth_confirmed"
    )


def test_still_image_kubric_sampling_aligns_physical_points_across_cameras():
    model, data = _tracking_model()
    object_id = model.body("object").id
    segmentation = np.full((4, 4, 3), -1, dtype=np.int32)
    segmentation[:, :, 0] = model.body_geomadr[object_id]
    segmentation[:, :, 1] = mujoco.mjtObj.mjOBJ_GEOM.value
    segmentation[:, :, 2] = object_id
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    env = SimpleNamespace(
        current_model=model,
        current_data=data,
        camera_manager=SimpleNamespace(
            registry={"wrist_camera": IdentityCamera(), "exo_camera_1": IdentityCamera()}
        ),
        render_depth_frame=lambda _camera_name: depth,
        render_segmentation_frame=lambda _camera_name: segmentation,
    )

    sampled = sample_aligned_kubric_points_for_frame(
        env,
        ("wrist_camera", "exo_camera_1"),
        img_width=4,
        img_height=4,
        max_points=8,
        seed=13,
        sampling_stride=1,
        max_sampled_fraction=1.0,
        include_background=False,
    )

    assert len(sampled["track_ids"]) == 8
    assert sampled["visibility_method"] == "four_neighbor_exact_geom_depth_support_v1"
    assert sampled["geom_ids"].shape == (8,)
    np.testing.assert_array_equal(sampled["point_object_names"], ["object"] * 8)
    assert sampled["num_raster_ambiguous_candidates_filtered"] == 0
    np.testing.assert_array_equal(sampled["track_ids"], np.arange(8))
    assert sampled["surface_normal_orientation"] == "toward_query_source_camera"
    assert sampled["normals_3d"].shape == (8, 3)
    assert sampled["local_normals"].shape == (8, 3)
    assert sampled["query_source_camera_normals"].shape == (8, 3)
    np.testing.assert_allclose(np.linalg.norm(sampled["normals_3d"], axis=1), 1.0, atol=2e-4)
    np.testing.assert_allclose(np.linalg.norm(sampled["local_normals"], axis=1), 1.0, atol=2e-4)
    np.testing.assert_allclose(
        np.linalg.norm(sampled["query_source_camera_normals"], axis=1),
        1.0,
        atol=2e-4,
    )
    assert set(sampled["surface_normal_methods"]).issubset(
        {"rendered_depth_exact_geom", "primitive_analytic"}
    )
    assert "unavailable" not in set(sampled["surface_normal_methods"])
    np.testing.assert_allclose(
        sampled["cameras"]["wrist_camera"]["points_2d"],
        sampled["cameras"]["exo_camera_1"]["points_2d"],
    )
    np.testing.assert_allclose(
        sampled["cameras"]["wrist_camera"]["points_3d"],
        sampled["cameras"]["exo_camera_1"]["points_3d"],
    )


def test_failure_targeted_still_sampling_saves_diagnostics_and_buckets():
    model, data = _tracking_model()
    object_id = model.body("object").id
    segmentation = np.full((4, 4, 3), -1, dtype=np.int32)
    segmentation[:, :, 0] = model.body_geomadr[object_id]
    segmentation[:, :, 1] = mujoco.mjtObj.mjOBJ_GEOM.value
    segmentation[:, :, 2] = object_id
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    env = SimpleNamespace(
        current_model=model,
        current_data=data,
        camera_manager=SimpleNamespace(
            registry={"wrist_camera": IdentityCamera(), "exo_camera_1": IdentityCamera()}
        ),
        render_depth_frame=lambda _camera_name: depth,
        render_segmentation_frame=lambda _camera_name: segmentation,
    )

    sampled = sample_aligned_kubric_points_for_frame(
        env,
        ("wrist_camera", "exo_camera_1"),
        img_width=4,
        img_height=4,
        max_points=8,
        seed=19,
        sampling_stride=1,
        max_sampled_fraction=1.0,
        include_background=False,
        sampling_mode="failure_targeted",
        failure_max_cross_view_candidates=8,
    )

    assert sampled["sampling_method"] == "failure_targeted"
    assert float(sampled["failure_min_source_edge_distance_px"]) == 2.0
    assert int(sampled["failure_max_cross_view_candidates"]) == 8
    assert int(sampled["num_raw_candidates"]) == 32
    assert int(sampled["num_candidates_before_cross_view_shortlist"]) == 32
    assert int(sampled["num_cross_view_candidates"]) == 8
    assert int(sampled["num_cross_view_candidates_shortlisted_out"]) == 24
    assert int(sampled["num_candidates"]) == 8
    assert np.all(sampled["source_edge_distance_px"] >= 2.0)
    assert len(sampled["sampling_buckets"]) == 8
    assert int(sampled["selected_bucket_counts"].sum()) == 8
    assert sampled["source_edge_distance_px"].shape == (8,)
    assert sampled["source_depth_m"].shape == (8,)
    assert sampled["depth_penalized_segment_area_fraction"].shape == (8,)
    assert sampled["visible_camera_count"].shape == (8,)
    for camera_points in sampled["cameras"].values():
        assert camera_points["in_frame"].shape == (8,)
        assert camera_points["point_depth"].shape == (8,)
        assert camera_points["depth_residual"].shape == (8,)
        assert camera_points["visibility_reason"].shape == (8,)
        assert not camera_points["raster_ambiguous"].any()


def test_still_sampling_restricts_candidates_to_explicit_target_bodies():
    model, data = _tracking_model()
    object_id = model.body("object").id
    gripper_id = model.body("robot_0/link").id
    background_id = model.body("background").id
    segmentation = np.full((4, 6, 3), -1, dtype=np.int32)
    segmentation[:, :, 0] = 0
    segmentation[:, :, 1] = mujoco.mjtObj.mjOBJ_GEOM.value
    segmentation[:, :2, 2] = object_id
    segmentation[:, 2:4, 2] = gripper_id
    segmentation[:, 4:, 2] = background_id
    depth = np.full((4, 6), 2.0, dtype=np.float32)
    env = SimpleNamespace(
        current_model=model,
        current_data=data,
        camera_manager=SimpleNamespace(registry={"camera": IdentityCamera()}),
        render_depth_frame=lambda _camera_name: depth,
        render_segmentation_frame=lambda _camera_name: segmentation,
    )

    sampled = sample_aligned_kubric_points_for_frame(
        env,
        ("camera",),
        img_width=6,
        img_height=4,
        max_points=6,
        seed=23,
        sampling_stride=1,
        max_sampled_fraction=1.0,
        include_background=False,
        eligible_body_ids={object_id, gripper_id},
        body_target_labels={
            object_id: "manipulated_object",
            gripper_id: "gripper",
        },
    )

    assert set(sampled["body_ids"]) == {object_id, gripper_id}
    target_names, target_counts = np.unique(sampled["point_target_labels"], return_counts=True)
    assert dict(zip(target_names, target_counts)) == {
        "gripper": 3,
        "manipulated_object": 3,
    }
    np.testing.assert_array_equal(sampled["eligible_body_ids"], sorted((object_id, gripper_id)))
    with pytest.raises(ValueError, match="include_background must be False"):
        sample_aligned_kubric_points_for_frame(
            env,
            ("camera",),
            img_width=6,
            img_height=4,
            max_points=2,
            include_background=True,
            eligible_body_ids={object_id, gripper_id},
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
