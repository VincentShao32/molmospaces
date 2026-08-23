from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np

from molmo_spaces.utils.point_tracking_utils import (
    _random_phase_strided_mask,
    get_object_body_ids,
    get_robot_body_ids,
    get_trackable_body_ids,
    sample_from_image,
    save_point_tracks,
    track_points_for_frame,
)


class IdentityCamera:
    fov = 90.0

    @staticmethod
    def get_pose() -> np.ndarray:
        return np.eye(4, dtype=np.float64)


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


def test_strided_mask_is_seeded_and_uses_one_grid_phase():
    mask_a = _random_phase_strided_mask(7, 9, 3, np.random.RandomState(11))
    mask_b = _random_phase_strided_mask(7, 9, 3, np.random.RandomState(11))

    assert np.array_equal(mask_a, mask_b)
    ys, xs = np.where(mask_a)
    assert len(ys) > 0
    assert len(set(ys % 3)) == 1
    assert len(set(xs % 3)) == 1
    assert _random_phase_strided_mask(7, 9, 1, np.random.RandomState(11)) is None


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
    )

    with np.load(output) as saved:
        assert saved["trajs_2d"].dtype == np.float32
        assert saved["body_ids"].dtype == np.int32
        np.testing.assert_array_equal(saved["query_frames"], [0, 1, 1])
        assert int(saved["num_sampled_from"]) == 12
