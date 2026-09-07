from types import SimpleNamespace

import mujoco
import numpy as np

from molmo_spaces.configs.camera_configs import (
    FrankaRandomizedDroidCameraSystem,
    MjcfCameraConfig,
    RandomizedExocentricCameraConfig,
)
from molmo_spaces.env.camera_manager import CameraManager


def test_camera_geometry_clearance_rejects_nearby_surface() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="2 2 0.1"/>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(current_model=model, current_data=data)

    assert CameraManager.camera_geometry_clearance(env, np.array([0.0, 0.0, 0.05])) < 0.10
    assert np.isclose(
        CameraManager.camera_geometry_clearance(env, np.array([0.0, 0.0, 0.20])),
        0.20,
    )


def test_camera_near_geometry_fraction_uses_metric_depth() -> None:
    manager = CameraManager()
    env = SimpleNamespace(
        render_depth_frame=lambda _camera_name: np.array(
            [[0.10, 0.20], [0.40, np.inf]], dtype=np.float32
        )
    )

    fraction = manager.camera_near_geometry_fraction(
        env,
        "test_camera",
        np.zeros(3),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 1.0, 0.0]),
        52.0,
        0.35,
    )

    assert fraction == 0.5
    assert not manager.registry.cameras


def test_randomized_droid_cameras_share_fov_and_have_clearance() -> None:
    cameras = FrankaRandomizedDroidCameraSystem().cameras

    assert len(cameras) == 4
    assert isinstance(cameras[0], MjcfCameraConfig)
    assert cameras[0].fov == 52.0
    assert cameras[0].fov_noise_degrees is None

    for camera in cameras[1:]:
        assert isinstance(camera, RandomizedExocentricCameraConfig)
        assert camera.fov == 52.0
        assert camera.fov_range is None
        assert camera.min_geometry_clearance == 0.10
        assert camera.near_geometry_distance == 0.35
        assert camera.max_near_geometry_fraction == 0.10
