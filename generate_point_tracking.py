"""Generate synthetic point tracking data from MolmoSpaces scenes.

Loads MuJoCo scenes, renders RGB + depth from a moving camera,
samples 3D surface points on objects, and tracks their 2D projections.

Usage:
    export MLSPACES_CACHE_DIR=/gpfs/scrubbed/yunbos/video_datasets/molmospaces/cache
    export MLSPACES_ASSETS_DIR=/gpfs/scrubbed/yunbos/video_datasets/molmospaces/molmospaces/assets
    export MUJOCO_GL=egl
    python generate_point_tracking.py --output_dir /path/to/output
"""

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm


def make_lookat_view(cam_pos, lookat, up_hint=np.array([0.0, 0.0, 1.0])):
    """Build a 4x4 world-to-camera view matrix."""
    forward = lookat - cam_pos
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-12

    R = np.stack([right, up, -forward], axis=0)
    view = np.eye(4)
    view[:3, :3] = R
    view[:3, 3] = -R @ cam_pos
    return view


def project_points(points_3d, view, fx, fy, cx, cy):
    """Project 3D world points to 2D pixel coords. Returns (N,2) coords and (N,) depths."""
    N = points_3d.shape[0]
    pts_h = np.hstack([points_3d, np.ones((N, 1))])
    pts_cam = (view @ pts_h.T).T[:, :3]

    depths = -pts_cam[:, 2]
    safe_z = np.where(depths < 1e-4, 1e-4, depths)

    px = fx * pts_cam[:, 0] / (-pts_cam[:, 2].clip(-np.inf, -1e-4)) + cx
    py = fy * (-pts_cam[:, 1]) / (-pts_cam[:, 2].clip(-np.inf, -1e-4)) + cy

    return np.stack([px, py], axis=1), depths


def check_occlusion_raycast(model, data, cam_pos, points_3d):
    """Check occlusion using raycasting from camera to each point.
    Returns (N,) boolean array where True = visible."""
    N = points_3d.shape[0]
    visible = np.zeros(N, dtype=bool)

    for i in range(N):
        direction = points_3d[i] - cam_pos
        dist_to_point = np.linalg.norm(direction)
        if dist_to_point < 1e-6:
            continue
        direction /= dist_to_point

        geomid = np.array([-1], dtype=np.int32)
        hit_dist = mujoco.mj_ray(
            model, data, cam_pos, direction, None, 1, -1, geomid
        )

        if hit_dist > 0 and abs(hit_dist - dist_to_point) < 0.02:
            visible[i] = True

    return visible


def sample_points_on_geoms(model, data, num_points=256, seed=42):
    """Sample 3D points on object surfaces by raycasting from the scene center."""
    rng = np.random.RandomState(seed)
    center = model.stat.center.copy()
    extent = model.stat.extent

    points = []
    geom_ids_list = []
    body_ids_list = []

    max_attempts = num_points * 100
    for _ in range(max_attempts):
        if len(points) >= num_points:
            break

        origin = center + rng.uniform(-extent * 0.8, extent * 0.8, 3)
        direction = rng.randn(3)
        direction /= np.linalg.norm(direction)

        geomid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(model, data, origin, direction, None, 1, -1, geomid)

        if dist > 0 and dist < 2 * extent and geomid[0] >= 0:
            hit_point = origin + dist * direction
            body_id = model.geom_bodyid[geomid[0]]
            # Only track points on non-world bodies
            if body_id > 0:
                points.append(hit_point)
                geom_ids_list.append(geomid[0])
                body_ids_list.append(body_id)

    if len(points) < num_points:
        print(f"  Warning: sampled {len(points)}/{num_points} points on non-world bodies")

    return np.array(points), np.array(geom_ids_list), np.array(body_ids_list)


def compute_local_coords(data, points_3d, body_ids):
    """Convert world points to body-local coordinates for tracking."""
    local_coords = np.zeros_like(points_3d)
    for i in range(len(points_3d)):
        bid = body_ids[i]
        body_pos = data.xpos[bid]
        body_rot = data.xmat[bid].reshape(3, 3)
        local_coords[i] = body_rot.T @ (points_3d[i] - body_pos)
    return local_coords


def transform_local_to_world(data, local_coords, body_ids):
    """Transform body-local coordinates back to world coordinates."""
    world_pts = np.zeros_like(local_coords)
    for i in range(len(local_coords)):
        bid = body_ids[i]
        body_pos = data.xpos[bid]
        body_rot = data.xmat[bid].reshape(3, 3)
        world_pts[i] = body_rot @ local_coords[i] + body_pos
    return world_pts


def find_valid_camera_positions(model, data, center, num_frames, radius=2.0, height=1.5,
                                num_candidates=360):
    """Find camera positions inside the room with clear line-of-sight to center.

    Casts rays outward from center to find wall distances, then places the
    camera well inside those limits.
    """
    lookat = center.copy()
    lookat[2] = height

    valid_positions = []
    angles = np.linspace(0, 2 * np.pi, num_candidates, endpoint=False)

    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        geomid = np.array([-1], dtype=np.int32)
        wall_dist = mujoco.mj_ray(model, data, lookat, direction, None, 1, -1, geomid)

        if wall_dist <= 0:
            max_r = radius
        else:
            max_r = min(radius, wall_dist * 0.6)

        if max_r < 0.3:
            continue

        cam_pos = lookat + direction * max_r
        cam_pos[2] = height

        # Verify line-of-sight back to center
        to_center = lookat - cam_pos
        dist_to_center = np.linalg.norm(to_center)
        to_center_norm = to_center / (dist_to_center + 1e-8)

        geomid2 = np.array([-1], dtype=np.int32)
        hit = mujoco.mj_ray(model, data, cam_pos, to_center_norm, None, 1, -1, geomid2)

        if hit <= 0 or hit >= dist_to_center * 0.95:
            valid_positions.append(cam_pos.copy())

    if len(valid_positions) < num_frames:
        print(f"  Warning: only {len(valid_positions)} valid camera positions found")
        if len(valid_positions) == 0:
            # Fallback: place camera at center looking around
            valid_positions = [center.copy() + np.array([0, 0, 0.5])] * num_frames

    # Sample num_frames positions evenly from the valid set
    indices = np.linspace(0, len(valid_positions) - 1, num_frames, dtype=int)
    return np.array([valid_positions[i] for i in indices])


def perturb_objects(model, data, rng, strength=5.0):
    """Apply random impulses to movable objects to create motion."""
    for body_id in range(1, model.nbody):
        if model.body_jntnum[body_id] > 0:
            jnt_start = model.body_jntadr[body_id]
            jnt_type = model.jnt_type[jnt_start]
            # Free joints (type 0) can translate and rotate
            if jnt_type == 0:
                data.qvel[model.jnt_dofadr[jnt_start]:model.jnt_dofadr[jnt_start] + 3] = (
                    rng.randn(3) * strength
                )
            # Hinge joints (type 3) can rotate
            elif jnt_type == 3:
                data.qvel[model.jnt_dofadr[jnt_start]] = rng.randn() * strength * 0.5


def generate_episode(
    scene_xml_path,
    output_dir,
    episode_id=0,
    num_frames=150,
    num_points=256,
    width=640,
    height=480,
    cam_radius=3.0,
    cam_height=1.5,
    warmup_steps=50,
):
    """Generate one episode of point tracking data."""
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    data = mujoco.MjData(model)

    # Warm up physics so objects settle under gravity
    for _ in range(warmup_steps):
        mujoco.mj_step(model, data)

    scene_center = model.stat.center.copy()
    scene_extent = model.stat.extent

    # Apply random perturbations to create motion
    rng = np.random.RandomState(episode_id + 42)
    perturb_objects(model, data, rng, strength=3.0)

    # Let perturbations propagate a bit
    for _ in range(10):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    # Sample points on object surfaces
    points_3d, geom_ids, body_ids = sample_points_on_geoms(
        model, data, num_points=num_points, seed=episode_id
    )
    if len(points_3d) == 0:
        print("  Skipping: no surface points found")
        return False

    # Store points in body-local coordinates for tracking through motion
    local_coords = compute_local_coords(data, points_3d, body_ids)

    # Camera trajectory: find positions inside the room
    cam_positions = find_valid_camera_positions(
        model, data, scene_center, num_frames,
        radius=min(cam_radius, scene_extent * 0.4),
        height=cam_height,
    )

    fovy_deg = 60.0
    fovy_rad = np.deg2rad(fovy_deg)
    fy = height / (2.0 * np.tan(fovy_rad / 2.0))
    fx = fy
    cx, cy = width / 2.0, height / 2.0
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    renderer = mujoco.FilamentRenderer(model, height=height, width=width)

    ep_dir = Path(output_dir) / f"episode_{episode_id:05d}"
    rgb_dir = ep_dir / "rgbs"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    N = len(points_3d)
    all_trajs_2d = np.zeros((num_frames, N, 2), dtype=np.float32)
    all_visibility = np.zeros((num_frames, N), dtype=np.float32)

    for t in tqdm(range(num_frames), desc=f"  Ep {episode_id}", leave=False):
        mujoco.mj_step(model, data)

        # Current 3D positions of tracked points (attached to bodies)
        current_pts = transform_local_to_world(data, local_coords, body_ids)

        # Camera for this frame
        cam_pos = cam_positions[t]
        lookat = scene_center.copy()
        lookat[2] = cam_height * 0.6  # look slightly below center

        view = make_lookat_view(cam_pos, lookat)

        # Render RGB using a programmatic camera via MjvCamera
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = np.linalg.norm(cam_pos - lookat)
        # Compute azimuth and elevation from cam_pos relative to lookat
        delta = cam_pos - lookat
        cam.azimuth = np.degrees(np.arctan2(delta[1], delta[0]))
        cam.elevation = np.degrees(np.arcsin(
            np.clip(delta[2] / (np.linalg.norm(delta) + 1e-8), -1, 1)
        ))

        renderer.update_scene(data, camera=cam)
        rgb = renderer.render().copy()

        # Project points to 2D
        coords_2d, depths = project_points(current_pts, view, fx, fy, cx, cy)
        all_trajs_2d[t] = coords_2d

        # Check occlusion via raycasting
        visible = check_occlusion_raycast(model, data, cam_pos, current_pts)

        # Also check if points are in frame
        for p in range(N):
            px_val, py_val = coords_2d[p]
            in_frame = (
                np.isfinite(px_val) and np.isfinite(py_val)
                and 0 <= px_val < width and 0 <= py_val < height
                and depths[p] > 0
            )
            all_visibility[t, p] = 1.0 if (in_frame and visible[p]) else 0.0

        np.save(rgb_dir / f"rgb_{t:05d}.npy", rgb)

    np.savez_compressed(
        ep_dir / "trajectory.npz",
        trajs_2d=all_trajs_2d,        # (T, N, 2) pixel coordinates
        visibility=all_visibility,      # (T, N) 1.0=visible 0.0=occluded/oob
        points_3d_initial=points_3d,   # (N, 3) initial world positions
        body_ids=body_ids,             # (N,) which body each point belongs to
        intrinsics=intrinsics,         # (3, 3) camera intrinsic matrix
    )

    renderer.close()
    print(f"  Saved episode {episode_id} -> {ep_dir} ({N} points, "
          f"avg visible: {all_visibility.mean():.1%})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate point tracking data from MolmoSpaces")
    parser.add_argument("--output_dir", type=str,
                        default="/gpfs/scrubbed/yunbos/video_datasets/molmospaces/point_tracking_data")
    parser.add_argument("--dataset", type=str, default="ithor",
                        choices=["ithor", "procthor-10k", "procthor-objaverse", "holodeck-objaverse"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=150)
    parser.add_argument("--num_points", type=int, default=256)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--cam_radius", type=float, default=3.0)
    parser.add_argument("--cam_height", type=float, default=1.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from molmo_spaces.molmo_spaces_constants import get_scenes
    from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path

    print(f"Loading {args.dataset} scenes ({args.split} split)...")
    scenes = get_scenes(args.dataset, args.split)

    scene_paths = []
    for split_name, split_data in scenes.items():
        if isinstance(split_data, dict):
            for idx, val in split_data.items():
                if val is None:
                    continue
                if isinstance(val, dict):
                    path = val.get("base")
                else:
                    path = val
                if path is not None:
                    scene_paths.append(path)
        elif isinstance(split_data, list):
            scene_paths.extend([p for p in split_data if p is not None])

    if not scene_paths:
        print("No scenes found!")
        return

    print(f"Found {len(scene_paths)} scenes")

    end_idx = min(args.start_idx + args.num_episodes, len(scene_paths))
    success_count = 0

    for ep_id, scene_idx in enumerate(range(args.start_idx, end_idx)):
        scene_path = scene_paths[scene_idx]
        print(f"\nEpisode {ep_id}: {Path(scene_path).name}")

        try:
            install_scene_with_objects_and_grasps_from_path(scene_path)
        except Exception as e:
            print(f"  Failed to install scene: {e}")
            continue

        if not Path(scene_path).exists():
            print(f"  Scene file not found after install")
            continue

        try:
            ok = generate_episode(
                scene_xml_path=scene_path,
                output_dir=args.output_dir,
                episode_id=ep_id,
                num_frames=args.num_frames,
                num_points=args.num_points,
                width=args.width,
                height=args.height,
                cam_radius=args.cam_radius,
                cam_height=args.cam_height,
            )
            if ok:
                success_count += 1
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Generated {success_count}/{end_idx - args.start_idx} episodes -> {args.output_dir}")


if __name__ == "__main__":
    main()
