"""Utilities for generating point tracking data from MuJoCo simulations.

Samples mesh vertices from non-world bodies, tracks them through simulation
by maintaining body-local coordinates, projects to 2D per camera, and
determines visibility via depth-buffer comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import numpy as np
from mujoco import MjData, MjModel

log = logging.getLogger(__name__)


def sample_mesh_vertices(
    model: MjModel,
    data: MjData,
    max_points: int = 5000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Collect mesh vertices from all non-world bodies and subsample.

    Allocates an equal point budget to every body, then randomly samples
    actual mesh vertices within each body so every object is represented
    and points sit exactly on the mesh surface.

    Returns:
        local_coords: (N, 3) body-local coordinates for tracking
        body_ids: (N,) int32 body id each point belongs to
        world_coords: (N, 3) initial world positions
        total_verts: total vertex count before subsampling
    """
    rng = np.random.RandomState(seed)

    per_body_verts: dict[int, list[np.ndarray]] = {}

    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH.value:
            continue

        body_id = model.geom_bodyid[geom_id]
        if body_id == 0:
            continue

        mesh_id = model.geom_dataid[geom_id]
        vertadr = model.mesh_vertadr[mesh_id]
        n_vert = model.mesh_vertnum[mesh_id]
        if n_vert == 0:
            continue

        verts_mesh_local = model.mesh_vert[vertadr : vertadr + n_vert]

        geom_pos = model.geom_pos[geom_id]
        geom_quat = model.geom_quat[geom_id]
        geom_rot = np.zeros((3, 3))
        mujoco.mju_quat2Mat(geom_rot.ravel(), geom_quat)
        verts_body_local = verts_mesh_local @ geom_rot.T + geom_pos

        if body_id not in per_body_verts:
            per_body_verts[body_id] = []
        per_body_verts[body_id].append(verts_body_local)

    if not per_body_verts:
        log.warning("No mesh vertices found on non-world bodies")
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            0,
        )

    for bid in per_body_verts:
        per_body_verts[bid] = np.concatenate(per_body_verts[bid], axis=0)

    total_verts = sum(v.shape[0] for v in per_body_verts.values())
    n_bodies = len(per_body_verts)

    body_ids_sorted = sorted(per_body_verts.keys())

    # When more bodies than budget, pick a random subset of bodies.
    if n_bodies > max_points:
        chosen_bodies = rng.choice(body_ids_sorted, size=max_points, replace=False)
        chosen_bodies.sort()
        body_ids_sorted = chosen_bodies.tolist()
        n_bodies = max_points

    per_body = max(1, max_points // n_bodies)
    remainder = max_points - per_body * n_bodies

    all_local = []
    all_body_ids = []
    all_world = []

    for i, bid in enumerate(body_ids_sorted):
        verts = per_body_verts[bid]
        n_alloc = per_body + (1 if i < remainder else 0)
        replace = n_alloc > len(verts)
        indices = rng.choice(len(verts), size=n_alloc, replace=replace)
        sampled = verts[indices]

        body_rot = data.xmat[bid].reshape(3, 3)
        body_pos = data.xpos[bid]
        world = sampled @ body_rot.T + body_pos

        all_local.append(sampled.astype(np.float32))
        all_body_ids.append(np.full(n_alloc, bid, dtype=np.int32))
        all_world.append(world.astype(np.float32))

    local_coords = np.concatenate(all_local, axis=0)
    body_ids = np.concatenate(all_body_ids, axis=0)
    world_coords = np.concatenate(all_world, axis=0)

    log.info(
        f"Sampled {len(local_coords)} point tracks from {n_bodies} bodies "
        f"({total_verts} total mesh vertices)"
    )

    return local_coords, body_ids, world_coords, total_verts


def get_object_body_ids(model: MjModel) -> set[int]:
    """Return body IDs of all manipulable objects (bodies with free joints + descendants)."""
    object_bids: set[int] = set()
    freejoints = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    root_bids = set(int(model.jnt_bodyid[j]) for j in freejoints)

    for root in root_bids:
        queue = [root]
        while queue:
            bid = queue.pop()
            object_bids.add(bid)
            children = np.where(model.body_parentid == bid)[0]
            queue.extend(int(c) for c in children)

    return object_bids


def get_robot_body_ids(model: MjModel, namespace: str = "robot_0/") -> set[int]:
    """Return body IDs of all robot bodies (identified by name prefix)."""
    robot_bids: set[int] = set()
    for bid in range(model.nbody):
        name = model.body(bid).name
        if name.startswith(namespace):
            robot_bids.add(bid)
    return robot_bids


def get_trackable_body_ids(model: MjModel) -> set[int]:
    """Return body IDs suitable for point tracking: objects + robot arm."""
    return get_object_body_ids(model) | get_robot_body_ids(model)


def sample_from_image(
    model: MjModel,
    data: MjData,
    camera,
    img_width: int,
    img_height: int,
    depth_frame: np.ndarray,
    seg_frame: np.ndarray,
    max_points: int = 256,
    seed: int = 0,
    object_body_ids: set[int] | None = None,
    prefer_body_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sample tracked points by picking visible pixels on objects.

    Like Kubric/TAP-Vid: pick random visible pixels from the rendered image,
    use segmentation to find which body they belong to, unproject to 3D, and
    convert to body-local coordinates for tracking. Every sampled point is
    guaranteed visible in this frame.

    Args:
        model: MjModel for mesh geometry lookup
        data: MjData with current body poses
        camera: Camera object (pos, forward, up, fov)
        img_width, img_height: rendered image dimensions
        depth_frame: (H, W) float32 metric depth
        seg_frame: (H, W, 3) int32 segmentation [geom_id, obj_type, body_id]
        max_points: number of points to sample
        seed: random seed
        object_body_ids: if provided, only sample from these body IDs
            (typically free-joint objects). If None, samples from all
            non-world bodies.
        prefer_body_ids: if provided, sample from these bodies first.
            Only falls back to other valid bodies if preferred bodies
            don't fill the budget.

    Returns:
        local_coords: (N, 3) body-local coords for tracking
        body_ids: (N,) int32 body id per point
        world_coords: (N, 3) initial world positions
        total_verts: total mesh vertex count across non-world bodies
    """
    rng = np.random.RandomState(seed)

    body_id_map = seg_frame[:, :, 2]  # (H, W) body id per pixel
    if object_body_ids is not None:
        valid_mask = np.isin(body_id_map, list(object_body_ids))
    else:
        valid_mask = body_id_map > 0

    if prefer_body_ids is not None and len(prefer_body_ids) > 0:
        prefer_mask = valid_mask & np.isin(body_id_map, list(prefer_body_ids))
        prefer_ys, prefer_xs = np.where(prefer_mask)
        if len(prefer_ys) > 0:
            n_prefer = min(max_points, len(prefer_ys))
            chosen_pref = rng.choice(len(prefer_ys), size=n_prefer, replace=False)
            leftover = max_points - n_prefer
            if leftover > 0:
                other_mask = valid_mask & ~prefer_mask
                other_ys, other_xs = np.where(other_mask)
                if len(other_ys) > 0:
                    n_other = min(leftover, len(other_ys))
                    chosen_other = rng.choice(len(other_ys), size=n_other, replace=False)
                    valid_ys = np.concatenate([prefer_ys[chosen_pref], other_ys[chosen_other]])
                    valid_xs = np.concatenate([prefer_xs[chosen_pref], other_xs[chosen_other]])
                else:
                    valid_ys = prefer_ys[chosen_pref]
                    valid_xs = prefer_xs[chosen_pref]
            else:
                valid_ys = prefer_ys[chosen_pref]
                valid_xs = prefer_xs[chosen_pref]
        else:
            valid_ys, valid_xs = np.where(valid_mask)
    else:
        valid_ys, valid_xs = np.where(valid_mask)

    if len(valid_ys) == 0:
        log.warning("No non-world body pixels visible — falling back to vertex sampling")
        return sample_mesh_vertices(model, data, max_points, seed)

    n_pick = min(max_points, len(valid_ys))
    chosen = rng.choice(len(valid_ys), size=n_pick, replace=False)
    pxs = valid_xs[chosen]
    pys = valid_ys[chosen]
    pixel_bids = body_id_map[pys, pxs]
    pixel_depths = depth_frame[pys, pxs]

    # Unproject pixels to 3D world coordinates
    cam2world = camera.get_pose()
    fovy_rad = np.radians(camera.fov)
    fy = (img_height / 2.0) / np.tan(fovy_rad / 2.0)
    fx = fy
    cx, cy = img_width / 2.0, img_height / 2.0

    # Camera-space 3D (X=right, Y=down, Z=forward)
    cam_x = (pxs.astype(np.float64) - cx) / fx * pixel_depths
    cam_y = (pys.astype(np.float64) - cy) / fy * pixel_depths
    cam_z = pixel_depths.astype(np.float64)
    pts_cam = np.stack([cam_x, cam_y, cam_z], axis=1)  # (N, 3)

    R = cam2world[:3, :3]
    t = cam2world[:3, 3]
    world_pts = (pts_cam @ R.T + t).astype(np.float32)

    # Count total mesh verts for metadata
    total_verts = 0
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH.value:
            continue
        if model.geom_bodyid[geom_id] == 0:
            continue
        total_verts += model.mesh_vertnum[model.geom_dataid[geom_id]]

    # Convert world points to body-local coordinates for rigid tracking.
    # Uses the exact unprojected surface point (no vertex snapping) so
    # every point is guaranteed visible in the sampling frame.
    local_coords = np.empty((n_pick, 3), dtype=np.float32)
    final_bids = pixel_bids.astype(np.int32)

    for bid in np.unique(final_bids):
        mask = final_bids == bid
        body_rot = data.xmat[bid].reshape(3, 3)
        body_pos = data.xpos[bid]
        local_coords[mask] = ((world_pts[mask] - body_pos) @ body_rot).astype(np.float32)

    log.info(
        f"Image-sampled {len(local_coords)} point tracks from "
        f"{len(np.unique(final_bids))} bodies ({total_verts} total mesh vertices)"
    )

    return local_coords, final_bids, world_pts, total_verts


def _build_camera_matrices(camera, img_width: int, img_height: int):
    """Build view matrix and intrinsics from a Camera object.

    Uses the same convention as CameraParameterSensor: cam2world from
    Camera.get_pose(), intrinsics from vertical FOV.

    Returns:
        world2cam: (4, 4) view matrix
        intrinsics: (3, 3) camera intrinsic matrix
    """
    cam2world = camera.get_pose()
    world2cam = np.linalg.inv(cam2world)

    fovy_rad = np.radians(camera.fov)
    fy = (img_height / 2.0) / np.tan(fovy_rad / 2.0)
    fx = fy
    cx = img_width / 2.0
    cy = img_height / 2.0

    intrinsics = np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
    )

    return world2cam, intrinsics


def track_points_for_frame(
    data,
    local_coords: np.ndarray,
    body_ids: np.ndarray,
    camera,
    img_width: int,
    img_height: int,
    depth_frame: np.ndarray,
    occlusion_tolerance: float = 0.03,
    precomputed_w2c: np.ndarray | None = None,
    precomputed_intrinsics: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D projections and visibility for all tracked points in one frame.

    Args:
        data: Object with .xpos and .xmat arrays (MjData or compatible)
        local_coords: (N, 3) body-local coordinates
        body_ids: (N,) body id per point
        camera: Camera object (used to build matrices if precomputed not given)
        img_width: Image width in pixels
        img_height: Image height in pixels
        depth_frame: (H, W) float32 rendered depth in meters
        occlusion_tolerance: Depth comparison tolerance in meters
        precomputed_w2c: Optional (4, 4) precomputed world-to-camera matrix
        precomputed_intrinsics: Optional (3, 3) precomputed intrinsic matrix

    Returns:
        coords_2d: (N, 2) float32 pixel coordinates
        visibility: (N,) float32 (1.0=visible, 0.0=occluded/oob)
        world_pts: (N, 3) float32 current world positions
    """
    N = len(local_coords)

    world_pts = np.empty((N, 3), dtype=np.float32)
    unique_bodies = np.unique(body_ids)
    for bid in unique_bodies:
        mask = body_ids == bid
        body_rot = data.xmat[bid].reshape(3, 3)
        body_pos = data.xpos[bid]
        world_pts[mask] = (local_coords[mask] @ body_rot.T + body_pos).astype(np.float32)

    if precomputed_w2c is not None and precomputed_intrinsics is not None:
        world2cam = precomputed_w2c
        intrinsics = precomputed_intrinsics
    else:
        world2cam, intrinsics = _build_camera_matrices(camera, img_width, img_height)

    pts_h = np.hstack([world_pts, np.ones((N, 1), dtype=np.float32)])
    pts_cam = (world2cam @ pts_h.T).T[:, :3]

    depths = pts_cam[:, 2]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    safe_z = np.where(depths < 1e-4, 1e-4, depths)
    px = fx * pts_cam[:, 0] / safe_z + cx
    py = fy * pts_cam[:, 1] / safe_z + cy

    coords_2d = np.stack([px, py], axis=1).astype(np.float32)

    in_frame = (
        np.isfinite(px)
        & np.isfinite(py)
        & (px >= 0)
        & (px < img_width)
        & (py >= 0)
        & (py < img_height)
        & (depths > 0)
    )

    visibility = np.zeros(N, dtype=np.float32)
    if depth_frame is not None and depth_frame.size > 0:
        in_frame_indices = np.where(in_frame)[0]
        if len(in_frame_indices) > 0:
            px_int = np.clip(px[in_frame_indices].astype(int), 0, img_width - 1)
            py_int = np.clip(py[in_frame_indices].astype(int), 0, img_height - 1)
            rendered_depth = depth_frame[py_int, px_int]
            point_depth = depths[in_frame_indices]
            not_occluded = (point_depth - rendered_depth) < occlusion_tolerance
            visibility[in_frame_indices] = np.where(not_occluded, 1.0, 0.0)
    else:
        visibility[in_frame] = 1.0

    return coords_2d, visibility, world_pts


def save_point_tracks(
    save_path: Path,
    trajs_2d: np.ndarray,
    visibility: np.ndarray,
    points_3d_initial: np.ndarray | None,
    points_3d: np.ndarray,
    body_ids: np.ndarray,
    intrinsics: np.ndarray,
    total_mesh_verts: int | None,
    query_frames: np.ndarray | None = None,
) -> None:
    """Save point tracking data to a compressed npz file."""
    data = dict(
        trajs_2d=trajs_2d.astype(np.float32),
        visibility=visibility.astype(np.float32),
        points_3d=points_3d.astype(np.float32),
        body_ids=body_ids.astype(np.int32),
        intrinsics=intrinsics.astype(np.float32),
    )
    if points_3d_initial is not None:
        data["points_3d_initial"] = points_3d_initial.astype(np.float32)
    if total_mesh_verts is not None:
        data["num_sampled_from"] = np.array(total_mesh_verts, dtype=np.int32)
    if query_frames is not None:
        data["query_frames"] = np.asarray(query_frames, dtype=np.int32)
    np.savez_compressed(save_path, **data)
