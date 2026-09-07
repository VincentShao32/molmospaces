"""Utilities for generating point tracking data from MuJoCo simulations.

Samples mesh vertices or rendered pixels, tracks their body-local coordinates
through simulation, and projects them into each camera. Kubric image sampling
balances space-time candidates by logical object and uses geometry identity
and depth to distinguish visibility, occlusion, and uncertain raster support.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
from mujoco import MjData, MjModel

log = logging.getLogger(__name__)

FAILURE_TARGET_BUCKET_NAMES = (
    "occlusion_edge",
    "cross_view_occlusion",
    "object_edge",
    "small_thin",
    "baseline",
)
DEFAULT_FAILURE_TARGET_FRACTIONS = (0.20, 0.15, 0.20, 0.20, 0.25)
DEFAULT_VISIBILITY_DEPTH_RELATIVE_TOLERANCE = 0.01
DEFAULT_VISIBILITY_DEPTH_ABSOLUTE_TOLERANCE_M = 0.001
RASTER_VISIBILITY_METHOD = "four_neighbor_exact_geom_depth_support_v1"
POINT_TRACK_VISIBILITY_REASON_NAMES = (
    "visible",
    "out_of_frame",
    "occluded_depth_confirmed",
    "raster_ambiguous",
)


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


def get_body_subtree_ids(model: MjModel, root_body_id: int) -> set[int]:
    """Return one MuJoCo body and every descendant in its kinematic subtree."""
    root_body_id = int(root_body_id)
    if not 0 <= root_body_id < model.nbody:
        raise ValueError(f"Invalid root body id {root_body_id} for nbody={model.nbody}")
    body_ids: set[int] = set()
    queue = [root_body_id]
    while queue:
        body_id = queue.pop()
        if body_id in body_ids:
            continue
        body_ids.add(body_id)
        queue.extend(int(child_id) for child_id in np.flatnonzero(model.body_parentid == body_id))
    return body_ids


def get_manipulation_target_body_ids(task) -> tuple[set[int], dict[int, str]]:
    """Resolve the active gripper and manipulated-object body subtrees.

    The pickup object is taken from ``task.config.task_config.pickup_obj_name``;
    the gripper roots come from the robot view's registered gripper move groups.
    Labels are returned alongside the union so saved NPZ files can verify the
    selected target class without relying on body-name reconstruction later.
    """
    env = task.env
    model = env.current_model
    task_config = task.config.task_config
    pickup_name = getattr(task_config, "pickup_obj_name", None)
    if not pickup_name:
        raise ValueError("gripper_and_pickup point targeting requires task_config.pickup_obj_name")

    object_manager = env.object_managers[env.current_batch_index]
    pickup_object = object_manager.get_object_by_name(pickup_name)
    pickup_body_ids = {int(body_id) for body_id in pickup_object.body_ids}
    if not pickup_body_ids:
        raise RuntimeError(f"Pickup object {pickup_name!r} has no MuJoCo bodies")

    robot_view = env.current_robot.robot_view
    gripper_body_ids: set[int] = set()
    for move_group_id in robot_view.get_gripper_movegroup_ids():
        gripper_group = robot_view.get_move_group(move_group_id)
        gripper_body_ids.update(get_body_subtree_ids(model, gripper_group.root_body_id))
    if not gripper_body_ids:
        raise RuntimeError("The active robot has no gripper body subtree")

    labels = {body_id: "gripper" for body_id in gripper_body_ids}
    labels.update({body_id: "manipulated_object" for body_id in pickup_body_ids})
    return gripper_body_ids | pickup_body_ids, labels


def get_kubric_segment_ids(
    model: MjModel,
    body_ids: np.ndarray,
    foreground_body_ids: set[int],
) -> np.ndarray:
    """Map rigid body IDs to Kubric-like logical instance segments.

    Every non-foreground body is collapsed into background segment 0. Bodies
    in the same connected foreground tree (for example, object descendants or
    robot links) share the highest foreground ancestor as their segment, while
    their original body IDs remain available for rigid point tracking.
    """
    body_ids = np.asarray(body_ids, dtype=np.int32)
    segment_ids = np.zeros_like(body_ids)
    foreground = {int(body_id) for body_id in foreground_body_ids}
    for body_id in np.unique(body_ids):
        body_id = int(body_id)
        if body_id not in foreground:
            continue
        segment_root = body_id
        parent_id = int(model.body_parentid[segment_root])
        while parent_id > 0 and parent_id in foreground:
            segment_root = parent_id
            parent_id = int(model.body_parentid[segment_root])
        segment_ids[body_ids == body_id] = segment_root
    return segment_ids


def get_kubric_segment_names(
    model: MjModel,
    segment_ids: np.ndarray,
) -> np.ndarray:
    """Return one durable logical-object label per Kubric-like segment ID.

    Segment zero is the single collapsed background class. Foreground segment
    IDs are logical body roots, whose MuJoCo names remain meaningful after the
    simulator is gone and allow downstream rows to retain object membership.
    """
    segment_ids = np.asarray(segment_ids, dtype=np.int32)
    names: list[str] = []
    for segment_id in segment_ids:
        segment_id = int(segment_id)
        if segment_id == 0:
            names.append("background")
            continue
        if not 0 < segment_id < model.nbody:
            raise ValueError(f"Invalid logical segment id {segment_id} for nbody={model.nbody}")
        names.append(model.body(segment_id).name or f"body_{segment_id}")
    return np.asarray(names, dtype=str)


def _random_phase_strided_mask(
    height: int, width: int, stride: int, rng: np.random.RandomState
) -> np.ndarray | None:
    """Boolean (H, W) mask: True on pixels aligned to a stride grid with random origin.

    Matches the Kubric-style idea: only pixels whose indices satisfy
    ``(y - oy) % stride == 0`` and ``(x - ox) % stride == 0`` for random
    ``oy, ox`` in ``[0, stride)``. Returns ``None`` when ``stride <= 1`` (no mask).
    """
    if stride <= 1:
        return None
    oy = int(rng.randint(0, stride))
    ox = int(rng.randint(0, stride))
    yy = np.arange(height, dtype=np.int32)[:, None]
    xx = np.arange(width, dtype=np.int32)[None, :]
    return ((yy - oy) % stride == 0) & ((xx - ox) % stride == 0)


def _apply_strided_mask(mask: np.ndarray, grid: np.ndarray | None) -> np.ndarray:
    if grid is None:
        return mask
    return mask & grid


def _fixed_phase_strided_mask(
    height: int, width: int, stride: int, phase_y: int, phase_x: int
) -> np.ndarray | None:
    """Return a spatial stride grid with an explicitly supplied phase."""
    if stride <= 1:
        return None
    phase_y = int(phase_y) % stride
    phase_x = int(phase_x) % stride
    yy = np.arange(height, dtype=np.int32)[:, None]
    xx = np.arange(width, dtype=np.int32)[None, :]
    return ((yy - phase_y) % stride == 0) & ((xx - phase_x) % stride == 0)


def get_kubric_num_to_sample(
    counts: np.ndarray,
    max_sampled_fraction: float,
    tracks_to_sample: int,
) -> np.ndarray:
    """Allocate a point budget using Kubric's per-segment balancing rule.

    Segments are considered from least to most abundant. At each step Kubric
    divides the remaining budget equally over the remaining segments, while
    capping a segment at ``max_sampled_fraction`` of its grid candidates.
    ``np.rint`` matches TensorFlow's round-to-nearest-even behavior.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1:
        raise ValueError(f"counts must be one-dimensional, got {counts.shape}")
    if tracks_to_sample < 0:
        raise ValueError("tracks_to_sample must be non-negative")
    if not 0.0 <= max_sampled_fraction <= 1.0:
        raise ValueError("max_sampled_fraction must be in [0, 1]")
    if len(counts) == 0:
        return np.zeros(0, dtype=np.int32)

    segment_order = np.argsort(counts, kind="stable")
    result = np.zeros(len(counts), dtype=np.int32)
    remaining = int(tracks_to_sample)
    for rank, segment_index in enumerate(segment_order):
        remaining_segments = len(counts) - rank
        wanted = int(np.rint(remaining / remaining_segments))
        capped = int(np.rint(counts[segment_index] * max_sampled_fraction))
        take = min(wanted, capped)
        result[segment_index] = take
        remaining -= take
    return result


def _logical_segment_map(
    model: MjModel,
    body_id_map: np.ndarray,
    foreground_body_ids: set[int],
) -> np.ndarray:
    """Map a rendered body-id image to foreground instances and background 0."""
    return get_kubric_segment_ids(
        model,
        np.asarray(body_id_map, dtype=np.int32).reshape(-1),
        foreground_body_ids,
    ).reshape(body_id_map.shape)


def _segment_boundary_band(
    segment_map: np.ndarray,
    valid_mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Return valid pixels within ``radius`` pixels of a segment boundary."""
    radius = max(0, int(radius))
    if radius == 0:
        return np.zeros_like(valid_mask, dtype=bool)

    height, width = segment_map.shape
    sentinel = np.iinfo(np.int32).min
    padded = np.pad(segment_map, 1, constant_values=sentinel)
    boundary = np.zeros_like(valid_mask, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        boundary |= valid_mask & (neighbor != segment_map)

    band = boundary.copy()
    frontier = boundary
    for _ in range(1, radius):
        padded_frontier = np.pad(frontier, 1, constant_values=False)
        expanded = np.zeros_like(valid_mask, dtype=bool)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            expanded |= padded_frontier[
                1 + dy : 1 + dy + height,
                1 + dx : 1 + dx + width,
            ]
        frontier = expanded & valid_mask & ~band
        band |= frontier
    return band


def candidate_mask_context_features(
    segment_map: np.ndarray,
    query_points: np.ndarray,
    segment_ids: np.ndarray,
    *,
    local_radius_px: int = 4,
    max_edge_distance_px: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure projected segment area, local support, and silhouette distance.

    ``query_points`` follows Kubric ``[t, y, x]`` ordering. Distances are measured
    from each candidate's pixel to the nearest pixel with a different logical
    segment id, up to ``max_edge_distance_px``. Values above that search radius
    are returned as ``max_edge_distance_px + 1``.
    """
    segment_map = np.asarray(segment_map, dtype=np.int32)
    query_points = np.asarray(query_points)
    segment_ids = np.asarray(segment_ids, dtype=np.int32)
    if segment_map.ndim != 2:
        raise ValueError("segment_map must be two-dimensional")
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape (N, 3) in [t, y, x] order")
    if len(query_points) != len(segment_ids):
        raise ValueError("query_points and segment_ids must align")

    n_points = len(segment_ids)
    height, width = segment_map.shape
    ys = np.floor(query_points[:, 1]).astype(np.int64)
    xs = np.floor(query_points[:, 2]).astype(np.int64)
    center_valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)

    unique_segments, counts = np.unique(segment_map, return_counts=True)
    area_lookup = {
        int(segment_id): float(count) / float(segment_map.size)
        for segment_id, count in zip(unique_segments, counts)
    }
    area_fraction = np.fromiter(
        (area_lookup.get(int(segment_id), 0.0) for segment_id in segment_ids),
        dtype=np.float32,
        count=n_points,
    )

    local_radius = max(0, int(local_radius_px))
    edge_radius = max(0, int(max_edge_distance_px))
    search_radius = max(local_radius, edge_radius)
    same_count = np.zeros(n_points, dtype=np.int32)
    local_count = np.zeros(n_points, dtype=np.int32)
    edge_distance_sq = np.full(n_points, np.inf, dtype=np.float32)

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            neighbor_y = ys + dy
            neighbor_x = xs + dx
            valid = (
                center_valid
                & (neighbor_y >= 0)
                & (neighbor_y < height)
                & (neighbor_x >= 0)
                & (neighbor_x < width)
            )
            if not np.any(valid):
                continue
            neighbor_segments = np.zeros(n_points, dtype=np.int32)
            neighbor_segments[valid] = segment_map[neighbor_y[valid], neighbor_x[valid]]
            same = valid & (neighbor_segments == segment_ids)
            if abs(dy) <= local_radius and abs(dx) <= local_radius:
                local_count += valid
                same_count += same
            distance_sq = dx * dx + dy * dy
            if 0 < distance_sq <= edge_radius * edge_radius:
                different = valid & ~same
                edge_distance_sq[different] = np.minimum(
                    edge_distance_sq[different], float(distance_sq)
                )

    local_support = np.divide(
        same_count,
        np.maximum(local_count, 1),
        dtype=np.float32,
    )
    edge_distance = np.full(n_points, float(edge_radius + 1), dtype=np.float32)
    found_edge = np.isfinite(edge_distance_sq)
    edge_distance[found_edge] = np.sqrt(edge_distance_sq[found_edge])
    return area_fraction, local_support, edge_distance


def _balanced_sample_without_replacement(
    candidate_indices: np.ndarray,
    segment_ids: np.ndarray,
    count: int,
    rng: np.random.RandomState,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Sample without replacement while distributing budget across segments.

    Optional weights are divided by their segment's candidate count. Equal
    weights therefore retain segment-balanced behavior, while nonuniform weights
    can softly prioritize candidates without letting large segments dominate.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if count <= 0 or len(candidate_indices) == 0:
        return np.zeros(0, dtype=np.int64)
    count = min(int(count), len(candidate_indices))
    pool_segments = np.asarray(segment_ids, dtype=np.int32)[candidate_indices]
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(segment_ids):
            raise ValueError("sample_weights must align with segment_ids")
        pool_weights = sample_weights[candidate_indices]
        _, inverse, segment_counts = np.unique(
            pool_segments, return_inverse=True, return_counts=True
        )
        pool_weights = np.divide(
            pool_weights,
            segment_counts[inverse],
            out=np.zeros_like(pool_weights),
            where=segment_counts[inverse] > 0,
        )
        pool_weights = np.nan_to_num(pool_weights, nan=0.0, posinf=0.0, neginf=0.0)
        pool_weights = np.maximum(pool_weights, np.finfo(np.float64).tiny)
        pool_weights /= pool_weights.sum()
        return rng.choice(
            candidate_indices,
            size=count,
            replace=False,
            p=pool_weights,
        )

    unique_segments, segment_counts = np.unique(pool_segments, return_counts=True)
    allocations = get_kubric_num_to_sample(segment_counts, 1.0, count)
    selected_parts: list[np.ndarray] = []
    for segment_id, allocation in zip(unique_segments, allocations):
        if allocation <= 0:
            continue
        segment_pool = candidate_indices[pool_segments == segment_id]
        selected_parts.append(rng.choice(segment_pool, size=int(allocation), replace=False))
    selected = np.concatenate(selected_parts) if selected_parts else np.zeros(0, dtype=np.int64)
    if len(selected) < count:
        remaining = np.setdiff1d(candidate_indices, selected, assume_unique=False)
        selected = np.concatenate(
            [
                selected,
                rng.choice(remaining, size=count - len(selected), replace=False),
            ]
        )
    rng.shuffle(selected)
    return selected


def select_failure_targeted_cross_view_shortlist_indices(
    segment_ids: np.ndarray,
    source_cameras: np.ndarray,
    source_edge_distance_px: np.ndarray,
    source_segment_area_fraction: np.ndarray,
    source_local_segment_support: np.ndarray,
    *,
    max_candidates: int,
    seed: int,
    edge_distance_px: float = 4.0,
    small_segment_area_fraction: float = 0.02,
    local_support_threshold: float = 0.60,
    priority_fraction: float = 0.40,
    prioritize_edges: bool = True,
    target_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Shortlist cheap source-view features before exact cross-view scoring.

    The shortlist is balanced over source-camera/logical-segment groups. A
    fixed share is first reserved for small/thin candidates and, when enabled,
    object-edge candidates. The remainder comes from non-priority candidates so
    likely cross-view occlusions and baseline controls retain broad coverage.
    Target labels, when present, are included in the balancing groups.
    """
    segment_ids = np.asarray(segment_ids, dtype=np.int32)
    source_cameras = np.asarray(source_cameras, dtype=str)
    source_edge_distance_px = np.asarray(source_edge_distance_px, dtype=np.float32)
    source_segment_area_fraction = np.asarray(source_segment_area_fraction, dtype=np.float32)
    source_local_segment_support = np.asarray(source_local_segment_support, dtype=np.float32)
    n_candidates = len(segment_ids)
    feature_arrays = (
        source_cameras,
        source_edge_distance_px,
        source_segment_area_fraction,
        source_local_segment_support,
    )
    if any(len(values) != n_candidates for values in feature_arrays):
        raise ValueError("Cross-view shortlist feature arrays must align")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if not 0.0 <= priority_fraction <= 1.0:
        raise ValueError("priority_fraction must be in [0, 1]")

    label_codes = None
    if target_labels is not None:
        target_labels = np.asarray(target_labels, dtype=str)
        if len(target_labels) != n_candidates:
            raise ValueError("target_labels must align with segment_ids")
        _, label_codes = np.unique(target_labels, return_inverse=True)

    if n_candidates <= max_candidates:
        return np.arange(n_candidates, dtype=np.int64)

    _, camera_codes = np.unique(source_cameras, return_inverse=True)
    group_columns = [camera_codes, segment_ids]
    if label_codes is not None:
        group_columns.append(label_codes)
    group_keys = np.column_stack(group_columns)
    _, group_ids = np.unique(group_keys, axis=0, return_inverse=True)
    group_ids = np.asarray(group_ids, dtype=np.int32)

    foreground = segment_ids != 0
    priority_mask = foreground & (
        (source_segment_area_fraction <= float(small_segment_area_fraction))
        | (source_local_segment_support <= float(local_support_threshold))
    )
    if prioritize_edges:
        priority_mask |= foreground & (source_edge_distance_px <= float(edge_distance_px))
    rng = np.random.RandomState(seed)
    priority_target = min(
        int(round(max_candidates * priority_fraction)),
        int(priority_mask.sum()),
    )
    priority_selected = _balanced_sample_without_replacement(
        np.flatnonzero(priority_mask),
        group_ids,
        priority_target,
        rng,
    )

    chosen = np.zeros(n_candidates, dtype=bool)
    chosen[priority_selected] = True
    remaining_target = max_candidates - len(priority_selected)
    broad_selected = _balanced_sample_without_replacement(
        np.flatnonzero(~priority_mask & ~chosen),
        group_ids,
        remaining_target,
        rng,
    )
    chosen[broad_selected] = True

    selected = np.concatenate([priority_selected, broad_selected])
    if len(selected) < max_candidates:
        fill_selected = _balanced_sample_without_replacement(
            np.flatnonzero(~chosen),
            group_ids,
            max_candidates - len(selected),
            rng,
        )
        selected = np.concatenate([selected, fill_selected])
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def depth_penalized_size_features(
    projected_area_fraction: np.ndarray,
    local_segment_support: np.ndarray,
    source_depth_m: np.ndarray,
    *,
    reference_depth_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct image-space size cues that otherwise favor distant surfaces.

    Perspective makes projected area fall approximately with depth squared and
    projected thickness fall approximately linearly with depth. Candidates at
    or closer than ``reference_depth_m`` are left unchanged. Beyond it, area is
    multiplied by the squared relative depth and local support by relative
    depth. This is intentionally a soft penalty rather than a maximum-depth
    cutoff, so genuinely small or thin distant objects can still qualify.
    """
    if reference_depth_m <= 0:
        raise ValueError("reference_depth_m must be positive")
    projected_area_fraction = np.asarray(projected_area_fraction, dtype=np.float32)
    local_segment_support = np.asarray(local_segment_support, dtype=np.float32)
    source_depth_m = np.asarray(source_depth_m, dtype=np.float32)
    if not (len(projected_area_fraction) == len(local_segment_support) == len(source_depth_m)):
        raise ValueError("Depth-penalized size feature arrays must align")

    depth_scale = np.full(len(source_depth_m), np.inf, dtype=np.float32)
    valid_depth = np.isfinite(source_depth_m) & (source_depth_m > 0)
    depth_scale[valid_depth] = np.maximum(
        1.0, source_depth_m[valid_depth] / float(reference_depth_m)
    )
    adjusted_area = projected_area_fraction * np.square(depth_scale)
    adjusted_support = np.minimum(1.0, local_segment_support * depth_scale)
    return adjusted_area.astype(np.float32), adjusted_support.astype(np.float32)


def soft_depth_sampling_weights(
    source_depth_m: np.ndarray,
    *,
    reference_depth_m: float = 1.0,
    minimum_weight: float = 0.10,
) -> np.ndarray:
    """Return inverse-square depth weights with a nonzero distant-point floor."""
    if reference_depth_m <= 0:
        raise ValueError("reference_depth_m must be positive")
    if not 0.0 <= minimum_weight <= 1.0:
        raise ValueError("minimum_weight must be in [0, 1]")
    source_depth_m = np.asarray(source_depth_m, dtype=np.float32)
    depth_scale = np.ones(len(source_depth_m), dtype=np.float32)
    valid_depth = np.isfinite(source_depth_m) & (source_depth_m > 0)
    depth_scale[valid_depth] = np.maximum(
        1.0, source_depth_m[valid_depth] / float(reference_depth_m)
    )
    weights = minimum_weight + (1.0 - minimum_weight) / np.square(depth_scale)
    weights[~valid_depth] = minimum_weight
    return weights.astype(np.float32)


def select_failure_targeted_candidate_indices(
    segment_ids: np.ndarray,
    source_edge_distance_px: np.ndarray,
    source_segment_area_fraction: np.ndarray,
    source_local_segment_support: np.ndarray,
    visibility_by_camera: np.ndarray,
    in_frame_by_camera: np.ndarray,
    occluder_edge_distance_px: np.ndarray,
    *,
    max_points: int,
    seed: int,
    target_fractions: tuple[float, ...] = DEFAULT_FAILURE_TARGET_FRACTIONS,
    edge_distance_px: float = 4.0,
    minimum_source_edge_distance_px: float = 2.0,
    small_segment_area_fraction: float = 0.02,
    local_support_threshold: float = 0.60,
    source_depth_m: np.ndarray | None = None,
    depth_penalty_reference_m: float = 1.0,
    depth_sampling_min_weight: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Select a balanced mixture of occlusion, edge, small/thin, and controls.

    Candidates closer than ``minimum_source_edge_distance_px`` to their logical
    segment boundary are ineligible for every bucket. The object-edge bucket
    therefore samples a safe annulus between that minimum inset and
    ``edge_distance_px``. When source depths are provided, small/thin eligibility
    uses depth-penalized area and support, and all four targeted buckets use soft
    inverse-depth sampling weights. The weights have a nonzero floor and the
    baseline quota is unweighted, so distant background coverage is retained.
    """
    segment_ids = np.asarray(segment_ids, dtype=np.int32)
    n_candidates = len(segment_ids)
    feature_arrays = (
        source_edge_distance_px,
        source_segment_area_fraction,
        source_local_segment_support,
        occluder_edge_distance_px,
    )
    if any(len(np.asarray(value)) != n_candidates for value in feature_arrays):
        raise ValueError("Failure-targeted feature arrays must align with segment_ids")
    visibility_by_camera = np.asarray(visibility_by_camera, dtype=bool)
    in_frame_by_camera = np.asarray(in_frame_by_camera, dtype=bool)
    if visibility_by_camera.shape != in_frame_by_camera.shape:
        raise ValueError("visibility_by_camera and in_frame_by_camera must align")
    if visibility_by_camera.ndim != 2 or visibility_by_camera.shape[1] != n_candidates:
        raise ValueError("camera visibility arrays must have shape (C, N)")
    if len(target_fractions) != len(FAILURE_TARGET_BUCKET_NAMES):
        raise ValueError(f"target_fractions must have {len(FAILURE_TARGET_BUCKET_NAMES)} values")
    if minimum_source_edge_distance_px < 0:
        raise ValueError("minimum_source_edge_distance_px must be non-negative")
    if minimum_source_edge_distance_px > edge_distance_px:
        raise ValueError("minimum_source_edge_distance_px must not exceed edge_distance_px")
    fractions = np.asarray(target_fractions, dtype=np.float64)
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("target_fractions must be non-negative and sum to 1")
    if max_points <= 0 or n_candidates == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype="<U24"),
            {name: 0 for name in FAILURE_TARGET_BUCKET_NAMES},
        )

    source_edge_distance_px = np.asarray(source_edge_distance_px, dtype=np.float32)
    safe_source_inset = source_edge_distance_px >= float(minimum_source_edge_distance_px)
    if not np.any(safe_source_inset):
        raise RuntimeError(
            "No failure-targeted candidates remain after applying the minimum "
            f"source edge inset of {minimum_source_edge_distance_px:g} px"
        )

    foreground = segment_ids != 0
    in_frame_occluded = np.any(in_frame_by_camera & ~visibility_by_camera, axis=0)
    adjusted_area = np.asarray(source_segment_area_fraction, dtype=np.float32)
    adjusted_support = np.asarray(source_local_segment_support, dtype=np.float32)
    depth_sample_weights = None
    if source_depth_m is not None:
        if len(np.asarray(source_depth_m)) != n_candidates:
            raise ValueError("source_depth_m must align with segment_ids")
        adjusted_area, adjusted_support = depth_penalized_size_features(
            adjusted_area,
            adjusted_support,
            source_depth_m,
            reference_depth_m=depth_penalty_reference_m,
        )
        depth_sample_weights = soft_depth_sampling_weights(
            source_depth_m,
            reference_depth_m=depth_penalty_reference_m,
            minimum_weight=depth_sampling_min_weight,
        )
    masks = {
        "occlusion_edge": safe_source_inset
        & in_frame_occluded
        & (np.asarray(occluder_edge_distance_px) <= edge_distance_px),
        "cross_view_occlusion": safe_source_inset & in_frame_occluded,
        "object_edge": safe_source_inset
        & foreground
        & (source_edge_distance_px <= edge_distance_px),
        "small_thin": safe_source_inset
        & foreground
        & (
            (adjusted_area <= small_segment_area_fraction)
            | (adjusted_support <= local_support_threshold)
        ),
        "baseline": safe_source_inset,
    }
    candidate_counts = {name: int(mask.sum()) for name, mask in masks.items()}

    raw_targets = fractions * int(max_points)
    targets = np.floor(raw_targets).astype(np.int64)
    for index in np.argsort(-(raw_targets - targets))[: int(max_points) - int(targets.sum())]:
        targets[index] += 1

    rng = np.random.RandomState(seed)
    chosen = np.zeros(n_candidates, dtype=bool)
    selected_parts: list[np.ndarray] = []
    bucket_parts: list[np.ndarray] = []
    for bucket_name, target in zip(FAILURE_TARGET_BUCKET_NAMES[:-1], targets[:-1]):
        pool = np.flatnonzero(masks[bucket_name] & ~chosen)
        picked = _balanced_sample_without_replacement(
            pool,
            segment_ids,
            int(target),
            rng,
            sample_weights=depth_sample_weights,
        )
        chosen[picked] = True
        selected_parts.append(picked)
        bucket_parts.append(np.full(len(picked), bucket_name, dtype="<U24"))

    selected_count = sum(len(part) for part in selected_parts)
    baseline_pool = np.flatnonzero(masks["baseline"] & ~chosen)
    baseline_picked = _balanced_sample_without_replacement(
        baseline_pool,
        segment_ids,
        min(int(max_points) - selected_count, len(baseline_pool)),
        rng,
    )
    chosen[baseline_picked] = True
    selected_parts.append(baseline_picked)
    bucket_parts.append(np.full(len(baseline_picked), "baseline", dtype="<U24"))

    selected = np.concatenate(selected_parts)
    buckets = np.concatenate(bucket_parts)
    if len(selected) < max_points:
        repeat_pool = selected if len(selected) else np.arange(n_candidates)
        repeated = rng.choice(repeat_pool, size=max_points - len(selected), replace=True)
        selected = np.concatenate([selected, repeated])
        buckets = np.concatenate([buckets, np.full(len(repeated), "fallback_repeat", dtype="<U24")])
    order = rng.permutation(len(selected))
    return selected[order], buckets[order], candidate_counts


def sample_kubric_candidates_from_image(
    model: MjModel,
    data: MjData,
    camera,
    img_width: int,
    img_height: int,
    depth_frame: np.ndarray,
    seg_frame: np.ndarray,
    frame_index: int,
    sampling_stride: int = 4,
    spatial_phase: tuple[int, int] = (0, 0),
    object_body_ids: set[int] | None = None,
    include_background: bool = False,
    dense_boundary_radius_px: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract all eligible candidates on one slice of Kubric's 3D grid.

    The temporal phase is handled by the caller. This function applies the
    shared spatial phase, keeps all valid pixels on that grid, unprojects their
    pixel centers, and returns query points in Kubric ``[t, y, x]`` order.

    With background enabled, every rendered MuJoCo geom is eligible, including
    world-body geoms (body 0). Empty renderer pixels are excluded via geom ID.
    Otherwise only ``object_body_ids`` are eligible.
    """
    if seg_frame.ndim != 3 or seg_frame.shape[2] < 3:
        raise ValueError("seg_frame must have shape (H, W, >=3) with geom and body IDs")

    stride = max(1, int(sampling_stride))
    geom_id_map = seg_frame[:, :, 0]
    object_type_map = seg_frame[:, :, 1]
    body_id_map = seg_frame[:, :, 2]
    valid_mask = (geom_id_map >= 0) & (object_type_map == mujoco.mjtObj.mjOBJ_GEOM.value)
    if not include_background:
        if object_body_ids is None:
            valid_mask &= body_id_map > 0
        else:
            valid_mask &= np.isin(body_id_map, list(object_body_ids))

    grid_mask = _fixed_phase_strided_mask(
        body_id_map.shape[0],
        body_id_map.shape[1],
        stride,
        spatial_phase[0],
        spatial_phase[1],
    )
    if dense_boundary_radius_px > 0:
        foreground_body_ids = object_body_ids
        if foreground_body_ids is None:
            foreground_body_ids = {
                int(body_id) for body_id in np.unique(body_id_map) if body_id > 0
            }
        segment_map = _logical_segment_map(model, body_id_map, foreground_body_ids)
        boundary_band = _segment_boundary_band(segment_map, valid_mask, dense_boundary_radius_px)
        if grid_mask is not None:
            valid_mask &= grid_mask | boundary_band
    else:
        valid_mask = _apply_strided_mask(valid_mask, grid_mask)
    pys, pxs = np.where(valid_mask)
    if len(pys) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    pixel_bids = body_id_map[pys, pxs]
    pixel_depths = depth_frame[pys, pxs]
    local_coords, body_ids, world_coords, _ = _unproject_and_localize(
        model,
        data,
        camera,
        img_width,
        img_height,
        pxs,
        pys,
        pixel_bids,
        pixel_depths,
        seed=0,
        use_pixel_centers=True,
    )
    query_points = np.column_stack(
        [
            np.full(len(pys), frame_index, dtype=np.float32),
            pys.astype(np.float32) + 0.5,
            pxs.astype(np.float32) + 0.5,
        ]
    )
    return local_coords, body_ids, world_coords, query_points


def select_kubric_candidate_indices(
    body_ids: np.ndarray,
    max_points: int,
    seed: int,
    max_sampled_fraction: float = 0.1,
    segment_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Select and pad candidate indices using Kubric's segment sampler.

    Returning indices keeps arbitrary arrays aligned with the sampled physical
    points. In particular, aligned multiview sampling uses them to preserve the
    camera that contributed each candidate while selecting one global pool.
    """
    n_candidates = len(body_ids)
    if n_candidates == 0 or max_points <= 0:
        return np.zeros(0, dtype=np.int64)

    if segment_ids is None:
        candidate_segment_ids = np.asarray(body_ids, dtype=np.int32)
    else:
        candidate_segment_ids = np.asarray(segment_ids, dtype=np.int32)
        if len(candidate_segment_ids) != n_candidates:
            raise ValueError("segment_ids must align with the candidate arrays")
    unique_segment_ids, counts = np.unique(candidate_segment_ids, return_counts=True)
    per_segment = get_kubric_num_to_sample(counts, max_sampled_fraction, max_points)
    rng = np.random.RandomState(seed)
    selected_parts: list[np.ndarray] = []
    for segment_id, n_sample in zip(unique_segment_ids, per_segment):
        if n_sample <= 0:
            continue
        candidate_indices = np.flatnonzero(candidate_segment_ids == segment_id)
        # Kubric uses tf.multinomial, which samples with replacement.
        selected_parts.append(rng.choice(candidate_indices, size=int(n_sample), replace=True))

    if not selected_parts:
        log.warning("Kubric allocation selected no points; using one candidate before padding")
        selected = np.array([rng.randint(0, n_candidates)], dtype=np.int64)
    else:
        selected = np.concatenate(selected_parts)

    # Kubric pads short samples by repeating a sampled row.
    if len(selected) < max_points:
        selected = np.concatenate(
            [selected, np.full(max_points - len(selected), selected[-1], dtype=np.int64)]
        )
    elif len(selected) > max_points:
        selected = selected[:max_points]

    return np.asarray(selected, dtype=np.int64)


def select_unambiguous_kubric_candidate_indices(
    body_ids: np.ndarray,
    max_points: int,
    seed: int,
    candidate_is_valid: Callable[[np.ndarray], np.ndarray],
    max_sampled_fraction: float = 0.1,
    segment_ids: np.ndarray | None = None,
    max_rejection_rounds: int = 32,
) -> tuple[np.ndarray, dict[str, int]]:
    """Keep Kubric's allocation; replace ambiguous tracks within each segment.

    The callback checks every recorded frame in the caller's relevant cameras:
    the owning camera for independent tracks, all cameras for shared tracks.
    Validation is lazy and cached per candidate, avoiding replay of the full
    (potentially million-point) pool. Sampling with replacement and padding
    remain Kubric behavior; clean original selections retain their positions.
    Exhausting a segment or the retry limit raises, never exports ambiguity.
    """
    if max_points <= 0 or len(body_ids) == 0:
        raise ValueError("Clean Kubric sampling requires candidates and a positive point budget")
    if max_rejection_rounds < 0:
        raise ValueError("max_rejection_rounds must be nonnegative")
    segments = np.asarray(body_ids if segment_ids is None else segment_ids, dtype=np.int32)
    selected = select_kubric_candidate_indices(
        body_ids, max_points, seed, max_sampled_fraction, segment_ids
    )
    # 0 = unchecked, 1 = valid, -1 = ambiguous in at least one observation.
    state = np.zeros(len(body_ids), dtype=np.int8)
    rng = np.random.RandomState(seed)
    replacements = 0
    initial_rejected = 0
    for round_index in range(max_rejection_rounds + 1):
        unchecked = np.unique(selected[state[selected] == 0])
        if len(unchecked):
            valid = np.asarray(candidate_is_valid(unchecked))
            if valid.shape != unchecked.shape or valid.dtype != np.bool_:
                raise ValueError("candidate_is_valid must return one boolean per candidate")
            state[unchecked] = np.where(valid, 1, -1)
        rejected_slots = np.flatnonzero(state[selected] < 0)
        if round_index == 0:
            initial_rejected = len(rejected_slots)
        if not len(rejected_slots):
            return selected, {
                "initial_rejected_tracks": initial_rejected,
                "validated_candidates": int(np.count_nonzero(state)),
                "rejected_candidates": int(np.count_nonzero(state < 0)),
                "replacement_draws": replacements,
                "replacement_rounds": round_index,
            }
        if round_index == max_rejection_rounds:
            raise RuntimeError(
                f"Clean Kubric sampling exhausted {max_rejection_rounds} replacement rounds; "
                f"{len(rejected_slots)} track slots still ambiguous"
            )
        rejected_segments = segments[selected[rejected_slots]]
        for segment in np.unique(rejected_segments):
            slots = rejected_slots[rejected_segments == segment]
            eligible = np.flatnonzero((segments == segment) & (state >= 0))
            if not len(eligible):
                raise RuntimeError(f"No unambiguous Kubric candidates remain in segment {segment}")
            selected[slots] = rng.choice(eligible, size=len(slots), replace=True)
            replacements += len(slots)
    raise AssertionError("unreachable")


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
    background_body_ids: set[int] | None = None,
    background_fraction: float = 0.0,
    image_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sample tracked points by picking visible pixels on objects.

    Like Kubric/TAP-Vid: pick random visible pixels from the rendered image,
    use segmentation to find which body they belong to, unproject to 3D, and
    convert to body-local coordinates for tracking. Every sampled point is
    guaranteed visible in this frame.

    When ``image_stride > 1``, candidates are restricted to a strided 2D grid
    with a random sub-pixel phase (same spirit as Kubric's ``sampling_stride``),
    then the usual uniform subsampling runs on that subset.

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
            non-world bodies. Ignored for pixels belonging to
            ``background_body_ids`` when a background budget is requested.
        prefer_body_ids: if provided, sample from these bodies first.
            Only falls back to other valid bodies if preferred bodies
            don't fill the budget.
        background_body_ids: if provided together with a positive
            ``background_fraction``, reserve ``background_fraction * max_points``
            points for pixels whose body is in this set. When ``None`` (the
            default) no explicit background budget is reserved — the sampler
            behaves as before. When set to an empty set, the complement of
            ``object_body_ids`` (i.e. all non-world bodies that are *not*
            trackable) is used.
        background_fraction: fraction of the budget to reserve for background
            bodies. Clamped to ``[0, 1]``. Ignored unless
            ``background_body_ids`` is not ``None``. If the requested
            background quota can't be filled (e.g. no background visible),
            the leftover is returned to the foreground budget.
        image_stride: If > 1, only pixels on a ``stride`` grid (random origin)
            are eligible. ``1`` keeps the original dense-pixel behavior.

    Returns:
        local_coords: (N, 3) body-local coords for tracking
        body_ids: (N,) int32 body id per point
        world_coords: (N, 3) initial world positions
        total_verts: total mesh vertex count across non-world bodies
    """
    rng = np.random.RandomState(seed)

    body_id_map = seg_frame[:, :, 2]  # (H, W) body id per pixel
    h, w = body_id_map.shape
    stride = max(1, int(image_stride))
    grid_mask = _random_phase_strided_mask(h, w, stride, rng)

    want_background_split = background_body_ids is not None and background_fraction > 0.0
    if want_background_split:
        background_fraction = float(np.clip(background_fraction, 0.0, 1.0))

        # Empty set => "everything that's a body but not in object_body_ids".
        # Non-empty => use explicitly-listed background bodies (rare, but useful
        # if a caller wants to restrict background to e.g. walls-only).
        if len(background_body_ids) == 0:
            all_body_mask = body_id_map > 0
            if object_body_ids is not None:
                bg_mask = all_body_mask & ~np.isin(body_id_map, list(object_body_ids))
            else:
                # object_body_ids=None means "everything goes to foreground",
                # so there's no implicit complement to draw background from.
                # Caller should pass an explicit set in this case.
                bg_mask = np.zeros_like(all_body_mask)
        else:
            bg_mask = np.isin(body_id_map, list(background_body_ids))

        if object_body_ids is not None:
            fg_mask = np.isin(body_id_map, list(object_body_ids)) & ~bg_mask
        else:
            fg_mask = (body_id_map > 0) & ~bg_mask

        bg_mask = _apply_strided_mask(bg_mask, grid_mask)
        fg_mask = _apply_strided_mask(fg_mask, grid_mask)

        # Budget split: background gets its requested share, foreground gets
        # the rest (including any leftover when background can't be filled).
        n_bg_target = int(round(max_points * background_fraction))
        bg_ys, bg_xs = np.where(bg_mask)
        n_bg = min(n_bg_target, len(bg_ys))
        n_fg_target = max_points - n_bg

        if prefer_body_ids is not None and len(prefer_body_ids) > 0:
            prefer_pix_mask = fg_mask & np.isin(body_id_map, list(prefer_body_ids))
            other_fg_mask = fg_mask & ~prefer_pix_mask
            pref_ys, pref_xs = np.where(prefer_pix_mask)
            oth_ys, oth_xs = np.where(other_fg_mask)
            n_pref = min(n_fg_target, len(pref_ys))
            n_oth = min(n_fg_target - n_pref, len(oth_ys))
            fg_ys_list, fg_xs_list = [], []
            if n_pref > 0:
                cp = rng.choice(len(pref_ys), size=n_pref, replace=False)
                fg_ys_list.append(pref_ys[cp])
                fg_xs_list.append(pref_xs[cp])
            if n_oth > 0:
                co = rng.choice(len(oth_ys), size=n_oth, replace=False)
                fg_ys_list.append(oth_ys[co])
                fg_xs_list.append(oth_xs[co])
            fg_ys_picked = np.concatenate(fg_ys_list) if fg_ys_list else np.empty(0, dtype=int)
            fg_xs_picked = np.concatenate(fg_xs_list) if fg_xs_list else np.empty(0, dtype=int)
        else:
            fg_ys_all, fg_xs_all = np.where(fg_mask)
            n_fg = min(n_fg_target, len(fg_ys_all))
            if n_fg > 0:
                cf = rng.choice(len(fg_ys_all), size=n_fg, replace=False)
                fg_ys_picked = fg_ys_all[cf]
                fg_xs_picked = fg_xs_all[cf]
            else:
                fg_ys_picked = np.empty(0, dtype=int)
                fg_xs_picked = np.empty(0, dtype=int)

        if n_bg > 0:
            cb = rng.choice(len(bg_ys), size=n_bg, replace=False)
            bg_ys_picked = bg_ys[cb]
            bg_xs_picked = bg_xs[cb]
        else:
            bg_ys_picked = np.empty(0, dtype=int)
            bg_xs_picked = np.empty(0, dtype=int)

        valid_ys = np.concatenate([fg_ys_picked, bg_ys_picked])
        valid_xs = np.concatenate([fg_xs_picked, bg_xs_picked])

        if len(valid_ys) == 0:
            log.warning("No non-world body pixels visible — falling back to vertex sampling")
            return sample_mesh_vertices(model, data, max_points, seed)

        # Skip the downstream "pick n_pick from valid_ys" step: valid_{ys,xs}
        # is already the final sampled set.
        pxs = valid_xs
        pys = valid_ys
        pixel_bids = body_id_map[pys, pxs]
        pixel_depths = depth_frame[pys, pxs]

        return _unproject_and_localize(
            model,
            data,
            camera,
            img_width,
            img_height,
            pxs,
            pys,
            pixel_bids,
            pixel_depths,
            seed,
        )

    # --- Single-budget path (original behavior, kept for back-compat) ---
    if object_body_ids is not None:
        valid_mask = np.isin(body_id_map, list(object_body_ids))
    else:
        valid_mask = body_id_map > 0

    valid_mask = _apply_strided_mask(valid_mask, grid_mask)

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

    return _unproject_and_localize(
        model,
        data,
        camera,
        img_width,
        img_height,
        pxs,
        pys,
        pixel_bids,
        pixel_depths,
        seed,
    )


def _unproject_and_localize(
    model: MjModel,
    data: MjData,
    camera,
    img_width: int,
    img_height: int,
    pxs: np.ndarray,
    pys: np.ndarray,
    pixel_bids: np.ndarray,
    pixel_depths: np.ndarray,
    seed: int,
    use_pixel_centers: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Unproject picked pixels to world, convert to body-local, count meshes.

    Shared tail for the single-budget and background-split paths of
    :func:`sample_from_image`.
    """
    cam2world = camera.get_pose()
    fovy_rad = np.radians(camera.fov)
    fy = (img_height / 2.0) / np.tan(fovy_rad / 2.0)
    fx = fy
    cx, cy = img_width / 2.0, img_height / 2.0

    center_offset = 0.5 if use_pixel_centers else 0.0
    raster_x = pxs.astype(np.float64) + center_offset
    raster_y = pys.astype(np.float64) + center_offset
    cam_x = (raster_x - cx) / fx * pixel_depths
    cam_y = (raster_y - cy) / fy * pixel_depths
    cam_z = pixel_depths.astype(np.float64)
    pts_cam = np.stack([cam_x, cam_y, cam_z], axis=1)

    R = cam2world[:3, :3]
    t = cam2world[:3, 3]
    world_pts = (pts_cam @ R.T + t).astype(np.float32)

    total_verts = 0
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH.value:
            continue
        if model.geom_bodyid[geom_id] == 0:
            continue
        total_verts += model.mesh_vertnum[model.geom_dataid[geom_id]]

    n_pick = len(pxs)
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

    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    return world2cam, intrinsics


def _occluder_edge_distances(
    segment_map: np.ndarray,
    points_2d: np.ndarray,
    occluded_mask: np.ndarray,
    max_edge_distance_px: int,
) -> np.ndarray:
    """Measure distance to the rendered occluder's silhouette for occluded points."""
    points_2d = np.asarray(points_2d)
    occluded_mask = np.asarray(occluded_mask, dtype=bool)
    result = np.full(len(points_2d), float(max_edge_distance_px + 1), dtype=np.float32)
    selected = np.flatnonzero(occluded_mask)
    if len(selected) == 0:
        return result
    xs = np.floor(points_2d[selected, 0]).astype(np.int64)
    ys = np.floor(points_2d[selected, 1]).astype(np.int64)
    height, width = segment_map.shape
    valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    selected = selected[valid]
    xs = xs[valid]
    ys = ys[valid]
    if len(selected) == 0:
        return result
    rendered_segments = segment_map[ys, xs]
    query_points = np.column_stack(
        [
            np.zeros(len(selected), dtype=np.float32),
            points_2d[selected, 1],
            points_2d[selected, 0],
        ]
    )
    _, _, edge_distance = candidate_mask_context_features(
        segment_map,
        query_points,
        rendered_segments,
        local_radius_px=0,
        max_edge_distance_px=max_edge_distance_px,
    )
    result[selected] = edge_distance
    return result


def _unproject_depth_pixels(
    depth: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Unproject raster pixel centers into the camera's CV coordinate frame."""
    z = np.asarray(depth[ys, xs], dtype=np.float64)
    x = (xs.astype(np.float64) + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0] * z
    y = (ys.astype(np.float64) + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1] * z
    return np.column_stack((x, y, z))


def _depth_surface_normals_for_camera(
    depth: np.ndarray,
    segmentation: np.ndarray,
    query_points: np.ndarray,
    geom_ids: np.ndarray,
    camera,
    img_width: int,
    img_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate normals from exact-geom horizontal and vertical depth tangents."""
    count = len(query_points)
    normals_world = np.full((count, 3), np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    if count == 0:
        return normals_world.astype(np.float32), valid

    xs = np.floor(query_points[:, 2]).astype(np.int64)
    ys = np.floor(query_points[:, 1]).astype(np.int64)
    height, width = depth.shape
    inside = (xs > 0) & (xs + 1 < width) & (ys > 0) & (ys + 1 < height)
    selected = np.flatnonzero(inside)
    if len(selected) == 0:
        return normals_world.astype(np.float32), valid

    sx, sy = xs[selected], ys[selected]
    wanted_geom = geom_ids[selected]
    geom_map = np.asarray(segmentation[:, :, 0], dtype=np.int32)
    center_ok = geom_map[sy, sx] == wanted_geom
    left_ok = geom_map[sy, sx - 1] == wanted_geom
    right_ok = geom_map[sy, sx + 1] == wanted_geom
    up_ok = geom_map[sy - 1, sx] == wanted_geom
    down_ok = geom_map[sy + 1, sx] == wanted_geom
    finite_center = np.isfinite(depth[sy, sx]) & (depth[sy, sx] > 0)
    finite_left = np.isfinite(depth[sy, sx - 1]) & (depth[sy, sx - 1] > 0)
    finite_right = np.isfinite(depth[sy, sx + 1]) & (depth[sy, sx + 1] > 0)
    finite_up = np.isfinite(depth[sy - 1, sx]) & (depth[sy - 1, sx] > 0)
    finite_down = np.isfinite(depth[sy + 1, sx]) & (depth[sy + 1, sx] > 0)
    left_ok &= finite_left
    right_ok &= finite_right
    up_ok &= finite_up
    down_ok &= finite_down
    usable = center_ok & finite_center & (left_ok | right_ok) & (up_ok | down_ok)
    selected = selected[usable]
    if len(selected) == 0:
        return normals_world.astype(np.float32), valid

    sx, sy = xs[selected], ys[selected]
    local = np.flatnonzero(usable)
    _, intrinsics = _build_camera_matrices(camera, img_width, img_height)
    center = _unproject_depth_pixels(depth, sx, sy, intrinsics)
    tangent_x = np.empty_like(center)
    tangent_y = np.empty_like(center)
    both_x = left_ok[local] & right_ok[local]
    both_y = up_ok[local] & down_ok[local]
    if np.any(both_x):
        tangent_x[both_x] = _unproject_depth_pixels(
            depth, sx[both_x] + 1, sy[both_x], intrinsics
        ) - _unproject_depth_pixels(depth, sx[both_x] - 1, sy[both_x], intrinsics)
    if np.any(~both_x):
        use_right = right_ok[local][~both_x]
        other_x = sx[~both_x] + np.where(use_right, 1, -1)
        other = _unproject_depth_pixels(depth, other_x, sy[~both_x], intrinsics)
        tangent_x[~both_x] = np.where(
            use_right[:, None], other - center[~both_x], center[~both_x] - other
        )
    if np.any(both_y):
        tangent_y[both_y] = _unproject_depth_pixels(
            depth, sx[both_y], sy[both_y] + 1, intrinsics
        ) - _unproject_depth_pixels(depth, sx[both_y], sy[both_y] - 1, intrinsics)
    if np.any(~both_y):
        use_down = down_ok[local][~both_y]
        other_y = sy[~both_y] + np.where(use_down, 1, -1)
        other = _unproject_depth_pixels(depth, sx[~both_y], other_y, intrinsics)
        tangent_y[~both_y] = np.where(
            use_down[:, None], other - center[~both_y], center[~both_y] - other
        )

    normals_camera = np.cross(tangent_x, tangent_y)
    lengths = np.linalg.norm(normals_camera, axis=1)
    good = np.isfinite(normals_camera).all(axis=1) & (lengths > 1e-10)
    normals_camera[good] /= lengths[good, None]
    cam2world = np.asarray(camera.get_pose(), dtype=np.float64)
    converted = normals_camera @ cam2world[:3, :3].T
    normals_world[selected[good]] = converted[good]
    valid[selected[good]] = True
    return normals_world.astype(np.float32), valid


def _geometry_surface_normals(
    model: MjModel,
    data: MjData,
    world_points: np.ndarray,
    geom_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytic primitive or nearest-triangle normals for MuJoCo geoms."""
    normals = np.full((len(world_points), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(world_points), dtype=bool)
    methods = np.full(len(world_points), "unavailable", dtype="<U32")
    mesh_cache = {}
    for geom_id in np.unique(geom_ids):
        rows = np.flatnonzero(geom_ids == geom_id)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        local_points = (world_points[rows] - position) @ rotation
        geom_type = int(model.geom_type[geom_id])
        size = np.maximum(np.asarray(model.geom_size[geom_id], dtype=np.float64), 1e-12)
        local_normals = np.full_like(local_points, np.nan)
        method = "primitive_analytic"

        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE.value:
            local_normals[:] = (0.0, 0.0, 1.0)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE.value:
            local_normals = local_points
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID.value:
            local_normals = local_points / (size**2)
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX.value:
            face_score = np.abs(local_points) / size
            axes = np.argmax(face_score, axis=1)
            local_normals[:] = 0.0
            local_normals[np.arange(len(rows)), axes] = np.where(
                local_points[np.arange(len(rows)), axes] >= 0, 1.0, -1.0
            )
        elif geom_type in {
            mujoco.mjtGeom.mjGEOM_CYLINDER.value,
            mujoco.mjtGeom.mjGEOM_CAPSULE.value,
        }:
            radial = np.linalg.norm(local_points[:, :2], axis=1)
            if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER.value:
                on_cap = np.abs(np.abs(local_points[:, 2]) - size[1]) < np.abs(radial - size[0])
                local_normals[:, :2] = local_points[:, :2]
                local_normals[:, 2] = 0.0
                local_normals[on_cap] = 0.0
                local_normals[on_cap, 2] = np.where(local_points[on_cap, 2] >= 0, 1.0, -1.0)
            else:
                cap_center_z = np.clip(local_points[:, 2], -size[1], size[1])
                local_normals = local_points - np.column_stack(
                    (np.zeros(len(rows)), np.zeros(len(rows)), cap_center_z)
                )
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH.value:
            try:
                import trimesh

                mesh_id = int(model.geom_dataid[geom_id])
                if mesh_id not in mesh_cache:
                    vertex_start = int(model.mesh_vertadr[mesh_id])
                    vertex_count = int(model.mesh_vertnum[mesh_id])
                    face_start = int(model.mesh_faceadr[mesh_id])
                    face_count = int(model.mesh_facenum[mesh_id])
                    mesh_cache[mesh_id] = trimesh.Trimesh(
                        vertices=np.asarray(
                            model.mesh_vert[vertex_start : vertex_start + vertex_count]
                        ),
                        faces=np.asarray(model.mesh_face[face_start : face_start + face_count]),
                        process=False,
                    )
                mesh = mesh_cache[mesh_id]
                _, _, face_ids = mesh.nearest.on_surface(local_points)
                local_normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float64)
                method = "mesh_nearest_face"
            except Exception as exc:
                log.warning("Could not compute mesh normals for geom %d: %s", geom_id, exc)
        else:
            log.warning("No geometry normal implementation for MuJoCo geom type %d", geom_type)

        lengths = np.linalg.norm(local_normals, axis=1)
        good = np.isfinite(local_normals).all(axis=1) & (lengths > 1e-10)
        local_normals[good] /= lengths[good, None]
        normals[rows[good]] = local_normals[good] @ rotation.T
        valid[rows[good]] = True
        methods[rows[good]] = method
    return normals.astype(np.float32), valid, methods


def estimate_aligned_surface_normals(
    model: MjModel,
    data: MjData,
    camera_names: list[str] | tuple[str, ...],
    camera_registry,
    rendered_depths: dict[str, np.ndarray],
    rendered_segmentations: dict[str, np.ndarray],
    query_points: np.ndarray,
    source_cameras: np.ndarray,
    world_points: np.ndarray,
    body_ids: np.ndarray,
    geom_ids: np.ndarray,
    img_width: int,
    img_height: int,
) -> dict[str, np.ndarray]:
    """Estimate one oriented surface normal for every selected physical point."""
    count = len(world_points)
    world_normals = np.full((count, 3), np.nan, dtype=np.float32)
    methods = np.full(count, "unavailable", dtype="<U32")
    resolved = np.zeros(count, dtype=bool)
    for camera_name in camera_names:
        rows = np.flatnonzero(source_cameras == camera_name)
        depth_normals, valid = _depth_surface_normals_for_camera(
            rendered_depths[camera_name],
            rendered_segmentations[camera_name],
            query_points[rows],
            geom_ids[rows],
            camera_registry[camera_name],
            img_width,
            img_height,
        )
        world_normals[rows[valid]] = depth_normals[valid]
        resolved[rows[valid]] = True
        methods[rows[valid]] = "rendered_depth_exact_geom"

    if not np.all(resolved):
        missing = np.flatnonzero(~resolved)
        fallback, valid, fallback_methods = _geometry_surface_normals(
            model, data, world_points[missing], geom_ids[missing]
        )
        world_normals[missing[valid]] = fallback[valid]
        resolved[missing[valid]] = True
        methods[missing[valid]] = fallback_methods[valid]
    if not np.all(resolved):
        missing_geoms = np.unique(geom_ids[~resolved]).tolist()
        raise RuntimeError(
            f"Could not compute surface normals for {(~resolved).sum()} points "
            f"on geoms {missing_geoms}"
        )

    # Resolve the normal sign consistently: point toward its query camera.
    camera_normals = np.empty_like(world_normals)
    for camera_name in camera_names:
        rows = np.flatnonzero(source_cameras == camera_name)
        cam2world = np.asarray(camera_registry[camera_name].get_pose(), dtype=np.float64)
        toward_camera = cam2world[:3, 3] - world_points[rows]
        flip = np.sum(world_normals[rows] * toward_camera, axis=1) < 0
        world_normals[rows[flip]] *= -1.0
        world2cam = np.linalg.inv(cam2world)
        camera_normals[rows] = world_normals[rows] @ world2cam[:3, :3].T

    local_normals = np.empty_like(world_normals)
    for body_id in np.unique(body_ids):
        rows = np.flatnonzero(body_ids == body_id)
        body_rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        local_normals[rows] = world_normals[rows] @ body_rotation

    for values, name in (
        (world_normals, "world"),
        (local_normals, "body-local"),
        (camera_normals, "query-camera"),
    ):
        lengths = np.linalg.norm(values, axis=1)
        if not np.isfinite(values).all() or not np.allclose(lengths, 1.0, atol=2e-4):
            raise RuntimeError(f"Generated invalid {name} surface normals")
    method_names, method_counts = np.unique(methods, return_counts=True)
    return {
        "normals_3d": world_normals.astype(np.float32),
        "local_normals": local_normals.astype(np.float32),
        "query_source_camera_normals": camera_normals.astype(np.float32),
        "surface_normal_methods": methods,
        "surface_normal_method_names": method_names,
        "surface_normal_method_counts": method_counts.astype(np.int64),
    }


def sample_aligned_kubric_points_for_frame(
    env,
    camera_names: list[str] | tuple[str, ...],
    img_width: int,
    img_height: int,
    max_points: int = 1000,
    seed: int = 0,
    sampling_stride: int = 4,
    max_sampled_fraction: float = 0.1,
    include_background: bool = True,
    sampling_mode: str = "kubric",
    failure_target_fractions: tuple[float, ...] = DEFAULT_FAILURE_TARGET_FRACTIONS,
    failure_edge_distance_px: float = 4.0,
    failure_min_source_edge_distance_px: float = 2.0,
    failure_max_cross_view_candidates: int = 15_000,
    failure_shortlist_prioritize_edges: bool = True,
    failure_small_segment_area_fraction: float = 0.02,
    failure_local_support_threshold: float = 0.60,
    failure_dense_boundary_radius_px: int = 4,
    failure_depth_penalty_reference_m: float = 1.0,
    failure_depth_sampling_min_weight: float = 0.10,
    eligible_body_ids: set[int] | None = None,
    body_target_labels: dict[int, str] | None = None,
    visibility_depth_relative_tolerance: float = (DEFAULT_VISIBILITY_DEPTH_RELATIVE_TOLERANCE),
    visibility_depth_absolute_tolerance_m: float = (DEFAULT_VISIBILITY_DEPTH_ABSOLUTE_TOLERANCE_M),
    exclude_raster_ambiguous: bool = True,
) -> dict:
    """Sample one shared physical 3D point set for a multiview still.

    Candidate pixels are pooled from every camera, selected once by logical
    segment, and projected back into all cameras. The ordered track ids therefore
    identify the same physical surface points in every view. Visibility requires
    renderer support from a matching exact geom and depth in the four pixels
    surrounding the continuous projection. By default candidates with ambiguous
    raster support in any camera are removed before sampling. ``failure_targeted``
    mode stratifies selection toward small/thin supports, silhouettes, and
    cross-view occlusions while retaining a baseline Kubric control bucket.
    """
    if max_points < 1:
        raise ValueError("max_points must be at least 1")
    if not camera_names:
        raise ValueError("At least one camera is required")
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("Camera names must be unique")
    if not 0.0 <= max_sampled_fraction <= 1.0:
        raise ValueError("max_sampled_fraction must be in [0, 1]")
    if sampling_mode not in {"kubric", "failure_targeted"}:
        raise ValueError("sampling_mode must be either 'kubric' or 'failure_targeted'")
    if sampling_mode == "failure_targeted":
        if failure_max_cross_view_candidates < max_points:
            raise ValueError("failure_max_cross_view_candidates must be at least max_points")
        if failure_min_source_edge_distance_px < 0:
            raise ValueError("failure_min_source_edge_distance_px must be non-negative")
        if failure_min_source_edge_distance_px > failure_edge_distance_px:
            raise ValueError(
                "failure_min_source_edge_distance_px must not exceed failure_edge_distance_px"
            )
    if eligible_body_ids is not None and include_background:
        raise ValueError(
            "include_background must be False when eligible_body_ids restricts candidates"
        )

    model = env.current_model
    data = env.current_data
    foreground_body_ids = (
        get_trackable_body_ids(model)
        if eligible_body_ids is None
        else {int(body_id) for body_id in eligible_body_ids}
    )
    invalid_body_ids = {body_id for body_id in foreground_body_ids if not 0 < body_id < model.nbody}
    if invalid_body_ids:
        raise ValueError(f"Invalid eligible body ids: {sorted(invalid_body_ids)}")
    if not foreground_body_ids:
        raise ValueError("At least one eligible foreground body is required")
    stride = max(1, int(sampling_stride))
    phase_rng = np.random.RandomState(seed + 4242)
    spatial_phase = tuple(int(value) for value in phase_rng.randint(0, stride, size=2))

    candidate_batches = []
    rendered_depths = {}
    rendered_segment_maps = {}
    rendered_segmentations = {}

    for camera_name in camera_names:
        if camera_name not in env.camera_manager.registry:
            raise KeyError(f"Camera {camera_name!r} is not registered")
        camera = env.camera_manager.registry[camera_name]
        depth = np.asarray(env.render_depth_frame(camera_name)).copy()
        segmentation = np.asarray(env.render_segmentation_frame(camera_name)).copy()
        if depth.shape != (img_height, img_width) or segmentation.shape != (
            img_height,
            img_width,
            3,
        ):
            raise ValueError(
                f"Camera {camera_name!r} rasters do not match the requested image dimensions"
            )
        rendered_depths[camera_name] = depth
        rendered_segmentations[camera_name] = segmentation
        body_id_map = np.asarray(segmentation[:, :, 2], dtype=np.int32)
        segment_map = _logical_segment_map(model, body_id_map, foreground_body_ids)
        rendered_segment_maps[camera_name] = segment_map
        local, body_ids, world, query = sample_kubric_candidates_from_image(
            model,
            data,
            camera,
            img_width,
            img_height,
            depth,
            segmentation,
            frame_index=0,
            sampling_stride=stride,
            spatial_phase=spatial_phase,
            object_body_ids=foreground_body_ids,
            include_background=include_background,
            dense_boundary_radius_px=(
                failure_dense_boundary_radius_px if sampling_mode == "failure_targeted" else 0
            ),
        )
        if len(local) == 0:
            continue
        query_x = np.floor(query[:, 2]).astype(np.int64)
        query_y = np.floor(query[:, 1]).astype(np.int64)
        logical_segment_ids = get_kubric_segment_ids(model, body_ids, foreground_body_ids)
        area_fraction, local_support, edge_distance = candidate_mask_context_features(
            segment_map,
            query,
            logical_segment_ids,
            local_radius_px=4,
            max_edge_distance_px=max(1, int(np.ceil(failure_edge_distance_px))),
        )
        candidate_batches.append(
            {
                "local_coords": local,
                "body_ids": body_ids,
                "geom_ids": np.asarray(segmentation[query_y, query_x, 0], dtype=np.int32),
                "world_coords": world,
                "query_points": query,
                "segment_ids": logical_segment_ids,
                "source_cameras": np.full(len(local), camera_name),
                "segment_area_fraction": area_fraction,
                "local_segment_support": local_support,
                "source_edge_distance": edge_distance,
            }
        )

    if not candidate_batches:
        raise RuntimeError("No eligible Kubric point candidates were visible in any camera")

    # Every value has one row per candidate. Slice the entire dictionary whenever
    # a filter or shortlist changes the pool, preserving physical point ordering.
    candidates = {
        key: np.concatenate([batch[key] for batch in candidate_batches], axis=0)
        for key in candidate_batches[0]
    }
    expected_target_labels = None
    if body_target_labels is not None:
        missing_labels = set(int(body_id) for body_id in candidates["body_ids"]) - set(
            int(body_id) for body_id in body_target_labels
        )
        if missing_labels:
            raise ValueError(
                f"Missing target labels for candidate body ids: {sorted(missing_labels)}"
            )
        candidates["target_labels"] = np.asarray(
            [body_target_labels[int(body_id)] for body_id in candidates["body_ids"]],
            dtype=str,
        )
        expected_target_labels = set(body_target_labels.values())
        missing_visible_targets = expected_target_labels - set(candidates["target_labels"])
        if missing_visible_targets:
            raise RuntimeError(
                f"No visible point candidates for target labels: {sorted(missing_visible_targets)}"
            )

    raw_candidate_count = len(candidates["local_coords"])
    source_edge_unsafe_candidate_count = 0
    candidates_before_cross_view_shortlist = len(candidates["local_coords"])
    cross_view_candidate_count = len(candidates["local_coords"])
    cross_view_shortlisted_out_count = 0
    if sampling_mode == "failure_targeted":
        safe_source_inset = candidates["source_edge_distance"] >= float(
            failure_min_source_edge_distance_px
        )
        source_edge_unsafe_candidate_count = int((~safe_source_inset).sum())
        if not np.any(safe_source_inset):
            raise RuntimeError(
                "No point candidates remain after applying the minimum source "
                f"edge inset of {failure_min_source_edge_distance_px:g} px"
            )
        if not np.all(safe_source_inset):
            candidates = {key: value[safe_source_inset] for key, value in candidates.items()}

        if expected_target_labels is not None:
            missing_safe_targets = expected_target_labels - set(candidates["target_labels"])
            if missing_safe_targets:
                raise RuntimeError(
                    "Minimum source edge inset removed every candidate for target "
                    f"labels: {sorted(missing_safe_targets)}"
                )

        candidates_before_cross_view_shortlist = len(candidates["local_coords"])
        shortlist_indices = select_failure_targeted_cross_view_shortlist_indices(
            candidates["segment_ids"],
            candidates["source_cameras"],
            candidates["source_edge_distance"],
            candidates["segment_area_fraction"],
            candidates["local_segment_support"],
            max_candidates=failure_max_cross_view_candidates,
            seed=seed + 7331,
            edge_distance_px=failure_edge_distance_px,
            prioritize_edges=failure_shortlist_prioritize_edges,
            small_segment_area_fraction=(failure_small_segment_area_fraction),
            local_support_threshold=failure_local_support_threshold,
            target_labels=candidates.get("target_labels"),
        )
        if len(shortlist_indices) < len(candidates["local_coords"]):
            candidates = {key: value[shortlist_indices] for key, value in candidates.items()}

        cross_view_candidate_count = len(candidates["local_coords"])
        cross_view_shortlisted_out_count = (
            candidates_before_cross_view_shortlist - cross_view_candidate_count
        )

        if expected_target_labels is not None:
            missing_shortlisted_targets = expected_target_labels - set(candidates["target_labels"])
            if missing_shortlisted_targets:
                raise RuntimeError(
                    "Cross-view shortlist removed every candidate for target labels: "
                    f"{sorted(missing_shortlisted_targets)}"
                )

    def project_points(camera_name, local_coords, body_ids, geom_ids):
        points_2d, visibility, points_3d, diagnostics = track_points_for_frame(
            data,
            local_coords,
            body_ids,
            env.camera_manager.registry[camera_name],
            img_width,
            img_height,
            rendered_depths[camera_name],
            return_diagnostics=True,
            segmentation_frame=rendered_segmentations[camera_name],
            geom_ids=geom_ids,
            visibility_depth_relative_tolerance=visibility_depth_relative_tolerance,
            visibility_depth_absolute_tolerance_m=visibility_depth_absolute_tolerance_m,
        )
        return {
            "points_2d": points_2d,
            "visibility": visibility,
            "points_3d": points_3d,
            **diagnostics,
        }

    cached_candidate_cameras: dict[str, dict[str, np.ndarray]] = {}
    candidates["occluder_edge_distance"] = np.full(
        len(candidates["local_coords"]),
        float(np.ceil(failure_edge_distance_px) + 1),
        dtype=np.float32,
    )
    candidates["source_depth"] = np.full(len(candidates["local_coords"]), np.nan, dtype=np.float32)
    candidates["depth_penalized_area"] = candidates["segment_area_fraction"]
    candidates["depth_penalized_support"] = candidates["local_segment_support"]
    candidate_bucket_counts: dict[str, int]
    raster_ambiguous_candidate_count = 0
    # Both modes must check all views before sampling when ambiguity is excluded.
    # Failure targeting also needs these projections to form its occlusion buckets.
    if sampling_mode == "failure_targeted" or exclude_raster_ambiguous:
        for camera_name in camera_names:
            projected = project_points(
                camera_name,
                candidates["local_coords"],
                candidates["body_ids"],
                candidates["geom_ids"],
            )
            cached_candidate_cameras[camera_name] = projected
            source_mask = candidates["source_cameras"] == camera_name
            candidates["source_depth"][source_mask] = projected["point_depth"][source_mask]
            if sampling_mode == "failure_targeted":
                occluded = projected["in_frame"] & (projected["visibility"] <= 0.5)
                occluder_edge = _occluder_edge_distances(
                    rendered_segment_maps[camera_name],
                    projected["points_2d"],
                    occluded,
                    max(1, int(np.ceil(failure_edge_distance_px))),
                )
                candidates["occluder_edge_distance"] = np.minimum(
                    candidates["occluder_edge_distance"], occluder_edge
                )

    if exclude_raster_ambiguous:
        ambiguous = np.any(
            [values["raster_ambiguous"] for values in cached_candidate_cameras.values()], axis=0
        )
        raster_ambiguous_candidate_count = int(ambiguous.sum())
        keep = ~ambiguous
        if not np.any(keep):
            raise RuntimeError("All multiview point candidates had ambiguous raster visibility")
        candidates = {key: value[keep] for key, value in candidates.items()}
        cached_candidate_cameras = {
            name: {key: value[keep] for key, value in values.items()}
            for name, values in cached_candidate_cameras.items()
        }
        if expected_target_labels is not None:
            missing_after_filter = expected_target_labels - set(candidates["target_labels"])
            if missing_after_filter:
                raise RuntimeError(
                    "Raster visibility filtering removed every candidate for target "
                    f"labels: {sorted(missing_after_filter)}"
                )

    if sampling_mode == "failure_targeted":
        candidates["depth_penalized_area"], candidates["depth_penalized_support"] = (
            depth_penalized_size_features(
                candidates["segment_area_fraction"],
                candidates["local_segment_support"],
                candidates["source_depth"],
                reference_depth_m=failure_depth_penalty_reference_m,
            )
        )
        visibility_matrix = np.stack(
            [cached_candidate_cameras[name]["visibility"] > 0.5 for name in camera_names]
        )
        in_frame_matrix = np.stack(
            [cached_candidate_cameras[name]["in_frame"] for name in camera_names]
        )

    def select_candidates(indices, budget, selection_seed):
        if sampling_mode == "kubric":
            selected = select_kubric_candidate_indices(
                candidates["body_ids"][indices],
                max_points=budget,
                seed=selection_seed,
                max_sampled_fraction=max_sampled_fraction,
                segment_ids=candidates["segment_ids"][indices],
            )
            return (
                selected,
                np.full(len(selected), "kubric", dtype="<U24"),
                {"kubric": len(indices)},
            )
        return select_failure_targeted_candidate_indices(
            candidates["segment_ids"][indices],
            candidates["source_edge_distance"][indices],
            candidates["segment_area_fraction"][indices],
            candidates["local_segment_support"][indices],
            visibility_matrix[:, indices],
            in_frame_matrix[:, indices],
            candidates["occluder_edge_distance"][indices],
            max_points=budget,
            seed=selection_seed,
            target_fractions=failure_target_fractions,
            edge_distance_px=failure_edge_distance_px,
            minimum_source_edge_distance_px=failure_min_source_edge_distance_px,
            small_segment_area_fraction=failure_small_segment_area_fraction,
            local_support_threshold=failure_local_support_threshold,
            source_depth_m=candidates["source_depth"][indices],
            depth_penalty_reference_m=failure_depth_penalty_reference_m,
            depth_sampling_min_weight=failure_depth_sampling_min_weight,
        )

    # One broad pool by default; explicit targets receive equal point budgets.
    if "target_labels" in candidates:
        target_groups = [
            np.flatnonzero(candidates["target_labels"] == name)
            for name in sorted(set(candidates["target_labels"]))
        ]
    else:
        target_groups = [np.arange(len(candidates["local_coords"]))]
    target_budgets = np.full(len(target_groups), max_points // len(target_groups), dtype=np.int64)
    target_budgets[: max_points % len(target_groups)] += 1
    selected_parts, bucket_parts = [], []
    candidate_bucket_counts = {}
    for target_index, (indices, budget) in enumerate(zip(target_groups, target_budgets)):
        target_selected, target_buckets, target_counts = select_candidates(
            indices, int(budget), seed + 9999 + target_index * 1009
        )
        selected_parts.append(indices[target_selected])
        bucket_parts.append(target_buckets)
        for name, count in target_counts.items():
            candidate_bucket_counts[name] = candidate_bucket_counts.get(name, 0) + count
    selected = np.concatenate(selected_parts)
    selected_buckets = np.concatenate(bucket_parts)
    if "target_labels" in candidates:
        order = np.random.RandomState(seed + 19999).permutation(len(selected))
        selected = selected[order]
        selected_buckets = selected_buckets[order]

    selected_local = np.asarray(candidates["local_coords"][selected], dtype=np.float32)
    selected_body_ids = np.asarray(candidates["body_ids"][selected], dtype=np.int32)
    selected_geom_ids = np.asarray(candidates["geom_ids"][selected], dtype=np.int32)
    selected_world = np.asarray(candidates["world_coords"][selected], dtype=np.float32)
    selected_segment_ids = np.asarray(candidates["segment_ids"][selected], dtype=np.int32)
    selected_object_names = get_kubric_segment_names(model, selected_segment_ids)
    selected_source_cameras = np.asarray(candidates["source_cameras"][selected])
    selected_segment_area_fraction = np.asarray(
        candidates["segment_area_fraction"][selected], dtype=np.float32
    )
    selected_local_segment_support = np.asarray(
        candidates["local_segment_support"][selected], dtype=np.float32
    )
    selected_source_edge_distance = np.asarray(
        candidates["source_edge_distance"][selected], dtype=np.float32
    )
    selected_occluder_edge_distance = np.asarray(
        candidates["occluder_edge_distance"][selected], dtype=np.float32
    )
    selected_source_depth = np.asarray(candidates["source_depth"][selected], dtype=np.float32)
    selected_depth_penalized_area = np.asarray(
        candidates["depth_penalized_area"][selected], dtype=np.float32
    )
    selected_depth_penalized_support = np.asarray(
        candidates["depth_penalized_support"][selected], dtype=np.float32
    )
    selected_target_labels = (
        None if "target_labels" not in candidates else candidates["target_labels"][selected]
    )
    selected_query_points = np.asarray(candidates["query_points"][selected], dtype=np.float32)
    surface_normals = estimate_aligned_surface_normals(
        model=model,
        data=data,
        camera_names=camera_names,
        camera_registry=env.camera_manager.registry,
        rendered_depths=rendered_depths,
        rendered_segmentations=rendered_segmentations,
        query_points=selected_query_points,
        source_cameras=selected_source_cameras,
        world_points=selected_world,
        body_ids=selected_body_ids,
        geom_ids=selected_geom_ids,
        img_width=img_width,
        img_height=img_height,
    )

    camera_points = {}
    for camera_name in camera_names:
        camera = env.camera_manager.registry[camera_name]
        if cached_candidate_cameras:
            projected = {
                key: value[selected] for key, value in cached_candidate_cameras[camera_name].items()
            }
        else:
            projected = project_points(
                camera_name, selected_local, selected_body_ids, selected_geom_ids
            )
            source_mask = selected_source_cameras == camera_name
            selected_source_depth[source_mask] = projected["point_depth"][source_mask]
        _, intrinsics = _build_camera_matrices(camera, img_width, img_height)
        camera_points[camera_name] = {**projected, "intrinsics": intrinsics}

    visible_camera_count = np.sum(
        np.stack(
            [camera_points[name]["visibility"] > 0.5 for name in camera_names],
            axis=0,
        ),
        axis=0,
    ).astype(np.int16)
    in_frame_occluded_camera_count = np.sum(
        np.stack(
            [
                camera_points[name]["in_frame"] & (camera_points[name]["visibility"] <= 0.5)
                for name in camera_names
            ],
            axis=0,
        ),
        axis=0,
    ).astype(np.int16)
    selected_bucket_names, selected_bucket_counts = np.unique(selected_buckets, return_counts=True)
    candidate_count_names = np.asarray(list(candidate_bucket_counts), dtype=str)
    candidate_count_values = np.asarray(list(candidate_bucket_counts.values()), dtype=np.int64)

    result = {
        "sampling_method": sampling_mode,
        "sampling_stride": stride,
        "sampling_phase": np.asarray((0, *spatial_phase), dtype=np.int32),
        "max_sampled_fraction": float(max_sampled_fraction),
        "include_background": bool(include_background),
        "visibility_method": RASTER_VISIBILITY_METHOD,
        "visibility_depth_relative_tolerance": float(visibility_depth_relative_tolerance),
        "visibility_depth_absolute_tolerance_m": float(visibility_depth_absolute_tolerance_m),
        "exclude_raster_ambiguous": bool(exclude_raster_ambiguous),
        "num_raw_candidates": raw_candidate_count,
        "num_source_edge_unsafe_candidates_excluded": (source_edge_unsafe_candidate_count),
        "num_candidates_before_cross_view_shortlist": (candidates_before_cross_view_shortlist),
        "num_cross_view_candidates": cross_view_candidate_count,
        "num_cross_view_candidates_shortlisted_out": (cross_view_shortlisted_out_count),
        "num_candidates": len(candidates["local_coords"]),
        "num_raster_ambiguous_candidates_filtered": (raster_ambiguous_candidate_count),
        "track_ids": np.arange(max_points, dtype=np.int32),
        "body_ids": selected_body_ids,
        "geom_ids": selected_geom_ids,
        "segment_ids": selected_segment_ids,
        "point_object_names": selected_object_names,
        "local_coords": selected_local,
        "points_3d": selected_world,
        "query_points": selected_query_points,
        "query_source_cameras": selected_source_cameras,
        "surface_normal_orientation": "toward_query_source_camera",
        "surface_normal_world_frame": "mujoco_world",
        "surface_normal_local_frame": "owning_body_local",
        "surface_normal_camera_frame": "query_source_camera_cv",
        **surface_normals,
        "sampling_buckets": np.asarray(selected_buckets, dtype=str),
        "selected_bucket_names": np.asarray(selected_bucket_names, dtype=str),
        "selected_bucket_counts": np.asarray(selected_bucket_counts, dtype=np.int64),
        "candidate_bucket_names": candidate_count_names,
        "candidate_bucket_counts": candidate_count_values,
        "failure_target_bucket_names": np.asarray(FAILURE_TARGET_BUCKET_NAMES, dtype=str),
        "failure_target_fractions": np.asarray(failure_target_fractions, dtype=np.float32),
        "source_segment_area_fraction": selected_segment_area_fraction,
        "source_local_segment_support": selected_local_segment_support,
        "source_edge_distance_px": selected_source_edge_distance,
        "source_depth_m": selected_source_depth,
        "depth_penalized_segment_area_fraction": selected_depth_penalized_area,
        "depth_penalized_local_segment_support": selected_depth_penalized_support,
        "occluder_edge_distance_px": selected_occluder_edge_distance,
        "visible_camera_count": visible_camera_count,
        "in_frame_occluded_camera_count": in_frame_occluded_camera_count,
        "failure_edge_distance_px": float(failure_edge_distance_px),
        "failure_min_source_edge_distance_px": float(failure_min_source_edge_distance_px),
        "failure_max_cross_view_candidates": int(failure_max_cross_view_candidates),
        "failure_shortlist_prioritize_edges": bool(failure_shortlist_prioritize_edges),
        "failure_small_segment_area_fraction": float(failure_small_segment_area_fraction),
        "failure_local_support_threshold": float(failure_local_support_threshold),
        "failure_dense_boundary_radius_px": int(failure_dense_boundary_radius_px),
        "failure_depth_penalty_reference_m": float(failure_depth_penalty_reference_m),
        "failure_depth_sampling_min_weight": float(failure_depth_sampling_min_weight),
        "cameras": camera_points,
    }
    if selected_target_labels is not None:
        result["point_target_labels"] = selected_target_labels
        result["eligible_body_ids"] = np.asarray(sorted(foreground_body_ids), dtype=np.int32)
    return result


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
    return_diagnostics: bool = False,
    segmentation_frame: np.ndarray | None = None,
    geom_ids: np.ndarray | None = None,
    visibility_depth_relative_tolerance: float = (DEFAULT_VISIBILITY_DEPTH_RELATIVE_TOLERANCE),
    visibility_depth_absolute_tolerance_m: float = (DEFAULT_VISIBILITY_DEPTH_ABSOLUTE_TOLERANCE_M),
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]
):
    """Compute 2D projections and visibility for all tracked points in one frame.

    Args:
        data: Object with .xpos and .xmat arrays (MjData or compatible)
        local_coords: (N, 3) body-local coordinates
        body_ids: (N,) body id per point
        camera: Camera object (used to build matrices if precomputed not given)
        img_width: Image width in pixels
        img_height: Image height in pixels
        depth_frame: (H, W) float32 rendered depth in meters
        occlusion_tolerance: Legacy single-pixel depth tolerance in meters. This
            is used only when segmentation/geom identity is unavailable.
        precomputed_w2c: Optional (4, 4) precomputed world-to-camera matrix
        precomputed_intrinsics: Optional (3, 3) precomputed intrinsic matrix
        segmentation_frame: Optional renderer segmentation with geom id in
            channel 0. When supplied together with ``geom_ids``, visibility is
            computed from the four raster pixels surrounding each continuous
            projection instead of the legacy nearest-pixel depth test.
        geom_ids: Exact source geom id for every tracked physical point.
        visibility_depth_relative_tolerance: Relative surface-depth agreement
            tolerance for renderer-supported visibility.
        visibility_depth_absolute_tolerance_m: Minimum absolute surface-depth
            agreement tolerance in meters.

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

    if (segmentation_frame is None) != (geom_ids is None):
        raise ValueError(
            "segmentation_frame and geom_ids must either both be supplied or both omitted"
        )
    if visibility_depth_relative_tolerance < 0:
        raise ValueError("visibility_depth_relative_tolerance must be non-negative")
    if visibility_depth_absolute_tolerance_m < 0:
        raise ValueError("visibility_depth_absolute_tolerance_m must be non-negative")
    if geom_ids is not None:
        geom_ids = np.asarray(geom_ids, dtype=np.int32)
        if geom_ids.shape != (N,):
            raise ValueError(f"geom_ids must have shape {(N,)}, got {geom_ids.shape}")
        segmentation_frame = np.asarray(segmentation_frame)
        if segmentation_frame.ndim != 3 or segmentation_frame.shape[2] < 1:
            raise ValueError("segmentation_frame must have shape (height, width, channels)")
        if segmentation_frame.shape[:2] != (img_height, img_width):
            raise ValueError(
                "segmentation_frame dimensions do not match the requested image size: "
                f"{segmentation_frame.shape[:2]} vs {(img_height, img_width)}"
            )

    visibility = np.zeros(N, dtype=np.float32)
    rendered_depth_values = np.full(N, np.nan, dtype=np.float32)
    depth_residual = np.full(N, np.nan, dtype=np.float32)
    visibility_tolerance = np.maximum(
        visibility_depth_absolute_tolerance_m,
        np.abs(depths) * visibility_depth_relative_tolerance,
    ).astype(np.float32)
    visibility_reason = np.full(N, "out_of_frame", dtype="<U32")
    raster_ambiguous = np.zeros(N, dtype=bool)
    matching_geom_neighbor_count = np.zeros(N, dtype=np.int8)
    min_matching_geom_depth_error = np.full(N, np.nan, dtype=np.float32)
    max_neighbor_depth = np.full(N, np.nan, dtype=np.float32)
    if depth_frame is not None and depth_frame.size > 0:
        in_frame_indices = np.where(in_frame)[0]
        if len(in_frame_indices) > 0:
            point_depth = depths[in_frame_indices]
            if segmentation_frame is None:
                px_int = np.clip(px[in_frame_indices].astype(int), 0, img_width - 1)
                py_int = np.clip(py[in_frame_indices].astype(int), 0, img_height - 1)
                rendered_depth = depth_frame[py_int, px_int]
                residual = point_depth - rendered_depth
                rendered_depth_values[in_frame_indices] = rendered_depth
                depth_residual[in_frame_indices] = residual
                not_occluded = residual < occlusion_tolerance
                visibility[in_frame_indices] = np.where(not_occluded, 1.0, 0.0)
                visibility_reason[in_frame_indices] = np.where(
                    not_occluded, "visible", "occluded_depth_confirmed"
                )
            else:
                # Kubric treats a projected coordinate as a continuous raster
                # location. Subtracting 0.5 converts from raster coordinates to
                # the pixel-center grid; the surrounding 2x2 pixels then provide
                # renderer support without a nearest-pixel boundary artifact.
                raster_x = px[in_frame_indices] - 0.5
                raster_y = py[in_frame_indices] - 0.5
                x0 = np.floor(raster_x).astype(np.int64)
                y0 = np.floor(raster_y).astype(np.int64)
                x1 = x0 + 1
                y1 = y0 + 1
                xs = np.stack([x0, x0, x1, x1], axis=1)
                ys = np.stack([y0, y1, y0, y1], axis=1)
                xs = np.clip(xs, 0, img_width - 1)
                ys = np.clip(ys, 0, img_height - 1)

                neighbor_depth = np.asarray(depth_frame[ys, xs], dtype=np.float32)
                geom_id_map = np.asarray(segmentation_frame[:, :, 0], dtype=np.int32)
                neighbor_geom = geom_id_map[ys, xs]
                finite_depth = np.isfinite(neighbor_depth)
                geom_match = neighbor_geom == geom_ids[in_frame_indices, None]
                depth_error = np.abs(neighbor_depth - point_depth[:, None])
                tolerance = visibility_tolerance[in_frame_indices, None]
                surface_support = geom_match & finite_depth & (depth_error <= tolerance)
                supported = np.any(surface_support, axis=1)

                finite_or_neg_inf = np.where(finite_depth, neighbor_depth, -np.inf)
                local_max_depth = np.max(finite_or_neg_inf, axis=1)
                has_finite_depth = np.any(finite_depth, axis=1)
                local_max_depth = np.where(has_finite_depth, local_max_depth, np.nan)
                depth_confirmed_occlusion = has_finite_depth & (
                    local_max_depth < point_depth - visibility_tolerance[in_frame_indices]
                )
                ambiguous = ~supported & ~depth_confirmed_occlusion

                visibility[in_frame_indices] = supported.astype(np.float32)
                visibility_reason[in_frame_indices] = np.where(
                    supported,
                    "visible",
                    np.where(
                        depth_confirmed_occlusion,
                        "occluded_depth_confirmed",
                        "raster_ambiguous",
                    ),
                )
                raster_ambiguous[in_frame_indices] = ambiguous
                matching_geom_neighbor_count[in_frame_indices] = np.sum(
                    geom_match, axis=1, dtype=np.int8
                )
                matching_errors = np.where(geom_match & finite_depth, depth_error, np.inf)
                local_min_error = np.min(matching_errors, axis=1)
                min_matching_geom_depth_error[in_frame_indices] = np.where(
                    np.isfinite(local_min_error), local_min_error, np.nan
                )
                max_neighbor_depth[in_frame_indices] = local_max_depth
                rendered_depth_values[in_frame_indices] = local_max_depth
                depth_residual[in_frame_indices] = point_depth - local_max_depth
    else:
        visibility[in_frame] = 1.0
        visibility_reason[in_frame] = "visible"

    if return_diagnostics:
        return (
            coords_2d,
            visibility,
            world_pts,
            {
                "in_frame": in_frame,
                "point_depth": depths.astype(np.float32),
                "rendered_depth": rendered_depth_values,
                "depth_residual": depth_residual,
                "visibility_tolerance": visibility_tolerance,
                "visibility_reason": visibility_reason,
                "raster_ambiguous": raster_ambiguous,
                "matching_geom_neighbor_count": matching_geom_neighbor_count,
                "min_matching_geom_depth_error": min_matching_geom_depth_error,
                "max_neighbor_depth": max_neighbor_depth,
            },
        )
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
    query_points: np.ndarray | None = None,
    sampling_method: str | None = None,
    sampling_stride: int | None = None,
    sampling_phase: np.ndarray | None = None,
    max_sampled_fraction: float | None = None,
    segment_ids: np.ndarray | None = None,
    track_ids: np.ndarray | None = None,
    aligned_across_cameras: bool | None = None,
    query_source_cameras: np.ndarray | None = None,
    geom_ids: np.ndarray | None = None,
    visibility_method: str | None = None,
    in_frame: np.ndarray | None = None,
    raster_ambiguous: np.ndarray | None = None,
    exclude_raster_ambiguous: bool | None = None,
    visibility_filter_stats: dict[str, int] | None = None,
    visibility_check_cameras: np.ndarray | None = None,
) -> None:
    """Save point tracks, with explicit masks for uncertain raster visibility.

    Kubric video tracks use exact geometry identity and depth support.
    ``visibility`` remains binary for compatibility; ambiguous entries are zero
    and must be excluded from supervision/scoring using ``visibility_valid``.
    Reasons are stored as uint8 codes indexing ``visibility_reason_names`` to
    avoid holding a large (T, N) Unicode array in memory for full videos.
    """
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
    if query_points is not None:
        data["query_points"] = np.asarray(query_points, dtype=np.float32)
    if sampling_method is not None:
        data["sampling_method"] = np.asarray(sampling_method)
    if sampling_stride is not None:
        data["sampling_stride"] = np.asarray(sampling_stride, dtype=np.int32)
    if sampling_phase is not None:
        data["sampling_phase"] = np.asarray(sampling_phase, dtype=np.int32)
    if max_sampled_fraction is not None:
        data["max_sampled_fraction"] = np.asarray(max_sampled_fraction, dtype=np.float32)
    if segment_ids is not None:
        data["segment_ids"] = np.asarray(segment_ids, dtype=np.int32)
    if track_ids is not None:
        data["track_ids"] = np.asarray(track_ids, dtype=np.int32)
    if aligned_across_cameras is not None:
        data["aligned_across_cameras"] = np.asarray(aligned_across_cameras, dtype=np.bool_)
    if query_source_cameras is not None:
        data["query_source_cameras"] = np.asarray(query_source_cameras, dtype=str)
    if geom_ids is not None:
        geom_ids = np.asarray(geom_ids, dtype=np.int32)
        if geom_ids.shape != (trajs_2d.shape[1],):
            raise ValueError("geom_ids must align with the point dimension of trajs_2d")
        data["geom_ids"] = geom_ids
    if visibility_method is not None:
        data["visibility_method"] = np.asarray(visibility_method)
    if (in_frame is None) != (raster_ambiguous is None):
        raise ValueError("in_frame and raster_ambiguous must both be supplied or both omitted")
    if visibility_method == RASTER_VISIBILITY_METHOD and (geom_ids is None or in_frame is None):
        raise ValueError("Exact-geom visibility requires geom_ids and per-frame visibility masks")
    if in_frame is not None:
        in_frame = np.asarray(in_frame, dtype=bool)
        raster_ambiguous = np.asarray(raster_ambiguous, dtype=bool)
        expected_shape = trajs_2d.shape[:2]
        if any(
            values.shape != expected_shape for values in (visibility, in_frame, raster_ambiguous)
        ):
            raise ValueError("Visibility arrays must have shape (frames, points) matching trajs_2d")
        visible = visibility > 0.5
        if np.any(visible & (~in_frame | raster_ambiguous)) or np.any(raster_ambiguous & ~in_frame):
            raise ValueError("Visible/ambiguous states are inconsistent with the in-frame mask")
        reasons = np.full(expected_shape, 2, dtype=np.uint8)
        reasons[~in_frame] = 1
        reasons[visible] = 0
        reasons[raster_ambiguous] = 3
        data.update(
            in_frame=in_frame,
            raster_ambiguous=raster_ambiguous,
            visibility_valid=~raster_ambiguous,
            visibility_reason_codes=reasons,
            visibility_reason_names=np.asarray(POINT_TRACK_VISIBILITY_REASON_NAMES),
        )
    if visibility_method == RASTER_VISIBILITY_METHOD:
        data["visibility_depth_relative_tolerance"] = np.asarray(
            DEFAULT_VISIBILITY_DEPTH_RELATIVE_TOLERANCE, dtype=np.float32
        )
        data["visibility_depth_absolute_tolerance_m"] = np.asarray(
            DEFAULT_VISIBILITY_DEPTH_ABSOLUTE_TOLERANCE_M, dtype=np.float32
        )
    if exclude_raster_ambiguous is not None:
        data["exclude_raster_ambiguous"] = np.asarray(exclude_raster_ambiguous, dtype=bool)
        if exclude_raster_ambiguous and (
            in_frame is None
            or geom_ids is None
            or visibility_method != RASTER_VISIBILITY_METHOD
            or np.any(raster_ambiguous)
        ):
            raise ValueError("Clean-track export requires exact-geom masks with no ambiguity")
    if visibility_filter_stats is not None:
        for key, value in visibility_filter_stats.items():
            data[f"visibility_filter_{key}"] = np.asarray(value, dtype=np.int64)
    if visibility_check_cameras is not None:
        camera_names = np.asarray(visibility_check_cameras, dtype=str)
        if camera_names.ndim != 1 or len(camera_names) == 0:
            raise ValueError("visibility_check_cameras must be a nonempty 1D list")
        data["visibility_check_cameras"] = camera_names
    np.savez_compressed(save_path, **data)
