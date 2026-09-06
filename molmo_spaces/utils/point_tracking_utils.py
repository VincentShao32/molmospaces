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
