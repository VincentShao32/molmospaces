"""Phase-stratified, synchronized multiview snapshot collection."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _scalar_int(value: Any) -> int:
    """Convert a scalar sensor value to an integer phase id."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected one policy phase value, got shape {array.shape}")
    return int(array.reshape(-1)[0])


def _jsonable(value: Any) -> Any:
    """Convert numpy-heavy snapshot metadata into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _safe_path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError(f"Invalid empty path component derived from {value!r}")
    return safe


class PhaseSnapshotCollector:
    """Uniformly sample synchronized camera sets from requested policy phases.

    Sampling uses one reservoir per phase, so every observed frame in a phase has
    equal probability of being retained without storing the complete video.  The
    standard object-manipulation policy reports ``gripper-open`` both before motion
    starts and while releasing the placed object.  The latter occurrence is exposed
    as the semantic phase ``release`` once ``place`` has been observed.
    """

    def __init__(
        self,
        *,
        phase_name_to_id: Mapping[str, int],
        camera_names: Sequence[str],
        required_phases: Sequence[str],
        samples_per_phase: int = 1,
        seed: int = 0,
    ) -> None:
        if samples_per_phase < 1:
            raise ValueError("samples_per_phase must be at least 1")
        if not camera_names:
            raise ValueError("At least one camera is required for phase snapshots")

        self.phase_name_to_id = {
            str(name): int(phase_id) for name, phase_id in phase_name_to_id.items()
        }
        self.phase_id_to_name = {phase_id: name for name, phase_id in self.phase_name_to_id.items()}
        if len(self.phase_id_to_name) != len(self.phase_name_to_id):
            raise ValueError("Policy phase ids must be unique")

        self.camera_names = tuple(camera_names)
        self.required_phases = tuple(dict.fromkeys(required_phases))
        if not self.required_phases:
            raise ValueError("At least one required phase must be configured")
        self.samples_per_phase = samples_per_phase

        available_phases = set(self.phase_name_to_id)
        if "gripper-open" in available_phases:
            available_phases.add("release")
        unknown = set(self.required_phases) - available_phases
        if unknown:
            raise ValueError(
                f"Required phase(s) {sorted(unknown)} are not exposed by the policy; "
                f"available phases are {sorted(available_phases)}"
            )

        self._seen_policy_phases: set[str] = set()
        self._counts = {phase: 0 for phase in self.required_phases}
        self._samples: dict[str, list[dict[str, Any]]] = {
            phase: [] for phase in self.required_phases
        }
        self._rngs = {
            phase: np.random.default_rng(np.random.SeedSequence([seed, index]))
            for index, phase in enumerate(self.required_phases)
        }

    def _semantic_phase(self, policy_phase: str) -> str:
        if policy_phase == "gripper-open":
            if "place" in self._seen_policy_phases:
                return "release"
            return "initial-gripper-open"
        return policy_phase

    def observe(
        self,
        observation: Any,
        frame_index: int,
        snapshot_payload_factory: Callable[[], Mapping[str, Any]] | None = None,
    ) -> str:
        """Consider one synchronized observation for phase-stratified sampling."""
        if isinstance(observation, list):
            if len(observation) != 1:
                raise ValueError("Phase snapshots currently require one environment per worker")
            frame_observation = observation[0]
        else:
            frame_observation = observation

        if "policy_phase" not in frame_observation:
            raise KeyError(
                "Observation has no policy_phase sensor. Register a PlannerPolicy "
                "before resetting the task."
            )

        phase_id = _scalar_int(frame_observation["policy_phase"])
        if phase_id not in self.phase_id_to_name:
            raise ValueError(
                f"Observed unknown policy phase id {phase_id}; mapping is {self.phase_name_to_id}"
            )
        policy_phase = self.phase_id_to_name[phase_id]
        semantic_phase = self._semantic_phase(policy_phase)
        self._seen_policy_phases.add(policy_phase)

        if semantic_phase not in self._samples:
            return semantic_phase

        missing_cameras = [
            camera_name for camera_name in self.camera_names if camera_name not in frame_observation
        ]
        if missing_cameras:
            raise KeyError(f"Observation is missing phase snapshot camera(s): {missing_cameras}")

        self._counts[semantic_phase] += 1
        count = self._counts[semantic_phase]
        retained = self._samples[semantic_phase]
        if len(retained) < self.samples_per_phase:
            replacement_index = len(retained)
        else:
            replacement_index = int(self._rngs[semantic_phase].integers(0, count))
            if replacement_index >= self.samples_per_phase:
                return semantic_phase

        snapshot = {
            "frame_index": int(frame_index),
            "phase": semantic_phase,
            "policy_phase": policy_phase,
            "policy_phase_id": phase_id,
            "images": {
                camera_name: np.asarray(frame_observation[camera_name]).copy()
                for camera_name in self.camera_names
            },
        }
        for key in ("qpos", "qvel", "robot_base_pose"):
            if key in frame_observation:
                snapshot[key] = copy.deepcopy(frame_observation[key])
        snapshot["camera_parameters"] = {
            camera_name: copy.deepcopy(frame_observation.get(f"sensor_param_{camera_name}"))
            for camera_name in self.camera_names
            if f"sensor_param_{camera_name}" in frame_observation
        }
        if snapshot_payload_factory is not None:
            snapshot.update(snapshot_payload_factory())

        if replacement_index == len(retained):
            retained.append(snapshot)
        else:
            retained[replacement_index] = snapshot
        return semantic_phase

    @property
    def missing_phases(self) -> tuple[str, ...]:
        return tuple(
            phase
            for phase in self.required_phases
            if len(self._samples[phase]) < self.samples_per_phase
        )

    @property
    def is_complete(self) -> bool:
        return not self.missing_phases

    def finalize(self, *, episode_seed: int) -> dict[str, Any]:
        """Return collected snapshots and coverage metadata."""
        return {
            "episode_seed": int(episode_seed),
            "camera_names": list(self.camera_names),
            "required_phases": list(self.required_phases),
            "samples_per_phase": self.samples_per_phase,
            "phase_frame_counts": dict(self._counts),
            "missing_phases": list(self.missing_phases),
            "complete": self.is_complete,
            "snapshots": {
                phase: sorted(samples, key=lambda sample: sample["frame_index"])
                for phase, samples in self._samples.items()
            },
        }


def discard_camera_frames(observation: Any, camera_names: Sequence[str]) -> None:
    """Drop rendered arrays after the policy and snapshot collector have consumed them."""
    observations = observation if isinstance(observation, list) else [observation]
    for frame_observation in observations:
        for camera_name in camera_names:
            frame_observation.pop(camera_name, None)
            frame_observation.pop(f"{camera_name}_depth", None)
            frame_observation.pop(f"{camera_name}_seg", None)


def save_phase_snapshot_episode(
    snapshot_data: Mapping[str, Any],
    *,
    output_dir: Path,
    episode_index: int,
    batch_suffix: str = "",
    scene_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write PNG image sets and JSON metadata for one accepted episode."""
    # Episode indices restart in each batch, so include its identity in the path.
    episode_dir = output_dir / f"episode_{episode_index:08d}{batch_suffix}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    episode_metadata: dict[str, Any] = {
        "episode_index": episode_index,
        "episode_seed": snapshot_data["episode_seed"],
        "camera_names": snapshot_data["camera_names"],
        "required_phases": snapshot_data["required_phases"],
        "samples_per_phase": snapshot_data["samples_per_phase"],
        "phase_frame_counts": snapshot_data["phase_frame_counts"],
        "complete": snapshot_data["complete"],
        "missing_phases": snapshot_data["missing_phases"],
        "scene": {},
        "snapshots": [],
    }
    if scene_metadata is not None:
        for key in (
            "task_type",
            "task_description",
            "text",
            "object_name",
            "place_receptacle_name",
            "policy_phases",
        ):
            if key in scene_metadata:
                episode_metadata["scene"][key] = _jsonable(scene_metadata[key])

    for phase in snapshot_data["required_phases"]:
        phase_dir_name = _safe_path_component(phase)
        phase_samples = snapshot_data["snapshots"][phase]
        for sample_index, sample in enumerate(phase_samples):
            sample_dir = episode_dir / phase_dir_name / f"sample_{sample_index:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            image_paths = {}
            for camera_name, image in sample["images"].items():
                filename = f"{_safe_path_component(camera_name)}.png"
                image_path = sample_dir / filename
                image_array = np.asarray(image)
                if image_array.dtype != np.uint8:
                    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
                Image.fromarray(image_array).save(image_path)
                image_paths[camera_name] = str(image_path.relative_to(episode_dir))

            sample_metadata = {
                key: _jsonable(value)
                for key, value in sample.items()
                if key not in {"images", "points"}
            }
            sample_metadata["sample_index"] = sample_index
            sample_metadata["images"] = image_paths
            if "points" in sample:
                point_data = sample["points"]
                point_path = sample_dir / "points.npz"
                flattened_points = {
                    key: value for key, value in point_data.items() if key != "cameras"
                }
                for camera_name, camera_points in point_data["cameras"].items():
                    for key, value in camera_points.items():
                        flattened_points[f"{camera_name}_{key}"] = value
                np.savez_compressed(point_path, **flattened_points)
                sample_metadata["points"] = {
                    "path": str(point_path.relative_to(episode_dir)),
                    "num_points": int(len(point_data["track_ids"])),
                    "sampling_method": str(point_data["sampling_method"]),
                    "aligned_across_cameras": True,
                }
            episode_metadata["snapshots"].append(sample_metadata)

    metadata_path = episode_dir / "metadata.json"
    metadata_path.write_text(json.dumps(episode_metadata, indent=2) + "\n")
    return {
        "episode_index": episode_index,
        "episode_seed": snapshot_data["episode_seed"],
        "episode_dir": episode_dir.name,
        "metadata": str(metadata_path.relative_to(output_dir)),
        "num_image_sets": len(episode_metadata["snapshots"]),
        "required_phases": snapshot_data["required_phases"],
        "complete": snapshot_data["complete"],
    }
