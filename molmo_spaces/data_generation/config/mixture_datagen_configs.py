"""Mixture datagen configs.

A mixture bundles several already-registered datagen configs into a single
sbatch job. The mixture runner (:mod:`molmo_spaces.data_generation.mixture_main`)
walks the components in order and runs each through
:class:`ParallelRolloutRunner` with that component's house budget; each
sub-config still uses its own ``num_workers`` for episode-level parallelism.

To add a new mixture, register a zero-arg factory that returns a
:class:`MixtureSpec`. Sub-configs are referenced by the same name they're
registered under in :mod:`molmo_spaces.data_generation.config_registry`.
"""

from molmo_spaces.data_generation.mixture_registry import (
    MixtureComponent,
    MixtureSpec,
    register_mixture,
)


@register_mixture("PointTrackTrioMixture")
def _point_track_trio() -> MixtureSpec:
    """Franka pick + Franka pick-and-place + RBY1 pick, with point tracks.

    Defaults to 2 houses x 3 episodes/house x 5000 points per component
    (smoke-test sizing). Override on the CLI:
        --override         <ConfigName>=<num_houses>
        --samples-override <ConfigName>=<episodes_per_house>
        --points-override  <ConfigName>=<point_track_num_points>
    """
    return MixtureSpec(
        components=(
            MixtureComponent(
                config_name="FrankaPickPointTrackDebug",
                num_houses=3000,
                samples_per_house=10,
                point_track_num_points=32000,
                point_track_include_background=True,
                point_track_background_fraction=0.4
            ),
            MixtureComponent(
                config_name="FrankaPickAndPlacePointTrack",
                num_houses=3000,
                samples_per_house=10,
                point_track_num_points=32000,
                point_track_include_background=True,
                point_track_background_fraction=0.4
            ),
            MixtureComponent(
                config_name="RBY1PickPointTrack",
                num_houses=3000,
                samples_per_house=10,
                point_track_num_points=32000,
                point_track_include_background=True,
                point_track_background_fraction=0.4
            ),
        ),
    )

@register_mixture("FrankaPickAndPlacePointTrackOnly")
def _franka_pick_and_place_point_track() -> MixtureSpec:
    """Single-component mixture: just FrankaPickAndPlacePointTrack at full scale."""
    return MixtureSpec(
        components=(
            MixtureComponent(
                config_name="FrankaPickAndPlacePointTrack",
                num_houses=4000,
                samples_per_house=3,
                point_track_num_points=32768,
                point_track_include_background=True,
                point_track_background_fraction=0.4,
                scene_dataset="procthor-10k",
                data_split="train",
            ),
        ),
    )

@register_mixture("FrankaPickPointTrackOnly")
def _franka_pick_point_track_only() -> MixtureSpec:
    """Single-component mixture: just FrankaPickPointTrack at full scale."""
    return MixtureSpec(
        components=(
            MixtureComponent(
                config_name="FrankaPickPointTrackDebug",
                num_houses=3000,
                samples_per_house=1,
                point_track_num_points=10000,
                point_track_include_background=True,
                point_track_background_fraction=0.2,
                scene_dataset="procthor-10k",
                data_split="train",
            ),
        ),
    )


@register_mixture("FrankaPickPointTrackAnimatedCamOnly")
def _franka_pick_point_track_animated_cam() -> MixtureSpec:
    """Single-component mixture: FrankaPickPointTrack with animated camera motion."""
    return MixtureSpec(
        components=(
            MixtureComponent(
                config_name="FrankaPickPointTrackAnimatedCam",
                num_houses=3000,
                samples_per_house=1,
                point_track_num_points=5000,
                point_track_include_background=True,
                point_track_background_fraction=0.2,
                scene_dataset="procthor-10k",
                data_split="train",
            ),
        ),
    )


@register_mixture("FrankaPickPointTrackWristOnly")
def _franka_pick_point_track_wrist_only() -> MixtureSpec:
    """Franka pick, one wrist-camera episode per house, with 5000 tracks."""
    return MixtureSpec(
        components=(
            MixtureComponent(
                config_name="FrankaPickPointTrackWristOnly",
                num_houses=3000,
                samples_per_house=1,
                point_track_num_points=5000,
                point_track_include_background=True,
                point_track_background_fraction=0.2,
                scene_dataset="procthor-10k",
                data_split="train",
            ),
        ),
    )


@register_mixture("RBY1PickPointTrackOnly")
def _rby1_pick_point_track() -> MixtureSpec:
    """Single-component mixture: just RBY1PickPointTrack at full scale."""
    return MixtureSpec(
        components=(
            MixtureComponent(
                config_name="RBY1PickPointTrack",
                num_houses=3000,
                samples_per_house=10,
                point_track_num_points=32768,
                point_track_include_background=True,
                point_track_background_fraction=0.4
            ),
        ),
    )


@register_mixture("RUMPickPointTrackOnly")
def _rum_pick_point_track() -> MixtureSpec:
    """Single-component mixture: just RUMPickPointTrack at full scale."""
    return MixtureSpec(
        components=(
            MixtureComponent(
                config_name="RUMPickPointTrack",
                num_houses=3000,
                samples_per_house=10,
                point_track_num_points=32768,
            ),
        ),
    )
