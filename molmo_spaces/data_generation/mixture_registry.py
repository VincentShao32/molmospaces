"""Registry and dataclasses for mixture datagen specs.

A *mixture* describes a set of existing :class:`MlSpacesExpConfig`-derived configs
(referenced by their registered name in
:mod:`molmo_spaces.data_generation.config_registry`) plus an absolute house count
to run for each. The mixture entry point (:mod:`molmo_spaces.data_generation.mixture_main`)
walks the components in order and runs each through the normal
:class:`ParallelRolloutRunner` in a single job.

This registry is intentionally separate from the per-config registry because a
mixture is meta-config: it is *not* itself an ``MlSpacesExpConfig`` and therefore
should not be looked up via :func:`get_config_class`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MixtureComponent:
    """One leg of a mixture.

    Attributes:
        config_name: Name of a registered :class:`MlSpacesExpConfig` subclass
            (i.e. a key in :mod:`molmo_spaces.data_generation.config_registry`).
        num_houses: How many distinct ProcThor houses to sample for this leg.
        samples_per_house: How many episodes to attempt per house (this maps
            directly onto ``task_sampler_config.samples_per_house`` in the
            sub-config). Total target episodes for this leg is roughly
            ``num_houses * samples_per_house``.
        point_track_num_points: Optional override for the sub-config's
            ``point_track_num_points`` field (number of point tracks generated
            per episode). ``None`` keeps whatever the sub-config defines.
            Components whose sub-config doesn't have this field will raise at
            instantiation time if a non-None value is supplied.
        point_track_include_background: Optional override for the sub-config's
            ``point_track_include_background`` field. When ``True``, reserves
            a fraction of the point-track budget for static scene geometry
            (walls/floor/furniture). ``None`` keeps the sub-config default.
        point_track_background_fraction: Optional override for the sub-config's
            ``point_track_background_fraction`` field (``[0, 1]``). Ignored
            unless background sampling is enabled. ``None`` keeps the
            sub-config default.
    """

    config_name: str
    num_houses: int
    samples_per_house: int
    point_track_num_points: int | None = None
    point_track_include_background: bool | None = None
    point_track_background_fraction: float | None = None


@dataclass(frozen=True)
class MixtureSpec:
    """A named mixture of datagen sub-configs."""

    components: tuple[MixtureComponent, ...]
    max_house_index: int = 99925  # Upper bound for sampling random house indices

    def __post_init__(self) -> None:
        # Catch the classic single-element-tuple bug ("components=(x)" instead
        # of "components=(x,)") at construction time, before sbatch eats 30s
        # of import overhead just to crash on the first iteration.
        if isinstance(self.components, MixtureComponent):
            raise TypeError(
                "MixtureSpec.components must be a tuple of MixtureComponent, "
                "but got a single MixtureComponent. Did you forget the trailing "
                "comma in `components=(MixtureComponent(...),)`?"
            )
        if not isinstance(self.components, tuple):
            # Tolerate list/other sequence by coercing -- frozen dataclass needs object.__setattr__.
            try:
                coerced = tuple(self.components)
            except TypeError as e:
                raise TypeError(
                    f"MixtureSpec.components must be an iterable of "
                    f"MixtureComponent, got {type(self.components).__name__}"
                ) from e
            object.__setattr__(self, "components", coerced)
        if len(self.components) == 0:
            raise ValueError("MixtureSpec.components must be non-empty.")
        for i, c in enumerate(self.components):
            if not isinstance(c, MixtureComponent):
                raise TypeError(
                    f"MixtureSpec.components[{i}] must be a MixtureComponent, "
                    f"got {type(c).__name__}"
                )


_MIXTURE_REGISTRY: dict[str, Callable[[], MixtureSpec]] = {}


def register_mixture(name: str, *, strict: bool = True):
    """Decorator: register a zero-arg factory that returns a :class:`MixtureSpec`.

    Using a factory (rather than a bare ``MixtureSpec`` instance) lets the
    registration site lazily reference other modules without import-time issues.
    """

    def decorator(fn: Callable[[], MixtureSpec]) -> Callable[[], MixtureSpec]:
        if name in _MIXTURE_REGISTRY:
            existing = _MIXTURE_REGISTRY[name]
            existing_id = f"{existing.__module__}.{existing.__name__}"
            new_id = f"{fn.__module__}.{fn.__name__}"
            if strict:
                raise ValueError(
                    f"Mixture '{name}' already registered as {existing_id}, "
                    f"trying to register as {new_id}"
                )
            log.warning(
                "Overriding existing mixture '%s'. Was %s, now %s",
                name,
                existing_id,
                new_id,
            )
        _MIXTURE_REGISTRY[name] = fn
        return fn

    return decorator


def get_mixture(name: str) -> MixtureSpec:
    if name not in _MIXTURE_REGISTRY:
        raise ValueError(
            f"Mixture '{name}' not found. Available mixtures: {list_mixtures()}"
        )
    return _MIXTURE_REGISTRY[name]()


def list_mixtures() -> list[str]:
    return sorted(_MIXTURE_REGISTRY)
