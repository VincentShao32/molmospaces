"""Run a mixture of datagen configs in a single job.

A mixture is a list of ``(registered_config_name, num_houses)`` pairs (defined
in :mod:`molmo_spaces.data_generation.config.mixture_datagen_configs`). For
each component, the runner instantiates the existing sub-config, overrides its
``task_sampler_config.house_inds`` to the requested absolute house count, and
hands it to :class:`ParallelRolloutRunner` -- so each component still uses its
own ``num_workers`` for episode-level parallelism inside this single job.

Usage::

    # Run a registered mixture with its default per-component house counts.
    python -m molmo_spaces.data_generation.mixture_main PointTrackTrioMixture

    # Override per-component house counts (repeatable).
    python -m molmo_spaces.data_generation.mixture_main PointTrackTrioMixture \\
        --override FrankaPickPointTrackDebug=2000 \\
        --override RBY1PickPointTrack=500

    # Override samples-per-house for one component (repeatable).
    python -m molmo_spaces.data_generation.mixture_main PointTrackTrioMixture \\
        --samples-override FrankaPickPointTrackDebug=20

    # Override point_track_num_points for one component (repeatable).
    python -m molmo_spaces.data_generation.mixture_main PointTrackTrioMixture \\
        --points-override FrankaPickPointTrackDebug=2048

    # Enable background point sampling for a component, with a custom fraction
    # of the point budget reserved for static scene geometry.
    python -m molmo_spaces.data_generation.mixture_main RUMPickPointTrackOnly \\
        --include-background RUMPickPointTrack=1 \\
        --bg-fraction RUMPickPointTrack=0.3

    # List available mixtures.
    python -m molmo_spaces.data_generation.mixture_main --list
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from molmo_spaces.data_generation.config_registry import get_config_class
from molmo_spaces.data_generation.main import auto_import_configs
from molmo_spaces.data_generation.mixture_registry import (
    MixtureComponent,
    MixtureSpec,
    get_mixture,
    list_mixtures,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.utils.cache_verifier import verify_and_repair_at_startup


MIXTURES_OUTPUT_ROOT = ASSETS_DIR / "experiment_output" / "datagen" / "mixtures"


def _parse_kv_int(arg: str) -> tuple[str, int]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"Expected format NAME=INT, got {arg!r}"
        )
    name, value = arg.split("=", 1)
    name = name.strip()
    try:
        return name, int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid integer in override {arg!r}: {e}"
        ) from e


def _parse_kv_bool(arg: str) -> tuple[str, bool]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"Expected format NAME=BOOL, got {arg!r}"
        )
    name, value = arg.split("=", 1)
    name = name.strip()
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on", "y", "t"):
        return name, True
    if v in ("0", "false", "no", "off", "n", "f"):
        return name, False
    raise argparse.ArgumentTypeError(
        f"Invalid bool in override {arg!r}: expected 0/1/true/false"
    )


def _parse_kv_float(arg: str) -> tuple[str, float]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"Expected format NAME=FLOAT, got {arg!r}"
        )
    name, value = arg.split("=", 1)
    name = name.strip()
    try:
        return name, float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid float in override {arg!r}: {e}"
        ) from e


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MolmoSpaces mixture datagen pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mixture_name",
        nargs="?",
        type=str,
        help="Name of the registered mixture to run (see --list).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        type=_parse_kv_int,
        metavar="CONFIG=NUM_HOUSES",
        help="Override a component's house count (repeatable).",
    )
    parser.add_argument(
        "--samples-override",
        action="append",
        default=[],
        type=_parse_kv_int,
        metavar="CONFIG=SAMPLES_PER_HOUSE",
        help="Override a component's samples_per_house (repeatable).",
    )
    parser.add_argument(
        "--points-override",
        action="append",
        default=[],
        type=_parse_kv_int,
        metavar="CONFIG=POINT_TRACK_NUM_POINTS",
        help="Override a component's point_track_num_points (repeatable). "
        "Only valid for sub-configs that expose this field.",
    )
    parser.add_argument(
        "--include-background",
        action="append",
        default=[],
        type=_parse_kv_bool,
        metavar="CONFIG=BOOL",
        help="Override a component's point_track_include_background "
        "(repeatable). Accepts 0/1/true/false. When enabled, part of the "
        "point-track budget is reserved for static scene geometry.",
    )
    parser.add_argument(
        "--bg-fraction",
        action="append",
        default=[],
        type=_parse_kv_float,
        metavar="CONFIG=FRACTION",
        help="Override a component's point_track_background_fraction "
        "(repeatable). Float in [0, 1].",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered mixtures and exit.",
    )
    return parser.parse_args()


def apply_overrides(
    spec: MixtureSpec,
    house_overrides: list[tuple[str, int]],
    samples_overrides: list[tuple[str, int]],
    points_overrides: list[tuple[str, int]] | None = None,
    bg_flag_overrides: list[tuple[str, bool]] | None = None,
    bg_fraction_overrides: list[tuple[str, float]] | None = None,
) -> MixtureSpec:
    """Return a new MixtureSpec with the requested per-component overrides."""
    house_map = dict(house_overrides)
    samples_map = dict(samples_overrides)
    points_map = dict(points_overrides or [])
    bg_flag_map = dict(bg_flag_overrides or [])
    bg_fraction_map = dict(bg_fraction_overrides or [])
    known_names = {c.config_name for c in spec.components}
    for name in (
        *house_map,
        *samples_map,
        *points_map,
        *bg_flag_map,
        *bg_fraction_map,
    ):
        if name not in known_names:
            raise ValueError(
                f"Override for {name!r} doesn't match any component in the "
                f"mixture (known: {sorted(known_names)})"
            )

    new_components = tuple(
        MixtureComponent(
            config_name=c.config_name,
            num_houses=house_map.get(c.config_name, c.num_houses),
            samples_per_house=samples_map.get(c.config_name, c.samples_per_house),
            point_track_num_points=points_map.get(
                c.config_name, c.point_track_num_points
            ),
            point_track_include_background=bg_flag_map.get(
                c.config_name, c.point_track_include_background
            ),
            point_track_background_fraction=bg_fraction_map.get(
                c.config_name, c.point_track_background_fraction
            ),
        )
        for c in spec.components
    )
    return MixtureSpec(components=new_components, max_house_index=spec.max_house_index)


def _instantiate_component(
    component: MixtureComponent,
    max_house_index: int,
    component_output_dir: Path,
):
    """Build the sub-config instance for one component, applying overrides."""
    cfg_cls = get_config_class(component.config_name)
    exp_config = cfg_cls()

    if not hasattr(exp_config, "task_sampler_config") or not hasattr(
        exp_config.task_sampler_config, "house_inds"
    ):
        raise ValueError(
            f"Component {component.config_name!r} doesn't expose "
            "task_sampler_config.house_inds; cannot apply num_houses override."
        )

    if component.num_houses > max_house_index:
        raise ValueError(
            f"Component {component.config_name!r}: requested {component.num_houses} "
            f"houses but max_house_index={max_house_index}"
        )
    exp_config.task_sampler_config.house_inds = random.sample(
        range(max_house_index), k=component.num_houses
    )
    exp_config.task_sampler_config.samples_per_house = component.samples_per_house

    if component.point_track_num_points is not None:
        if not hasattr(exp_config, "point_track_num_points"):
            raise ValueError(
                f"Component {component.config_name!r} sub-config does not "
                f"expose `point_track_num_points`; remove the override or "
                f"pick a point-track-capable config."
            )
        exp_config.point_track_num_points = component.point_track_num_points

    if component.point_track_include_background is not None:
        if not hasattr(exp_config, "point_track_include_background"):
            raise ValueError(
                f"Component {component.config_name!r} sub-config does not "
                f"expose `point_track_include_background`."
            )
        exp_config.point_track_include_background = (
            component.point_track_include_background
        )

    if component.point_track_background_fraction is not None:
        frac = component.point_track_background_fraction
        if not 0.0 <= frac <= 1.0:
            raise ValueError(
                f"Component {component.config_name!r}: "
                f"point_track_background_fraction must be in [0, 1], got {frac}"
            )
        if not hasattr(exp_config, "point_track_background_fraction"):
            raise ValueError(
                f"Component {component.config_name!r} sub-config does not "
                f"expose `point_track_background_fraction`."
            )
        exp_config.point_track_background_fraction = frac

    exp_config.output_dir = component_output_dir
    return exp_config


def main() -> int:
    print("[BOOT] entering mixture_main()", flush=True)
    args = get_args()

    print("[BOOT] auto-importing configs...", flush=True)
    auto_import_configs()
    print("[BOOT] auto-import done", flush=True)

    if args.list:
        names = list_mixtures()
        if not names:
            print("(no mixtures registered)")
        else:
            for name in names:
                spec = get_mixture(name)
                comps = ", ".join(
                    f"{c.config_name}={c.num_houses}" for c in spec.components
                )
                print(f"{name}: {comps}")
        return 0

    if not args.mixture_name:
        print("Error: mixture_name is required (or pass --list).", file=sys.stderr)
        return 2

    spec = get_mixture(args.mixture_name)
    spec = apply_overrides(
        spec,
        args.override,
        args.samples_override,
        args.points_override,
        args.include_background,
        args.bg_fraction,
    )

    # Self-heal any partially-extracted resource cache entries before workers
    # boot. Cheap (just Path.exists) and prevents silent worker death from
    # missing .obj/.png files left behind by a previously-killed extract.
    verify_and_repair_at_startup()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mixture_run_dir = MIXTURES_OUTPUT_ROOT / args.mixture_name / timestamp
    os.makedirs(mixture_run_dir, exist_ok=True)
    print(f"[BOOT] mixture run dir: {mixture_run_dir}", flush=True)

    spec_dump_path = mixture_run_dir / "mixture_spec.json"
    spec_dump_path.write_text(
        json.dumps(
            {
                "mixture_name": args.mixture_name,
                "max_house_index": spec.max_house_index,
                "components": [asdict(c) for c in spec.components],
            },
            indent=2,
        )
    )

    results: list[dict] = []
    grand_success = 0
    grand_total = 0
    for component in spec.components:
        component_dir = mixture_run_dir / component.config_name
        os.makedirs(component_dir, exist_ok=True)
        print(
            f"\n[MIXTURE] === Running component '{component.config_name}' "
            f"({component.num_houses} houses) ===",
            flush=True,
        )

        try:
            exp_config = _instantiate_component(
                component, spec.max_house_index, component_dir
            )
            exp_config.save_config()
            runner = ParallelRolloutRunner(exp_config)
            success_count, total_count = runner.run()
        except Exception as e:
            print(
                f"[MIXTURE] Component {component.config_name!r} crashed: {e}",
                flush=True,
            )
            traceback.print_exc()
            results.append(
                {
                    "config_name": component.config_name,
                    "num_houses": component.num_houses,
                    "success_count": 0,
                    "total_count": 0,
                    "error": repr(e),
                }
            )
            continue

        grand_success += success_count
        grand_total += total_count
        results.append(
            {
                "config_name": component.config_name,
                "num_houses": component.num_houses,
                "success_count": success_count,
                "total_count": total_count,
            }
        )
        print(
            f"[MIXTURE] Component {component.config_name!r}: "
            f"success={success_count}, total={total_count}",
            flush=True,
        )

    summary = {
        "mixture_name": args.mixture_name,
        "timestamp": timestamp,
        "grand_success_count": grand_success,
        "grand_total_count": grand_total,
        "components": results,
    }
    (mixture_run_dir / "mixture_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[MIXTURE] === Mixture complete ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
