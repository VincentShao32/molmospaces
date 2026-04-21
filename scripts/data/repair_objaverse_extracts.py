"""Compatibility shim -- delegates to the generic ``repair_extracts``.

The original objaverse-only repair has been generalised: the same kind of
partial-extract corruption can happen for *any* (data_type, source) pair,
not just objaverse objects. See :mod:`scripts.data.repair_extracts` and
:mod:`molmo_spaces.utils.cache_verifier` for details.

This module is kept so existing call sites / docs that reference
``python -m scripts.data.repair_objaverse_extracts`` keep working. It just
restricts the verifier to ``--data-type objects`` and forwards all other
flags through.

Usage examples (unchanged behaviour for objaverse cleanup)::

    python -m scripts.data.repair_objaverse_extracts             # repair objaverse + thor
    python -m scripts.data.repair_objaverse_extracts --dry-run   # only report
"""

from __future__ import annotations

import argparse
import logging
import sys

from molmo_spaces.molmo_spaces_constants import (
    get_resource_manager,
    resource_manager_log_level,
)
from molmo_spaces.utils.cache_verifier import (
    format_reports,
    verify_resource_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which archives look broken; don't modify anything.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging for the resource manager too.",
    )
    parser.add_argument(
        "--uid",
        action="append",
        default=[],
        help=argparse.SUPPRESS,  # legacy flag -- ignored; full scan is fast.
    )
    args = parser.parse_args()

    if args.uid:
        print(
            "[repair_objaverse_extracts] note: --uid is no longer needed; "
            "the verifier scans the entire objects cache automatically.",
            file=sys.stderr,
        )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.verbose:
        resource_manager_log_level(logging.DEBUG)

    manager = get_resource_manager()
    reports = verify_resource_cache(
        manager=manager,
        data_types=["objects"],
        dry_run=args.dry_run,
    )

    print(format_reports(reports))
    total_still_broken = sum(len(r.still_broken_archives) for r in reports)
    if args.dry_run:
        return 0
    return 2 if total_still_broken else 0


if __name__ == "__main__":
    sys.exit(main())
