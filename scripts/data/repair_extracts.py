"""Repair partial resource-manager extracts (any data type/source).

The :class:`molmospaces_resources.ResourceManager` writes a per-archive
``.<package>_complete_extract`` flag into the cache once an archive has been
extracted, and on subsequent runs it short-circuits on flag presence
*without* re-validating that the files are still on disk. A previously
killed / interrupted extraction can therefore look "complete" to the
manager even when essential files are missing -- which surfaces in datagen
as::

    HouseInvalidForTask: Scene setup failed during compilation.
    Details: Setup error: Error: Error opening file '<cache>/.../*.obj':
    No such file or directory

This script generalises the old objaverse-only repair:
    * cross-checks every flagged archive against its ``CompactPathTrie``
      manifest of expected leaf paths;
    * for any archive that's missing files, removes the bogus flag and any
      stale partial files, and reinvokes ``manager.install_packages`` to
      properly re-fetch + re-extract.

Note: the data-generation entry points (``main.py`` and ``mixture_main.py``)
already invoke this same logic at startup via
``molmo_spaces.utils.cache_verifier.verify_and_repair_at_startup``. This
script is the manual / one-shot interface for that machinery (e.g. for
sanity-checking the cache outside of a job).

Usage::

    python -m scripts.data.repair_extracts                 # repair everything
    python -m scripts.data.repair_extracts --dry-run       # only report
    python -m scripts.data.repair_extracts --data-type objects --data-type scenes
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
        "--data-type",
        action="append",
        default=[],
        help="Restrict to this data type (e.g. 'objects', 'scenes', 'grasps'). "
        "Repeatable. Default: every data type known to the resource manager.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report broken archives; don't modify anything.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging for the resource manager too.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.verbose:
        resource_manager_log_level(logging.DEBUG)

    manager = get_resource_manager()
    data_types = args.data_type or None
    reports = verify_resource_cache(
        manager=manager,
        data_types=data_types,
        dry_run=args.dry_run,
    )

    total_broken = sum(len(r.broken_archives) for r in reports)
    total_repaired = sum(len(r.repaired_archives) for r in reports)
    total_still_broken = sum(len(r.still_broken_archives) for r in reports)
    errors = [r for r in reports if r.error]

    print(format_reports(reports))
    print(
        f"\nSummary: broken={total_broken} repaired={total_repaired} "
        f"still_broken={total_still_broken} errors={len(errors)}"
    )

    if args.dry_run:
        return 0
    if total_still_broken > 0 or errors:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
