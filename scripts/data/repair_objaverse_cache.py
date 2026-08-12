"""Repair objaverse cache by removing stale ``_complete_extract`` flags.

The molmospaces_resources extraction pipeline marks each archive complete by
touching ``.<archive>_complete_extract``. On later runs, ``install_packages``
short-circuits on that flag and never re-downloads. If an earlier extract got
killed after chmod-ing files read-only, subsequent extracts silently swallow
``PermissionError`` in ``_safe_extract`` and still touch the flag, leaving a
populated flag over an empty (or partial) uid dir. From that point on the
resource manager believes the archive is cached even though its files are
missing.

This script scans the cache, identifies uid dirs that are empty (or missing),
removes the corresponding stale complete-extract flag, and optionally removes
the empty dir. After running it, re-run the usual bulk download to refill the
missing archives.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def repair(
    data_type: str,
    source: str,
    *,
    dry_run: bool = False,
    delete_empty_dirs: bool = True,
    remove_partial: bool = False,
) -> None:
    from molmo_spaces.molmo_spaces_constants import get_resource_manager

    mgr = get_resource_manager()
    cache_dest = mgr.cache_path(data_type, source)
    print(f"cache_dest: {cache_dest}")

    if not cache_dest.exists():
        print("cache_dest does not exist; nothing to repair")
        return

    tries = mgr.tries(data_type, source)
    print(f"archives in trie: {len(tries)}")

    cleaned_flags = 0
    removed_dirs = 0
    fully_ok = 0
    partial = 0
    missing_dir = 0

    for pkg, trie in tries.items():
        # The only uid for objaverse is the top-level folder; for other
        # sources each archive may include multiple files under a shared root.
        leaves = trie.leaf_paths()
        if not leaves:
            continue
        # Derive the single top-level dir each archive writes into by the
        # longest common prefix of leaves.
        first = Path(leaves[0])
        top = first.parts[0] if first.parts else ""
        uid_dir = cache_dest / top if top else None

        have_any = False
        all_present = True
        if uid_dir is not None and uid_dir.exists():
            for p in leaves:
                exists = (cache_dest / p).exists()
                if exists:
                    have_any = True
                else:
                    all_present = False
        else:
            all_present = False

        flag = cache_dest / f".{pkg.replace('/', '__')}_complete_extract"

        if all_present:
            fully_ok += 1
            continue

        # Not fully present -> flag should not exist. Drop it so the next
        # install will re-fetch this archive.
        if flag.exists():
            if dry_run:
                pass
            else:
                flag.unlink()
            cleaned_flags += 1

        if not have_any:
            if uid_dir is not None and uid_dir.exists() and delete_empty_dirs:
                if dry_run:
                    pass
                else:
                    try:
                        uid_dir.rmdir()
                    except OSError:
                        # Dir had hidden files etc.; leave it.
                        pass
                removed_dirs += 1
            else:
                missing_dir += 1
        else:
            partial += 1
            if remove_partial and uid_dir is not None and uid_dir.exists():
                if not dry_run:
                    for child in uid_dir.iterdir():
                        try:
                            child.unlink()
                        except OSError:
                            pass

    print("")
    print(f"fully extracted (unchanged): {fully_ok}")
    print(f"stale flags removed:        {cleaned_flags}")
    print(f"empty dirs removed:         {removed_dirs}")
    print(f"missing uid dirs:           {missing_dir}")
    print(f"partial uid dirs (kept):    {partial}")
    if dry_run:
        print("(dry run -- no filesystem changes were made)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-type", default="objects")
    ap.add_argument("--source", default="objaverse")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--keep-empty-dirs",
        action="store_true",
        help="Leave empty uid dirs in place (default is to remove them).",
    )
    ap.add_argument(
        "--remove-partial",
        action="store_true",
        help="Also wipe uid dirs that have *some* files (safer to re-extract).",
    )
    args = ap.parse_args()

    repair(
        args.data_type,
        args.source,
        dry_run=args.dry_run,
        delete_empty_dirs=not args.keep_empty_dirs,
        remove_partial=args.remove_partial,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
