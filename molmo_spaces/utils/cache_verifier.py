"""Resource-manager cache verifier and self-healing repair.

Background
----------
``molmospaces_resources.ResourceManager`` writes a per-archive
``.<package>_complete_extract`` flag file into the cache once an archive has
been extracted, and on subsequent runs the manager *short-circuits on flag
presence without re-validating that the files are still on disk*.

That short-circuit means a partial extraction (e.g. job killed mid-extract,
filesystem hiccup, manual cleanup of a single file) can become "permanently
extracted" from the manager's point of view, even when essential files
(``*.obj``, ``*.png``, etc.) are missing. Workers then fail with::

    HouseInvalidForTask: Scene setup failed during compilation.
    Details: Setup error: Error: Error opening file
    '<cache>/objects/<source>/<version>/.../<file>.obj': No such file or directory

This module solves that by using the same per-archive ``CompactPathTrie`` the
manager itself loads from ``COMBINED_TRIES_NAME`` to enumerate every file an
archive *should* have extracted, then comparing against what's actually on
disk. Anything missing => clear the flag, remove stale partial files, and
re-invoke ``manager.install_packages`` to re-fetch + re-extract the affected
archives.

Robot assets under ``<MLSPACES_ASSETS_DIR>/robots/<name>/`` are typically
symlinks into ``generated_cache``. A partial robot extract leaves
``_*_complete_extract`` flags while some ``*.obj`` targets are missing, which
makes MuJoCo fail at compile time with "No such file or directory". The
startup hook therefore also walks those trees for **dangling symlinks** and
re-installs the affected robot archives (same spirit as archive repair).

Public entry points:

* :func:`verify_resource_cache` -- library function. Returns a per
  ``(data_type, source)`` summary.
* :func:`find_robots_with_dangling_symlinks` -- detect broken
  ``<ASSETS_DIR>/robots/<name>/`` symlink trees (partial robot extracts).
* :func:`repair_robot_installs` -- clear extract flags and re-install named robots.
* :func:`verify_and_repair_at_startup` -- convenience wrapper used by the
  data-generation entry points. Honours ``MLSPACES_SKIP_CACHE_VERIFY=1`` to
  short-circuit, and only complains -- never crashes -- if anything goes
  wrong with the verification itself.

The verifier is intentionally cheap (just ``Path.exists`` calls) so it's
safe to run unconditionally at every job start.
"""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from molmospaces_resources.manager import ResourceManager


logger = logging.getLogger("molmo_spaces.cache_verifier")


def _complete_extract_flag(package: str, cache_dest: Path) -> Path:
    """Mirror :func:`molmospaces_resources.file_utils._complete_extract_flag`."""
    return cache_dest / f".{package.replace('/', '__')}_complete_extract"


def _force_unlink(path: Path) -> None:
    """``unlink`` that survives read-only files / missing files."""
    try:
        if path.is_file() or path.is_symlink():
            try:
                path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
            path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.debug("Could not remove %s: %s", path, e)


@dataclass
class SourceVerifyReport:
    """Per-(data_type, source) outcome of a verification pass."""

    data_type: str
    source: str
    cache_dir: Path
    archives_checked: int = 0
    broken_archives: list[str] = field(default_factory=list)
    repaired_archives: list[str] = field(default_factory=list)
    still_broken_archives: list[str] = field(default_factory=list)
    error: str | None = None  # set if the source itself failed to load

    @property
    def has_action(self) -> bool:
        return bool(self.broken_archives or self.error)


def _iter_data_type_sources(
    manager: "ResourceManager",
    data_types: Iterable[str] | None,
) -> Iterable[tuple[str, str]]:
    versions = getattr(manager, "versions", {}) or {}
    selected = list(data_types) if data_types is not None else list(versions.keys())
    for data_type in selected:
        for source in versions.get(data_type, {}):
            yield data_type, source


def _find_broken_packages(
    cache_dest: Path, tries
) -> tuple[int, list[str]]:
    """Return ``(num_archives_with_flag, broken_archive_names)``.

    An archive is considered broken when its ``_complete_extract`` flag exists
    but at least one path in ``trie.leaf_paths()`` is missing on disk.
    """
    if not cache_dest.is_dir():
        return 0, []

    flag_suffix = "_complete_extract"
    flagged_packages: list[str] = []
    for entry in cache_dest.iterdir():
        name = entry.name
        if not name.startswith("."):
            continue
        if not name.endswith(flag_suffix):
            continue
        # ".<pkg>_complete_extract" -> "<pkg>"
        pkg = name[1 : -len(flag_suffix)]
        flagged_packages.append(pkg)

    broken: list[str] = []
    for pkg in flagged_packages:
        trie = tries.get(pkg) if tries is not None else None
        if trie is None:
            # No trie (unindexed archive); we can't verify -- trust the flag.
            continue
        try:
            leaves = trie.leaf_paths()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(
                "Could not read leaf_paths for %s: %s", pkg, e
            )
            continue
        for rel in leaves:
            if not (cache_dest / rel).exists():
                broken.append(pkg)
                break

    return len(flagged_packages), broken


def _clear_partial_extract(
    cache_dest: Path, package: str, leaf_paths: list[str]
) -> None:
    """Remove the bogus complete-flag and any partial leaf files for *package*."""
    flag = _complete_extract_flag(package, cache_dest)
    if flag.exists():
        try:
            flag.unlink()
        except OSError as e:
            logger.warning("Failed to remove flag %s: %s", flag, e)

    # Remove leaf files that *do* exist so re-extract gets a clean slate.
    # We deliberately don't recursively rmdir empty parent dirs -- they're
    # harmless and the next extract will re-use them.
    for rel in leaf_paths:
        _force_unlink(cache_dest / rel)


def verify_resource_cache(
    manager: "ResourceManager" | None = None,
    data_types: Iterable[str] | None = None,
    *,
    dry_run: bool = False,
    repair: bool = True,
) -> list[SourceVerifyReport]:
    """Walk every (data_type, source) and verify on-disk extracts.

    Args:
        manager: A ``ResourceManager`` instance. If ``None``, the module-level
            singleton from :func:`molmo_spaces.molmo_spaces_constants.get_resource_manager`
            is used.
        data_types: If given, restrict checks to these data types
            (e.g. ``["objects", "scenes"]``). Default is all known data types
            in ``manager.versions``.
        dry_run: When ``True``, only report; never modify disk.
        repair: When ``False``, broken archives are reported but not repaired
            (useful for callers that want to drive their own retry loop).

    Returns:
        One :class:`SourceVerifyReport` per ``(data_type, source)`` checked.
    """
    if manager is None:
        from molmo_spaces.molmo_spaces_constants import get_resource_manager

        manager = get_resource_manager()

    reports: list[SourceVerifyReport] = []
    for data_type, source in _iter_data_type_sources(manager, data_types):
        try:
            cache_dest = manager.cache_path(data_type, source)
        except Exception as e:  # pragma: no cover - defensive
            reports.append(
                SourceVerifyReport(
                    data_type=data_type,
                    source=source,
                    cache_dir=Path("?"),
                    error=f"cache_path failed: {e!r}",
                )
            )
            continue

        if not cache_dest.is_dir():
            # Source hasn't been touched on this machine yet -- nothing to verify.
            continue

        try:
            tries = manager.tries(data_type, source)
        except Exception as e:
            reports.append(
                SourceVerifyReport(
                    data_type=data_type,
                    source=source,
                    cache_dir=cache_dest,
                    error=f"tries() failed: {e!r}",
                )
            )
            continue

        archives_checked, broken = _find_broken_packages(cache_dest, tries)
        report = SourceVerifyReport(
            data_type=data_type,
            source=source,
            cache_dir=cache_dest,
            archives_checked=archives_checked,
            broken_archives=sorted(broken),
        )

        if not broken or dry_run or not repair:
            reports.append(report)
            continue

        # Perform the repair: clear flags + partial files, then re-install.
        for pkg in report.broken_archives:
            trie = tries.get(pkg)
            leaves = trie.leaf_paths() if trie is not None else []
            _clear_partial_extract(cache_dest, pkg, leaves)

        try:
            manager.install_packages(
                data_type, {source: report.broken_archives}, skip_linking=True
            )
        except Exception as e:
            report.error = f"install_packages failed: {e!r}"
            report.still_broken_archives = list(report.broken_archives)
            reports.append(report)
            continue

        # Re-validate post-repair.
        _, still_broken = _find_broken_packages(cache_dest, tries)
        repaired = [p for p in report.broken_archives if p not in still_broken]
        report.repaired_archives = sorted(repaired)
        report.still_broken_archives = sorted(
            p for p in report.broken_archives if p in set(still_broken)
        )
        reports.append(report)

    return reports


def format_reports(reports: list[SourceVerifyReport]) -> str:
    lines: list[str] = []
    for r in reports:
        if not r.has_action and not r.repaired_archives:
            continue
        lines.append(
            f"  [{r.data_type}/{r.source}] checked={r.archives_checked} "
            f"broken={len(r.broken_archives)} "
            f"repaired={len(r.repaired_archives)} "
            f"still_broken={len(r.still_broken_archives)}"
            + (f" error={r.error}" if r.error else "")
        )
        for pkg in r.broken_archives:
            tag = (
                "OK"
                if pkg in r.repaired_archives
                else ("FAIL" if pkg in r.still_broken_archives else "?")
            )
            lines.append(f"      [{tag}] {pkg}")
    if not lines:
        return "  (no broken extracts found)"
    return "\n".join(lines)


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


def find_robots_with_dangling_symlinks(robots_symlink_root: Path) -> list[str]:
    """Return sorted robot names under ``robots_symlink_root`` that have a broken symlink.

    A symlink is broken when ``Path.is_symlink()`` is true and ``Path.exists()``
    is false (missing target in the content cache).
    """
    if not robots_symlink_root.is_dir():
        return []
    broken: list[str] = []
    for robot_dir in sorted(robots_symlink_root.iterdir()):
        if not robot_dir.is_dir():
            continue
        has_dangling = False
        try:
            for p in robot_dir.rglob("*"):
                try:
                    if p.is_symlink() and not p.exists():
                        has_dangling = True
                        logger.debug("Dangling symlink: %s -> %s", p, p.readlink())
                        break
                except OSError as e:
                    logger.debug("Symlink check failed for %s: %s", p, e)
                    has_dangling = True
                    break
        except OSError as e:
            logger.warning("Could not walk robot dir %s: %s", robot_dir, e)
            continue
        if has_dangling:
            broken.append(robot_dir.name)
    return broken


def repair_robot_installs(manager: "ResourceManager", robot_names: list[str]) -> None:
    """Clear extract-complete flags and re-install listed robots (refreshes symlinks)."""
    for name in robot_names:
        try:
            cache_dest = manager.cache_path("robots", name)
        except Exception as e:
            logger.warning("cache_path(robots, %s) failed: %s", name, e)
            continue
        if cache_dest.is_dir():
            for flag in cache_dest.glob(".*_complete_extract"):
                _force_unlink(flag)
        try:
            pkgs = manager.find_all_packages_for_source("robots", name)
        except Exception as e:
            logger.warning("find_all_packages_for_source(robots, %s) failed: %s", name, e)
            continue
        if not pkgs:
            logger.warning(
                "No robot archive packages found for %r; remove %s manually or set "
                "MLSPACES_FORCE_INSTALL when fetching.",
                name,
                cache_dest,
            )
            continue
        try:
            manager.install_packages("robots", {name: pkgs}, skip_linking=False)
            logger.info("Re-installed robot %r (%d package(s))", name, len(pkgs))
        except Exception as e:
            logger.warning("install_packages failed for robot %r: %s", name, e)


def verify_and_repair_at_startup(
    data_types: Iterable[str] | None = ("objects", "scenes", "grasps", "robots"),
) -> list[SourceVerifyReport]:
    """Best-effort startup hook used by the data-generation entry points.

    Behaviour:

    * If ``MLSPACES_SKIP_CACHE_VERIFY=1``, returns immediately (no scan).
    * Otherwise scans the requested data types using
      :func:`verify_resource_cache` in dry-run mode (cheap; just ``Path.exists``
      lookups against each archive's manifest trie).
    * Independently scans ``<ASSETS_DIR>/robots/*/`` for **dangling symlinks**
      (partial robot extracts). Those are repaired even when archive tries are
      absent or report a clean bill of health.
    * If 0 broken archives **and** 0 broken robot symlinks -> prints healthy and returns.
    * If >=1 broken archives -> **always** auto-repairs those in place.
    * If >=1 broken robot symlink trees -> clears the robot's extract flags and
      re-runs ``install_packages`` for that robot (may re-download).

    Never raises: any verification failure is logged but the pipeline still
    proceeds (worst case: the job hits the same per-house failure it would
    have hit without the hook).
    """
    if _truthy_env("MLSPACES_SKIP_CACHE_VERIFY"):
        print("[CACHE-VERIFY] skipped (MLSPACES_SKIP_CACHE_VERIFY set)", flush=True)
        return []

    print(
        f"[CACHE-VERIFY] scanning data_types={list(data_types) if data_types else 'ALL'}",
        flush=True,
    )

    # Phase 1: dry-run scan -- always cheap.
    try:
        scan_reports = verify_resource_cache(
            data_types=data_types, dry_run=True
        )
    except Exception as e:  # pragma: no cover - defensive
        print(f"[CACHE-VERIFY] WARNING: scan crashed: {e!r}", flush=True)
        return []

    archive_broken = sum(len(r.broken_archives) for r in scan_reports)

    robot_broken: list[str] = []
    try:
        from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

        robot_broken = find_robots_with_dangling_symlinks(ASSETS_DIR / "robots")
    except Exception as e:  # pragma: no cover - defensive
        print(f"[CACHE-VERIFY] WARNING: robot symlink scan crashed: {e!r}", flush=True)

    if archive_broken == 0 and not robot_broken:
        print(
            "[CACHE-VERIFY] all extracts and robot symlinks look healthy.",
            flush=True,
        )
        return scan_reports

    reports: list[SourceVerifyReport] = list(scan_reports)

    if archive_broken:
        print(
            f"[CACHE-VERIFY] found {archive_broken} broken archive(s); "
            "auto-repairing (this may take a while)...",
            flush=True,
        )
        try:
            reports = verify_resource_cache(data_types=data_types, dry_run=False)
        except Exception as e:  # pragma: no cover - defensive
            print(f"[CACHE-VERIFY] WARNING: archive repair crashed: {e!r}", flush=True)

        total_repaired = sum(len(r.repaired_archives) for r in reports)
        total_still_broken = sum(len(r.still_broken_archives) for r in reports)
        print(
            f"[CACHE-VERIFY] archives: repaired {total_repaired}, "
            f"still broken {total_still_broken}.",
            flush=True,
        )
        print(format_reports(reports), flush=True)

    if robot_broken:
        print(
            f"[CACHE-VERIFY] dangling symlinks under robots/: {robot_broken}; "
            "re-installing those robot package(s)...",
            flush=True,
        )
        try:
            from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_resource_manager

            mgr = get_resource_manager()
            repair_robot_installs(mgr, robot_broken)
            still = find_robots_with_dangling_symlinks(ASSETS_DIR / "robots")
            if still:
                print(
                    f"[CACHE-VERIFY] WARNING: robots still have dangling symlinks "
                    f"after repair: {still}",
                    flush=True,
                )
            else:
                print("[CACHE-VERIFY] robot symlink repair complete.", flush=True)
        except Exception as e:  # pragma: no cover - defensive
            print(f"[CACHE-VERIFY] WARNING: robot repair crashed: {e!r}", flush=True)

    return reports
