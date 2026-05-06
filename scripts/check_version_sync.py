#!/usr/bin/env python3
"""Pre-commit hook to verify version consistency across files.

simplevecdb.__init__ derives __version__ dynamically from installed package
metadata (``importlib.metadata.version("simplevecdb")``), so there is no
static version literal in __init__.py to compare against.

This script validates that:
1. ``pyproject.toml`` has a well-formed SemVer version.
2. The latest ``CHANGELOG.md`` entry header matches that version, so a
   release tag can never ship with a stale changelog.
"""

import re
import sys
from pathlib import Path


_PYPROJECT_VERSION_RE = re.compile(
    r'^version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE
)
_CHANGELOG_HEADING_RE = re.compile(
    r"^##\s+\[(?P<version>[^\]]+)\]", re.MULTILINE
)
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+([.-][a-zA-Z0-9]+)*$")


def extract_pyproject_version(pyproject_path: Path) -> str | None:
    match = _PYPROJECT_VERSION_RE.search(pyproject_path.read_text())
    return match.group(1) if match else None


def extract_latest_changelog_version(changelog_path: Path) -> str | None:
    """Return the first ``## [X.Y.Z]`` header in CHANGELOG.md, ignoring
    placeholder ``[Unreleased]`` headings."""
    for match in _CHANGELOG_HEADING_RE.finditer(changelog_path.read_text()):
        version = match.group("version").strip()
        if version.lower() == "unreleased":
            continue
        return version
    return None


def main() -> int:
    repo_root = Path(__file__).parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    changelog_path = repo_root / "CHANGELOG.md"

    if not pyproject_path.exists():
        print(f"❌ {pyproject_path} not found", file=sys.stderr)
        return 1

    pyproject_version = extract_pyproject_version(pyproject_path)
    if not pyproject_version:
        print("❌ Could not extract version from pyproject.toml", file=sys.stderr)
        return 1

    if not _SEMVER_RE.match(pyproject_version):
        print(
            f"❌ pyproject.toml version {pyproject_version!r} is not a valid SemVer",
            file=sys.stderr,
        )
        return 1

    if changelog_path.exists():
        changelog_version = extract_latest_changelog_version(changelog_path)
        if changelog_version is None:
            print(
                "❌ CHANGELOG.md has no released-version heading "
                "(expected '## [X.Y.Z] - YYYY-MM-DD').",
                file=sys.stderr,
            )
            return 1
        if changelog_version != pyproject_version:
            print(
                f"❌ Version mismatch: pyproject.toml={pyproject_version!r} "
                f"but latest CHANGELOG entry is {changelog_version!r}. "
                "Update CHANGELOG.md before tagging the release.",
                file=sys.stderr,
            )
            return 1
        print(
            f"✅ versions in sync: pyproject={pyproject_version}, "
            f"CHANGELOG={changelog_version}"
        )
    else:
        print(f"✅ pyproject version OK: {pyproject_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
