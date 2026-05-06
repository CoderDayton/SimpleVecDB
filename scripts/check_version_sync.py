#!/usr/bin/env python3
"""Pre-commit hook to verify pyproject.toml has a parseable version.

simplevecdb.__init__ derives __version__ dynamically from installed package
metadata (`importlib.metadata.version("simplevecdb")`), so there is no static
version literal in __init__.py to compare against. This script therefore
only validates that pyproject.toml has a well-formed version field.
"""

import re
import sys
from pathlib import Path


def extract_pyproject_version(pyproject_path: Path) -> str | None:
    """Extract version from pyproject.toml."""
    content = pyproject_path.read_text()
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return match.group(1) if match else None


def main() -> int:
    """Validate pyproject.toml version field."""
    repo_root = Path(__file__).parent.parent
    pyproject_path = repo_root / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"❌ {pyproject_path} not found", file=sys.stderr)
        return 1

    pyproject_version = extract_pyproject_version(pyproject_path)
    if not pyproject_version:
        print("❌ Could not extract version from pyproject.toml", file=sys.stderr)
        return 1

    if not re.match(r"^\d+\.\d+\.\d+([.-][a-zA-Z0-9]+)*$", pyproject_version):
        print(
            f"❌ pyproject.toml version {pyproject_version!r} is not a valid SemVer",
            file=sys.stderr,
        )
        return 1

    print(f"✅ pyproject version OK: {pyproject_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
