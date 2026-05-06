#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

# Files to update. simplevecdb.__init__ derives __version__ dynamically via
# importlib.metadata, so no version literal lives there.
FILES = [
    Path("pyproject.toml"),
]


def get_current_version():
    """Read version from pyproject.toml"""
    content = FILES[0].read_text()
    match = re.search(r'version = "(\d+\.\d+\.\d+)"', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)


def bump_semver(current: str, part: str) -> str:
    """Bump major, minor, or patch version."""
    major, minor, patch = map(int, current.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    elif part == "minor":
        return f"{major}.{minor + 1}.0"
    elif part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    return part  # Assume it's a specific version string


def update_file(path: Path, old_ver: str, new_ver: str):
    """Update version in file content. Uses anchored regex to avoid replacing
    incidental occurrences of the version string elsewhere in the file."""
    content = path.read_text()

    if path.name == "pyproject.toml":
        pattern = re.compile(
            r'^(version\s*=\s*)"' + re.escape(old_ver) + r'"',
            flags=re.MULTILINE,
        )
        replacement = r'\g<1>"' + new_ver + r'"'
    else:
        return

    new_content, count = pattern.subn(replacement, content)
    if count == 0:
        print(f"Warning: Could not find anchored version {old_ver!r} in {path}")
        return

    path.write_text(new_content)
    print(f"Updated {path}")


def main():
    parser = argparse.ArgumentParser(description="Bump version of SimpleVecDB")
    parser.add_argument(
        "version", help="New version (x.y.z) or part to bump (major, minor, patch)"
    )
    args = parser.parse_args()

    try:
        current_ver = get_current_version()
        new_ver = bump_semver(current_ver, args.version)

        print(f"Bumping version: {current_ver} -> {new_ver}")

        for file_path in FILES:
            if file_path.exists():
                update_file(file_path, current_ver, new_ver)
            else:
                print(f"Warning: File not found: {file_path}")

        print("\nDone! Don't forget to:")
        print(f"  git add {' '.join(str(f) for f in FILES)}")
        print(f'  git commit -m "Bump version to {new_ver}"')
        print(f"  git tag v{new_ver}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
