"""Check that release metadata agrees with the canonical Pixi workspace."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

# Only current-release fields belong here. Historical acquisition records,
# earlier changelog sections and synthetic distribution fixtures keep their versions.
VERSION_FIELDS: dict[str, tuple[str, ...]] = {
    "pyproject.toml": (r'(?m)^version = "(?P<version>[^"\r\n]+)"\r?$',),
    "pixi.toml": (r'(?m)^version = "(?P<version>[^"\r\n]+)"\r?$',),
    "src/pymhm/__init__.py": (r'(?m)^__version__ = "(?P<version>[^"\r\n]+)"\r?$',),
    "CITATION.cff": (r'(?m)^version: "(?P<version>[^"\r\n]+)"\r?$',),
    "recipe/recipe.yaml": (r'(?m)^  version: "(?P<version>[^"\r\n]+)"\r?$',),
    "README.md": (
        r"\[!\[Version: (?P<version>[^\]\r\n]+)\]\(",
        r"https://img\.shields\.io/badge/version-(?P<version>[^/\r\n]+)-21918c\.svg",
        r"(?m)^Version (?P<version>\S+) is research software",
    ),
    "docs/index.md": (r"(?m)^This is version (?P<version>[^,\s]+),",),
}


def version_files(root: Path, version: str, replacement: str | None = None) -> dict[Path, bytes]:
    """Validate all declared current-version fields and optionally prepare their updates.

    The returned bytes preserve unrelated text and original line endings. Each
    declared field must occur exactly once and already contain ``version``;
    inconsistent metadata is rejected before release preparation writes anything.
    """
    updates = {}
    for relative, patterns in VERSION_FIELDS.items():
        path = root / relative
        text = path.read_bytes().decode("utf-8")
        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if len(matches) != 1:
                raise ValueError(f"{relative}: expected one current-version field")
            match = matches[0]
            if match["version"] != version:
                raise ValueError(
                    f"{relative}: version {match['version']!r} differs from {version!r}"
                )
            if replacement is not None:
                text = text[: match.start("version")] + replacement + text[match.end("version") :]
        if replacement is not None:
            updates[path] = text.encode("utf-8")
    return updates


def validate_metadata(root: Path, tag: str | None = None) -> str:
    """Check versions, dependencies and Python bounds, returning the package version."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pixi = tomllib.loads((root / "pixi.toml").read_text(encoding="utf-8"))
    for field in ("name", "version", "description"):
        if project[field] != pixi["workspace"][field]:
            raise ValueError(f"PyPI and Pixi disagree on {field}")
    expected = [
        f"{name}{constraint}"
        for name, constraint in pixi["dependencies"].items()
        if name != "python"
    ]
    if sorted(project["dependencies"]) != sorted(expected):
        raise ValueError("Runtime dependencies differ between Pixi and PyPI metadata")
    if project["requires-python"] != pixi["dependencies"]["python"]:
        raise ValueError("Python constraints differ between Pixi and PyPI metadata")
    version = str(project["version"])
    version_files(root, version)
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Expected tag v{version}, received {tag!r}")
    return version


def main() -> None:
    """Validate shared metadata and, optionally, a requested release tag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Release tag; must equal v followed by the version")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        version = validate_metadata(root, args.tag)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"Validated pymhm {version}: versions, runtime dependencies, Python constraint")


if __name__ == "__main__":
    main()
