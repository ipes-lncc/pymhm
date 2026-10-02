"""Check that release metadata agrees with the canonical Pixi workspace."""

from __future__ import annotations

import argparse
import ast
import tomllib
from pathlib import Path


def main() -> None:
    """Validate shared metadata and, optionally, a requested release tag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Release tag; must equal v followed by the version")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pixi = tomllib.loads((root / "pixi.toml").read_text(encoding="utf-8"))
    for field in ("name", "version", "description"):
        if project[field] != pixi["workspace"][field]:
            raise SystemExit(f"PyPI and Pixi disagree on {field}")
    expected = [
        f"{name}{constraint}"
        for name, constraint in pixi["dependencies"].items()
        if name != "python"
    ]
    if sorted(project["dependencies"]) != sorted(expected):
        raise SystemExit("Runtime dependencies differ between Pixi and PyPI metadata")
    if project["requires-python"] != pixi["dependencies"]["python"]:
        raise SystemExit("Python constraints differ between Pixi and PyPI metadata")
    version = project["version"]
    module = ast.parse((root / "src/pymhm/__init__.py").read_text(encoding="utf-8"))
    versions = [
        ast.literal_eval(statement.value)
        for statement in module.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        )
    ]
    if versions != [version]:
        raise SystemExit("Package __version__ differs from release metadata")
    if args.tag is not None and args.tag != f"v{version}":
        raise SystemExit(f"Expected tag v{version}, received {args.tag!r}")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    if f'version: "{version}"' not in citation:
        raise SystemExit("CITATION.cff version differs from release metadata")
    recipe = (root / "recipe/recipe.yaml").read_text(encoding="utf-8")
    if f'  version: "{version}"' not in recipe:
        raise SystemExit("Conda recipe version differs from release metadata")
    print(f"Validated pymhm {version}: versions, runtime dependencies, Python constraint")


if __name__ == "__main__":
    main()
