"""Execute scientific notebooks and retain outputs without modifying sources."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from notebook_data import (
    DEFAULT_MAX_BYTES,
    dependency_plan,
    required_archives,
    required_images,
    select_notebooks,
    selected_notebook_ids,
    validate_archives,
)


def output_notebook_path(root: Path, source: Path) -> Path:
    """Keep problem folders in outputs and separate external sources by path digest.

    External notebook paths are encoded by the SHA-256 of the absolute
    UTF-8 path, preventing equal basenames from overwriting each other's outputs.
    Source notebooks are never overwritten.
    """
    source = source.resolve()
    base = (root / "notebooks").resolve()
    output = root / "build" / "notebooks"
    if source.is_relative_to(base):
        return output / source.relative_to(base)
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    return output / "external" / digest / source.name


def main() -> None:
    """Run selected notebooks with a bounded timeout and reproducible paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", help="Notebook IDs, paths or problem folders (default: all)"
    )
    parser.add_argument("--timeout", type=int, default=600, help="Cell timeout in seconds")
    parser.add_argument(
        "--max-data-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Explicit size budget for selected computed field archives",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        notebooks = select_notebooks(root, args.paths, allow_external=True)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not notebooks:
        raise SystemExit("No notebooks found")
    if args.timeout < 1:
        raise SystemExit("Cell timeout must be positive")
    selected = selected_notebook_ids(root, notebooks)
    dependencies = required_archives(root, selected)
    try:
        validate_archives(
            root,
            dependency_plan(root, dependencies, required_images(root, selected)),
            args.max_data_bytes,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output = root / "build" / "notebooks"
    output.mkdir(parents=True, exist_ok=True)
    kernel_environment = {
        **os.environ,
        "MPLBACKEND": "module://matplotlib_inline.backend_inline",
    }
    for path in notebooks:
        print(f"Executing {path}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        )
        client.execute(env=kernel_environment)
        destination = output_notebook_path(root, path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination)
    print(f"Executed {len(notebooks)} notebooks; outputs: {output}")


if __name__ == "__main__":
    main()
