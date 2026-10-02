"""Execute scientific notebooks and retain outputs without modifying sources."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from notebook_data import DEFAULT_MAX_BYTES, dependency_plan, required_archives, validate_archives


def main() -> None:
    """Run selected notebooks with a bounded timeout and reproducible paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Notebook paths (default: all)")
    parser.add_argument("--timeout", type=int, default=600, help="Cell timeout in seconds")
    parser.add_argument(
        "--max-data-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Explicit size budget for selected computed field archives",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebooks = args.paths or sorted((root / "notebooks").glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No notebooks found")
    if args.timeout < 1:
        raise SystemExit("Cell timeout must be positive")
    selected = {path.name.split("_", 1)[0] for path in notebooks}
    dependencies = {
        identifier: paths
        for identifier, paths in required_archives(root).items()
        if identifier in selected
    }
    try:
        validate_archives(root, dependency_plan(root, dependencies), args.max_data_bytes)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output = root / "build" / "notebooks"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    for path in notebooks:
        print(f"Executing {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        )
        client.execute()
        nbformat.write(notebook, output / path.name)
    print(f"Executed {len(notebooks)} notebooks; outputs: {output}")


if __name__ == "__main__":
    main()
