"""Execute the exact UFL definitions in the transient spatial teaching notebook.

The notebook owns the source, forms and equations. This helper only exports
its tagged definitions into an importable spawn module and records diagnostics.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from threadpoolctl import threadpool_limits

from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.io.workspace import case_workspace, source_file, source_identity

ROOT = case_workspace()
NOTEBOOK = ROOT / "notebooks/transport/spatial_refinement.ipynb"


def exported_definitions() -> str:
    """Read the complete, explicitly tagged teaching definitions without rewriting them."""
    notebook = json.loads(NOTEBOOK.read_text())
    sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "pymhm-transient-spatial-definition" in cell.get("metadata", {}).get("tags", [])
    ]
    if len(sources) != 1:
        raise ValueError("The transient tutorial must declare exactly one definition cell")
    return sources[0]


def main() -> None:
    """Acquire fresh analytical-error rows from unchanged exported notebook equations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--final", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--classical", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("An existing numerical record cannot be overwritten")
    hashes = current_source_manifest(
        source_identity(
            ROOT,
            [
                *sorted((ROOT / "src/pymhm").rglob("*.py")),
                Path(__file__),
                NOTEBOOK,
                ROOT / "pixi.lock",
                ROOT / "pixi.toml",
                ROOT / "pyproject.toml",
            ],
        ),
        packages=("pymhm",),
    )
    record: dict[str, Any] = {
        "schema": "pymhm-native-transient-spatial-v1",
        "case": "c(t,x,y)=t*sin(pi*x)*sin(pi*y)",
        "operator": "capacity=1,diffusion=1,beta=(0.25,-0.5),reaction=0.5",
        "space": "native local P2/r2, scalar P1 traces; one retained mean",
        "time_integrator": "backward Euler",
        "dt": args.dt,
        "final_time": args.final,
        "source_sha256": hashes,
        "classical": args.classical,
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pymhm-transient-forms-") as temporary, threadpool_limits(1):
        module_path = Path(temporary) / "pymhm_transient_spatial_equations.py"
        module_path.write_text(exported_definitions())
        sys.path.insert(0, temporary)
        try:
            equations = importlib.import_module("pymhm_transient_spatial_equations")
            for n in args.levels:
                if args.classical:
                    row = equations.classical_row(n, args.dt, args.final)
                else:
                    row = equations.spatial_row(n, args.dt, args.final, workers=args.workers)
                record["rows"].append(row)
                args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
                if not row["accepted"]:
                    raise RuntimeError(f"Physical equations or error quadrature rejected n={n}")
        finally:
            sys.path.remove(temporary)
            sys.modules.pop("pymhm_transient_spatial_equations", None)
    if any(file_digest(source_file(name, root=ROOT)) != digest for name, digest in hashes.items()):
        raise RuntimeError("An executed numerical definition changed during acquisition")


if __name__ == "__main__":
    main()
