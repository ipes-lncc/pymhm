"""Acquire bounded initial wave studies with explicit physical case provenance.

Each case reports three declared levels, original-equation checks and quadrature
controls. Initial refinement studies do not certify an asymptotic regime or a
historical paper reproduction. Reference increments are distinct from exact errors.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from pymhm.io.provenance import current_source_manifest

ROOT = Path(__file__).resolve().parents[1]
CASES = ("helmholtz", "elastic-wave", "three-layer", "marmousi")


def digest(path: Path) -> str:
    """Hash the actual source or artifact bytes with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes(extra: tuple[Path, ...] = ()) -> dict[str, str]:
    """Identify all core, lock and already loaded example/private numerical owners."""
    import sys

    files = set((ROOT / "src/pymhm").rglob("*.py"))
    files.update((ROOT / name) for name in ("pixi.lock", "pixi.toml", "pyproject.toml"))
    files.update(extra)
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if (
                path.is_file()
                and path.suffix == ".py"
                and path.is_relative_to(ROOT)
                and not path.is_relative_to(ROOT / ".pixi")
            ):
                files.add(path)
    return current_source_manifest(
        {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(files)}
    )


def require_original(values: dict[str, float], threshold: float = 1e-10) -> None:
    """Require finite original-equation diagnostics under the declared unchanged threshold."""
    if any(not np.isfinite(value) or value > threshold for value in values.values()):
        raise ArithmeticError(f"Original equation check failed at {threshold}: {values}")


def quadrature_change(low: dict[str, float], high: dict[str, float]) -> float:
    """Return the largest relative quadrature change across separately reported fields."""
    if low.keys() != high.keys() or not low:
        raise ValueError("Quadrature controls require the same nonempty norm fields")
    if any(not np.isfinite(value) or value < 0 for value in (*low.values(), *high.values())):
        raise ValueError("Quadrature norms must be finite and nonnegative")
    return max(
        abs(low[key] - high[key]) / max(high[key], float(np.finfo(float).tiny)) for key in low
    )


def write(path: Path, record: dict[str, Any]) -> None:
    """Expose a completed JSON record atomically, preserving literal newlines."""
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plot(record: dict[str, Any], path: Path) -> None:
    """Render separately named field curves for exact errors or qualified increments."""
    matplotlib = importlib.import_module("matplotlib")

    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    rows = record["rows"]
    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    plotted = False
    for key in record["plot_fields"]:
        x = [row["level"] for row in rows if key in row.get("norms", {})]
        y = [row["norms"][key] for row in rows if key in row.get("norms", {})]
        if x and np.all(np.asarray(y) > 0):
            axis.loglog(x, y, "o-", label=key.replace("_", " "))
            plotted = True
    if not plotted:
        raise ValueError("No positive measured field norms available for the initial figure")
    axis.set(xlabel=record["level_label"], ylabel=record["norm_label"])
    axis.set_title(record["figure_title"])
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=160)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def run(case: str, output: Path, *, level: int | None = None) -> dict[str, Any]:
    """Run one source-fenced initial case and retain every completed numerical level."""
    if case not in CASES or output.exists():
        raise ValueError("Select a known case and a fresh output directory")
    module = importlib.import_module("examples.minimal_wave_" + case.replace("-", "_"))
    module_source = module.__file__
    if module_source is None:
        raise RuntimeError("A concrete public example source owner is required")
    before = source_hashes((Path(__file__), Path(module_source)))
    output.mkdir(parents=True)
    generation_id = str(uuid.uuid4())
    snapshot = output / "executed-sources"
    snapshot.mkdir()
    for name, expected in before.items():
        path = ROOT / name
        if digest(path) != expected:
            raise RuntimeError("Source changed before the original acquisition")
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write(snapshot / "manifest.json", {"source_sha256": before, "generation_id": generation_id})
    started = perf_counter()
    with threadpool_limits(1):
        record = (
            module.acquire(output) if level is None else module.acquire(output, levels=(level,))
        )
        pools = threadpool_info()
        if any(pool["num_threads"] != 1 for pool in pools):
            raise RuntimeError("The initial numerical study requires one native thread")
    after = source_hashes(tuple(ROOT / name for name in before))
    changed = [name for name, value in before.items() if after.get(name) != value]
    if changed:
        raise RuntimeError("Executed numerical sources changed: " + ", ".join(changed))
    record.update(
        schema="pymhm-initial-wave-convergence-v1",
        case=case,
        generation_id=generation_id,
        source_sha256=before,
        sources_changed_during_run=changed,
        elapsed_seconds=perf_counter() - started,
        native_threads=1,
        workers=1,
        native_runtime=[pool | {"sha256": digest(Path(pool["filepath"]))} for pool in pools],
        original_equation_relative_threshold=1e-10,
        historical_literal_reproduction=False,
    )
    write(output / "study.json", record)
    print(
        json.dumps(
            {"case": case, "levels": len(record["rows"]), "seconds": record["elapsed_seconds"]}
        ),
        flush=True,
    )
    return record


def assemble(case: str, output: Path, directories: tuple[Path, ...]) -> dict[str, Any]:
    """Combine accepted immutable levels and reintegrate fields without a new PDE solve."""
    if output.exists() or case not in CASES or not directories:
        raise ValueError("A known case, input levels and fresh assembled output are required")
    module = importlib.import_module("examples.minimal_wave_" + case.replace("-", "_"))
    module_source = module.__file__
    if module_source is None:
        raise RuntimeError("A concrete public example source owner is required")
    before = source_hashes((Path(__file__), Path(module_source)))
    output.mkdir(parents=True)
    with threadpool_limits(1):
        record = module.combine(output, directories)
    after = source_hashes(tuple(ROOT / name for name in before))
    if any(after.get(name) != value for name, value in before.items()):
        raise RuntimeError("A data-only numerical consumer changed during integration")
    record.update(
        schema="pymhm-initial-wave-convergence-v1",
        case=case,
        generation_id=str(uuid.uuid4()),
        consumer_source_sha256=before,
        pde_solves_during_assembly=0,
        original_equation_relative_threshold=1e-10,
        native_threads=1,
        workers=1,
        historical_literal_reproduction=False,
    )
    write(output / "study.json", record)
    return record


def main() -> None:
    """Acquire one explicit case under an external process/RSS/deadline controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--level", type=int)
    parser.add_argument("--assemble-from", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.plot_only:
        record = json.loads((args.output / "study.json").read_text())
        plot(record, args.output / "convergence.png")
        write(
            args.output / "figures.json",
            {
                "study_sha256": digest(args.output / "study.json"),
                "plot_consumer_sha256": digest(Path(__file__)),
                "figure_sha256": {
                    name: digest(args.output / name)
                    for name in ("convergence.png", "convergence.svg")
                },
                "pde_solves": 0,
            },
        )
    elif args.case and args.assemble_from:
        assemble(args.case, args.output, tuple(args.assemble_from))
    elif args.case:
        run(args.case, args.output, level=args.level)
    else:
        parser.error("--case is required for an acquisition")


if __name__ == "__main__":
    main()
