"""Integrate physical HPC4e errors against independently assembled classical fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from hpc4e_data import DATA_DIRECTORY, load_data
from hpc4e_fields import RectangularElasticityField, compare_fields
from threadpoolctl import threadpool_limits

from pymhm.io.provenance import current_source_manifest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "build/results/hpc4e"


def compare(job: tuple[Path, Path, Path, int]) -> dict:
    """Measure a single archived field in physical L2 and compliance norms."""
    approximation_path, reference_path, data_directory, order = job
    material = load_data(data_directory)
    with threadpool_limits(limits=1):
        norms = compare_fields(
            RectangularElasticityField.load(approximation_path),
            RectangularElasticityField.load(reference_path),
            material,
            order,
        )
    return dict(approximation=approximation_path.stem, reference=reference_path.stem, norms=norms)


def main() -> None:
    """Compare archived classical/MHM fields at two or more exact quadratures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--reference", type=Path, default=ARCHIVES / "classical-rt2-512x256-mumps.npz"
    )
    parser.add_argument(
        "--approximations",
        type=Path,
        nargs="+",
        default=[
            ARCHIVES / "classical-rt1-512x256-pardiso.npz",
            *(ARCHIVES / f"mhm-s{s}.npz" for s in (1, 2, 4, 8)),
        ],
    )
    parser.add_argument("--orders", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "examples/results/hpc4e/reference-refinement.json"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if len(set(args.orders)) < 2 or any(order < 4 for order in args.orders):
        parser.error("at least two distinct quadrature orders >= 4 are required for RT2")
    sources = [
        Path(__file__),
        Path(__file__).with_name("hpc4e_fields.py"),
        Path(__file__).with_name("hpc4e_data.py"),
        Path(__file__).with_name("results") / "hpc4e/dataset.json",
        *(
            ROOT / f"src/pymhm/{name}.py"
            for name in (
                "fem/hdiv/tensor_rt",
                "_legacy/models/elasticity/stress_tensor",
                "_legacy/models/darcy/cartesian",
                "io/reservoir",
            )
        ),
    ]
    source_hashes = current_source_manifest(
        {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        }
    )
    input_paths = [*args.approximations, args.reference]
    if len({path.name for path in input_paths}) != len(input_paths):
        parser.error("comparison archives must have distinct filenames")
    inputs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    jobs = [
        (path, args.reference, args.data_dir, order)
        for path in args.approximations
        for order in args.orders
    ]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        rows = []
        for row in pool.map(compare, jobs):
            rows.append(row)
            print(json.dumps(row), flush=True)
    if inputs != {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}:
        raise RuntimeError("HPC4E comparison archives changed during integration")
    if source_hashes != current_source_manifest(
        {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        }
    ):
        raise RuntimeError("HPC4E comparison sources changed during integration")
    report = dict(
        comparison="Physical L2 and compliance norms on the common material-aligned grid",
        normalization="Each difference is divided by the RT2 reference norm",
        input_sha256=inputs,
        source_sha256=source_hashes,
        source_changed=False,
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".acquiring")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
