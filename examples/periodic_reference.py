"""Acquire current-source conforming Qk periodic references; Q5 is the default."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from pymhm.io.provenance import (
    current_source_manifest,
    optional_file_digest,
    workspace_git_dirty,
    workspace_revision,
)
from pymhm.io.workspace import source_file, source_identity

if __package__:
    from ._entrypoint import prepare_example_imports
else:
    from _entrypoint import prepare_example_imports

    prepare_example_imports(__file__, __package__)

if __package__:
    from .verify_periodic import (
        ARTIFACTS,
        ROOT,
        case_conventions,
        factor_x,
        factor_y,
        fingerprint,
        validate_case_provenance,
    )
else:
    from examples.verify_periodic import (
        ARTIFACTS,
        ROOT,
        case_conventions,
        factor_x,
        factor_y,
        fingerprint,
        validate_case_provenance,
    )

from pymhm.core.validation import positive_int
from pymhm.fem.scalar.quadrilateral import qk_basis, qk_space
from pymhm.fem.scalar.separable import interval_nodal_quadrature as _line_data
from pymhm.linalg.linear import SolverUnavailableError
from pymhm.linalg.separable import solve_separable_krylov
from pymhm.materials.separable import SeparableField
from pymhm.meshes.cartesian import CartesianMacroMesh


def sources() -> dict[str, str]:
    """Fingerprint the executed acquisition and all numerical dependency owners."""
    paths = [Path(__file__), source_file("examples/verify_periodic.py", root=ROOT)]
    paths += [
        source_file(f"src/pymhm/{name}.py", root=ROOT)
        for name in (
            "linalg/separable",
            "_legacy/models/darcy/separable",
            "_legacy/models/darcy/conforming",
            "_legacy/models/darcy/cartesian",
            "linalg/linear",
            "meshes/triangle",
            "fem/scalar/operators",
            "fem/quadrature/material",
            "core/contracts",
            "fem/scalar/triangle",
        )
    ]
    return current_source_manifest(
        source_identity(
            ROOT,
            (
                path
                for path in paths
                if path.name not in ("pixi.lock", "pixi.toml", "pyproject.toml") or path.is_file()
            ),
        ),
        packages=("pymhm", "examples"),
    )


def _array_digest(values: np.ndarray) -> str:
    """Bind an executed real array's dtype, dimensions and ordered coefficients."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _executed_basis(degree: int, order: int) -> dict[str, np.ndarray]:
    """Retain the owner's actual tensor-factor tabulation and ordered nodal basis.

    The separated operator uses these one-dimensional values and reference
    derivatives in each coordinate. The nodal cardinal matrix fixes the Qk
    coordinate convention, with the x index fastest on the unit square.
    """
    physical, weights, values, derivative, _, _ = _line_data(1, (0.0, 1.0), degree, order)
    _, nodes = qk_space(CartesianMacroMesh(1), degree)
    cardinals = qk_basis(degree, nodes)[0]
    return {
        "basis_nodes": nodes,
        "basis_cardinal_matrix": cardinals,
        "basis_quadrature_points": physical[0],
        "basis_quadrature_weights": weights,
        "basis_values_1d": values,
        "basis_reference_derivatives_1d": derivative,
    }


def _check_field(pressure: np.ndarray, residual: float, n: int, degree: int) -> None:
    """Require finite nodal coefficients, the unchanged true criterion and zero BC."""
    width = n * degree + 1
    if (
        pressure.shape != (width**2,)
        or np.iscomplexobj(pressure)
        or not np.isfinite(pressure).all()
        or not np.isfinite(residual)
        or not 0 <= residual <= 1e-10
    ):
        raise ValueError("reference arrays violate the original-equation acquisition contract")
    grid = pressure.reshape(width, width)
    if np.any(grid[[0, -1]]) or np.any(grid[:, [0, -1]]):
        raise ValueError("reference pressure violates strong homogeneous exterior Dirichlet data")


def _check_archive(
    path: Path, record: dict, basis: dict[str, np.ndarray], n: int, degree: int
) -> None:
    """Verify canonical basis coordinates and effective native coefficient precision."""
    precision = record.get("refinement_precision")
    if precision not in {"double", "extended"}:
        raise ValueError("reference acquisition requires explicit double or extended precision")
    dtype = np.dtype(np.longdouble if precision == "extended" else float)
    with np.load(path, allow_pickle=False) as arrays:
        _check_field(arrays["pressure"], float(arrays["residual"]), n, degree)
        if (
            record.get("pressure_dtype") != arrays["pressure"].dtype.str
            or arrays["pressure"].dtype != dtype
            or record.get("coefficient_precision_bits")
            != np.finfo(arrays["pressure"].dtype).nmant + 1
            or record.get("coefficient_precision_bits") != np.finfo(dtype).nmant + 1
            or record.get("relative_equation_residual") != float(arrays["residual"])
            or record.get("basis_sha256")
            != {name: _array_digest(values) for name, values in basis.items()}
            or any(
                name not in arrays
                or arrays[name].dtype != values.dtype
                or not np.array_equal(arrays[name], values)
                for name, values in basis.items()
            )
        ):
            raise ValueError(
                "reference arrays, basis or precision differ from their acquisition contract"
            )


def validate_reference_archive(path: Path, record: dict) -> None:
    """Validate a manifested Qk coefficient vector before physical evaluation.

    Archived ordered nodal coordinates and cardinal/assembly matrices must
    match the shared Qk owner used by ``ConformingQuadrilateralSolution``.
    The evaluator's numerical owners and the physical case are source guarded.
    Native NPZ coefficients require the declared effective precision;
    they are not a portable high/remainder representation. Legacy manifests
    without this basis contract are handled separately by their consumers.
    """
    n = positive_int(record.get("n"), "n")
    degree = positive_int(record.get("degree"), "degree")
    order = positive_int(record.get("quadrature_order"), "order", minimum=degree + 1)
    validate_case_provenance(record)
    owner_hashes = record.get("source_sha256", {})
    owners = (
        "_legacy/models/darcy/cartesian",
        "_legacy/models/darcy/conforming",
        "meshes/triangle",
        "fem/scalar/operators",
        "_legacy/models/darcy/separable",
    )
    if (
        any(
            owner_hashes.get(f"src/pymhm/{owner}.py")
            != fingerprint(source_file(f"src/pymhm/{owner}.py", root=ROOT))
            for owner in owners
        )
        or record.get("operator_dtype") != np.dtype(float).str
        or record.get("operator_precision_bits") != np.finfo(float).nmant + 1
        or record.get("coefficient_storage") != "native-real-npz"
        or record.get("schema") != "pymhm-conforming-periodic-reference-v1"
        or record.get("assembly") != "lor"
        or record.get("solver") != "low-order-refined-pyamg-cg"
        or record.get("refinement_precision") not in {"double", "extended"}
        or (
            record.get("refinement_precision") == "extended"
            and np.finfo(np.longdouble).eps >= np.finfo(float).eps
        )
        or record.get("original_equation_relative_criterion") != 1e-10
        or record.get("dofs") != (n * degree + 1) ** 2
        or record.get("basis_convention")
        != f"Q{degree} equidistant nodal cardinal functions; x index fastest"
        or record.get("boundary_convention")
        != "strong homogeneous exterior Dirichlet pressure at nodal dofs"
        or record.get("archive_sha256") != fingerprint(path)
    ):
        raise ValueError("reference evaluation owners, basis or precision contract mismatch")
    _check_archive(path, record, _executed_basis(degree, order), n, degree)


def run(
    n: int,
    order: int,
    native_threads: int = 1,
    *,
    degree: int = 5,
    refinement_precision: Literal["double", "extended"] = "extended",
    artifacts: Path | None = None,
    records: Path | None = None,
) -> None:
    """Acquire a unit-square Qk reference with strong zero exterior pressure.

    ``degree`` defaults to Q5. At least ``degree + 1`` Gauss points per axis are
    required so the recorded quadrature is the one executed by the shared
    operator. Float64 operators retain their 53-bit significands; extended
    coefficient accumulation and its effective native precision are explicit.
    ``refinement_precision='double'`` supports hosts without a wider long double;
    an explicit extended request is rejected on those hosts. Both modes retain
    the same original-equation residual criterion.
    This acquisition does not certify continuum accuracy or reference resolution.
    """
    n = positive_int(n, "n")
    degree = positive_int(degree, "degree")
    order = positive_int(order, "order", minimum=degree + 1)
    native_threads = positive_int(native_threads, "native_threads")
    if refinement_precision not in {"double", "extended"}:
        raise ValueError("refinement_precision must be double or extended")
    if refinement_precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        raise SolverUnavailableError("extended refinement requires a wider long-double type")
    original = sources()
    lockfile_digest = optional_file_digest(ROOT / "pixi.lock")
    directory = ARTIFACTS if artifacts is None else Path(artifacts)
    record_directory = ROOT / "examples/results" if records is None else Path(records)
    path = directory / f"reference-q{degree}-{n}-current-order{order}-lor.npz"
    record_path = record_directory / f"periodic-reference-q{degree}-{n}-order{order}.json"
    basis = _executed_basis(degree, order)
    if path.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        validate_case_provenance(record)
        if (
            record.get("source_sha256") != original
            or record.get("lockfile_sha256") != lockfile_digest
            or record.get("archive_sha256") != fingerprint(path)
            or (record.get("n"), record.get("degree"), record.get("quadrature_order"))
            != (n, degree, order)
            or record.get("assembly") != "lor"
            or record.get("operator_dtype") != np.dtype(float).str
            or record.get("operator_precision_bits") != np.finfo(float).nmant + 1
            or record.get("coefficient_storage") != "native-real-npz"
            or record.get("refinement_precision") != refinement_precision
        ):
            raise ValueError("existing reference differs from its acquisition contract")
        validate_reference_archive(path, record)
        return
    if path.exists() or record_path.exists():
        raise ValueError("reference acquisition requires a matching archive and manifest")
    start = time.perf_counter()
    with threadpool_limits(native_threads):
        solution = solve_separable_krylov(
            CartesianMacroMesh(n),
            degree=degree,
            permeability=SeparableField(((1.0, 1.0), (factor_x, factor_y))),
            source=SeparableField(((np.sin, np.sin),)),
            quadrature_order=order,
            refinement_precision=refinement_precision,
        )
        native_libraries = [
            {key: value for key, value in info.items() if key != "filepath"}
            for info in threadpool_info()
        ]
    if sources() != original or optional_file_digest(ROOT / "pixi.lock") != lockfile_digest:
        raise RuntimeError("reference sources or lockfile changed during acquisition")
    _check_field(solution.pressure, solution.relative_equation_residual, n, degree)
    directory.mkdir(parents=True, exist_ok=True)
    record_directory.mkdir(parents=True, exist_ok=True)
    temporary_archive = path.with_suffix(".npz.part")
    with temporary_archive.open("wb") as stream:
        np.savez_compressed(
            stream,
            pressure=solution.pressure,
            residual=solution.relative_equation_residual,
            **basis,
        )
    revision = workspace_revision(ROOT)
    dirty = workspace_git_dirty(ROOT)
    record = {
        "schema": "pymhm-conforming-periodic-reference-v1",
        "n": n,
        "degree": degree,
        "quadrature_order": order,
        "dofs": len(solution.pressure),
        "residual": solution.residual,
        "relative_equation_residual": solution.relative_equation_residual,
        "iterations": solution.iterations,
        "preconditioner_levels": solution.preconditioner_levels,
        "solver": "low-order-refined-pyamg-cg",
        "assembly": "lor",
        "basis_convention": f"Q{degree} equidistant nodal cardinal functions; x index fastest",
        "basis_sha256": {name: _array_digest(values) for name, values in basis.items()},
        "boundary_convention": "strong homogeneous exterior Dirichlet pressure at nodal dofs",
        "pressure_dtype": solution.pressure.dtype.str,
        "coefficient_storage": "native-real-npz",
        "coefficient_precision_bits": np.finfo(solution.pressure.dtype).nmant + 1,
        "operator_dtype": np.dtype(float).str,
        "operator_precision_bits": np.finfo(float).nmant + 1,
        "original_equation_relative_criterion": 1e-10,
        "refinement_precision": refinement_precision,
        "lockfile_sha256": lockfile_digest,
        "git_revision": revision,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "native_threads_requested": native_threads,
        "native_libraries_during_solve": native_libraries,
        "source_sha256": original,
        "case_conventions": case_conventions(),
        "material_case_provenance_verified": True,
        "archive": path.name,
        "archive_sha256": fingerprint(temporary_archive),
        "elapsed_seconds": time.perf_counter() - start,
    }
    _check_archive(temporary_archive, record, basis, n, degree)
    temporary_record = record_path.with_suffix(".json.part")
    temporary_record.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    snapshot_directory = directory / "acquisition-sources"
    snapshot_directory.mkdir(exist_ok=True)
    for name, digest in original.items():
        (snapshot_directory / f"{digest}.py").write_bytes(
            (source_file(name, root=ROOT)).read_bytes()
        )
    if lockfile_digest is not None:
        (snapshot_directory / f"{lockfile_digest}.lock").write_bytes(
            (ROOT / "pixi.lock").read_bytes()
        )
    if sources() != original or optional_file_digest(ROOT / "pixi.lock") != lockfile_digest:
        raise RuntimeError("reference sources or lockfile changed before publication")
    temporary_archive.replace(path)
    temporary_record.replace(record_path)
    print(json.dumps(record, allow_nan=False), flush=True)


def main() -> None:
    """Acquire requested nested Qk resolutions, retaining Q5/order10 defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument(
        "--refinement-precision", choices=("double", "extended"), default="extended"
    )
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--records", type=Path, default=ROOT / "examples/results")
    args = parser.parse_args()
    if min(*args.sizes, args.degree, args.native_threads) < 1 or args.order < args.degree + 1:
        parser.error("sizes, degree and threads must be positive; order must be >= degree + 1")
    for n in args.sizes:
        run(
            n,
            args.order,
            args.native_threads,
            degree=args.degree,
            refinement_precision=args.refinement_precision,
            artifacts=args.artifacts,
            records=args.records,
        )


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.periodic_reference").main()
