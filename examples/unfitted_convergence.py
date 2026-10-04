"""Trace h/p and material-contrast studies for CAMWA 209 (2026), Section 6.

The smooth problem follows the printed macro mesh, forcing and trace spaces.
Local P8 resolutions are explicit numerical controls because the paper does
not supply its local discretization. The contrast study declares ell=2 and
S2 delta=1/6, which Figure 7 does not itself restate. Absolute broken gradient
errors are kept distinct from material-weighted energy and physical flux errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from threadpoolctl import threadpool_limits

from examples.archive_precision import precision_fields
from examples.layered_poisson import LayeredPoissonSeries
from examples.unfitted_geometry import macro_mesh
from examples.unfitted_trace_family import ScalarTraceFamily
from pymhm._legacy.models.darcy.primal import DarcySolution, _assembly_quadrature_order, solve_darcy
from pymhm.core.validation import positive_int
from pymhm.fem.quadrature.material import (
    fit_material_faces,
    fit_material_mesh,
    material_triangle_quadrature,
)
from pymhm.fem.scalar.triangle import element_tabulate, multiindices, nodal_space
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import tensor_values
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/unfitted/convergence"
SOURCES = (
    "examples/unfitted_convergence.py",
    "examples/archive_precision.py",
    "examples/unfitted_trace_family.py",
    "examples/unfitted_campaign.py",
    "examples/unfitted_geometry.py",
    "examples/layered_poisson.py",
    "src/pymhm/_legacy/models/darcy/primal.py",
    "src/pymhm/fem/scalar/triangle.py",
    "src/pymhm/core/contracts.py",
    "src/pymhm/core/refinement.py",
    "src/pymhm/linalg/linear.py",
    "src/pymhm/meshes/triangle.py",
    "src/pymhm/fem/scalar/operators.py",
    "src/pymhm/fem/quadrature/material.py",
    "src/pymhm/io/reservoir.py",
    "src/pymhm/meshes/roundoff.py",
    "src/pymhm/meshes/refinement.py",
)


def source_hashes() -> dict[str, str]:
    """Identify actual executed numerical sources without exposing local paths."""
    return current_source_manifest(
        {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    )


def smooth_field(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sin(2πx)sin(2πy) and its independently differentiated gradient."""
    x, y = points.T
    pressure = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    gradient = (
        2
        * np.pi
        * np.column_stack(
            (
                np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y),
                np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
            )
        )
    )
    return pressure, gradient


def smooth_source(points: np.ndarray) -> np.ndarray:
    """Return minus the Laplacian of the manufactured sine field."""
    return 8 * np.pi**2 * smooth_field(points)[0]


class ScalarErrorFields(Protocol):
    """Original broken cardinal coefficients and material for physical norms."""

    @property
    def local_meshes(self) -> tuple[TriangleMesh, ...]:
        """Expose the original affine local triangulations."""
        ...

    @property
    def pressure(self) -> tuple[np.ndarray, ...]:
        """Expose complete broken cardinal pressure coefficient vectors."""
        ...

    @property
    def permeability(self) -> Any:
        """Expose the physical diffusion tensor or material field."""
        ...

    @property
    def degree(self) -> int:
        """Declare the cardinal polynomial degree in every local mesh."""
        ...


def error_norms(solution: ScalarErrorFields, exact: Callable, order: int) -> dict[str, float]:
    """Integrate distinct physical norms using material cuts and original local fields."""
    totals = np.zeros(8, dtype=np.longdouble)
    for mesh, coefficients in zip(solution.local_meshes, solution.pressure, strict=True):
        bary, weights, material = material_triangle_quadrature(mesh, solution.permeability, order)
        points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
        expected, expected_gradient = exact(points.reshape(-1, 2))
        dofs, _, basis, gradient, _ = element_tabulate(mesh, solution.degree, bary)
        values = np.einsum("ti,tqi->tq", coefficients[dofs], basis)
        actual_gradient = np.einsum("ti,tqia->tqa", coefficients[dofs], gradient)
        tensors = tensor_values(material, points.reshape(-1, 2)).reshape(*weights.shape, 2, 2)
        expected = expected.reshape(weights.shape)
        expected_gradient = expected_gradient.reshape(*weights.shape, 2)
        delta = actual_gradient - expected_gradient
        flux_delta = np.einsum("tqab,tqb->tqa", tensors, delta)
        flux = np.einsum("tqab,tqb->tqa", tensors, expected_gradient)
        integrands = (
            (values - expected) ** 2,
            np.sum(delta**2, axis=-1),
            np.sum(flux_delta**2, axis=-1),
            np.sum(delta * flux_delta, axis=-1),
            expected**2,
            np.sum(expected_gradient**2, axis=-1),
            np.sum(flux**2, axis=-1),
            np.sum(expected_gradient * flux, axis=-1),
        )
        for index, integrand in enumerate(integrands):
            totals[index] += np.sum(mesh.areas[:, None] * weights * integrand, dtype=np.longdouble)
    result = {}
    for index, name in enumerate(("pressure", "gradient", "flux", "energy")):
        error, norm = float(np.sqrt(totals[index])), float(np.sqrt(totals[index + 4]))
        result[f"{name}_absolute"] = error
        result[f"{name}_reference_norm"] = norm
        result[f"{name}_relative"] = error / norm
    return result


def save_field(solution: DarcySolution, path: Path) -> str:
    """Persist complete primal physical fields with their cardinal basis and trace maps.

    Pressure and physical normal-flux coefficients retain all digits as portable
    high/correction/tail float64 components. Local nodal DOFs, coordinates and
    barycentric multiindices declare the executed cardinal basis. Complete fields
    include original-equation defect-source corrections; replay evaluates these
    fields without repeating the solve. Unused coarse coordinates are omitted
    because ``DarcySolution`` does not expose their executed retained matrix E.
    """
    if solution.formulation != "primal":
        raise ValueError("unfitted field archive requires primal cardinal pressure fields")
    skeleton = solution.skeleton
    arrays = dict(
        field_archive_version=np.array(2),
        macro_points=skeleton.mesh.points,
        macro_cells=skeleton.mesh.cells,
        macro_faces=skeleton.mesh.faces,
        macro_face_cells=skeleton.mesh.face_cells,
        macro_normals=skeleton.mesh.normals,
        macro_signs=skeleton.mesh.signs,
        macro_cell_faces=skeleton.mesh.cell_faces,
        degree=np.array(solution.degree),
        coefficient_precision_bits=np.array(
            max(
                np.finfo(np.asarray(values).dtype).nmant + 1
                for values in (*solution.pressure, solution.hybrid.trace)
            )
        ),
        basis_multiindices=multiindices(solution.degree),
        **precision_fields("trace", solution.hybrid.trace),
    )
    for face, space in enumerate(skeleton.faces):
        arrays[f"trace_breaks_{face}"] = np.array(space.breaks)
        arrays[f"trace_degrees_{face}"] = np.array(space.degrees)
        arrays[f"trace_continuous_{face}"] = np.array(space.continuous)
        arrays[f"trace_dofs_{face}"] = skeleton.dofs(face)
    for index, (mesh, pressure) in enumerate(
        zip(solution.local_meshes, solution.pressure, strict=True)
    ):
        dofs, nodes = nodal_space(mesh, solution.degree)
        if np.shape(pressure) != (len(nodes),):
            raise ValueError("unfitted pressure coefficients differ from the declared nodal basis")
        arrays[f"points_{index}"] = mesh.points
        arrays[f"cells_{index}"] = mesh.cells
        arrays[f"nodal_dofs_{index}"] = dofs
        arrays[f"nodal_points_{index}"] = nodes
        arrays.update(precision_fields(f"pressure_{index}", pressure))
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smooth_configurations(maximum_segments: int) -> list[tuple[int, int]]:
    """Return both printed trace sweeps, with duplicate configurations executed once."""
    return [(ell, s) for ell in range(4) for s in (1, 2, 4, 8, 16, 32) if s <= maximum_segments] + [
        (4, s) for s in (1, 2, 4) if s <= maximum_segments
    ]


def main() -> None:
    """Acquire complete cases atomically under unchanged numerical-source provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("smooth", "contrast"), required=True)
    parser.add_argument("--refinement", type=int, required=True)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument(
        "--assembly-order",
        type=int,
        help="Gauss points per Duffy coordinate/boundary interval "
        "(default: degree+3; floor: degree+2)",
    )
    parser.add_argument(
        "--norm-orders",
        nargs="+",
        type=int,
        help="Distinct independent norm Gauss counts (default: degree+3 degree+5)",
    )
    parser.add_argument("--maximum-segments", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--local-solver", default="scipy")
    parser.add_argument("--names", nargs="+", help="Acquire only these explicitly named cases")
    parser.add_argument("--output", type=Path, default=DATA)
    parser.add_argument(
        "--contrasts", nargs="+", type=float, default=[10, 100, 1000, 10000, 100000, 1000000]
    )
    args = parser.parse_args()
    degree = positive_int(args.degree if args.study == "smooth" else 4, "degree")
    requested_order = degree + 3 if args.assembly_order is None else args.assembly_order
    assembly_order = _assembly_quadrature_order(degree, requested_order)
    norm_orders = sorted(
        positive_int(order, "norm_orders")
        for order in ([degree + 3, degree + 5] if args.norm_orders is None else args.norm_orders)
    )
    if len(norm_orders) != len(set(norm_orders)):
        raise ValueError("norm_orders must contain distinct Gauss counts")
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = dict(
        study=args.study,
        local_degree=degree,
        local_refinement=args.refinement,
        local_assembly_workers=args.workers if args.study == "smooth" else 1,
        local_solver=args.local_solver,
        requested_names=args.names,
        maximum_segments=args.maximum_segments if args.study == "smooth" else None,
        contrasts=args.contrasts if args.study == "contrast" else None,
        requested_assembly_order=requested_order,
        assembly_order=assembly_order,
        norm_orders=norm_orders,
        quadrature_convention="Gauss points per Duffy coordinate; boundary count also has "
        "trace degree+2 floor; trace coupling has a separate exact polynomial rule",
        material_fitted_local_meshes=args.study == "contrast",
        contrast_delta="1/6 (declared control, not restated in Figure 7)",
        local_refinement_precision="double" if args.study == "smooth" else "extended",
        hybrid_refinement_steps=0 if args.study == "smooth" else 3,
    )
    acquisition = f"{args.study}-p{degree}-r{args.refinement}"
    if assembly_order != degree + 3:
        acquisition += f"-q{assembly_order}"
    if norm_orders != [degree + 3, degree + 5]:
        acquisition += "-nq" + "-".join(str(order) for order in norm_orders)
    path = args.output / f"{acquisition}.json"
    before = source_hashes()
    record = (
        json.loads(path.read_text())
        if path.exists()
        else dict(
            configuration=configuration,
            source_sha256=before,
            source_changed_during_run=False,
            paper="Chaumont-Frelet, Paredes, Valentin, CAMWA 209 (2026), Section 6",
            doi="10.1016/j.camwa.2026.01.016",
            primary_error="absolute broken H1 seminorm of pressure error",
            cases=[],
        )
    )
    if record["configuration"] != configuration or record["source_sha256"] != before:
        raise ValueError("preserve previous acquisition: requested configuration or sources differ")

    def checkpoint() -> None:
        """Publish only complete rows whose physical archive and source bytes are identified."""
        if source_hashes() != before:
            raise RuntimeError("numerical sources changed during the acquisition")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path)

    with threadpool_limits(1):
        if args.study == "smooth":
            start = time.monotonic()
            family = ScalarTraceFamily.prepare(
                macro_mesh(0),
                trace_degree=4,
                segments=args.maximum_segments,
                local_degree=degree,
                local_refinement=args.refinement,
                source=smooth_source,
                quadrature_order=configuration["assembly_order"],
                workers=args.workers,
                local_solver=args.local_solver,
            )
            record["preparation_seconds"] = time.monotonic() - start
            configurations = [
                (f"ell{ell}-s{s}", ell, s)
                for ell, s in smooth_configurations(args.maximum_segments)
            ]
        else:
            configurations = [
                (f"{setting}-contrast{int(contrast)}", setting, contrast)
                for contrast in args.contrasts
                for setting in ("S0", "S2")
            ]
        if args.names is not None:
            available = {row[0] for row in configurations}
            if len(args.names) != len(set(args.names)) or not set(args.names) <= available:
                raise ValueError("requested names must be distinct available configurations")
            configurations = [row for row in configurations if row[0] in args.names]
        for name, first, second in configurations:
            if any(row["name"] == name for row in record["cases"]):
                continue
            start = time.monotonic()
            if args.study == "smooth":
                solution, diagnostics = family.solve(first, second)
                exact = smooth_field
                row = dict(
                    name=name,
                    trace_degree=first,
                    segments=second,
                    macro_diameter=0.5,
                    H=0.5 / second,
                    **diagnostics,
                )
            else:
                mesh = macro_mesh(0 if first == "S0" else 1 / 6)
                material = CartesianCellField(np.array([[float(second), 1.0]]), (1.0, 0.5))
                skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
                if first == "S2":
                    skeleton = fit_material_faces(skeleton, material)
                local = tuple(
                    fit_material_mesh(mesh.submesh(i, args.refinement), material)
                    for i in range(len(mesh.cells))
                )
                solution = solve_darcy(
                    mesh,
                    permeability=material,
                    source=1,
                    skeleton=skeleton,
                    degree=4,
                    local_refinement=args.refinement,
                    local_meshes=local,
                    quadrature_order=configuration["assembly_order"],
                    local_refinement_precision="extended",
                    hybrid_refinement_steps=configuration["hybrid_refinement_steps"],
                    local_solver=args.local_solver,
                )
                exact = LayeredPoissonSeries(second, 1023).evaluate
                row = dict(
                    name=name,
                    setting=first,
                    contrast=second,
                    delta=0 if first == "S0" else 1 / 6,
                    trace_degree=2,
                    reference_series_odd_modes=1023,
                )
            row.update(
                solve_seconds=time.monotonic() - start,
                trace_dofs=solution.skeleton.size,
                macro_cells=len(solution.local_meshes),
                fine_cells=sum(len(m.cells) for m in solution.local_meshes),
                global_dofs=solution.skeleton.size + len(solution.local_meshes),
                macro_balance_max=float(np.max(np.abs(solution.conservation_residuals()))),
            )
            row["norms"] = {
                f"quadrature_{order}": error_norms(solution, exact, order)
                for order in configuration["norm_orders"]
            }
            if args.study == "contrast":
                row["series_2047_control"] = error_norms(
                    solution,
                    LayeredPoissonSeries(second, 2047).evaluate,
                    configuration["norm_orders"][-1],
                )
            archive = args.output / f"{acquisition}-{name}.npz"
            row.update(
                archive=archive.name,
                archive_sha256=save_field(solution, archive),
                total_seconds=time.monotonic() - start,
            )
            record["cases"].append(row)
            checkpoint()
            print(json.dumps(row), flush=True)
        record["complete"] = True
        checkpoint()


if __name__ == "__main__":
    main()
