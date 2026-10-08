"""Acquire three small Darcy/scalar resolutions with current physical conventions.

These initial studies keep the stated approximation spaces and physical data.
Only scalar norms, actual checked-solve diagnostics and reproducible source
provenance are persisted. No coefficient vector, replay certificate, converged
reference or matched historical reproduction is asserted.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
from threadpoolctl import threadpool_limits

from examples import pgmhm_campaign, solve_spe10, solve_unusual
from examples.formulations.application import cartesian_darcy as solve_darcy_quadrilateral
from examples.formulations.application import hdiv_darcy as solve_darcy_hdiv3d
from examples.formulations.application import petrov_galerkin_diffusion as solve_pgmhm
from examples.formulations.application import transport as solve_rad
from examples.solve_mapped_well import WellData
from pymhm.fem.scalar.quadrilateral import (
    quadrilateral_trace_coupling,
    rectangle_intersection_quadrature,
)
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.linalg.linear import LinearFactorization, accurate_residual
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.hexahedron import HexMesh
from pymhm.meshes.mixed import AffineMixedMesh
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]


def source_hashes() -> dict[str, str]:
    """Identify every portable core owner, imported case helper and the Pixi lock."""
    paths = list((ROOT / "src/pymhm").rglob("*.py"))
    paths += [Path(__file__), ROOT / "pixi.lock"]
    paths += [
        ROOT / "examples" / name
        for name in (
            "pgmhm_campaign.py",
            "mh_campaign.py",
            "solve_spe10.py",
            "solve_unusual.py",
            "solve_mapped_well.py",
            "field_sampling.py",
            "archive_precision.py",
        )
    ]
    paths.append(ROOT / "examples/results/spe10/layer-36.npz")
    return current_source_manifest(
        {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)
        }
    )


@contextmanager
def observe_checked_solves() -> Iterator[list[dict[str, Any]]]:
    """Observe original augmented RHS columns; preserve the owner's solves and tolerances.

    These are Euclidean coefficient-row residuals of the actually executed
    linear systems. They are distinct from physical field norms and from a
    separately assembled uncondensed full PDE verification.
    """
    rows: list[dict[str, Any]] = []
    original = LinearFactorization.solve

    def observed(self: Any, rhs: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, rhs, *args, **kwargs)
        forcing = np.asarray(rhs, dtype=np.result_type(np.asarray(rhs).dtype, result.dtype))
        defect = accurate_residual(self._matrix, forcing, result)
        norms = np.atleast_1d(np.linalg.norm(defect, axis=0))
        denominators = np.atleast_1d(np.linalg.norm(forcing, axis=0))
        relative = [
            float(value / scale) if scale else None
            for value, scale in zip(norms, denominators, strict=True)
        ]
        rows.append(
            dict(
                matrix_shape=list(self._matrix.shape),
                rhs_columns=len(norms),
                owner_rtol=self.rtol,
                owner_atol=self.atol,
                original_defect_norm_max=float(norms.max()),
                original_rhs_relative_max=max(
                    (value for value in relative if value is not None), default=0.0
                ),
                zero_rhs_columns=int(np.count_nonzero(denominators == 0)),
                original_owner_criterion_passed=True,
            )
        )
        return result

    LinearFactorization.solve = observed
    try:
        yield rows
    finally:
        LinearFactorization.solve = original


def scalar_case(case: str, level: int) -> tuple[Any, dict[str, Any]]:
    """Solve PGMHM P3/P1 or UNUSUAL P1/P0 with the current published case helpers."""
    mesh = TriangleMesh.unit_square(level)
    if case == "pgmhm":
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
        solution = solve_pgmhm(
            mesh,
            source=pgmhm_campaign.source,
            skeleton=skeleton,
            degree=3,
            local_refinement=1,
            stabilization_parameter=0.1,
            quadrature_order=8,
            backend="serial",
            workers=1,
        )
        norms = {str(order): pgmhm_campaign.norms(solution, order) for order in (10, 12)}
        balance = float(np.max(np.abs(solution.conservation_residuals(enriched=True))))
        row = dict(
            level=level,
            h=float(mesh.lengths.max()),
            approximation="P3 locals/P1 traces, one fine triangle per macro",
            alpha=0.1,
            assembly_order=8,
            norm_orders=[10, 12],
            norms=norms,
            pressure_l2_error=norms["12"]["enriched_pressure_l2"],
            flux_l2_error=norms["12"]["enriched_flux_l2"],
            global_condensed_residual=solution.hybrid.residual,
            enriched_macro_balance_max=balance,
            flux_convention=(
                "-grad(enriched pressure), a broken raw gradient; "
                "only the enriched normal multiplier is macro conservative"
            ),
            physical_case=(
                "unit square, sin(2*pi*x)*sin(2*pi*y), K=1, source=8*pi^2*p, weak pressure zero"
            ),
        )
    else:
        options, exact, gradient = solve_unusual.configuration("smooth", 1.0)
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0) for _ in mesh.faces))
        neumann = {
            int(face): 0.0 for face in mesh.boundary_faces if abs(mesh.normals[face, 1]) > 0.99
        }
        solution = solve_rad(
            mesh,
            skeleton=skeleton,
            degree=1,
            local_refinement=2,
            stabilization="unusual",
            dirichlet_enforcement="strong",
            neumann=neumann,
            quadrature_order=10,
            **options,
        )
        norms = {
            str(order): solve_unusual.errors(solution, 1.0, exact, gradient, order)
            for order in (12, 20)
        }
        row = dict(
            level=level,
            h=float(mesh.lengths.max()),
            approximation="P1 locals/P0 traces, one red local refinement",
            assembly_order=10,
            norm_orders=[12, 20],
            norms=norms,
            pressure_l2_error=norms["20"]["scalar_l2"],
            flux_l2_error=norms["20"]["flux_l2"],
            global_condensed_residual=solution.hybrid.residual,
            flux_convention="physical diffusion flux=-grad(u), a broken raw gradient",
            physical_case=(
                "reaction-diffusion -Delta(u)+u=1; epsilon=1; strong zero pressure "
                "on vertical sides, zero normal flux on horizontal sides"
            ),
            inverse_constant="P1 published m=1/3; negative UNUSUAL strong-residual stabilization",
        )
    row.update(
        macro_cells=len(mesh.cells),
        fine_cells=sum(len(fine.cells) for fine in solution.local_meshes),
    )
    return solution, row


def well_case(level: int) -> tuple[Any, dict[str, Any]]:
    """Solve tetrahedral P1-pressure mixed Darcy on the unchanged octagonal annulus."""
    data = WellData()
    hexa = HexMesh.annular_prism(
        np.geomspace(data.inner_radius, data.outer_radius, 5), data.height, 8
    )
    mesh = AffineMixedMesh.from_extruded_hexahedra(hexa.points, hexa.cells, "tetrahedron")
    neumann = {
        int(face): 0.0
        for face in mesh.boundary_faces
        if np.ptp(mesh.points[mesh.faces[face], 2]) < 1e-12
    }
    solution = solve_darcy_hdiv3d(
        mesh,
        pressure_degree=1,
        local_refinement=level,
        permeability=data.permeability / data.viscosity,
        dirichlet=data.pressure,
        neumann=neumann,
        quadrature_order=5,
        boundary_quadrature_order=12,
        backend="serial",
        workers=1,
        global_rtol=1e-12,
        global_refinement_precision="extended",
    )
    norms = {str(order): solution.errors(data.pressure, data.flux, order) for order in (7, 10)}
    diameter = max(
        float(np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1).max())
        for fine in solution.local_meshes
        for vertices in fine.points[fine.cells]
    )
    return solution, dict(
        level=level,
        h=diameter,
        h_convention="maximum physical fine-cell vertex diameter, metres",
        approximation=(
            "tetrahedron 18-dimensional H(div) flux/P1 pressure; P1 coarse normal moments"
        ),
        macro_cells=len(mesh.cells),
        fine_cells=sum(len(fine.cells) for fine in solution.local_meshes),
        assembly_order=5,
        boundary_order=12,
        norm_orders=[7, 10],
        norms=norms,
        pressure_l2_error=norms["10"]["pressure_l2"],
        flux_l2_error=norms["10"]["flux_l2"],
        global_condensed_residual=solution.hybrid.residual,
        original_physical_block_residual_max=float(np.max(solution.physical_residuals)),
        fine_pressure_moment_defect_max=float(
            max(np.max(np.abs(value)) for value in solution.equilibrium_residuals())
        ),
        flux_convention="physical H(div) Darcy flux; contravariant affine Piola",
        physical_case=(
            "Dupuit-Thiem radial pressure on a fixed planar octagonal annulus, "
            "radii 0.2/50m, height 10m; radial pressure on polygonal walls "
            "and zero outward flux on caps"
        ),
        refinement="local fine factor1/2/4 with macro factor1 and identical polygonal domain",
    )


def spe_trace_compatibility(
    mesh: CartesianMacroMesh, skeleton: SkeletonSpace, level: int
) -> list[dict[str, Any]]:
    """Require injective executed trace coupling in each rectangle orientation class.

    This campaign fixes Q1 locals, congruent rectangular macros and continuous
    piecewise P1 traces with two segments. Full column rank is a necessary local
    compatibility condition, not a global inf-sup or uniqueness certificate.
    In particular r2 has only eight boundary nodes for twelve trace coefficients.
    """
    representatives: dict[tuple[int, ...], int] = {}
    for cell, signs in enumerate(mesh.signs):
        representatives.setdefault(tuple(int(value) for value in signs), cell)
    checks = []
    for signs, cell in representatives.items():
        matrix = quadrilateral_trace_coupling(mesh, cell, mesh.submesh(cell, level), skeleton, 1)
        rank = int(np.linalg.matrix_rank(matrix))
        if rank < matrix.shape[1]:
            raise ValueError(
                f"Q1/r{level} cannot represent this P1C0/s2 trace: "
                f"local coupling rank {rank} < {matrix.shape[1]} columns"
            )
        checks.append(dict(cell=cell, signs=list(signs), shape=list(matrix.shape), rank=rank))
    return checks


def spe_case(level: int) -> tuple[Any, list[dict[str, Any]]]:
    """Solve unchanged layer36, Q1 locals/P1 continuous traces on66 fixed macros."""
    mesh = CartesianMacroMesh(6, 11, (0, 1200, 0, 2200))
    skeleton = SkeletonSpace(
        mesh, tuple(FaceSpace.uniform(1, 2, continuous=True) for _ in mesh.faces)
    )
    compatibility = spe_trace_compatibility(mesh, skeleton, level)
    neumann = {int(face): 0.0 for face in mesh.boundary_faces if abs(mesh.normals[face, 0]) > 0.5}
    solution = solve_darcy_quadrilateral(
        mesh,
        permeability=solve_spe10.load_layer(),
        source=0.0,
        dirichlet=solve_spe10.pressure_boundary,
        neumann=neumann,
        skeleton=skeleton,
        local_refinement=level,
        degree=1,
        quadrature_order=5,
        backend="serial",
        workers=1,
    )
    return solution, compatibility


def spe_norms(solution: Any, previous: Any, order: int) -> dict[str, float]:
    """Integrate own-field norms and nested local increments on exact pixel intersections.

    Rectangular local meshes share the same fixed macrogrid and refine by factors
    two. Quadrature partitions at every current fine edge and material pixel.
    Coarse fields are evaluated in their own Q1 basis through the public owner.
    """
    totals = np.zeros(4, dtype=np.longdouble)
    for cell, fine in enumerate(solution.local_meshes):
        origins = fine.points[fine.cells[:, 0]]
        reference, weights = rectangle_intersection_quadrature(
            origins, fine.spacing, solution.permeability, order
        )
        physical = origins[:, None] + reference * fine.spacing
        p, q = solution.evaluate(cell, physical.reshape(-1, 2))
        weight = (weights * np.prod(fine.spacing)).ravel()
        totals[:2] += [
            np.sum(weight * p**2, dtype=np.longdouble),
            np.sum(weight * np.sum(q**2, axis=1), dtype=np.longdouble),
        ]
        if previous is not None:
            old_p, old_q = previous.evaluate(cell, physical.reshape(-1, 2))
            totals[2:] += [
                np.sum(weight * (p - old_p) ** 2, dtype=np.longdouble),
                np.sum(weight * np.sum((q - old_q) ** 2, axis=1), dtype=np.longdouble),
            ]
    values = np.sqrt(totals)
    result = dict(pressure_l2_norm=float(values[0]), flux_l2_norm=float(values[1]))
    if previous is not None:
        result.update(
            pressure_l2_increment=float(values[2]),
            flux_l2_increment=float(values[3]),
            pressure_relative_increment=float(values[2] / values[0]),
            flux_relative_increment=float(values[3] / values[1]),
        )
    return result


def run(case: str, output: Path) -> dict[str, Any]:
    """Acquire one bounded three-level study, preserving progress after each level."""
    if output.exists():
        raise ValueError("A new output directory is required; prior execution bytes are preserved")
    output.mkdir(parents=True)
    hashes = source_hashes()
    snapshot = output / "executed-sources"
    for name in hashes:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    report: dict[str, Any] = dict(
        schema="pymhm-minimal-darcy-convergence-v1",
        case=case,
        acquisition_id=str(uuid4()),
        source_sha256=hashes,
        rows=[],
        complete=False,
        persisted_coefficients=False,
        replay_claim=False,
        independently_assembled_reference=False,
        interpretation=(
            "Initial three-resolution study; neither a converged numerical baseline "
            "nor historical paper reproduction"
        ),
        residual_convention=(
            "Actual checked original augmented linear systems, by RHS column; "
            "Euclidean coefficient-row norms, separate from physical field errors. "
            "These diagnostics alone are not full uncondensed PDE/uniqueness/inf-sup certificates."
        ),
    )
    previous = None
    levels = (4, 8, 16) if case == "spe10" else (1, 2, 4) if case == "mixedwell" else (2, 4, 8)
    report["requested_levels"] = list(levels)
    (output / "minimal-convergence.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with threadpool_limits(1):
        for level in levels:
            started = perf_counter()
            with observe_checked_solves() as checks:
                if case in {"pgmhm", "unusual"}:
                    solution, row = scalar_case(case, level)
                elif case == "mixedwell":
                    solution, row = well_case(level)
                else:
                    solution, compatibility = spe_case(level)
                    norms = {str(order): spe_norms(solution, previous, order) for order in (5, 7)}
                    row = dict(
                        level=level,
                        h=200.0 / level,
                        macro_cells=66,
                        fine_cells=66 * level**2,
                        approximation=(
                            "Q1 locals/continuous P1 trace with 2segments; fixed 6x11macrogrid"
                        ),
                        assembly_order=5,
                        norm_orders=[5, 7],
                        norms=norms,
                        trace_coupling_compatibility=compatibility,
                        global_condensed_residual=solution.hybrid.residual,
                        macro_balance_max=float(np.max(np.abs(solution.conservation_residuals()))),
                        flux_convention="physical Darcy flux=-Kgrad(p), a broken raw gradient",
                        physical_case=(
                            "SPE10 Model2 layer36; exact original Cartesian material, "
                            "1200x2200ft; bottom pressure1/top0, zero flux on vertical sides"
                        ),
                        units=(
                            "coordinates ft, permeability mD, flux mD/ft; "
                            "unchanged published-case driver conventions"
                        ),
                        refinement=(
                            "local r4/8/16 only; fixed macro mesh and fixed trace restriction; "
                            "no exact pressure solution"
                        ),
                    )
                    previous = solution
            row.update(
                elapsed_seconds=perf_counter() - started,
                checked_original_solves=checks,
                checked_original_solve_count=len(checks),
            )
            report["rows"].append(row)
            if hashes != source_hashes():
                raise ValueError("An executed numerical source changed during acquisition")
            (output / "minimal-convergence.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    dict(
                        case=case,
                        level=level,
                        seconds=row["elapsed_seconds"],
                        global_residual=row["global_condensed_residual"],
                    )
                ),
                flush=True,
            )
    report["complete"] = True
    report["source_changed_during_run"] = False
    (output / "minimal-convergence.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    """Acquire one explicitly selected small study under an external resource guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pgmhm", "unusual", "spe10", "mixedwell"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.case, args.output)


if __name__ == "__main__":
    main()
