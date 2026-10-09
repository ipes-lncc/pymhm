"""Check scalar tutorial local Galerkin error at fixed data and trace spaces.

This acquisition delegates every solve, evaluation and physical norm to its
existing owner. It doubles the local subdivisions at one fixed macro mesh and
compares the two executed fields without averaging discontinuous pressures or
normal traces. These are local-refinement controls, not macro rate measurements.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import (
    moment_diffusion,
    petrov_galerkin_diffusion,
    three_field_diffusion,
)
from examples.formulations.mixed_darcy import define_rt_darcy, recover_rt_darcy
from examples.minimal_scalar_convergence import _mh2m_original, hybrid_original
from examples.tutorial_scalar_acquisition import (
    _mshho_original,
    _pgmhm_original,
    moment_source,
    physical_errors,
    source,
    source_manifest,
    three_field_source,
)
from pymhm import ExecutionConfig, FaceSpace, SkeletonSpace, TriangleMesh, assemble
from pymhm.fem.traces.pressure_2d import PressureTraceSpace
from pymhm.io.provenance import file_digest
from pymhm.postprocessing.fields import DiscreteField
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.sampling import TrianglePointLocator
from pymhm.postprocessing.velocity import polynomial_darcy_velocity

REFINEMENTS = {"mixed-mhm": 2, "mshho": 2, "pgmhm": 1, "mh2m": 4}


def solve_variant(method: str, mesh: TriangleMesh, refinement: int) -> tuple[Any, dict[str, Any]]:
    """Preserve the qualified spaces/data, changing only the fine subdivision.

    The compatible MH2M Gamma/Lambda segmentation remains one/two, so refining
    its local mesh strengthens M1 while retaining M2. MsHHO retains the projected
    source and extended moment arithmetic. PGMHM preserves its enrichment.
    """
    execution = {"backend": "process", "workers": 16}
    if method == "mixed-mhm":
        definition = define_rt_darcy(
            mesh, degree=1, source=source, local_refinement=refinement, quadrature_order=8
        )
        system = assemble(definition.problem, execution=ExecutionConfig(**execution))
        coefficients = system.solve()
        solution = recover_rt_darcy(definition, system, coefficients)
        original = hybrid_original(
            system,
            coefficients,
            dict(definition.problem.fixed),
            system.global_load[: system.trace_size],
        )
    elif method == "mshho":
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0) for _ in mesh.faces))
        solution = moment_diffusion(
            mesh,
            skeleton=skeleton,
            cell_degree=0,
            degree=3,
            local_refinement=refinement,
            source=moment_source,
            quadrature_order=8,
            reconstruction_precision="extended",
            local_refinement_precision="extended",
        )
        original = _mshho_original(solution)
    elif method == "pgmhm":
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
        solution = petrov_galerkin_diffusion(
            mesh,
            skeleton=skeleton,
            degree=3,
            local_refinement=refinement,
            source=source,
            stabilization_parameter=0.1,
            quadrature_order=8,
            **execution,
        )
        original = _pgmhm_original(solution)
    elif method == "mh2m":
        solution = three_field_diffusion(
            mesh,
            pressure_trace=PressureTraceSpace.uniform(mesh, degree=2),
            flux_space=SkeletonSpace(
                mesh, tuple(FaceSpace.uniform(1, subdivisions=2) for _ in mesh.faces)
            ),
            degree=2,
            local_refinement=refinement,
            source=three_field_source,
            quadrature_order=8,
            **execution,
        )
        original = _mh2m_original(solution)
    else:
        raise ValueError(f"Unsupported scalar tutorial method: {method}")
    return solution, original


def field_references(
    method: str, mesh: TriangleMesh, solution: Any
) -> tuple[Callable[[Any], np.ndarray], Callable[[Any], np.ndarray]]:
    """Build live pressure/physical-gradient callbacks in the executed field basis.

    DG pressure remains discontinuous by fine cell. RT flux is converted by the
    shared owner to its exactly equivalent vector polynomial. Other fluxes are
    minus the raw pressure gradient for identity permeability; MH2M compares the
    positive broken gradient because that is its declared rate observable.
    All quadrature points are strictly inside the nested fine integration cells.
    """
    meshes = (
        solution.local_meshes
        if method in {"mixed-mhm", "pgmhm"}
        else tuple(data.mesh for data in solution.local)
    )
    values = solution.enriched_pressure if method == "pgmhm" else solution.pressure
    fields = tuple(
        DiscreteField(
            nodal_field("pressure", fine, solution.degree, discontinuous=method == "mixed-mhm"),
            np.asarray(coefficients).ravel(),
        )
        for fine, coefficients in zip(meshes, values, strict=True)
    )
    vectors = (
        tuple(polynomial_darcy_velocity(solution, cell) for cell in range(len(mesh.cells)))
        if method == "mixed-mhm"
        else None
    )
    locator = TrianglePointLocator(mesh)

    def pressure(points: Any) -> np.ndarray:
        """Dispatch analytical comparisons to the original incident macro field."""
        points = np.asarray(points)
        owners = locator.locate(points)
        result = np.empty(len(points))
        for cell in np.unique(owners):
            selected = owners == cell
            result[selected] = fields[cell].evaluate(points[selected])
        return result

    def derivative(points: Any) -> np.ndarray:
        """Return original physical RT flux, raw nodal flux, or MH2M gradient."""
        points = np.asarray(points)
        owners = locator.locate(points)
        result = np.empty((len(points), 2))
        for cell in np.unique(owners):
            selected = owners == cell
            result[selected] = (
                vectors[cell].evaluate(points[selected])
                if vectors is not None
                else fields[cell].gradient(points[selected]) * (1 if method == "mh2m" else -1)
            )
        return result

    return pressure, derivative


def field_difference(solution: Any, method: str, references: tuple[Any, Any], order: int) -> dict:
    """Delegate full pairwise norms to the refined solution's integration owner."""
    options = {"enriched": True} if method == "pgmhm" else {}
    pressure, derivative = references
    key = "gradient_l2" if method == "mh2m" else "flux_l2"
    value = (
        solution.gradient_l2_error(derivative, order=order)
        if method == "mh2m"
        else solution.flux_l2_error(derivative, order=order, **options)
    )
    return {"pressure_l2": solution.l2_error(pressure, order=order, **options), key: value}


def acquire_local_controls(root: Path, *, n: int = 8) -> dict[str, Any]:
    """Acquire one independent two-grid control per qualified scalar method.

    The records contain actual pairwise field differences, original equation
    checks and independent error/difference quadrature, without treating changes
    in errors as substitutes for differences between the reconstructed fields.
    """
    sources = source_manifest(root)
    result: dict[str, Any] = {
        "method": "Fixed-macro scalar local-refinement controls",
        "source_sha256": sources,
        "source_changed_during_run": False,
        "macro_resolution": n,
        "assembly_quadrature": 8,
        "error_quadratures": [10, 12],
        "scope": "Same physical problem and trace spaces; doubled local subdivisions only",
        "rows": [],
    }
    mesh = TriangleMesh.unit_square(n)
    for method, refinement in REFINEMENTS.items():
        start = perf_counter()
        with threadpool_limits(1):
            coarse, coarse_original = solve_variant(method, mesh, refinement)
            fine, fine_original = solve_variant(method, mesh, 2 * refinement)
            references = field_references(method, mesh, coarse)
            replay = field_difference(coarse, method, references, 12)
            if max(replay.values()) > 1e-10:
                raise ArithmeticError(f"The executed coarse field basis does not replay: {replay}")
            differences = {
                str(order): field_difference(fine, method, references, order) for order in (10, 12)
            }
            errors = {
                label: {str(order): physical_errors(field, method, order) for order in (10, 12)}
                for label, field in (("coarse", coarse), ("fine", fine))
            }
            discrepancy = max(
                abs(differences["10"][key] / differences["12"][key] - 1)
                for key in differences["10"]
            )
            scaled_quad_change = max(
                abs(differences["10"][key] - differences["12"][key]) / errors["coarse"]["12"][key]
                for key in differences["10"]
            )
            norm_discrepancy = max(
                abs(measure["10"][key] / measure["12"][key] - 1)
                for measure in errors.values()
                for key in measure["10"]
            )
            # A difference of two equivalent fields can be at roundoff. Its
            # integration uncertainty is scaled to the physical error whose
            # local contribution is assessed, with no artificial absolute floor.
            # Preserve the raw relative change as data rather than requesting
            # relative accuracy on a quantity indistinguishable from zero.
            if max(scaled_quad_change, norm_discrepancy) > 1e-9:
                raise ArithmeticError("Independent two-grid field quadratures disagree")
        if any(file_digest(root / name) != digest for name, digest in sources.items()):
            raise RuntimeError("An executed numerical source changed during local control")
        result["rows"].append(
            {
                "method": method,
                "local_subdivisions": [refinement, 2 * refinement],
                "physical_errors": errors,
                "pairwise_field_difference": differences,
                "difference_to_coarse_error": {
                    key: value / errors["coarse"]["12"][key]
                    for key, value in differences["12"].items()
                },
                "coarse_basis_replay_difference": replay,
                "difference_quadrature_relative_change": discrepancy,
                "difference_quadrature_change_relative_to_coarse_error": scaled_quad_change,
                "norm_quadrature_relative_change": norm_discrepancy,
                "original_equations": {"coarse": coarse_original, "fine": fine_original},
                "seconds": perf_counter() - start,
            }
        )
        print(method, result["rows"][-1]["difference_to_coarse_error"], flush=True)
    return result


def main() -> None:
    """Write the separately attributed current local-control record."""
    root = Path(__file__).resolve().parents[1]
    output = root / "examples/results/tutorial-methods/scalar-local-controls-current.json"
    output.write_text(json.dumps(acquire_local_controls(root), indent=2) + "\n")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.tutorial_scalar_local_control").main()
