"""Acquire a conforming UFL reference for the smooth anisotropic elasticity case.

The constant Kelvin tensor, exact sine displacement and strong zero exterior
data match the separate primal-MHM refinement study. UFL differentiates the
physical stress to derive the force, independently of the MHM analytic Hessian
callbacks. Shared package owners assemble, map coordinates and solve the
original free equations. Physical norms use independent quadrature rules.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm import TriangleMesh, compile_form
from pymhm.backends.spaces import bind_space, coefficient_map
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from pymhm.linalg.linear import accurate_residual, factorize

ROOT = case_workspace()
KELVIN_STIFFNESS = np.array([[5.0, 1.0, 0.4], [1.0, 4.0, 0.3], [0.4, 0.3, 2.0]])


def physical_stress(displacement: Any) -> Any:
    """Apply C in the orthonormal (xx, yy, sqrt(2)xy) Kelvin representation.

    The returned symmetric tensor is the physical Cauchy stress; its Frobenius
    norm counts both off-diagonal entries. No pressure or pseudotraction is
    substituted for this tensor.
    """
    import ufl

    strain = ufl.sym(ufl.grad(displacement))
    kelvin = ufl.as_vector((strain[0, 0], strain[1, 1], np.sqrt(2) * strain[0, 1]))
    stress = ufl.dot(ufl.as_matrix(KELVIN_STIFFNESS), kelvin)
    return ufl.as_matrix(((stress[0], stress[2] / np.sqrt(2)), (stress[2] / np.sqrt(2), stress[1])))


def exact_forms(domain: Any) -> tuple[Any, Any, Any]:
    """Differentiate the stated analytical displacement and its stress in UFL."""
    import ufl

    x = ufl.SpatialCoordinate(domain)
    displacement = ufl.as_vector(
        (
            ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1]),
            ufl.sin(2 * np.pi * x[0]) * ufl.sin(np.pi * x[1]),
        )
    )
    stress = physical_stress(displacement)
    return displacement, stress, -ufl.div(stress)


def reference_row(resolution: int) -> dict[str, Any]:
    """Solve global conforming P3 elasticity and measure displacement and stress.

    The grid is the same unit-square diagonal triangulation as the MHM study,
    with two triangles per square. Zero displacement is imposed strongly on
    all exterior nodes. Only original free equilibrium rows enter the stated
    relative residual; the imposed rows represent the essential condition.
    """
    import basix
    import basix.ufl
    import ufl
    from dolfinx import fem

    started = perf_counter()
    mesh = TriangleMesh.unit_square(resolution)
    native = bind_space(
        mesh,
        basix.ufl.element(
            "Lagrange", "triangle", 3, shape=(2,), lagrange_variant=basix.LagrangeVariant.equispaced
        ),
    )
    try:
        u, v = ufl.TrialFunction(native.space), ufl.TestFunction(native.space)
        domain = native.space.mesh
        exact_u, exact_sigma, force = exact_forms(domain)
        measure = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 14})
        matrix = compile_form(ufl.inner(physical_stress(u), ufl.sym(ufl.grad(v))) * measure)
        load = compile_form(ufl.inner(force, v) * measure)
        _, nodes = nodal_space(mesh, 3)
        exterior = np.flatnonzero(np.any(np.isclose(nodes, 0) | np.isclose(nodes, 1), axis=1))
        boundary = coefficient_map(native)[(2 * exterior[:, None] + np.arange(2)).ravel()]
        free = np.setdiff1d(np.arange(native.size), boundary)
        operator, forcing = matrix[free][:, free], load[free]
        with factorize(operator, rtol=1e-10, atol=0.0) as solver:
            free_values = solver.solve(forcing)
        original_residual = accurate_residual(operator, forcing, free_values)
        original_relative = float(np.linalg.norm(original_residual) / np.linalg.norm(forcing))
        numerical = fem.Function(native.space)
        numerical.x.array[free] = free_values
        numerical.x.scatter_forward()
        differences = {
            "displacement_l2": numerical - exact_u,
            "stress_l2": physical_stress(numerical) - exact_sigma,
        }
        errors = {}
        for order in (14, 18):
            quadrature = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": order})
            errors[str(order)] = {
                name: float(
                    np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(delta, delta) * quadrature)))
                )
                for name, delta in differences.items()
            }
        changes = {
            name: abs(errors["14"][name] - errors["18"][name]) / errors["18"][name]
            for name in differences
        }
        accepted = original_relative <= 1e-10 and max(changes.values()) <= 1e-8
        return {
            "resolution": resolution,
            "grid_spacing": 1.0 / resolution,
            "triangle_diameter": float(np.sqrt(2) / resolution),
            "triangles": len(mesh.cells),
            "degree": 3,
            "dofs": native.size,
            "free_dofs": len(free),
            **errors["18"],
            "original_free_equilibrium_relative_residual": original_relative,
            "strong_boundary_displacement_linf": float(np.max(abs(numerical.x.array[boundary]))),
            "quadrature": {
                "assembly_degree": 14,
                "error_degrees": [14, 18],
                "errors": errors,
                "relative_changes": changes,
                "relative_change_criterion": 1e-8,
            },
            "wall_seconds": perf_counter() - started,
            "accepted": bool(accepted),
        }
    finally:
        native.close()


def source_hashes() -> dict[str, str]:
    """Freeze this acquisition, all numerical package owners and locked profiles."""
    paths = [
        Path(__file__),
        *sorted((ROOT / "src/pymhm").rglob("*.py")),
        ROOT / "pixi.lock",
        ROOT / "pixi.toml",
        ROOT / "pyproject.toml",
    ]
    return current_source_manifest(source_identity(ROOT, paths), packages=("pymhm",))


def main() -> None:
    """Archive an attributed native reference without overwriting any record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("An existing numerical record cannot be overwritten")
    sources = source_hashes()
    record = {
        "schema": "pymhm-native-anisotropic-primal-elasticity-reference-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": sources,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("pymhm", "numpy", "scipy", "fenics-dolfinx", "fenics-basix", "fenics-ufl")
        },
        "native_threads": 1,
        "domain": "Unit-square diagonal triangles; two triangles per square",
        "tensor_kelvin": KELVIN_STIFFNESS.tolist(),
        "exact_displacement": "(sin(pi*x)*sin(pi*y), sin(2*pi*x)*sin(pi*y))",
        "source": (
            "minus divergence of the physical Cauchy stress, differentiated independently in UFL"
        ),
        "boundary": "strong zero displacement on the entire exterior",
        "space": "global conforming vector P3, equispaced Basix nodes",
        "field_norms": "absolute physical displacement L2 and symmetric Cauchy-stress Frobenius L2",
        "solver": "shared checked SciPy LU; original free-equilibrium criterion1e-10",
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        for level in args.levels:
            row = reference_row(level)
            record["rows"].append(row)
            args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
            if not row["accepted"]:
                raise ValueError(
                    "Original physical equilibrium or independent error quadrature rejected"
                )
    unchanged = all(
        file_digest(source_file(name, root=ROOT)) == digest for name, digest in sources.items()
    )
    if not unchanged:
        raise RuntimeError("An executed reference source changed during acquisition")
    record["source_changed_during_acquisition"] = False
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
