"""Measurements and displays for the reaction-layer introduction.

The notebook declares every physical form and solve. Helpers read already
computed fields or assemble a separately supplied conforming reference form.
Matplotlib is loaded only by plotting functions; field measurements and
reference assembly remain available without visualization dependencies.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from examples.introduction.vector import (
    display_samples,
    plot_errors,
    plot_fields,
    profile_segments,
    rates,
)
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import nodal_space, reference_basis, tabulate
from pymhm.meshes.triangle import TriangleMesh


def scalar_error_norms(
    meshes: Sequence[TriangleMesh],
    coefficients: Sequence[Any],
    degree: int,
    exact: Callable[[np.ndarray], np.ndarray],
    exact_gradient: Callable[[np.ndarray], np.ndarray],
    diffusion: float,
    *,
    order: int = 20,
) -> dict[str, float]:
    """Integrate broken scalar and raw physical-flux errors over all fine cells.

    ``order`` counts Gaussian points per Duffy coordinate. Flux is
    ``-diffusion*grad(u_h)``; no H(div) reconstruction or interface smoothing
    enters these norms. Nodal extrema and errors are reported separately.
    """
    bary, weights = triangle_quadrature(order)
    totals = np.zeros(3)
    maximum, minimum, nodal = -np.inf, np.inf, 0.0
    for mesh, values in zip(meshes, coefficients, strict=True):
        dofs, nodes, basis, gradient, _ = tabulate(mesh, degree, bary)
        points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells], optimize=True)
        truth = exact(points.reshape(-1, 2)).reshape(points.shape[:2])
        grad_truth = exact_gradient(points.reshape(-1, 2)).reshape(points.shape)
        if hasattr(values, "values_and_gradient"):
            # Explicit cell owners retain independent one-sided fine-cell gradients.
            owners = np.repeat(np.arange(len(mesh.cells)), len(bary))
            numerical, derivative = values.values_and_gradient(points.reshape(-1, 2), cells=owners)
            delta = numerical.reshape(points.shape[:2]) - truth
            grad_delta = derivative.reshape(points.shape) - grad_truth
            values = values.portable_coefficients
        else:
            # The independent classical reference supplies its declared nodal vector.
            delta = values[dofs] @ basis.T - truth
            grad_delta = (
                np.einsum("ti,tqia->tqa", values[dofs], gradient, optimize=True) - grad_truth
            )
        integrands = (delta**2, np.sum(grad_delta**2, axis=-1), truth**2)
        totals += [float(mesh.areas @ (value @ weights)) for value in integrands]
        maximum = max(maximum, float(values.max()))
        minimum = min(minimum, float(values.min()))
        nodal = max(nodal, float(np.max(np.abs(values - exact(nodes)))))
    scalar, derivative, norm = np.sqrt(totals)
    return {
        "scalar_l2": float(scalar),
        "flux_l2": float(diffusion * derivative),
        "gradient_l2": float(derivative),
        "scalar_relative_l2": float(scalar / norm),
        "nodal_error": nodal,
        "nodal_minimum": minimum,
        "nodal_maximum": maximum,
    }


def plot_rad_primary(
    cases: Any, rows: Any, reference_rows: Any, references: Any, truth: Any, *, reports: Path
) -> None:
    """Display measured fields and curves without selecting or solving a formulation."""
    plot_fields_local = partial(plot_fields, reports=reports)
    plot_errors_local = partial(plot_errors, reports=reports)
    macro, skeleton, meshes, fields = cases[16, "MHM-USFEM"]
    data_us = display_samples(meshes, fields, 1, subdivisions=3)
    data_gal = display_samples(
        cases[16, "MHM-Galerkin"][2], cases[16, "MHM-Galerkin"][3], 1, subdivisions=3
    )
    ref_mesh, ref_values = references[truth.epsilon, 128]
    data_ref = display_samples((ref_mesh,), (ref_values,), 2, subdivisions=1)
    plot_fields_local(
        macro,
        {
            "u exact": (data_ref["points"], data_ref["cells"], truth.value(data_ref["points"])),
            "u MHM-Galerkin": (data_gal["points"], data_gal["cells"], data_gal["values"]),
            "u MHM-USFEM": (data_us["points"], data_us["cells"], data_us["values"]),
            "u CG2 reference": (data_ref["points"], data_ref["cells"], data_ref["values"]),
            "error MHM-USFEM": (
                data_us["points"],
                data_us["cells"],
                data_us["values"] - truth.value(data_us["points"]),
            ),
            "flux magnitude MHM-USFEM": (
                data_us["points"],
                data_us["cells"],
                truth.epsilon * np.linalg.norm(data_us["gradient"], axis=1),
            ),
        },
        "rad-fields",
    )
    H = np.array([1 / n for n in (2, 4, 8, 16)])
    errors = {
        f"{method} {norm}": np.array([r[norm] for r in rows if r["method"] == method])
        for method in ("MHM-Galerkin", "MHM-USFEM")
        for norm in ("scalar_l2", "flux_l2")
    }
    plot_errors_local(H, errors, "rad-convergence")
    for name, values in errors.items():
        print(name, "rates:", rates(H, values))
    for epsilon in (truth.epsilon, 1e-5):
        selected = [r for r in reference_rows if r["epsilon"] == epsilon]
        plot_errors_local(
            np.array([1 / r["nx"] for r in selected]),
            {name: np.array([r[name] for r in selected]) for name in ("scalar_l2", "flux_l2")},
            f"rad-reference-eps{epsilon:g}",
        )


def plot_rad_severe(
    macro: TriangleMesh,
    challenging: Any,
    references: Any,
    sweep: Any,
    truth_factory: Callable[[float], Any],
    *,
    reports: Path,
) -> None:
    """Display measured fields and curves without selecting or solving a formulation."""
    plot_fields_local = partial(plot_fields, reports=reports)
    severe = 1e-5
    gm, gu = challenging[severe, "MHM-Galerkin"]
    ug = display_samples(gm, gu, 1, subdivisions=3)
    um, uu_values = challenging[severe, "MHM-USFEM"]
    uu = display_samples(um, uu_values, 1, subdivisions=3)
    rmesh, rvalues = references[severe, 1024]
    ur = display_samples((rmesh,), (rvalues,), 2, subdivisions=1)
    plot_fields_local(
        macro,
        {
            "severe layer: u exact": (
                ur["points"],
                ur["cells"],
                truth_factory(severe).value(ur["points"]),
            ),
            "severe layer: MHM-Galerkin": (ug["points"], ug["cells"], ug["values"]),
            "severe layer: MHM-USFEM": (uu["points"], uu["cells"], uu["values"]),
            "severe layer: CG2 reference": (ur["points"], ur["cells"], ur["values"]),
            "severe layer: Galerkin scalar error": (
                ug["points"],
                ug["cells"],
                ug["values"] - truth_factory(severe).value(ug["points"]),
            ),
            "severe layer: USFEM scalar error": (
                uu["points"],
                uu["cells"],
                uu["values"] - truth_factory(severe).value(uu["points"]),
            ),
        },
        "rad-severe-fields",
    )
    for row in sweep:
        if row["epsilon"] == severe:
            print(
                row["method"],
                "overshoot =",
                row["overshoot"],
                "maximum nodal error =",
                row["nodal_error"],
                "scalar L2 =",
                row["scalar_l2"],
            )


def plot_rad_refined(
    resolved_cases: Any,
    resolved_rows: Any,
    references: Any,
    truth_factory: Callable[[float], Any],
    *,
    reports: Path,
) -> None:
    """Display measured fields and curves without selecting or solving a formulation."""
    plot_fields_local = partial(plot_fields, reports=reports)
    plot_errors_local = partial(plot_errors, reports=reports)
    rmesh, rvalues = references[1e-5, 1024]
    ur = display_samples((rmesh,), (rvalues,), 2, subdivisions=1)
    refined_macro = resolved_cases[1e-5, 16, "MHM-USFEM"][0]
    gm, gu = resolved_cases[1e-5, 16, "MHM-Galerkin"][2:]
    resolved_gal = display_samples(gm, gu, 1, subdivisions=2)
    um, uu_values = resolved_cases[1e-5, 16, "MHM-USFEM"][2:]
    resolved_us = display_samples(um, uu_values, 1, subdivisions=2)
    plot_fields_local(
        refined_macro,
        {
            "refined layer: u exact": (
                ur["points"],
                ur["cells"],
                truth_factory(1e-5).value(ur["points"]),
            ),
            "refined layer: MHM-Galerkin": (
                resolved_gal["points"],
                resolved_gal["cells"],
                resolved_gal["values"],
            ),
            "refined layer: MHM-USFEM": (
                resolved_us["points"],
                resolved_us["cells"],
                resolved_us["values"],
            ),
            "refined layer: CG2 reference": (ur["points"], ur["cells"], ur["values"]),
            "refined layer: Galerkin scalar error": (
                resolved_gal["points"],
                resolved_gal["cells"],
                resolved_gal["values"] - truth_factory(1e-5).value(resolved_gal["points"]),
            ),
            "refined layer: USFEM scalar error": (
                resolved_us["points"],
                resolved_us["cells"],
                resolved_us["values"] - truth_factory(1e-5).value(resolved_us["points"]),
            ),
        },
        "rad-resolved-fields",
    )
    gal_flux = -1e-5 * resolved_gal["gradient"]
    us_flux = -1e-5 * resolved_us["gradient"]
    plot_fields_local(
        refined_macro,
        {
            "Exact flux magnitude": (
                ur["points"],
                ur["cells"],
                1e-5 * np.linalg.norm(truth_factory(1e-5).gradient(ur["points"]), axis=1),
            ),
            "CG2 flux magnitude": (
                ur["points"],
                ur["cells"],
                1e-5 * np.linalg.norm(ur["gradient"], axis=1),
            ),
            "MHM-Galerkin flux magnitude": (
                resolved_gal["points"],
                resolved_gal["cells"],
                np.linalg.norm(gal_flux, axis=1),
            ),
            "MHM-USFEM flux magnitude": (
                resolved_us["points"],
                resolved_us["cells"],
                np.linalg.norm(us_flux, axis=1),
            ),
            "Galerkin flux error magnitude": (
                resolved_gal["points"],
                resolved_gal["cells"],
                np.linalg.norm(
                    gal_flux + 1e-5 * truth_factory(1e-5).gradient(resolved_gal["points"]), axis=1
                ),
            ),
            "USFEM flux error magnitude": (
                resolved_us["points"],
                resolved_us["cells"],
                np.linalg.norm(
                    us_flux + 1e-5 * truth_factory(1e-5).gradient(resolved_us["points"]), axis=1
                ),
            ),
        },
        "rad-resolved-flux-fields",
    )
    refined_H = np.array([1 / resolution for resolution in (4, 8, 16)])
    refined_errors = {
        f"{method} {norm}": np.array(
            [
                row[norm]
                for row in resolved_rows
                if row["epsilon"] == 1e-5 and row["method"] == method
            ]
        )
        for method in ("MHM-Galerkin", "MHM-USFEM")
        for norm in ("scalar_l2", "flux_l2")
    }
    plot_errors_local(refined_H, refined_errors, "rad-resolved-convergence")
    for label, error in refined_errors.items():
        print(label, "rates:", rates(refined_H, error))


def plot_rad_profiles(
    macro: TriangleMesh,
    challenging: Any,
    resolved_cases: Any,
    references: Any,
    truth_factory: Callable[[float], Any],
    *,
    epsilon_primary: float,
    underresolved_subdivisions: int = 2,
    face_subdivisions: int = 4,
    refined_subdivisions: int = 8,
    reports: Path,
) -> None:
    """Display measured fields and curves without selecting or solving a formulation."""
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for column, epsilon in enumerate((epsilon_primary, 1e-5)):
        for row, family in enumerate(("underresolved", "refined")):
            ax = axes[row, column]
            x = np.linspace(0, 1, 2001)
            ax.plot(
                x,
                truth_factory(epsilon).value(np.column_stack((x, x * 0 + 0.37))),
                "k--",
                label="exact",
            )
            if family == "underresolved":
                profile_macro = macro
                parameters = (
                    f"macro n=8; face segments=1; local subdivisions={underresolved_subdivisions}"
                )
            else:
                profile_macro = resolved_cases[epsilon, 16, "MHM-USFEM"][0]
                parameters = (
                    f"macro n=16; face segments={face_subdivisions}; "
                    f"local subdivisions={refined_subdivisions}"
                )
            crossings: set[float] = set()
            for method in ("MHM-Galerkin", "MHM-USFEM"):
                meshes, fields = (
                    challenging[epsilon, method]
                    if family == "underresolved"
                    else resolved_cases[epsilon, 16, method][2:]
                )
                for index, (position, values) in enumerate(
                    profile_segments(profile_macro, meshes, fields, 1)
                ):
                    ax.plot(
                        position,
                        values,
                        color={"MHM-Galerkin": "tab:blue", "MHM-USFEM": "tab:orange"}[method],
                        label=method if index == 0 else None,
                    )
                    crossings.update((float(position[0]), float(position[-1])))
            for cross in sorted(crossings):
                ax.axvline(cross, color=".7", alpha=0.6, linewidth=0.5)
            rmesh, rvalues = references[epsilon, 128 if epsilon == epsilon_primary else 1024]
            locations = np.asarray(
                Triangulation(
                    rmesh.points[:, 0], rmesh.points[:, 1], triangles=rmesh.cells
                ).get_trifinder()(x, np.full_like(x, 0.37)),
                dtype=np.int64,
            )
            vertices = rmesh.points[rmesh.cells[locations]]
            inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
            local = np.einsum(
                "qij,qj->qi", inverse, np.column_stack((x, x * 0 + 0.37)) - vertices[:, 0]
            )
            bary = np.column_stack((1 - local.sum(axis=1), local))
            dofs, _ = nodal_space(rmesh, 2)
            reference_profile = np.einsum(
                "qi,qi->q", reference_basis(2, bary)[0], rvalues[dofs[locations]]
            )
            ax.plot(
                x, reference_profile, linestyle=":", color="tab:green", label="fine CG2 reference"
            )
            ax.set(
                xlabel="x at y=0.37",
                ylabel="u",
                title=f"{family}; epsilon={epsilon:g}\n{parameters}",
            )
            ax.legend(fontsize=8)
    fig.savefig(reports / "rad-layer-profiles.png", dpi=160)
    plt.show()


def scalar_reference(
    epsilon: float,
    nx: int,
    ny: int,
    form_factory: Callable[[Any, float], tuple[Any, Any, Any, Any]],
    exact: Any,
    *,
    reports: Path,
    root: Path,
) -> tuple[tuple[TriangleMesh, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Acquire a conforming P2 solution of the supplied reaction-layer UFL forms.

    The notebook owns operator, source, exact field and quadrature. This helper
    imposes its stated zero vertical nodal data and leaves horizontal data
    natural, uses the generic global-equation API and records physical norms.
    """
    import basix.ufl
    import dolfinx
    import ufl

    from pymhm import Equation, MultiscaleProblem, assemble
    from pymhm.backends.spaces import bind_space, create_native_mesh
    from pymhm.core.equations import compile_form

    mesh = TriangleMesh.unit_square(nx, ny)
    domain = create_native_mesh(mesh)
    space = dolfinx.fem.functionspace(domain, basix.ufl.element("Lagrange", "triangle", 2))
    binding = bind_space(mesh, space)
    try:
        a, load, truth, dx = form_factory(space, epsilon)
        matrix, forcing = compile_form(a), compile_form(load)
        coordinates = space.tabulate_dof_coordinates()[:, :2]
        fixed = np.flatnonzero(
            np.isclose(coordinates[:, 0], 0, atol=1e-12)
            | np.isclose(coordinates[:, 0], 1, atol=1e-12)
        )
        problem = MultiscaleProblem.from_global(
            Equation(matrix, forcing), len(forcing), fixed=dict.fromkeys(fixed, 0.0)
        )
        coefficients = assemble(problem).solve().trace
        numerical = dolfinx.fem.Function(space)
        numerical.x.array[:] = coefficients
        difference = numerical - truth
        scalar = float(np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(difference**2 * dx))))
        gradient = float(
            np.sqrt(
                dolfinx.fem.assemble_scalar(
                    dolfinx.fem.form(ufl.inner(ufl.grad(difference), ufl.grad(difference)) * dx)
                )
            )
        )
        _, nodes = nodal_space(mesh, 2)
        nodal = binding.to_portable(coefficients)
        metrics = dict(
            epsilon=epsilon,
            nx=nx,
            ny=ny,
            triangles=len(mesh.cells),
            scalar_l2=scalar,
            flux_l2=epsilon * gradient,
            gradient_l2=gradient,
            nodal_error=float(np.max(abs(nodal - exact.value(nodes)))),
        )
        state = reports / f"rad-reference-eps{epsilon:g}-{nx}x{ny}-state.npz"
        np.savez_compressed(
            state,
            points=mesh.points,
            cells=mesh.cells,
            scalar_coefficients=nodal,
            scalar_degree=np.array(2),
            epsilon=np.array(epsilon),
        )
        archive = dict(
            path=str(state.relative_to(root)),
            sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
            basis_convention="Basix canonical equispaced P2 nodal coefficients",
        )
        print("CG2 reference", epsilon, (nx, ny), scalar, epsilon * gradient)
        return (mesh, nodal), metrics, archive
    finally:
        binding.close()


def record_rad_case(
    macro: TriangleMesh,
    skeleton: Any,
    system: Any,
    solution: Any,
    exact: Any,
    *,
    order: int,
    name: str,
    reports: Path,
    root: Path,
    named: bool = False,
    **metadata: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Measure and archive an already solved scalar case, using its executed coefficients."""
    from examples.introduction.vector import preserve_state

    fields = solution.field("scalar")
    meshes = tuple(field.mesh for field in fields)
    coefficients = tuple(field.portable_coefficients for field in fields)
    metrics = scalar_error_norms(
        meshes,
        fields if named else coefficients,
        1,
        exact.value,
        exact.gradient,
        exact.epsilon,
        order=order,
    )
    row = dict(
        macro_cells=len(macro.cells),
        fine_cells=sum(len(m.cells) for m in meshes),
        trace_dofs=skeleton.size,
        residual=float(solution.residual),
        **metrics,
        **metadata,
    )
    archive = preserve_state(name, skeleton, system, solution, reports=reports, root=root)
    return (macro, skeleton, meshes, coefficients), row, archive
