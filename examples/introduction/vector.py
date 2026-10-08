"""Physical field measurements and displays for the vector introduction notebooks.

Operators are supplied by notebook formulas. These helpers integrate physical
fields, record executed coordinates and plot independently sampled macro sides.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from pymhm.fem.reference import physical_simplex_tabulation
from pymhm.fem.scalar.operators import p1_geometry, triangle_quadrature
from pymhm.fem.scalar.triangle import multiindices, nodal_space, reference_basis
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.fields import DiscreteField
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.sampling import TrianglePointLocator

Array = NDArray[np.float64]


def evaluate_named_displacement(field: Any, points: Array) -> tuple[Array, Array]:
    """Read named displacement and its raw physical Jacobian from the executed space."""
    locator = TrianglePointLocator(field.mesh)
    return field.values_and_gradient(points, cells=locator.locate(points))


# The explicit Basix evaluator below also supplies the independent classical reference.
@dataclass
class TriangleVectorEvaluator:
    """Evaluate actual continuous Pk vector coefficients through their recorded nodal basis."""

    mesh: TriangleMesh
    degree: int
    coefficients: Array

    def __post_init__(self) -> None:
        """Capture the nodal basis and a validated geometry-only point locator."""
        self.locator = TrianglePointLocator(self.mesh)
        self.field = DiscreteField(
            nodal_field("vector", self.mesh, self.degree, components=2), self.coefficients.ravel()
        )

    def __call__(self, points: Array) -> tuple[Array, Array]:
        """Evaluate a validated incident fine cell, without averaging an interface."""
        return self.field.values_and_gradient(points, cells=self.locator.locate(points))


@dataclass
class BrokenVectorEvaluator:
    """Evaluate independent macrocell vectors, retaining one-sided interface values."""

    macro: TriangleMesh
    local_evaluators: tuple[Callable[[Array], tuple[Array, Array]], ...]

    def __post_init__(self) -> None:
        """Cache the actual macrogeometry without constructing a fictitious field."""
        self.locator = TrianglePointLocator(self.macro)

    def __call__(self, points: Array) -> tuple[Array, Array]:
        """Read the independently supplied evaluator of each validated macro owner."""
        owners = self.locator.locate(points)
        value, gradient = np.empty((len(points), 2)), np.empty((len(points), 2, 2))
        for owner in np.unique(owners):
            selected = owners == owner
            value[selected], gradient[selected] = self.local_evaluators[owner](points[selected])
        return value, gradient


def cauchy_stress(points: Array, gradient: Array, *, modulus: Callable[[Array], Array]) -> Array:
    """Return symmetric plane-strain stress with lambda=mu=micro_modulus."""
    values = modulus(points)
    return values[:, None, None] * (
        gradient
        + gradient.swapaxes(1, 2)
        + np.trace(gradient, axis1=1, axis2=2)[:, None, None] * np.eye(2)
    )


def elasticity_errors(
    first: Any,
    second: Any,
    points: Array,
    weights: Array,
    *,
    modulus: Callable[[Array], Array],
    batch_size: int = 32768,
) -> dict[str, float]:
    """Integrate physical displacement L2, stress Frobenius L2 and strain-energy differences."""
    sums = np.zeros(6)
    for start in range(0, len(points), batch_size):
        selection = slice(start, start + batch_size)
        x, w = points[selection], weights[selection]
        u, grad = first(x)
        target, gradref = second(x)
        delta = grad - gradref
        strain_delta = (delta + delta.swapaxes(1, 2)) / 2
        strain_ref = (gradref + gradref.swapaxes(1, 2)) / 2
        stress_delta, stress_ref = (
            cauchy_stress(x, delta, modulus=modulus),
            cauchy_stress(x, gradref, modulus=modulus),
        )
        sums += np.asarray(
            [
                w @ np.sum((u - target) ** 2, axis=1),
                w @ np.sum(stress_delta**2, axis=(1, 2)),
                w @ np.sum(stress_delta * strain_delta, axis=(1, 2)),
                w @ np.sum(target**2, axis=1),
                w @ np.sum(stress_ref**2, axis=(1, 2)),
                w @ np.sum(stress_ref * strain_ref, axis=(1, 2)),
            ]
        )
    norms = np.sqrt(sums)
    return dict(
        displacement_L2=float(norms[0]),
        stress_L2=float(norms[1]),
        energy=float(norms[2]),
        displacement_relative=float(norms[0] / norms[3]),
        stress_relative=float(norms[1] / norms[4]),
        energy_relative=float(norms[2] / norms[5]),
    )


def elasticity_panel(
    macro: TriangleMesh,
    evaluator: Any,
    quantity: str,
    refinement: int = 16,
    *,
    modulus: Callable[[Array], Array],
) -> tuple[Array, NDArray[np.int64], Array]:
    """Sample each macro triangle separately for displacement, stress or material panels."""
    points, triangles, values = [], [], []
    count = 0
    for cell in range(len(macro.cells)):
        local = macro.submesh(cell, refinement)
        physical = local.points
        center = macro.points[macro.cells[cell]].mean(axis=0)
        inside = physical + 1e-8 * (center - physical)
        if quantity == "modulus":
            value = modulus(inside)
        else:
            side_evaluator = (
                evaluator.local_evaluators[cell]
                if isinstance(evaluator, BrokenVectorEvaluator)
                else evaluator
            )
            displacement, gradient = side_evaluator(inside)
            if quantity == "displacement_x":
                value = displacement[:, 0]
            elif quantity == "displacement_magnitude":
                value = np.linalg.norm(displacement, axis=1)
            elif quantity == "stress_xx":
                value = cauchy_stress(inside, gradient, modulus=modulus)[:, 0, 0]
            elif quantity == "stress_magnitude":
                value = np.linalg.norm(
                    cauchy_stress(inside, gradient, modulus=modulus), axis=(1, 2)
                )
            else:
                raise ValueError("unknown elasticity field quantity")
        points.append(physical)
        triangles.append(local.cells + count)
        values.append(value)
        count += len(physical)
    return np.vstack(points), np.vstack(triangles), np.concatenate(values)


def triangle_grid_quadrature(n: int, order: int) -> tuple[Array, Array]:
    """Integrate on actual common fine triangles, resolving gradient jumps."""
    common = TriangleMesh.unit_square(n)
    bary, weights = triangle_quadrature(order)
    points = np.einsum("qi,tia->tqa", bary, common.points[common.cells])
    return points.reshape(-1, 2), (common.areas[:, None] * weights).reshape(-1)


def dirichlet_solve(matrix: Any, load: Array, dofs: NDArray[np.int64], values: Array) -> Array:
    """Solve the supplied global equation with explicitly prescribed nodal coordinates."""
    from pymhm import Equation, MultiscaleProblem, assemble

    problem = MultiscaleProblem.from_global(
        Equation(matrix, load), len(load), fixed=dict(zip(dofs, values, strict=True))
    )
    return assemble(problem).solve().trace


def plot_field_panels(
    macro_mesh: Any,
    panels: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Plot independent nodal scalar panels and their actual macrofaces.

    Each panel supplies physical points, its explicit triangular connectivity
    and values. Duplicate coordinates are retained, so broken one-sided fields
    are never averaged across a macroface. Each field has its own color scale.
    """
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    count = len(panels)
    if not count:
        raise ValueError("provide at least one field panel")
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    size = figsize or (4.1 * columns, 3.6 * rows)
    figure, axes = plt.subplots(rows, columns, figsize=size, squeeze=False, layout="constrained")
    for axis, (label, (points, triangles, values)) in zip(axes.flat, panels.items(), strict=False):
        coordinates = np.asarray(points)
        triangulation = Triangulation(coordinates[:, 0], coordinates[:, 1], triangles=triangles)
        artist = axis.tripcolor(triangulation, values, shading="gouraud", rasterized=True)
        axis.add_collection(
            LineCollection(
                macro_mesh.points[macro_mesh.faces], colors="0.2", linewidths=0.65, zorder=3
            )
        )
        axis.set(title=label, xlabel="x", ylabel="y", aspect="equal")
        figure.colorbar(artist, ax=axis, shrink=0.87, pad=0.025)
    for axis in list(axes.flat)[count:]:
        axis.set_visible(False)
    return figure


def observed_rates(mesh_sizes: Any, errors: Any) -> np.ndarray:
    """Return log(error[i]/error[i+1])/log(H[i]/H[i+1]) without assumed orders."""
    h, e = np.asarray(mesh_sizes, dtype=float), np.asarray(errors, dtype=float)
    if h.ndim != 1 or e.shape != h.shape or len(h) < 2:
        raise ValueError("provide at least two matching refinement levels")
    if np.any(h <= 0) or np.any(np.diff(h) >= 0) or np.any(e <= 0):
        raise ValueError("mesh sizes must decrease and measured errors must be positive")
    return np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])


def plot_convergence(mesh_sizes: Any, errors: Mapping[str, Any]) -> Any:
    """Plot actual errors and observed successive rates in separate readable axes."""
    import matplotlib.pyplot as plt

    h = np.asarray(mesh_sizes, dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for label, values in errors.items():
        e = np.asarray(values, dtype=float)
        axes[0].loglog(h, e, "o-", label=label)
        axes[1].semilogx(h[1:], observed_rates(h, e), "o-", label=label)
    axes[0].set(xlabel="H", ylabel="Measured error")
    axes[1].set(xlabel="H", ylabel="Observed rate")
    for axis in axes:
        axis.invert_xaxis()
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    return figure


def elasticity_reference(
    resolution: int,
    form_factory: Callable[[Any, int], Any],
    boundary: Callable[[Array], Array],
    *,
    degree: int = 2,
) -> tuple[TriangleVectorEvaluator, dict[str, int]]:
    """Assemble the notebook-supplied classical UFL energy and strong boundary data.

    The supplied form_factory owns the physical operator and quadrature.
    This acquisition helper owns the conforming mesh, coefficient map and
    generic global solve. It does not select a multiscale physical solver.
    """
    import basix
    import basix.ufl

    from pymhm.backends.spaces import bind_space
    from pymhm.core.equations import compile_form

    mesh = TriangleMesh.unit_square(resolution)
    element = basix.ufl.element(
        "Lagrange",
        "triangle",
        degree,
        lagrange_variant=basix.LagrangeVariant.equispaced,
        shape=(2,),
    )
    binding = bind_space(mesh, element)
    try:
        mapping = binding.mapping
        matrix = compile_form(form_factory(binding.space, resolution))
        _, nodes = nodal_space(mesh, degree)
        exterior = np.flatnonzero(np.any(np.isclose(nodes, 0) | np.isclose(nodes, 1), axis=1))
        portable = (2 * exterior[:, None] + np.arange(2)).ravel()
        coefficients = dirichlet_solve(
            matrix, np.zeros(len(mapping)), mapping[portable], boundary(nodes[exterior]).ravel()
        )
        evaluator = TriangleVectorEvaluator(mesh, degree, coefficients[mapping].reshape(-1, 2))
        return evaluator, dict(
            square_grid=resolution, triangles=len(mesh.cells), displacement_unknowns=len(mapping)
        )
    finally:
        binding.close()


def plot_elasticity_fields(
    macro: TriangleMesh,
    mhm: BrokenVectorEvaluator,
    reference: TriangleVectorEvaluator,
    coarse: TriangleVectorEvaluator,
    modulus: Callable[[Array], Array],
) -> Any:
    """Show the nine original material, displacement and Cauchy-stress panels."""

    def difference(points: Array) -> tuple[Array, Array]:
        """Read the physical displacement and gradient difference."""
        first, second = mhm(points), reference(points)
        return first[0] - second[0], first[1] - second[1]

    specifications = (
        ("Lamé modulus (lambda = mu)", reference, "modulus"),
        ("MHM displacement x", mhm, "displacement_x"),
        ("Reference displacement x", reference, "displacement_x"),
        ("MHM Cauchy stress xx", mhm, "stress_xx"),
        ("Reference Cauchy stress xx", reference, "stress_xx"),
        ("Coarse Galerkin stress xx", coarse, "stress_xx"),
        ("Displacement difference magnitude", difference, "displacement_magnitude"),
        ("MHM Cauchy stress magnitude", mhm, "stress_magnitude"),
        ("Reference stress magnitude", reference, "stress_magnitude"),
    )
    return plot_field_panels(
        macro,
        {
            label: elasticity_panel(macro, evaluator, quantity, 32, modulus=modulus)
            for label, evaluator, quantity in specifications
        },
        figsize=(15, 13),
    )


def nodal_evaluation(
    mesh: TriangleMesh, degree: int, bary: np.ndarray, *, gradient: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate declared nodal coordinates through the package's native Basix kernel."""
    dofs, _ = nodal_space(mesh, degree)
    geometry, _ = p1_geometry(mesh)
    values, first, _ = physical_simplex_tabulation(
        "triangle",
        degree,
        bary,
        nodes=multiindices(degree) / degree,
        reference_gradients=geometry[:, 1:],
        nderiv=1 if gradient else 0,
    )
    return dofs, values, first


def flow_error_norms(
    meshes: Sequence[TriangleMesh],
    velocity: Sequence[np.ndarray],
    pressure: Sequence[np.ndarray],
    velocity_degree: int,
    pressure_degree: int,
    exact: Any,
    *,
    order: int = 20,
    named_velocity: Sequence[Any] | None = None,
    named_pressure: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Integrate velocity, pressure, gradient and macro mass measurements.

    Pressure must already use the zero-volume-mean gauge. Macro mass is the
    largest absolute macro integral of div(u_h); ``divergence_l2`` measures
    the different fine-cell incompressibility defect. Raw gradients and
    pseudostress are broken fields, without a conservative reconstruction.
    """
    bary, weights = triangle_quadrature(order)
    totals = np.zeros(5)
    mass, mean = 0.0, 0.0
    for cell, (mesh, u, p) in enumerate(zip(meshes, velocity, pressure, strict=True)):
        udofs, ubasis, gradients = nodal_evaluation(mesh, velocity_degree, bary)
        pdofs, pbasis, _ = nodal_evaluation(mesh, pressure_degree, bary, gradient=False)
        points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells], optimize=True)
        flat = points.reshape(-1, 2)
        if named_velocity is not None and named_pressure is not None:
            owners = np.repeat(np.arange(len(mesh.cells)), len(bary))
            numerical_u, numerical_gradient = named_velocity[cell].values_and_gradient(
                flat, cells=owners
            )
            delta_u = numerical_u.reshape(points.shape) - exact.velocity(flat).reshape(points.shape)
            numerical_p = (
                named_pressure[cell].evaluate(flat, cells=owners).reshape(points.shape[:2])
            )
            derivative = numerical_gradient.reshape((*points.shape[:2], 2, 2))
        else:
            # Explicit coefficient evaluation for independent references and basis replay.
            delta_u = np.einsum("qi,tia->tqa", ubasis, u[udofs], optimize=True) - exact.velocity(
                flat
            ).reshape(points.shape)
            numerical_p = p[pdofs] @ pbasis.T
            derivative = np.einsum("tqia,tic->tqca", gradients, u[udofs], optimize=True)
        delta_p = numerical_p - exact.pressure(flat).reshape(points.shape[:2])
        delta_gradient = derivative - exact.gradient(flat).reshape(derivative.shape)
        divergence = np.trace(derivative, axis1=-2, axis2=-1)
        delta_stress = exact.viscosity * delta_gradient - delta_p[..., None, None] * np.eye(2)
        integrands = (
            np.sum(delta_u**2, axis=-1),
            delta_p**2,
            np.sum(delta_gradient**2, axis=(-2, -1)),
            divergence**2,
            np.sum(delta_stress**2, axis=(-2, -1)),
        )
        totals += [float(mesh.areas @ (value @ weights)) for value in integrands]
        mass = max(mass, abs(float(mesh.areas @ (divergence @ weights))))
        mean += float(mesh.areas @ (numerical_p @ weights))
    errors = np.sqrt(totals)
    return dict(
        velocity_l2=float(errors[0]),
        pressure_l2=float(errors[1]),
        velocity_h1_seminorm=float(errors[2]),
        divergence_l2=float(errors[3]),
        pseudostress_l2=float(errors[4]),
        macro_mass_defect=mass,
        pressure_integral=mean,
    )


def preserve_state(
    name: str, skeleton: SkeletonSpace, system: Any, solution: Any, *, reports: Path, root: Path
) -> dict[str, Any]:
    """Archive executed coefficients, orientation, local lifts and actual retained bases."""
    payload = {
        "macro_points": skeleton.mesh.points,
        "macro_cells": skeleton.mesh.cells,
        "macro_faces": skeleton.mesh.faces,
        "macro_signs": skeleton.mesh.signs,
        "trace": solution.trace,
        "skeleton_components": np.array(skeleton.components),
    }
    basis_digest = hashlib.sha256()
    for face, space in enumerate(skeleton.faces):
        payload[f"face_breaks_{face}"] = np.asarray(space.breaks)
        payload[f"face_degrees_{face}"] = np.asarray(space.degrees)
    for cell, (response, data, field, coarse) in enumerate(
        zip(
            system.responses,
            system.local_metadata,
            solution.fields,
            solution.coarse,
            strict=True,
        )
    ):
        payload[f"local_points_{cell}"] = data["mesh"].points
        payload[f"local_cells_{cell}"] = data["mesh"].cells
        payload[f"field_{cell}"] = field
        payload[f"coarse_{cell}"] = coarse
        payload[f"source_{cell}"] = response.source
        payload[f"lifts_{cell}"] = response.lifts
        payload[f"basis_{cell}"] = response.retained_basis
        payload[f"trace_dofs_{cell}"] = response.problem.trace_dofs
        if "free" in data:
            payload[f"free_nodes_{cell}"] = data["free"]
        if "velocity_nodes" in data:
            payload[f"velocity_nodes_{cell}"] = np.array(data["velocity_nodes"])
            payload[f"velocity_degree_{cell}"] = np.array(data["velocity_degree"])
            payload[f"pressure_degree_{cell}"] = np.array(data["pressure_degree"])
        else:
            payload[f"scalar_degree_{cell}"] = np.array(1)
        basis_digest.update(str(response.retained_basis.shape).encode())
        basis_digest.update(response.retained_basis.tobytes())
    path = reports / f"{name}-state.npz"
    np.savez_compressed(path, **payload)
    # Replay this executed basis literally at two BLAS thread counts.
    with np.load(path) as saved:
        for count in (1, 2):
            with threadpool_limits(count):
                for cell in range(len(system.responses)):
                    reconstructed = (
                        saved[f"source_{cell}"]
                        - saved[f"lifts_{cell}"] @ saved["trace"][saved[f"trace_dofs_{cell}"]]
                        + saved[f"basis_{cell}"] @ saved[f"coarse_{cell}"]
                    )
                    np.testing.assert_allclose(
                        reconstructed, saved[f"field_{cell}"], rtol=1e-11, atol=1e-12
                    )
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "executed_basis_sha256": basis_digest.hexdigest(),
        "replay_blas_threads": [1, 2],
    }


def display_samples(
    meshes: Sequence[TriangleMesh],
    fields: Sequence[np.ndarray],
    degree: int,
    subdivisions: int = 2,
) -> dict:
    """Evaluate Basix polynomials without merging any incident macro traces."""
    template = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, subdivisions)
    bary = np.column_stack((1 - template.points.sum(axis=1), template.points))
    points, cells, values, derivatives = [], [], [], []
    offset = 0
    for mesh, field in zip(meshes, fields, strict=True):
        dofs, phi, gradients = nodal_evaluation(mesh, degree, bary)
        coordinates = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
        value = np.einsum("qi,ti...->tq...", phi, field[dofs])
        gradient = np.einsum("tqia,ti...->tq...a", gradients, field[dofs])
        points.append(coordinates.reshape(-1, 2))
        cells.append(
            (
                template.cells[None]
                + offset
                + np.arange(len(mesh.cells))[:, None, None] * len(bary)
            ).reshape(-1, 3)
        )
        values.append(value.reshape((-1, *value.shape[2:])))
        derivatives.append(gradient.reshape((-1, *gradient.shape[2:])))
        offset += len(mesh.cells) * len(bary)
    return dict(
        points=np.concatenate(points),
        cells=np.concatenate(cells),
        values=np.concatenate(values),
        gradient=np.concatenate(derivatives),
    )


def plot_fields(macro: TriangleMesh, panels: dict, name: str, *, reports: Path) -> None:
    """Show separate one-sided display arrays and each actual macroface."""
    columns = min(3, len(panels))
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.3 * columns, 3.7 * rows),
        squeeze=False,
        layout="constrained",
    )
    for ax, (label, data) in zip(axes.flat, panels.items(), strict=False):
        points, cells, values = data
        artist = ax.tripcolor(
            Triangulation(points[:, 0], points[:, 1], triangles=cells),
            values,
            shading="gouraud",
            rasterized=True,
        )
        ax.add_collection(
            LineCollection(
                macro.points[macro.faces].tolist(), colors=".25", linewidths=0.35, alpha=0.7
            )
        )
        ax.set(title=label, xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(artist, ax=ax, shrink=0.85, pad=0.025)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    fig.savefig(reports / f"{name}.png", dpi=160)
    plt.show()


def rates(H: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Compute successive measured slopes without imposing a theoretical order."""
    return np.log(errors[:-1] / errors[1:]) / np.log(H[:-1] / H[1:])


def plot_errors(H: np.ndarray, errors: dict, name: str, *, reports: Path) -> None:
    """Show measured norms and successive rates in separate axes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for label, error in errors.items():
        axes[0].loglog(H, error, "o-", label=label)
        axes[1].semilogx(H[1:], rates(H, error), "o-", label=label)
    axes[0].set(xlabel="H", ylabel="Absolute error")
    axes[1].set(xlabel="H", ylabel="Observed rate")
    for ax, samples in zip(axes, (H, H[1:]), strict=True):
        ax.set_xticks(samples, labels=[f"{value:.3g}" for value in samples])
        ax.tick_params(axis="x", which="minor", labelbottom=False)
        ax.invert_xaxis()
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(reports / f"{name}.png", dpi=160)
    plt.show()


def evaluate_incident(
    mesh: TriangleMesh, field: np.ndarray, degree: int, points: np.ndarray
) -> np.ndarray:
    """Evaluate from this specified macrocell, including its own boundary limits."""
    incident, chosen = TrianglePointLocator(mesh).coordinates(points)
    dofs, _ = nodal_space(mesh, degree)
    phi = reference_basis(degree, chosen)[0]
    return np.einsum("qi,qi...->q...", phi, field[dofs[incident]])


def profile_segments(
    macro: TriangleMesh,
    meshes: Sequence[TriangleMesh],
    fields: Sequence[np.ndarray],
    degree: int,
    height: float = 0.37,
) -> list:
    """Return separate horizontal segments; each endpoint retains its incident value."""
    segments = []
    for cell, vertices in enumerate(macro.points[macro.cells]):
        intersections = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (vertices[i, 1] - height) * (vertices[j, 1] - height) < 0:
                fraction = (height - vertices[i, 1]) / (vertices[j, 1] - vertices[i, 1])
                intersections.append(vertices[i, 0] + fraction * (vertices[j, 0] - vertices[i, 0]))
        if len(intersections) == 2:
            x = np.linspace(min(intersections), max(intersections), 101)
            points = np.column_stack((x, np.full_like(x, height)))
            segments.append((x, evaluate_incident(meshes[cell], fields[cell], degree, points)))
    return segments


def execution_provenance(notebook: str, *, root: Path, **metadata: Any) -> dict[str, Any]:
    """Record current notebook/lock identities and installed numerical-library versions."""
    import importlib.metadata

    from examples.introduction.provenance import source_digests

    support = tuple(
        root / "examples/introduction" / name
        for name in ("vector.py", "transport.py", "provenance.py")
    )
    owners = tuple((root / "src/pymhm").rglob("*.py"))
    return {
        "notebook": notebook,
        "notebook_sha256": hashlib.sha256((root / notebook).read_bytes()).hexdigest(),
        "pixi_lock_sha256": hashlib.sha256((root / "pixi.lock").read_bytes()).hexdigest(),
        "support_sha256": source_digests(root, support),
        "source_sha256": source_digests(root, owners),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "fenics-basix", "fenics-dolfinx", "pymhm")
        },
        "basis_convention": (
            "Basix equispaced Pk in PyMHM nodal_space order; no local nullspace modes"
        ),
        "reference_project": "DOLFINx",
        "reference_source_url": "https://docs.fenicsproject.org/dolfinx/v0.9.0/python/",
        **metadata,
    }


def brinkman_reference(
    resolution: int,
    form_factory: Callable[[Any], tuple[Any, Any, Any, Any, Any, Any]],
    exact: Any,
    *,
    reports: Path,
    root: Path,
) -> tuple[tuple[TriangleMesh, Array, Array], dict[str, Any], dict[str, Any]]:
    """Acquire a conforming P2/P1 solution of notebook-supplied mixed UFL forms.

    The notebook owns operator, source, exact fields, pressure moment and
    quadrature. Full exterior velocity is imposed on nodal coefficients, and
    the supplied physical pressure integral is constrained to zero. Assembly
    and boundary/constraint elimination use the generic public equation API.
    """
    import basix.ufl
    import dolfinx
    import ufl

    from pymhm import Equation, MultiscaleProblem, assemble
    from pymhm.backends.spaces import bind_space, create_native_mesh
    from pymhm.core.equations import compile_form

    mesh = TriangleMesh.unit_square(resolution)
    domain = create_native_mesh(mesh)
    element = basix.ufl.mixed_element(
        [
            basix.ufl.element("Lagrange", "triangle", 2, shape=(2,)),
            basix.ufl.element("Lagrange", "triangle", 1),
        ]
    )
    space = dolfinx.fem.functionspace(domain, element)
    binding = bind_space(mesh, space)
    try:
        a, load, pressure_form, truth_u, truth_p, dx = form_factory(space)
        matrix, forcing, moment = compile_form(a), compile_form(load), compile_form(pressure_form)
        velocity_space, vmap = space.sub(0).collapse()
        coordinates = velocity_space.tabulate_dof_coordinates()[:, :2]
        boundary = np.flatnonzero(
            np.any(
                np.isclose(coordinates, 0, atol=1e-12) | np.isclose(coordinates, 1, atol=1e-12),
                axis=1,
            )
        )
        fixed = np.asarray(vmap)[(2 * boundary[:, None] + np.arange(2)).ravel()]
        prescribed = exact.velocity(coordinates[boundary]).ravel()
        problem = MultiscaleProblem.from_global(
            Equation(matrix, forcing),
            len(forcing),
            fixed=dict(zip(fixed, prescribed, strict=True)),
            constraints=((moment, 0.0),),
        )
        coefficients = assemble(problem).solve().trace
        numerical = dolfinx.fem.Function(space)
        numerical.x.array[:] = coefficients
        uh, ph = ufl.split(numerical)
        du, dp = uh - truth_u, ph - truth_p
        metrics = dict(
            velocity_l2=float(
                np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(du, du) * dx)))
            ),
            pressure_l2=float(np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(dp**2 * dx)))),
            velocity_h1_seminorm=float(
                np.sqrt(
                    dolfinx.fem.assemble_scalar(
                        dolfinx.fem.form(ufl.inner(ufl.grad(du), ufl.grad(du)) * dx)
                    )
                )
            ),
            pressure_integral=float(moment @ coefficients),
        )
        assert abs(metrics["pressure_integral"]) < 1e-9
        velocity = binding.to_portable(coefficients, component=0).reshape(-1, 2)
        pressure = binding.to_portable(coefficients, component=1)
        state = reports / f"brinkman-reference-n{resolution}-state.npz"
        np.savez_compressed(
            state,
            points=mesh.points,
            cells=mesh.cells,
            velocity_coefficients=velocity,
            pressure_coefficients=pressure,
            velocity_degree=np.array(2),
            pressure_degree=np.array(1),
            viscosity=np.array(exact.viscosity),
            drag=np.array(exact.drag),
        )
        archive = dict(
            path=str(state.relative_to(root)),
            sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
            basis_convention=(
                "Basix canonical equispaced P2 velocity/P1 pressure nodal coefficients"
            ),
        )
        print("Conforming Taylor-Hood", resolution, metrics)
        return (
            (mesh, velocity, pressure),
            dict(n=resolution, triangles=len(mesh.cells), total_dofs=len(forcing), **metrics),
            archive,
        )
    finally:
        binding.close()


def record_brinkman_case(
    macro: TriangleMesh,
    skeleton: Any,
    system: Any,
    solution: Any,
    exact: Any,
    *,
    velocity_degree: int,
    pressure_degree: int,
    order: int,
    name: str,
    reports: Path,
    root: Path,
    named: bool = False,
    gauge: Any = None,
    **metadata: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Measure physical flow fields and archive an already solved mixed case.

    The zero physical pressure mean and macro conservation are checked
    separately from fine-cell divergence and raw pseudostress errors. Supplying
    a gauge requests the existing small-case singular-value compatibility check.
    """
    from scipy import sparse

    velocity_fields, pressure_fields = solution.field("velocity"), solution.field("pressure")
    meshes = tuple(field.mesh for field in velocity_fields)
    velocity = tuple(field.portable_coefficients.reshape(-1, 2) for field in velocity_fields)
    pressure = tuple(field.portable_coefficients for field in pressure_fields)
    metrics = flow_error_norms(
        meshes,
        velocity,
        pressure,
        velocity_degree,
        pressure_degree,
        exact,
        order=order,
        named_velocity=velocity_fields if named else None,
        named_pressure=pressure_fields if named else None,
    )
    assert abs(metrics["pressure_integral"]) < 1e-9
    assert metrics["macro_mass_defect"] < 1e-9
    row = dict(
        H=float(macro.lengths[macro.cell_faces].max()),
        macro_cells=len(macro.cells),
        fine_cells=sum(len(mesh.cells) for mesh in meshes),
        trace_dofs=skeleton.size,
        residual=float(solution.residual),
        **metrics,
        **metadata,
    )
    if gauge is not None:
        gauge_row = gauge[0]
        gauged = sparse.bmat(
            [
                [system.matrix, sparse.csc_matrix(gauge_row[:, None])],
                [sparse.csc_matrix(gauge_row[None]), None],
            ]
        ).toarray()
        singular_values = np.linalg.svd(gauged, compute_uv=False)
        row.update(
            smallest_gauged_singular_value=float(singular_values[-1]), gauged_dimension=len(gauged)
        )
        assert singular_values[-1] > 1e-10 * singular_values[0]
    archive = preserve_state(name, skeleton, system, solution, reports=reports, root=root)
    return (macro, meshes, velocity, pressure, pressure_degree), row, archive


def plot_brinkman_convergence(
    rows: Any, reference_rows: Any, control_rows: Any, *, reports: Path
) -> None:
    """Display physical measured fields and rates without changing a formulation."""
    from functools import partial

    partial(plot_fields, reports=reports)
    plot_errors_local = partial(plot_errors, reports=reports)
    H = np.array([r["H"] for r in rows if r["method"] == "MHM-USFEM"])
    errors = {
        f"{method}: {norm}": np.array([row[norm] for row in rows if row["method"] == method])
        for method in ("MHM Taylor-Hood", "MHM-USFEM")
        for norm in ("velocity_l2", "pressure_l2")
    }
    plot_errors_local(H, errors, "brinkman-convergence")
    for label, error in errors.items():
        print(label, "rates:", rates(H, error))
    reference_H = np.sqrt(2) / np.array([r["n"] for r in reference_rows])
    plot_errors_local(
        reference_H,
        {
            name: np.array([r[name] for r in reference_rows])
            for name in ("velocity_l2", "pressure_l2", "velocity_h1_seminorm")
        },
        "brinkman-reference-convergence",
    )
    control_H = np.array([r["H"] for r in control_rows if r["method"] == "MHM-USFEM"])
    control_errors = {
        f"{method}: {norm}": np.array([r[norm] for r in control_rows if r["method"] == method])
        for method in ("MHM Taylor-Hood", "MHM-USFEM")
        for norm in ("velocity_l2", "pressure_l2")
    }
    plot_errors_local(control_H, control_errors, "brinkman-enriched-convergence")
    for label, error in control_errors.items():
        print("enriched", label, "rates:", rates(control_H, error))


def plot_brinkman_family_rates(published_rows: Any, *, reports: Path) -> list[dict[str, Any]]:
    """Display physical measured fields and rates without changing a formulation."""
    from functools import partial

    partial(plot_fields, reports=reports)
    plot_errors_local = partial(plot_errors, reports=reports)
    published_rates = []
    for ell in (0, 1, 2):
        selected = [row for row in published_rows if row["ell"] == ell]
        H = np.array([row["H"] for row in selected])
        errors = {
            name: np.array([row[name] for row in selected])
            for name in ("velocity_l2", "pressure_l2", "velocity_h1_seminorm")
        }
        plot_errors_local(H, errors, f"brinkman-single-element-ell{ell}-convergence")
        measured = {name: rates(H, values).tolist() for name, values in errors.items()}
        published_rates.append(
            dict(
                ell=ell,
                local_degree=ell + 2,
                levels=[r["n"] for r in selected],
                measured=measured,
                literature={
                    "velocity_l2": ell + 2,
                    "pressure_l2": ell + 1,
                    "velocity_h1_seminorm": ell + 1,
                },
            )
        )
        print("single-element", ell, "measured rates", measured)

    return published_rates


def plot_brinkman_family_fields(
    published_cases: Any, levels: Any, references: Any, truth: Any, *, reports: Path
) -> None:
    """Display physical measured fields and rates without changing a formulation."""
    from functools import partial

    plot_fields_local = partial(plot_fields, reports=reports)
    partial(plot_errors, reports=reports)
    macro, meshes, velocity, pressure, degree = published_cases[2, levels[2][-1]]
    family_velocity = display_samples(meshes, velocity, degree, subdivisions=4)
    family_pressure = display_samples(meshes, pressure, degree, subdivisions=4)
    rmesh, ru, rp = references[128]
    reference_velocity = display_samples((rmesh,), (ru,), 2, subdivisions=1)
    reference_pressure = display_samples((rmesh,), (rp,), 1, subdivisions=1)
    plot_fields_local(
        macro,
        {
            "velocity magnitude exact": (
                family_velocity["points"],
                family_velocity["cells"],
                np.linalg.norm(truth.velocity(family_velocity["points"]), axis=1),
            ),
            "velocity magnitude\nsingle-element USFEM P4/P4": (
                family_velocity["points"],
                family_velocity["cells"],
                np.linalg.norm(family_velocity["values"], axis=1),
            ),
            "velocity magnitude\nTaylor-Hood reference": (
                reference_velocity["points"],
                reference_velocity["cells"],
                np.linalg.norm(reference_velocity["values"], axis=1),
            ),
            "pressure exact": (
                family_pressure["points"],
                family_pressure["cells"],
                truth.pressure(family_pressure["points"]),
            ),
            "pressure\nsingle-element USFEM P4/P4": (
                family_pressure["points"],
                family_pressure["cells"],
                family_pressure["values"],
            ),
            "pressure Taylor-Hood reference": (
                reference_pressure["points"],
                reference_pressure["cells"],
                reference_pressure["values"],
            ),
            "velocity error magnitude\nsingle-element USFEM P4/P4": (
                family_velocity["points"],
                family_velocity["cells"],
                np.linalg.norm(
                    family_velocity["values"] - truth.velocity(family_velocity["points"]),
                    axis=1,
                ),
            ),
            "pressure error\nsingle-element USFEM P4/P4": (
                family_pressure["points"],
                family_pressure["cells"],
                family_pressure["values"] - truth.pressure(family_pressure["points"]),
            ),
        },
        "brinkman-single-element-fields",
    )


def plot_brinkman_enriched_fields(
    enriched_cases: Any, references: Any, truth: Any, *, reports: Path
) -> None:
    """Display physical measured fields and rates without changing a formulation."""
    from functools import partial

    plot_fields_local = partial(plot_fields, reports=reports)
    partial(plot_errors, reports=reports)
    macro, meshes, velocity, pressure, pdegree = enriched_cases[16, "MHM-USFEM"]
    us = display_samples(meshes, velocity, 2, 2)
    usp = display_samples(meshes, pressure, pdegree, 2)
    _, gmeshes, gu, gp, gpd = enriched_cases[16, "MHM Taylor-Hood"]
    gal = display_samples(gmeshes, gu, 2, 2)
    galp = display_samples(gmeshes, gp, gpd, 2)
    rmesh, ru, rp = references[128]
    ref = display_samples((rmesh,), (ru,), 2, 1)
    refp = display_samples((rmesh,), (rp,), 1, 1)
    plot_fields_local(
        macro,
        {
            "velocity magnitude exact": (
                us["points"],
                us["cells"],
                np.linalg.norm(truth.velocity(us["points"]), axis=1),
            ),
            "velocity magnitude MHM Taylor-Hood enriched": (
                gal["points"],
                gal["cells"],
                np.linalg.norm(gal["values"], axis=1),
            ),
            "velocity magnitude MHM-USFEM enriched": (
                us["points"],
                us["cells"],
                np.linalg.norm(us["values"], axis=1),
            ),
            "pressure exact": (usp["points"], usp["cells"], truth.pressure(usp["points"])),
            "pressure MHM Taylor-Hood": (galp["points"], galp["cells"], galp["values"]),
            "pressure MHM-USFEM": (usp["points"], usp["cells"], usp["values"]),
            "velocity magnitude Taylor-Hood reference": (
                ref["points"],
                ref["cells"],
                np.linalg.norm(ref["values"], axis=1),
            ),
            "velocity error magnitude MHM-USFEM": (
                us["points"],
                us["cells"],
                np.linalg.norm(us["values"] - truth.velocity(us["points"]), axis=1),
            ),
            "pressure error MHM-USFEM": (
                usp["points"],
                usp["cells"],
                usp["values"] - truth.pressure(usp["points"]),
            ),
        },
        "brinkman-fields",
    )

    plot_fields_local(
        macro,
        {
            "pressure Taylor-Hood reference": (refp["points"], refp["cells"], refp["values"]),
            "pressure error MHM Taylor-Hood": (
                galp["points"],
                galp["cells"],
                galp["values"] - truth.pressure(galp["points"]),
            ),
            "pressure error MHM-USFEM": (
                usp["points"],
                usp["cells"],
                usp["values"] - truth.pressure(usp["points"]),
            ),
        },
        "brinkman-pressure-comparison",
    )


def plot_brinkman_profiles(enriched_cases: Any, truth: Any, *, reports: Path) -> None:
    """Display physical measured fields and rates without changing a formulation."""
    from functools import partial

    partial(plot_fields, reports=reports)
    partial(plot_errors, reports=reports)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    x = np.linspace(0, 1, 2001)
    points = np.column_stack((x, x * 0 + 0.37))
    axes[0].plot(x, truth.velocity(points)[:, 1], "k--", label="exact velocity component y")
    axes[1].plot(x, truth.pressure(points), "k--", label="exact pressure")
    for method in ("MHM Taylor-Hood", "MHM-USFEM"):
        m, meshes, u, p, pd = enriched_cases[16, method]
        for index, (position, values) in enumerate(profile_segments(m, meshes, u, 2)):
            axes[0].plot(
                position,
                values[:, 1],
                color={"MHM Taylor-Hood": "tab:blue", "MHM-USFEM": "tab:orange"}[method],
                label=method if index == 0 else None,
            )
            for cross in (position[0], position[-1]):
                axes[0].axvline(cross, color=".7", linewidth=0.4, alpha=0.6)
        for index, (position, values) in enumerate(profile_segments(m, meshes, p, pd)):
            axes[1].plot(
                position,
                values,
                color={"MHM Taylor-Hood": "tab:blue", "MHM-USFEM": "tab:orange"}[method],
                label=method if index == 0 else None,
            )
            for cross in (position[0], position[-1]):
                axes[1].axvline(cross, color=".7", linewidth=0.4, alpha=0.6)
    axes[0].set(
        xlim=(0.85, 1),
        xlabel="x at y=0.37",
        ylabel="velocity component y",
        title="Boundary-layer profile",
    )
    axes[1].set(
        xlim=(0, 1),
        xlabel="x at y=0.37",
        ylabel="pressure",
        title="One-sided pressure profiles; mean zero",
    )
    for ax in axes:
        ax.legend(fontsize=8)
    fig.savefig(reports / "brinkman-profiles.png", dpi=160)
    plt.show()


def inspect_elasticity_solution(system: Any, solution: Any, macro: TriangleMesh) -> None:
    """Report original residual and local trace injectivity for the executed primal space."""
    print(
        dict(
            global_unknowns=system.matrix.shape[0],
            largest_local_unknowns=max(len(field) for field in solution.fields),
            original_equations_relative_residual=solution.raw_residual,
        )
    )
    singular_values = np.linalg.svd(system.responses[0].problem.coupling, compute_uv=False)
    print(
        dict(
            local_trace_pairing_smallest_singular_value=float(singular_values[-1]),
            local_trace_pairing_condition=float(singular_values[0] / singular_values[-1]),
        )
    )
    assert singular_values[-1] > 1e-12 * singular_values[0]
    point = macro.points[macro.cells[0]].mean(axis=0, keepdims=True)
    print(
        "First macrocell displacement at its center:",
        solution.field("displacement")[0].evaluate(point),
    )
