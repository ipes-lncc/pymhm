"""Reusable field, quadrature, visualization and archive support for scalar introductions.

The notebooks retain their physical data and mathematical formulations. This
module operates on already declared fields, matrices and quadrature partitions;
it does not select or solve a multiscale method. Physical flux is -K grad(p),
where gradients are raw element derivatives rather than H(div) reconstructions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from examples.field_sampling import sample_field as sample_nodal_fields
from examples.introduction.provenance import source_digests
from pymhm.fem.scalar.operators import p1_geometry, triangle_quadrature
from pymhm.fem.scalar.quadrilateral import qk_basis, qk_space, quadrilateral_quadrature
from pymhm.fem.scalar.triangle import nodal_space, reference_basis
from pymhm.linalg.linear import solve_linear
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import tensor_values
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.conforming import ConformingQuadrilateralSolution
from pymhm.postprocessing.fields import DiscreteField
from pymhm.postprocessing.nodal import nodal_field

Array = NDArray[np.float64]
Evaluator = Callable[[Array], tuple[Array, Array]]


def grid_quadrature(
    bounds: tuple[float, float, float, float], shape: tuple[int, int], order: int = 5
) -> tuple[Array, Array]:
    """Return Gauss points and physical weights on an explicitly resolved grid.

    The caller chooses a common grid resolving every compared finite-element
    interface and material pixel; the function never infers hidden jumps.
    """
    grid = CartesianMacroMesh(*shape, bounds)
    reference, unit_weights = quadrilateral_quadrature(order)
    origins = grid.points[grid.cells[:, 0]]
    return (origins[:, None] + reference * grid.spacing).reshape(-1, 2), np.tile(
        unit_weights * np.prod(grid.spacing), len(grid.cells)
    )


def physical_errors(
    first: Evaluator,
    second: Evaluator,
    permeability: Any,
    points: Array,
    weights: Array,
    *,
    batch_size: int = 65536,
) -> dict[str, float]:
    """Integrate L2 pressure, L2 physical flux and K-inverse flux-energy differences.

    Fields are evaluated directly in their executed coefficient bases on the
    same quadrature, never through image samples or averaged gradients.
    """
    totals = np.zeros(6)
    for begin in range(0, len(points), batch_size):
        selected = slice(begin, begin + batch_size)
        x, w = points[selected], weights[selected]
        p, grad = first(x)
        pref, gradref = second(x)
        k = tensor_values(permeability, x)
        dq = -np.einsum("qab,qb->qa", k, grad - gradref)
        qref = -np.einsum("qab,qb->qa", k, gradref)
        totals += np.asarray(
            [
                w @ (p - pref) ** 2,
                w @ np.sum(dq**2, axis=1),
                w @ np.einsum("qa,qa->q", dq, np.linalg.solve(k, dq[..., None])[..., 0]),
                w @ pref**2,
                w @ np.sum(qref**2, axis=1),
                w @ np.einsum("qa,qa->q", qref, np.linalg.solve(k, qref[..., None])[..., 0]),
            ]
        )
    values = np.sqrt(totals)
    return dict(
        pressure_L2=float(values[0]),
        flux_L2=float(values[1]),
        flux_energy=float(values[2]),
        pressure_relative=float(values[0] / values[3]),
        flux_relative=float(values[1] / values[4]),
        energy_relative=float(values[2] / values[5]),
    )


def rectangle_panel(
    macro: CartesianMacroMesh,
    evaluator: Evaluator,
    *,
    quantity: str = "pressure",
    permeability: Any = 1.0,
    points_per_side: int = 21,
) -> tuple[Array, NDArray[np.int64], Array]:
    """Sample each macro rectangle independently, retaining two interface limits.

    Evaluation points approach boundary nodes from their macrocell interior.
    Plot coordinates stay on the true interfaces. ``quantity`` is pressure,
    permeability_xx, permeability_yy or physical Darcy flux magnitude.
    """
    axis = np.linspace(0, 1, points_per_side)
    x, y = np.meshgrid(axis, axis)
    reference = np.column_stack((x.ravel(), y.ravel()))
    indices = np.arange(points_per_side**2).reshape(points_per_side, points_per_side)
    a, b = indices[:-1, :-1].ravel(), indices[:-1, 1:].ravel()
    c, d = indices[1:, 1:].ravel(), indices[1:, :-1].ravel()
    base = np.vstack((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
    points, triangles, values = [], [], []
    for cell, vertices in enumerate(macro.cells):
        lower = macro.points[vertices[0]]
        physical = lower + reference * macro.spacing
        center = macro.points[vertices].mean(axis=0)
        inside = np.nextafter(physical, center)
        if quantity in {"permeability_xx", "permeability_yy"}:
            component = 0 if quantity == "permeability_xx" else 1
            value = tensor_values(permeability, inside)[:, component, component]
        else:
            pressure, gradient = evaluator(inside)
            if quantity == "pressure":
                value = pressure
            elif quantity == "flux_magnitude":
                flux = -np.einsum("qab,qb->qa", tensor_values(permeability, inside), gradient)
                value = np.linalg.norm(flux, axis=1)
            else:
                raise ValueError("unknown Darcy plot quantity")
        points.append(physical)
        triangles.append(base + cell * len(reference))
        values.append(value)
    return np.vstack(points), np.vstack(triangles), np.concatenate(values)


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
    from matplotlib.ticker import NullLocator

    h = np.asarray(mesh_sizes, dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for label, values in errors.items():
        e = np.asarray(values, dtype=float)
        axes[0].loglog(h, e, "o-", label=label)
        axes[1].semilogx(h[1:], observed_rates(h, e), "o-", label=label)
    axes[0].set(xlabel="H", ylabel="Measured error")
    axes[1].set(xlabel="H", ylabel="Observed rate")
    for axis, locations in zip(axes, (h, h[1:]), strict=True):
        axis.set_xticks(locations, [f"{value:.4g}" for value in locations])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.invert_xaxis()
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    return figure


def load_spe10_layer(root: Path, number: int = 36) -> tuple[CartesianCellField, dict[str, Any]]:
    """Read and verify an unchanged complete horizontal layer from pinned OPM data.

    Coordinates are feet, permeability is mD, axes are x,y. ``number`` is
    one-based; no rescaling, clipping, smoothing or crop is applied.
    """
    folder = root / "examples/results/spe10"
    provenance = json.loads((folder / "dataset.json").read_text())
    layer = next(row for row in provenance["layers"] if row["layer_one_based"] == number)
    path = folder / layer["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != layer["sha256"]:
        raise ValueError("SPE10 layer archive does not match its pinned SHA256")
    with np.load(path) as archive:
        horizontal = archive["permeability"][..., :2]
        tensors = np.zeros((*horizontal.shape[:2], 2, 2))
        tensors[..., 0, 0] = horizontal[..., 0]
        tensors[..., 1, 1] = horizontal[..., 1]
        spacing = tuple(archive["spacing"])
    return CartesianCellField(tensors, spacing), {**provenance, "selected_layer": layer}


def fine_quad_panel(
    fine: CartesianMacroMesh, degree: int, nodal: Array, material: CartesianCellField, quantity: str
) -> tuple[Array, NDArray[np.int64], Array]:
    """Sample each actual fine quadrilateral separately with its incident pixel tensor."""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    basis, derivative = qk_basis(degree, corners)
    dofs, _ = qk_space(fine, degree)
    coefficients = nodal[dofs]
    if quantity == "pressure":
        values = coefficients @ basis.T
    elif quantity == "flux_magnitude":
        gradient = np.einsum("ti,qia->tqa", coefficients, derivative / fine.spacing)
        # Pixel-fitted cells have constant K. Query interiors, never shared material vertices.
        incident_K = tensor_values(material, fine.points[fine.cells].mean(axis=1))
        flux = -np.einsum("tab,tqb->tqa", incident_K, gradient)
        values = np.linalg.norm(flux, axis=2)
    else:
        raise ValueError("quantity must be pressure or physical Darcy flux magnitude")
    points = fine.points[fine.cells].reshape(-1, 2)
    offset = 4 * np.arange(len(fine.cells), dtype=np.int64)
    triangles = np.asarray(
        np.vstack((offset[:, None] + [0, 1, 2], offset[:, None] + [0, 2, 3])), dtype=np.int64
    )
    return points, triangles, values.reshape(-1)


def broken_quad_panel(
    system: Any, solution: Any, degree: int, material: CartesianCellField, quantity: str
) -> tuple[Array, NDArray[np.int64], Array]:
    """Collect disconnected fine-element panels without merging any interface values."""
    points, triangles, values = [], [], []
    count = 0
    for field in solution.field("pressure"):
        local_points, local_triangles, local_values = fine_quad_panel(
            field.mesh, degree, field.portable_coefficients, material, quantity
        )
        points.append(local_points)
        triangles.append(local_triangles + count)
        values.append(local_values)
        count += len(local_points)
    return np.vstack(points), np.vstack(triangles), np.concatenate(values)


def fine_line_profile(
    fine: CartesianMacroMesh, degree: int, nodal: Array, material: CartesianCellField, x_line: float
) -> tuple[Array, Array, Array]:
    """Evaluate disconnected vertical fine-element profiles with each incident K."""
    column = int(np.floor((x_line - fine.points[0, 0]) / fine.spacing[0]))
    if not 0 <= column < fine.nx:
        raise ValueError("the profile must lie inside this mesh in the x direction")
    assert fine.ny is not None
    ny = fine.ny
    sample_y = np.linspace(0.0, 1.0, 5)
    reference = np.empty((ny, len(sample_y), 2))
    reference[:, :, 0] = (x_line - fine.points[0, 0]) / fine.spacing[0] - column
    reference[:, :, 1] = sample_y
    owners = np.arange(ny) * fine.nx + column
    dofs, _ = qk_space(fine, degree)
    local = nodal[dofs[owners]]
    basis, derivative = qk_basis(degree, reference.reshape(-1, 2))
    basis = basis.reshape(ny, len(sample_y), -1)
    derivative = derivative.reshape(ny, len(sample_y), -1, 2) / fine.spacing
    pressure = np.einsum("tqi,ti->tq", basis, local)
    gradient = np.einsum("tqia,ti->tqa", derivative, local)
    incident_K = tensor_values(material, fine.points[fine.cells[owners]].mean(axis=1))
    flux = -np.einsum("tab,tqb->tqa", incident_K, gradient)
    y = fine.points[0, 1] + (np.arange(ny)[:, None] + sample_y) * fine.spacing[1]
    # NaN separators preserve both independent limits at every fine interface.
    segments = [
        np.pad(values, ((0, 0), (0, 1)), constant_values=np.nan).ravel()
        for values in (y, pressure, flux[:, :, 1])
    ]
    return segments[0], segments[1], segments[2]


def evaluate_qk(
    mesh: CartesianMacroMesh, degree: int, coefficients: Array, points: Array
) -> tuple[Array, Array]:
    """Evaluate a global Cartesian field through its shared public numerical owner.

    An internal interface uses its positive incident cell. The domain boundary
    uses the adjacent interior cell. No material coefficient is averaged.
    """
    return ConformingQuadrilateralSolution(mesh, degree, coefficients, 1.0, 0.0).evaluate(points)


def evaluate_bound_pressure(
    macro: CartesianMacroMesh, fields: tuple[Any, ...], points: Array
) -> tuple[Array, Array]:
    """Evaluate named pressure and gradient on each owning macrocell, without averaging."""
    assert macro.ny is not None
    coordinates = (points - macro.points[0]) / macro.spacing
    indices = np.clip(np.floor(coordinates).astype(int), 0, [macro.nx - 1, int(macro.ny) - 1])
    owners = indices[:, 1] * macro.nx + indices[:, 0]
    pressure, gradient = np.empty(len(points)), np.empty((len(points), 2))
    for cell in np.unique(owners):
        selected = owners == cell
        pressure[selected], gradient[selected] = fields[cell].values_and_gradient(points[selected])
    return pressure, gradient


def native_scalar_space(mesh: Any, degree: int) -> tuple[Any, Any, Any]:
    """Bind an explicit equispaced Lagrange degree on a triangle or quadrilateral.

    The public binding owns topology, native resources and portable ordering.
    Returns native mesh, native function space and executed coefficient map.
    """
    import basix
    import basix.ufl

    from pymhm.backends.spaces import bind_space

    cell = "quadrilateral" if isinstance(mesh, CartesianMacroMesh) else "triangle"
    element = basix.ufl.element(
        "Lagrange",
        cell,
        degree,
        lagrange_variant=basix.LagrangeVariant.equispaced,
    )
    binding = bind_space(mesh, element)
    return binding.mesh, binding.space, binding.mapping


@dataclass(frozen=True)
class ScalarField:
    """A physical Pk nodal coefficient vector on one existing triangular mesh."""

    mesh: TriangleMesh
    degree: int
    values: np.ndarray

    @cached_property
    def dofs(self) -> np.ndarray:
        """Use the package's declared nodal ordering without changing its basis."""
        return nodal_space(self.mesh, self.degree)[0]

    @cached_property
    def locator(self) -> Any:
        """Find triangles from original connectivity, without retriangulating nodes."""
        from matplotlib.tri import Triangulation

        return Triangulation(
            self.mesh.points[:, 0], self.mesh.points[:, 1], triangles=self.mesh.cells
        ).get_trifinder()

    @cached_property
    def geometry(self) -> np.ndarray:
        """Return the package's physical barycentric gradients."""
        return p1_geometry(self.mesh)[0]

    @cached_property
    def definition(self) -> DiscreteField:
        """Archive the actual portable polynomial basis and its immutable coefficients."""
        return DiscreteField(nodal_field("pressure", self.mesh, self.degree), self.values)


def evaluate_scalar(field: ScalarField, points: Array, *, cells: Any = None) -> tuple[Array, Array]:
    """Evaluate a declared Pk field with the original one-sided triangle locator.

    Values and raw gradients use the public saved-basis field evaluator. Passing
    explicit cell owners preserves the original incident-cell convention.
    """
    selected = (
        np.asarray(field.locator(*points.T), dtype=int) if cells is None else np.asarray(cells)
    )
    if np.any(selected < 0):
        raise ValueError("evaluation point lies outside the declared field mesh")
    return field.definition.values_and_gradient(points, cells=selected)


def evaluate_incident_reference(
    reference: ScalarField, points: Array, incident_centers: Array
) -> tuple[Array, Array]:
    """Evaluate original coordinates from the side of their incident sampling cells.

    For nested triangular comparison meshes, a small interior probe selects the
    reference owner. Its length is 1e-7 times the smallest reference altitude,
    capped at the distance to the incident center. Only ownership uses that
    probe: pressure and raw gradient are evaluated at the original coordinates,
    with the selected cells explicitly supplied to the public field evaluator.
    """
    direction = np.asarray(incident_centers) - points
    if direction.shape != points.shape:
        raise ValueError("incident centers must match the physical sampling points")
    altitude = 2 * reference.mesh.areas[:, None] / reference.mesh.lengths[reference.mesh.cell_faces]
    radius = 1e-7 * float(np.min(altitude))
    distance = np.linalg.norm(direction, axis=1)
    fraction = np.ones(len(points))
    np.divide(radius, distance, out=fraction, where=distance > 0)
    probe = points + np.minimum(fraction, 1.0)[:, None] * direction
    owners = np.asarray(reference.locator(*probe.T), dtype=int)
    return evaluate_scalar(reference, points, cells=owners)


def sample_field(fields: Sequence[ScalarField], refinement: int = 2) -> dict[str, np.ndarray]:
    """Sample each incident triangle using the shared scalar/vector field sampler.

    All fields in this call have the same degree. Interface coordinates remain
    duplicated, preserving independent values and gradients on both sides.
    """
    if not fields or len({field.degree for field in fields}) != 1:
        raise ValueError("provide fields with one common polynomial degree")
    return sample_nodal_fields(
        tuple(field.mesh for field in fields),
        tuple(field.values for field in fields),
        fields[0].degree,
        refinement=refinement,
    )


def compare_scalar_fields(
    fields: Sequence[ScalarField],
    reference: ScalarField,
    coefficient: Callable[[np.ndarray], np.ndarray],
    *,
    order: int = 8,
) -> dict[str, float]:
    """Integrate on the finest reference partition resolving all compared meshes.

    This notebook uses nested uniform triangular grids. Gaussian points lie
    strictly inside each reference triangle; no interface averaging is used.
    The scalar coefficient multiplies the gradient in physical flux norms.
    """
    bary, weights = triangle_quadrature(order)
    basis, derivative, _ = reference_basis(reference.degree, bary)
    errors, norms = np.zeros(3), np.zeros(3)
    for start in range(0, len(reference.mesh.cells), 256):
        stop = min(start + 256, len(reference.mesh.cells))
        triangles = reference.mesh.cells[start:stop]
        points = np.einsum("qi,tia->tqa", bary, reference.mesh.points[triangles])
        flat = points.reshape(-1, 2)
        pressure, gradient = np.empty(len(flat)), np.empty((len(flat), 2))
        assigned = np.zeros(len(flat), dtype=bool)
        for field in fields:
            inside = np.asarray(field.locator(*flat.T)) >= 0
            if np.any(inside):
                pressure[inside], gradient[inside] = evaluate_scalar(field, flat[inside])
                assigned[inside] = True
        if not np.all(assigned):
            raise ValueError("compared fields do not cover the physical integration domain")
        nodal = reference.values[reference.dofs[start:stop]]
        pref = (nodal @ basis.T).ravel()
        gref = np.einsum(
            "qib,tba,ti->tqa",
            derivative,
            reference.geometry[start:stop],
            nodal,
        ).reshape(-1, 2)
        a = coefficient(flat)
        measure = (reference.mesh.areas[start:stop, None] * weights).ravel()
        dp, dg = pressure - pref, gradient - gref
        dq, qref = -a[:, None] * dg, -a[:, None] * gref
        errors += np.array(
            [
                measure @ dp**2,
                measure @ np.sum(dq**2, axis=1),
                measure @ (a * np.sum(dg**2, axis=1)),
            ]
        )
        norms += np.array(
            [
                measure @ pref**2,
                measure @ np.sum(qref**2, axis=1),
                measure @ (a * np.sum(gref**2, axis=1)),
            ]
        )
    if np.any(norms <= 0):
        raise ValueError("relative norms require a nonzero reference field")
    absolute, relative = np.sqrt(errors), np.sqrt(errors / norms)
    return dict(
        pressure_L2=float(absolute[0]),
        flux_L2=float(absolute[1]),
        flux_energy=float(absolute[2]),
        pressure_relative=float(relative[0]),
        flux_relative=float(relative[1]),
        energy_relative=float(relative[2]),
    )


def plot_field_panels(
    macro_mesh: Any,
    panels: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    figsize: tuple[float, float] | None = None,
    color_limits: Mapping[str, tuple[float, float]] | None = None,
    emphasize_macro: bool = False,
) -> Any:
    """Plot independent nodal scalar panels and their actual macrofaces.

    Each panel supplies physical points, its explicit triangular connectivity
    and values. Duplicate coordinates are retained, so broken one-sided fields
    are never averaged across a macroface. Each field has its own colorbar.
    ``color_limits`` optionally sets physical minimum/maximum values by label,
    allowing related fields to use matching scales without merging colorbars.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.tri import Triangulation

    count = len(panels)
    if not count:
        raise ValueError("provide at least one field panel")
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    size = figsize or (
        (4.3 * columns, 3.8 * rows) if emphasize_macro else (4.1 * columns, 3.6 * rows)
    )
    limits_by_label = {} if color_limits is None else dict(color_limits)
    if emphasize_macro:
        for prefix in ("Pressure:", "Flux magnitude:"):
            labels = [label for label in panels if label.startswith(prefix)]
            maximum = max((float(np.max(panels[label][2])) for label in labels), default=1.0)
            limits_by_label.update({label: (0.0, maximum) for label in labels})
    figure, axes = plt.subplots(rows, columns, figsize=size, squeeze=False, layout="constrained")
    for axis, (label, (points, triangles, values)) in zip(axes.flat, panels.items(), strict=False):
        coordinates = np.asarray(points)
        triangulation = Triangulation(coordinates[:, 0], coordinates[:, 1], triangles=triangles)
        limits: dict[str, Any] = {}
        if label in limits_by_label:
            lower, upper = limits_by_label[label]
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError("color limits must be finite and increasing")
            limits = {"vmin": lower, "vmax": upper}
        if emphasize_macro and label == "Signed pressure difference":
            bound = max(float(np.max(abs(values))), np.finfo(float).tiny)
            limits = dict(vmin=-bound, vmax=bound, cmap="RdBu_r")
        artist = axis.tripcolor(triangulation, values, shading="gouraud", rasterized=True, **limits)
        if emphasize_macro:
            axis.add_collection(
                LineCollection(
                    macro_mesh.points[macro_mesh.faces].tolist(),
                    colors="white",
                    linewidths=1.0,
                    alpha=0.85,
                )
            )
            axis.add_collection(
                LineCollection(
                    macro_mesh.points[macro_mesh.faces].tolist(), colors="#24343c", linewidths=0.45
                )
            )
        else:
            axis.add_collection(
                LineCollection(
                    macro_mesh.points[macro_mesh.faces].tolist(),
                    colors="0.2",
                    linewidths=0.65,
                    zorder=3,
                )
            )
        axis.set(title=label, xlabel="x", ylabel="y", aspect="equal")
        figure.colorbar(artist, ax=axis, shrink=0.85 if emphasize_macro else 0.87, pad=0.025)
    for axis in list(axes.flat)[count:]:
        axis.set_visible(False)
    return figure


def dirichlet_solve(matrix: Any, load: Array, dofs: Any, values: Array) -> Array:
    """Lift stated essential coefficients and solve the remaining classical equation.

    This is coefficient elimination for an already assembled arbitrary operator;
    no physical form or boundary identification is selected by this helper.
    The lifting product A_free,fixed @ g follows the supplied essential-coordinate
    ordering, retaining the caller's literal pairing of indices and values.
    """
    operator = sparse.csc_matrix(matrix)
    pressure = np.zeros(len(load))
    pressure[dofs] = values
    free = np.setdiff1d(np.arange(len(load)), dofs)
    forcing = load[free] - operator[free][:, dofs] @ pressure[dofs]
    pressure[free] = solve_linear(operator[free][:, free], forcing)
    return pressure


def archive_cell(
    archive: dict[str, Any],
    cell: int,
    field: ScalarField,
    *,
    executed: DiscreteField | None = None,
    trace_binding: Any = None,
    **arrays: Any,
) -> None:
    """Store incident geometry and literal executed arrays, without rebuilding a basis.

    Supplied names become name_cell keys. An executed native field additionally
    retains its original basis matrix, basis points, portable map and digest;
    the trace binding retains its original trial map and global coordinates.
    """
    archive.update(
        {
            f"points_{cell}": field.mesh.points,
            f"cells_{cell}": field.mesh.cells,
            f"pressure_{cell}": field.values,
        }
    )
    archive.update({f"{name}_{cell}": value for name, value in arrays.items()})
    if executed is not None:
        descriptor = executed.definition.descriptor
        if descriptor is None:
            raise TypeError("executed field archive requires its native descriptor")
        archive.update(
            {
                f"executed_pressure_{cell}": executed.coefficients,
                f"executed_pressure_mapping_{cell}": descriptor.mapping,
                f"executed_pressure_basis_{cell}": descriptor.basis_matrix,
                f"executed_pressure_basis_points_{cell}": descriptor.basis_points,
                f"pressure_basis_digest_{cell}": np.array(executed.basis_digest),
            }
        )
    if trace_binding is not None:
        archive.update(
            {
                f"trace_trial_map_{cell}": trace_binding.trial_map,
                f"gamma_dofs_{cell}": trace_binding.dofs,
            }
        )


def save_scalar_report(
    root: Path,
    notebook: str,
    record: Mapping[str, Any],
    archive: Mapping[str, Any],
    figures: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
) -> Path:
    """Save stated scientific results, executed arrays and full source provenance.

    Numerical quantities are supplied explicitly by the notebook. Backend and
    source digests identify the executed implementation and locked environment.
    """
    import basix
    import dolfinx
    import ufl

    output = root / "build" / "introduction" / notebook if output_dir is None else output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(output / (name + ".png"), dpi=160)
    result = dict(record)
    result.update(
        {
            "operator_language": "executed UFL",
            "native_backend": {
                "dolfinx": dolfinx.__version__,
                "basix": basix.__version__,
                "ufl": ufl.__version__,
            },
            "notebook_sha256": hashlib.sha256(
                (root / "notebooks" / "introduction" / (notebook + ".ipynb")).read_bytes()
            ).hexdigest(),
            "source_sha256": {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((root / "src" / "pymhm").rglob("*.py"))
            },
            "pixi_lock_sha256": hashlib.sha256((root / "pixi.lock").read_bytes()).hexdigest(),
            "support_sha256": source_digests(
                root,
                (
                    Path(__file__),
                    root / "examples/field_sampling.py",
                    root / "examples/introduction/provenance.py",
                ),
            ),
            "scope": (
                "Original introductory case; finite local Galerkin realization; "
                "numerical classical baseline"
            ),
        }
    )
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output / "executed-fields.npz", **archive)
    return output


def plot_reference_differences(mesh_sizes: Any, differences: Sequence[Mapping[str, float]]) -> Any:
    """Plot measured pressure/flux differences and their observed rates.

    Successive differences are reference-resolution indicators rather than
    exact-solution errors. Each physical field retains its own labeled axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    spacing = np.asarray(mesh_sizes)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for axis, key, label in zip(
        axes,
        ("pressure_relative", "flux_relative"),
        ("Pressure difference / reference norm", "Flux difference / reference norm"),
        strict=True,
    ):
        values = np.array([row[key] for row in differences])
        rate = observed_rates(spacing, values)[0]
        axis.loglog(spacing, values, "o-", label=f"observed difference rate = {rate:.2f}")
        axis.set_xticks(spacing, labels=[f"{h:.3g}" for h in spacing])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set(xlabel="Finer classical mesh spacing", ylabel=label)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=9)
    return figure


def plot_material(
    macro: TriangleMesh,
    coefficient: Callable[[Array], Array],
    *,
    resolution: int = 64,
    limits: tuple[float, float] = (1.0, 3.0),
) -> Any:
    """Plot a stated scalar material with the actual incident macro mesh."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.tri import Triangulation

    preview = TriangleMesh.unit_square(resolution)
    figure, axis = plt.subplots(figsize=(5.4, 4.2), layout="constrained")
    artist = axis.tripcolor(
        Triangulation(preview.points[:, 0], preview.points[:, 1], triangles=preview.cells),
        coefficient(preview.points),
        shading="gouraud",
        vmin=limits[0],
        vmax=limits[1],
    )
    axis.add_collection(
        LineCollection(macro.points[macro.faces].tolist(), colors="white", linewidths=1.1)
    )
    axis.add_collection(
        LineCollection(macro.points[macro.faces].tolist(), colors="#24343c", linewidths=0.45)
    )
    axis.set(
        title="Multiscale permeability and actual macro mesh",
        xlabel="x",
        ylabel="y",
        aspect="equal",
    )
    figure.colorbar(artist, ax=axis, label="Permeability aε")
    return figure


def plot_scalar_comparison(
    macro: TriangleMesh,
    fields: Sequence[ScalarField],
    reference: ScalarField,
    coefficient: Callable[[Array], Array],
    *,
    numerical_label: str,
) -> Any:
    """Plot pressure, signed differences and physical flux on an incident-cell sample.

    The same six panels retain matching numerical/reference color ranges and
    independent colorbars. Neither gradient nor interface values are averaged.
    """
    samples = sample_field(fields, refinement=2)
    points, triangles = samples["points"], samples["cells"]
    pressure, gradient = samples["values"], samples["gradient"]
    reference_pressure, reference_gradient = evaluate_incident_reference(
        reference, points, samples["incident_centers"]
    )
    material = tensor_values(coefficient, points)
    flux = -np.einsum("qab,qb->qa", material, gradient)
    reference_flux = -np.einsum("qab,qb->qa", material, reference_gradient)
    panels = {
        "Pressure: classical P2": (points, triangles, reference_pressure),
        f"Pressure: {numerical_label}": (points, triangles, pressure),
        "Signed pressure difference": (points, triangles, pressure - reference_pressure),
        "Flux magnitude: classical P2": (points, triangles, np.linalg.norm(reference_flux, axis=1)),
        f"Flux magnitude: {numerical_label}": (points, triangles, np.linalg.norm(flux, axis=1)),
        "Flux difference magnitude": (
            points,
            triangles,
            np.linalg.norm(flux - reference_flux, axis=1),
        ),
    }
    return plot_field_panels(macro, panels, emphasize_macro=True)


def executed_array_digest(*arrays: Any) -> str:
    """Digest literal executed arrays in declared order without regenerating coordinates."""
    digest = hashlib.sha256()
    for values in arrays:
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()
