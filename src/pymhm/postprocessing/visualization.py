"""Optional PyVista conversion without merging broken finite-element fields.

The numerical core never imports PyVista. Converters return ordinary PyVista
objects, so slicing, glyphs, export and interactive views use its native API.
Structured arrays use explicit cell shapes and x-fastest (Fortran) ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pymhm.core.validation import positive_int
from pymhm.fem.scalar.triangle import nodal_space, reference_basis
from pymhm.meshes.triangle import TriangleMesh

Array = NDArray[Any]
_PREFIX = "pymhm:"


def _pyvista() -> ModuleType:
    """Load the optional backend with an actionable installation error."""
    try:
        return import_module("pyvista")
    except (ImportError, OSError) as exc:
        raise ImportError("PyVista/VTK could not load; install pymhm[visualization].") from exc


def _real(values: Any, name: str) -> Array:
    """Reject complex, nonnumeric and nonfinite data before any real conversion."""
    array = np.asarray(values)
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.iscomplexobj(array)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must contain finite real numbers")
    return array.copy()


def _field(name: str, values: Any, shape: tuple[int, ...]) -> Array:
    """Validate scalar/vector entity data and pad planar vectors with zero z."""
    if not isinstance(name, str) or not name or name.startswith(_PREFIX):
        raise ValueError("field names must be nonempty strings outside the pymhm: namespace")
    array = _real(values, name)
    if array.shape == shape:
        return array.reshape(-1, order="F")
    if array.shape[:-1] != shape or array.shape[-1] not in (2, 3):
        raise ValueError(f"{name!r} must have shape {shape} or {shape}+(2 or 3,)")
    vector = array.reshape((-1, array.shape[-1]), order="F")
    return np.column_stack((vector, np.zeros(len(vector)))) if vector.shape[1] == 2 else vector


def _triangle_grid(
    points: Array, cells: Array, point_data: Mapping[str, Array], cell_data: Mapping[str, Array]
) -> Any:
    """Construct a VTK triangle grid using independent, explicitly supplied vertices."""
    pv = _pyvista()
    connectivity = np.column_stack((np.full(len(cells), 3), cells)).ravel()
    xyz = np.column_stack((points, np.zeros(len(points))))
    grid = pv.UnstructuredGrid(connectivity, np.full(len(cells), pv.CellType.TRIANGLE), xyz)
    grid.point_data.update(point_data)
    grid.cell_data.update(cell_data)
    return grid


def triangle_grid(
    mesh: TriangleMesh,
    *,
    point_data: Mapping[str, Any] | None = None,
    cell_data: Mapping[str, Any] | None = None,
) -> Any:
    """Convert a planar triangular mesh and explicitly associated scalar/vector data.

    Point arrays have shape ``(npoints,)`` or ``(npoints, 2|3)``. Cell arrays
    have the analogous cell shape. Two-component vectors receive a zero z
    component. Cell values remain cell data; no cell-to-point averaging occurs.
    Input geometry and fields are copied into the returned UnstructuredGrid.
    """
    points = {
        name: _field(name, value, (len(mesh.points),)) for name, value in (point_data or {}).items()
    }
    cells = {
        name: _field(name, value, (len(mesh.cells),)) for name, value in (cell_data or {}).items()
    }
    return _triangle_grid(mesh.points, mesh.cells, points, cells)


def broken_triangle_grid(
    local_meshes: Sequence[TriangleMesh],
    fields: Sequence[Any],
    degree: int = 1,
    *,
    name: str = "field",
    subdivision: int | None = None,
    macro_mesh: TriangleMesh | None = None,
) -> Any:
    """Sample broken P0–P4 scalar/vector fields without identifying coincident points.

    For degree zero, each local array has one value per fine triangle. Higher
    degrees use :func:`pymhm.fem.scalar.triangle.nodal_space` ordering. Every fine triangle
    gets a separate display triangulation, preserving both macro and fine-cell
    jumps. Values are exact at sample points; straight display triangles are
    a visualization approximation of higher-order fields, not an error norm.
    ``subdivision`` defaults to twice the degree, or one for P0. Supplying the
    macro mesh stores its actual geometry for automatic overlays in plot_field.
    """
    degree = positive_int(degree, "degree", 0)
    if degree > 4:
        raise ValueError("degree must lie between zero and four")
    subdivision = positive_int(
        max(1, 2 * degree) if subdivision is None else subdivision, "subdivision"
    )
    if not local_meshes or len(local_meshes) != len(fields):
        raise ValueError("local meshes and fields must be nonempty sequences of equal length")
    if macro_mesh is not None and len(macro_mesh.cells) != len(local_meshes):
        raise ValueError("one local mesh is required per macrocell")
    template = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, subdivision)
    bary = np.column_stack((1 - template.points.sum(axis=1), template.points))
    points, cells, values, macro_ids, fine_ids = [], [], [], [], []
    offset = 0
    for macro, (mesh, field) in enumerate(zip(local_meshes, fields, strict=True)):
        dofs, nodes = nodal_space(mesh, degree) if degree else (None, mesh.cells)
        coefficients = _field(name, field, (len(nodes),))
        if degree:
            basis = reference_basis(degree, bary)[0]
            sampled = np.einsum("qi,ti...->tq...", basis, coefficients[dofs])
        else:
            sampled = np.repeat(coefficients[:, None], len(bary), axis=1)
        coordinates = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        points.append(coordinates.reshape(-1, 2))
        cells.append(
            (
                template.cells[None]
                + offset
                + np.arange(len(mesh.cells))[:, None, None] * len(bary)
            ).reshape(-1, 3)
        )
        values.append(sampled.reshape((-1, *sampled.shape[2:])))
        macro_ids.append(np.full(len(mesh.cells) * len(template.cells), macro))
        fine_ids.append(np.repeat(np.arange(len(mesh.cells)), len(template.cells)))
        offset += len(mesh.cells) * len(bary)
    if len({value.shape[1:] for value in values}) != 1:
        raise ValueError("all local fields must have the same number of components")
    grid = _triangle_grid(
        np.concatenate(points),
        np.concatenate(cells),
        {name: np.concatenate(values)},
        {
            _PREFIX + "macro_cell": np.concatenate(macro_ids),
            _PREFIX + "fine_cell": np.concatenate(fine_ids),
        },
    )
    if macro_mesh is not None:
        grid.field_data[_PREFIX + "macro_points"] = macro_mesh.points.copy()
        grid.field_data[_PREFIX + "macro_cells"] = macro_mesh.cells.copy()
    return grid


def macro_edges(mesh: TriangleMesh) -> Any:
    """Return each actual macroface once as a planar PyVista PolyData line."""
    points = np.column_stack((mesh.points, np.zeros(len(mesh.points))))
    lines = np.column_stack((np.full(len(mesh.faces), 2), mesh.faces)).ravel()
    return _pyvista().PolyData(points, lines=lines)


def _coordinates(values: Sequence[float], dimension: int, name: str, *, spacing: bool) -> Array:
    """Validate physical image origin/spacing, allowing XY pairs for planar images."""
    array = _real(values, name)
    if array.shape == (2,) and dimension == 2:
        array = np.append(array, 1.0 if spacing else 0.0)
    if array.shape != (3,) or (spacing and np.any(array <= 0)):
        raise ValueError(f"{name} needs three coordinates (two for 2D), with positive spacing")
    return array


def structured_cell_grid(
    cell_shape: Sequence[int],
    *,
    cell_data: Mapping[str, Any],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> Any:
    """Create a uniform 2D/3D cell-centered image with unambiguous array ordering.

    ``cell_shape=(nx, ny)`` or ``(nx, ny, nz)`` explicitly specifies geometry.
    Scalar fields have that shape; vectors append an axis of size two or three.
    The x/I index varies fastest in VTK storage (Fortran array order). No axis
    is reversed, no physical unit is converted, and no cell field is smoothed.
    A 2D grid has one plane of points, not a fictitious layer of volume cells.
    ``origin`` denotes the lower cell corner and ``spacing`` the physical cell
    widths. Both accept triples; XY pairs are also accepted for 2D grids.
    """
    shape = tuple(cell_shape)
    if len(shape) not in (2, 3):
        raise ValueError("cell_shape must explicitly contain two or three dimensions")
    shape = tuple(positive_int(size, "cell dimension") for size in shape)
    fields = {name: _field(name, value, shape) for name, value in cell_data.items()}
    step = _coordinates(spacing, len(shape), "spacing", spacing=True)
    corner = _coordinates(origin, len(shape), "origin", spacing=False)
    dimensions = tuple(size + 1 for size in shape) + ((1,) if len(shape) == 2 else ())
    grid = _pyvista().ImageData(dimensions=dimensions, spacing=step, origin=corner)
    grid.cell_data.update(fields)
    return grid


def plot_field(
    grid: Any,
    scalars: str,
    *,
    macro_mesh: TriangleMesh | None = None,
    off_screen: bool = False,
    window_size: tuple[int, int] = (1000, 700),
    **mesh_options: Any,
) -> Any:
    """Create a composable Plotter with field colors and an actual macro overlay.

    Return the open plotter; the caller owns ``show``, ``screenshot`` and
    ``close``. Embedded macro geometry from broken_triangle_grid is used unless
    ``macro_mesh`` overrides it. Scalar association defaults to cell data when
    both associations contain the same name; pass ``preference='point'`` to
    choose nodal data. Two-dimensional data use an XY camera. Rendering uses
    no geometry cleaning, point merging or cell-to-point conversion.
    """
    if scalars not in grid.point_data and scalars not in grid.cell_data:
        raise ValueError(f"field {scalars!r} is not present in the grid")
    if macro_mesh is None and _PREFIX + "macro_points" in grid.field_data:
        macro_mesh = TriangleMesh(
            grid.field_data[_PREFIX + "macro_points"], grid.field_data[_PREFIX + "macro_cells"]
        )
    plotter = _pyvista().Plotter(off_screen=off_screen, window_size=window_size)
    try:
        options = {"lighting": False, "smooth_shading": False, "preference": "cell"} | mesh_options
        plotter.add_mesh(grid, scalars=scalars, **options)
        if macro_mesh is not None:
            edges = macro_edges(macro_mesh)
            if grid.bounds[4] == grid.bounds[5]:
                edges.points[:, 2] = grid.bounds[4]
            plotter.add_mesh(edges, color="black", line_width=2)
        if grid.bounds[4] == grid.bounds[5]:
            plotter.view_xy()
        else:
            plotter.view_isometric()
    except Exception:
        plotter.close()
        raise
    return plotter
