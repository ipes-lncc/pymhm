"""Cartesian quadrilateral MHM with continuous tensor-product Qk local pressures.

The skeleton uses the same signed physical normal flux as triangular Darcy.
Only axis-aligned Cartesian rectangles are supported; curved and bilinear
non-affine quadrilaterals require a different geometric transformation.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, cast

import numpy as np
from numpy.polynomial import polynomial
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm._geometry_roundoff import cartesian_coordinates, coordinate_difference
from pymhm.elements import boundary_data, scalar_values, tensor_values, vector_values
from pymhm.hybrid import HybridSolution, HybridSystem, LocalAssembly, LocalProblem
from pymhm.mesh import FloatArray, IntArray, SkeletonSpace, TriangleMesh, positive_int
from pymhm.reservoir import CartesianCellField


@dataclass(frozen=True)
class CartesianMacroMesh:
    """Uniform rectangular mesh with counterclockwise cells and oriented faces.

    ``bounds=(xmin,xmax,ymin,ymax)`` and ``nx,ny`` describe the physical domain.
    The first adjacent cell defines each face normal. A missing ``ny`` uses
    ``nx``. Arrays are immutable; cell and point numbering run fastest in x.
    The geometric face interface is compatible with ``SkeletonSpace``.
    """

    nx: int = 1
    ny: int | None = None
    bounds: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    points: FloatArray = field(init=False, repr=False)
    cells: IntArray = field(init=False, repr=False)
    faces: IntArray = field(init=False, repr=False)
    cell_faces: IntArray = field(init=False, repr=False)
    signs: IntArray = field(init=False, repr=False)
    face_cells: IntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the rectangle and build its signed cell-face incidence."""
        nx = positive_int(self.nx, "nx")
        ny = nx if self.ny is None else positive_int(self.ny, "ny")
        if np.iscomplexobj(self.bounds):
            raise ValueError("bounds must be real")
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError("bounds must contain four finite coordinates")
        x0, x1, y0, y1 = bounds
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bounds must have positive width and height")
        x, y = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
        points = np.column_stack((np.tile(x, ny + 1), np.repeat(y, nx + 1)))
        if len(np.unique(points[:, 0])) != nx + 1 or len(np.unique(points[:, 1])) != ny + 1:
            raise ValueError("cell spacing is not resolvable at the coordinate scale")
        i, j = np.meshgrid(np.arange(nx), np.arange(ny))
        a = (j * (nx + 1) + i).ravel()
        cells = np.column_stack((a, a + 1, a + nx + 2, a + nx + 1))
        # Face IDs follow first encounter in counterclockwise row-major cells.
        # Closed formulas preserve that public numbering without Python dictionaries.
        horizontal = np.empty((ny + 1, nx), dtype=np.int64)
        vertical = np.empty((ny, nx + 1), dtype=np.int64)
        columns = np.arange(nx)
        horizontal[0] = np.where(columns == 0, 0, 3 * columns + 1)
        horizontal[1] = np.where(columns == 0, 2, 3 * columns + 3)
        vertical[0, 0] = 3
        vertical[0, 1:] = np.where(columns == 0, 1, 3 * columns + 2)
        starts = 3 * nx + 1 + np.arange(ny - 1) * (2 * nx + 1)
        horizontal[2:] = starts[:, None] + np.where(columns == 0, 1, 2 * columns + 2)
        vertical[1:, 0] = starts + 2
        vertical[1:, 1:] = starts[:, None] + np.where(columns == 0, 0, 2 * columns + 1)
        count = (ny + 1) * nx + (nx + 1) * ny
        faces = np.empty((count, 2), dtype=np.int64)
        neighbors = np.full((count, 2), -1, dtype=np.int64)
        grid = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
        hstart, hend = grid[:, :-1].copy(), grid[:, 1:].copy()
        hstart[1:], hend[1:] = hend[1:].copy(), hstart[1:].copy()
        faces[horizontal.ravel()] = np.column_stack((hstart.ravel(), hend.ravel()))
        vstart, vend = grid[:-1].copy(), grid[1:].copy()
        vstart[:, 0], vend[:, 0] = vend[:, 0].copy(), vstart[:, 0].copy()
        faces[vertical.ravel()] = np.column_stack((vstart.ravel(), vend.ravel()))
        ids = np.arange(nx * ny).reshape(ny, nx)
        neighbors[horizontal[0], 0] = ids[0]
        neighbors[horizontal[1:].ravel(), 0] = ids.ravel()
        neighbors[horizontal[1:-1].ravel(), 1] = ids[1:].ravel()
        neighbors[vertical[:, 0], 0] = ids[:, 0]
        neighbors[vertical[:, 1:].ravel(), 0] = ids.ravel()
        neighbors[vertical[:, 1:-1].ravel(), 1] = ids[:, 1:].ravel()
        cell_faces = np.column_stack(
            (
                horizontal[:-1].ravel(),
                vertical[:, 1:].ravel(),
                horizontal[1:].ravel(),
                vertical[:, :-1].ravel(),
            )
        )
        signs = np.column_stack(
            (
                np.where(j.ravel() == 0, 1, -1),
                np.ones(nx * ny, dtype=np.int64),
                np.ones(nx * ny, dtype=np.int64),
                np.where(i.ravel() == 0, 1, -1),
            )
        )
        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "bounds", tuple(float(v) for v in bounds))
        for name, value in (
            ("points", points),
            ("cells", cells),
            ("faces", np.asarray(faces, dtype=np.int64)),
            ("cell_faces", cell_faces),
            ("signs", signs),
            ("face_cells", np.asarray(neighbors, dtype=np.int64)),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def spacing(self) -> FloatArray:
        """Return the physical widths of one rectangular cell."""
        x0, x1, y0, y1 = self.bounds
        return np.array([(x1 - x0) / self.nx, (y1 - y0) / cast(int, self.ny)])

    @property
    def areas(self) -> FloatArray:
        """Return one positive physical area per cell."""
        return np.full(len(self.cells), np.prod(self.spacing))

    @property
    def lengths(self) -> FloatArray:
        """Return the physical lengths of the oriented mesh faces."""
        return np.linalg.norm(self.points[self.faces[:, 1]] - self.points[self.faces[:, 0]], axis=1)

    @property
    def normals(self) -> FloatArray:
        """Return unit normals pointing out of each face's first adjacent cell."""
        tangent = self.points[self.faces[:, 1]] - self.points[self.faces[:, 0]]
        return np.column_stack((tangent[:, 1], -tangent[:, 0])) / self.lengths[:, None]

    @property
    def boundary_faces(self) -> IntArray:
        """Return faces with a single neighboring rectangle."""
        return np.flatnonzero(self.face_cells[:, 1] == -1)

    def submesh(self, cell: int, subdivisions: int | tuple[int, int]) -> "CartesianMacroMesh":
        """Refine one rectangle independently in its two coordinate directions."""
        positive_int(cell, "cell", 0)
        if cell >= len(self.cells):
            raise ValueError("cell index outside mesh")
        nx, ny = _refinement(subdivisions)
        lower, upper = self.points[self.cells[cell, [0, 2]]]
        return CartesianMacroMesh(nx, ny, (lower[0], upper[0], lower[1], upper[1]))


def _refinement(value: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize a scalar or rectangular pair of local subdivision counts."""
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("local_refinement requires two integers")
        return positive_int(value[0], "local_refinement"), positive_int(
            value[1], "local_refinement"
        )
    n = positive_int(value, "local_refinement")
    return n, n


def quadrilateral_quadrature(order: int = 4) -> tuple[FloatArray, FloatArray]:
    """Return product Gauss points on [0,1]^2 and unit-sum integration weights."""
    points, weights = leggauss(positive_int(order, "quadrature order"))
    points, weights = (points + 1) / 2, weights / 2
    return (
        np.array([(x, y) for y in points for x in points]),
        np.array([wx * wy for wy in weights for wx in weights]),
    )


@lru_cache(maxsize=16)
def _cardinals(degree: int) -> tuple[FloatArray, ...]:
    """Build the exact nodal cardinal polynomials in ascending power order."""
    nodes = np.linspace(0, 1, degree + 1)
    return tuple(
        polynomial.polyfromroots(np.delete(nodes, i)) / np.prod(nodes[i] - np.delete(nodes, i))
        for i in range(degree + 1)
    )


def _cardinal_values(degree: int, points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Evaluate nodal polynomials and derivatives without high-order cancellation.

    The product recurrence remains defined at every interpolation node. The
    established low-order path is retained to preserve stored Q1--Q4 fields.
    """
    if degree <= 4:
        coefficients = _cardinals(degree)
        return (
            np.column_stack([polynomial.polyval(points, c) for c in coefficients]),
            np.column_stack(
                [polynomial.polyval(points, polynomial.polyder(c)) for c in coefficients]
            ),
        )
    nodes = np.linspace(0, 1, degree + 1)
    values = np.ones((len(points), degree + 1))
    derivatives = np.zeros_like(values)
    for i, node in enumerate(nodes):
        for other in np.delete(nodes, i):
            factor = (points - other) / (node - other)
            derivatives[:, i] = derivatives[:, i] * factor + values[:, i] / (node - other)
            values[:, i] *= factor
    return values, derivatives


def qk_basis(degree: int, points: Any) -> tuple[FloatArray, FloatArray]:
    """Evaluate equidistant Qk basis and reference gradients at unit-square points.

    Nodes and basis functions are ordered first in x, then in y. Polynomial
    derivatives are evaluated analytically, including exactly at nodal points.
    """
    degree = positive_int(degree, "degree")
    if np.iscomplexobj(points):
        raise ValueError("reference points must be real")
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not np.isfinite(points).all()
        or np.any(points < -1e-12)
        or np.any(points > 1 + 1e-12)
    ):
        raise ValueError("reference points must be finite with shape (n,2) in [0,1]^2")
    x, dx = _cardinal_values(degree, points[:, 0])
    y, dy = _cardinal_values(degree, points[:, 1])
    basis = np.einsum("qi,qj->qij", y, x).reshape(len(points), (degree + 1) ** 2)
    gradient = np.stack((np.einsum("qi,qj->qij", y, dx), np.einsum("qi,qj->qij", dy, x)), axis=-1)
    return basis, gradient.reshape(len(points), (degree + 1) ** 2, 2)


def qk_space(mesh: CartesianMacroMesh, degree: int) -> tuple[IntArray, FloatArray]:
    """Return the shared nodal DOFs and coordinates of a continuous Qk space."""
    degree = positive_int(degree, "degree")
    nx, ny = mesh.nx * degree, cast(int, mesh.ny) * degree
    x0, x1, y0, y1 = mesh.bounds
    nodes = np.array(
        [(x, y) for y in np.linspace(y0, y1, ny + 1) for x in np.linspace(x0, x1, nx + 1)]
    )
    local = np.array([j * (nx + 1) + i for j in range(degree + 1) for i in range(degree + 1)])
    origin = np.array(
        [
            j * degree * (nx + 1) + i * degree
            for j in range(cast(int, mesh.ny))
            for i in range(mesh.nx)
        ]
    )
    return (origin[:, None] + local).astype(np.int64), nodes


def _grid_resolves_material(mesh: CartesianMacroMesh, field: CartesianCellField) -> bool:
    """Determine whether every material line coincides with a fine-grid line."""
    if len(field.spacing) != 2:
        raise ValueError("quadrilateral material integration requires a two-dimensional field")
    spacing = mesh.spacing
    intervals = np.asarray(field.spacing) / spacing
    bounds = np.asarray(mesh.bounds)
    offsets, offset_error = cartesian_coordinates(bounds[[0, 2]], np.asarray(field.origin), spacing)
    _, width_error = coordinate_difference(bounds[[1, 3]], bounds[[0, 2]])
    spacing_error = width_error / np.array([mesh.nx, mesh.ny])
    interval_error = abs(intervals) * spacing_error / (spacing - spacing_error)
    offset_error += abs(offsets) * spacing_error / (spacing - spacing_error)
    values = np.r_[intervals, offsets]
    tolerance = np.maximum(
        64 * np.finfo(float).eps * max(1.0, np.max(abs(values))),
        2 * np.r_[interval_error, offset_error],
    )
    return bool(np.all(abs(values - np.rint(values)) <= tolerance))


def _cartesian_rectangle_quadrature(
    origins: FloatArray,
    spacing: FloatArray,
    field: CartesianCellField,
    order: int,
) -> tuple[FloatArray, FloatArray]:
    """Integrate each rectangle separately on its exact intersections with pixels.

    Reference points have shape (cells, points, 2); weights have shape
    (cells, points) and sum to one on each rectangle. Zero-weight padding
    combines variable intersection counts without extrapolating the material.
    """
    if len(field.spacing) != 2:
        raise ValueError("quadrilateral material integration requires a two-dimensional field")
    pixel_spacing = np.asarray(field.spacing)
    pixel_origin = np.asarray(field.origin)
    shape = np.asarray(field.values.shape[:2])
    lower, lower_error = cartesian_coordinates(origins, pixel_origin, pixel_spacing)
    upper, upper_error = cartesian_coordinates(origins + spacing, pixel_origin, pixel_spacing)
    tolerance = np.maximum(
        64 * np.finfo(float).eps * shape, 2 * np.maximum(lower_error, upper_error)
    )
    if np.any(lower < -tolerance) or np.any(upper > shape + tolerance):
        raise ValueError("a fine rectangle lies outside the Cartesian material field")
    lower = np.where(
        np.isclose(lower, np.rint(lower), rtol=0, atol=tolerance), np.rint(lower), lower
    )
    upper = np.where(
        np.isclose(upper, np.rint(upper), rtol=0, atol=tolerance), np.rint(upper), upper
    )
    first = np.floor(lower).astype(int)
    last = np.ceil(upper).astype(int) - 1
    parts = np.max(last - first + 1, axis=0)
    gauss, weights = leggauss(positive_int(order, "quadrature order"))
    gauss, weights = (gauss + 1) / 2, weights / 2
    axes, axis_weights = [], []
    for axis in range(2):
        indices = first[:, axis, None] + np.arange(parts[axis])
        physical_lo = pixel_origin[axis] + indices * pixel_spacing[axis]
        physical_hi = pixel_origin[axis] + (indices + 1) * pixel_spacing[axis]
        end = origins[:, axis, None] + spacing[axis]
        lo = np.minimum(np.maximum(physical_lo, origins[:, axis, None]), end)
        hi = np.maximum(np.minimum(physical_hi, end), lo)
        a = (lo - origins[:, axis, None]) / spacing[axis]
        width = (hi - lo) / spacing[axis]
        axes.append(a[..., None] + width[..., None] * gauss)
        axis_weights.append(width[..., None] * weights)
    count = len(origins)
    x = np.broadcast_to(axes[0][:, None, None, :, :], (count, parts[1], order, parts[0], order))
    y = np.broadcast_to(axes[1][:, :, :, None, None], x.shape)
    combined_weights = axis_weights[1][:, :, :, None, None] * axis_weights[0][:, None, None, :, :]
    reference = np.stack((x, y), axis=-1).reshape(count, -1, 2)
    return np.clip(reference, 0, 1), combined_weights.reshape(count, -1)


def quadrilateral_operators(
    mesh: CartesianMacroMesh,
    degree: int = 1,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    order: int = 4,
) -> tuple[Any, Any, FloatArray]:
    """Assemble sparse Qk diffusion, mass and load using bounded cell batches.

    CartesianCellField coefficients are integrated over exact intersections
    with material pixels, retaining a uniform-rule fast path on aligned grids.
    Arbitrary callback discontinuities are not inferred: those require an
    aligned grid or a declared quadrature-sensitivity study.
    """
    degree = positive_int(degree, "degree")
    order = max(positive_int(order, "order"), degree + 1)
    reference, weights = quadrilateral_quadrature(order)
    dofs, nodes = qk_space(mesh, degree)
    basis, gradient = qk_basis(degree, reference)
    gradient = gradient / mesh.spacing
    area = float(np.prod(mesh.spacing))
    width = basis.shape[1]
    n = len(nodes)
    cuts = isinstance(permeability, CartesianCellField) and not _grid_resolves_material(
        mesh, permeability
    )
    blocks, loads = [], []
    for begin in range(0, len(mesh.cells), 256):
        origins = mesh.points[mesh.cells[begin : begin + 256, 0]]
        if cuts:
            cell_reference, cell_weights = _cartesian_rectangle_quadrature(
                origins, mesh.spacing, permeability, order
            )
            cell_basis, cell_gradient = qk_basis(degree, cell_reference.reshape(-1, 2))
            cell_basis = cell_basis.reshape(len(origins), -1, width)
            cell_gradient = cell_gradient.reshape(len(origins), -1, width, 2) / mesh.spacing
            points = origins[:, None, :] + cell_reference * mesh.spacing
            tensors = tensor_values(permeability, points.reshape(-1, 2)).reshape(
                len(origins), -1, 2, 2
            )
            force = scalar_values(source, points.reshape(-1, 2)).reshape(len(origins), -1)
            blocks.append(
                area
                * np.einsum(
                    "tq,tqia,tqab,tqjb->tij",
                    cell_weights,
                    cell_gradient,
                    tensors,
                    cell_gradient,
                    optimize=True,
                )
            )
            loads.append(area * np.einsum("tq,tqi,tq->ti", cell_weights, cell_basis, force))
        else:
            points = origins[:, None, :] + reference[None, :, :] * mesh.spacing
            tensors = tensor_values(permeability, points.reshape(-1, 2)).reshape(
                len(origins), len(weights), 2, 2
            )
            force = scalar_values(source, points.reshape(-1, 2)).reshape(len(origins), len(weights))
            blocks.append(
                area
                * np.einsum(
                    "q,qia,tqab,qjb->tij", weights, gradient, tensors, gradient, optimize=True
                )
            )
            loads.append(area * np.einsum("q,qi,tq->ti", weights, basis, force))
    row = np.repeat(dofs, width, axis=1).ravel()
    column = np.tile(dofs, (1, width)).ravel()
    matrix = sparse.coo_matrix(
        (np.concatenate(blocks).ravel(), (row, column)), shape=(n, n)
    ).tocsc()
    mass_block = area * np.einsum("q,qi,qj->ij", weights, basis, basis)
    mass = sparse.coo_matrix(
        (np.tile(mass_block.ravel(), len(dofs)), (row, column)), shape=(n, n)
    ).tocsc()
    load = np.bincount(dofs.ravel(), weights=np.concatenate(loads).ravel(), minlength=n)
    return matrix, mass, load


def quadrilateral_trace_coupling(
    coarse: CartesianMacroMesh,
    cell: int,
    fine: CartesianMacroMesh,
    skeleton: SkeletonSpace,
    degree: int,
) -> FloatArray:
    """Integrate signed Qk traces, splitting at fine and skeleton breakpoints."""
    _, nodes = qk_space(fine, degree)
    width = sum(skeleton.faces[f].size for f in coarse.cell_faces[cell])
    matrix = np.zeros((len(nodes), width))
    offset = 0
    nx, ny = fine.nx * degree, cast(int, fine.ny) * degree
    # Coordinate-ordered boundary nodes; the global face parameter may reverse them.
    edge_nodes = (
        np.arange(nx + 1),
        np.arange(ny + 1) * (nx + 1) + nx,
        ny * (nx + 1) + np.arange(nx + 1),
        np.arange(ny + 1) * (nx + 1),
    )
    for side, face in enumerate(coarse.cell_faces[cell]):
        space = skeleton.faces[face]
        ids = edge_nodes[side]
        start, end = coarse.points[coarse.faces[face]]
        tangent = end - start
        length = coarse.lengths[face]
        parameter_nodes = (nodes[ids] - start) @ tangent / length**2
        gauss, w = leggauss(max(space.degrees) + degree + 1)
        for begin in range(0, len(ids) - 1, degree):
            indices = ids[begin : begin + degree + 1]
            t0, t1 = parameter_nodes[begin], parameter_nodes[begin + degree]
            lo, hi = sorted((float(t0), float(t1)))
            cuts = sorted({lo, hi, *(b for b in space.breaks if lo < b < hi)})
            for left, right in zip(cuts[:-1], cuts[1:], strict=True):
                parameter = left + (gauss + 1) * (right - left) / 2
                s = (parameter - t0) / (t1 - t0)
                local_basis, _ = _cardinal_values(degree, s)
                weights = w * (right - left) / 2 * length * coarse.signs[cell, side]
                matrix[indices, offset : offset + space.size] += local_basis.T @ (
                    weights[:, None] * space.evaluate(parameter)
                )
        offset += space.size
    return matrix


@dataclass(frozen=True)
class _QuadTask:
    """Picklable geometry and coefficients for one complete worker-local solve."""

    mesh: CartesianMacroMesh
    cell: int
    refinement: tuple[int, int]
    skeleton: SkeletonSpace
    degree: int
    permeability: Any
    source: Any
    order: int


def _assemble_quad(task: _QuadTask) -> LocalAssembly:
    """Build one original Qk Neumann problem and return reconstruction metadata."""
    fine = task.mesh.submesh(task.cell, task.refinement)
    matrix, mass, load = quadrilateral_operators(
        fine, task.degree, permeability=task.permeability, source=task.source, order=task.order
    )
    coupling = quadrilateral_trace_coupling(task.mesh, task.cell, fine, task.skeleton, task.degree)
    kernel = np.ones((matrix.shape[0], 1))
    constraints = mass @ kernel
    return LocalAssembly(
        LocalProblem(
            matrix, coupling, load, task.skeleton.cell_dofs(task.cell), kernel, constraints
        ),
        (fine, constraints[:, 0]),
    )


@dataclass(frozen=True)
class QuadrilateralDarcySolution:
    """Broken continuous-Qk pressure and conservative macro normal-flux trace.

    ``flux`` stores samples of the raw physical field −K grad p at fine-cell
    centers. It is generally not H(div)-conforming. ``evaluate`` preserves
    distinct values on neighboring macrocells, without averaging.
    """

    skeleton: SkeletonSpace
    local_meshes: tuple[CartesianMacroMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    permeability: Any
    source: Any
    degree: int
    quadrature_order: int

    def evaluate(self, cell: int, points: Any) -> tuple[FloatArray, FloatArray]:
        """Sample pressure and physical flux inside one specified macrocell.

        At fine-cell interfaces the cell on the positive coordinate side is
        selected, except on the outer boundary. No interpolation across a
        permeability jump or a macroface is performed.
        """
        positive_int(cell, "cell", 0)
        if cell >= len(self.local_meshes):
            raise ValueError("cell index outside mesh")
        if np.iscomplexobj(points):
            raise ValueError("sample points must be real")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("sample points must be finite with shape (n,2)")
        mesh = self.local_meshes[cell]
        x0, x1, y0, y1 = mesh.bounds
        coordinates = (points - [x0, y0]) / mesh.spacing
        counts = np.array([mesh.nx, mesh.ny], dtype=int)
        if np.any(coordinates < -1e-10) or np.any(coordinates > counts + 1e-10):
            raise ValueError("sample points lie outside the selected macrocell")
        nearest = np.rint(coordinates)
        coordinates = np.where(
            np.isclose(coordinates, nearest, rtol=0, atol=32 * np.finfo(float).eps * counts),
            nearest,
            coordinates,
        )
        indices = np.clip(np.floor(coordinates).astype(int), 0, counts - 1)
        reference = np.clip(coordinates - indices, 0, 1)
        width = mesh.nx * self.degree + 1
        offsets = np.array(
            [j * width + i for j in range(self.degree + 1) for i in range(self.degree + 1)]
        )
        origins = (indices[:, 1] * width + indices[:, 0]) * self.degree
        basis, gradients = qk_basis(self.degree, reference)
        coefficients = self.pressure[cell][origins[:, None] + offsets]
        values = np.einsum("qi,qi->q", coefficients, basis)
        grad = np.einsum("qi,qia->qa", coefficients, gradients / mesh.spacing)
        # A discontinuous material must use the same one-sided fine cell as
        # the polynomial gradient, including at a macrocell upper boundary.
        centers = np.array([x0, y0]) + (indices + 0.5) * mesh.spacing
        on_interface = np.isclose(reference, 0, rtol=0, atol=32 * np.finfo(float).eps) | (
            np.isclose(reference, 1, rtol=0, atol=32 * np.finfo(float).eps)
        )
        material_points = np.where(on_interface, np.nextafter(points, centers), points)
        flux = -np.einsum("qab,qb->qa", tensor_values(self.permeability, material_points), grad)
        return values, flux

    def _error(self, exact: Any, order: int, flux: bool) -> float:
        """Integrate one pressure or vector-flux error without cosmetic averaging."""
        reference, weights = quadrilateral_quadrature(order)
        total = 0.0
        evaluator = vector_values if flux else scalar_values
        for cell, mesh in enumerate(self.local_meshes):
            for begin in range(0, len(mesh.cells), 256):
                origins = mesh.points[mesh.cells[begin : begin + 256, 0]]
                points = origins[:, None, :] + reference[None, :, :] * mesh.spacing
                flat = points.reshape(-1, 2)
                values = self.evaluate(cell, flat)[int(flux)]
                difference = values - evaluator(exact, flat)
                squares = np.sum(difference**2, axis=1) if flux else difference**2
                total += float(
                    np.prod(mesh.spacing) * np.sum(squares.reshape(-1, len(weights)) @ weights)
                )
        return float(np.sqrt(total))

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the pressure L2 error with independently chosen quadrature."""
        return self._error(exact, order, False)

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the raw physical-flux L2 error with independent quadrature."""
        return self._error(exact, order, True)

    def conservation_residuals(self, order: int | None = None) -> FloatArray:
        """Return integrated macro trace outflow minus volume source per cell."""
        reference, weights = quadrilateral_quadrature(
            self.quadrature_order if order is None else order
        )
        residuals = []
        for cell, fine in enumerate(self.local_meshes):
            flux = 0.0
            for side, face in enumerate(self.skeleton.mesh.cell_faces[cell]):
                space = self.skeleton.faces[face]
                t, w = space.quadrature(max(space.degrees) + 2)
                flux += (
                    self.skeleton.mesh.signs[cell, side]
                    * self.skeleton.mesh.lengths[face]
                    * ((w @ space.evaluate(t)) @ self.hybrid.trace[self.skeleton.dofs(int(face))])
                )
            integral = 0.0
            for begin in range(0, len(fine.cells), 256):
                points = fine.points[fine.cells[begin : begin + 256, 0], None, :] + (
                    reference[None, :, :] * fine.spacing
                )
                values = scalar_values(self.source, points.reshape(-1, 2))
                integral += float(
                    np.prod(fine.spacing) * np.sum(values.reshape(-1, len(weights)) @ weights)
                )
            residuals.append(float(flux) - integral)
        return np.asarray(residuals)


def solve_darcy_quadrilateral(
    mesh: CartesianMacroMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int | tuple[int, int] = 4,
    degree: int = 1,
    quadrature_order: int = 4,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> QuadrilateralDarcySolution:
    """Solve Darcy MHM on Cartesian macrorectangles with continuous Qk locals.

    Dirichlet pressure is imposed weakly on faces not listed in ``neumann``;
    Neumann values are outward physical normal fluxes. Pure Neumann problems
    require compatibility and impose the physical mean pressure. The local
    refinement may differ between x and y, independently of trace segments.
    Assembly and condensation both execute inside the selected CPU workers.
    Process execution requires picklable coefficient callbacks and the usual
    guarded script entry point. Sparse local operators reuse one factorization
    for every source and trace right-hand side.
    """
    if not isinstance(mesh, CartesianMacroMesh):
        raise TypeError("quadrilateral Darcy requires CartesianMacroMesh")
    degree = positive_int(degree, "degree")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 1)
    refinement = _refinement(local_refinement)
    skeleton = SkeletonSpace(cast(TriangleMesh, mesh)) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("quadrilateral Darcy requires a scalar skeleton on the supplied mesh")
    boundary, fixed = boundary_data(skeleton, dirichlet, neumann, order=order)
    tasks = [
        _QuadTask(mesh, cell, refinement, skeleton, degree, permeability, source, order)
        for cell in range(len(mesh.cells))
    ]
    system = HybridSystem.from_local_factory(
        _assemble_quad,
        tasks,
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    local_meshes, means = zip(*system.local_metadata, strict=True)
    pure_neumann = neumann is not None and set(neumann) == set(mesh.boundary_faces)
    constraints = (
        [system.mean_constraint(means, mean_pressure * float(np.sum(mesh.areas)))]
        if pure_neumann
        else None
    )
    hybrid = system.solve(solver=solver, fixed=fixed, constraints=constraints)
    solution = QuadrilateralDarcySolution(
        skeleton,
        tuple(local_meshes),
        hybrid.fields,
        (),
        hybrid,
        permeability,
        source,
        degree,
        order,
    )
    flux = tuple(
        solution.evaluate(cell, fine.points[fine.cells].mean(axis=1))[1]
        for cell, fine in enumerate(local_meshes)
    )
    return QuadrilateralDarcySolution(
        skeleton,
        tuple(local_meshes),
        hybrid.fields,
        flux,
        hybrid,
        permeability,
        source,
        degree,
        order,
    )
