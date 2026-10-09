"""Small, auditable triangular finite-element assembly and quadrature kernels."""

from collections.abc import Mapping
from math import fsum
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy import sparse

from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar._integration import (
    boundary_moments,
    diffusion_blocks,
    ordinary_product_range,
)
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.materials.evaluation import (
    scalar_values as scalar_values,
)
from pymhm.materials.evaluation import (
    tensor_values as tensor_values,
)
from pymhm.materials.evaluation import (
    vector_values as vector_values,
)
from pymhm.meshes.triangle import TriangleMesh


def _boundary_moments(
    basis: FloatArray, weights: FloatArray, values: FloatArray
) -> NDArray[np.floating[Any]]:
    """Integrate real boundary moments with compiled binary64 compensation.

    The declared basis, quadrature and field values retain their ordering and
    magnitudes. Exceptional exponents use a wider native real type where
    available and retain that result until the physical face measure is
    applied. Platforms without that type retain compensated real moments.
    No moment is clipped or replaced by a nullspace projection.
    """
    if not all(ordinary_product_range(array) for array in (basis, weights, values)):
        if np.finfo(np.longdouble).eps < np.finfo(float).eps:
            return basis.astype(np.longdouble).T @ (
                weights[:, None].astype(np.longdouble) * values.astype(np.longdouble)
            )
        result = np.array(
            [
                [fsum(basis[:, i] * weights * values[:, j]) for j in range(values.shape[1])]
                for i in range(basis.shape[1])
            ]
        )
        if not np.isfinite(result).all():
            raise ValueError("boundary moment products exceed the native real range")
        return result
    return boundary_moments(
        np.asarray(basis, dtype=float),
        np.asarray(weights, dtype=float),
        np.asarray(values, dtype=float),
    )


def triangle_quadrature(order: int = 4) -> tuple[FloatArray, FloatArray]:
    """Return barycentric Duffy-product Gauss points and unit-sum weights."""
    t, w = leggauss(positive_int(order, "quadrature order"))
    t, w = (t + 1) / 2, w / 2
    bary = np.array([(1 - a - (1 - a) * b, a, (1 - a) * b) for a in t for b in t])
    weights = np.array([2 * wa * wb * (1 - a) for a, wa in zip(t, w, strict=True) for wb in w])
    return bary, weights


def p1_geometry(mesh: TriangleMesh) -> tuple[FloatArray, FloatArray]:
    """Return affine P1 gradients per cell and positive integration areas."""
    vertices = mesh.points[mesh.cells]
    matrices = np.concatenate((np.ones((len(vertices), 3, 1)), vertices), axis=2)
    gradients = np.linalg.inv(matrices)[:, 1:, :].swapaxes(1, 2)
    return gradients, mesh.areas


def _scalar_diffusion_blocks(
    weights: FloatArray,
    gradients: FloatArray,
    tensors: FloatArray,
    measures: FloatArray | float,
) -> FloatArray:
    """Accumulate scalar gradient Gram blocks, then return binary64 entries.

    Trailing axes are q, (q,basis,dimension), (q,dimension,dimension),
    respectively; leading cell axes broadcast. Measures have only cell axes.
    A compiled binary64 kernel compensates Cartesian contractions and the
    quadrature sum. Leading cell axes and singleton quadrature axes broadcast
    before entering that leaf. No kernel projection or operator-entry
    truncation is applied; the represented inputs remain unchanged. Exceptional
    exponents retain native NumPy accumulation within the platform's available
    real range; representable intermediate products remain a platform-dependent
    capability.
    """
    count = np.broadcast_shapes(weights.shape[-1:], gradients.shape[-3:-2], tensors.shape[-3:-2])[0]
    axes = np.broadcast_shapes(
        weights.shape[:-1], gradients.shape[:-3], tensors.shape[:-3], np.shape(measures)
    )
    cells = int(np.prod(axes, dtype=np.int64))
    width, dimension = gradients.shape[-2:]
    w = np.broadcast_to(np.asarray(weights, dtype=float), (*axes, count)).reshape(cells, count)
    g = np.broadcast_to(
        np.asarray(gradients, dtype=float), (*axes, count, width, dimension)
    ).reshape(cells, count, width, dimension)
    k = np.broadcast_to(
        np.asarray(tensors, dtype=float), (*axes, count, dimension, dimension)
    ).reshape(cells, count, dimension, dimension)
    m = np.broadcast_to(np.asarray(measures, dtype=float), axes).reshape(cells)
    if not all(ordinary_product_range(array) for array in (w, g, k, m)):
        return _exceptional_diffusion_blocks(w, g, k, m).reshape(*axes, width, width)
    return diffusion_blocks(w, g, k, m).reshape(*axes, width, width)


def _exceptional_diffusion_blocks(
    weights: FloatArray, gradients: FloatArray, tensors: FloatArray, measures: FloatArray
) -> FloatArray:
    """Retain extreme-exponent products through native real quadrature integration.

    NumPy supplies the tensor contraction in its native extended real type.
    On platforms where that type is binary64 the same product convention and
    compensated quadrature sum retain the portable accumulation contract.
    Only the final operator entries narrow to binary64. A finite final entry
    may involve much larger intermediate products on very small cells.
    """
    w, g, k, m = (
        np.asarray(array, dtype=np.longdouble) for array in (weights, gradients, tensors, measures)
    )
    total = np.zeros((g.shape[0], g.shape[2], g.shape[2]), dtype=np.longdouble)
    correction = np.zeros_like(total)
    for q in range(g.shape[1]):
        block = np.einsum(
            "...,...ia,...ab,...jb,...->...ij",
            w[:, q],
            g[:, q],
            k[:, q],
            g[:, q],
            m,
            optimize=False,
        )
        combined = total + block
        correction += np.where(
            abs(total) >= abs(block), (total - combined) + block, (block - combined) + total
        )
        total = combined
    result = np.asarray(total + correction, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("diffusion products exceed the native real range")
    return result


def p1_operators(
    mesh: TriangleMesh,
    diffusion: Any = 1.0,
    source: Any = 0.0,
    reaction: Any = 0.0,
    advection: Any = (0.0, 0.0),
    order: int = 4,
) -> tuple[Any, Any, FloatArray]:
    """Assemble diffusion, mass and load with conventional advective transport.

    The bilinear form is ``(K grad u,grad v)+(beta.grad u,v)+(c u,v)``.
    This operator is not the skew conservative Robin formulation of RAD-MHM.
    """
    from pymhm.fem.quadrature.material import material_triangle_quadrature

    bary, weights, material = material_triangle_quadrature(mesh, diffusion, order)
    from pymhm.fem.reference import physical_simplex_tabulation

    geometry, areas = p1_geometry(mesh)
    basis, derivatives, _ = physical_simplex_tabulation(
        "triangle", 1, bary, nodes=np.eye(3), reference_gradients=geometry[:, 1:], nderiv=1
    )
    gradients = derivatives[:, 0]
    vertices = mesh.points[mesh.cells]
    points = np.einsum("tqi,tij->tqj", bary, vertices)
    flat = points.reshape(-1, 2)
    tensors = tensor_values(material, flat).reshape(*weights.shape, 2, 2)
    coefficients = scalar_values(reaction, flat).reshape(len(areas), -1)
    if np.any(coefficients < 0):
        raise ValueError("reaction must be nonnegative")
    forces = scalar_values(source, flat).reshape(len(areas), -1)
    velocity = vector_values(advection, flat).reshape(*weights.shape, 2)
    local_mass = np.einsum("tq,tqi,tqj,t->tij", weights, basis, basis, areas)
    blocks = _scalar_diffusion_blocks(weights, gradients[:, None], tensors, areas)
    blocks += np.einsum("tq,tq,tqi,tqj,t->tij", weights, coefficients, basis, basis, areas)
    blocks += np.einsum("tq,tqi,tqa,tja,t->tij", weights, basis, velocity, gradients, areas)
    rhs = np.einsum("tq,tq,tqi,t->ti", weights, forces, basis, areas)
    n = len(mesh.points)
    rows = np.repeat(mesh.cells, 3, axis=1).ravel()
    cols = np.tile(mesh.cells, (1, 3)).ravel()
    matrix = sparse.coo_matrix((blocks.ravel(), (rows, cols)), shape=(n, n)).tocsc()
    mass = sparse.coo_matrix((local_mass.ravel(), (rows, cols)), shape=(n, n)).tocsc()
    load = np.bincount(mesh.cells.ravel(), weights=rhs.ravel(), minlength=n)
    return matrix, mass, load


def face_integration(
    mesh: TriangleMesh, cell: int, fine: TriangleMesh, skeleton: SkeletonSpace
) -> tuple[FloatArray, FloatArray]:
    """Integrate skeleton basis against P1 traces and fine boundary flux DOFs.

    Quadrature is split at both fine-mesh vertices and skeleton breaks. The
    second result maps trace coefficients to integrated outward RT0 boundary
    fluxes, ordered by ``fine.boundary_faces``.
    """
    ndofs = sum(skeleton.faces[f].size for f in mesh.cell_faces[cell])
    p1 = np.zeros((len(fine.points), ndofs))
    rt = np.zeros((len(fine.boundary_faces), ndofs))
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        space = skeleton.faces[face]
        start, end = mesh.points[mesh.faces[face]]
        tangent = end - start
        length2 = tangent @ tangent
        sign = mesh.signs[cell, side]
        order = max(space.degrees) + 2
        gauss, gauss_weights = leggauss(order)
        for row, fine_face in enumerate(fine.boundary_faces):
            nodes = fine.faces[fine_face]
            coords = fine.points[nodes]
            t = (coords - start) @ tangent / length2
            projected = start + t[:, None] * tangent
            if (
                not np.allclose(coords, projected, atol=1e-12, rtol=1e-12)
                or min(t) < -1e-12
                or max(t) > 1 + 1e-12
            ):
                continue
            lo, hi = np.clip(np.sort(t), 0, 1)
            breaks = sorted({lo, hi, *(b for b in space.breaks if lo < b < hi)})
            for left, right in zip(breaks[:-1], breaks[1:], strict=True):
                parameter = left + (gauss + 1) * (right - left) / 2
                weights = gauss_weights * (right - left) / 2 * np.sqrt(length2) * sign
                basis = space.evaluate(parameter)
                from pymhm.fem.traces.scalar import edge_basis

                nodal = edge_basis(1, (parameter - t[0]) / (t[1] - t[0]))
                p1[nodes, offset : offset + space.size] += nodal.T @ (weights[:, None] * basis)
                rt[row, offset : offset + space.size] += weights @ basis
        offset += space.size
    return p1, rt


def boundary_data(
    skeleton: SkeletonSpace,
    dirichlet: Any,
    neumann: dict[int, Any] | None = None,
    *,
    neumann_components: dict[int, dict[int, Any]] | None = None,
    order: int = 5,
) -> tuple[FloatArray, dict[int, float]]:
    """Integrate primary fields and L2-project full or componentwise boundary fluxes.

    ``neumann_components[face][component]`` prescribes a Cartesian component
    of the multiplier as a scalar field. Other components retain Dirichlet
    data. Full and componentwise Neumann declarations cannot share a face.
    Components refer to the skeleton's interleaved Cartesian ordering; no
    normal/tangential rotation or outward-sign conversion is performed here.
    """
    positive_int(order, "boundary quadrature order")
    neumann = {} if neumann is None else neumann
    neumann_components = {} if neumann_components is None else neumann_components
    boundary_faces = set(skeleton.mesh.boundary_faces)
    for data in (neumann, neumann_components):
        if not isinstance(data, Mapping):
            raise ValueError("Neumann data must map boundary faces to prescribed fields")
        faces = {positive_int(face, "boundary face", 0) for face in data}
        if not faces.issubset(boundary_faces):
            raise ValueError("Neumann data may only be imposed on boundary faces")
    if set(neumann) & set(neumann_components):
        raise ValueError("full and componentwise Neumann data must not overlap on a face")
    for components in neumann_components.values():
        if not isinstance(components, Mapping) or not components:
            raise ValueError("component Neumann data must be a nonempty component-to-field map")
        for component in components:
            if positive_int(component, "boundary component", 0) >= skeleton.components:
                raise ValueError("boundary component lies outside the skeleton components")
    load = np.zeros(skeleton.size)
    fixed: dict[int, float] = {}
    for face in skeleton.mesh.boundary_faces:
        space = skeleton.faces[face]
        parameter, weights = space.quadrature(max(order, max(space.degrees) + 2))
        start, end = skeleton.mesh.points[skeleton.mesh.faces[face]]
        points = start + parameter[:, None] * (end - start)
        basis = space.evaluate(parameter)
        evaluator = scalar_values if skeleton.components == 1 else vector_values
        prescribed = np.zeros(skeleton.components, dtype=bool)
        if face in neumann:
            values = evaluator(neumann[face], points).reshape(len(points), skeleton.components)
            prescribed[:] = True
        else:
            components = neumann_components.get(face, {})
            values = (
                evaluator(dirichlet, points).reshape(len(points), skeleton.components)
                if len(components) < skeleton.components
                else np.empty((len(points), skeleton.components))
            )
            for component, field in components.items():
                values[:, component] = scalar_values(field, points)
                prescribed[component] = True
        # Keep the same quadrature in the trace equations and volume constraint.
        # Compensate the reduction so opposite oriented faces do not acquire
        # different rounding errors in their constant displacement moments.
        moments = _boundary_moments(basis, weights, values)
        dofs = skeleton.dofs(int(face)).reshape(-1, skeleton.components)
        if np.any(prescribed):
            coefficients = np.linalg.solve(
                basis.T @ (weights[:, None] * basis),
                np.asarray(moments[:, prescribed], dtype=float),
            ).ravel()
            fixed.update(
                zip(dofs[:, prescribed].ravel().tolist(), coefficients.tolist(), strict=True)
            )
        if not np.all(prescribed):
            load[dofs[:, ~prescribed].ravel()] = (
                moments[:, ~prescribed] * skeleton.mesh.lengths[face]
            ).ravel()
    return load, fixed


def rt0_operators(
    mesh: TriangleMesh, permeability: Any = 1.0, source: Any = 0.0, order: int = 4
) -> tuple[Any, Any, FloatArray]:
    """Assemble RT0 flux mass, integrated divergence and P0 source moments.

    A flux DOF is the integral across a globally oriented face. In a cell the
    basis associated with side (i,j) is ``sign*(x-opposite)/(2*area)``.
    """
    from pymhm.fem.hdiv.rt import rt_basis
    from pymhm.fem.quadrature.material import material_triangle_quadrature

    bary, weights, material = material_triangle_quadrature(mesh, permeability, order)
    vertices = mesh.points[mesh.cells]
    points = np.einsum("tqi,tij->tqj", bary, vertices)
    inverse = np.linalg.inv(tensor_values(material, points.reshape(-1, 2))).reshape(
        *weights.shape, 2, 2
    )
    basis = rt_basis(mesh, 0, bary)[0]
    blocks = np.einsum("tq,tqia,tqab,tqjb,t->tij", weights, basis, inverse, basis, mesh.areas)
    nfaces = len(mesh.faces)
    mass = sparse.coo_matrix(
        (
            blocks.ravel(),
            (
                np.repeat(mesh.cell_faces, 3, axis=1).ravel(),
                np.tile(mesh.cell_faces, (1, 3)).ravel(),
            ),
        ),
        shape=(nfaces, nfaces),
    ).tocsc()
    divergence = sparse.coo_matrix(
        (mesh.signs.ravel(), (np.repeat(np.arange(len(vertices)), 3), mesh.cell_faces.ravel())),
        shape=(len(vertices), nfaces),
    ).tocsc()
    load = (
        np.sum(
            scalar_values(source, points.reshape(-1, 2)).reshape(weights.shape) * weights, axis=1
        )
        * mesh.areas
    )
    return mass, divergence, load


def rt0_evaluate(mesh: TriangleMesh, flux: FloatArray, barycentric: FloatArray) -> FloatArray:
    """Evaluate RT0 at common (q,3) or cellwise (t,q,3) barycentric points."""
    from pymhm.fem.hdiv.rt import rt_basis

    basis = rt_basis(mesh, 0, barycentric)[0]
    return np.einsum("tqia,ti->tqa", basis, flux[mesh.cell_faces])
