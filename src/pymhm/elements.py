"""Small, auditable triangular finite-element assembly and quadrature kernels."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.mesh import FloatArray, SkeletonSpace, TriangleMesh, positive_int


def triangle_quadrature(order: int = 4) -> tuple[FloatArray, FloatArray]:
    """Return barycentric Duffy-product Gauss points and unit-sum weights."""
    t, w = leggauss(positive_int(order, "quadrature order"))
    t, w = (t + 1) / 2, w / 2
    bary = np.array([(1 - a - (1 - a) * b, a, (1 - a) * b) for a in t for b in t])
    weights = np.array([2 * wa * wb * (1 - a) for a, wa in zip(t, w, strict=True) for wb in w])
    return bary, weights


def scalar_values(field: Any, points: FloatArray) -> FloatArray:
    """Evaluate a finite real scalar coefficient on points shaped ``(n, 2)``."""
    value = field(points) if callable(field) else field
    if np.iscomplexobj(value):
        raise ValueError("scalar coefficient must be real")
    result = np.broadcast_to(np.asarray(value, dtype=float), (len(points),)).copy()
    if not np.isfinite(result).all():
        raise ValueError("scalar coefficient must be finite")
    return result


def vector_values(field: Any, points: FloatArray) -> FloatArray:
    """Evaluate a finite real two-component coefficient with point-major storage."""
    value = field(points) if callable(field) else field
    if np.iscomplexobj(value):
        raise ValueError("vector coefficient must be real")
    result = np.broadcast_to(np.asarray(value, dtype=float), (len(points), 2)).copy()
    if not np.isfinite(result).all():
        raise ValueError("vector coefficient must be finite")
    return result


def tensor_values(field: Any, points: FloatArray) -> FloatArray:
    """Evaluate an isotropic scalar or symmetric positive-definite 2x2 tensor."""
    raw = field(points) if callable(field) else field
    if np.iscomplexobj(raw):
        raise ValueError("diffusion tensor must be real")
    value = np.asarray(raw, dtype=float)
    if value.ndim == 0 or value.shape == (len(points),):
        result = np.broadcast_to(value, (len(points),))[:, None, None] * np.eye(2)
    else:
        result = np.broadcast_to(value, (len(points), 2, 2)).copy()
    if (
        not np.isfinite(result).all()
        or not np.allclose(result, result.swapaxes(1, 2), rtol=1e-12, atol=1e-14)
        or np.any(np.linalg.eigvalsh(result) <= 0)
    ):
        raise ValueError("diffusion tensor must be finite, symmetric and positive definite")
    return result


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
    NumPy's widest real type accumulates the quadrature and tensor contractions
    before the final binary64 rounding. No kernel projection or operator-entry
    truncation is applied. On platforms with binary64 longdouble this uses
    that platform's real precision.
    """
    operands = tuple(
        np.asarray(value, dtype=np.longdouble) for value in (weights, gradients, tensors, measures)
    )
    return np.asarray(
        np.einsum(
            "...q,...qia,...qab,...qjb,...->...ij",
            *operands[:3],
            operands[1],
            operands[3],
            optimize=True,
        ),
        dtype=np.float64,
    )


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
    from pymhm.cut_cells import material_triangle_quadrature

    bary, weights, material = material_triangle_quadrature(mesh, diffusion, order)
    from pymhm.element_backends import physical_simplex_tabulation

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
                from pymhm.scalar_boundary import edge_basis

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
        # Accumulate before narrowing so opposite oriented faces do not acquire
        # different rounding errors in their constant displacement moments.
        moments = basis.astype(np.longdouble).T @ (
            weights[:, None].astype(np.longdouble) * values.astype(np.longdouble)
        )
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
    from pymhm.cut_cells import material_triangle_quadrature
    from pymhm.rt import rt_basis

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
    from pymhm.rt import rt_basis

    basis = rt_basis(mesh, 0, barycentric)[0]
    return np.einsum("tqia,ti->tqa", basis, flux[mesh.cell_faces])
