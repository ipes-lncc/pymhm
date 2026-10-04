"""Three-dimensional AFW compliance, divergence and weak-symmetry moments on tetrahedra."""

from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core.validation import FloatArray
from pymhm.fem.assembly import assemble_element_blocks as _scatter
from pymhm.fem.hdiv.family_3d import (
    HDiv3DFamily,
    cell_quadrature,
    face_polynomials,
    face_quadrature,
    face_shape,
)
from pymhm.materials.evaluation import scalar_values_3d, vector_values_3d
from pymhm.meshes.mixed import AffineMixedMesh, hdiv3d_basis, hdiv3d_dofs, hdiv3d_face_offsets


def rigid_values(points: FloatArray, center: FloatArray) -> FloatArray:
    """Evaluate translations and e_i cross (x-center), ordered as three plus three modes."""
    values = np.zeros((*points.shape[:-1], 3, 6))
    values[..., :3] = np.eye(3)
    for axis in range(3):
        values[..., :, 3 + axis] = np.cross(np.eye(3)[axis], points - center)
    return values


def compliance_action(
    tensors: FloatArray, points: FloatArray, lame_lambda: Any, lame_mu: Any, compliance: Any
) -> tuple[FloatArray, float]:
    """Apply full-tensor compliance, retaining a cancellation-free isotropic deviatoric split.

    General Cartesian compliance has shape (9,9), acts on row-major tensor
    entries, is self-adjoint and positive definite, and preserves symmetric
    and skew subspaces. Its skew extension is explicit. Isotropic lambda may
    be infinite; mu must be finite and positive. The returned scale is the
    largest sampled spherical compliance tr(A I)/3.
    """
    shape = tensors.shape[:2]
    if compliance is not None:
        raw = compliance(points) if callable(compliance) else compliance
        if np.iscomplexobj(raw):
            raise ValueError("stress compliance must be real")
        material = np.asarray(raw, dtype=float)
        if material.shape[-2:] != (9, 9):
            raise ValueError("3D stress compliance requires Cartesian (9,9) shape")
        material = np.broadcast_to(material, (len(points), 9, 9)).reshape(*shape, 9, 9)
        if not np.isfinite(material).all():
            raise ValueError("stress compliance must be finite")
        tolerance = 64 * np.finfo(float).eps * np.max(abs(material), axis=(-2, -1))
        if np.any(np.max(abs(material - material.swapaxes(-1, -2)), axis=(-2, -1)) > tolerance):
            raise ValueError("stress compliance must be self-adjoint")
        transpose = np.arange(9).reshape(3, 3).T.ravel()
        if np.any(
            np.max(abs(material - material[..., transpose, :][..., transpose]), axis=(-2, -1))
            > tolerance
        ):
            raise ValueError("stress compliance must preserve symmetric and skew tensors")
        if np.any(np.linalg.eigvalsh(material)[..., 0] <= 0):
            raise ValueError("stress compliance must be positive definite on full tensors")
        action = np.einsum("tqab,tqib->tqia", material, tensors.reshape(*tensors.shape[:3], 9))
        identity = np.eye(3).ravel()
        scale = float(np.max(np.einsum("a,tqab,b->tq", identity, material, identity)) / 3)
        return action.reshape(tensors.shape), scale
    mu = scalar_values_3d(lame_mu, points).reshape(shape)
    raw = lame_lambda(points) if callable(lame_lambda) else lame_lambda
    if np.iscomplexobj(raw):
        raise ValueError("Lamé lambda must be real")
    lam = np.broadcast_to(np.asarray(raw, dtype=float), (len(points),)).reshape(shape)
    if np.any(mu <= 0) or np.any(np.isnan(lam)) or np.any(lam < 0):
        raise ValueError("Lamé mu must be positive and lambda nonnegative, including infinity")
    scale = np.maximum(mu, lam)
    inverse_bulk = (1 / scale) / (
        2 * (mu / scale) + 3 * np.divide(lam, scale, out=np.ones_like(lam), where=np.isfinite(lam))
    )
    trace = np.trace(tensors, axis1=-2, axis2=-1)
    deviator = tensors - trace[..., None, None] * np.eye(3) / 3
    action = deviator / (2 * mu[..., None, None, None])
    action += inverse_bulk[..., None, None, None] * trace[..., None, None] * np.eye(3) / 3
    return action, float(inverse_bulk.max())


def mixed_elasticity_operators_3d(
    mesh: AffineMixedMesh,
    family: HDiv3DFamily,
    *,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    source: Any = (0.0, 0.0, 0.0),
    compliance: Any = None,
    quadrature_order: int = 5,
) -> tuple:
    """Assemble AFW stress compliance, displacement divergence and three skew moments.

    Stress rows use full BDM_k and both displacement and axial weak rotation
    use discontinuous P_(k-1). Row/coordinate components are interleaved within
    each scalar/vector moment. Source is the physical force in -div(sigma)=f.
    Returns M,D,R,f,plain stress-trace moments, weighted trace moments and the
    physical bulk-compliance gauge scale.
    """
    if (
        mesh.kind != "tetrahedron"
        or family.kind != mesh.kind
        or family.normal_degree != family.pressure_degree + 1
    ):
        raise ValueError("AFW elasticity requires complete tetrahedral BDM_k/P_(k-1) spaces")
    xi, weights = cell_quadrature(mesh.kind, quadrature_order)
    vectors, divergence, scalar = hdiv3d_basis(mesh, family, xi)
    physical = mesh.geometry(xi)
    nc, nq, width = vectors.shape[:3]
    tensors = np.zeros((nc, nq, 3 * width, 3, 3))
    for row in range(3):
        tensors[:, :, row::3, row] = vectors
    action, scale = compliance_action(
        tensors, physical.reshape(-1, 3), lame_lambda, lame_mu, compliance
    )
    w = mesh.determinants[:, None] * weights
    mass = np.einsum("tq,tqiab,tqjab->tij", w, action, tensors, optimize=True)
    scalar_div = np.einsum("tq,qi,tqj->tij", w, scalar, divergence, optimize=True)
    nscl = scalar.shape[1]
    div = np.zeros((nc, 3 * nscl, 3 * width))
    for row in range(3):
        div[:, row::3, row::3] = scalar_div
    asym = np.zeros_like(div)
    for row, (i, j) in enumerate(((1, 2), (2, 0), (0, 1))):
        asym[:, row::3] = np.einsum(
            "tq,qi,tqj->tij", w, scalar, tensors[..., i, j] - tensors[..., j, i]
        )
    force = np.einsum(
        "tq,qi,tqa->tia",
        w,
        scalar,
        vector_values_3d(source, physical.reshape(-1, 3)).reshape(nc, nq, 3),
    )
    dofs = (3 * hdiv3d_dofs(mesh, family)[:, :, None] + np.arange(3)).reshape(nc, -1)
    ns, nu = int(dofs.max()) + 1, 3 * nscl * nc
    uid = np.arange(nu).reshape(nc, -1)
    plain = np.bincount(
        dofs.ravel(),
        weights=np.einsum("tq,tqi->ti", w, np.trace(tensors, axis1=-2, axis2=-1)).ravel(),
        minlength=ns,
    )
    weighted = np.bincount(
        dofs.ravel(),
        weights=np.einsum("tq,tqi->ti", w, np.trace(action, axis1=-2, axis2=-1)).ravel(),
        minlength=ns,
    )
    return (
        _scatter(mass, dofs, dofs, (ns, ns)),
        _scatter(div, uid, dofs, (nu, ns)),
        _scatter(asym, uid, dofs, (nu, ns)),
        force.ravel(),
        plain,
        weighted,
        scale,
    )


def rigid_coefficients(
    mesh: AffineMixedMesh, family: HDiv3DFamily, center: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return exact rigid DG coefficients, physical moments and fine-face dual values."""
    xi, weights = cell_quadrature(mesh.kind, family.pressure_degree + 2)
    scalar = family.tabulate(xi)[2]
    physical = mesh.geometry(xi)
    fields = rigid_values(physical, center)
    moments = np.einsum("q,qi,tqak,t->tiak", weights, scalar, fields, mesh.determinants)
    mass = scalar.T @ (weights[:, None] * scalar)
    coefficients = np.linalg.solve(mass, moments.transpose(1, 0, 2, 3).reshape(len(mass), -1))
    coefficients = coefficients.reshape(len(mass), len(mesh.cells), 3, 6).transpose(1, 0, 2, 3)
    coefficients /= mesh.determinants[:, None, None, None]
    boundary = []
    uv, face_weights = face_quadrature(3, family.normal_degree + 2)
    tests = face_polynomials(uv, 3, family.normal_degree)
    face_mass = tests.T @ (face_weights[:, None] * tests)
    for face in mesh.boundary_faces:
        points = face_shape(uv, 3) @ mesh.points[mesh.faces[face]]
        moments_face = np.einsum("q,qi,qak->iak", face_weights, tests, rigid_values(points, center))
        boundary.append(
            np.linalg.solve(face_mass, moments_face.reshape(len(tests[0]), -1)).reshape(-1, 3, 6)
        )
    return (
        coefficients.reshape(-1, 6),
        moments.reshape(-1, 6),
        np.concatenate(boundary).reshape(-1, 6),
    )


def boundary_selector(mesh: AffineMixedMesh, family: HDiv3DFamily, size: int) -> Any:
    """Select interleaved physical canonical stress-row moments on all fine exterior faces."""
    offsets = hdiv3d_face_offsets(mesh, family)
    scalar = np.concatenate([np.arange(offsets[f], offsets[f + 1]) for f in mesh.boundary_faces])
    rows = (3 * scalar[:, None] + np.arange(3)).ravel()
    return sparse.coo_matrix(
        (np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(size, len(rows))
    ).tocsc()
