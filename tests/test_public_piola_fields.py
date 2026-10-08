"""Executed moment bases survive physical field replay and component divergence."""

import pickle
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.fem.hdiv.bdm_family import BDMFamily
from pymhm.fem.hdiv.family_3d import HDiv3DFamily
from pymhm.fem.hdiv.mapped import mapped_rt_basis, mapped_rt_dofs
from pymhm.fem.hdiv.rt import rt_basis, rt_dofs
from pymhm.fem.hdiv.tensor_rt import tensor_rt_basis, tensor_rt_dofs
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.hexahedron import HexMesh
from pymhm.meshes.mixed import AffineMixedMesh, hdiv3d_basis, hdiv3d_dofs
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.fields import DiscreteField
from pymhm.postprocessing.modal import modal_field
from pymhm.postprocessing.piola import ReferenceVectorBasis, hdiv_field, piola_field


@pytest.fixture(params=("rt", "bdm", "tensor", "hex", "tetra-classic", "tetra", "prism"))
def executed(request):
    """Return the actual owner tables and polynomial coordinates for each family."""
    key = request.param
    if key in {"rt", "bdm"}:
        mesh = TriangleMesh.unit_square()
        family, degree, enrichment = ("RT", 2, 0) if key == "rt" else (BDMFamily(2, 1), 0, 0)
    elif key == "tensor":
        mesh, family, degree, enrichment = CartesianMacroMesh(2, 1), "tensor-RT", 1, 1
    elif key == "hex":
        mesh, family, degree, enrichment = HexMesh.unit_cube(), "mapped-RT", 1, 0
    else:
        kind = "prism" if key == "prism" else "tetrahedron"
        mesh = AffineMixedMesh.unit_cube(kind=kind)
        family = HDiv3DFamily(kind, 1, 1) if key == "tetra-classic" else HDiv3DFamily(kind, 2, 2)
        degree = enrichment = 0
    reference = np.full((2, mesh.points.shape[1]), 0.15)
    reference[1] += 0.03
    if isinstance(family, BDMFamily):
        basis, div = family.basis(mesh, np.c_[1 - reference.sum(axis=1), reference])
        dofs = family.dofs(mesh)
    elif isinstance(family, HDiv3DFamily):
        basis, div, _ = hdiv3d_basis(mesh, family, reference)
        dofs = hdiv3d_dofs(mesh, family)
    elif family == "RT":
        basis, div = rt_basis(mesh, degree, np.c_[1 - reference.sum(axis=1), reference])
        dofs = rt_dofs(mesh, degree)
    elif family == "tensor-RT":
        basis, div, _ = tensor_rt_basis(mesh, degree, enrichment, reference)
        dofs = tensor_rt_dofs(mesh, degree, enrichment)
    else:
        basis, div, _ = mapped_rt_basis(mesh, degree, reference)
        dofs = mapped_rt_dofs(mesh, degree)
    if isinstance(mesh, HexMesh):
        physical = mesh.geometry(reference)[0][0]
    elif hasattr(mesh, "jacobian"):
        physical = mesh.points[mesh.cells[0, 0]] + reference @ mesh.jacobian[0].T
    elif isinstance(mesh, CartesianMacroMesh):
        physical = mesh.points[mesh.cells[0, 0]] + reference * mesh.spacing
    else:
        vertices = mesh.points[mesh.cells[0]]
        physical = vertices[0] + reference @ (vertices[1:] - vertices[0])
    return mesh, family, degree, enrichment, physical, basis[0], div[0], dofs


def test_literal_moment_values_tensor_rows_and_divergence_replay(executed, monkeypatch):
    mesh, family, degree, enrichment, physical, basis, div, dofs = executed
    components = mesh.points.shape[1]
    coefficients = np.linspace(-0.4, 0.8, (dofs.max() + 1) * components)
    options = dict(degree=degree, enrichment=enrichment, components=components)
    value = DiscreteField(hdiv_field("stress", mesh, family, **options), coefficients)
    divergence = DiscreteField(
        hdiv_field("stress_divergence", mesh, family, divergence=True, **options), coefficients
    )
    local = coefficients.reshape(-1, components)[dofs[0]]
    expected = np.einsum("qia,ic->qca", basis, local)
    expected_div = div @ local
    owners = np.zeros(len(physical), dtype=int)
    assert_allclose(value.evaluate(physical, cells=owners), expected, atol=1e-12, rtol=1e-10)
    assert_allclose(
        divergence.evaluate(physical, cells=owners), expected_div, atol=1e-12, rtol=1e-10
    )
    archive = pickle.dumps((value, divergence))
    import pymhm.postprocessing.piola as owner

    def forbidden(*args, **kwargs):
        raise AssertionError("replay must consume the archived executed moment basis")

    monkeypatch.setattr(owner, "create_reference_element", forbidden)
    monkeypatch.setattr(owner, "tetrahedral_candidate_coefficients", forbidden)
    monkeypatch.setattr(owner, "bernstein_basis_matrix", forbidden)
    for threads in (1, 2):
        with threadpool_limits(threads):
            restored, restored_div = pickle.loads(archive)
            assert restored.basis_digest == value.basis_digest
            assert not restored.definition.evaluator.reference_basis.matrix.flags.writeable
            assert_allclose(
                restored.evaluate(physical, cells=owners), expected, atol=1e-12, rtol=1e-10
            )
            assert_allclose(
                restored_div.evaluate(physical, cells=owners), expected_div, atol=1e-12, rtol=1e-10
            )


def test_scalar_piola_dofs_orientation_and_input_contract(executed):
    mesh, family, degree, enrichment, physical, basis, div, dofs = executed
    coefficients = np.linspace(-0.2, 0.7, dofs.max() + 1)
    field = hdiv_field("flux", mesh, family, degree=degree, enrichment=enrichment)
    expected = np.einsum("qia,i->qa", basis, coefficients[dofs[0]])
    assert_allclose(field.evaluator(mesh, coefficients, physical), expected, atol=1e-12, rtol=1e-10)
    divergence = hdiv_field(
        "divergence", mesh, family, degree=degree, enrichment=enrichment, divergence=True
    )
    assert_allclose(
        divergence.evaluator(mesh, coefficients, physical),
        div @ coefficients[dofs[0]],
        atol=1e-12,
        rtol=1e-10,
    )
    with pytest.raises(ValueError, match="coefficients"):
        field.evaluator(mesh, coefficients[:-1], physical)
    with pytest.raises(ValueError, match="must agree"):
        piola_field(
            "bad",
            mesh,
            reference_basis=field.evaluator.reference_basis,
            dofs=dofs,
            orientation=np.eye(dofs.shape[1]),
        )
    with pytest.raises(ValueError, match="nonnegative integers"):
        piola_field(
            "bad",
            mesh,
            reference_basis=field.evaluator.reference_basis,
            dofs=dofs.astype(float),
            orientation=field.evaluator.orientation,
        )


@pytest.mark.parametrize("mode", ["unknown", "prism_bernstein"])
def test_explicit_reference_basis_conventions(mode):
    with pytest.raises(ValueError):
        ReferenceVectorBasis("triangle", 1, 2, np.eye(3), np.eye(3), mode)
    with pytest.raises(ValueError, match="matrices"):
        ReferenceVectorBasis("triangle", 1, 2, np.ones(3), np.eye(3))
    with pytest.raises(ValueError):
        ReferenceVectorBasis("triangle", 1, 2, np.eye(3, dtype=complex), np.eye(3))
    with pytest.raises(ValueError, match="unknown"):
        hdiv_field("unknown", TriangleMesh.unit_square(), "not-a-family")


@pytest.mark.parametrize("convention", ["monomial", "legendre"])
def test_modal_complete_tensor_values_gradients_and_one_sided_replay(convention):
    mesh = CartesianMacroMesh(2, 1)
    powers = tuple(product(range(2), repeat=2))
    coefficients = np.arange(2 * len(powers) * 2, dtype=float) / 8
    field = DiscreteField(
        modal_field("u", mesh, powers, convention=convention, components=2), coefficients
    )
    points = np.array([[0.5, 0.3], [0.5, 0.3]])
    cells = np.array([0, 1])
    restored = pickle.loads(pickle.dumps(field))
    assert not np.allclose(
        field.evaluate(points, cells=cells)[0], field.evaluate(points, cells=cells)[1]
    )
    assert_allclose(
        restored.evaluate(points, cells=cells), field.evaluate(points, cells=cells), atol=0
    )
    step = 1e-6
    p = np.array([[0.2, 0.3]])
    derivative = np.stack(
        [
            (field.evaluate(p + step * np.eye(2)[a]) - field.evaluate(p - step * np.eye(2)[a]))
            / (2 * step)
            for a in range(2)
        ],
        axis=-1,
    )
    assert_allclose(field.gradient(p), derivative, atol=1e-9, rtol=1e-9)
    scalar = DiscreteField(modal_field("p", mesh, powers, convention=convention), coefficients[:8])
    assert scalar.evaluate(p).shape == (1,)
    assert scalar.gradient(p).shape == (1, 2)
    with pytest.raises(ValueError, match="coefficients"):
        scalar.definition.evaluator(mesh, [1], p)
    with pytest.raises(ValueError, match="nonnegative"):
        modal_field("bad", mesh, ((-1, 0),))
    with pytest.raises(ValueError, match="convention"):
        modal_field("bad", mesh, powers, convention="unknown")
