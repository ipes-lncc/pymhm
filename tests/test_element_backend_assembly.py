"""Native tabulators preserve physical MHM fields and declared global gauges."""

from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm.assembly import HybridProblem, assemble_hybrid
from pymhm.darcy3d import TriangularSkeleton, _boundary, tetra_trace_coupling
from pymhm.elements import boundary_data
from pymhm.hybrid import LocalAssembly, LocalProblem
from pymhm.lagrange import element_tabulate, nodal_space, scalar_operators, trace_coupling
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.tetrahedral import TetraMesh, tetra_nodal_space, tetra_operators
from pymhm.variational import GlobalForm


def exact_pressure(points):
    """Return ``1+sum(x_i**2)``; its independently differentiated source is -2d."""
    return 1 + np.sum(points * points, axis=1)


def build_cell(cell, *, macro, skeleton, backend, homogeneous):
    """Use existing assembly/trace owners with an explicitly selected tabulator."""
    fine = macro.submesh(cell, 2)
    source = 0.0 if homogeneous else -2.0 * macro.points.shape[1]
    if isinstance(macro, TriangleMesh):
        matrix, mass, load = scalar_operators(fine, 2, source=source, element_backend=backend)
        coupling = trace_coupling(macro, cell, fine, skeleton, 2)
        _, nodes = nodal_space(fine, 2)
    else:
        matrix, mass, load = tetra_operators(fine, 2, source=source, element_backend=backend)
        coupling = tetra_trace_coupling(macro, cell, fine, skeleton, 2)
        _, nodes = tetra_nodal_space(fine, 2)
    moments = np.asarray(mass.sum(axis=1)).reshape(-1, 1)
    problem = LocalProblem(
        matrix,
        coupling,
        load,
        skeleton.cell_dofs(cell),
        kernel=np.ones((len(nodes), 1)),
        constraints=moments,
    )
    return LocalAssembly(problem, {"nodes": nodes})


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("boundary", ["homogeneous", "nonhomogeneous", "neumann"])
def test_basix_assembled_cells_preserve_physical_fields_and_moments(dimension, boundary):
    """Check full oriented multi-cell equations, not only standalone basis values."""
    pytest.importorskip("basix")
    homogeneous = boundary == "homogeneous"
    macro = TriangleMesh.unit_square() if dimension == 2 else TetraMesh.unit_cube()
    skeleton = (
        SkeletonSpace(macro, tuple(FaceSpace.uniform(1) for _ in macro.faces))
        if dimension == 2
        else TriangularSkeleton(macro, degree=0)
    )
    natural = {}
    if boundary == "neumann":
        for face in macro.boundary_faces:
            normal = macro.normals[face].copy()
            natural[int(face)] = lambda x, normal=normal: -2 * x @ normal
    pressure = 0.0 if homogeneous else exact_pressure
    if dimension == 2:
        load, fixed = boundary_data(skeleton, pressure, natural, order=5)
    else:
        load, fixed = _boundary(skeleton, pressure, natural, 5)
    form = GlobalForm(skeleton.size, (1,) * len(macro.cells), boundary_load=load)
    systems, solutions = [], []
    for backend in ("portable", "basix"):
        provider = partial(
            build_cell, macro=macro, skeleton=skeleton, backend=backend, homogeneous=homogeneous
        )
        system = assemble_hybrid(HybridProblem(form, provider, range(len(macro.cells))))
        constraints = None
        if boundary == "neumann":
            weights = [r.problem.constraints[:, 0] for r in system.responses]
            constraints = [system.mean_constraint(weights, 1 + dimension / 3)]
        solution = system.solve(fixed=fixed, constraints=constraints)
        assert solution.raw_residual < 1e-10
        for field, record, response in zip(
            solution.fields, system.local_metadata, system.responses, strict=True
        ):
            expected = np.zeros(len(field)) if homogeneous else exact_pressure(record["nodes"])
            assert_allclose(field, expected, atol=3e-11)
            local = response.problem
            assert_allclose(
                local.matrix @ field + local.coupling @ solution.trace[local.trace_dofs],
                local.load,
                atol=2e-12,
            )
        systems.append(system)
        solutions.append(solution)
    assert_allclose(systems[0].matrix.toarray(), systems[1].matrix.toarray(), atol=2e-12)
    assert_allclose(systems[0].rhs, systems[1].rhs, atol=2e-12)
    assert_allclose(solutions[0].trace, solutions[1].trace, atol=2e-11)
    for old, new in zip(systems[0].responses, systems[1].responses, strict=True):
        assert_array_equal(old.problem.trace_dofs, new.problem.trace_dofs)
        assert_array_equal(old.problem.coarse_basis, new.problem.coarse_basis)
        assert_allclose(old.problem.matrix.toarray(), new.problem.matrix.toarray(), atol=3e-12)
        assert_allclose(old.problem.constraints, new.problem.constraints, atol=2e-14)


def test_per_cell_tabulation_rejects_unknown_backend_without_loading_it():
    mesh = TriangleMesh.unit_square()
    bary = np.stack((np.eye(3), np.eye(3)))
    with pytest.raises(ValueError, match="backend"):
        element_tabulate(mesh, 1, bary, backend="other")
