"""Bitwise face ownership and bounded geometry work for prepared scalar traces."""

import numpy as np
import pytest

import pymhm.scalar_boundary as owner
from pymhm.lagrange import nodal_space
from pymhm.longest_edge import refine_longest_edge
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.scalar_boundary import edge_basis, prepare_scalar_trace, scalar_trace


def legacy(pieces, degree, coefficients, parameter):
    """Evaluate the original ordered overlap rule without an interval index."""
    values = np.empty(len(parameter))
    covered = np.zeros(len(parameter), dtype=bool)
    for positions, ids in pieces:
        lo, hi = np.sort(positions)
        selected = (parameter >= lo - 1e-13) & (parameter <= hi + 1e-13)
        local = (parameter[selected] - positions[0]) / (positions[1] - positions[0])
        values[selected] = edge_basis(degree, local) @ coefficients[ids]
        covered[selected] = True
    if not covered.all():
        raise ValueError("not covered")
    return values


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_orientations_endpoints_and_supplied_local_meshes_are_bitwise(degree):
    """Both incident fields retain endpoint ownership, reversed edges and fine partitions."""
    macro = TriangleMesh.unit_square(1)
    face = int(np.flatnonzero(macro.face_cells[:, 1] >= 0)[0])
    for cell in (0, 1):
        fine = macro.submesh(cell, 3)
        fine = refine_longest_edge(fine, np.arange(len(fine.cells)) % 3 == 0).mesh
        n = len(nodal_space(fine, degree)[1])
        coefficients = np.random.default_rng(171 + cell).normal(size=n)
        prepared = prepare_scalar_trace(macro, fine, face, degree)
        knots = np.unique(np.concatenate([p for p, _ in prepared.pieces]))
        parameters = np.r_[
            np.linspace(0, 1, 79),
            knots,
            knots - 1e-13,
            knots + 1e-13,
            np.nextafter(knots - 1e-13, -np.inf),
            np.nextafter(knots + 1e-13, np.inf),
        ]
        parameters = parameters[(parameters >= 0) & (parameters <= 1)][::-1]
        np.testing.assert_array_equal(
            prepared.evaluate(coefficients, parameters),
            legacy(prepared.pieces, degree, coefficients, parameters),
        )
        np.testing.assert_array_equal(
            scalar_trace(macro, fine, face, degree, coefficients, parameters),
            legacy(prepared.pieces, degree, coefficients, parameters),
        )
        for array in (
            prepared.starts,
            prepared.ends,
            prepared.prefix_maximum,
            prepared.order,
            *(a for p in prepared.pieces for a in p),
        ):
            assert not array.flags.writeable
        with pytest.raises(ValueError, match="macroface"):
            prepared.evaluate(coefficients, np.array([-1, 1.1]))
        assert prepared.evaluate(coefficients, np.empty(0)).shape == (0,)


def test_preparation_is_once_and_search_skips_disjoint_pieces(monkeypatch):
    """Repeated subinterval evaluations do not scan or evaluate every fine edge."""
    mesh = TriangleMesh.unit_square(1)
    face = int(np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0])
    fine = mesh.submesh(0, 16)
    original_pieces, original_basis = owner.edge_pieces, owner.edge_basis
    counts = {"geometry": 0, "basis": 0}

    def pieces(*args):
        """Count geometry reconstruction calls independently of sample count."""
        counts["geometry"] += 1
        return original_pieces(*args)

    def basis(*args):
        """Count only polynomial pieces actually evaluated."""
        counts["basis"] += 1
        return original_basis(*args)

    monkeypatch.setattr(owner, "edge_pieces", pieces)
    monkeypatch.setattr(owner, "edge_basis", basis)
    prepared = prepare_scalar_trace(mesh, fine, face, 1)
    for index in range(64):
        prepared.evaluate(np.arange(len(fine.points)), np.array([(index + 0.5) / 64]))
    assert counts == {"geometry": 1, "basis": 64}


def test_general_overlaps_and_empty_preparations_preserve_last_assignment(monkeypatch):
    """The prefix index handles nested intervals rather than assuming a valid sorted partition."""
    pieces = (
        (np.array([1.0, 0.0]), np.array([0, 1])),
        (np.array([0.25, 0.3]), np.array([2, 3])),
        (np.array([0.7, 0.5]), np.array([4, 5])),
    )
    monkeypatch.setattr(owner, "edge_pieces", lambda *args: iter(pieces))
    mesh = TriangleMesh.unit_square(1)
    prepared = prepare_scalar_trace(mesh, mesh, 0, 1)
    x = np.array([0.2, 0.26, 0.4, 0.55, 0.8])
    c = np.arange(6.0)
    np.testing.assert_array_equal(prepared.evaluate(c, x), legacy(pieces, 1, c, x))
    monkeypatch.setattr(owner, "edge_pieces", lambda *args: iter(()))
    empty = prepare_scalar_trace(mesh, mesh, 0, 1)
    assert empty.evaluate(c, np.empty(0)).shape == (0,)
    with pytest.raises(ValueError, match="macroface"):
        empty.evaluate(c, np.array([0.5]))


class LegacyPrepared:
    """Preserve the original polynomial evaluation arithmetic for estimator comparison."""

    def __init__(self, prepared):
        """Keep exactly the same oriented fine pieces."""
        self.pieces = prepared.pieces
        self.degree = prepared.degree

    def evaluate(self, coefficients, parameter):
        """Use original ordered masks including endpoint overlap."""
        return legacy(self.pieces, self.degree, coefficients, parameter)


def test_estimators_bitwise_and_one_preparation_per_incident_face(monkeypatch):
    """Both common jump indicators retain all integrated values on supplied fine meshes."""
    import pymhm.darcy_jump_estimator as darcy
    import pymhm.scalar_adaptive as transport
    from pymhm.darcy import solve_darcy
    from pymhm.transport import solve_transport

    mesh = TriangleMesh.unit_square(1)
    fine = (mesh.submesh(0, 3), mesh.submesh(1, 5))
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 2) for _ in mesh.faces))
    solutions = (
        solve_transport(
            mesh,
            skeleton=skeleton,
            local_meshes=fine,
            degree=2,
            source=1,
            dirichlet_enforcement="strong",
        ),
        solve_darcy(mesh, skeleton=skeleton, local_meshes=fine, degree=2, source=1),
    )
    for module, solution in zip((transport, darcy), solutions, strict=True):
        calls = []

        def prepare(*args, _calls=calls):
            """Count each actual geometry construction."""
            _calls.append((args[2], id(args[1])))
            return prepare_scalar_trace(*args)

        monkeypatch.setattr(module, "prepare_scalar_trace", prepare)
        actual = (
            module.estimate_transport_faces(solution, transport.TransportBounds(1, 0, 0))
            if module is transport
            else module.estimate_darcy_jumps(solution)
        )
        assert len(calls) == (2 if module is transport else 6)
        assert len(calls) == len(set(calls))
        monkeypatch.setattr(
            module,
            "prepare_scalar_trace",
            lambda *args: LegacyPrepared(prepare_scalar_trace(*args)),
        )
        expected = (
            module.estimate_transport_faces(solution, transport.TransportBounds(1, 0, 0))
            if module is transport
            else module.estimate_darcy_jumps(solution)
        )
        if module is transport:
            for a, b in zip(actual.values, expected.values, strict=True):
                np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_array_equal(actual.face_squared, expected.face_squared)


def test_pgmhm_prepared_trace_is_reused_for_flux_and_sparse_sampling(monkeypatch):
    """Field evaluation reuses owned geometry and retains the first/last ownership contracts."""
    from scipy import sparse

    from pymhm.pgmhm import _trace_matrix, solve_pgmhm

    mesh = TriangleMesh.unit_square(1)
    solution = solve_pgmhm(
        mesh,
        stabilization_parameter=0.1,
        degree=2,
        local_refinement=3,
        source=lambda p: 1 + p[:, 0],
        dirichlet=lambda p: 2 + p[:, 0] + p[:, 1],
    )
    saved = owner.edge_pieces

    def forbidden(*args):
        """Repeated flux evaluation must not rebuild prepared face geometry."""
        raise AssertionError("geometry reconstructed")

    monkeypatch.setattr(owner, "edge_pieces", forbidden)
    for data in solution.penalties:
        points = np.r_[np.linspace(0, 1, 19), data.breaks]
        first = int(mesh.face_cells[data.face, 0])
        where = int(np.flatnonzero(mesh.cell_faces[first] == data.face)[0])
        jump = sum(
            sign * legacy(trace.pieces, 2, solution.pressure[cell], points)
            for (cell, sign, _), trace in zip(data.sides, data.traces, strict=True)
        )
        expected = solution.skeleton.faces[data.face].evaluate(points) @ solution.hybrid.trace[
            solution.skeleton.dofs(data.face)
        ] - data.coefficient * (jump - data.boundary_value(points))
        np.testing.assert_array_equal(
            solution.normal_flux(first, data.face, points), mesh.signs[first, where] * expected
        )
        for (cell, _, _), trace in zip(data.sides, data.traces, strict=True):
            fine = solution.local_meshes[cell]
            count = len(nodal_space(fine, 2)[1])
            rows = []
            columns = []
            values = []
            covered = np.zeros(len(points), dtype=bool)
            for positions, ids in trace.pieces:
                lo, hi = np.sort(positions)
                selected = np.flatnonzero(
                    (points >= lo - 1e-13) & (points <= hi + 1e-13) & ~covered
                )
                basis = edge_basis(
                    2, (points[selected] - positions[0]) / (positions[1] - positions[0])
                )
                rows.extend(np.repeat(selected, len(ids)))
                columns.extend(np.tile(ids, len(selected)))
                values.extend(basis.ravel())
                covered[selected] = True
            expected_matrix = sparse.coo_matrix(
                (values, (rows, columns)), shape=(len(points), count)
            ).tocsr()
            actual = _trace_matrix(mesh, fine, data.face, 2, points, trace)
            assert (actual != expected_matrix).nnz == 0
    monkeypatch.setattr(owner, "edge_pieces", saved)
