"""One-sided polynomial and primal Darcy fields for passive transport."""

from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.spatial import cKDTree

from pymhm.cut_cells import cartesian_trace_values
from pymhm.darcy import DarcySolution
from pymhm.elements import p1_geometry, tensor_values
from pymhm.lagrange import multiindices, nodal_space, reference_basis
from pymhm.mesh import FloatArray, IntArray, TriangleMesh, positive_int
from pymhm.planar_material import PlanarMaterial
from pymhm.reservoir import CartesianCellField
from pymhm.scalar_boundary import edge_basis, edge_pieces


class _TriangleLocator:
    """Locate actual incident triangles without averaging independent field traces."""

    def __init__(self, mesh: TriangleMesh) -> None:
        """Cache affine geometry; a centroid tree supplies candidates only."""
        self.mesh = mesh
        vertices = mesh.points[mesh.cells]
        self.centers = vertices.mean(axis=1)
        self.origins = vertices[:, 0]
        self.inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        self.tree = cKDTree(self.centers)
        self.bary_gradient = p1_geometry(mesh)[0]

    def locate(self, points: Any) -> IntArray:
        """Select a containing cell with an exhaustive fallback for skew partitions."""
        if np.iscomplexobj(points):
            raise ValueError("evaluation points must be real")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("evaluation points must have finite shape (n,2)")
        count = len(self.mesh.cells)
        candidates = np.asarray(self.tree.query(points, k=min(8, count))[1]).reshape(
            len(points), min(8, count)
        )
        result = np.full(len(points), -1, dtype=np.int64)
        tolerance = 64 * np.finfo(float).eps
        for row, (point, ids) in enumerate(zip(points, candidates, strict=True)):
            bary = np.einsum("tij,tj->ti", self.inverse[ids], point - self.origins[ids])
            inside = np.all(bary >= -tolerance, axis=1) & (bary.sum(axis=1) <= 1 + tolerance)
            if np.any(inside):
                result[row] = ids[np.flatnonzero(inside)[0]]
            else:
                bary = np.einsum("tij,tj->ti", self.inverse, point - self.origins)
                inside = np.all(bary >= -tolerance, axis=1) & (bary.sum(axis=1) <= 1 + tolerance)
                if not np.any(inside):
                    raise ValueError("evaluation point lies outside the supplied macroelement")
                result[row] = np.flatnonzero(inside)[0]
        return result

    def _coordinates(
        self, points: Any, cells: Any = None
    ) -> tuple[FloatArray, IntArray, FloatArray]:
        """Validate explicitly selected sides and return their barycentric coordinates."""
        raw = np.asarray(points)
        if np.iscomplexobj(raw):
            raise ValueError("evaluation points must be real")
        points = np.asarray(raw, dtype=float)
        owners = self.locate(points) if cells is None else np.asarray(cells)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.isfinite(points).all()
            or owners.shape != (len(points),)
            or not np.issubdtype(owners.dtype, np.integer)
            or np.any(owners < 0)
            or np.any(owners >= len(self.mesh.cells))
        ):
            raise ValueError("provide finite points and valid integer incident-cell indices")
        reference = np.einsum("pab,pb->pa", self.inverse[owners], points - self.origins[owners])
        bary = np.column_stack((1 - reference.sum(axis=1), reference))
        if np.any(bary < -256 * np.finfo(float).eps):
            raise ValueError("evaluation point lies outside its declared incident triangle")
        return points, owners, bary


class PolynomialDarcyVelocity(_TriangleLocator):
    """An exact cellwise polynomial representation of a Darcy vector field.

    ``values`` contains the physical vector at equispaced Pk nodes in each fine
    triangle, in :func:`pymhm.lagrange.multiindices` order. For RT or BDM fields
    these nodal coordinates are an equivalent representation of the executed
    vector polynomial, rather than a projection or a smoothing operation.
    H(div) continuity and equilibrium belong to the supplied Darcy solution;
    this low-level constructor does not certify either property.
    """

    def __init__(self, mesh: TriangleMesh, values: Any, degree: int) -> None:
        """Validate and copy physical cell coefficients, retaining independent sides."""
        super().__init__(mesh)
        self.degree = positive_int(degree, "polynomial degree")
        raw = np.asarray(values)
        shape = (len(mesh.cells), (self.degree + 1) * (self.degree + 2) // 2, 2)
        if np.iscomplexobj(raw) or raw.shape != shape or not np.isfinite(raw).all():
            raise ValueError("polynomial velocity requires finite real cellwise Pk vector values")
        self.values = np.array(raw, dtype=float, copy=True)
        self.values.setflags(write=False)

    def evaluate(self, points: Any, cells: Any = None) -> FloatArray:
        """Evaluate each polynomial on its declared side, without nearest-node sampling."""
        _, owners, bary = self._coordinates(points, cells)
        basis = reference_basis(self.degree, bary)[0]
        return np.einsum("pi,pia->pa", basis, self.values[owners])

    def __call__(self, points: FloatArray) -> FloatArray:
        """Evaluate containing fine cells, choosing one side at an interior edge."""
        return self.evaluate(points)

    def gradient(self, points: Any, cells: Any = None) -> FloatArray:
        """Return grad(q)[component, derivative] by exact polynomial differentiation."""
        _, owners, bary = self._coordinates(points, cells)
        derivative = reference_basis(self.degree, bary)[1]
        gradient = np.einsum("pin,pnb->pib", derivative, self.bary_gradient[owners])
        return np.einsum("pia,pib->pab", self.values[owners], gradient)

    def divergence(self, points: FloatArray, cells: Any = None) -> FloatArray:
        """Return the physical polynomial divergence, without source projection."""
        return np.trace(self.gradient(points, cells), axis1=1, axis2=2)


class PrimalDarcyVelocity(_TriangleLocator):
    """The explicit raw field -K grad(p_h), with material derivatives provided.

    The field can have normal jumps inside a macrocell and is not asserted to
    be H(div)-conforming or fine-cell conservative. ``permeability_gradient``
    has axes (point, tensor row, tensor column, derivative). For a piecewise
    constant material this broken derivative is zero. Material traces use the
    specified incident triangle, including Cartesian/planar interfaces.

    Transport uses the raw vector in volume integrals and the Darcy skeletal
    multiplier as the numerical normal velocity. The conservative Galerkin
    form is integrated directly; it does not replace distributional interface
    derivatives by the broken divergence. The boundary term contains half this
    numerical normal velocity, preserving the L11 Robin multiplier convention.
    This explicitly declared discrete flux pair is distinct from an H(div)
    vector field. Strong-residual stabilization is not provided for this pair.
    """

    def __init__(self, solution: DarcySolution, cell: int, permeability_gradient: Any) -> None:
        """Retain actual primal geometry and require an explicit broken tensor derivative."""
        if solution.formulation != "primal" or permeability_gradient is None:
            raise ValueError("raw primal velocity requires a primal solution and material gradient")
        super().__init__(solution.local_meshes[cell])
        self.solution = solution
        self.cell = cell
        self.degree = solution.degree
        self.material = solution.permeability
        self.material_gradient = permeability_gradient
        dofs, _ = nodal_space(self.mesh, self.degree)
        self.pressure = solution.pressure[cell][dofs].copy()
        self.pressure.setflags(write=False)

    def _data(
        self, points: Any, cells: Any = None, material_values: Any = None
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate physical pressure derivatives and unambiguous material traces."""
        points, owners, bary = self._coordinates(points, cells)
        _, first, second = reference_basis(self.degree, bary)
        gradients = np.einsum("pin,pna->pia", first, self.bary_gradient[owners])
        hessians = np.einsum(
            "pinm,pna,pmb->piab", second, self.bary_gradient[owners], self.bary_gradient[owners]
        )
        gradient = np.einsum("pi,pia->pa", self.pressure[owners], gradients)
        hessian = np.einsum("pi,piab->pab", self.pressure[owners], hessians)
        material = self.material if material_values is None else material_values
        if material_values is None and isinstance(material, CartesianCellField):
            material = cartesian_trace_values(material, points, self.centers[owners])
        elif isinstance(material, PlanarMaterial):
            material = material.trace_values(points, self.centers[owners])
        return tensor_values(material, points), gradient, hessian

    def evaluate(
        self, points: Any, cells: Any = None, *, material_values: Any = None
    ) -> FloatArray:
        """Evaluate -K grad(p_h), optionally retaining exact material-quadrature samples."""
        tensor, gradient, _ = self._data(points, cells, material_values)
        return -np.einsum("pab,pb->pa", tensor, gradient)

    def __call__(self, points: FloatArray) -> FloatArray:
        """Evaluate one containing side of the unsmoothed raw Darcy field."""
        return self.evaluate(points)

    def gradient(self, points: Any, cells: Any = None) -> FloatArray:
        """Differentiate the raw velocity, retaining all tensor derivative components."""
        points = np.asarray(points)
        tensor, gradient, hessian = self._data(points, cells)
        raw = (
            self.material_gradient(points)
            if callable(self.material_gradient)
            else self.material_gradient
        )
        derivative = np.asarray(raw)
        if np.iscomplexobj(derivative) or derivative.shape not in (
            (),
            (2, 2, 2),
            (len(points), 2, 2, 2),
        ):
            raise ValueError("material gradient requires finite real shape (n,2,2,2) or zero")
        if not np.isfinite(derivative).all() or (derivative.ndim == 0 and derivative != 0):
            raise ValueError("material gradient requires finite real shape (n,2,2,2) or zero")
        derivative = np.broadcast_to(derivative, (len(points), 2, 2, 2))
        return -np.einsum("pabc,pb->pac", derivative, gradient) - np.einsum(
            "pab,pbc->pac", tensor, hessian
        )

    def divergence(self, points: FloatArray, cells: Any = None) -> FloatArray:
        """Return broken div(-K grad(p_h)); interface distributions are separate."""
        return np.trace(self.gradient(points, cells), axis1=1, axis2=2)

    def normal_trace(self, face: int, points: FloatArray) -> FloatArray:
        """Return the Darcy numerical normal velocity in the global macroface orientation."""
        points, _, _ = self._coordinates(points)
        skeleton = self.solution.skeleton
        if face not in skeleton.mesh.cell_faces[self.cell]:
            raise ValueError("numerical normal velocity requires a face of this macrocell")
        start, end = skeleton.mesh.points[skeleton.mesh.faces[face]]
        tangent = end - start
        parameter = (points - start) @ tangent / (tangent @ tangent)
        if (
            not np.allclose(points, start + parameter[:, None] * tangent, atol=1e-12, rtol=0)
            or np.any(parameter < -1e-12)
            or np.any(parameter > 1 + 1e-12)
        ):
            raise ValueError("numerical normal points must lie on the supplied macroface")
        return (
            skeleton.faces[face].evaluate(parameter)
            @ self.solution.hybrid.trace[skeleton.dofs(face)]
        )

    def advection_boundary_matrix(
        self, degree: int, order: int, *, faces: tuple[int, ...] | None = None
    ) -> Any:
        """Integrate half the numerical normal velocity on every macro boundary side.

        The volume form is ``-(v_raw*u, grad(test))``. This term supplies the
        corresponding L11 half-advection Robin boundary contribution. Integration
        splits at both fine-edge and Darcy-trace breakpoints, independently of the
        concentration skeleton; no pressure-gradient normal is substituted.
        """
        macro = self.solution.skeleton.mesh
        count = len(nodal_space(self.mesh, degree)[1])
        matrix = sparse.lil_matrix((count, count))
        gauss, weights = leggauss(max(order, degree + self.degree + 2))
        for side, face in enumerate(macro.cell_faces[self.cell]):
            if faces is not None and face not in faces:
                continue
            start, end = macro.points[macro.faces[face]]
            trace = self.solution.skeleton.faces[face]
            for positions, ids in edge_pieces(macro, self.mesh, int(face), degree):
                lo, hi = sorted(positions)
                cuts = np.r_[lo, [x for x in trace.breaks if lo < x < hi], hi]
                for left, right in zip(cuts[:-1], cuts[1:], strict=True):
                    parameter = left + (gauss + 1) * (right - left) / 2
                    local = (parameter - positions[0]) / (positions[1] - positions[0])
                    points = start + parameter[:, None] * (end - start)
                    normal = macro.signs[self.cell, side] * self.normal_trace(int(face), points)
                    basis = edge_basis(degree, local)
                    measure = (right - left) * macro.lengths[face] / 2
                    block = basis.T @ ((weights * normal * measure / 2)[:, None] * basis)
                    matrix[np.ix_(ids, ids)] += block
        return matrix.tocsc()


def polynomial_darcy_velocity(solution: Any, cell: int) -> PolynomialDarcyVelocity:
    """Convert an executed triangular RT/BDM field to its equivalent vector polynomial.

    Supported solutions are ``RTDarcySolution``, ``BDMDarcySolution`` and
    ``MomentFluxSolution``. Their own field evaluators preserve canonical moments,
    Piola transforms and orientation. Reconstruction conservation remains exactly
    the conservation imposed by the chosen reconstruction; no extra constraint is
    introduced by this conversion.
    """
    from pymhm.darcy_mixed import BDMDarcySolution, _evaluate
    from pymhm.darcy_rt import RTDarcySolution
    from pymhm.reconstruction_moments import MomentFluxSolution
    from pymhm.rt import rt_evaluate

    if isinstance(solution, BDMDarcySolution):
        mesh = solution.local_meshes[cell]
        degree = solution.family.polynomial_degree
        bary = multiindices(degree) / degree
        values = _evaluate(solution.family, mesh, solution.flux[cell], bary)[0]
    elif isinstance(solution, (RTDarcySolution, MomentFluxSolution)):
        mesh = solution.local_meshes[cell]
        degree = solution.degree + 1
        bary = multiindices(degree) / degree
        values = rt_evaluate(mesh, solution.flux[cell], solution.degree, bary)[0]
    else:
        raise ValueError("require a triangular RT or BDM Darcy field")
    return PolynomialDarcyVelocity(mesh, values, degree)
