"""Physical coefficient records and norms, independent of method solver dispatch.

These records interpret explicitly assembled coefficient vectors in their
executed finite-element bases. Construction performs no PDE solve, condensation
or solver selection. Named FieldDefinition views provide the general-purpose
field API; these records additionally supply established physical diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from pymhm.core.contracts import HybridSolution
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.core.validation import real_array as _real
from pymhm.fem.hdiv.bdm_family import BDMFamily
from pymhm.fem.hdiv.bdm_forms import bdm_evaluate, bdm_pressure_basis, bdm_trace_map
from pymhm.fem.hdiv.family_3d import HDiv3DFamily, cell_quadrature
from pymhm.fem.hdiv.mixed import triangle_pressure_basis
from pymhm.fem.hdiv.mixed_3d import Mixed3DSkeleton
from pymhm.fem.hdiv.rt import rt_evaluate
from pymhm.fem.hdiv.rt_trace import rt_trace_map
from pymhm.fem.hdiv.tensor_rt import (
    _material_rule,
    tensor_rt_basis,
    tensor_rt_dofs,
    tensor_rt_trace_map,
)
from pymhm.fem.loads import split_point_sources
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.neumann import NeumannMaps
from pymhm.fem.scalar.operators import rt0_evaluate, rt0_operators, triangle_quadrature
from pymhm.fem.scalar.quadrilateral import qk_basis, quadrilateral_quadrature
from pymhm.fem.scalar.tetrahedron import tetra_tabulate, tetrahedron_quadrature
from pymhm.fem.scalar.triangle import element_tabulate, tabulate
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.jump import FaceJumpForm
from pymhm.fem.traces.pressure_2d import PressureTraceSpace
from pymhm.fem.traces.pressure_3d import PressureTraceSpace3D, boundary_rules, broken_face_basis
from pymhm.fem.traces.scalar import scalar_trace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.fem.vector.operators import _lagrange
from pymhm.materials.compliance import compressibility_values
from pymhm.materials.evaluation import (
    scalar_values,
    scalar_values_3d,
    tensor_values,
    tensor_values_3d,
    vector_values,
    vector_values_3d,
)
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.mixed import AffineMixedMesh, hdiv3d_basis, hdiv3d_dofs
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class DarcySolution:
    """Broken pressure and conservative skeleton flux of a Darcy solve.

    Primal pressure coefficients are continuous Pk nodal values on each macrocell.
    Mixed pressure coefficients
    are P0 cell values and the flux is RT0 conforming, with integrated face DOFs.
    For the primal formulation, ``flux`` stores samples of ``-K grad(p)`` at
    fine-cell centroids; these are not piecewise-constant coefficients when
    the degree exceeds one or K varies. Error norms evaluate the full field
    at quadrature points. Raw primal flux is generally not H(div)-conforming.
    """

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    formulation: str
    permeability: Any
    source: Any
    quadrature_order: int = 4
    degree: int = 1
    point_sources: tuple[FloatArray, ...] = ()

    def l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate pressure error with independent higher-order quadrature."""
        total = 0.0
        for mesh, pressure in zip(self.local_meshes, self.pressure, strict=True):
            bary, weights, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            if self.formulation == "primal":
                dofs, _, basis, _, _ = element_tabulate(mesh, self.degree, bary)
                values = np.einsum("ti,tqi->tq", pressure[dofs], basis)
            else:
                values = np.broadcast_to(pressure[:, None], weights.shape)
            difference = values - scalar_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ np.sum(difference**2 * weights, axis=1))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate physical flux error; primal flux is evaluated as -K grad p."""
        total = 0.0
        for mesh, pressure, flux in zip(self.local_meshes, self.pressure, self.flux, strict=True):
            bary, weights, material = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            if self.formulation == "mixed":
                values = rt0_evaluate(mesh, flux, bary)
            else:
                dofs, _, _, gradients, _ = element_tabulate(mesh, self.degree, bary)
                grad = np.einsum("ti,tqia->tqa", pressure[dofs], gradients)
                tensors = tensor_values(material, points.reshape(-1, 2)).reshape(
                    *weights.shape, 2, 2
                )
                values = -np.einsum("tqab,tqb->tqa", tensors, grad)
            difference = values - vector_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ np.sum(np.sum(difference**2, axis=2) * weights, axis=1))
        return float(np.sqrt(total))

    def conservation_residuals(self, order: int | None = None) -> FloatArray:
        """Return macro flux minus source, using assembly quadrature by default.

        Supply a higher order to measure source integration error separately.
        """
        residuals = []
        for cell, mesh in enumerate(self.local_meshes):
            bary, weights, _ = material_triangle_quadrature(
                mesh, self.permeability, self.quadrature_order if order is None else order
            )
            flux = 0.0
            for side, face in enumerate(self.skeleton.mesh.cell_faces[cell]):
                space = self.skeleton.faces[face]
                parameter, w = space.quadrature(max(space.degrees) + 2)
                moments = w @ space.evaluate(parameter)
                flux += (
                    self.skeleton.mesh.signs[cell, side]
                    * self.skeleton.mesh.lengths[face]
                    * float(moments @ self.hybrid.trace[self.skeleton.dofs(int(face))])
                )
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            integral = mesh.areas @ np.sum(
                scalar_values(self.source, points.reshape(-1, 2)).reshape(weights.shape) * weights,
                axis=1,
            )
            if self.point_sources:
                integral += self.point_sources[cell][:, 2].sum()
            residuals.append(flux - integral)
        return np.asarray(residuals)

    def fine_conservation_residuals(self) -> tuple[FloatArray, ...]:
        """Return RT0 cellwise mass defects; reject raw primal flux diagnostics."""
        if self.formulation != "mixed":
            raise ValueError("fine-cell conservation requires the mixed RT0 formulation")
        residuals = []
        for cell, (mesh, flux) in enumerate(zip(self.local_meshes, self.flux, strict=True)):
            force = rt0_operators(mesh, self.permeability, self.source, self.quadrature_order)[2]
            if self.point_sources:
                force += np.array(
                    [
                        part[:, 2].sum()
                        for part in split_point_sources(mesh, self.point_sources[cell])
                    ]
                )
            residuals.append(np.sum(flux[mesh.cell_faces] * mesh.signs, axis=1) - force)
        return tuple(residuals)


@dataclass(frozen=True)
class Darcy3DSolution:
    """Broken tetrahedral pressure and physical gradient flux with conservative trace.

    ``flux`` evaluates ``-K grad(p)`` and is generally not H(div)-conforming.
    Macro conservation refers to ``hybrid.trace``, not this raw gradient field.
    """

    skeleton: TriangularSkeleton
    local_meshes: tuple[TetraMesh, ...]
    pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    permeability: Any
    source: Any
    quadrature_order: int

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate pressure and physical flux at common fine-cell barycentric points."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.local_meshes):
            raise ValueError("cell outside local meshes")
        fine = self.local_meshes[cell]
        dofs, _, basis, gradient = tetra_tabulate(fine, self.degree, bary)
        pressure = self.pressure[cell][dofs] @ basis.T
        grad = np.einsum("ti,tqij->tqj", self.pressure[cell][dofs], gradient)
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        diffusion = tensor_values_3d(self.permeability, points.reshape(-1, 3)).reshape(
            *points.shape[:2], 3, 3
        )
        return pressure, -np.einsum("tqij,tqj->tqi", diffusion, grad)

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the broken pressure error with independently selected positive quadrature."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            values = self.evaluate(cell, bary)[0]
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            expected = scalar_values_3d(exact, points.reshape(-1, 3)).reshape(values.shape)
            total += float(fine.volumes @ ((values - expected) ** 2 @ weights))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the full physical flux, including P2 gradient and variable diffusion."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            values = self.evaluate(cell, bary)[1]
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            expected = _real(
                exact(points.reshape(-1, 3)) if callable(exact) else exact, "exact flux"
            )
            try:
                expected = np.broadcast_to(expected, (len(points.reshape(-1, 3)), 3)).reshape(
                    values.shape
                )
            except ValueError as exc:
                raise ValueError("exact flux must return three components per point") from exc
            total += float(fine.volumes @ (np.sum((values - expected) ** 2, axis=2) @ weights))
        return float(np.sqrt(total))

    def conservation_residuals(self, order: int | None = None) -> FloatArray:
        """Return outward skeletal flux minus integrated source in every macrocell."""
        mesh = self.skeleton.mesh
        bary, weights = tetrahedron_quadrature(self.quadrature_order if order is None else order)
        residuals = []
        for cell, fine in enumerate(self.local_meshes):
            total = sum(
                mesh.signs[cell, side]
                * mesh.areas[face]
                * (
                    self.skeleton.face_weights(int(face))
                    @ np.array(
                        [
                            self.hybrid.trace[
                                self.skeleton.subtriangle_dofs(int(face), segment)
                            ].mean()
                            for segment in range(len(self.skeleton.face_partition(int(face))))
                        ]
                    )
                )
                for side, face in enumerate(mesh.cell_faces[cell])
            )
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            load = scalar_values_3d(self.source, points.reshape(-1, 3)).reshape(len(fine.cells), -1)
            residuals.append(total - float(fine.volumes @ (load @ weights)))
        return np.asarray(residuals)


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


@dataclass(frozen=True)
class RTDarcySolution:
    """Locally H(div)-conforming RTm flux and discontinuous nodal Pm pressure.

    Mathematical degree m>=0 corresponds to Basix RT element degree m+1.
    MHM traces prescribe physical normal flux, while classical solutions use a
    globally conforming fine mesh. The latter have no skeletal multiplier.
    """

    local_meshes: tuple[TriangleMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    degree: int
    permeability: Any
    source: Any
    quadrature_order: int
    skeleton: SkeletonSpace | None = None
    hybrid: HybridSolution | None = None
    residual: float = 0.0

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the discontinuous pressure error without centroidal substitution."""
        total = 0.0
        for mesh, p in zip(self.local_meshes, self.pressure, strict=True):
            bary, w, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            value = np.einsum("ti,tqi->tq", p, triangle_pressure_basis(self.degree, bary))
            error = value - scalar_values(exact, points.reshape(-1, 2)).reshape(w.shape)
            total += float(mesh.areas @ np.sum(w * error**2, axis=1))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the complete physical H(div) vector error."""
        total = 0.0
        for mesh, q in zip(self.local_meshes, self.flux, strict=True):
            bary, w, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            value = rt_evaluate(mesh, q, self.degree, bary)[0]
            error = value - vector_values(exact, points.reshape(-1, 2)).reshape(value.shape)
            total += float(mesh.areas @ np.sum(w * np.sum(error**2, axis=2), axis=1))
        return float(np.sqrt(total))

    def divergence_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the full divergence error, distinguishing projection from pointwise balance."""
        total = 0.0
        for mesh, q in zip(self.local_meshes, self.flux, strict=True):
            bary, weights, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            divergence = rt_evaluate(mesh, q, self.degree, bary)[1]
            exact_values = scalar_values(exact, points.reshape(-1, 2)).reshape(weights.shape)
            total += float(mesh.areas @ np.sum(weights * (divergence - exact_values) ** 2, axis=1))
        return float(np.sqrt(total))

    def fine_equilibrium_residuals(self) -> tuple[FloatArray, ...]:
        """Return every DG-Pm moment of div(q)-f on each fine triangle."""
        result = []
        for mesh, q in zip(self.local_meshes, self.flux, strict=True):
            bary, w, _ = material_triangle_quadrature(
                mesh, self.permeability, self.quadrature_order
            )
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            divergence = rt_evaluate(mesh, q, self.degree, bary)[1]
            defect = divergence - scalar_values(self.source, points.reshape(-1, 2)).reshape(w.shape)
            result.append(
                np.einsum(
                    "tq,tqi,tq,t->ti",
                    w,
                    triangle_pressure_basis(self.degree, bary),
                    defect,
                    mesh.areas,
                )
            )
        return tuple(result)

    def fine_conservation_residuals(self) -> tuple[FloatArray, ...]:
        """Return integrated fine-cell flux/source imbalance via the partition of unity."""
        return tuple(moment.sum(axis=1) for moment in self.fine_equilibrium_residuals())

    def conservation_residuals(self) -> FloatArray:
        """Sum fine-cell physical balances over each local mesh/macrocell."""
        return np.array([defect.sum() for defect in self.fine_conservation_residuals()])

    def normal_flux_residuals(self) -> tuple[FloatArray, ...]:
        """Compare all local exterior RT normal moments with the physical MHM trace."""
        if self.skeleton is None or self.hybrid is None:
            return ()
        defects = []
        count = self.degree + 1
        for cell, (mesh, q) in enumerate(zip(self.local_meshes, self.flux, strict=True)):
            indices = (count * mesh.boundary_faces[:, None] + np.arange(count)).ravel()
            mapping = rt_trace_map(self.skeleton.mesh, cell, mesh, self.skeleton, self.degree)
            defects.append(q[indices] - mapping @ self.hybrid.trace[self.skeleton.cell_dofs(cell)])
        return tuple(defects)


@dataclass(frozen=True)
class BDMDarcySolution:
    """Conforming BDM(k,n) flux and complete discontinuous pressure from a MHM solve.

    Flux stores integrated normal moments and cell moments as specified in
    ``pymhm.fem.hdiv.bdm_family``. The default BDM2/P1 uses the equivalent coordinates
    in ``pymhm.fem.hdiv.bdm``. Pressure has degree k+n-1 and cardinal coefficients.
    The skeletal multiplier is the physical flux q.n in the macro orientation.
    """

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    permeability: Any
    source: Any
    quadrature_order: int
    family: BDMFamily = BDMFamily()

    def l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate broken pressure error against an analytical pressure."""
        total = 0.0
        for mesh, pressure in zip(self.local_meshes, self.pressure, strict=True):
            bary, weights, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            values = np.einsum("ti,tqi->tq", pressure, bdm_pressure_basis(self.family, bary))
            error = values - scalar_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ np.sum(error**2 * weights, axis=1))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate physical BDM flux error without gradient recovery or smoothing."""
        total = 0.0
        for mesh, flux in zip(self.local_meshes, self.flux, strict=True):
            bary, weights, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            values, _ = bdm_evaluate(self.family, mesh, flux, bary)
            error = values - vector_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ np.sum(np.sum(error**2, axis=2) * weights, axis=1))
        return float(np.sqrt(total))

    def divergence_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate divergence error; exact=source measures strong equilibrium error."""
        total = 0.0
        for mesh, flux in zip(self.local_meshes, self.flux, strict=True):
            bary, weights, _ = material_triangle_quadrature(mesh, self.permeability, order)
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            _, values = bdm_evaluate(self.family, mesh, flux, bary)
            error = values - scalar_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ np.sum(error**2 * weights, axis=1))
        return float(np.sqrt(total))

    def fine_equilibrium_residuals(self) -> tuple[FloatArray, ...]:
        """Return all complete pressure moments of div(q)-f in every fine triangle."""
        residuals = []
        for mesh, flux in zip(self.local_meshes, self.flux, strict=True):
            bary, weights, _ = material_triangle_quadrature(
                mesh, self.permeability, self.quadrature_order
            )
            points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
            _, divergence = bdm_evaluate(self.family, mesh, flux, bary)
            source = scalar_values(self.source, points.reshape(-1, 2)).reshape(divergence.shape)
            residuals.append(
                np.einsum(
                    "tq,tqi,tq,t->ti",
                    weights,
                    bdm_pressure_basis(self.family, bary),
                    divergence - source,
                    mesh.areas,
                )
            )
        return tuple(residuals)

    def fine_conservation_residuals(self) -> tuple[FloatArray, ...]:
        """Return integrated div(q)-f over each fine cell, using assembly quadrature."""
        return tuple(residual.sum(axis=1) for residual in self.fine_equilibrium_residuals())

    def conservation_residuals(self) -> FloatArray:
        """Return macro flux minus source integrals, with the signed skeleton flux."""
        defects = []
        coarse = self.skeleton.mesh
        for cell, fine in enumerate(self.local_meshes):
            bary, weights, _ = material_triangle_quadrature(
                fine, self.permeability, self.quadrature_order
            )
            points = np.einsum("tqi,tij->tqj", bary, fine.points[fine.cells])
            source = scalar_values(self.source, points.reshape(-1, 2)).reshape(points.shape[:2])
            defect = -float(fine.areas @ np.sum(source * weights, axis=1))
            for side, face in enumerate(coarse.cell_faces[cell]):
                parameter, w = self.skeleton.faces[face].quadrature(max(4, self.family.degree + 1))
                values = self.skeleton.faces[face].evaluate(parameter)
                defect += (
                    coarse.signs[cell, side]
                    * coarse.lengths[face]
                    * float(w @ values @ self.hybrid.trace[self.skeleton.dofs(int(face))])
                )
            defects.append(defect)
        return np.asarray(defects)

    def normal_flux_residuals(self) -> tuple[FloatArray, ...]:
        """Return all boundary normal-moment differences between BDM flux and skeleton."""
        defects = []
        for cell, (fine, flux) in enumerate(zip(self.local_meshes, self.flux, strict=True)):
            count = self.family.degree + 1
            dofs = (count * fine.boundary_faces[:, None] + np.arange(count)).ravel()
            mapping = bdm_trace_map(self.family, self.skeleton.mesh, cell, fine, self.skeleton)
            defects.append(flux[dofs] - mapping @ self.hybrid.trace[self.skeleton.cell_dofs(cell)])
        return tuple(defects)


@dataclass(frozen=True)
class TensorRTDarcySolution:
    """Enriched rectangular RT flux, modal Q_s pressure and physical skeleton flux."""

    skeleton: SkeletonSpace
    local_meshes: tuple[CartesianMacroMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    enrichment: int
    permeability: Any
    source: Any
    quadrature_order: int

    def evaluate(self, cell: int, points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate pressure, physical flux and divergence on every local rectangle."""
        fine = self.local_meshes[cell]
        basis, div, pressure = tensor_rt_basis(fine, self.degree, self.enrichment, points)
        coefficients = self.flux[cell][tensor_rt_dofs(fine, self.degree, self.enrichment)]
        return (
            np.einsum(
                "ti,tqi->tq",
                self.pressure[cell],
                np.broadcast_to(pressure, (*basis.shape[:2], pressure.shape[-1])),
            ),
            np.einsum("tqia,ti->tqa", basis, coefficients),
            np.einsum("tqi,ti->tq", div, coefficients),
        )

    def errors(self, pressure: Any, flux: Any, divergence: Any, order: int = 8) -> dict[str, float]:
        """Integrate three independent physical L2 errors without smoothing interfaces."""
        errors = np.zeros(3)
        for cell, fine in enumerate(self.local_meshes):
            points, weights = _material_rule(fine, self.permeability, order)
            physical = fine.points[fine.cells[:, 0], None] + points * fine.spacing
            flat = physical.reshape(-1, 2)
            p, q, d = self.evaluate(cell, points)
            errors[0] += fine.areas @ (
                np.sum((p - scalar_values(pressure, flat).reshape(p.shape)) ** 2 * weights, axis=1)
            )
            errors[1] += fine.areas @ (
                np.sum(
                    np.sum((q - vector_values(flux, flat).reshape(q.shape)) ** 2, axis=-1)
                    * weights,
                    axis=1,
                )
            )
            errors[2] += fine.areas @ (
                np.sum(
                    (d - scalar_values(divergence, flat).reshape(d.shape)) ** 2 * weights, axis=1
                )
            )
        return dict(zip(("pressure_l2", "flux_l2", "divergence_l2"), np.sqrt(errors), strict=True))

    def equilibrium_residuals(self) -> tuple[FloatArray, ...]:
        """Return all Q_s moments of div(q)-f in each local fine cell."""
        residuals = []
        for cell, fine in enumerate(self.local_meshes):
            points, weights = _material_rule(fine, self.permeability, self.quadrature_order)
            physical = fine.points[fine.cells[:, 0], None] + points * fine.spacing
            _, _, pressure = tensor_rt_basis(fine, self.degree, self.enrichment, points)
            d = self.evaluate(cell, points)[2]
            force = scalar_values(self.source, physical.reshape(-1, 2)).reshape(d.shape)
            residuals.append(np.einsum("t,tq,tqi,tq->ti", fine.areas, weights, pressure, d - force))
        return tuple(residuals)

    def normal_flux_residuals(self) -> tuple[FloatArray, ...]:
        """Return all fine-boundary moments of q.n minus the represented macro trace."""
        residuals = []
        for cell, fine in enumerate(self.local_meshes):
            ids = (
                (self.degree + 1) * fine.boundary_faces[:, None] + np.arange(self.degree + 1)
            ).ravel()
            mapping = tensor_rt_trace_map(
                cast(CartesianMacroMesh, self.skeleton.mesh), cell, fine, self.skeleton, self.degree
            )
            residuals.append(
                self.flux[cell][ids] - mapping @ self.hybrid.trace[self.skeleton.cell_dofs(cell)]
            )
        return tuple(residuals)


@dataclass(frozen=True)
class Mixed3DDarcySolution:
    """Affine H(div) flux and discontinuous pressure with their MHM skeleton."""

    skeleton: Mixed3DSkeleton
    family: HDiv3DFamily
    local_meshes: tuple[AffineMixedMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    source: Any
    physical_residuals: FloatArray

    def evaluate(self, cell: int, points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate pressure, physical flux and divergence in every local cell."""
        mesh = self.local_meshes[cell]
        basis, div, p = hdiv3d_basis(mesh, self.family, points)
        local = self.flux[cell][hdiv3d_dofs(mesh, self.family)]
        return (
            self.pressure[cell] @ p.T,
            np.einsum("tqia,ti->tqa", basis, local),
            np.einsum("tqi,ti->tq", div, local),
        )

    def errors(self, pressure: Any, flux: Any, order: int = 6) -> dict[str, float]:
        """Integrate physical pressure/vector-flux L2 errors with positive quadrature."""
        points, w = cell_quadrature(self.family.kind, order)
        errors = np.zeros(2)
        for cell, mesh in enumerate(self.local_meshes):
            physical = mesh.geometry(points)
            p, q, _ = self.evaluate(cell, points)
            pe = scalar_values_3d(pressure, physical.reshape(-1, 3)).reshape(p.shape)
            qe = vector_values_3d(flux, physical.reshape(-1, 3)).reshape(q.shape)
            errors += [
                np.sum(mesh.determinants[:, None] * w * (p - pe) ** 2),
                np.sum(mesh.determinants[:, None] * w * np.sum((q - qe) ** 2, axis=2)),
            ]
        return dict(pressure_l2=float(np.sqrt(errors[0])), flux_l2=float(np.sqrt(errors[1])))

    def equilibrium_residuals(self, order: int = 6) -> tuple[FloatArray, ...]:
        """Return every pressure-tested fine-cell divergence/source defect."""
        points, w = cell_quadrature(self.family.kind, order)
        p = self.family.tabulate(points)[2]
        result = []
        for cell, mesh in enumerate(self.local_meshes):
            div = self.evaluate(cell, points)[2]
            f = scalar_values_3d(self.source, mesh.geometry(points).reshape(-1, 3)).reshape(
                div.shape
            )
            result.append(np.einsum("t,q,qi,tq->ti", mesh.determinants, w, p, div - f))
        return tuple(result)


@dataclass(frozen=True)
class VectorSolution:
    """Reconstructed velocity/displacement and optional P1 pressure fields."""

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    values: tuple[FloatArray, ...]
    pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    pressure_degree: int = 1

    def l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate the error in velocity or displacement."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, values in zip(self.local_meshes, self.values, strict=True):
            dofs, _, basis, _ = _lagrange(mesh, self.degree, bary)
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            difference = np.einsum("qi,tia->tqa", basis, values[dofs]) - vector_values(
                exact, points.reshape(-1, 2)
            ).reshape(len(mesh.cells), len(weights), 2)
            total += float(mesh.areas @ (np.sum(difference**2, axis=2) @ weights))
        return float(np.sqrt(total))

    def pressure_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate pressure error with the same global gauge as the exact field."""
        if not self.pressure:
            raise ValueError("elasticity solution has no pressure field")
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, pressure in zip(self.local_meshes, self.pressure, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            dofs, _, basis, _ = _lagrange(mesh, self.pressure_degree, bary)
            difference = pressure[dofs] @ basis.T - scalar_values(
                exact, points.reshape(-1, 2)
            ).reshape(len(mesh.cells), len(weights))
            total += float(mesh.areas @ (difference**2 @ weights))
        return float(np.sqrt(total))

    def divergence_l2(self) -> float:
        """Measure the pointwise divergence norm of the reconstructed vector field."""
        bary, weights = triangle_quadrature(4)
        total = 0.0
        for mesh, values in zip(self.local_meshes, self.values, strict=True):
            dofs, _, _, gradients = _lagrange(mesh, self.degree, bary)
            divergence = np.einsum("tqia,tia->tq", gradients, values[dofs])
            total += float(mesh.areas @ (divergence**2 @ weights))
        return float(np.sqrt(total))


@dataclass(frozen=True)
class ScalarSolution:
    """Broken Pk scalar field reconstructed from a hybrid system."""

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    values: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int = 1
    strong_dirichlet: bool = False
    natural_faces: tuple[int, ...] = ()

    def l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate scalar error using quadrature independent of assembly."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, values in zip(self.local_meshes, self.values, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            dofs, _, basis, _, _ = tabulate(mesh, self.degree, bary)
            difference = values[dofs] @ basis.T - scalar_values(
                exact, points.reshape(-1, 2)
            ).reshape(len(mesh.cells), len(weights))
            total += float(mesh.areas @ (difference**2 @ weights))
        return float(np.sqrt(total))


@dataclass(frozen=True)
class RAD3DSolution:
    """Broken tetrahedral scalar field and full conservative physical flux.

    ``evaluate`` returns ``u`` and ``-K grad(u)+beta*u``. This physical flux
    differs from the skeletal Robin multiplier by ``beta*u/2``.
    """

    local_meshes: tuple[TetraMesh, ...]
    values: tuple[FloatArray, ...]
    degree: int
    diffusion: Any
    velocity: Any
    hybrid: HybridSolution | None = None
    skeleton: TriangularSkeleton | None = None
    residual: float = 0.0

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate scalar and physical flux without averaging across macro interfaces."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.local_meshes):
            raise ValueError("cell outside local meshes")
        fine = self.local_meshes[cell]
        dofs, _, basis, gradient = tetra_tabulate(fine, self.degree, bary)
        value = self.values[cell][dofs] @ basis.T
        derivative = np.einsum("ti,tqia->tqa", self.values[cell][dofs], gradient)
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        tensor = tensor_values_3d(self.diffusion, points.reshape(-1, 3)).reshape(
            *points.shape[:2], 3, 3
        )
        beta = vector_values_3d(self.velocity, points.reshape(-1, 3)).reshape(points.shape)
        return value, -np.einsum("tqab,tqb->tqa", tensor, derivative) + beta * value[:, :, None]

    def l2_error(self, exact: Any, order: int = 7) -> float:
        """Integrate the scalar error with a positive tetrahedral rule."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            value = self.evaluate(cell, bary)[0]
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            expected = scalar_values_3d(exact, points.reshape(-1, 3)).reshape(value.shape)
            total += float(fine.volumes @ ((value - expected) ** 2 @ weights))
        return float(np.sqrt(total))

    def h1_seminorm_error(self, exact_gradient: Any, order: int = 7) -> float:
        """Integrate the complete broken physical gradient error."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            dofs, _, _, gradient = tetra_tabulate(fine, self.degree, bary)
            values = np.einsum("ti,tqia->tqa", self.values[cell][dofs], gradient)
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            expected = vector_values_3d(exact_gradient, points.reshape(-1, 3)).reshape(values.shape)
            total += float(fine.volumes @ (np.sum((values - expected) ** 2, axis=-1) @ weights))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact_flux: Any, order: int = 7) -> float:
        """Integrate the complete conservative physical flux error."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            values = self.evaluate(cell, bary)[1]
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            expected = vector_values_3d(exact_flux, points.reshape(-1, 3)).reshape(values.shape)
            total += float(fine.volumes @ (np.sum((values - expected) ** 2, axis=-1) @ weights))
        return float(np.sqrt(total))


@dataclass(frozen=True)
class Flow3DSolution:
    """Broken velocity, pressure and grad-grad pseudostress on tetrahedral local meshes.

    The scalar ``skeleton`` supplies face geometry and scalar modes. Its hybrid
    trace interleaves three canonical negative-traction coefficients per mode.
    Velocity and pressure are continuous only inside each macrocell. The raw
    pseudostress is not claimed to be globally H(div)-conforming.
    """

    skeleton: TriangularSkeleton
    local_meshes: tuple[TetraMesh, ...]
    velocity: tuple[FloatArray, ...]
    pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    pressure_degree: int
    viscosity: float
    advection: Any
    formulation: str

    def _index(self, cell: int) -> int:
        """Validate a macrocell index before array access."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.local_meshes):
            raise ValueError("cell outside local meshes")
        return cell

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate one-sided velocity and pressure on all fine tetrahedra of a macrocell."""
        cell = self._index(cell)
        fine = self.local_meshes[cell]
        udofs, _, ubasis, _ = tetra_tabulate(fine, self.degree, bary)
        pdofs, _, pbasis, _ = tetra_tabulate(fine, self.pressure_degree, bary)
        return np.einsum("qi,tia->tqa", ubasis, self.velocity[cell][udofs]), self.pressure[cell][
            pdofs
        ] @ pbasis.T

    def gradient(self, cell: int, bary: FloatArray) -> FloatArray:
        """Evaluate grad(u)[component, derivative] without averaging macro interfaces."""
        cell = self._index(cell)
        dofs, _, _, gradient = tetra_tabulate(self.local_meshes[cell], self.degree, bary)
        return np.einsum("tia,tqib->tqab", self.velocity[cell][dofs], gradient)

    def pseudostress(self, cell: int, bary: FloatArray) -> FloatArray:
        """Return -nu*grad(u)+p*I+u tensor beta/2, whose normal is the MHM multiplier."""
        velocity, pressure = self.evaluate(cell, bary)
        fine = self.local_meshes[cell]
        points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
        beta = vector_values_3d(self.advection, points.reshape(-1, 3)).reshape(velocity.shape)
        return (
            -self.viscosity * self.gradient(cell, bary)
            + pressure[..., None, None] * np.eye(3)
            + np.einsum("tqa,tqb->tqab", velocity, beta) / 2
        )

    def _error(self, exact: Any, kind: str, order: int) -> float:
        """Integrate one selected physical squared error using positive volume quadrature."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(-1, 3)
            if kind == "gradient":
                values = self.gradient(cell, bary)
                expected = _real(exact(points) if callable(exact) else exact, "exact gradient")
                expected = np.broadcast_to(expected, (len(points), 3, 3)).reshape(values.shape)
                difference = np.sum((values - expected) ** 2, axis=(-1, -2))
            elif kind == "velocity":
                values = self.evaluate(cell, bary)[0]
                expected = vector_values_3d(exact, points).reshape(values.shape)
                difference = np.sum((values - expected) ** 2, axis=-1)
            else:
                values = self.evaluate(cell, bary)[1]
                expected = scalar_values_3d(exact, points).reshape(values.shape)
                difference = (values - expected) ** 2
            total += float(fine.volumes @ (difference @ weights))
        return float(np.sqrt(total))

    def l2_error(self, exact: Any, order: int = 7) -> float:
        """Return the complete broken velocity L2 error."""
        return self._error(exact, "velocity", order)

    def pressure_l2_error(self, exact: Any, order: int = 7) -> float:
        """Return pressure L2 error with the explicitly imposed physical gauge."""
        return self._error(exact, "pressure", order)

    def h1_seminorm_error(self, exact_gradient: Any, order: int = 7) -> float:
        """Return the complete broken velocity-gradient L2 error."""
        return self._error(exact_gradient, "gradient", order)

    def divergence_l2(self, order: int = 7) -> float:
        """Integrate the raw pointwise velocity divergence; no H(div) recovery is implied."""
        bary, weights = tetrahedron_quadrature(order)
        return float(
            np.sqrt(
                sum(
                    float(
                        fine.volumes
                        @ (np.trace(self.gradient(cell, bary), axis1=-2, axis2=-1) ** 2 @ weights)
                    )
                    for cell, fine in enumerate(self.local_meshes)
                )
            )
        )


@dataclass(frozen=True)
class ElasticitySolution(VectorSolution):
    """Displacement, Herrmann pressure and physical Cauchy-stress reconstruction."""

    lame_lambda: Any = 1.0
    lame_mu: Any = 1.0
    formulation: str = "gals"
    stabilization: tuple[float, ...] = ()

    def gradient(self, cell: int, bary: FloatArray) -> FloatArray:
        """Evaluate the broken displacement gradient, indexed fine cell/point/i/j."""
        dofs, _, _, derivative, _ = tabulate(self.local_meshes[cell], self.degree, bary)
        return np.einsum("tqij,tia->tqaj", derivative, self.values[cell][dofs])

    def stress(self, cell: int, bary: FloatArray) -> FloatArray:
        """Evaluate symmetric Cauchy stress using pressure without lambda cancellation."""
        derivative = self.gradient(cell, bary)
        dofs, _, basis, _, _ = tabulate(self.local_meshes[cell], self.pressure_degree, bary)
        pressure = self.pressure[cell][dofs] @ basis.T
        mesh = self.local_meshes[cell]
        points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        mu = scalar_values(self.lame_mu, points.reshape(-1, 2)).reshape(pressure.shape)
        return mu[..., None, None] * (derivative + derivative.swapaxes(-1, -2)) - (
            pressure[..., None, None] * np.eye(2)
        )

    def h1_seminorm_error(self, exact_gradient: Any, order: int = 6) -> float:
        """Integrate the full broken displacement-gradient error, not the strain norm."""
        return self._tensor_error(exact_gradient, order, stress=False)

    def stress_l2_error(self, exact_stress: Any, order: int = 6) -> float:
        """Integrate the Frobenius error of the full symmetric Cauchy stress."""
        return self._tensor_error(exact_stress, order, stress=True)

    def _tensor_error(self, exact: Any, order: int, *, stress: bool) -> float:
        """Share physical tensor quadrature between gradient and stress diagnostics."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for cell, mesh in enumerate(self.local_meshes):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            data = exact(points.reshape(-1, 2)) if callable(exact) else exact
            target = np.broadcast_to(np.asarray(data), (points.shape[0] * len(bary), 2, 2))
            if not np.isfinite(target).all() or np.iscomplexobj(target):
                raise ValueError("exact tensor must be finite and real")
            value = self.stress(cell, bary) if stress else self.gradient(cell, bary)
            error = value - target.reshape(value.shape)
            total += float(mesh.areas @ (np.sum(error**2, axis=(-1, -2)) @ weights))
        return float(np.sqrt(total))

    def compressibility_l2(self, order: int = 6) -> float:
        """Measure ``div(u)+p/lambda`` without asserting pointwise satisfaction."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for cell, mesh in enumerate(self.local_meshes):
            divergence = np.trace(self.gradient(cell, bary), axis1=-2, axis2=-1)
            dofs, _, basis, _, _ = tabulate(mesh, self.pressure_degree, bary)
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            compliance = compressibility_values(self.lame_lambda, points.reshape(-1, 2)).reshape(
                divergence.shape
            )
            residual = divergence + (self.pressure[cell][dofs] @ basis.T) * compliance
            total += float(mesh.areas @ (residual**2 @ weights))
        return float(np.sqrt(total))


@dataclass(frozen=True)
class GaLS3DSolution:
    """Displacement, Herrmann pressure and raw symmetric Cauchy stress in three dimensions."""

    skeleton: TriangularSkeleton
    local_meshes: tuple[TetraMesh, ...]
    displacement: tuple[FloatArray, ...]
    pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    pressure_degree: int
    lame_lambda: Any
    lame_mu: Any
    formulation: str
    stabilization: tuple[float, ...]

    def _index(self, cell: int) -> int:
        """Check a macrocell index before local array access."""
        index = positive_int(cell, "cell", 0)
        if index >= len(self.local_meshes):
            raise ValueError("cell outside local meshes")
        return index

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate one-sided displacement and pressure on the local tetrahedra."""
        cell = self._index(cell)
        fine = self.local_meshes[cell]
        dofs, _, basis, _ = tetra_tabulate(fine, self.degree, bary)
        pdofs, _, pbasis, _ = tetra_tabulate(fine, self.pressure_degree, bary)
        return np.einsum("qi,tia->tqa", basis, self.displacement[cell][dofs]), self.pressure[cell][
            pdofs
        ] @ pbasis.T

    def gradient(self, cell: int, bary: FloatArray) -> FloatArray:
        """Return the broken displacement gradient indexed component then derivative."""
        cell = self._index(cell)
        dofs, _, _, derivative = tetra_tabulate(self.local_meshes[cell], self.degree, bary)
        return np.einsum("tia,tqib->tqab", self.displacement[cell][dofs], derivative)

    def stress(self, cell: int, bary: FloatArray) -> FloatArray:
        """Return 2*mu*epsilon(u)-p*I without multiplying lambda by a small divergence."""
        gradient = self.gradient(cell, bary)
        pressure = self.evaluate(cell, bary)[1]
        fine = self.local_meshes[cell]
        points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
        mu = scalar_values_3d(self.lame_mu, points.reshape(-1, 3)).reshape(pressure.shape)
        return mu[..., None, None] * (gradient + gradient.swapaxes(-1, -2)) - pressure[
            ..., None, None
        ] * np.eye(3)

    def _error(self, exact: Any, kind: str, order: int) -> float:
        """Integrate a physical scalar, vector or tensor error without interface averaging."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(-1, 3)
            if kind == "displacement":
                value = self.evaluate(cell, bary)[0]
                target = vector_values_3d(exact, points).reshape(value.shape)
                error = np.sum((value - target) ** 2, axis=-1)
            elif kind == "pressure":
                value = self.evaluate(cell, bary)[1]
                target = scalar_values_3d(exact, points).reshape(value.shape)
                error = (value - target) ** 2
            else:
                value = self.stress(cell, bary) if kind == "stress" else self.gradient(cell, bary)
                target = _real(exact(points) if callable(exact) else exact, "exact tensor")
                target = np.broadcast_to(target, (len(points), 3, 3)).reshape(value.shape)
                error = np.sum((value - target) ** 2, axis=(-1, -2))
            total += float(fine.volumes @ (error @ weights))
        return float(np.sqrt(total))

    def l2_error(self, exact: Any, order: int = 7) -> float:
        """Integrate the displacement L2 error."""
        return self._error(exact, "displacement", order)

    def pressure_l2_error(self, exact: Any, order: int = 7) -> float:
        """Integrate physical Herrmann-pressure error with the imposed mean convention."""
        return self._error(exact, "pressure", order)

    def h1_seminorm_error(self, exact_gradient: Any, order: int = 7) -> float:
        """Integrate the full broken displacement-gradient error."""
        return self._error(exact_gradient, "gradient", order)

    def stress_l2_error(self, exact_stress: Any, order: int = 7) -> float:
        """Integrate the complete symmetric Cauchy-stress Frobenius error."""
        return self._error(exact_stress, "stress", order)

    def compressibility_l2(self, order: int = 7) -> float:
        """Measure div(u)+p/lambda in physical L2, including zero inverse lambda at infinity."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
            compliance = compressibility_values(self.lame_lambda, points.reshape(-1, 3)).reshape(
                points.shape[:2]
            )
            residual = (
                np.trace(self.gradient(cell, bary), axis1=-2, axis2=-1)
                + self.evaluate(cell, bary)[1] * compliance
            )
            total += float(fine.volumes @ (residual**2 @ weights))
        return float(np.sqrt(total))


@dataclass(frozen=True)
class MHSolution:
    """Broken pressure and modified Robin multiplier of the elliptic MH method.

    ``hybrid.trace`` represents (-K grad(p)-p sigma).n in the globally oriented
    skeleton basis. Use ``normal_flux`` for the physical one-sided outward
    flux lambda+p sigma.n. Raw volume fluxes are -K grad(p). These distinctions
    prevent using the Robin multiplier as a conservative Darcy trace.
    """

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    system: HybridSystem
    permeability: Any
    source: Any
    degree: int
    quadrature_order: int
    robin_parameter: float
    origin: FloatArray
    ellipticity_lower_bound: float
    boundary_pressure: dict[int, FloatArray]
    global_matrix: Any
    global_rhs: FloatArray

    @property
    def global_coefficients(self) -> FloatArray:
        """Return Robin lambda followed by pressure on sorted Neumann faces.

        These coefficients satisfy global_matrix/global_rhs before any mean
        augmentation. The system attribute retains the underlying Robin
        condensation, whose response objects reconstruct the volume pressure.
        """
        return np.concatenate((self.hybrid.trace, *self.boundary_pressure.values()))

    def _primal_fields(self) -> DarcySolution:
        """Reuse only the established volume-field integration, not Darcy trace diagnostics."""
        return DarcySolution(
            self.skeleton,
            self.local_meshes,
            self.pressure,
            self.flux,
            self.hybrid,
            "primal",
            self.permeability,
            self.source,
            self.quadrature_order,
            self.degree,
        )

    def l2_error(self, exact: Any, order: int = 8) -> float:
        """Integrate pressure error by independent physical quadrature."""
        return self._primal_fields().l2_error(exact, order)

    def flux_l2_error(self, exact: Any, order: int = 8) -> float:
        """Integrate the raw physical Darcy flux error, not the Robin multiplier error."""
        return self._primal_fields().flux_l2_error(exact, order)

    def normal_flux(self, cell: int, face: int, parameter: Any) -> FloatArray:
        """Evaluate outward physical flux on one incident side of a macroface.

        The Robin multiplier is shared with opposite orientations. Its added
        pressure trace is one-sided; pointwise physical-flux continuity is not
        implied when the discrete pressure jumps.
        """
        mesh = self.skeleton.mesh
        if face not in mesh.cell_faces[cell]:
            raise ValueError("face must be incident to the supplied macrocell")
        side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
        sign = mesh.signs[cell, side]
        t = np.asarray(parameter, dtype=float)
        if t.ndim != 1 or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
            raise ValueError("face parameters must be finite points in [0,1]")
        start, end = mesh.points[mesh.faces[face]]
        points = start + t[:, None] * (end - start)
        pressure = scalar_trace(
            mesh, self.local_meshes[cell], face, self.degree, self.pressure[cell], t
        )
        multiplier = (
            self.skeleton.faces[face].evaluate(t) @ self.hybrid.trace[self.skeleton.dofs(face)]
        )
        sigma_n = self.robin_parameter * ((points - self.origin) @ mesh.normals[face]) / 2
        return sign * (multiplier + pressure * sigma_n)

    def conservation_residuals(self) -> FloatArray:
        """Return integral(lambda+p sigma.n)-integral(f) for every macrocell."""
        values = []
        for response, pressure, (_, robin) in zip(
            self.system.responses, self.pressure, self.system.local_metadata, strict=True
        ):
            problem = response.problem
            values.append(
                np.sum(
                    problem.coupling @ self.hybrid.trace[problem.trace_dofs]
                    + robin @ pressure
                    - problem.load
                )
            )
        return np.asarray(values)


@dataclass(frozen=True)
class MH3DSolution:
    """Tetrahedral MH pressure, Robin multiplier and physical boundary diagnostics.

    The multiplier is (q-p sigma).n_global, with sigma=nu(x-origin)/3.
    Raw volume flux is q=-K grad(p); normal_flux_moments restores p sigma.n.
    The Neumann boundary extension is symmetric indefinite, even though the
    local Robin matrices and the Dirichlet Schur operator are positive definite.
    """

    skeleton: TriangularSkeleton
    local_meshes: tuple[TetraMesh, ...]
    pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    system: HybridSystem
    degree: int
    permeability: Any
    source: Any
    quadrature_order: int
    robin_parameter: float
    origin: FloatArray
    boundary_pressure: dict[int, FloatArray]
    global_matrix: Any
    global_rhs: FloatArray

    def _fields(self) -> Darcy3DSolution:
        """Reuse only physical volume evaluation, not the Darcy trace convention."""
        return Darcy3DSolution(
            self.skeleton,
            self.local_meshes,
            self.pressure,
            self.hybrid,
            self.degree,
            self.permeability,
            self.source,
            self.quadrature_order,
        )

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate the full pressure and physical flux at fine-cell barycentric points."""
        return self._fields().evaluate(cell, bary)

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the broken pressure error using an independent volume rule."""
        return self._fields().l2_error(exact, order)

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the raw physical flux error, not the Robin multiplier error."""
        return self._fields().flux_l2_error(exact, order)

    @property
    def global_coefficients(self) -> FloatArray:
        """Order Robin lambda before auxiliary pressures on sorted Neumann faces."""
        return np.concatenate((self.hybrid.trace, *self.boundary_pressure.values()))

    def normal_flux_moments(self, cell: int, face: int) -> FloatArray:
        """Integrate outward physical flux against all conormal modes on one side."""
        mesh = self.skeleton.mesh
        if face not in mesh.cell_faces[cell]:
            raise ValueError("face must be incident to the supplied macrocell")
        side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
        normal = mesh.signs[cell, side] * mesh.normals[face]
        result = np.zeros(len(self.skeleton.dofs(face)))
        for found, dofs, values, points, weights, bary in boundary_rules(
            mesh, cell, self.local_meshes[cell], self.degree, self.quadrature_order
        ):
            if found != face:
                continue
            basis = broken_face_basis(self.skeleton, face, bary)
            multiplier = mesh.signs[cell, side] * (
                basis @ self.hybrid.trace[self.skeleton.dofs(face)]
            )
            pressure = values @ self.pressure[cell][dofs]
            coefficient = self.robin_parameter * ((points - self.origin) @ normal) / 3
            result += basis.T @ (weights * (multiplier + pressure * coefficient))
        return result

    def conservation_residuals(self) -> FloatArray:
        """Return integrated outward physical flux minus source in every macrocell."""
        return np.array(
            [
                np.sum(
                    response.problem.coupling @ self.hybrid.trace[response.problem.trace_dofs]
                    + metadata[1] @ pressure
                    - response.problem.load
                )
                for response, pressure, metadata in zip(
                    self.system.responses, self.pressure, self.system.local_metadata, strict=True
                )
            ]
        )


@dataclass(frozen=True)
class MH2MSolution:
    """Pressure trace, broken local pressure, and outward physical flux moments.

    ``conormal`` stores lambda=A grad(p).n separately on each macrocell; its
    negative is the physical normal flux. Continuity holds against Gamma test
    functions. Neither pointwise normal continuity nor fine-cell equilibrium
    follows from this weak condition. Raw fluxes are -A grad(p_h).
    """

    trace_space: PressureTraceSpace
    flux_space: SkeletonSpace
    local: tuple[NeumannMaps, ...]
    trace: FloatArray
    pressure: tuple[FloatArray, ...]
    conormal: tuple[FloatArray, ...]
    matrix: Any
    rhs: FloatArray
    free_dofs: IntArray
    degree: int
    permeability: Any
    residual: float

    def conservation_residuals(self) -> FloatArray:
        """Return integral(q.n)-integral(f) per macrocell from conormal moments."""
        return np.array(
            [
                -data.flux_integrals @ lam - data.load.sum()
                for data, lam in zip(self.local, self.conormal, strict=True)
            ]
        )

    def trace_moment_residuals(self) -> tuple[FloatArray, ...]:
        """Return Lambda-tested differences between local pressure and Gamma trace."""
        return tuple(
            data.boundary_coupling.T @ p - data.trace_pairing @ self.trace[data.trace_dofs]
            for data, p in zip(self.local, self.pressure, strict=True)
        )

    def local_equation_residuals(self) -> tuple[FloatArray, ...]:
        """Return original nodal equations A p-B lambda-f, without condensation."""
        return tuple(
            data.stiffness @ p - data.boundary_coupling @ lam - data.load
            for data, p, lam in zip(self.local, self.pressure, self.conormal, strict=True)
        )

    def _error(self, exact: Any, order: int, derivative: bool, flux: bool) -> float:
        """Integrate a physical field error, cutting Cartesian coefficient interfaces."""
        total = 0.0
        for data, coefficients in zip(self.local, self.pressure, strict=True):
            bary, weights, material = material_triangle_quadrature(
                cast(TriangleMesh, data.mesh),
                self.permeability,
                positive_int(order, "quadrature order"),
            )
            dofs, _, basis, gradients, _ = element_tabulate(
                cast(TriangleMesh, data.mesh), self.degree, bary
            )
            points = np.einsum("tqi,tia->tqa", bary, data.mesh.points[data.mesh.cells])
            if derivative:
                field_values = np.einsum("tqia,ti->tqa", gradients, coefficients[dofs])
                if flux:
                    tensor = tensor_values(material, points.reshape(-1, 2)).reshape(
                        *weights.shape, 2, 2
                    )
                    field_values = -np.einsum("tqab,tqb->tqa", tensor, field_values)
                defect = field_values - vector_values(exact, points.reshape(-1, 2)).reshape(
                    field_values.shape
                )
                squared = np.sum(defect**2, axis=-1)
            else:
                values = np.einsum("tqi,ti->tq", basis, coefficients[dofs])
                defect = values - scalar_values(exact, points.reshape(-1, 2)).reshape(weights.shape)
                squared = defect**2
            total += float(np.sum(cast(TriangleMesh, data.mesh).areas[:, None] * weights * squared))
        return float(np.sqrt(total))

    def l2_error(self, exact: Any, order: int = 8) -> float:
        """Return the broken pressure L2 error using independent quadrature."""
        return self._error(exact, order, False, False)

    def gradient_l2_error(self, exact: Any, order: int = 8) -> float:
        """Return the unweighted broken H1 seminorm error."""
        return self._error(exact, order, True, False)

    def flux_l2_error(self, exact: Any, order: int = 8) -> float:
        """Return the raw physical Darcy flux L2 error; this is not the trace error."""
        return self._error(exact, order, True, True)


@dataclass(frozen=True)
class MH2M3DSolution:
    """Continuous pressure trace and reconstructed broken tetrahedral fields.

    Each macrocell has independent outward conormal coefficients
    lambda=K grad(p).n. Their negatives are physical normal fluxes. Global
    continuity is tested by Gamma, not pointwise. The local complement has
    boundary mean zero; a global pure-Neumann gauge fixes the volume mean.
    """

    trace_space: PressureTraceSpace3D
    flux_space: TriangularSkeleton
    local: tuple[NeumannMaps, ...]
    trace: FloatArray
    pressure: tuple[FloatArray, ...]
    conormal: tuple[FloatArray, ...]
    matrix: Any
    rhs: FloatArray
    free_dofs: IntArray
    degree: int
    permeability: Any
    source: Any
    quadrature_order: int
    residual: float

    @property
    def local_meshes(self) -> tuple[TetraMesh, ...]:
        """Return each conforming local tetrahedral partition."""
        return tuple(cast(TetraMesh, data.mesh) for data in self.local)

    def _fields(self) -> Darcy3DSolution:
        """Reuse only volume-field evaluation; this trace is not a Darcy flux skeleton."""
        hybrid = HybridSolution(self.trace, (), self.pressure, self.residual, np.empty(0))
        return Darcy3DSolution(
            self.flux_space,
            self.local_meshes,
            self.pressure,
            hybrid,
            self.degree,
            self.permeability,
            self.source,
            self.quadrature_order,
        )

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate complete reconstructed pressure and raw physical flux."""
        return self._fields().evaluate(cell, bary)

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the broken pressure error with independent tetrahedral quadrature."""
        return self._fields().l2_error(exact, order)

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the raw physical flux error, keeping the full permeability tensor."""
        return self._fields().flux_l2_error(exact, order)

    def conservation_residuals(self) -> FloatArray:
        """Return integrated outward physical conormal flux minus source per macrocell."""
        return np.array(
            [
                -data.flux_integrals @ lam - data.load.sum()
                for data, lam in zip(self.local, self.conormal, strict=True)
            ]
        )

    def trace_moment_residuals(self) -> tuple[FloatArray, ...]:
        """Return Lambda-tested differences of the local pressure and Gamma trace."""
        return tuple(
            data.boundary_coupling.T @ p - data.trace_pairing @ self.trace[data.trace_dofs]
            for data, p in zip(self.local, self.pressure, strict=True)
        )

    def local_equation_residuals(self) -> tuple[FloatArray, ...]:
        """Return original nodal equations A p-B lambda-f before condensation."""
        return tuple(
            data.stiffness @ p - data.boundary_coupling @ lam - data.load
            for data, p, lam in zip(self.local, self.pressure, self.conormal, strict=True)
        )


@dataclass(frozen=True)
class MsHHOLocal:
    """One constrained Galerkin lift and its represented local moment operator.

    ``energy`` is R.T A R, retaining the represented assembly's small
    antisymmetry. For the exactly symmetric diffusion form it is the energy
    Hessian. Projecting this matrix to its symmetric part would change the
    original finite equations.
    """

    mesh: Any
    reconstruction: FloatArray
    moments: FloatArray
    energy: FloatArray
    load: FloatArray
    cell_count: int
    integral: FloatArray


@dataclass(frozen=True)
class MsHHOSolution:
    """Broken pressure fields, common face moments and represented condensed operator."""

    skeleton: Any
    local: tuple[MsHHOLocal, ...]
    pressure: tuple[FloatArray, ...]
    face_moments: FloatArray
    cell_moments: tuple[FloatArray, ...]
    matrix: Any
    residual: float
    degree: int
    permeability: Any
    source_variant: str
    local_refinement_precision: str = "double"

    def l2_error(self, exact: Any, order: int = 8) -> float:
        """Integrate broken pressure error with independent quadrature."""
        bary, weights = triangle_quadrature(positive_int(order, "quadrature order"))
        total = 0.0
        for local, values in zip(self.local, self.pressure, strict=True):
            mesh = local.mesh
            dofs, _, basis, _, _ = tabulate(mesh, self.degree, bary)
            points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
            exact_values = scalar_values(exact, points.reshape(-1, 2)).reshape(points.shape[:2])
            error = values[dofs] @ basis.T - exact_values
            total += float(mesh.areas @ (error**2 @ weights))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact: Any, order: int = 8) -> float:
        """Integrate the physical raw flux -A grad(u); no H(div) claim is made."""
        from pymhm.materials.evaluation import vector_values

        bary, weights = triangle_quadrature(positive_int(order, "quadrature order"))
        total = 0.0
        for local, values in zip(self.local, self.pressure, strict=True):
            mesh = local.mesh
            dofs, _, _, gradient, _ = tabulate(mesh, self.degree, bary)
            points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
            flat = points.reshape(-1, 2)
            tensor = tensor_values(self.permeability, flat).reshape(*points.shape[:2], 2, 2)
            flux = -np.einsum("tqab,tqib,ti->tqa", tensor, gradient, values[dofs])
            error = flux - vector_values(exact, flat).reshape(flux.shape)
            total += float(mesh.areas @ (np.sum(error**2, axis=-1) @ weights))
        return float(np.sqrt(total))


@dataclass(frozen=True)
class MsHHO3DSolution(MsHHOSolution):
    """Broken tetrahedral pressure reconstructions with original-macroface moments."""

    def l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate pressure error in physical volume using independent quadrature."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for local, field in zip(self.local, self.pressure, strict=True):
            fine = local.mesh
            dofs, _, basis, _ = tetra_tabulate(fine, self.degree, bary)
            physical = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
            target = scalar_values_3d(exact, physical.reshape(-1, 3)).reshape(physical.shape[:2])
            difference = field[dofs] @ basis.T - target
            total += float(fine.volumes @ (difference**2 @ weights))
        return float(np.sqrt(total))

    def flux_l2_error(self, exact: Any, order: int = 6) -> float:
        """Integrate the physical raw flux -A grad(p), without an H(div) assertion."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for local, field in zip(self.local, self.pressure, strict=True):
            fine = local.mesh
            dofs, _, _, gradient = tetra_tabulate(fine, self.degree, bary)
            physical = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
            material = tensor_values_3d(self.permeability, physical.reshape(-1, 3)).reshape(
                *physical.shape[:2], 3, 3
            )
            flux = -np.einsum("tqab,tqib,ti->tqa", material, gradient, field[dofs])
            exact_flux = vector_values_3d(exact, physical.reshape(-1, 3)).reshape(flux.shape)
            total += float(fine.volumes @ (np.sum((flux - exact_flux) ** 2, axis=-1) @ weights))
        return float(np.sqrt(total))


@dataclass(frozen=True)
class PGMHMSolution:
    """Base and enriched Darcy fields of the residual-based Petrov-Galerkin method.

    ``pressure`` is equation (31); ``enriched_pressure`` is equation (34).
    ``hybrid.trace`` stores the unenriched physical flux multiplier. Physical
    macro conservation requires ``normal_flux(..., enriched=True)``.
    Both volume flux fields are raw gradients, not H(div) reconstructions.
    """

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    pressure: tuple[FloatArray, ...]
    enriched_pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    system: HybridSystem
    permeability: Any
    source: Any
    degree: int
    quadrature_order: int
    stabilization_parameter: float
    ellipticity_lower_bound: float
    penalties: tuple[FaceJumpForm, ...]
    enrichment_loads: tuple[FloatArray, ...]

    def _fields(self, enriched: bool) -> DarcySolution:
        """Reuse only physical volume norm integration with the selected pressure field."""
        values = self.enriched_pressure if enriched else self.pressure
        return DarcySolution(
            self.skeleton,
            self.local_meshes,
            values,
            tuple(np.empty(0) for _ in values),
            self.hybrid,
            "primal",
            self.permeability,
            self.source,
            self.quadrature_order,
            self.degree,
        )

    def l2_error(self, exact: Any, order: int = 8, *, enriched: bool = False) -> float:
        """Integrate base or enriched pressure error in the physical L2 norm."""
        return self._fields(enriched).l2_error(exact, order)

    def flux_l2_error(self, exact: Any, order: int = 8, *, enriched: bool = False) -> float:
        """Integrate -K grad(p) error with the selected base or enriched pressure."""
        return self._fields(enriched).flux_l2_error(exact, order)

    def normal_flux(
        self, cell: int, face: int, parameter: Any, *, enriched: bool = True
    ) -> FloatArray:
        """Evaluate one-sided outward flux, including the residual enrichment by default."""
        mesh = self.skeleton.mesh
        if face not in mesh.cell_faces[cell]:
            raise ValueError("face must be incident to the supplied macrocell")
        t = np.asarray(parameter, dtype=float)
        if t.ndim != 1 or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
            raise ValueError("face parameters must be finite points in [0,1]")
        result = self.skeleton.faces[face].evaluate(t) @ self.hybrid.trace[self.skeleton.dofs(face)]
        data = next((item for item in self.penalties if item.face == face), None)
        if enriched and data is not None:
            jump = np.zeros(len(t))
            for (neighbor, sign, _), trace in zip(data.sides, data.traces, strict=True):
                jump += sign * trace.evaluate(self.pressure[neighbor], t)
            result -= data.coefficient * (jump - data.boundary_value(t))
        side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
        return mesh.signs[cell, side] * result

    def conservation_residuals(self, *, enriched: bool = True) -> FloatArray:
        """Return integral(q.n)-integral(f); only the enriched multiplier conserves macros."""
        residuals = []
        for response, extra in zip(self.system.responses, self.enrichment_loads, strict=True):
            p = response.problem
            residual = p.coupling @ self.hybrid.trace[p.trace_dofs] - p.load
            if enriched:
                residual = residual + extra
            residuals.append(np.sum(residual))
        return np.asarray(residuals)
