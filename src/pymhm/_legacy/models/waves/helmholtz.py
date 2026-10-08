"""Multiscale Hybrid-Mixed Helmholtz with absorbing and Dirichlet boundaries.

The complex problem is represented exactly by interleaved real/imaginary
coordinates. No damping, shifted operator or approximate local kernel is added.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.loads import point_load_vector, split_point_sources
from pymhm.fem.scalar.helmholtz import (
    acoustic_space,
    complex_values,
    complex_vector,
    edge_rule,
    material_trace,
    real_matrix,
    real_vector,
    volume_forms,
)
from pymhm.fem.traces.helmholtz import PolynomialNeumannTrace, helmholtz_skeleton
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.acoustics import HelmholtzSolution
from pymhm.postprocessing.acoustics import (
    local_helmholtz_error_squared as local_helmholtz_error_squared,
)


def _boundary_map(value: Any, mesh: Any) -> dict[int, Any]:
    """Interpret a scalar/callback as every exterior face, or validate a face map."""
    exterior = mesh.boundary_faces
    result = (
        {}
        if value is None
        else dict(value)
        if isinstance(value, Mapping)
        else dict.fromkeys(exterior, value)
    )
    if not result.keys() <= set(exterior):
        raise ValueError("boundary maps must contain only exterior face IDs")
    return result


def _boundary_values(value: Any, points: FloatArray, normal: FloatArray) -> Any:
    """Evaluate scalar boundary data whose callable convention includes the normal."""
    return complex_values(
        value(points, np.broadcast_to(normal, points.shape)) if callable(value) else value, points
    )


@dataclass(frozen=True)
class _HelmholtzFactory:
    """Portable complete local assembly followed by the shared exact condensation."""

    mesh: Any
    skeleton: SkeletonSpace
    omega: float
    degree: int
    refinement: int
    order: int
    density: Any
    modulus: Any
    source: Any
    point_sources: tuple[FloatArray, ...]
    dirichlet: Any
    absorbing: dict[int, Any]
    neumann: dict[int, Any]
    pml_stretch: Any
    _normals: FloatArray = field(init=False, repr=False)
    _boundary_faces: frozenset[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Evaluate immutable global geometry once before dispatching local problems."""
        normals = np.array(self.mesh.normals, copy=True)
        normals.setflags(write=False)
        object.__setattr__(self, "_normals", normals)
        object.__setattr__(self, "_boundary_faces", frozenset(self.mesh.boundary_faces))

    def __call__(self, cell: int) -> LocalAssembly:
        """Assemble rho^-1 diffusion, negative mass and outgoing-wave impedance."""
        fine = self.mesh.submesh(cell, self.refinement)
        stiffness, mass, load = volume_forms(
            fine, self.degree, self.density, self.modulus, self.source, self.order, self.pml_stretch
        )
        if len(self.point_sources[cell]):
            load += point_load_vector(fine, self.degree, self.point_sources[cell])
        matrix = (stiffness - self.omega**2 * mass).astype(complex).tolil()
        _, nodes = acoustic_space(fine, self.degree)
        widths = [self.skeleton.faces[face].size for face in self.mesh.cell_faces[cell]]
        coupling = np.zeros((len(nodes), sum(widths)))
        boundary = np.zeros(sum(widths), dtype=complex)
        prescribed: dict[int, complex] = {}
        offset = 0
        for side, face in enumerate(self.mesh.cell_faces[cell]):
            space = self.skeleton.faces[face]
            normal = self._normals[face] * self.mesh.signs[cell, side]
            face_mass = np.zeros((space.size, space.size))
            constant_neumann: complex | None = None
            neumann_samples_constant = True
            face_load = np.zeros(space.size, dtype=complex)
            for ids, parameter, points, weights, basis in edge_rule(
                self.mesh,
                fine,
                cell,
                face,
                self.degree,
                space,
                self.order,
                (self.density, self.modulus, self.dirichlet),
            ):
                face_basis = space.evaluate(parameter)
                if face in self.absorbing:
                    interior = self.mesh.points[self.mesh.cells[cell]].mean(axis=0)
                    density = material_trace(self.density, points, interior)
                    modulus = material_trace(self.modulus, points, interior)
                    block = basis.T @ ((weights / np.sqrt(density * modulus))[:, None] * basis)
                    matrix[np.ix_(ids, ids)] -= 1j * self.omega * block
                    value = _boundary_values(self.absorbing[face], points, normal)
                    np.add.at(load, ids, basis.T @ (weights * value))
                else:
                    coupling[ids, offset : offset + space.size] += (
                        self.mesh.signs[cell, side] * basis.T @ (weights[:, None] * face_basis)
                    )
                    if face in self.neumann:
                        data = self.neumann[face]
                        values = (
                            data.evaluate(parameter)
                            if isinstance(data, PolynomialNeumannTrace)
                            else _boundary_values(data, points, normal)
                        )
                        if constant_neumann is None:
                            constant_neumann = complex(values[0])
                        neumann_samples_constant &= bool(np.all(values == constant_neumann))
                        face_mass += face_basis.T @ (weights[:, None] * face_basis)
                        face_load += face_basis.T @ (weights * values)
                    elif face in self._boundary_faces:
                        boundary[offset : offset + space.size] += face_basis.T @ (
                            weights * complex_values(self.dirichlet, points)
                        )
            if face in self.absorbing:
                prescribed.update(dict.fromkeys(range(offset, offset + space.size), 0j))
            elif face in self.neumann:
                # This preserves a constant in the executed quadrature data, including
                # callbacks. It does not establish constancy away from those samples.
                coefficients = (
                    self.neumann[face].coefficients_on(space)
                    if isinstance(self.neumann[face], PolynomialNeumannTrace)
                    else constant_neumann * space.constant_coefficients()
                    if neumann_samples_constant and constant_neumann is not None
                    else np.linalg.solve(face_mass, face_load)
                )
                prescribed.update(
                    dict(zip(range(offset, offset + space.size), coefficients, strict=True))
                )
            offset += space.size
        problem = LocalProblem(
            real_matrix(matrix.tocsc()),
            sparse.kron(coupling, sparse.eye(2)).toarray(),
            real_vector(load),
            self.skeleton.cell_dofs(cell),
        )
        return LocalAssembly(problem, (fine, mass, real_vector(boundary), prescribed))


def solve_helmholtz(
    mesh: TriangleMesh | PolygonMesh | CartesianMacroMesh,
    *,
    omega: float,
    density: Any = 1.0,
    bulk_modulus: Any = 1.0,
    source: Any = 0j,
    point_sources: Any = (),
    dirichlet: Any = 0j,
    absorbing: Any = 0j,
    neumann: Any = None,
    pml_stretch: Any = None,
    skeleton: SkeletonSpace | None = None,
    degree: int = 3,
    local_refinement: int = 2,
    quadrature_order: int = 8,
    solver: str = "scipy",
    local_solver: str = "scipy",
    local_refinement_precision: Literal["double", "extended"] = "double",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> HelmholtzSolution:
    """Solve -div(rho^-1 grad u)-omega²*kappa^-1*u=f by the 2020 MHM method.

    ``absorbing`` maps exterior faces to g in rho^-1*du/dn-i*omega*u/sqrt(kappa*rho)=g;
    a scalar or callable selects every exterior face, and None selects none.
    Absorbing and Neumann callables receive (points, outward_unit_normals).
    ``neumann`` uses physical q.n=-rho^-1*du/dn on a disjoint exterior subset.
    ``PolynomialNeumannTrace`` supplies declared Legendre data in the oriented
    macroface parameter, preserving degree under polynomial trace restriction.
    Generic callbacks are projected by quadrature; equality of sampled constant
    data does not establish analytic constancy between the samples.
    Remaining exterior faces prescribe ``dirichlet(points)`` weakly. Complex
    fields use component-interleaved real coordinates, exactly equivalent to
    complex algebra. No pressure gauge is inserted at a positive frequency.
    Scalar Cartesian materials and isotropic PlanarMaterial inputs receive
    interface-fitted integration; integration alone does not enrich the local
    approximation space to represent a derivative jump.

    ``point_sources`` supplies finite real (x,y,strength) rows. Discrete Dirac
    functionals evaluate local Pk/Qk test functions at the physical source point;
    sources on macro interfaces are shared by the incident angular sectors.
    A point is evaluated once within each continuous local space. In two
    dimensions a Dirac source has no finite continuum H1 energy norm; use
    sampled field or appropriate weaker norms for such comparisons.

    ``local_solver`` selects the source/trace factorization independently of
    the global ``solver``. ``local_refinement_precision="extended"`` retains
    wider corrected local responses where the NumPy platform supports them;
    factorization and the original residual tolerance are unchanged.

    Local resonances and incompatible trace spaces are rejected by the shared
    factorization/rank and original-equation residual checks. Sufficient paper
    resolution assumptions are stronger than algebraic solvability; a passing
    solve does not establish wave accuracy or remove Helmholtz pollution.

    ``pml_stretch`` optionally supplies diagonal complex coordinate stretches
    (s_x,s_y). Section 3.3 then uses D=diag(s_y/s_x,s_x/s_y) in diffusion and
    d=s_x*s_y in the negative mass term. The source is the transformed equation's
    RHS. Select absorbing=None: this variant supports Dirichlet or physical
    conormal data on the exterior, without combining an untransformed absorbing
    condition with PML coefficients. Stretches must have positive real parts and
    nonnegative imaginary parts; they do not by themselves establish a PML
    truncation-error bound. No coordinate damping is inserted when omitted.
    """
    if not isinstance(mesh, (TriangleMesh, PolygonMesh, CartesianMacroMesh)):
        raise TypeError("Helmholtz requires a planar triangular, polygonal or Cartesian mesh")
    if np.iscomplexobj(omega) or not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be a finite positive angular frequency")
    degree = positive_int(degree, "degree")
    refinement = positive_int(local_refinement, "local_refinement")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    skeleton = helmholtz_skeleton(mesh, omega) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 2:
        raise ValueError("Helmholtz requires a two-component skeleton on the supplied mesh")
    order = max(order, degree + max(max(face.degrees) for face in skeleton.faces) + 2)
    absorbing_map, neumann_map = _boundary_map(absorbing, mesh), _boundary_map(neumann, mesh)
    if absorbing_map.keys() & neumann_map.keys():
        raise ValueError("absorbing and Neumann face sets must be disjoint")
    if pml_stretch is not None and absorbing_map:
        raise ValueError("PML requires absorbing=None; specify exterior Dirichlet/Neumann data")
    factory = _HelmholtzFactory(
        mesh,
        skeleton,
        float(omega),
        degree,
        refinement,
        order,
        density,
        bulk_modulus,
        source,
        split_point_sources(mesh, point_sources),
        dirichlet,
        absorbing_map,
        neumann_map,
        pml_stretch,
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        local_solver=local_solver,
        local_refinement_precision=local_refinement_precision,
        backend=backend,
        workers=workers,
    )
    fixed = {}
    for response, (_, _, boundary, prescribed) in zip(
        system.responses, system.local_metadata, strict=True
    ):
        ids = response.problem.trace_dofs
        system.rhs[ids] -= boundary
        system.load_scale[ids] += abs(boundary)
        for scalar, value in prescribed.items():
            fixed[int(ids[2 * scalar])] = float(value.real)
            fixed[int(ids[2 * scalar + 1])] = float(value.imag)
    hybrid = system.solve(solver=solver, fixed=fixed)
    local_meshes = tuple(data[0] for data in system.local_metadata)
    return HelmholtzSolution(
        skeleton,
        local_meshes,
        tuple(complex_vector(field) for field in hybrid.fields),
        complex_vector(hybrid.trace),
        hybrid,
        system,
        float(omega),
        density,
        bulk_modulus,
        source,
        degree,
        order,
        absorbing_map,
        pml_stretch,
        factory.point_sources,
    )
