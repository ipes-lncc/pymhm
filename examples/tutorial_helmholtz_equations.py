"""User-defined complex acoustic forms with outgoing data on every exterior face.

This focused tutorial reuses reference bases, material quadrature, edge maps
and the exact real/imaginary embedding. It declares its own volume, impedance
and global continuity forms. Its affine manufactured patch is separate from
the archived wave and PML convergence studies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core.assembly import SolverConfig
from pymhm.core.contracts import LocalAssembly
from pymhm.core.equations import Equation, LocalEquations, compile_local_equations
from pymhm.core.multiscale import MultiscaleProblem, MultiscaleSystem, assemble
from pymhm.core.validation import positive_int
from pymhm.execution.cpu import ExecutionConfig
from pymhm.fem.loads import point_load_vector, split_point_sources
from pymhm.fem.scalar.helmholtz import (
    acoustic_quadrature,
    acoustic_space,
    complex_values,
    complex_vector,
    edge_rule,
    material_trace,
    volume_forms,
)
from pymhm.fem.traces.helmholtz import PolynomialNeumannTrace, helmholtz_skeleton
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.linalg.complex import realify_operator, realify_vector
from pymhm.postprocessing.acoustics import HelmholtzSolution
from pymhm.postprocessing.nodal import nodal_field


@dataclass(frozen=True)
class AcousticCell:
    """Physical mesh/material data for a declared local acoustic form."""

    macro: Any
    skeleton: SkeletonSpace
    cell: int
    omega: float
    degree: int
    refinement: int
    order: int
    density: Any
    modulus: Any
    source: Any
    boundary: Any
    dirichlet: Any = 0.0
    neumann: Any = None
    pml_stretch: Any = None
    point_sources: Any = ()


@dataclass(frozen=True)
class AcousticAssemblyProvider:
    """Compile declared acoustic forms for portable response/archive acquisition.

    This compatibility adapter selects no package method. It constructs the
    same :func:`acoustic_equations` used by the generic multiscale application,
    then compiles their original local A/B/C blocks through the public compiler.
    The returned metadata includes the Dirichlet functional and prescribed
    coordinates for response-store callers that explicitly assemble balance.
    """

    mesh: Any
    skeleton: SkeletonSpace
    omega: float
    degree: int
    refinement: int
    order: int
    density: Any
    modulus: Any
    source: Any
    point_sources: Any
    dirichlet: Any
    absorbing: Any
    neumann: Any
    pml_stretch: Any

    def __call__(self, cell: int) -> LocalAssembly:
        """Return checked original coefficient equations in the declared local trace map."""
        forms = acoustic_equations(
            AcousticCell(
                self.mesh,
                self.skeleton,
                cell,
                self.omega,
                self.degree,
                self.refinement,
                self.order,
                self.density,
                self.modulus,
                self.source,
                self.absorbing,
                self.dirichlet,
                self.neumann,
                self.pml_stretch,
                self.point_sources[cell],
            )
        )
        compiled = compile_local_equations(forms)
        return LocalAssembly(compiled.problem, compiled.metadata)


def acoustic_equations(item: AcousticCell) -> LocalEquations:
    """Declare diffusion minus frequency mass with exterior outgoing impedance.

    The local equation is A p+B lambda=f. Interior B maps canonical physical
    normal-flux moments, and C=-B.T declares pressure continuity globally.
    Absorbing exterior coordinates are zero placeholders; the actual physical
    flux there follows from pressure and the prescribed impedance data.
    Complex coordinates are interleaved Re/Im through the common exact maps.
    """
    macro, skeleton, cell = item.macro, item.skeleton, item.cell
    fine = macro.submesh(cell, item.refinement)
    stiffness, mass, load = volume_forms(
        fine, item.degree, item.density, item.modulus, item.source, item.order, item.pml_stretch
    )
    if len(item.point_sources):
        load += point_load_vector(fine, item.degree, item.point_sources)
    matrix = (stiffness - item.omega**2 * mass).astype(complex).tolil()
    _, nodes = acoustic_space(fine, item.degree)
    width = sum(skeleton.faces[face].size for face in macro.cell_faces[cell])
    pairing = np.zeros((len(nodes), width))
    boundary = np.zeros(width, dtype=complex)
    prescribed: dict[int, complex] = {}
    absorbing = boundary_faces(item.boundary, macro)
    neumann = boundary_faces(item.neumann, macro)
    offset = 0
    exterior = set(macro.boundary_faces)
    for side, face in enumerate(macro.cell_faces[cell]):
        space = skeleton.faces[face]
        normal = macro.normals[face] * macro.signs[cell, side]
        face_mass = np.zeros((space.size, space.size))
        face_load = np.zeros(space.size, dtype=complex)
        constant_neumann: complex | None = None
        constant_samples = True
        for ids, parameter, points, weights, basis in edge_rule(
            macro,
            fine,
            cell,
            face,
            item.degree,
            space,
            item.order,
            (item.density, item.modulus, item.dirichlet),
        ):
            face_basis = space.evaluate(parameter)
            if face in absorbing:
                interior = macro.points[macro.cells[cell]].mean(axis=0)
                density = material_trace(item.density, points, interior)
                modulus = material_trace(item.modulus, points, interior)
                matrix[np.ix_(ids, ids)] -= (
                    1j
                    * item.omega
                    * (basis.T @ ((weights / np.sqrt(density * modulus))[:, None] * basis))
                )
                data = (
                    absorbing[face](points, np.broadcast_to(normal, points.shape))
                    if callable(absorbing[face])
                    else absorbing[face]
                )
                values = complex_values(data, points)
                np.add.at(load, ids, basis.T @ (weights * values))
            else:
                pairing[ids, offset : offset + space.size] += (
                    macro.signs[cell, side] * basis.T @ (weights[:, None] * face_basis)
                )
                if face in neumann:
                    datum = neumann[face]
                    values = (
                        datum.evaluate(parameter)
                        if isinstance(datum, PolynomialNeumannTrace)
                        else complex_values(
                            datum(points, np.broadcast_to(normal, points.shape))
                            if callable(datum)
                            else datum,
                            points,
                        )
                    )
                    if constant_neumann is None:
                        constant_neumann = complex(values[0])
                    constant_samples &= bool(np.all(values == constant_neumann))
                    face_mass += face_basis.T @ (weights[:, None] * face_basis)
                    face_load += face_basis.T @ (weights * values)
                elif face in exterior:
                    boundary[offset : offset + space.size] += face_basis.T @ (
                        weights * complex_values(item.dirichlet, points)
                    )
        if face in absorbing:
            prescribed.update(dict.fromkeys(range(offset, offset + space.size), 0j))
        elif face in neumann:
            coefficients = (
                neumann[face].coefficients_on(space)
                if isinstance(neumann[face], PolynomialNeumannTrace)
                else constant_neumann * space.constant_coefficients()
                if constant_samples and constant_neumann is not None
                else np.linalg.solve(face_mass, face_load)
            )
            prescribed.update(
                dict(zip(range(offset, offset + space.size), coefficients, strict=True))
            )
        offset += space.size
    coupling = sparse.kron(pairing, sparse.eye(2)).toarray()
    return LocalEquations(
        a=realify_operator(matrix.tocsc()),
        L=realify_vector(load),
        b=coupling,
        c=-coupling.T,
        dofs=skeleton.cell_dofs(cell),
        g=-realify_vector(boundary),
        metadata=(fine, mass, realify_vector(boundary), prescribed),
        field_data=(nodal_field("pressure", fine, item.degree, components=2),),
    )


def boundary_faces(value: Any, mesh: Any) -> dict[int, Any]:
    """Declare exterior face data: None selects none, a callback selects all.

    A mapping chooses individual macrofaces. No interior face can be assigned
    boundary data. Scalar pressure data is complex; normal-dependent callbacks
    receive physical points and outward unit normals.
    """
    result = (
        {}
        if value is None
        else dict(value)
        if isinstance(value, Mapping)
        else dict.fromkeys(mesh.boundary_faces, value)
    )
    if not result.keys() <= set(mesh.boundary_faces):
        raise ValueError("boundary maps must contain only exterior macrofaces")
    return result


def acoustic_problem(
    mesh: Any,
    *,
    omega: float,
    source: Any = 0j,
    dirichlet: Any = 0j,
    absorbing: Any = 0j,
    neumann: Any = None,
    density: Any = 1.0,
    bulk_modulus: Any = 1.0,
    pml_stretch: Any = None,
    point_sources: Any = (),
    skeleton: SkeletonSpace | None = None,
    degree: int = 3,
    local_refinement: int = 2,
    quadrature_order: int = 8,
) -> MultiscaleProblem[AcousticCell]:
    """Declare complex acoustic volume, face and continuity equations explicitly.

    Absorbing data is ``rho^-1 dp/dn-i*omega*p/sqrt(rho*kappa)=g``.
    Neumann data is physical ``q.n=-rho^-1 dp/dn``. Remaining exterior faces
    prescribe pressure weakly. PML declares the transformed operator and
    therefore requires absorbing=None. Sources use physical volume units;
    finite point loads retain the common sector-sharing convention. At positive
    frequency no mean-pressure gauge is inserted. Local resonance or trace
    incompatibility must still be rejected by the common algebraic checks.
    """
    if np.iscomplexobj(omega) or not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be a finite positive angular frequency")
    degree = positive_int(degree, "degree")
    refinement = positive_int(local_refinement, "local_refinement")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    space = helmholtz_skeleton(mesh, omega) if skeleton is None else skeleton
    if space.mesh is not mesh or space.components != 2:
        raise ValueError("acoustic traces need interleaved Re/Im coordinates on this mesh")
    order = max(order, degree + max(max(face.degrees) for face in space.faces) + 2)
    absorbing_map, neumann_map = boundary_faces(absorbing, mesh), boundary_faces(neumann, mesh)
    if absorbing_map.keys() & neumann_map.keys():
        raise ValueError("absorbing and Neumann faces must be disjoint")
    if pml_stretch is not None and absorbing_map:
        raise ValueError("PML requires absorbing=None")
    loads = split_point_sources(mesh, point_sources)
    items = tuple(
        AcousticCell(
            mesh,
            space,
            cell,
            float(omega),
            degree,
            refinement,
            order,
            density,
            bulk_modulus,
            source,
            absorbing_map,
            dirichlet,
            neumann_map,
            pml_stretch,
            loads[cell],
        )
        for cell in range(len(mesh.cells))
    )
    return MultiscaleProblem(
        Equation(0, 0), acoustic_equations, items, space.size, (0,) * len(items)
    )


def acoustic_prescribed(system: MultiscaleSystem) -> dict[int, float]:
    """Read projected physical Neumann coordinates and zero absorbing placeholders."""
    fixed = {}
    for response, metadata in zip(system.responses, system.local_metadata, strict=True):
        ids = response.problem.trace_dofs
        for scalar, value in metadata[3].items():
            fixed[int(ids[2 * scalar])] = float(value.real)
            fixed[int(ids[2 * scalar + 1])] = float(value.imag)
    return fixed


def recover_acoustic(
    problem: MultiscaleProblem[AcousticCell], system: MultiscaleSystem, solution: Any
) -> HelmholtzSolution:
    """Expose physical complex fields from the generic core's real coefficient basis."""
    item = tuple(problem.items)[0]
    return HelmholtzSolution(
        item.skeleton,
        tuple(record[0] for record in system.local_metadata),
        tuple(complex_vector(field) for field in solution.fields),
        complex_vector(solution.trace),
        solution,
        system,
        item.omega,
        item.density,
        item.modulus,
        item.source,
        item.degree,
        item.order,
        boundary_faces(item.boundary, item.macro),
        item.pml_stretch,
        tuple(cell.point_sources for cell in problem.items),
    )


def solve_acoustic(
    mesh: Any,
    *,
    solver: str = "scipy",
    local_solver: str = "scipy",
    local_refinement_precision: Any = "double",
    backend: Any = "serial",
    workers: int | None = None,
    **forms: Any,
) -> HelmholtzSolution:
    """Execute the user-written acoustic equations with generic assembly and solve.

    This example is a composition of public equations and operator kernels,
    not a dispatch to a package acoustic method. ``acoustic_problem`` contains
    the full mathematical declaration; users may replace any of its blocks.
    """
    problem = acoustic_problem(mesh, **forms)
    system = assemble(
        problem,
        execution=ExecutionConfig(backend=backend, workers=workers),
        solvers=SolverConfig(
            local_solver=local_solver,
            global_solver=solver,
            local_refinement_precision=local_refinement_precision,
        ),
    )
    solution = system.solve(fixed=acoustic_prescribed(system))
    return recover_acoustic(problem, system, solution)


def solve_patch(
    mesh: Any,
    *,
    omega: float,
    source: Any,
    boundary: Any,
    degree: int = 2,
    local_refinement: int = 2,
    quadrature_order: int = 8,
    density: Any = 1.0,
    modulus: Any = 1.0,
) -> tuple[MultiscaleSystem, Any]:
    """Solve the explicitly declared all-impedance patch through the generic core."""
    problem = acoustic_problem(
        mesh,
        omega=omega,
        source=source,
        absorbing=boundary,
        degree=degree,
        local_refinement=local_refinement,
        quadrature_order=quadrature_order,
        density=density,
        bulk_modulus=modulus,
    )
    space = tuple(problem.items)[0].skeleton
    fixed = {int(index): 0.0 for face in mesh.boundary_faces for index in space.dofs(face)}
    problem = replace(problem, fixed=fixed)
    system = assemble(problem)
    return system, system.solve()


def pressure_error(
    system: MultiscaleSystem, solution: Any, exact: Any, degree: int = 2, order: int = 8
) -> float:
    """Integrate the complex physical pressure error on independent local meshes."""
    total = 0.0
    for metadata, values in zip(system.local_metadata, solution.fields, strict=True):
        fine = metadata[0]
        dofs, points, weights, basis, _, _ = acoustic_quadrature(fine, degree, 1.0, order)
        field = np.einsum("tqi,ti->tq", basis, complex_vector(values)[dofs])
        truth = complex_values(exact, points.reshape(-1, 2)).reshape(weights.shape)
        total += float(np.sum(weights * abs(field - truth) ** 2))
    return float(np.sqrt(total))


def macro_balances(system: MultiscaleSystem, solution: Any) -> Any:
    """Evaluate the executed constant-test equations separately in each macrocell."""
    result = []
    for response, values in zip(system.responses, solution.fields, strict=True):
        problem = response.problem
        defect = (
            problem.matrix @ values
            + problem.coupling @ solution.trace[problem.trace_dofs]
            - problem.load
        )
        result.append(np.sum(complex_vector(defect)))
    return np.asarray(result)
