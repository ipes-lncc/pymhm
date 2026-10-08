"""Reuse Helmholtz responses and factors when only absorbing incident data vary.

Every changed local load is assembled by the original Helmholtz factory. The
shared local and global solvers check exact operator identity before reusing
factors, and the complete reconstructed solution retains its original equations.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from typing import Any

import numpy as np

from examples.helmholtz_trace_family import verify_helmholtz_solution
from examples.local_response_cache import array_identity, operator_identity
from examples.tutorial_helmholtz_equations import AcousticAssemblyProvider
from pymhm.core.contracts import LocalResponse
from pymhm.core.system import HybridSystem
from pymhm.fem.scalar.helmholtz import complex_vector
from pymhm.linalg.linear import LinearFactorization, factorize
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.acoustics import HelmholtzSolution


class IncidentFamily:
    """Context-managed fixed-operator family with complete physical field checks.

    The prepared solution must use a Cartesian macro mesh, no PML and absorption
    on every exterior face. Materials, volumetric/point sources, local mesh,
    trace basis and quadrature are fixed. Only the impedance datum changes.
    The actual response and global trace dtypes retain their respective
    double or extended correction conventions during subsequent solves.
    Factorizations remain private to this serial process and close explicitly.
    """

    def __init__(self, prepared: HelmholtzSolution) -> None:
        """Retain the executed response basis and validate the declared boundary family."""
        mesh = prepared.skeleton.mesh
        if (
            not isinstance(mesh, CartesianMacroMesh)
            or prepared.pml_stretch is not None
            or set(prepared.absorbing) != set(mesh.boundary_faces)
        ):
            raise ValueError(
                "incident families require Cartesian meshes and full exterior absorption"
            )
        self.prepared = prepared
        self._local_precision = (
            "extended"
            if any(
                np.finfo(np.asarray(array).dtype).eps < np.finfo(float).eps
                for response in prepared.system.responses
                for array in (response.source, response.lifts)
            )
            else "double"
        )
        self._global_precision = (
            "extended"
            if np.finfo(prepared.hybrid.trace.dtype).eps < np.finfo(float).eps
            else "double"
        )
        self.factory = AcousticAssemblyProvider(
            mesh,
            prepared.skeleton,
            prepared.omega,
            prepared.degree,
            prepared.local_meshes[0].nx,
            prepared.quadrature_order,
            prepared.density,
            prepared.bulk_modulus,
            prepared.source,
            prepared.point_sources,
            0j,
            prepared.absorbing,
            {},
            None,
        )
        exterior = set(prepared.absorbing)
        self.boundary_cells = tuple(
            cell
            for cell, faces in enumerate(mesh.cell_faces)
            if any(face in exterior for face in faces)
        )
        self.fixed = {
            int(dof): 0.0 for face in prepared.absorbing for dof in prepared.skeleton.dofs(face)
        }
        self.free = np.setdiff1d(np.arange(prepared.skeleton.size), list(self.fixed))
        self._stack: ExitStack | None = None
        self._local_factors: dict[int, LinearFactorization] = {}
        self._global_factor: LinearFactorization | None = None
        self.last_field_diagnostics: dict[str, float] | None = None

    def __enter__(self) -> IncidentFamily:
        """Prepare reusable source and global factors with exception-safe cleanup."""
        if self._stack is not None:
            raise RuntimeError("incident family is already open")
        self._stack = ExitStack()
        try:
            for cell in self.boundary_cells:
                problem = self.prepared.system.responses[cell].problem
                self._local_factors[cell] = self._stack.enter_context(
                    factorize(problem.condensation_matrix())
                )
            if len(self.free):
                self._global_factor = self._stack.enter_context(
                    factorize(self.prepared.system.matrix[self.free][:, self.free])
                )
        except BaseException:
            self.close()
            raise
        return self

    def solve(self, absorbing_data: Any) -> HelmholtzSolution:
        """Reassemble incident loads and solve using unchanged harmonic response columns."""
        if self._stack is None:
            raise RuntimeError("incident family must be open")
        self.last_field_diagnostics = None
        factory = replace(
            self.factory, absorbing=dict.fromkeys(self.prepared.absorbing, absorbing_data)
        )
        responses = list(self.prepared.system.responses)
        metadata = list(self.prepared.system.local_metadata)
        for cell in self.boundary_cells:
            assembled = factory(cell)
            original = responses[cell]
            problem = assembled.problem
            if operator_identity(problem) != operator_identity(original.problem) or array_identity(
                problem.trace_dofs
            ) != array_identity(original.problem.trace_dofs):
                raise ValueError(
                    "incident data changed the prepared local operator or injection map"
                )
            source = problem.reconstruct(
                np.zeros(problem.coupling.shape[1]),
                np.empty(0),
                factorization=self._local_factors[cell],
                refinement_precision=self._local_precision,
            )
            responses[cell] = LocalResponse(
                problem, source, original.lifts, original.coarse_vectors
            )
            metadata[cell] = assembled.metadata
        system = HybridSystem.from_responses(responses, metadata=metadata)
        for response, (_, _, boundary, _) in zip(system.responses, metadata, strict=True):
            dofs = response.problem.trace_dofs
            system.rhs[dofs] -= boundary
            system.load_scale[dofs] += abs(boundary)
        hybrid = system.solve(
            fixed=self.fixed,
            factorization=self._global_factor,
            refinement_precision=self._global_precision,
        )
        solution = replace(
            self.prepared,
            pressure=tuple(complex_vector(field) for field in hybrid.fields),
            trace=complex_vector(hybrid.trace),
            hybrid=hybrid,
            system=system,
            absorbing=factory.absorbing,
        )
        self.last_field_diagnostics = verify_helmholtz_solution(solution)
        return solution

    def close(self) -> None:
        """Release every local/global native factor, including after a failed solve."""
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        self._local_factors.clear()
        self._global_factor = None

    def __exit__(self, *exc: Any) -> None:
        """Close factors when leaving the execution context."""
        self.close()
