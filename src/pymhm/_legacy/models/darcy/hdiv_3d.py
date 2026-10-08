"""Mixed MHM Darcy with tetrahedral BDFM and affine prismatic local spaces."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm._legacy.models.darcy._mixed import normal_flux_blocks
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.hdiv.family_3d import (
    HDiv3DFamily,
    face_size,
)
from pymhm.fem.hdiv.mixed_3d import Mixed3DSkeleton as Mixed3DSkeleton
from pymhm.fem.hdiv.mixed_3d import _FacePartition as _FacePartition
from pymhm.fem.hdiv.mixed_3d import _partition as _partition
from pymhm.fem.hdiv.mixed_3d import hdiv3d_boundary_data as _boundary
from pymhm.fem.hdiv.mixed_3d import hdiv3d_operators as hdiv3d_operators
from pymhm.fem.hdiv.mixed_3d import hdiv3d_trace_mapping as _trace_mapping
from pymhm.meshes.mixed import AffineMixedMesh, hdiv3d_face_offsets
from pymhm.postprocessing.solutions import Mixed3DDarcySolution as Mixed3DDarcySolution


@dataclass(frozen=True)
class _LocalFactory:
    """Assemble one mixed Neumann problem and its physical constant pressure mode."""

    skeleton: Mixed3DSkeleton
    family: HDiv3DFamily
    refinement: int
    permeability: Any
    source: Any
    order: int

    def __call__(self, cell: int) -> LocalAssembly:
        """Return the physical saddle, trace map and dimensional pressure constraint."""
        fine = self.skeleton.mesh.submesh(cell, self.refinement)
        M, D, f, moment = hdiv3d_operators(
            fine,
            self.family,
            permeability=self.permeability,
            source=self.source,
            quadrature_order=self.order,
        )
        nq, npres = M.shape[0], len(f)
        offsets = hdiv3d_face_offsets(fine, self.family)
        ids = np.concatenate(
            [np.arange(offsets[face], offsets[face + 1]) for face in fine.boundary_faces]
        )
        nb = len(ids)
        mapping = _trace_mapping(self.skeleton, cell, fine, self.family.normal_degree)
        A, B, load = normal_flux_blocks(M, D, f, ids, mapping)
        constant = np.zeros((len(fine.cells), self.family.pressure_size))
        constant[:, 0] = 1
        boundary = np.zeros(nb)
        cursor = 0
        for face in fine.boundary_faces:
            boundary[cursor] = 1
            cursor += face_size(len(fine.faces[face]), self.family.normal_degree)
        kernel = np.r_[np.zeros(nq), constant.ravel(), boundary][:, None]
        moment = np.r_[np.zeros(nq), moment, np.zeros(nb)]
        scale = np.sqrt(float(np.max(M.diagonal())))
        scaling = np.r_[np.full(nq, 1 / scale), np.full(npres + nb, scale)]
        transform = sparse.diags(scaling)
        problem = LocalProblem(
            transform @ A @ transform,
            scaling[:, None] * B,
            scaling * load,
            self.skeleton.cell_dofs(cell),
            kernel=kernel / scaling[:, None],
            constraints=(moment * scaling)[:, None],
        )
        return LocalAssembly(problem, (fine, nq, npres, scaling, moment * scaling))


def solve_darcy_hdiv3d(
    mesh: AffineMixedMesh,
    *,
    pressure_degree: int = 1,
    normal_degree: int = 1,
    trace_degree: int = 1,
    subdivisions: int = 1,
    local_refinement: int = 2,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    quadrature_order: int = 5,
    boundary_quadrature_order: int | None = None,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    global_rtol: float = 1e-10,
    global_refinement_precision: Literal["double", "extended"] = "double",
) -> Mixed3DDarcySolution:
    """Solve Darcy with independent normal order and interior mixed enrichment.

    Pressure degree 1 selects tetrahedral BDFM with 18 modes or prism flux with
    27 modes. Tetrahedral pressure degree 2 retains every zero-normal P3 bubble
    (32 modes) at the default normal degree one. General admissible orders are
    those of HDiv3DFamily. Pressure uses ordinary composition;
    flux uses contravariant Piola on affine cells. With trace_degree=normal_degree and
    subdivisions=local_refinement the trace is the complete classical fine-grid
    conforming space. Neumann values are physical outward normal flux; pressure
    is otherwise imposed weakly. Materials must be resolved by the local mesh
    or the supplied quadrature; coefficient-interface fitting is not implicit.
    """
    family = HDiv3DFamily(mesh.kind, pressure_degree, normal_degree)
    r = positive_int(local_refinement, "local refinement")
    skeleton = Mixed3DSkeleton(mesh, trace_degree, subdivisions)
    if trace_degree > normal_degree:
        raise ValueError("trace degree must not exceed local normal degree")
    if r % subdivisions:
        raise ValueError("trace subdivisions must divide local refinement")
    order = max(positive_int(quadrature_order, "quadrature order"), pressure_degree + 2)
    neumann = {} if neumann is None else neumann
    boundary_order = (
        order
        if boundary_quadrature_order is None
        else positive_int(boundary_quadrature_order, "boundary quadrature order")
    )
    load, fixed = _boundary(skeleton, dirichlet, neumann, boundary_order)
    system = HybridSystem.from_local_factory(
        _LocalFactory(skeleton, family, r, permeability, source, order),
        range(len(mesh.cells)),
        boundary_load=load,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    gauges = None
    if set(neumann) == set(mesh.boundary_faces):
        if not np.isfinite(mean_pressure):
            raise ValueError("mean pressure must be finite")
        gauges = [
            system.mean_constraint(
                [data[4] for data in system.local_metadata],
                float(mean_pressure) * sum(mesh.volumes),
            )
        ]
    hybrid = system.solve(
        solver=solver,
        fixed=fixed,
        constraints=gauges,
        rtol=global_rtol,
        refinement_precision=global_refinement_precision,
    )
    pressure, flux, residuals = [], [], []
    for response, data, field in zip(
        system.responses, system.local_metadata, hybrid.fields, strict=True
    ):
        fine, nq, npres, scale, _ = data
        problem = response.problem
        trace = hybrid.trace[problem.trace_dofs]
        defect = (problem.matrix @ field + problem.coupling @ trace - problem.load) / scale
        action = (
            abs(problem.matrix) @ abs(field)
            + abs(problem.coupling) @ abs(trace)
            + abs(problem.load)
        ) / scale
        residual = [
            float(
                np.linalg.norm(defect[part])
                / max(np.linalg.norm(action[part]), np.finfo(float).tiny)
            )
            for part in (slice(0, nq), slice(nq, nq + npres), slice(nq + npres, None))
        ]
        if max(residual) > 1e-10:
            raise ValueError(
                "physical mixed block residual exceeds the declared backward-error criterion"
            )
        field = scale * field
        flux.append(field[:nq])
        pressure.append(field[nq : nq + npres].reshape(len(fine.cells), -1))
        residuals.append(residual)
    return Mixed3DDarcySolution(
        skeleton,
        family,
        tuple(data[0] for data in system.local_metadata),
        tuple(pressure),
        tuple(flux),
        hybrid,
        source,
        np.array(residuals),
    )
