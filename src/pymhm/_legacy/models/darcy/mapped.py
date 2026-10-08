"""Three-dimensional mixed MHM on trilinear hexahedra with contravariant RT Piola maps.

Flux DOFs are integral tensor-Legendre normal moments. Skeleton unknowns are
reference-face flux densities, not physical polynomials on distorted faces.
Pressure uses the ordinary scalar pullback Q_k. On a nonaffine cell,
``div(RT_k) = Q_k / det(J)``; its physical divergence need not belong to the
scalar pressure space, while all tested pressure moments remain conservative.
"""

from collections.abc import Iterator as Iterator
from dataclasses import dataclass
from itertools import product as product
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.core.contracts import HybridSolution as HybridSolution
from pymhm.core.contracts import LocalAssembly as LocalAssembly
from pymhm.core.contracts import LocalProblem as LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray as FloatArray
from pymhm.core.validation import IntArray as IntArray
from pymhm.core.validation import positive_int as positive_int
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.hdiv.mapped import (
    _modal as _modal,
)
from pymhm.fem.hdiv.mapped import (
    _reference_rt_map as _reference_rt_map,
)
from pymhm.fem.hdiv.mapped import (
    mapped_rt_basis as mapped_rt_basis,
)
from pymhm.fem.hdiv.mapped import (
    mapped_rt_dofs as mapped_rt_dofs,
)
from pymhm.fem.hdiv.mapped_forms import (
    HexSkeleton,
    mapped_boundary_data,
    mapped_rt_operators,
    mapped_rt_trace_mapping,
    quadrature_slices,
)
from pymhm.materials.evaluation import scalar_values_3d as scalar_values_3d
from pymhm.materials.evaluation import tensor_values_3d as tensor_values_3d
from pymhm.materials.evaluation import vector_values_3d as vector_values_3d
from pymhm.meshes.hexahedron import (
    _CORNERS as _CORNERS,
)
from pymhm.meshes.hexahedron import (
    _FACE_CORNERS as _FACE_CORNERS,
)
from pymhm.meshes.hexahedron import (
    _UV as _UV,
)
from pymhm.meshes.hexahedron import (
    HexMesh as HexMesh,
)
from pymhm.meshes.hexahedron import (
    QuadratureOrder as QuadratureOrder,
)
from pymhm.meshes.hexahedron import (
    _face_coordinate_transform as _face_coordinate_transform,
)
from pymhm.meshes.hexahedron import (
    _geometry as _geometry,
)
from pymhm.meshes.hexahedron import (
    _points as _points,
)
from pymhm.meshes.hexahedron import (
    cube_quadrature as cube_quadrature,
)
from pymhm.postprocessing.mapped import MappedRTDarcySolution

_QUADRATURE_BATCH_ENTRIES = 2_000_000


@dataclass(frozen=True)
class _Factory:
    """Build one mixed Neumann local problem with a physical pressure mean constraint."""

    mesh: HexMesh
    skeleton: HexSkeleton
    degree: int
    refinement: int
    permeability: Any
    source: Any
    order: QuadratureOrder

    def __call__(self, cell: int) -> LocalAssembly:
        """Assemble a flux/pressure/boundary-pressure saddle and its joint constant kernel."""
        fine, reference = self.mesh.submesh(cell, self.refinement)
        M, D, f, moment = _operators(fine, self.degree, self.permeability, self.source, self.order)
        nq, npres = M.shape[0], len(f)
        count = (self.degree + 1) ** 2
        nb = count * len(fine.boundary_faces)
        indices = (count * fine.boundary_faces[:, None] + np.arange(count)).ravel()
        selector = sparse.coo_matrix(
            (np.ones(nb), (indices, np.arange(nb))), shape=(nq, nb)
        ).tocsc()
        zero = sparse.csc_matrix((npres, nb))
        A = sparse.bmat(
            [[M, -D.T, selector], [-D, None, zero], [selector.T, zero.T, None]], format="csc"
        )
        mapping = _trace_map(self.mesh, cell, fine, reference, self.skeleton, self.degree)
        B = np.zeros((nq + npres + nb, mapping.shape[1]))
        B[-nb:] = -mapping
        constant = np.zeros((len(fine.cells), (self.degree + 1) ** 3))
        constant[:, 0] = 1
        boundary = np.zeros((len(fine.boundary_faces), count))
        boundary[:, 0] = 1
        kernel = np.r_[np.zeros(nq), constant.ravel(), boundary.ravel()][:, None]
        weights = np.r_[np.zeros(nq), moment, np.zeros(nb)]
        # A symmetric change of physical units balances flux and pressure; it
        # leaves the Schur complement and physical pressure moment unchanged.
        scale = float(np.sqrt(np.max(M.diagonal())))
        scaling = np.r_[np.full(nq, 1 / scale), np.full(npres + nb, scale)]
        transform = sparse.diags(scaling)
        problem = LocalProblem(
            transform @ A @ transform,
            scaling[:, None] * B,
            scaling * np.r_[np.zeros(nq), -f, np.zeros(nb)],
            self.skeleton.cell_dofs(cell),
            kernel=kernel / scaling[:, None],
            constraints=(weights * scaling)[:, None],
        )
        return LocalAssembly(problem, (fine, reference, weights * scaling, nq, npres, scaling))


def solve_darcy_mapped_rt(
    mesh: HexMesh,
    *,
    degree: int = 1,
    trace_degree: int | None = None,
    subdivisions: int = 1,
    local_refinement: int = 2,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    quadrature_order: QuadratureOrder = 5,
    solver: str = "scipy",
    local_solver: str = "scipy",
    global_rtol: float = 1e-10,
    global_refinement_precision: Literal["double", "extended"] = "double",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> MappedRTDarcySolution:
    """Solve a three-dimensional mixed MHM problem using mapped tensor RT_k/Q_k locals.

    Reference flux-density traces have Q_l degree l<=k, with aligned uniform
    subfaces. ``subdivisions=local_refinement`` and l=k yields the classical
    globally conforming fine-mesh mixed operator after local condensation.
    Boundary Neumann values are physical outward flux densities. Discontinuous
    coefficients must be resolved by the geometric cells or supplied quadrature.
    The volume rule may use a tuple of three positive Gauss orders; boundary
    integration then uses their maximum in each face direction. Quadrature is
    accumulated in bounded batches without modifying its points or weights.
    ``global_rtol`` controls the retained saddle's linear solve independently
    of the fixed 1e-10 physical block criterion. Explicit extended refinement
    retains correction digits in global variables and reconstructed fields;
    it requires a wider long-double type and does not guarantee that the
    physical criterion can be reached for every ill-conditioned problem.
    """
    if np.iscomplexobj(global_rtol) or not np.isfinite(global_rtol) or global_rtol <= 0:
        raise ValueError("global_rtol must be finite and positive")
    if global_refinement_precision not in {"double", "extended"}:
        raise ValueError("global_refinement_precision must be double or extended")
    k = positive_int(degree, "RT degree", 0)
    r = positive_int(local_refinement, "local refinement")
    skeleton = HexSkeleton(mesh, k if trace_degree is None else trace_degree, subdivisions)
    if skeleton.degree > k or r % skeleton.subdivisions:
        raise ValueError(
            "trace degree must not exceed RT degree and subdivisions must divide refinement"
        )
    orders = quadrature_order if isinstance(quadrature_order, tuple) else (quadrature_order,) * 3
    if len(orders) != 3:
        raise ValueError("volume quadrature needs exactly three axis orders")
    validated = tuple(max(positive_int(v, "quadrature order"), k + 2) for v in orders)
    order = validated if isinstance(quadrature_order, tuple) else validated[0]
    neumann = {} if neumann is None else neumann
    boundary, fixed = _boundary(skeleton, dirichlet, neumann, max(validated))
    system = HybridSystem.from_local_factory(
        _Factory(mesh, skeleton, k, r, permeability, source, order),
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    gauges = None
    if set(neumann) == set(mesh.boundary_faces):
        mean = float(mean_pressure)
        if not np.isfinite(mean):
            raise ValueError("mean pressure must be finite")
        moments = [data[2] for data in system.local_metadata]
        # Only the first pressure basis integrates to volume on affine cells;
        # geometric moments of higher modes must not be included in that sum.
        volume = sum(
            np.sum(
                _geometry(mesh.points[mesh.cells[i]][None], cube_quadrature(order)[0])[2]
                * cube_quadrature(order)[1]
            )
            for i in range(len(mesh.cells))
        )
        gauges = [system.mean_constraint(moments, mean * volume)]
    hybrid = system.solve(
        solver=solver,
        fixed=fixed,
        constraints=gauges,
        rtol=global_rtol,
        refinement_precision=global_refinement_precision,
    )
    pressure, flux, residuals = [], [], []
    for response, metadata, local_field in zip(
        system.responses, system.local_metadata, hybrid.fields, strict=True
    ):
        fine, _, _, nq, npres, scaling = metadata
        problem = response.problem
        trace = hybrid.trace[problem.trace_dofs]
        defect = (problem.matrix @ local_field + problem.coupling @ trace - problem.load) / scaling
        action = (
            abs(problem.matrix) @ abs(local_field)
            + abs(problem.coupling) @ abs(trace)
            + abs(problem.load)
        ) / scaling
        block_residuals = [
            float(
                np.linalg.norm(defect[part])
                / max(np.linalg.norm(action[part]), np.finfo(float).tiny)
            )
            for part in (slice(0, nq), slice(nq, nq + npres), slice(nq + npres, None))
        ]
        if max(block_residuals) > 1e-10:
            raise ValueError(
                "physical mixed block residual exceeds the declared backward-error criterion"
            )
        residuals.append(block_residuals)
        local_field = scaling * local_field
        flux.append(local_field[:nq])
        pressure.append(local_field[nq : nq + npres].reshape(len(fine.cells), -1))
    return MappedRTDarcySolution(
        skeleton,
        tuple(data[0] for data in system.local_metadata),
        tuple(pressure),
        tuple(flux),
        hybrid,
        k,
        permeability,
        source,
        order,
        np.array(residuals),
    )


_operators = mapped_rt_operators
_quadrature_slices = quadrature_slices
_trace_map = mapped_rt_trace_mapping
_boundary = mapped_boundary_data

_scatter = assemble_element_blocks
