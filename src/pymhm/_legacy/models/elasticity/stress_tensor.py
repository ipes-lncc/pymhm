"""Weakly symmetric mixed MHM elasticity on Cartesian rectangles.

Stress rows use RT normal degree k and interior order s=k+n. Displacement is
Q_s squared, whereas independent rotation is total-degree P_s. This distinction
is essential to the quadrilateral family in the 2021 weak-symmetry analysis.
"""

from functools import partial
from typing import Any, Literal, cast

import numpy as np
from scipy import sparse

from pymhm._legacy.models.elasticity.boundary import require_compatible_displacement_flux
from pymhm._legacy.models.elasticity.mixed_pressure import _boundary_volume_flux
from pymhm._legacy.models.elasticity.stress import _rigid_values
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.hdiv.tensor_rt import _trace_map
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.quadrilateral import _grid_resolves_material
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.vector.stress_tensor import complete_rotation_basis
from pymhm.fem.vector.stress_tensor import tensor_rigid_moments as _modal_rigid
from pymhm.fem.vector.stress_tensor import tensor_stress_operators as _operators
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import vector_values
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.stress_tensor import (
    TensorRTElasticitySolution as TensorRTElasticitySolution,
)

_rotation_basis = complete_rotation_basis


def _local(
    cell: int,
    *,
    mesh: CartesianMacroMesh,
    skeleton: SkeletonSpace,
    degree: int,
    enrichment: int,
    refinement: int,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    compliance: Any = None,
) -> LocalAssembly:
    """Augment a mixed Neumann problem with boundary displacement and three rigid modes."""
    fine = mesh.submesh(cell, refinement)
    for coefficient in (lame_lambda, lame_mu):
        if isinstance(coefficient, CartesianCellField) and not _grid_resolves_material(
            fine, coefficient
        ):
            raise ValueError("Cartesian Lame material interfaces must align with fine rectangles")
    mass, div, asym, force, trace, bulk_trace, bulk_max = _operators(
        fine, degree, enrichment, lame_lambda, lame_mu, source, order, compliance
    )
    ns, nu, nr = mass.shape[0], div.shape[0], asym.shape[0]
    nb = 2 * (degree + 1) * len(fine.boundary_faces)
    rows = (2 * (degree + 1) * fine.boundary_faces[:, None] + np.arange(2 * (degree + 1))).ravel()
    selector = sparse.coo_matrix((np.ones(nb), (rows, np.arange(nb))), shape=(ns, nb)).tocsc()
    matrix = sparse.bmat(
        [
            [mass, div.T, asym.T, -selector],
            [
                div,
                sparse.csc_matrix((nu, nu)),
                sparse.csc_matrix((nu, nr)),
                sparse.csc_matrix((nu, nb)),
            ],
            [
                asym,
                sparse.csc_matrix((nr, nu)),
                sparse.csc_matrix((nr, nr)),
                sparse.csc_matrix((nr, nb)),
            ],
            [
                -selector.T,
                sparse.csc_matrix((nb, nu)),
                sparse.csc_matrix((nb, nr)),
                sparse.csc_matrix((nb, nb)),
            ],
        ],
        format="csc",
    )
    size = ns + nu + nr + nb
    center = mesh.points[mesh.cells[cell]].mean(axis=0)
    rigid, moments = _modal_rigid(fine, degree + enrichment, center)
    kernel = np.zeros((size, 3))
    kernel[ns : ns + nu] = rigid
    nrot = nr // len(fine.cells)
    kernel[ns + nu : ns + nu + nr : nrot, 2] = -1
    boundary = _rigid_values(fine.points[fine.faces[fine.boundary_faces]], center)
    coefficients = np.zeros((len(boundary), degree + 1, 2, 3))
    coefficients[:, 0], coefficients[:, 1] = (
        boundary.mean(axis=1),
        (boundary[:, 1] - boundary[:, 0]) / 2,
    )
    kernel[-nb:] = coefficients.reshape(-1, 3)
    constraints = np.zeros_like(kernel)
    constraints[ns : ns + nu] = moments
    mapping = np.kron(_trace_map(mesh, cell, fine, skeleton, degree), np.eye(2))
    coupling = np.zeros((size, mapping.shape[1]))
    coupling[-nb:] = -mapping
    problem = LocalProblem(
        matrix,
        coupling,
        np.r_[np.zeros(ns), -force, np.zeros(nr + nb)],
        skeleton.cell_dofs(cell),
        kernel,
        constraints,
    )
    global_center = mesh.areas @ mesh.points[mesh.cells].mean(axis=1) / mesh.areas.sum()
    global_moments = np.zeros_like(kernel)
    global_moments[ns : ns + nu] = _modal_rigid(fine, degree + enrichment, global_center)[1]
    return LocalAssembly(
        problem,
        (
            fine,
            ns,
            nu,
            nr,
            np.pad(trace, (0, size - ns)),
            np.pad(bulk_trace, (0, size - ns)),
            bulk_max,
            global_moments,
        ),
    )


def solve_elasticity_tensor_rt(
    mesh: CartesianMacroMesh,
    *,
    degree: int = 1,
    enrichment: int = 0,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    compliance: Any = None,
    source: Any = (0.0, 0.0),
    dirichlet: Any = (0.0, 0.0),
    traction: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 2,
    quadrature_order: int = 6,
    rigid_moments: Any = (0.0, 0.0, 0.0),
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> TensorRTElasticitySolution:
    """Solve the Cartesian RT_k/RT-enriched weak-symmetry family with physical gauges.

    k>=1 is required. Fine stress normal degree is k; interior RT order s=k+n
    yields displacement Q_s squared and rotation P_s. The skeleton has P1 on
    interior macrofaces and full fine Pk on exterior faces by default. Custom
    normal traces must align with fine edges and have degree at most k.
    Traction means physical outward sigma n; the multiplier is its negative.
    Fully prescribed displacement uses the exact bulk-compliance identity at
    finite lambda and mean(-trace(sigma)/2) at infinite lambda. Pure traction
    prescribes three integrated displacement moments against global rigid modes.
    ``compliance`` optionally supplies a positive full Cartesian anisotropic
    compliance in (xx,xy,yx,yy) coordinates, replacing the Lame material. Its
    explicit skew extension must preserve symmetric tensors; the physical
    boundary identity then uses tr(A sigma).
    Coefficients may be smooth callbacks; material discontinuities must align
    with the fine grid for the ordinary per-rectangle Gaussian rule used here.
    """
    k = positive_int(degree, "stress degree", 1)
    n = positive_int(enrichment, "enrichment", 0)
    s = k + n
    refinement = positive_int(local_refinement, "local_refinement")
    order = positive_int(quadrature_order, "quadrature_order", s + 2)
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    boundary_faces = set(mesh.boundary_faces)
    skeleton = skeleton or SkeletonSpace(
        cast(Any, mesh),
        tuple(
            FaceSpace.uniform(k, refinement) if f in boundary_faces else FaceSpace.uniform(1)
            for f in range(len(mesh.faces))
        ),
        2,
    )
    if skeleton.mesh is not mesh or skeleton.components != 2:
        raise ValueError("mixed elasticity requires a two-component skeleton on this mesh")
    for face in skeleton.faces:
        if max(face.degrees) > k or not np.allclose(
            np.array(face.breaks) * refinement,
            np.round(np.array(face.breaks) * refinement),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("RT trace degrees exceed k or segments do not align with fine edges")
    data = {} if traction is None else traction
    if data and mean_pressure != 0:
        raise ValueError("mean_pressure requires full displacement boundaries")
    negative = {
        face: (lambda points, value=value: -vector_values(value, points))
        for face, value in data.items()
    }
    boundary, fixed = boundary_data(skeleton, dirichlet, negative, order=order)
    factory = partial(
        _local,
        mesh=mesh,
        skeleton=skeleton,
        degree=k,
        enrichment=n,
        refinement=refinement,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        compliance=compliance,
        source=source,
        order=order,
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=-boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    metadata = system.local_metadata
    gauges = []
    if not data:
        scale = max(entry[6] for entry in metadata)
        flux = _boundary_volume_flux(skeleton, boundary)
        if scale == 0:
            require_compatible_displacement_flux(
                flux, _boundary_volume_flux(skeleton, boundary, absolute=True)
            )
            gauges.append(
                system.mean_constraint(
                    [entry[4] for entry in metadata], -2 * mean_pressure * sum(mesh.areas)
                )
            )
        else:
            if mean_pressure != 0:
                raise ValueError("mean_pressure is only a gauge in the incompressible limit")
            gauges.append(
                system.mean_constraint([entry[5] / scale for entry in metadata], flux / scale)
            )
    if set(data) == boundary_faces:
        targets = np.asarray(rigid_moments)
        if targets.shape != (3,) or np.iscomplexobj(targets) or not np.isfinite(targets).all():
            raise ValueError("rigid_moments must contain three finite real moments")
        gauges.extend(
            system.mean_constraint([entry[7][:, i] for entry in metadata], targets[i])
            for i in range(3)
        )
    hybrid = system.solve(solver=solver, fixed=fixed, constraints=gauges)
    stress, displacement, rotation = [], [], []
    for values, entry in zip(hybrid.fields, metadata, strict=True):
        fine, ns, nu, nr = entry[:4]
        stress.append(values[:ns].reshape(-1, 2))
        displacement.append(values[ns : ns + nu].reshape(len(fine.cells), (s + 1) ** 2, 2))
        rotation.append(
            values[ns + nu : ns + nu + nr].reshape(len(fine.cells), (s + 1) * (s + 2) // 2)
        )
    return TensorRTElasticitySolution(
        skeleton,
        tuple(entry[0] for entry in metadata),
        tuple(stress),
        tuple(displacement),
        tuple(rotation),
        hybrid,
        k,
        n,
        source,
        order,
    )
