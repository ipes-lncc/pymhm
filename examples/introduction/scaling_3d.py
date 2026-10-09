"""Reproducible three-dimensional CPU/GPU scaling measurements and presentation for Darcy.

The notebook retains the public local/global equation workflow. This module
owns timers, norm integration, archives and figures; full campaigns preserve
the original fine grids, trace spaces and inclusive timing conventions.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import random
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

from examples.introduction.provenance import (
    execution_source_manifest,
)
from pymhm import LocalContext, MeshHierarchy, TraceBinding, bind_problem
from pymhm.core.assembly import SolverConfig
from pymhm.core.contracts import LocalAssembly, LocalProblem, LocalResponse
from pymhm.core.equations import Equation, LocalEquations, compile_form, compile_local_equations
from pymhm.core.multiscale import assemble
from pymhm.core.system import HybridSystem
from pymhm.execution.cpu import ExecutionConfig, map_local
from pymhm.fem.reference import tensor_lagrange_tabulation
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace
from pymhm.meshes.hexahedron import HexMesh

ROOT = case_workspace()
OUTPUT = ROOT / "build/introduction/darcy_3d_workspace_scalability"
OUTPUT.mkdir(parents=True, exist_ok=True)
PUBLISHED = ROOT / "benchmarks/results/execution/introduction-3d-workspace-lu-20261005/results.json"
ACCELERATED_CAMPAIGN = ROOT / "benchmarks/results/execution/introduction-3d-accelerators-20261005"
DEFAULT_ANISOTROPY = np.array([[2.0, 0.3, 0.2], [0.3, 1.5, 0.1], [0.2, 0.1, 1.0]])


@dataclass(frozen=True)
class DarcyData:
    """Declare the medium period and constant symmetric positive anisotropy.

    The scalar medium multiplier is ``exp(prod(sin(2*pi*x_i/period)))``.
    Coordinates have shape ``(...,3)`` and tensor values ``(...,3,3)``.
    The exact pressure is ``prod(sin(pi*x_i))``. Integer domain lengths in x
    retain homogeneous exterior pressure and a fixed physical medium scale.
    """

    period: float = 0.1
    anisotropy: tuple[tuple[float, ...], ...] = tuple(map(tuple, DEFAULT_ANISOTROPY))

    def __post_init__(self) -> None:
        """Reject nonpositive periods and nonsymmetric or nonpositive material tensors."""
        matrix = np.asarray(self.anisotropy)
        if not np.isfinite(self.period) or self.period <= 0:
            raise ValueError("period must be finite and positive")
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or not np.array_equal(matrix, matrix.T)
            or np.linalg.eigvalsh(matrix)[0] <= 0
        ):
            raise ValueError(
                "anisotropy must be a finite symmetric positive definite 3 by 3 tensor"
            )


DEFAULT_DATA = DarcyData()


def medium(points: np.ndarray, data: DarcyData) -> np.ndarray:
    """Evaluate the scalar multiplier, whose bounds are ``exp(-1)`` and ``exp(1)``."""
    return np.exp(np.prod(np.sin((2 * math.pi / data.period) * points), axis=-1))


def permeability(points: np.ndarray, data: DarcyData) -> np.ndarray:
    """Return the full physical diffusion tensor ``m(x)*D`` at every point."""
    return medium(points, data)[..., None, None] * np.asarray(data.anisotropy)


def exact_pressure(points: np.ndarray) -> np.ndarray:
    """Return the scalar exact pressure, with homogeneous data on integer boxes."""
    return np.prod(np.sin(math.pi * points), axis=-1)


def exact_gradient(points: np.ndarray) -> np.ndarray:
    """Differentiate pressure analytically in physical x, y and z coordinates."""
    sine, cosine = np.sin(math.pi * points), np.cos(math.pi * points)
    return math.pi * np.stack(
        [
            cosine[..., i] * np.prod(sine[..., [j for j in range(3) if j != i]], axis=-1)
            for i in range(3)
        ],
        axis=-1,
    )


def exact_hessian(points: np.ndarray) -> np.ndarray:
    """Return all physical pressure second derivatives, including mixed terms."""
    sine, cosine = np.sin(math.pi * points), np.cos(math.pi * points)
    result = np.empty((*points.shape[:-1], 3, 3))
    for i in range(3):
        for j in range(3):
            result[..., i, j] = (
                -(math.pi**2) * np.prod(sine, axis=-1)
                if i == j
                else math.pi**2 * cosine[..., i] * cosine[..., j] * sine[..., 3 - i - j]
            )
    return result


def exact_flux(points: np.ndarray, data: DarcyData) -> np.ndarray:
    """Return Darcy flux ``-K grad(p_exact)``, preserving all three components."""
    return -np.einsum("...ij,...j->...i", permeability(points, data), exact_gradient(points))


def source(points: np.ndarray, data: DarcyData) -> np.ndarray:
    """Apply minus divergence with independent analytic material and pressure derivatives."""
    omega = 2 * math.pi / data.period
    sine, cosine = np.sin(omega * points), np.cos(omega * points)
    multiplier = medium(points, data)
    grad_multiplier = (
        omega
        * multiplier[..., None]
        * np.stack(
            [
                cosine[..., i] * np.prod(sine[..., [j for j in range(3) if j != i]], axis=-1)
                for i in range(3)
            ],
            axis=-1,
        )
    )
    matrix = np.asarray(data.anisotropy)
    return -multiplier * np.einsum("ij,...ij->...", matrix, exact_hessian(points)) - np.einsum(
        "...i,ij,...j->...", grad_multiplier, matrix, exact_gradient(points)
    )


def symbolic_data(domain: Any, omega: Any, anisotropy: Any) -> tuple[Any, Any, Any, Any]:
    """Declare the physical exact fields and source using updateable UFL constants.

    ``omega=2*pi/period`` is spatially constant and ``anisotropy`` is a symmetric
    positive definite tensor. SpatialCoordinate reads the current physical mesh;
    no material interpolation or numerical differentiation is introduced.
    """
    import ufl

    x = ufl.SpatialCoordinate(domain)
    pressure = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1]) * ufl.sin(np.pi * x[2])
    multiplier = ufl.exp(ufl.sin(omega * x[0]) * ufl.sin(omega * x[1]) * ufl.sin(omega * x[2]))
    tensor = multiplier * anisotropy
    flux = -ufl.dot(tensor, ufl.grad(pressure))
    return pressure, tensor, flux, ufl.div(flux)


def ufl_data(domain: Any, data: DarcyData) -> tuple[Any, Any, Any, Any]:
    """Bind physical application data to the explicit symbolic field definitions."""
    import ufl

    return symbolic_data(domain, 2 * np.pi / data.period, ufl.as_matrix(data.anisotropy))


def verify_source(data: DarcyData = DEFAULT_DATA) -> dict[str, float]:
    """Check analytic source independently by centered differences of physical exact flux."""
    points = np.array([[0.237, 0.419, 0.561], [0.613, 0.728, 0.137], [0.832, 0.147, 0.349]])
    step, divergence = 2e-6, np.zeros(len(points))
    for axis in range(3):
        shift = np.eye(3)[axis] * step
        divergence += (
            exact_flux(points + shift, data)[:, axis] - exact_flux(points - shift, data)[:, axis]
        ) / (2 * step)
    forcing = source(points, data)
    np.testing.assert_allclose(forcing, divergence, rtol=3e-8, atol=3e-8)
    return {
        "maximum_source_absolute_difference": float(np.max(np.abs(forcing - divergence))),
        "anisotropy_minimum_eigenvalue": float(np.linalg.eigvalsh(data.anisotropy)[0]),
        "anisotropy_maximum_eigenvalue": float(np.linalg.eigvalsh(data.anisotropy)[-1]),
    }


def macro_mesh(count: int = 4, length: int = 1) -> HexMesh:
    """Make Cartesian macrohexahedra with fixed physical width 1/count in all axes."""
    nx, ny, nz = count * length, count, count
    points = np.array(
        list(
            product(
                np.linspace(0, length, nx + 1), np.linspace(0, 1, ny + 1), np.linspace(0, 1, nz + 1)
            )
        )
    )
    corners = tuple(product((0, 1), repeat=3))
    cells = np.array(
        [
            [(i + a) * (ny + 1) * (nz + 1) + (j + b) * (nz + 1) + k + c for a, b, c in corners]
            for i, j, k in product(range(nx), range(ny), range(nz))
        ]
    )
    return HexMesh(points, cells)


@dataclass
class CellWorkspace:
    """Own one compatible native workspace and proven geometric application data."""

    native: Any
    omega: Any
    anisotropy: Any
    reference_geometry: np.ndarray
    relative_nodes: np.ndarray
    unsigned_coupling: np.ndarray
    physical_moments: np.ndarray
    build_index: int
    assembly_count: int = 0

    def close(self) -> None:
        """Release the native workspace and all retained application arrays."""
        native, self.native = self.native, None
        try:
            if native is not None:
                native.close()
        finally:
            self.omega = self.anisotropy = None
            self.reference_geometry = np.empty((0, 3))
            self.relative_nodes = np.empty((0, 3))
            self.unsigned_coupling = np.empty((0, 0))
            self.physical_moments = np.empty((0, 1))


def create_cell_workspace(
    extent: np.ndarray,
    *,
    refinement: int,
    trace_degree: int,
    quadrature_degree: int,
    data: DarcyData,
    build_index: int,
) -> CellWorkspace:
    """Compile geometry-compatible forms once and tabulate geometric face columns.

    The mesh starts at the origin. Subsequent calls update its geometry to
    the cell's physical position while preserving its topology and ordering.
    Each trace function is tabulated in increasing physical tangent axes.
    Native form assembly returns owned arrays, so later buffer reuse cannot
    mutate previously returned local equations or coefficient archives.
    """
    import dolfinx
    import ufl
    from mpi4py import MPI

    from pymhm.backends.workspace import compile_form_bundle, create_workspace

    domain = dolfinx.mesh.create_box(
        MPI.COMM_SELF,
        [np.zeros(3), extent],
        [refinement] * 3,
        cell_type=dolfinx.mesh.CellType.hexahedron,
    )
    V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
    p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    omega = dolfinx.fem.Constant(domain, np.float64(2 * np.pi / data.period))
    anisotropy = dolfinx.fem.Constant(domain, np.asarray(data.anisotropy, dtype=float))
    _, tensor, _, forcing = symbolic_data(domain, omega, anisotropy)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})
    forms = {
        "operator": ufl.inner(ufl.dot(tensor, ufl.grad(p)), ufl.grad(v)) * dx,
        "load": forcing * v * dx,
        "moment": v * dx,
    }
    facets, markers = [], []
    for side in range(6):
        axis, endpoint = divmod(side, 2)
        value = float(endpoint * extent[axis])
        located = dolfinx.mesh.locate_entities_boundary(
            domain,
            2,
            lambda x, axis=axis, value=value: np.isclose(x[axis], value, rtol=0, atol=1e-12),
        )
        facets.extend(located.tolist())
        markers.extend([side + 1] * len(located))
    ordering = np.argsort(facets)
    tags = dolfinx.mesh.meshtags(
        domain,
        2,
        np.asarray(facets, dtype=np.int32)[ordering],
        np.asarray(markers, dtype=np.int32)[ordering],
    )
    ds = ufl.Measure(
        "ds",
        domain=domain,
        subdomain_data=tags,
        metadata={"quadrature_degree": 2 * trace_degree + 2},
    )
    T = V if trace_degree == 1 else dolfinx.fem.functionspace(domain, ("Lagrange", trace_degree))
    trace_lift = dolfinx.fem.Function(T)
    trace_points = T.tabulate_dof_coordinates()[:, :3]
    reference_nodes = np.array(list(product(np.linspace(0, 1, trace_degree + 1), repeat=2)))
    width = len(reference_nodes)
    for side in range(6):
        forms[f"face{side}"] = trace_lift * v * ds(side + 1)
    bundle = compile_form_bundle(forms, domain.comm)
    native = create_workspace(bundle, domain)
    try:
        nodes = V.tabulate_dof_coordinates()[:, :3].copy()
        unsigned = np.empty((len(nodes), 6 * width))
        for side in range(6):
            tangents = [axis for axis in range(3) if axis != side // 2]
            uv = np.clip(trace_points[:, tangents] / extent[tangents], 0, 1)
            values = tensor_lagrange_tabulation(
                "quadrilateral", trace_degree, uv, nodes=reference_nodes, nderiv=0
            )[0]
            for index in range(width):
                trace_lift.x.array[:] = values[:, index]
                unsigned[:, side * width + index] = native.assemble(f"face{side}")
        moments = native.assemble("moment")[:, None]
        return CellWorkspace(
            native,
            omega,
            anisotropy,
            domain.geometry.x.copy(),
            nodes,
            unsigned,
            moments,
            build_index,
        )
    except BaseException:
        native.close()
        raise


def assemble_cell_equations(
    record: CellWorkspace,
    local: LocalContext,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    refinement: int,
    trace_degree: int,
    data: DarcyData,
    builds: int,
    started: float,
    key: tuple[Any, ...],
) -> LocalEquations:
    """Remount A/f for current material/position and declare explicit trace orientation.

    The returned matrices and metadata own their data independently of reusable
    native buffers. This application reuses B and physical moments because their
    forms contain only the fixed cell geometry and trace basis, never K or f.
    """
    from pymhm.backends.workspace import update_workspace

    geometry = record.reference_geometry + lower
    update_workspace(
        record.native,
        geometry=geometry,
        constants={
            record.omega: np.float64(2 * np.pi / data.period),
            record.anisotropy: np.asarray(data.anisotropy, dtype=float),
        },
    )
    A = record.native.assemble("operator")
    f = record.native.assemble("load")
    record.assembly_count += 1
    B = record.unsigned_coupling
    coordinates = record.relative_nodes + lower
    moments = record.physical_moments.copy()
    return local.equations(
        a=A,
        L=f,
        b=B,
        c=-B.T,
        kernel=np.ones((len(coordinates), 1)),
        moments=moments,
        metadata={
            "coordinates": coordinates,
            "bounds": np.vstack((lower, upper)),
            "refinement": refinement,
            "fine_cells": refinement**3,
            "physical_moment": moments[:, 0],
            "local_assembly_seconds": time.perf_counter() - started,
            "workspace_builds_in_thread": builds,
            "workspace_instance_assembly_count": record.assembly_count,
            "workspace_process": os.getpid(),
            "workspace_thread": threading.get_ident(),
            "workspace_key": repr(key),
            "material_period": data.period,
        },
    )


@dataclass(frozen=True)
class TensorFaceInterface:
    """Declare Cartesian tensor-face coordinates through the custom-space contract.

    The native workspace integrates unsigned face shapes. This adapter alone
    declares their canonical/outward-normal coefficient transport. Mathematical
    signs in the local and global equations remain explicit in their forms.
    """

    mesh: HexMesh
    degree: int

    @property
    def size(self) -> int:
        """Count the independent Qk coefficients on every global macroface."""
        return len(self.mesh.faces) * (self.degree + 1) ** 2

    def binding(self, cell: int) -> TraceBinding:
        """Declare numbering, basis identity and both geometric coefficient maps."""
        width = (self.degree + 1) ** 2
        dofs = np.concatenate(
            [face * width + np.arange(width) for face in self.mesh.cell_faces[cell]]
        )
        outward = np.diag(np.repeat(self.mesh.signs[cell], width))
        return TraceBinding(
            dofs,
            outward,
            basis_id=f"Cartesian face Q{self.degree}; increasing physical tangent axes",
            require_injective=True,
        )


@dataclass
class LocalProvider:
    """Assemble independent Q1 Neumann cells through isolated native workspaces.

    A cache is created lazily in each calling thread of each spawned worker.
    Pickle transfers only application data, never native FEM/PETSc resources.
    The geometry-only face and volume-moment blocks are independent of the
    material in this application; the backend does not infer that property.
    All material matrices and loads are assembled again for every cell.
    """

    macro: HexMesh
    refinement: int
    data: DarcyData = DEFAULT_DATA
    trace_degree: int = 1
    quadrature_degree: int = 8
    workspace_capacity: int = 4
    cell_data: dict[int, DarcyData] | None = None
    _thread_state: Any = dataclass_field(init=False, repr=False)
    _registry_lock: Any = dataclass_field(init=False, repr=False)
    _caches: list[Any] = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate structural counts and initialize process-local Python state."""
        for name in ("refinement", "trace_degree", "quadrature_degree", "workspace_capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._reset_native_state()

    def _reset_native_state(self) -> None:
        """Initialize empty thread-isolated caches without importing a native backend."""
        self._thread_state = threading.local()
        self._registry_lock = threading.Lock()
        self._caches = []

    def __getstate__(self) -> dict[str, Any]:
        """Serialize application declarations and exclude every native instance."""
        return {
            name: getattr(self, name)
            for name in (
                "macro",
                "refinement",
                "data",
                "trace_degree",
                "quadrature_degree",
                "workspace_capacity",
                "cell_data",
            )
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Create empty native caches after a cross-platform spawn transfer."""
        for name, value in state.items():
            setattr(self, name, value)
        self.__post_init__()

    def prepare_runtime(self) -> None:
        """Preload the selected FEM libraries before the worker's native thread limits."""
        from importlib import import_module

        for module in ("dolfinx", "ufl", "mpi4py.MPI"):
            import_module(module)

    def close(self) -> None:
        """Close all worker/thread caches after running jobs have joined."""
        with self._registry_lock:
            caches, self._caches = self._caches, []
        failure = None
        for cache in caches:
            try:
                cache.close()
            except BaseException as error:
                if failure is None:
                    failure = error
        self._thread_state = threading.local()
        if failure is not None:
            raise failure

    def _thread_cache(self) -> Any:
        """Return the bounded cache belonging exclusively to the calling thread."""
        from pymhm.backends.workspace import WorkspaceCache

        if not hasattr(self._thread_state, "cache"):
            self._thread_state.cache = WorkspaceCache(capacity=self.workspace_capacity)
            self._thread_state.builds = 0
            with self._registry_lock:
                self._caches.append(self._thread_state.cache)
        return self._thread_state.cache

    def _build(self, extent: np.ndarray) -> CellWorkspace:
        """Delegate compatible geometry/form construction to its free function."""
        index = self._thread_state.builds + 1
        record = create_cell_workspace(
            extent,
            refinement=self.refinement,
            trace_degree=self.trace_degree,
            quadrature_degree=self.quadrature_degree,
            data=self.data,
            build_index=index,
        )
        self._thread_state.builds = index
        return record

    def local_mesh(self, cell: int) -> HexMesh:
        """Describe the matching portable fine mesh without allocating native objects."""
        return self.macro.submesh(cell, self.refinement)[0]

    def __call__(self, local: LocalContext) -> LocalEquations:
        """Assemble this cell's A/f and declare oriented coupling and constant kernel."""
        started = time.perf_counter()
        cell = local.cell
        corners = self.macro.points[self.macro.cells[cell]]
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        extent = upper - lower
        if np.any(extent <= 0):
            raise ValueError("the Cartesian application's cell extents must be positive")
        expected = lower + np.array(list(product((0, 1), repeat=3))) * extent
        geometry_tolerance = 32 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(corners))))
        if not np.allclose(corners, expected, rtol=0, atol=geometry_tolerance):
            raise ValueError(
                "this application requires Cartesian boxes in tensor-product vertex order"
            )
        key = (
            "hexahedron",
            self.refinement,
            1,
            self.trace_degree,
            self.quadrature_degree,
            tuple(extent),
            np.dtype(float).str,
        )
        cache = self._thread_cache()
        record = cache.get(key, lambda: self._build(extent))
        current_data = self.data if self.cell_data is None else self.cell_data.get(cell, self.data)
        return assemble_cell_equations(
            record,
            local,
            lower,
            upper,
            refinement=self.refinement,
            trace_degree=self.trace_degree,
            data=current_data,
            builds=self._thread_state.builds,
            started=started,
            key=key,
        )


def audit_local(provider: LocalProvider, cell: int = 0) -> dict[str, float]:
    """Check local identities and close the audit provider on success or failure."""
    try:
        problem = bind_problem(
            MeshHierarchy(provider.macro, provider.local_mesh),
            TensorFaceInterface(provider.macro, provider.trace_degree),
            provider,
            global_equation=Equation(0, 0),
            retained=1,
        )
        equations = problem.local_provider(cell)
        matrix, coupling = equations.a, equations.b
        ones = np.ones(matrix.shape[0])
        lower, upper = equations.metadata["bounds"]
        lengths = upper - lower
        width = (provider.trace_degree + 1) ** 2
        kernel_action = float(np.linalg.norm(matrix @ ones))
        matrix_scale = float(np.max(np.asarray(abs(matrix).sum(axis=1))))
        assert kernel_action <= 1e-11 * matrix_scale * np.sqrt(len(ones))
        moment_volume = float(np.sum(equations.moments))
        np.testing.assert_allclose(moment_volume, np.prod(lengths), rtol=1e-12, atol=1e-14)
        integrals = ones @ coupling
        for side in range(6):
            area = float(np.prod(lengths[np.arange(3) != side // 2]))
            np.testing.assert_allclose(
                np.sum(integrals[side * width : (side + 1) * width]),
                provider.macro.signs[cell, side] * area,
                rtol=1e-12,
                atol=1e-14,
            )
        return {
            "constant_kernel_action_norm": kernel_action,
            "physical_volume": moment_volume,
            "trace_column_rank": int(np.linalg.matrix_rank(coupling)),
        }
    finally:
        provider.close()


def run_mhm(
    fine_n: int,
    *,
    workers: int = 1,
    backend: str = "process",
    solver: str = "scipy",
    data: DarcyData = DEFAULT_DATA,
    length: int = 1,
    trace_degree: int = 1,
    macro_count: int = 4,
    quadrature_degree: int = 8,
) -> tuple[dict[str, Any], Any, Any, HexMesh]:
    """Time a fresh workspace provider, all independent solves and global reconstruction."""
    if fine_n % macro_count:
        raise ValueError("fine_n must be divisible by macro_count")
    with threadpool_limits(1):
        started = time.perf_counter()
        macro = macro_mesh(macro_count, length)
        provider_type = LocalProvider if backend == "serial" else SpawnLocalProvider
        provider_data = (
            data
            if backend == "serial"
            else worker_module.DarcyData(period=data.period, anisotropy=data.anisotropy)
        )
        provider = provider_type(
            macro,
            fine_n // macro_count,
            data=provider_data,
            trace_degree=trace_degree,
            quadrature_degree=quadrature_degree,
        )
        interface_type = (
            TensorFaceInterface if backend == "serial" else worker_module.TensorFaceInterface
        )
        problem = bind_problem(
            MeshHierarchy(macro, provider.local_mesh),
            interface_type(macro, trace_degree),
            provider,
            global_equation=Equation(0, 0),
            retained=1,
        )
        prepared = time.perf_counter()
        system = assemble(
            problem,
            execution=ExecutionConfig(
                backend, workers, native_threads=1, batch_size=workers, pipeline=True
            ),
            solvers=SolverConfig(local_solver=solver, global_solver="scipy"),
        )
        assembled = time.perf_counter()
        solution = system.solve()
        finished = time.perf_counter()
        builds: dict[tuple[int, int], int] = {}
        for metadata in system.local_metadata:
            identity = (metadata["workspace_process"], metadata["workspace_thread"])
            builds[identity] = max(builds.get(identity, 0), metadata["workspace_builds_in_thread"])
        record = {
            "method": "MHM native workspace",
            "fine_n": fine_n,
            "length": length,
            "workers": workers,
            "backend": backend,
            "local_solver": solver,
            "trace_degree": trace_degree,
            "material_period": data.period,
            "setup": prepared - started,
            "local_and_global_assembly": assembled - prepared,
            "global_solve_reconstruction": finished - assembled,
            "total": finished - started,
            "fine_cells": fine_n**3 * length,
            "macro_cells": len(macro.cells),
            "local_nodal_dofs": (fine_n // macro_count + 1) ** 3,
            "trace_dofs": problem.trace_size,
            "global_dofs": system.matrix.shape[0],
            "global_relative_residual": float(solution.residual),
            "local_assembly_seconds_sum": sum(
                m["local_assembly_seconds"] for m in system.local_metadata
            ),
            "assembly_quadrature_degree": quadrature_degree,
            "native_workspace_builds": sum(builds.values()),
            "workspace_worker_thread_counts": [
                {"process": key[0], "thread": key[1], "builds": value}
                for key, value in sorted(builds.items())
            ],
            "material_operator_assemblies": len(system.local_metadata),
            "material_load_assemblies": len(system.local_metadata),
            "reuse_scope": (
                "mesh/space/native forms and geometry-only B/C; no numerical "
                "factor or response reuse"
            ),
        }
    return record, system, solution, macro


@dataclass
class NativeState:
    """Keep native functions in their original executed coordinates for untimed norms."""

    domain: Any
    space: Any
    pressure: Any
    data: DarcyData
    grid: tuple[int, int, int]
    length: int
    gathered_coefficients: np.ndarray | None = None


def solve_classical(
    fine_n: int,
    *,
    data: DarcyData = DEFAULT_DATA,
    solver: str = "pyamg",
    length: int = 1,
    quadrature_degree: int = 8,
    native_threads: int = 1,
) -> tuple[dict[str, Any], NativeState]:
    """Time fresh native Q1 geometry, UFL assembly, boundary elimination and solve.

    Imports and JIT warmup belong to a separate explicitly recorded warmup.
    Native assembly, sparse-copy transfer and fresh AMG setup remain in every
    measured run. No factorization or AMG hierarchy is reused between runs.
    The physical relative residual is checked on the original reduced matrix.
    FEM assembly uses one native thread in this single-process reference.
    native_threads selects the solver library budget, including MKL for PARDISO.
    """
    import dolfinx
    import ufl
    from mpi4py import MPI

    from pymhm.linalg.linear import preload_solver_backend, solve_linear

    if fine_n < 2 or isinstance(fine_n, bool) or length < 1 or isinstance(length, bool):
        raise ValueError(
            "fine_n and length must be positive integer counts, with fine_n at least two"
        )
    if (
        isinstance(native_threads, bool)
        or not isinstance(native_threads, int)
        or native_threads < 1
    ):
        raise ValueError("native_threads must be a positive integer solver budget")
    # Preload only the selected library; numerical factors remain fresh.
    preload_solver_backend(solver)
    grid = fine_n * length, fine_n, fine_n
    with threadpool_limits(1):
        started = time.perf_counter()
        domain = dolfinx.mesh.create_box(
            MPI.COMM_SELF,
            [np.zeros(3), np.array([float(length), 1.0, 1.0])],
            grid,
            cell_type=dolfinx.mesh.CellType.hexahedron,
        )
        V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
        p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        _, tensor, _, forcing = ufl_data(domain, data)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})
        a = ufl.inner(ufl.dot(tensor, ufl.grad(p)), ufl.grad(v)) * dx
        L = forcing * v * dx
        coordinates = V.tabulate_dof_coordinates()[:, :3]
        upper = np.array([length, 1, 1])
        boundary = np.any(
            np.isclose(coordinates, 0, rtol=0, atol=1e-13)
            | np.isclose(coordinates, upper, rtol=0, atol=1e-13),
            axis=1,
        )
        free = np.flatnonzero(~boundary)
        prepared = time.perf_counter()
        matrix, load = compile_form(a), compile_form(L)
        reduced, rhs = matrix[free][:, free], load[free]
        assembled = time.perf_counter()
        coefficients = np.zeros(len(coordinates))
        with threadpool_limits(native_threads):
            coefficients[free] = solve_linear(
                reduced,
                rhs,
                solver=solver,
                rtol=1e-10,
                atol=0,
                maxiter=500 if solver == "pyamg" else None,
                near_nullspace=np.ones((len(free), 1)) if solver == "pyamg" else None,
                refinement_precision="double",
                refinement_steps=2,
                equilibration="none",
            )
            solver_native_pools = threadpool_info()
        pressure = dolfinx.fem.Function(V)
        pressure.x.array[:] = coefficients
        completed = time.perf_counter()
        residual = float(np.linalg.norm(reduced @ coefficients[free] - rhs) / np.linalg.norm(rhs))
        assert residual <= 1e-10
        ncells = domain.topology.index_map(3).size_local
        assert ncells == math.prod(grid) and len(coefficients) == math.prod(n + 1 for n in grid)
        nnz = int(reduced.nnz)
    return (
        {
            "setup": prepared - started,
            "assembly_transfer_elimination": assembled - prepared,
            "solve_reconstruct": completed - assembled,
            "total": completed - started,
            "equation_relative_L2_residual": residual,
            "fine_grid": list(grid),
            "fine_cells": ncells,
            "global_algebraic_size": len(free),
            "matrix_nnz": nnz,
            "solver": solver,
            "assembly_quadrature_degree": quadrature_degree,
            "assembly_native_threads": 1,
            "solver_native_threads_requested": native_threads,
            "solver_native_pools": solver_native_pools,
            "timing_scope": (
                "fresh native setup, assembly/copy/BC elimination, solver set"
                "up/solve, complete field reconstruction"
            ),
        },
        NativeState(domain, V, pressure, data, grid, length),
    )


def solve_classical_mpi(
    fine_n: int,
    *,
    comm: Any,
    data: DarcyData = DEFAULT_DATA,
    length: int = 1,
    quadrature_degree: int = 8,
) -> tuple[dict[str, Any], NativeState]:
    """Build and solve independent conforming Q1 with distributed PETSc CG/GAMG.

    Every rank uses one native thread. Collective barriers and rank maxima
    account for synchronization, mesh partitioning, assembly, a fresh AMG
    hierarchy and complete distributed field reconstruction. The constant
    near-nullspace is an AMG candidate, not a pressure gauge or exact nullspace.
    All manually created PETSc resources are destroyed before returning.
    """
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    grid = fine_n * length, fine_n, fine_n
    matrix = load = ksp = candidate = defect = None
    with threadpool_limits(1):
        comm.Barrier()
        started = time.perf_counter()
        domain = dolfinx.mesh.create_box(
            comm,
            [np.zeros(3), np.array([float(length), 1.0, 1.0])],
            grid,
            cell_type=dolfinx.mesh.CellType.hexahedron,
        )
        V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
        p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        _, tensor, _, forcing = ufl_data(domain, data)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})
        a = dolfinx.fem.form(ufl.inner(ufl.dot(tensor, ufl.grad(p)), ufl.grad(v)) * dx)
        L = dolfinx.fem.form(forcing * v * dx)
        domain.topology.create_connectivity(2, 3)
        facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
        bc = dolfinx.fem.dirichletbc(PETSc.ScalarType(0), dofs, V)
        pressure = dolfinx.fem.Function(V)
        comm.Barrier()
        prepared = time.perf_counter()
        try:
            matrix = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
            matrix.assemble()
            load = dolfinx.fem.petsc.assemble_vector(L)
            dolfinx.fem.petsc.apply_lifting(load, [a], bcs=[[bc]])
            load.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(load, [bc])
            candidate = PETSc.NullSpace().create(constant=True, comm=comm)
            matrix.setNearNullSpace(candidate)
            comm.Barrier()
            assembled = time.perf_counter()
            ksp = PETSc.KSP().create(comm)
            ksp.setOperators(matrix)
            ksp.setType("cg")
            ksp.getPC().setType("gamg")
            ksp.setTolerances(rtol=1e-10, atol=0, max_it=500)
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            ksp.setUp()
            comm.Barrier()
            hierarchy_prepared = time.perf_counter()
            ksp.solve(load, pressure.x.petsc_vec)
            pressure.x.scatter_forward()
            owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
            coordinates = V.tabulate_dof_coordinates()[:owned, :3]
            lattice = np.rint(coordinates * fine_n).astype(np.int64)
            addresses = np.ravel_multi_index(lattice.T, tuple(n + 1 for n in grid))
            parts = comm.gather((addresses, pressure.x.array[:owned].copy()), root=0)
            canonical = None
            if comm.rank == 0:
                canonical = np.empty(math.prod(n + 1 for n in grid))
                indices = np.concatenate([part[0] for part in parts])
                assert len(indices) == len(canonical) and len(np.unique(indices)) == len(indices)
                canonical[indices] = np.concatenate([part[1] for part in parts])
            comm.Barrier()
            completed = time.perf_counter()
            defect = load.duplicate()
            matrix.mult(pressure.x.petsc_vec, defect)
            defect.axpy(-1, load)
            residual = float(defect.norm() / load.norm())
            reason, iterations = int(ksp.getConvergedReason()), int(ksp.getIterationNumber())
            assert reason > 0 and residual <= 1e-10, (reason, iterations, residual)
            ncells = int(comm.allreduce(domain.topology.index_map(3).size_local, op=MPI.SUM))
            ndofs = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)
            assert ncells == math.prod(grid) and ndofs == math.prod(n + 1 for n in grid)
        finally:
            for resource in (defect, ksp, candidate, load, matrix):
                if resource is not None:
                    resource.destroy()
        elapsed = {
            "setup": prepared - started,
            "assembly_transfer_elimination": assembled - prepared,
            "hierarchy_setup": hierarchy_prepared - assembled,
            "solve_gather_reconstruct": completed - hierarchy_prepared,
            "solve_reconstruct": completed - assembled,
            "total": completed - started,
        }
        elapsed = {
            name: float(comm.allreduce(value, op=MPI.MAX)) for name, value in elapsed.items()
        }
    return (
        {
            **elapsed,
            "mpi_ranks": int(comm.size),
            "native_threads_per_rank": 1,
            "fine_grid": list(grid),
            "fine_cells": ncells,
            "global_algebraic_size": ndofs,
            "solver": "PETSc CG/GAMG",
            "iterations": iterations,
            "convergence_reason": reason,
            "equation_relative_L2_residual": residual,
            "assembly_quadrature_degree": quadrature_degree,
            "near_nullspace": "one constant candidate; no global pressure gauge",
            "timing_scope": (
                "fresh distributed mesh/setup, synchronized assembly/BC elimi"
                "nation, fresh GAMG hierarchy/solve, ghost synchronization, c"
                "anonical physical-node gather and complete reconstruction"
            ),
        },
        NativeState(domain, V, pressure, data, grid, length, canonical),
    )


def peak_rss_bytes() -> int:
    """Return process lifetime high-water RSS; Linux reports KiB, macOS bytes."""
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else 1024 * value


def solve_mpi_lu(
    fine_n: int, *, comm: Any, data: DarcyData = DEFAULT_DATA, quadrature_degree: int = 8
) -> tuple[dict[str, Any], NativeState]:
    """Assemble unchanged global Q1 forms and solve with distributed pivoted MUMPS LU."""
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    if not PETSc.Sys.hasExternalPackage("mumps"):
        raise RuntimeError(
            "The pinned PETSc installation must provide MUMPS; no backend substitution"
        )
    matrix = load = ksp = defect = None
    grid = (fine_n,) * 3
    with threadpool_limits(1):
        comm.Barrier()
        started = time.perf_counter()
        domain = dolfinx.mesh.create_box(
            comm, [np.zeros(3), np.ones(3)], grid, cell_type=dolfinx.mesh.CellType.hexahedron
        )
        space = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
        pressure_trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        _, tensor, _, forcing = ufl_data(domain, data)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})
        a = dolfinx.fem.form(
            ufl.inner(ufl.dot(tensor, ufl.grad(pressure_trial)), ufl.grad(test)) * dx
        )
        L = dolfinx.fem.form(forcing * test * dx)
        domain.topology.create_connectivity(2, 3)
        facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
        boundary_dofs = dolfinx.fem.locate_dofs_topological(space, 2, facets)
        bc = dolfinx.fem.dirichletbc(PETSc.ScalarType(0), boundary_dofs, space)
        pressure = dolfinx.fem.Function(space)
        comm.Barrier()
        prepared = time.perf_counter()
        try:
            matrix = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
            matrix.assemble()
            load = dolfinx.fem.petsc.assemble_vector(L)
            dolfinx.fem.petsc.apply_lifting(load, [a], bcs=[[bc]])
            load.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(load, [bc])
            comm.Barrier()
            assembled = time.perf_counter()
            ksp = PETSc.KSP().create(comm)
            ksp.setOperators(matrix)
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("lu")
            pc.setFactorSolverType("mumps")
            ksp.setUp()
            comm.Barrier()
            factored = time.perf_counter()
            ksp.solve(load, pressure.x.petsc_vec)
            pressure.x.scatter_forward()
            owned = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
            coordinates = space.tabulate_dof_coordinates()[:owned, :3]
            lattice = np.rint(coordinates * fine_n).astype(np.int64)
            addresses = np.ravel_multi_index(lattice.T, tuple(n + 1 for n in grid))
            parts = comm.gather((addresses, pressure.x.array[:owned].copy()), root=0)
            canonical = None
            if comm.rank == 0:
                indices = np.concatenate([part[0] for part in parts])
                canonical = np.empty(math.prod(n + 1 for n in grid))
                assert len(indices) == len(canonical) and len(np.unique(indices)) == len(indices)
                canonical[indices] = np.concatenate([part[1] for part in parts])
            comm.Barrier()
            completed = time.perf_counter()
            peak_by_rank = comm.gather(peak_rss_bytes(), root=0)
            max_peak = int(comm.allreduce(peak_rss_bytes(), op=MPI.MAX))
            sum_peak = int(comm.allreduce(peak_rss_bytes(), op=MPI.SUM))
            defect = load.duplicate()
            matrix.mult(pressure.x.petsc_vec, defect)
            defect.axpy(-1, load)
            residual = float(defect.norm() / load.norm())
            reason = int(ksp.getConvergedReason())
            assert reason > 0 and residual <= 1e-10, (reason, residual)
            ncells = int(comm.allreduce(domain.topology.index_map(3).size_local, op=MPI.SUM))
            ndofs = int(space.dofmap.index_map.size_global * space.dofmap.index_map_bs)
            assert ncells == math.prod(grid) and ndofs == math.prod(n + 1 for n in grid)
            factor = pc.getFactorMatrix()
            diagnostics = {
                "mumps_infog": {
                    str(index): int(factor.getMumpsInfog(index))
                    for index in (16, 17, 18, 19, 21, 22)
                },
                "mumps_icntl": {
                    str(index): int(factor.getMumpsIcntl(index))
                    for index in (7, 14, 22, 23, 24, 28, 29, 35)
                },
                "mumps_memory_unit": (
                    "MB; 16/17 estimated max/sum, 18/19 allocated max/sum, 21/22 used max/sum"
                ),
            }
        finally:
            for native in (defect, ksp, load, matrix):
                if native is not None:
                    native.destroy()
        elapsed = {
            "setup": prepared - started,
            "assembly_transfer_elimination": assembled - prepared,
            "factor_setup": factored - assembled,
            "solve_gather_reconstruct": completed - factored,
            "total": completed - started,
        }
        elapsed = {
            name: float(comm.allreduce(value, op=MPI.MAX)) for name, value in elapsed.items()
        }
    return (
        {
            **elapsed,
            **diagnostics,
            "solver": "PETSc PREONLY/LU/MUMPS",
            "mpi_ranks": comm.size,
            "native_threads_per_rank": 1,
            "fine_grid": list(grid),
            "fine_cells": ncells,
            "global_algebraic_size": ndofs,
            "assembly_quadrature_degree": quadrature_degree,
            "free_pressure_dofs": (fine_n - 1) ** 3,
            "stored_pressure_dofs": ndofs,
            "factor_pressure_rows": ndofs,
            "equation_relative_L2_residual": residual,
            "convergence_reason": reason,
            "peak_rss_bytes_by_rank": peak_by_rank,
            "maximum_rank_peak_rss_bytes": max_peak,
            "sum_rank_peak_rss_bytes": sum_peak,
            "memory_scope": (
                "Process lifetime high-water RSS sampled before physical norm"
                "s; sum of individual maxima need not occur simultaneously"
            ),
            "pressure_gauge": (
                "none; full strong Dirichlet pressure determines its nonzero physical mean"
            ),
            "matrix_reuse": False,
            "factor_reuse_across_acquisitions": False,
            "timing_scope": (
                "fresh native geometry/form/setup, synchronized assembly/stro"
                "ng boundary handling, complete MUMPS analysis/factorization,"
                " triangular solve, ghost synchronization, canonical physical"
                "-node gather and reconstruction"
            ),
        },
        NativeState(domain, space, pressure, data, grid, 1, canonical),
    )


def classical_physical_errors(
    state: NativeState, *, quadrature_degree: int = 10
) -> dict[str, float]:
    """Integrate physical p/q errors against exact fields with independent native UFL.

    Weak-scaling errors are normalized by sqrt(volume); dimensional absolute
    errors and exact field norms are retained separately. Physical flux is the
    broken conforming gradient field, without any H(div) reconstruction claim.
    This operation is scientific validation and is outside performance timing.
    """
    import dolfinx
    import ufl
    from mpi4py import MPI

    exact, tensor, flux, _ = ufl_data(state.domain, state.data)
    numerical_flux = -ufl.dot(tensor, ufl.grad(state.pressure))
    dx = ufl.Measure("dx", domain=state.domain, metadata={"quadrature_degree": quadrature_degree})
    dp, dq = state.pressure - exact, numerical_flux - flux
    integrands = {
        "pressure": dp**2,
        "flux": ufl.inner(dq, dq),
        "exact_pressure": exact**2,
        "exact_flux": ufl.inner(flux, flux),
    }
    values = {
        name: math.sqrt(
            max(
                0.0,
                float(
                    state.domain.comm.allreduce(
                        dolfinx.fem.assemble_scalar(dolfinx.fem.form(value * dx)), op=MPI.SUM
                    )
                ),
            )
        )
        for name, value in integrands.items()
    }
    pressure_integral = float(
        state.domain.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(state.pressure * dx)), op=MPI.SUM
        )
    )
    exact_integral = (1 - math.cos(math.pi * state.length)) * 4 / math.pi**3
    volume = float(state.length)
    return {
        "pressure_L2": values["pressure"],
        "flux_L2": values["flux"],
        "pressure_L2_per_sqrt_volume": values["pressure"] / math.sqrt(volume),
        "flux_L2_per_sqrt_volume": values["flux"] / math.sqrt(volume),
        "pressure_relative_L2": values["pressure"] / values["exact_pressure"],
        "flux_relative_L2": values["flux"] / values["exact_flux"],
        "exact_pressure_L2": values["exact_pressure"],
        "exact_flux_L2": values["exact_flux"],
        "pressure_integral": pressure_integral,
        "exact_pressure_integral": exact_integral,
        "error_quadrature_degree": quadrature_degree,
    }


def canonical_coefficients(state: NativeState) -> tuple[np.ndarray, np.ndarray]:
    """Export literal native Q1 coefficients into x/y/z lexicographic grid order.

    The transformation is only a coordinate permutation. No interpolation,
    fitted basis, numerical nullspace or smoothing enters this data contract.
    """
    if state.domain.comm.size != 1:
        raise ValueError("canonical grid export requires a serial native state")
    coordinates = state.space.tabulate_dof_coordinates()[:, :3]
    spacing = 1 / state.grid[1]
    indices = np.rint(coordinates / spacing).astype(np.int64)
    shape = tuple(n + 1 for n in state.grid)
    order = np.ravel_multi_index(indices.T, shape)
    assert len(np.unique(order)) == math.prod(shape)
    points, coefficients = np.empty_like(coordinates), np.empty(len(coordinates))
    points[order], coefficients[order] = coordinates, state.pressure.x.array
    return points, coefficients


def mhm_physical_errors(
    system: Any, solution: Any, *, data: DarcyData = DEFAULT_DATA, order: int = 5
) -> dict[str, float]:
    """Integrate actual broken pressure/vector-flux errors with explicit material data.

    Native node order is recovered only through integer lattice addresses.
    Independent local fields are integrated without smoothing their interfaces.
    Absolute errors and volume-normalized errors are kept as separate values.
    """
    from pymhm.meshes.hexahedron import cube_quadrature

    reference, weights = cube_quadrature(order)
    corners = np.array(list(product((0, 1), repeat=3)))
    phi, derivative = tensor_lagrange_tabulation(
        "hexahedron", 1, reference, nodes=corners, nderiv=1
    )
    pressure_square, flux_square, volume = 0.0, 0.0, 0.0
    for metadata, field_values in zip(system.local_metadata, solution.fields, strict=True):
        nodes = metadata["coordinates"]
        r = metadata["refinement"]
        lower, upper = metadata["bounds"]
        spacing = (upper - lower) / r
        lattice = np.rint((nodes - lower) / spacing).astype(np.int64)
        addresses = np.ravel_multi_index(lattice.T, (r + 1,) * 3)
        assert len(np.unique(addresses)) == (r + 1) ** 3
        permutation = np.argsort(addresses)
        values = np.asarray(field_values)[permutation]
        cells = np.array(
            [
                [(i + a) * (r + 1) ** 2 + (j + b) * (r + 1) + k + c for a, b, c in corners]
                for i, j, k in product(range(r), repeat=3)
            ]
        )
        for begin in range(0, len(cells), 256):
            ids = cells[begin : begin + 256]
            coefficients = values[ids]
            origins = nodes[permutation][ids[:, 0]]
            points = origins[:, None, :] + reference[None, :, :] * spacing
            dp = coefficients @ phi.T - exact_pressure(points)
            gradient = np.einsum("ti,qid->tqd", coefficients, derivative / spacing)
            dq = np.einsum(
                "tqij,tqj->tqi", permeability(points, data), gradient - exact_gradient(points)
            )
            determinant = float(np.prod(spacing))
            pressure_square += determinant * float(np.einsum("q,tq,tq->", weights, dp, dp))
            flux_square += determinant * float(np.einsum("q,tqd,tqd->", weights, dq, dq))
        volume += float(np.prod(upper - lower))
    return {
        "pressure_L2": float(np.sqrt(pressure_square)),
        "flux_L2": float(np.sqrt(flux_square)),
        "pressure_L2_per_sqrt_volume": float(np.sqrt(pressure_square / volume)),
        "flux_L2_per_sqrt_volume": float(np.sqrt(flux_square / volume)),
        "error_gauss_order": order,
    }


def slice_fields(
    system: Any, solution: Any, plane: float = 0.37, *, data: DarcyData = DEFAULT_DATA
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample Q1 pressure and physical flux at independent cut fine-cell centroids."""
    corners = np.array(list(product((0, 1), repeat=3)))
    point_blocks, pressure_blocks, flux_blocks = [], [], []
    for metadata, field in zip(system.local_metadata, solution.fields, strict=True):
        lower, upper = metadata["bounds"]
        if not lower[2] < plane < upper[2]:
            continue
        nodes = metadata["coordinates"]
        r = metadata["refinement"]
        spacing = (upper - lower) / r
        lattice = np.rint((nodes - lower) / spacing).astype(np.int64)
        addresses = np.ravel_multi_index(lattice.T, (r + 1,) * 3)
        assert len(np.unique(addresses)) == (r + 1) ** 3
        permutation = np.argsort(addresses)
        coordinates, values = nodes[permutation], np.asarray(field)[permutation]
        k = int((plane - lower[2]) / spacing[2])
        ids = np.array(
            [
                [(i + a) * (r + 1) ** 2 + (j + b) * (r + 1) + k + c for a, b, c in corners]
                for i, j in product(range(r), repeat=2)
            ]
        )
        origins = coordinates[ids[:, 0]]
        reference = np.array([[0.5, 0.5, (plane - origins[0, 2]) / spacing[2]]])
        phi, derivative = tensor_lagrange_tabulation(
            "hexahedron", 1, reference, nodes=corners, nderiv=1
        )
        points = origins + reference[0] * spacing
        pressure = (values[ids] @ phi.T)[:, 0]
        gradient = np.einsum("ti,id->td", values[ids], derivative[0] / spacing)
        flux = -np.einsum("tij,tj->ti", permeability(points, data), gradient)
        point_blocks.append(points)
        pressure_blocks.append(pressure)
        flux_blocks.append(flux)
    return (
        np.concatenate(point_blocks),
        np.concatenate(pressure_blocks),
        np.concatenate(flux_blocks),
    )


def classical_at_points(state: NativeState, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate literal conforming Q1 coefficients in their coordinate-addressed cells."""
    _, coefficients = canonical_coefficients(state)
    n = state.grid[1]
    cells = np.floor(points * n).astype(np.int64)
    cells = np.minimum(cells, np.array(state.grid) - 1)
    reference = points * n - cells
    corners = np.array(list(product((0, 1), repeat=3)))
    ids = np.stack(
        [
            np.ravel_multi_index((cells + corner).T, tuple(i + 1 for i in state.grid))
            for corner in corners
        ],
        axis=1,
    )
    phi, derivative = tensor_lagrange_tabulation(
        "hexahedron", 1, reference, nodes=corners, nderiv=1
    )
    pressure = np.einsum("ti,ti->t", coefficients[ids], phi)
    gradient = np.einsum("ti,tid->td", coefficients[ids], derivative * n)
    flux = -np.einsum("tij,tj->ti", permeability(points, state.data), gradient)
    return pressure, flux


def plot_mhm_slice(
    system: Any,
    solution: Any,
    macro: HexMesh,
    classical: NativeState,
    *,
    data: DarcyData = DEFAULT_DATA,
) -> list[Path]:
    """Compare exact/classical/local fields with distinct macro overlays and colorbars."""
    points, pressure, flux = slice_fields(system, solution, data=data)
    exact_p, exact_q = exact_pressure(points), exact_flux(points, data)
    classical_p, classical_q = classical_at_points(classical, points)
    families = {
        "pressure": (
            [
                ("Exact pressure", exact_p),
                ("Classical Q1 pressure", classical_p),
                ("MHM pressure", pressure),
                ("Material multiplier", medium(points, data)),
                ("Classical pressure error", classical_p - exact_p),
                ("MHM pressure error", pressure - exact_p),
            ]
        ),
        "flux": (
            [
                ("Exact Darcy flux magnitude", np.linalg.norm(exact_q, axis=1)),
                ("Classical Darcy flux magnitude", np.linalg.norm(classical_q, axis=1)),
                ("MHM Darcy flux magnitude", np.linalg.norm(flux, axis=1)),
                ("Material multiplier", medium(points, data)),
                (
                    "Classical vector-flux error magnitude",
                    np.linalg.norm(classical_q - exact_q, axis=1),
                ),
                ("MHM vector-flux error magnitude", np.linalg.norm(flux - exact_q, axis=1)),
            ]
        ),
    }
    destinations = []
    x_lines, y_lines = np.unique(macro.points[:, 0]), np.unique(macro.points[:, 1])
    for name, quantities in families.items():
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
        shared_min = min(float(np.min(values)) for _, values in quantities[:3])
        shared_max = max(float(np.max(values)) for _, values in quantities[:3])
        for index, (ax, (title, values)) in enumerate(zip(axes.flat, quantities, strict=True)):
            options = {"vmin": shared_min, "vmax": shared_max} if index < 3 else {}
            artist = ax.scatter(
                points[:, 0], points[:, 1], c=values, s=35, marker="s", cmap="viridis", **options
            )
            fig.colorbar(artist, ax=ax, shrink=0.8)
            for x in x_lines:
                ax.axvline(
                    x,
                    color="black",
                    linewidth=0.7,
                    path_effects=[patheffects.withStroke(linewidth=1.7, foreground="white")],
                )
            for y in y_lines:
                ax.axhline(
                    y,
                    color="black",
                    linewidth=0.7,
                    path_effects=[patheffects.withStroke(linewidth=1.7, foreground="white")],
                )
            ax.set(
                title=title,
                xlabel="x",
                ylabel="y",
                aspect="equal",
                xlim=(x_lines[0], x_lines[-1]),
                ylim=(y_lines[0], y_lines[-1]),
            )
        fig.suptitle("Darcy fields at z = 0.37; black lines mark macrofaces")
        destination = OUTPUT / f"small_{name}_fields.png"
        fig.savefig(destination, dpi=160)
        plt.show()
        destinations.append(destination)
    return destinations


def archive_fields(system: Any, solution: Any, destination: Path) -> dict[str, Any]:
    """Persist executed basis/lifts/geometry and verify reconstruction in their coordinates."""
    arrays: dict[str, np.ndarray] = {"global_trace": solution.trace}
    largest = 0.0
    basis_digests: dict[str, str] = {}
    for cell, (response, coefficients, field, metadata) in enumerate(
        zip(system.responses, solution.coarse, solution.fields, system.local_metadata, strict=True)
    ):
        local_trace = solution.trace[response.problem.trace_dofs]
        for native_threads in (1, 2):
            with threadpool_limits(native_threads):
                replay = response.reconstruct(local_trace, coefficients)
                rotated = replace(response, coarse_vectors=-response.retained_basis)
                rotated_replay = rotated.reconstruct(local_trace, -coefficients)
                np.testing.assert_allclose(replay, field, rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(rotated_replay, field, rtol=1e-13, atol=1e-13)
                largest = max(
                    largest,
                    float(np.max(np.abs(replay - field), initial=0)),
                    float(np.max(np.abs(rotated_replay - field), initial=0)),
                )
        basis = np.ascontiguousarray(response.retained_basis)
        header = json.dumps(
            {"shape": basis.shape, "dtype": basis.dtype.str}, sort_keys=True
        ).encode()
        basis_digests[str(cell)] = hashlib.sha256(header + basis.tobytes()).hexdigest()
        prefix = f"cell_{cell}_"
        for name, values in {
            "source": response.source,
            "trace_lifts": response.lifts,
            "retained_basis": response.retained_basis,
            "trace_dofs": response.problem.trace_dofs,
            "coarse_coefficients": coefficients,
            "field_coefficients": field,
            "coordinates": metadata["coordinates"],
            "bounds": metadata["bounds"],
            "physical_moments": response.problem.constraints,
            "declared_kernel": response.problem.kernel,
        }.items():
            arrays[prefix + name] = np.asarray(values)
    np.savez_compressed(destination, **arrays)
    return {
        "path": destination.name,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "maximum_coefficient_replay_difference": largest,
        "cells": len(system.responses),
        "executed_basis_sha256": basis_digests,
        "replay_native_threads": [1, 2],
        "equivalent_retained_rotation": "sign reversal with matching coefficient reversal",
    }


def summarize_scaling(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare inclusive complete samples with fixed strong or per-worker weak work."""
    summaries = []
    for section in ("strong", "weak"):
        families: dict[tuple[Any, ...], dict[int, list[dict[str, Any]]]] = {}
        for sample in record.get(section, []):
            if sample.get("backend") != "process":
                continue
            workers = int(sample["workers"])
            fine_cells = int(sample["fine_cells"])
            macro_cells = int(sample["macro_cells"])
            work = (
                (fine_cells / workers, macro_cells / workers)
                if section == "weak"
                else (sample.get("length", 1), fine_cells, macro_cells)
            )
            family = (
                sample["fine_n"],
                sample.get("trace_degree"),
                sample.get("local_solver"),
                sample.get("material_period", 0.1),
                work,
            )
            families.setdefault(family, {}).setdefault(workers, []).append(sample)
        for family, configurations in sorted(families.items()):
            times = {
                workers: [
                    float(sample.get("total_including_parent_startup_seconds", sample["total"]))
                    for sample in samples
                ]
                for workers, samples in configurations.items()
            }
            medians = {workers: float(np.median(values)) for workers, values in times.items()}
            baseline_workers = 1 if section == "strong" else min(configurations)
            baseline = medians.get(baseline_workers)
            for workers, samples in sorted(configurations.items()):
                sample = samples[0]
                median = medians[workers]
                ratio = None if baseline is None else baseline / median
                summaries.append(
                    {
                        "study": section,
                        "fine_n": family[0],
                        "trace_degree": family[1],
                        "local_solver": family[2],
                        "material_period": family[3],
                        "workers": workers,
                        "domain_length": sample.get("length", 1),
                        "fine_cells": sample["fine_cells"],
                        "macro_cells": sample["macro_cells"],
                        "local_fine_cells_per_worker": sample["fine_cells"] / workers,
                        "macro_cells_per_worker": sample["macro_cells"] / workers,
                        "baseline_workers": baseline_workers,
                        "samples": len(samples),
                        "median_seconds": median,
                        "minimum_seconds": min(times[workers]),
                        "maximum_seconds": max(times[workers]),
                        "speedup_or_weak_efficiency": ratio,
                        "strong_efficiency": ratio / workers
                        if ratio is not None and section == "strong"
                        else None,
                        "timing_scope": (
                            "inclusive launch when recorded; complete solver pipeline otherwise"
                        ),
                    }
                )
    return summaries


def plot_scaling(rows: list[dict[str, Any]]) -> Path | None:
    """Plot strong self-speedups and weak efficiencies from their measured baselines."""
    if not rows:
        return None
    fig = plt.figure(figsize=(11, 6.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(4.5, 1.35))
    axes = [fig.add_subplot(grid[0, column]) for column in range(2)]
    legend_axes = [fig.add_subplot(grid[1, column]) for column in range(2)]
    for study, ax in zip(("strong", "weak"), axes, strict=True):
        families = sorted(
            {
                (
                    row["fine_n"],
                    row["trace_degree"],
                    row["local_solver"],
                    row["material_period"],
                    row["local_fine_cells_per_worker"] if study == "weak" else None,
                    row["macro_cells_per_worker"] if study == "weak" else None,
                )
                for row in rows
                if row["study"] == study
            }
        )
        for family in families:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["study"] == study
                    and (
                        row["fine_n"],
                        row["trace_degree"],
                        row["local_solver"],
                        row["material_period"],
                        row["local_fine_cells_per_worker"] if study == "weak" else None,
                        row["macro_cells_per_worker"] if study == "weak" else None,
                    )
                    == family
                ),
                key=lambda row: row["workers"],
            )
            selected = [row for row in selected if row["speedup_or_weak_efficiency"] is not None]
            if selected:
                label = f"n={family[0]}, trace Q{family[1]}, {family[2]}, ε={family[3]}"
                if study == "weak":
                    label += (
                        f", {family[5]:g} macros/process, base={selected[0]['baseline_workers']}"
                    )
                ax.plot(
                    [row["workers"] for row in selected],
                    [row["speedup_or_weak_efficiency"] for row in selected],
                    "o-",
                    label=label,
                )
    axes[0].set(title="Strong process self-speedup", xlabel="Processes", ylabel="T(1) / T(P)")
    axes[1].set(title="Focused weak efficiency", xlabel="Processes", ylabel="T(P_base) / T(P)")
    fig.suptitle("ε: permeability spatial period", fontsize=12)
    for ax, legend_ax in zip(axes, legend_axes, strict=True):
        legend_ax.set_axis_off()
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            ax.text(
                0.5,
                0.5,
                "No completed measurements",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        if handles:
            legend_ax.legend(handles, labels, fontsize=7, loc="center", framealpha=1)
    destination = OUTPUT / "scaling_summary.png"
    fig.savefig(destination, dpi=160)
    plt.show()
    return destination


def accuracy_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Select actual physical norms with separate classical/MHM spaces and solver labels."""
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for section in ("strong", "classical"):
        for sample in record.get(section, []):
            errors = sample.get("physical_errors")
            if not errors:
                continue
            fine_n = sample.get("fine_n", sample.get("fine_grid", [None, None])[1])
            if fine_n is None:
                continue
            label = (
                f"MHM / {sample.get('local_solver', 'unspecified')}"
                if section == "strong"
                else (
                    f"Classical / {sample.get('solver', 'unspecified')} / "
                    f"{sample.get('mpi_ranks', 1)} ranks"
                )
            )
            label += f" / ε={sample.get('material_period', 0.1)}"
            key = label, fine_n, sample.get("trace_degree")
            rows.setdefault(
                key,
                {
                    "label": label,
                    "fine_n": fine_n,
                    "pressure_L2": errors["pressure_L2"],
                    "flux_L2": errors["flux_L2"],
                },
            )
    return list(rows.values())


def plot_accuracy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Plot separately integrated errors and return observed classical refinement rates."""
    if not rows:
        return []
    fig = plt.figure(figsize=(11, 7), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(4.5, 1.8))
    axes = [fig.add_subplot(grid[0, column]) for column in range(2)]
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.set_axis_off()
    classical_rates = []
    for label in sorted({row["label"] for row in rows}):
        selected = sorted(
            (row for row in rows if row["label"] == label), key=lambda row: row["fine_n"]
        )
        fine_counts = [row["fine_n"] for row in selected]
        for ax, field, title in zip(
            axes,
            ("pressure_L2", "flux_L2"),
            ("Pressure error", "Physical vector-flux error"),
            strict=True,
        ):
            ax.loglog(fine_counts, [row[field] for row in selected], "o-", label=label)
            ax.set(title=title, xlabel="Fine cells per unit direction", ylabel="L2 error")
            ax.grid(alpha=0.25)
        if label.startswith("Classical"):
            for first, second in zip(selected[:-1], selected[1:], strict=True):
                denominator = math.log(second["fine_n"] / first["fine_n"])
                classical_rates.append(
                    {
                        "method": label,
                        "from_n": first["fine_n"],
                        "to_n": second["fine_n"],
                        "pressure_rate": math.log(first["pressure_L2"] / second["pressure_L2"])
                        / denominator,
                        "flux_rate": math.log(first["flux_L2"] / second["flux_L2"]) / denominator,
                    }
                )
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, fontsize=8, loc="center", framealpha=1)
    fig.suptitle(
        "Local-resolution sweep; macro mesh and MHM trace are fixed\nε: permeability spatial period"
    )
    fig.savefig(OUTPUT / "physical_accuracy_by_resolution.png", dpi=160)
    plt.show()
    return classical_rates


def archive_local(system: Any, destination: Path, cell: int = 0) -> None:
    """Archive original Neumann coefficients and the exact executed kernel/moments."""
    problem = system.responses[cell].problem
    A = problem.matrix.tocsr()
    np.savez_compressed(
        destination,
        A_data=A.data,
        A_indices=A.indices,
        A_indptr=A.indptr,
        A_shape=A.shape,
        B=problem.coupling,
        f=problem.load,
        Z=problem.kernel,
        C=problem.constraints,
        trace_dofs=problem.trace_dofs,
    )


def load_problem(path: Path) -> LocalProblem:
    """Load the executed CSR, face columns and physical moments without changing them."""
    with np.load(path, allow_pickle=False) as archive:
        operator = sparse.csr_matrix(
            (archive["A_data"], archive["A_indices"], archive["A_indptr"]),
            shape=tuple(archive["A_shape"]),
        )
        return LocalProblem(
            operator,
            archive["B"],
            archive["f"],
            archive["trace_dofs"],
            kernel=archive["Z"],
            constraints=archive["C"],
        )


def original_rows(problem: LocalProblem, columns: np.ndarray) -> dict[str, Any]:
    """Independently fit the auxiliary moments and check every original KKT row."""
    rhs = np.column_stack((problem.load, problem.coupling))
    residual = rhs.astype(np.longdouble) - problem.matrix @ columns.astype(np.longdouble)
    constraints = problem.constraints.astype(np.longdouble)
    gram = np.asarray(constraints.T @ constraints, dtype=float)
    moments = np.linalg.solve(gram, np.asarray(constraints.T @ residual, dtype=float))
    residual -= constraints @ moments.astype(np.longdouble)
    gauge = constraints.T @ columns.astype(np.longdouble)
    row_norms = np.linalg.norm(np.asarray(residual, dtype=float), axis=0)
    gauge_norms = np.linalg.norm(np.asarray(gauge, dtype=float), axis=0)
    augmented_norms = np.hypot(row_norms, gauge_norms)
    rhs_norms = np.linalg.norm(rhs, axis=0)
    bounds = 1e-10 * rhs_norms
    assert np.all(augmented_norms <= bounds), (augmented_norms, bounds)
    return {
        "physical_absolute_residuals": row_norms.tolist(),
        "physical_relative_residuals": np.divide(
            row_norms, rhs_norms, out=np.zeros_like(row_norms), where=rhs_norms > 0
        ).tolist(),
        "augmented_relative_residuals": np.divide(
            augmented_norms, rhs_norms, out=np.zeros_like(augmented_norms), where=rhs_norms > 0
        ).tolist(),
        "physical_accepted_bounds": bounds.tolist(),
        "gauge_absolute_defects": np.max(np.abs(gauge), axis=0).astype(float).tolist(),
    }


def condense_shard(
    solver: str, slot: int, archives: tuple[str, ...]
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    """Own one fixed device or CPU shard in a fresh spawn process and return host data."""
    cp = None
    metadata: dict[str, Any] = {"solver": solver, "slot": slot, "operator_count": len(archives)}
    if solver == "amgx":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(slot)
        import cupy as cp

        assert cp.cuda.runtime.getDeviceCount() == 1
        cp.cuda.runtime.deviceSynchronize()
        properties = cp.cuda.runtime.getDeviceProperties(0)
        before_free, total = cp.cuda.runtime.memGetInfo()
        metadata.update(
            {
                "physical_device": slot,
                "visible_device": 0,
                "visible_device_count": cp.cuda.runtime.getDeviceCount(),
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "name": properties["name"].decode(),
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver": cp.cuda.runtime.driverGetVersion(),
                "cupy_version": cp.__version__,
                "memory_total": total,
                "free_memory_before_native_condensation": before_free,
            }
        )

    from threadpoolctl import threadpool_info, threadpool_limits

    from pymhm.linalg.linear import preload_solver_backend

    preload_solver_backend(solver)
    result = []
    try:
        with threadpool_limits(1):
            for archive in archives:
                problem = load_problem(Path(archive))
                result.append((Path(archive).name, problem.condense(solver)))
            metadata["native_pools_inside_limits"] = threadpool_info()
    finally:
        if cp is not None:
            cp.cuda.runtime.deviceSynchronize()
    if cp is not None:
        after_free, _ = cp.cuda.runtime.memGetInfo()
        metadata["free_memory_after_native_cleanup"] = after_free
    return result, metadata


def run_gpu_component(
    archives: tuple[Path, ...], solver: str, slots: tuple[int, ...]
) -> tuple[float, list[tuple[str, Any]], list[dict[str, Any]]]:
    """Include fresh spawn, import, load, native lifecycle, return and shutdown."""
    beginning = time.perf_counter()
    with ExitStack() as resources:
        pools = [
            resources.enter_context(
                ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            )
            for _ in slots
        ]
        futures = [
            pools[index].submit(
                GpuCondenseShard,
                solver,
                slot,
                tuple(str(path) for path in archives[index :: len(slots)]),
            )
            for index, slot in enumerate(slots)
        ]
        completed = [future.result() for future in futures]
    elapsed = time.perf_counter() - beginning
    responses = sorted(
        [item for result, _ in completed for item in result], key=lambda item: item[0]
    )
    return elapsed, responses, [metadata for _, metadata in completed]


def validate_gpu_responses(responses: list[tuple[str, Any]]) -> dict[str, Any]:
    """Check every original source/face row and physical gauge after timed return."""
    import numpy as np

    rows = []
    for name, response in responses:
        columns = np.column_stack((response.source, response.lifts))
        control = original_rows(response.problem, columns)
        retained = response.retained_basis.astype(np.longdouble)
        original_moments = response.problem.constraints.astype(
            np.longdouble
        ).T @ response.problem.coarse_basis.astype(np.longdouble)
        moment_defect = response.problem.constraints.astype(np.longdouble).T @ (
            retained - response.problem.coarse_basis.astype(np.longdouble)
        )
        assert np.linalg.norm(moment_defect) <= 1e-10 * np.linalg.norm(original_moments)
        rows.append(
            {
                "archive": name,
                "dofs": response.problem.matrix.shape[0],
                "nnz": response.problem.matrix.nnz,
                "source_plus_trace_rhs": columns.shape[1],
                "represented_kernel_correction": response.problem._correct_kernel,
                "executed_rhs_width": columns.shape[1]
                + (response.problem.kernel.shape[1] if response.problem._correct_kernel else 0),
                "max_physical_relative_residual": max(control["physical_relative_residuals"]),
                "max_augmented_relative_residual": max(control["augmented_relative_residuals"]),
                "max_gauge_absolute_defect": max(control["gauge_absolute_defects"]),
                "retained_moment_defect_norm": float(np.linalg.norm(moment_defect)),
                "retained_basis_original_action_norm": float(
                    np.linalg.norm(response.problem.matrix @ retained)
                ),
                "retained_basis_sha256": hashlib.sha256(
                    np.ascontiguousarray(response.retained_basis).tobytes()
                ).hexdigest(),
            }
        )
    return {
        "all_original_physical_columns_accepted": True,
        "rtol": 1e-10,
        "atol": 0.0,
        "local_rows": rows,
    }


@dataclass
class FullGpuAssembleOnly:
    """Own one CPU worker's displayed provider and native lifetime cleanup."""

    provider: Any

    def __post_init__(self) -> None:
        """Bind the same custom interface once in each worker's portable state."""
        self.problem = bind_problem(
            MeshHierarchy(self.provider.macro, self.provider.local_mesh),
            TensorFaceInterface(self.provider.macro, self.provider.trace_degree),
            self.provider,
            global_equation=Equation(0, 0),
            retained=1,
        )

    def __call__(self, cell: int) -> LocalAssembly:
        """Delegate form assembly and return validated portable coefficients."""
        compiled = compile_local_equations(self.problem.local_provider(cell))
        return LocalAssembly(compiled.problem, compiled.metadata)

    def prepare_runtime(self) -> None:
        """Forward environment preparation before the CPU worker applies its budget."""
        self.provider.prepare_runtime()

    def close(self) -> None:
        """Release this worker's native form, geometry and buffer resources."""
        self.provider.close()


@dataclass(frozen=True)
class FullGpuResponseArrays:
    """Return executed responses without transferring the original matrix again."""

    source: np.ndarray
    lifts: np.ndarray
    coarse_vectors: np.ndarray | None


def restore_full_gpu_response(
    assembled: LocalAssembly, arrays: FullGpuResponseArrays
) -> LocalResponse:
    """Attach unchanged arrays to the coordinator's original local problem."""
    problem = assembled.problem
    shapes = ((problem.matrix.shape[0],), problem.coupling.shape, problem.coarse_basis.shape)
    for value, shape in zip(
        (arrays.source, arrays.lifts, arrays.coarse_vectors), shapes, strict=True
    ):
        if value is not None and (
            np.iscomplexobj(value) or value.shape != shape or not np.isfinite(value).all()
        ):
            raise ValueError("Response transport must preserve finite real array shapes")
    return LocalResponse(problem, arrays.source, arrays.lifts, arrays.coarse_vectors)


def condense_full_gpu_shard(
    physical_device: int,
    indexed: tuple[tuple[int, LocalAssembly], ...],
    solver: str,
) -> tuple[list[tuple[int, FullGpuResponseArrays]], dict[str, Any]]:
    """Delegate fresh local solves to one device and return only host arrays."""
    if solver not in {"cudss", "amgx"}:
        raise ValueError("Select cudss or amgx for this GPU example")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    began = time.perf_counter()
    import cupy as cp

    from pymhm.linalg.linear import preload_solver_backend

    preload_solver_backend(solver)
    cp.cuda.runtime.deviceSynchronize()
    initialized = time.perf_counter()
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError("Each GPU process requires exactly one visible device")
    rows = []
    try:
        with threadpool_limits(1), cp.cuda.Device(0):
            for index, assembled in indexed:
                response = assembled.problem.condense(solver)
                rows.append(
                    (
                        index,
                        FullGpuResponseArrays(
                            response.source, response.lifts, response.coarse_vectors
                        ),
                    )
                )
            native_pools = threadpool_info()
    finally:
        cp.cuda.runtime.deviceSynchronize()
    completed = time.perf_counter()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    return rows, {
        "physical_device": physical_device,
        "visible_device": 0,
        "name": properties["name"].decode(),
        "pid": os.getpid(),
        "operator_count": len(indexed),
        "native_threads": 1,
        "native_pools_inside_limits": native_pools,
        "import_and_initialization_seconds": initialized - began,
        "condensation_and_synchronization_seconds": completed - initialized,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "cuda_path": os.environ.get("CUDA_PATH"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "factor_reuse": "Fresh matrix-dependent factors/hierarchy for each macrocell",
    }


def run_full_gpu_mhm(
    fine_n: int,
    *,
    devices: tuple[int, ...],
    assembly_workers: int,
    solver: str = "cudss",
    data: DarcyData | None = None,
    length: int = 1,
    trace_degree: int = 1,
    quadrature_degree: int = 8,
) -> tuple[dict[str, Any], HybridSystem, Any, HexMesh]:
    """Time displayed CPU assembly, dedicated GPU solves and original global solve."""
    if fine_n % 4 or assembly_workers < 1:
        raise ValueError("fine_n must be divisible by four; use positive CPU workers")
    if not devices or len(devices) != len(set(devices)) or min(devices) < 0:
        raise ValueError("Select distinct nonnegative physical GPU indices")
    data = DarcyData() if data is None else data
    captured = current_source_manifest(
        {
            "importable_full_gpu_worker": full_gpu_source_sha256,
            **execution_source_manifest(
                "introduction/darcy_3d_parallel_scalability.ipynb", workspace=ROOT
            ),
        }
    )
    with threadpool_limits(1):
        started = time.perf_counter()
        macro = macro_mesh(4, length)
        provider = full_gpu_module.LocalProvider(
            macro,
            fine_n // 4,
            data=full_gpu_module.DarcyData(period=data.period, anisotropy=data.anisotropy),
            trace_degree=trace_degree,
            quadrature_degree=quadrature_degree,
        )
        factory = full_gpu_module.FullGpuAssembleOnly(provider)
        assemblies = tuple(
            map_local(
                factory,
                range(len(macro.cells)),
                backend="process",
                workers=assembly_workers,
                native_threads=1,
                batch_size=assembly_workers,
            )
        )
        assembled = time.perf_counter()
        indexed = tuple(enumerate(assemblies))
        with ExitStack() as resources:
            pools = [
                resources.enter_context(
                    ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
                )
                for _ in devices
            ]
            futures = [
                pool.submit(
                    full_gpu_module.condense_full_gpu_shard,
                    device,
                    indexed[slot :: len(devices)],
                    solver,
                )
                for slot, (pool, device) in enumerate(zip(pools, devices, strict=True))
            ]
            completed = [future.result() for future in futures]
        ordered = sorted((item for rows, _ in completed for item in rows), key=lambda item: item[0])
        if [index for index, _ in ordered] != list(range(len(assemblies))):
            raise RuntimeError("GPU responses must cover every macrocell exactly once")
        responses = tuple(
            restore_full_gpu_response(assemblies[index], arrays) for index, arrays in ordered
        )
        condensed = time.perf_counter()
        system = HybridSystem.from_responses(
            responses, metadata=tuple(item.metadata for item in assemblies)
        )
        reduced = time.perf_counter()
        solution = system.solve()
        finished = time.perf_counter()
    record = {
        "fine_n": fine_n,
        "domain_length": length,
        "local_solver": solver,
        "assembly_workers": assembly_workers,
        "native_threads_per_worker": 1,
        "devices": list(devices),
        "gpu_processes": len(devices),
        "fine_cells": fine_n**3 * length,
        "macro_cells": len(macro.cells),
        "material_period": data.period,
        "anisotropy": data.anisotropy,
        "assembly_quadrature_degree": quadrature_degree,
        "trace_degree": trace_degree,
        "source_manifest": captured,
        "cpu_setup_assembly_pool_seconds": assembled - started,
        "gpu_condensation_pool_seconds": condensed - assembled,
        "ordered_global_assembly_seconds": reduced - condensed,
        "global_solve_reconstruction_seconds": finished - reduced,
        "complete_workflow_seconds": finished - started,
        "global_relative_residual": float(solution.residual),
        "gpu_worker_metadata": [metadata for _, metadata in completed],
        "timing_scope": "Fresh CPU/GPU pools through global reconstruction; no overlap",
        "matrix_factor_reuse": False,
    }
    return record, system, solution, macro


def published_accelerator_figures(record: dict[str, Any]) -> list[tuple[Path, str]]:
    """Resolve the record's published figure paths and their recorded captions."""
    plots = record["plots"]
    entries = plots.items() if isinstance(plots, dict) else enumerate(plots)
    figures = []
    for name, entry in entries:
        if isinstance(entry, str):
            filename, caption = entry, str(name)
        elif isinstance(entry, dict):
            if entry.get("available") is False:
                print(f"{name}: {entry.get('reason', 'No completed measurement is recorded')}")
                continue
            filename = entry.get(
                "figure", entry.get("path", entry.get("file", entry.get("filename")))
            )
            caption = str(entry.get("caption", entry.get("title", name)))
        else:
            raise TypeError("Each published figure must declare its file path")
        if filename is None:
            raise ValueError(f"Published figure {name!r} has no file path")
        destination = ACCELERATED_CAMPAIGN / str(filename)
        if not destination.is_file():
            repository_destination = ROOT / str(filename)
            if repository_destination.is_file():
                destination = repository_destination
            else:
                print(f"Historical accelerator figure unavailable in this workspace: {filename}")
                continue
        figures.append((destination, caption))
    return figures


# Real module identity is preserved in spawn and persisted source provenance.
worker_module: Any = sys.modules[__name__]
full_gpu_module: Any = worker_module
SpawnLocalProvider = LocalProvider
GpuCondenseShard = condense_shard
worker_source = Path(__file__).read_text()
worker_sha256 = hashlib.sha256(worker_source.encode()).hexdigest()
full_gpu_source_sha256 = worker_sha256
export_import_seconds = 0.0


def run_study(
    *,
    RUN_SMALL_REPRODUCTION: bool = True,
    RUN_LARGE_CAMPAIGN: bool = False,
    RUN_GPU_COMPONENT: bool = False,
    RUN_FULL_GPU_WORKFLOW: bool = False,
    data: DarcyData = DEFAULT_DATA,
    trace_degree: int = 1,
    quadrature_degree: int = 8,
    small_fine_n: int = 16,
) -> dict[str, Any]:
    """Execute CPU/GPU acquisitions, exact-field checks, archives and available panels."""
    DATA = data
    REPETITIONS = 1 if not RUN_LARGE_CAMPAIGN else 3
    STRONG_FINE_COUNTS = (small_fine_n,) if not RUN_LARGE_CAMPAIGN else (64, 96, 128)
    WORKER_COUNTS = (1, 2) if not RUN_LARGE_CAMPAIGN else (1, 8, 16, 32)
    WEAK_FINE_N = small_fine_n if not RUN_LARGE_CAMPAIGN else 64
    WEAK_WORKERS = (1, 2) if not RUN_LARGE_CAMPAIGN else (1, 2, 4, 8)
    RANDOM_SEED = 731
    LOCAL_SOLVERS = ("scipy", "pyamg")
    CLASSICAL_SOLVERS = ("scipy", "pyamg")

    reproduction = {
        "metadata": {
            "repetitions": REPETITIONS,
            "native_threads": 1,
            "trace_degree": trace_degree,
            "material_period": data.period,
            "workspace_reuse": (
                "native geometry/spaces/form buffers, geometry-only B/C; A/f "
                "and local solvers fresh per cell"
            ),
            "assembly_quadrature_degree": quadrature_degree,
            "worker_source_sha256": worker_sha256,
            "python": platform.python_version(),
            "source_manifest": current_source_manifest(
                {
                    **execution_source_manifest(
                        "introduction/darcy_3d_parallel_scalability.ipynb", workspace=ROOT
                    ),
                }
            ),
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "versions": {
                name: version(name)
                for name in ("numpy", "scipy", "fenics-basix", "fenics-dolfinx", "pyamg")
            },
            "native_libraries": threadpool_info(),
        },
        "warmup": [],
        "strong": [],
        "weak": [],
        "classical": [],
    }
    last_mhm = None
    last_classical = None

    if RUN_SMALL_REPRODUCTION or RUN_LARGE_CAMPAIGN:
        controls = audit_local(
            LocalProvider(
                macro_mesh(),
                max(4, STRONG_FINE_COUNTS[0] // 4),
                data=DATA,
                trace_degree=trace_degree,
                quadrature_degree=quadrature_degree,
            )
        )
        reproduction["accuracy_controls"] = {"source": verify_source(DATA), "local": controls}
        warmup_n = 16
        for solver in CLASSICAL_SOLVERS:
            classical_warm, _ = solve_classical(
                warmup_n, data=DATA, solver=solver, quadrature_degree=quadrature_degree
            )
            reproduction["warmup"].append({"method": "classical", **classical_warm})
        for solver in LOCAL_SOLVERS:
            for workers in WORKER_COUNTS:
                record, _, _, _ = run_mhm(
                    warmup_n,
                    workers=workers,
                    backend="process",
                    solver=solver,
                    data=DATA,
                    trace_degree=trace_degree,
                    quadrature_degree=quadrature_degree,
                )
                reproduction["warmup"].append(record)

        rng = random.Random(RANDOM_SEED)
        for fine_n in STRONG_FINE_COUNTS:
            configurations = [("classical", 1, solver) for solver in CLASSICAL_SOLVERS]
            configurations += [
                (backend, workers, solver)
                for solver in LOCAL_SOLVERS
                for backend, workers in [
                    ("serial", 1),
                    *(("process", workers) for workers in WORKER_COUNTS),
                ]
            ]
            for repetition in range(REPETITIONS):
                order = configurations.copy()
                rng.shuffle(order)
                for backend, workers, solver in order:
                    if backend == "classical":
                        record, state = solve_classical(
                            fine_n, data=DATA, solver=solver, quadrature_degree=quadrature_degree
                        )
                        record.update(
                            {
                                "method": "classical",
                                "fine_n": fine_n,
                                "workers": 1,
                                "material_period": DATA.period,
                                "repetition": repetition,
                                "physical_errors": classical_physical_errors(state),
                            }
                        )
                        reproduction["classical"].append(record)
                        last_classical = state
                    else:
                        record, system, solution, macro = run_mhm(
                            fine_n,
                            workers=workers,
                            backend=backend,
                            solver=solver,
                            data=DATA,
                            trace_degree=trace_degree,
                            quadrature_degree=quadrature_degree,
                        )
                        record.update(
                            {
                                "repetition": repetition,
                                "physical_errors": mhm_physical_errors(
                                    system, solution, data=DATA, order=7
                                ),
                            }
                        )
                        reproduction["strong"].append(record)
                        last_mhm = system, solution, macro
                    print(
                        {
                            "fine_n": fine_n,
                            "backend": backend,
                            "workers": workers,
                            "solver": solver,
                            "total_seconds": record["total"],
                            "physical_errors": record["physical_errors"],
                        }
                    )

    if RUN_LARGE_CAMPAIGN:
        for solver in LOCAL_SOLVERS:
            for repetition in range(REPETITIONS):
                order = list(WEAK_WORKERS)
                rng.shuffle(order)
                for workers in order:
                    record, system, solution, _ = run_mhm(
                        WEAK_FINE_N,
                        workers=workers,
                        backend="process",
                        solver=solver,
                        data=DATA,
                        length=workers,
                        trace_degree=trace_degree,
                        quadrature_degree=quadrature_degree,
                    )
                    record.update(
                        {
                            "repetition": repetition,
                            "physical_errors": mhm_physical_errors(
                                system, solution, data=DATA, order=7
                            ),
                        }
                    )
                    reproduction["weak"].append(record)

    if RUN_SMALL_REPRODUCTION or RUN_LARGE_CAMPAIGN:
        (OUTPUT / "reproduction.json").write_text(json.dumps(reproduction, indent=2) + "\n")

    if last_mhm is not None and last_classical is not None:
        plot_mhm_slice(*last_mhm, last_classical, data=DATA)

    if last_mhm is not None:
        reproduction["archive"] = archive_fields(
            last_mhm[0], last_mhm[1], OUTPUT / "fields_and_executed_bases.npz"
        )
        (OUTPUT / "reproduction.json").write_text(json.dumps(reproduction, indent=2) + "\n")

    published = json.loads(PUBLISHED.read_text()) if PUBLISHED.exists() else None

    if published is not None:
        print(json.dumps(published.get("metadata", {}), indent=2))
        print(
            {
                section: len(published.get(section, []))
                for section in ("strong", "weak", "classical", "gpu")
            }
        )
    else:
        print(
            "No published campaign record is installed; the small reproduction remains available."
        )

    scaling_summary = summarize_scaling(published if published is not None else reproduction)

    print(scaling_summary)

    plot_scaling(scaling_summary)

    field_error_rows = accuracy_rows(published if published is not None else reproduction)

    reproduction["observed_classical_rates"] = plot_accuracy(field_error_rows)

    GPU_DEVICES = (0, 1)

    GPU_REPETITIONS = 3

    if RUN_GPU_COMPONENT:
        if last_mhm is None:
            raise ValueError("Execute a CPU formulation first to provide original local operators")
        gpu_digest = worker_sha256
        local_directory = OUTPUT / "original_local_operators"
        local_directory.mkdir(exist_ok=True)
        for cell in range(len(last_mhm[0].responses)):
            archive_local(last_mhm[0], local_directory / f"cell-{cell:03d}.npz", cell)
        original_archives = tuple(sorted(local_directory.glob("cell-*.npz")))
        gpu_records = []
        for repetition in range(GPU_REPETITIONS):
            seconds, responses, device_metadata = run_gpu_component(
                original_archives, "amgx", GPU_DEVICES
            )
            gpu_records.append(
                {
                    "repetition": repetition,
                    "full_component_seconds": seconds,
                    "devices": list(GPU_DEVICES),
                    "worker_metadata": device_metadata,
                    "scope": "original local condensation component; no FEM or global solve",
                    "controls": validate_gpu_responses(responses),
                    "worker_source_sha256": gpu_digest,
                }
            )
        (OUTPUT / "gpu_component.json").write_text(json.dumps(gpu_records, indent=2) + "\n")

    FULL_GPU_FINE_N = 16

    FULL_GPU_ASSEMBLY_WORKERS = 2

    FULL_GPU_DEVICES = (0, 1)

    FULL_GPU_SOLVERS = ("cudss", "amgx")

    FULL_GPU_DATA = DarcyData(period=0.137)

    full_gpu_reproduction: list[dict[str, Any]] = []

    if RUN_FULL_GPU_WORKFLOW:
        if len(FULL_GPU_DEVICES) != 2 or FULL_GPU_ASSEMBLY_WORKERS % 2:
            raise ValueError(
                "This strong/weak demonstration requires two GPUs and an even CPU count"
            )
        # Warm the same displayed native form kernels; all measured pools remain fresh.
        audit_local(
            LocalProvider(
                macro_mesh(),
                max(4, FULL_GPU_FINE_N // 4),
                data=FULL_GPU_DATA,
                trace_degree=trace_degree,
                quadrature_degree=quadrature_degree,
            )
        )
        configurations = (
            ("strong", 1, FULL_GPU_ASSEMBLY_WORKERS, FULL_GPU_DEVICES[:1]),
            ("strong", 1, FULL_GPU_ASSEMBLY_WORKERS, FULL_GPU_DEVICES),
            ("weak", 1, FULL_GPU_ASSEMBLY_WORKERS // 2, FULL_GPU_DEVICES[:1]),
            ("weak", 2, FULL_GPU_ASSEMBLY_WORKERS, FULL_GPU_DEVICES),
        )
        for solver in FULL_GPU_SOLVERS:
            for study, length, cpu_workers, devices in configurations:
                record, system, solution, macro = run_full_gpu_mhm(
                    FULL_GPU_FINE_N,
                    devices=devices,
                    assembly_workers=cpu_workers,
                    solver=solver,
                    data=FULL_GPU_DATA,
                    length=length,
                    trace_degree=trace_degree,
                    quadrature_degree=quadrature_degree,
                )
                record["study"] = study
                record["controls"] = validate_gpu_responses(
                    [
                        (f"cell-{index:03d}", response)
                        for index, response in enumerate(system.responses)
                    ]
                )
                defects = []
                for response, field in zip(system.responses, solution.fields, strict=True):
                    problem = response.problem
                    vector = (
                        problem.matrix @ field
                        + problem.coupling @ solution.trace[problem.trace_dofs]
                        - problem.load
                    )
                    relative = np.linalg.norm(vector) / max(np.linalg.norm(problem.load), 1e-300)
                    assert relative <= 1e-10
                    defects.append(float(relative))
                record["reconstructed_physical_equation_max_relative_residual"] = max(defects)
                record["physical_errors"] = mhm_physical_errors(
                    system, solution, data=FULL_GPU_DATA, order=7
                )
                # Independent CPU execution of the same forms checks the actual field.
                _, cpu_system, cpu_solution, _ = run_mhm(
                    FULL_GPU_FINE_N,
                    workers=min(2, cpu_workers),
                    solver="scipy",
                    length=length,
                    data=FULL_GPU_DATA,
                    trace_degree=trace_degree,
                    quadrature_degree=quadrature_degree,
                )
                differences = []
                for gpu_response, cpu_response, gpu_field, cpu_field in zip(
                    system.responses,
                    cpu_system.responses,
                    solution.fields,
                    cpu_solution.fields,
                    strict=True,
                ):
                    np.testing.assert_array_equal(
                        gpu_response.problem.matrix.data, cpu_response.problem.matrix.data
                    )
                    np.testing.assert_allclose(
                        gpu_response.retained_basis,
                        cpu_response.retained_basis,
                        rtol=1e-10,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(gpu_field, cpu_field, rtol=1e-10, atol=1e-12)
                    differences.append(float(np.max(np.abs(gpu_field - cpu_field))))
                record["cpu_scipy_field_maximum_absolute_difference"] = max(differences)
                label = f"{solver}-{study}-L{length}-cpu{cpu_workers}-gpu{len(devices)}"
                record["archive"] = archive_fields(
                    system, solution, OUTPUT / (label + "-executed-fields.npz")
                )
                full_gpu_reproduction.append(record)
                print(
                    {
                        "configuration": label,
                        "complete_workflow_seconds": record["complete_workflow_seconds"],
                        "pressure_L2": record["physical_errors"]["pressure_L2"],
                        "flux_L2": record["physical_errors"]["flux_L2"],
                        "physical_residual": max(defects),
                    }
                )
        (OUTPUT / "full_cpu_gpu_workflow.json").write_text(
            json.dumps(full_gpu_reproduction, indent=2) + "\n"
        )

    ACCELERATED_CAMPAIGN = (
        ROOT / "benchmarks/results/execution/introduction-3d-accelerators-20261005"
    )

    ACCELERATED_RESULTS = ACCELERATED_CAMPAIGN / "results.json"

    accelerated_publication = None

    if ACCELERATED_RESULTS.exists():
        publication_bytes = ACCELERATED_RESULTS.read_bytes()
        accelerated_publication = json.loads(publication_bytes)
        required_sections = {
            "metadata",
            "acquisitions",
            "configurations",
            "study_membership",
            "plots",
        }
        if not required_sections.issubset(accelerated_publication):
            raise ValueError("The accelerator record is missing its declared sections")
        print("Published JSON SHA256:", hashlib.sha256(publication_bytes).hexdigest())
        print(json.dumps(accelerated_publication["metadata"], indent=2))
        print("Recorded configurations:")
        print(json.dumps(accelerated_publication["configurations"], indent=2))
        print("Study membership:")
        print(json.dumps(accelerated_publication["study_membership"], indent=2))
        print("Acquisitions:", len(accelerated_publication["acquisitions"]))
    else:
        print("No accelerator campaign record is installed; no measurements are inferred.")

    if accelerated_publication is not None:
        from IPython.display import Image, Markdown, display

        publication_figures = published_accelerator_figures(accelerated_publication)
        print("Published figures:", len(publication_figures))
        for image_path, caption in publication_figures:
            display(Markdown(caption))
            display(Image(filename=str(image_path)))
    return reproduction
