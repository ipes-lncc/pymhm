"""Classical weakly symmetric RT/Q elasticity for the pinned HPC4E material section.

This original DOLFINx/UFL assembly has globally conforming stress and no MHM
trace restriction. RT degree k is mathematical normal degree; Basix uses k+1.
Displacement is discontinuous Qk squared and independent rotation is total Pk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from pymhm.io.provenance import current_source_manifest

if __package__:
    from .hpc4e_data import BOUNDS, DATA_DIRECTORY, LENGTH_SCALE, STRESS_SCALE, HPC4EData, load_data
else:
    from hpc4e_data import BOUNDS, DATA_DIRECTORY, LENGTH_SCALE, STRESS_SCALE, HPC4EData, load_data

from pymhm._legacy.models.elasticity.stress_tensor import _rotation_basis
from pymhm.fem.hdiv.tensor_rt import tensor_rt_basis
from pymhm.fem.scalar.quadrilateral import quadrilateral_quadrature
from pymhm.meshes.cartesian import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/hpc4e"
ARCHIVES = ROOT / "build/results/hpc4e"


@dataclass(frozen=True)
class ReferenceField:
    """Canonical physical-rectangle RTk/Qk/Pk coefficients, with x varying fastest."""

    stress: np.ndarray
    displacement: np.ndarray
    rotation: np.ndarray
    nx: int
    ny: int
    degree: int
    bounds: tuple[float, float, float, float] = BOUNDS

    def __post_init__(self) -> None:
        """Validate finite canonical arrays before evaluating archived physical fields."""
        if self.degree not in (1, 2) or min(self.nx, self.ny) < 1:
            raise ValueError("positive grid counts and RT degree one or two required")
        k, count = self.degree, self.nx * self.ny
        shapes = (
            (count, 2 * (k + 1) * (k + 2), 2),
            (count, (k + 1) ** 2, 2),
            (count, (k + 1) * (k + 2) // 2),
        )
        for values, shape in zip(
            (self.stress, self.displacement, self.rotation), shapes, strict=True
        ):
            if (
                np.shape(values) != shape
                or np.iscomplexobj(values)
                or not np.isfinite(values).all()
            ):
                raise ValueError("coefficient arrays must have the finite real canonical shapes")
        if (
            len(self.bounds) != 4
            or not np.isfinite(self.bounds).all()
            or (self.bounds[1] <= self.bounds[0] or self.bounds[3] <= self.bounds[2])
        ):
            raise ValueError("ordered finite physical bounds required")

    def evaluate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate each owning cell polynomial without averaging interface values."""
        if np.iscomplexobj(points):
            raise ValueError("evaluation points must be finite real coordinate pairs")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("evaluation points must be finite coordinate pairs")
        spacing = np.array(
            [
                (self.bounds[1] - self.bounds[0]) / self.nx,
                (self.bounds[3] - self.bounds[2]) / self.ny,
            ]
        )
        scaled = (points - np.array(self.bounds)[[0, 2]]) / spacing
        if np.any(scaled < -1e-11) or np.any(scaled > [self.nx + 1e-11, self.ny + 1e-11]):
            raise ValueError("evaluation points lie outside the reference section")
        ids = np.clip(np.floor(scaled).astype(int), 0, [self.nx - 1, self.ny - 1])
        local = scaled - ids
        cell = ids[:, 0] + self.nx * ids[:, 1]
        one = CartesianMacroMesh(bounds=(0.0, spacing[0], 0.0, spacing[1]))
        basis, _, scalar = tensor_rt_basis(one, self.degree, 0, local)
        rotation = _rotation_basis(self.degree, local)
        return (
            np.einsum("qia,qib->qba", basis[0], self.stress[cell]),
            np.einsum("qi,qia->qa", scalar, self.displacement[cell]),
            np.einsum("qi,qi->q", rotation, self.rotation[cell]),
        )

    def save(self, path: Path) -> None:
        """Store canonical coefficients and the complete physical grid description."""
        np.savez_compressed(
            path,
            stress=self.stress,
            displacement=self.displacement,
            rotation=self.rotation,
            nx=self.nx,
            ny=self.ny,
            degree=self.degree,
            bounds=self.bounds,
        )


def load_field(path: Path) -> ReferenceField:
    """Reload a classical field independently of native finite-element numbering."""
    with np.load(path) as data:
        return ReferenceField(
            data["stress"],
            data["displacement"],
            data["rotation"],
            int(data["nx"]),
            int(data["ny"]),
            int(data["degree"]),
            tuple(data["bounds"]),
        )


def difference(fine: ReferenceField, coarse: ReferenceField, *, order: int = 5) -> dict[str, float]:
    """Integrate physical L2 differences on the common nested rectangular partition.

    Relative quantities use the fine field norm as denominator. Tensor Gauss
    order five exactly integrates squared RT2 fields, Q2 displacement and P2
    rotation on each rectangle; no samples on interfaces enter these norms.
    """
    if fine.bounds != coarse.bounds or fine.nx % coarse.nx or fine.ny % coarse.ny:
        raise ValueError("matching physical bounds and nested rectangular grids required")
    if order < max(fine.degree, coarse.degree) + 2:
        raise ValueError("quadrature must integrate the squared RT polynomials exactly")
    reference, weights = quadrilateral_quadrature(order)
    spacing = np.array(
        [(fine.bounds[1] - fine.bounds[0]) / fine.nx, (fine.bounds[3] - fine.bounds[2]) / fine.ny]
    )
    origin = np.array(fine.bounds)[[0, 2]]
    sums = np.zeros((2, 3))
    for begin in range(0, fine.nx * fine.ny, 512):
        ids = np.arange(begin, min(begin + 512, fine.nx * fine.ny))
        xy = np.column_stack((ids % fine.nx, ids // fine.nx))
        points = origin + (xy[:, None, :] + reference[None]) * spacing
        f, c = fine.evaluate(points.reshape(-1, 2)), coarse.evaluate(points.reshape(-1, 2))
        for component, (fv, cv) in enumerate(zip(f, c, strict=True)):
            actual = fv.reshape(len(ids), len(reference), -1)
            delta = (fv - cv).reshape(actual.shape)
            sums[:, component] += np.array(
                [
                    np.einsum("tqi,tqi,q->", delta, delta, weights),
                    np.einsum("tqi,tqi,q->", actual, actual, weights),
                ]
            ) * np.prod(spacing)
    result = {}
    for name, (error, norm) in zip(
        ("stress", "displacement", "rotation"), np.sqrt(sums).T, strict=True
    ):
        result[f"{name}_l2"] = float(error)
        result[f"{name}_reference_l2"] = float(norm)
        result[f"{name}_relative"] = (
            float(error / norm) if norm else (0.0 if error == 0 else float("inf"))
        )
    return result


def _export(
    domain: Any, function: Any, nx: int, ny: int, degree: int
) -> tuple[ReferenceField, float]:
    """Fit native physical values to the exchange basis and check independent sample points."""
    import dolfinx

    spacing = np.array([1 / nx, 0.45 / ny])
    one = CartesianMacroMesh(bounds=(0.0, spacing[0], 0.0, spacing[1]))
    points, _ = quadrilateral_quadrature(degree + 2)
    vectors, _, scalar = tensor_rt_basis(one, degree, 0, points)
    rotation = _rotation_basis(degree, points)
    vector_matrix = vectors[0].transpose(0, 2, 1).reshape(-1, vectors.shape[2])
    inverse = [np.linalg.pinv(matrix) for matrix in (vector_matrix, scalar, rotation)]
    count = domain.topology.index_map(2).size_local
    cells = np.arange(count, dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(domain, 2, cells)[:, :2]
    origins = np.rint(midpoints / spacing - 0.5) * spacing
    xy = np.rint(origins / spacing).astype(int)
    canonical = xy[:, 0] + nx * xy[:, 1]
    if len(np.unique(canonical)) != count:
        raise ValueError("native quadrilateral cells do not match the declared Cartesian grid")
    functions = [function.sub(i).collapse() for i in range(5)]
    stress = np.empty((count, vectors.shape[2], 2))
    displacement = np.empty((count, scalar.shape[1], 2))
    rotations = np.empty((count, rotation.shape[1]))
    for begin in range(0, count, 512):
        selection = cells[begin : begin + 512]
        physical = origins[selection, None] + points[None] * spacing
        xyz = np.column_stack((physical.reshape(-1, 2), np.zeros(physical.size // 2)))
        owners = np.repeat(selection, len(points))
        for component in range(5):
            values = functions[component].eval(xyz, owners).reshape(len(selection), len(points), -1)
            if component < 2:
                stress[selection, :, component] = values.reshape(len(selection), -1) @ inverse[0].T
            elif component < 4:
                displacement[selection, :, component - 2] = values[..., 0] @ inverse[1].T
            else:
                rotations[selection] = values[..., 0] @ inverse[2].T
    pieces = domain.comm.allgather((canonical, stress, displacement, rotations))
    numbering = np.concatenate([piece[0] for piece in pieces])
    if not np.array_equal(np.sort(numbering), np.arange(nx * ny)):
        raise ValueError("owned native cells do not cover the declared grid exactly once")
    order = np.argsort(numbering)
    arrays = [np.concatenate([piece[i] for piece in pieces])[order] for i in range(1, 4)]
    result = ReferenceField(*arrays, nx, ny, degree)
    del pieces, arrays
    check = np.array([[0.173, 0.231], [0.659, 0.417], [0.389, 0.813]])
    error, scale = 0.0, 0.0
    for begin in range(0, count, 2048):
        selection = cells[begin : begin + 2048]
        physical = (origins[selection, None] + check[None] * spacing).reshape(-1, 2)
        xyz = np.column_stack((physical, np.zeros(len(physical))))
        owners = np.repeat(selection, len(check))
        actual = result.evaluate(physical)
        for component in range(5):
            native = functions[component].eval(xyz, owners)
            expected = (
                actual[0][:, component]
                if component < 2
                else actual[1][:, component - 2, None]
                if component < 4
                else actual[2][:, None]
            )
            error = max(error, float(np.max(abs(native - expected))))
            scale = max(scale, float(np.max(abs(native))))
    from mpi4py import MPI

    error = domain.comm.allreduce(error, op=MPI.MAX)
    scale = domain.comm.allreduce(scale, op=MPI.MAX)
    relative = error / max(scale, np.finfo(float).tiny)
    if relative > 2e-10:
        raise ValueError(f"physical coefficient conversion failed: {relative}")
    return result, relative


def _own_petsc(resources: ExitStack, handle: Any) -> Any:
    """Register an owned PETSc handle immediately, including failure paths."""
    resources.callback(handle.destroy)
    return handle


def _restore_petsc_option(options: Any, key: str, previous: dict[str, str]) -> None:
    """Restore an overwritten caller option, or remove only the driver-created key."""
    if key in previous:
        options[key] = previous[key]
    else:
        del options[key]


def sparsity_estimate(nx: int, ny: int, degree: int) -> dict[str, int]:
    """Count native RTk/Qk/Pk unknowns and the uncompressed UFL element graph.

    Each interior edge identifies 2(k+1) stress coefficients. Its dense
    overlap is counted twice by the two adjacent element blocks and is
    therefore subtracted once. This count includes structural numerical zeros.
    """
    if degree not in (1, 2) or min(nx, ny) < 1:
        raise ValueError("positive grid counts and RT degree one or two required")
    local = 4 * (degree + 1) * (degree + 2) + 2 * (degree + 1) ** 2
    local += (degree + 1) * (degree + 2) // 2
    shared = 2 * (degree + 1)
    interior_faces = (nx - 1) * ny + nx * (ny - 1)
    return {
        "native_dofs": local * nx * ny - shared * interior_faces,
        "structural_nonzeros": local**2 * nx * ny - shared**2 * interior_faces,
    }


def _solve(
    nx: int,
    ny: int,
    degree: int,
    data: HPC4EData | None,
    *,
    threads: int = 8,
    workspace_limit_mb: int = 45000,
    factorization: str = "lu",
    solver_python: Path | None = None,
    equilibration: Literal["none", "symmetric"] = "none",
    comm: Any = None,
    refinement_precision: Literal["double", "extended"] = "double",
    out_of_core_directory: Path | None = None,
) -> tuple[ReferenceField, dict]:
    """Assemble native stress equilibrium and solve the original mixed equations.

    ``data=None`` selects a represented polynomial patch with nonzero side/top
    tractions, zero bottom displacement and constant Lamé coefficients. Dataset
    solves require every material pixel boundary to lie on the classical grid.
    """
    import resource

    import basix
    import basix.ufl
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    if degree not in (1, 2) or min(nx, ny, threads, workspace_limit_mb) < 1:
        raise ValueError("positive grid/thread counts and RT degree one or two required")
    if data is not None and (nx % data.young.shape[0] or ny % data.young.shape[1]):
        raise ValueError("the classical grid must align with every material pixel")
    if factorization != "pypardiso-symmetric-matching" and not PETSc.Sys.hasExternalPackage(
        "mumps"
    ):
        raise RuntimeError("the independent reference requires PETSc/MUMPS")
    comm = MPI.COMM_SELF if comm is None else comm
    distributed = comm.size > 1
    estimate = sparsity_estimate(nx, ny, degree)
    index_limit = np.iinfo(PETSc.IntType).max
    if estimate["native_dofs"] > index_limit:
        raise ValueError("the global dimension requires a 64-bit PETSc index build")
    if not distributed and estimate["structural_nonzeros"] > index_limit:
        raise ValueError("serial CSR exceeds PETSc index width; use explicit MPI distribution")
    if distributed and factorization == "pypardiso-symmetric-matching":
        raise ValueError("distributed reference requires native MUMPS LU or LDLt")
    with ExitStack() as resources:
        started = perf_counter()
        domain = dolfinx.mesh.create_rectangle(
            comm,
            np.array([[0.0, 0.0], [1.0, 0.45]]),
            [nx, ny],
            cell_type=dolfinx.mesh.CellType.quadrilateral,
        )
        rt = basix.ufl.element("RT", "quadrilateral", degree + 1)
        scalar = basix.ufl.element(
            "DG", "quadrilateral", degree, lagrange_variant=basix.LagrangeVariant.equispaced
        )
        rotation = basix.ufl.element(
            "DPC",
            "quadrilateral",
            degree,
            discontinuous=True,
            dpc_variant=basix.DPCVariant.simplex_equispaced,
        )
        space = dolfinx.fem.functionspace(
            domain, basix.ufl.mixed_element([rt, rt, scalar, scalar, rotation])
        )
        sx, sy, ux, uy, rot = ufl.TrialFunctions(space)
        tx, ty, vx, vy, w = ufl.TestFunctions(space)
        sigma, tau = ufl.as_tensor((sx, sy)), ufl.as_tensor((tx, ty))
        if data is None:
            lam, mu = 2.0, 1.0
            force = ufl.as_vector((0.0, -11.0))
        else:
            dg = dolfinx.fem.functionspace(domain, ("DG", 0))
            centers = dg.tabulate_dof_coordinates()[:, :2]
            lam, mu, fy = [dolfinx.fem.Function(dg) for _ in range(3)]
            fields = data.fields()
            lam.x.array[:] = fields[0](centers)
            mu.x.array[:] = fields[1](centers)
            fy.x.array[:] = fields[2](centers)[:, 1]
            force = ufl.as_vector((0.0, fy))
        compliance = (sigma - lam / (2 * mu + 2 * lam) * ufl.tr(sigma) * ufl.Identity(2)) / (2 * mu)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * degree + 4})
        form = dolfinx.fem.form(
            (
                ufl.inner(compliance, tau)
                + ufl.inner(ufl.as_vector((ux, uy)), ufl.div(tau))
                + ufl.inner(ufl.div(sigma), ufl.as_vector((vx, vy)))
                + rot * (tau[0, 1] - tau[1, 0])
                + w * (sigma[0, 1] - sigma[1, 0])
            )
            * dx
        )
        linear = dolfinx.fem.form(-ufl.inner(force, ufl.as_vector((vx, vy))) * dx)
        original = _own_petsc(resources, dolfinx.fem.petsc.assemble_matrix(form))
        original.assemble()
        if distributed and comm.rank == 0:
            print(f"Native UFL matrix assembled on {comm.size} MPI ranks", flush=True)
        matrix_storage = None
        if distributed:
            if __package__:
                from .hpc4e_parallel import checked_solve, compact_matrix, symmetric_equilibration
            else:
                from hpc4e_parallel import checked_solve, compact_matrix, symmetric_equilibration
            original, matrix_storage = compact_matrix(original, resources)
            if comm.rank == 0:
                print(f"Verified and compacted native matrix: {matrix_storage}", flush=True)
        forcing = _own_petsc(resources, dolfinx.fem.petsc.assemble_vector(linear))
        forcing.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        owned_size = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
        facets = dolfinx.mesh.locate_entities_boundary(
            domain,
            1,
            lambda x: (abs(x[0]) < 1e-12) | (abs(x[0] - 1) < 1e-12) | (abs(x[1] - 0.45) < 1e-12),
        )
        prescribed = []
        g = _own_petsc(resources, original.createVecRight())
        g.set(0)
        for row in range(2):
            collapsed, _ = space.sub(row).collapse()
            value = dolfinx.fem.Function(collapsed)
            if data is None:
                value.interpolate(
                    (lambda x: np.stack((8 * x[1], x[0])))
                    if row == 0
                    else (lambda x: np.stack((x[0], 10 * x[1])))
                )
            parent_dofs, collapsed_dofs = dolfinx.fem.locate_dofs_topological(
                (space.sub(row), collapsed), 1, facets
            )
            owned = parent_dofs < owned_size
            g.array[parent_dofs[owned]] = value.x.array[collapsed_dofs[owned]]
            prescribed.extend(parent_dofs[owned])
        prescribed = np.unique(prescribed).astype(PETSc.IntType)
        rhs = _own_petsc(resources, original.createVecLeft())
        rhs.array[:] = forcing.array[:owned_size]
        matrix = _own_petsc(resources, original.copy())
        global_prescribed = prescribed + matrix.getOwnershipRange()[0]
        matrix.zeroRowsColumns(global_prescribed, diag=1.0, x=g, b=rhs)
        if distributed:
            # Exact symmetric row/column elimination preserves the checked UFL symmetry.
            matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        vector = _own_petsc(resources, matrix.createVecRight())
        balanced_matrix = None
        diagonal = None
        if distributed and equilibration == "symmetric":
            if comm.rank == 0:
                print("Applying five distributed Ruiz congruences", flush=True)
            diagonal = symmetric_equilibration(matrix, resources)
        elif equilibration == "symmetric" and factorization != "pypardiso-symmetric-matching":
            from scipy import sparse

            from pymhm.linalg.linear import LinearFactorization, _symmetric_equilibration

            pointer, indices, values = matrix.getValuesCSR()
            original_csr = sparse.csr_matrix((values, indices, pointer), shape=matrix.getSize())
            balanced, diagonal = _symmetric_equilibration(original_csr)
            balanced_matrix = PETSc.Mat().createAIJ(
                size=balanced.shape,
                csr=(balanced.indptr, balanced.indices, balanced.data),
                comm=MPI.COMM_SELF,
            )
            _own_petsc(resources, balanced_matrix)
            balanced_matrix.assemble()
            del balanced
        factor_matrix = matrix if balanced_matrix is None else balanced_matrix
        ksp = _own_petsc(resources, PETSc.KSP().create(comm))
        ksp.setOperators(factor_matrix)
        ksp.setType("preonly")
        if factorization == "ldlt" and distributed:
            if factor_matrix.isSymmetricKnown() != (True, True):
                raise ValueError("symmetric MUMPS requires the verified UFL operator")
            factor_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            factor_matrix.setOption(PETSc.Mat.Option.SPD, False)
        elif factorization == "ldlt":
            from scipy import sparse

            from pymhm.linalg.linear import _hermitian

            pointer, indices, values = factor_matrix.getValuesCSR()
            _hermitian(
                sparse.csr_matrix((values, indices, pointer), shape=factor_matrix.getSize()),
                "symmetric MUMPS",
            )
            factor_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            factor_matrix.setOption(PETSc.Mat.Option.SPD, False)
        ksp.getPC().setType("cholesky" if factorization == "ldlt" else "lu")
        ksp.getPC().setFactorSolverType("mumps")
        prefix = "hpc4e_reference_"
        ksp.setOptionsPrefix(prefix)
        options = PETSc.Options()
        settings = {
            "mat_mumps_icntl_7": 5,
            "mat_mumps_icntl_14": 100,
            "mat_mumps_icntl_10": 5,
            "mat_mumps_icntl_23": workspace_limit_mb,
            "mat_mumps_icntl_22": int(out_of_core_directory is not None),
        }
        previous_options = options.getAll()
        for key, value in settings.items():
            resources.callback(_restore_petsc_option, options, prefix + key, previous_options)
            options[prefix + key] = value
        if out_of_core_directory is not None:
            key = prefix + "mat_mumps_ooc_tmpdir"
            resources.callback(_restore_petsc_option, options, key, previous_options)
            options[key] = str(out_of_core_directory.resolve())
        ksp.setFromOptions()
        assembled = perf_counter()
        print(
            f"Assembled classical RT{degree}/Q{degree}/P{degree}: "
            f"{matrix.getSize()[0]:,} DOFs; {factorization} starts",
            flush=True,
        )
        factorization_record = None
        distributed_history = None
        if distributed:
            vector, distributed_history = checked_solve(
                ksp,
                original,
                rhs,
                forcing,
                prescribed,
                g,
                np.ones(owned_size) if diagonal is None else diagonal,
                resources,
                refinement_precision=refinement_precision,
            )
        elif factorization == "pypardiso-symmetric-matching":
            import subprocess
            from tempfile import TemporaryDirectory

            from scipy import sparse

            with TemporaryDirectory(prefix="pymhm-hpc4e-") as temporary:
                algebra = Path(temporary)
                pointer, indices, values = matrix.getValuesCSR()
                exported = sparse.csr_matrix((values, indices, pointer), shape=matrix.getSize())
                sparse.save_npz(algebra / "matrix.npz", exported, compressed=False)
                np.save(algebra / "rhs.npy", rhs.array, allow_pickle=False)
                subprocess.run(
                    [
                        str(solver_python),
                        str(ROOT / "examples/solve_hpc4e_algebra.py"),
                        str(algebra / "matrix.npz"),
                        str(algebra / "rhs.npy"),
                        str(algebra / "solution.npy"),
                        str(algebra / "report.json"),
                        "--threads",
                        str(threads),
                        "--solver",
                        factorization,
                        "--equilibration",
                        equilibration,
                    ],
                    check=True,
                )
                vector.array[:] = np.load(algebra / "solution.npy", allow_pickle=False)
                factorization_record = json.loads((algebra / "report.json").read_text())
        elif diagonal is not None:

            def solve_scaled(forcing_array: np.ndarray) -> np.ndarray:
                """Reuse native factors through the shared original-equation refinement gate."""
                with ExitStack() as vectors:
                    scaled_rhs = _own_petsc(vectors, rhs.duplicate())
                    scaled_rhs.array[:] = diagonal * forcing_array
                    state = _own_petsc(vectors, vector.duplicate())
                    ksp.solve(scaled_rhs, state)
                    if ksp.getConvergedReason() <= 0:
                        raise RuntimeError(f"MUMPS failed with reason {ksp.getConvergedReason()}")
                    return diagonal * state.array.copy()

            with LinearFactorization(
                original_csr, solve_scaled, lambda: None, "petsc", 1e-11, 0.0
            ) as factors:
                vector.array[:] = factors.solve(
                    rhs.array, refinement_precision=refinement_precision
                )
        else:
            ksp.solve(rhs, vector)
        solved = perf_counter()
        if factorization_record is None and ksp.getConvergedReason() <= 0:
            raise RuntimeError(f"MUMPS failed with reason {ksp.getConvergedReason()}")
        residual = _own_petsc(resources, original.createVecLeft())
        original.mult(vector, residual)
        defect = residual.array - forcing.array[:owned_size]
        free = np.ones(len(defect), dtype=bool)
        free[prescribed] = False
        relative = np.sqrt(comm.allreduce(float(np.sum(defect[free] ** 2)), op=MPI.SUM)) / max(
            np.sqrt(
                comm.allreduce(float(np.sum(forcing.array[:owned_size][free] ** 2)), op=MPI.SUM)
            ),
            np.finfo(float).tiny,
        )
        if relative > 1e-10:
            raise ValueError(f"original free-equation relative residual exceeds 1e-10: {relative}")
        function = dolfinx.fem.Function(space)
        function.x.array[:owned_size] = vector.array
        function.x.scatter_forward()
        sxh, syh, uxh, uyh, _ = ufl.split(function)
        sigma_h = ufl.as_tensor((sxh, syh))
        displacement_h = ufl.as_vector((uxh, uyh))
        equilibrium = ufl.div(sigma_h) + force
        equilibrium_l2 = float(
            np.sqrt(
                max(
                    0.0,
                    comm.allreduce(
                        dolfinx.fem.assemble_scalar(
                            dolfinx.fem.form(ufl.inner(equilibrium, equilibrium) * dx)
                        ),
                        op=MPI.SUM,
                    ),
                )
            )
        )
        force_l2 = float(
            np.sqrt(
                comm.allreduce(
                    dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(force, force) * dx)),
                    op=MPI.SUM,
                )
            )
        )
        compliance_h = (sigma_h - lam / (2 * mu + 2 * lam) * ufl.tr(sigma_h) * ufl.Identity(2)) / (
            2 * mu
        )
        energy = float(
            comm.allreduce(
                dolfinx.fem.assemble_scalar(
                    dolfinx.fem.form(ufl.inner(compliance_h, sigma_h) * dx)
                ),
                op=MPI.SUM,
            )
        )
        work = float(
            comm.allreduce(
                dolfinx.fem.assemble_scalar(
                    dolfinx.fem.form(ufl.inner(force, displacement_h) * dx)
                ),
                op=MPI.SUM,
            )
        )
        _, rotation_map = space.sub(4).collapse()
        rotation_map = np.asarray(rotation_map)
        weak_symmetry_max = float(
            comm.allreduce(
                float(np.max(abs(defect[rotation_map[rotation_map < owned_size]]), initial=0)),
                op=MPI.MAX,
            )
        )
        if equilibrium_l2 > 1e-8 * force_l2:
            raise ValueError(
                f"strong cell equilibrium failed: relative {equilibrium_l2 / force_l2}"
            )
        energy_relative = abs(energy - work) / max(abs(energy), abs(work), np.finfo(float).tiny)
        if data is not None and energy_relative > 1e-8:
            raise ValueError(f"compliance energy/body work identity failed: {energy_relative}")
        field, conversion = _export(domain, function, nx, ny, degree)
        peak_rss = comm.allgather(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**3 if sys.platform == "darwin" else 1024**2)
        )
        metadata = dict(
            nx=nx,
            ny=ny,
            degree=degree,
            native_dofs=vector.getSize(),
            free_dofs=int(comm.allreduce(int(free.sum()), op=MPI.SUM)),
            mpi_ranks=comm.size,
            petsc_index_bits=np.iinfo(PETSc.IntType).bits,
            matrix_storage=matrix_storage,
            sparsity_estimate=estimate,
            distributed_refinement_history=distributed_history,
            refinement_precision=refinement_precision,
            residual=float(relative),
            conversion_relative=conversion,
            equilibrium_l2=equilibrium_l2,
            equilibrium_relative=equilibrium_l2 / force_l2,
            weak_symmetry_moment_max=weak_symmetry_max,
            compliance_energy=energy,
            body_work=work,
            energy_work_relative=energy_relative if data is not None else None,
            peak_rss_gib=sum(peak_rss),
            peak_rss_per_rank_gib=peak_rss,
            peak_rss_convention="sum of each rank peak; an upper bound on concurrent owned RSS",
            assembly_seconds=assembled - started,
            factor_solve_seconds=solved - assembled,
            total_seconds=perf_counter() - started,
            quadrature_degree=2 * degree + 4,
            formulation=(
                "Globally conforming row-wise RTk, DG Qk displacement, "
                "DPC total Pk rotation; no MHM restriction"
            ),
            bottom="zero displacement in the natural mixed boundary functional",
            sides_top="prescribed zero physical traction"
            if data is not None
            else "exact nonzero polynomial traction",
            bounds=list(BOUNDS),
            length_scale_m=LENGTH_SCALE,
            stress_scale_Pa=STRESS_SCALE,
            software=dict(
                dolfinx=dolfinx.__version__,
                basix=basix.__version__,
                petsc=PETSc.Sys.getVersion(),
                python=platform.python_version(),
            ),
            requested_threads=threads,
            factorization=factorization,
            algebra_process=factorization_record,
            equilibration=equilibration,
            algebra_refinement_rtol=1e-11 if diagonal is not None or distributed else None,
            mumps_options=settings,
            factor_storage="out-of-core" if out_of_core_directory is not None else "in-core",
        )
        return field, metadata


def solve(
    nx: int,
    ny: int,
    degree: int,
    data: HPC4EData | None,
    *,
    threads: int = 8,
    workspace_limit_mb: int = 45000,
    factorization: str = "lu",
    solver_python: Path | None = None,
    equilibration: Literal["none", "symmetric"] = "none",
    comm: Any = None,
    refinement_precision: Literal["double", "extended"] = "double",
    out_of_core_directory: Path | None = None,
) -> tuple[ReferenceField, dict]:
    """Assemble and validate the native mixed equations with controlled native pools.

    ``factorization='ldlt'`` selects symmetric indefinite MUMPS LDLT; the
    default is unsymmetric LU. The pypardiso-symmetric-matching option sends
    the CSR matrix to solve_hpc4e_algebra.py using solver_python; its explicit
    equilibration option consumes the shared factorizer without mixing Intel
    and PETSc libraries in one process. Native MUMPS also accepts symmetric
    equilibration, using the same congruence and original-residual checks.
    Its inner correction target is 1e-11 before the final float64 physical
    equation check at 1e-10; no physical acceptance tolerance is relaxed.
    An explicit ``out_of_core_directory`` selects MUMPS disk-backed factor
    storage; it changes neither the operator nor the original-equation gates.
    An explicit ``comm`` partitions native cells, matrix rows and MUMPS factors
    across MPI ranks. Only the validated canonical field archive is replicated.
    The distributed path discards exactly zero matrix entries, retains saddle
    diagonals and uses the same five Ruiz sweeps. ``refinement_precision`` may
    explicitly retain long-double correction digits before the float64 field
    is checked. Distributed forcing and diagnostics include MPI reductions.
    All routes use the identical UFL matrix,
    physical boundary elimination and original-equation residual criterion.
    Native libraries are loaded before applying ``threadpool_limits`` so
    BLAS implementations such as BLIS and OpenMP pools are both controlled
    throughout assembly, factorization and field reconstruction. The record
    reports the actual discovered pool settings, without machine-local paths.
    Owned PETSc objects are destroyed on success or failure, and temporary
    option values restore any caller settings when the solve exits.
    """
    import importlib

    from threadpoolctl import threadpool_info, threadpool_limits

    if refinement_precision not in {"double", "extended"}:
        raise ValueError("refinement_precision must be double or extended")
    if factorization not in ("lu", "ldlt", "pypardiso-symmetric-matching"):
        raise ValueError("factorization must be lu, ldlt or pypardiso-symmetric-matching")
    if equilibration not in ("none", "symmetric"):
        raise ValueError("equilibration must be none or symmetric")
    if factorization == "pypardiso-symmetric-matching" and (
        solver_python is None or not Path(solver_python).is_file()
    ):
        raise ValueError("the isolated solver requires an existing solver_python executable")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if out_of_core_directory is not None:
        if factorization == "pypardiso-symmetric-matching":
            raise ValueError("out-of-core storage requires native MUMPS")
        out_of_core_directory = Path(out_of_core_directory)
        out_of_core_directory.mkdir(parents=True, exist_ok=True)
    for name in ("dolfinx.fem.petsc", "petsc4py.PETSc"):
        importlib.import_module(name)
    with threadpool_limits(limits=threads):
        field, record = _solve(
            nx,
            ny,
            degree,
            data,
            threads=threads,
            workspace_limit_mb=workspace_limit_mb,
            factorization=factorization,
            solver_python=solver_python,
            equilibration=equilibration,
            comm=comm,
            refinement_precision=refinement_precision,
            out_of_core_directory=out_of_core_directory,
        )
        record["native_threadpools"] = [
            {key: value for key, value in pool.items() if key != "filepath"}
            for pool in threadpool_info()
        ]
    return field, record


def main() -> None:
    """Run one checked native reference and preserve full coefficients outside the source tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--degree", type=int, choices=(1, 2), default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--factorization", choices=("lu", "ldlt", "pypardiso-symmetric-matching"), default="lu"
    )
    parser.add_argument("--solver-python", type=Path)
    parser.add_argument("--equilibration", choices=("none", "symmetric"), default="none")
    parser.add_argument("--mpi", action="store_true", help="assemble and solve on MPI.COMM_WORLD")
    parser.add_argument("--refinement-precision", choices=("double", "extended"), default="double")
    parser.add_argument("--workspace-limit-mb", type=int, default=45000)
    parser.add_argument("--out-of-core-directory", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--patch", action="store_true")
    args = parser.parse_args()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD if args.mpi else MPI.COMM_SELF
    if MPI.COMM_WORLD.size > 1 and not args.mpi:
        raise ValueError("multiple MPI ranks require the explicit --mpi option")
    data = None if args.patch else load_data(args.data_dir, download=args.download)
    sources = [
        Path(__file__),
        Path(__file__).with_name("hpc4e_data.py"),
        ROOT / "src/pymhm/fem/hdiv/tensor_rt.py",
        ROOT / "src/pymhm/_legacy/models/darcy/cartesian.py",
        ROOT / "src/pymhm/_legacy/models/elasticity/stress_tensor.py",
        ROOT / "src/pymhm/linalg/linear.py",
    ]
    if comm.size > 1:
        sources.append(Path(__file__).with_name("hpc4e_parallel.py"))
    if args.factorization == "pypardiso-symmetric-matching":
        sources.append(ROOT / "examples/solve_hpc4e_algebra.py")
    source_hashes = current_source_manifest(
        {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        }
    )
    snapshots = ARCHIVES / "acquisition-sources"
    snapshots.mkdir(parents=True, exist_ok=True)
    for path in sources if comm.rank == 0 else []:
        (snapshots / f"{source_hashes[path.relative_to(ROOT).as_posix()]}.py").write_bytes(
            path.read_bytes()
        )
    field, record = solve(
        args.nx,
        args.ny,
        args.degree,
        data,
        threads=args.threads,
        workspace_limit_mb=args.workspace_limit_mb,
        factorization=args.factorization,
        solver_python=args.solver_python,
        equilibration=args.equilibration,
        comm=comm,
        refinement_precision=args.refinement_precision,
        out_of_core_directory=args.out_of_core_directory,
    )
    changed = any(
        hashlib.sha256(path.read_bytes()).hexdigest()
        != source_hashes[path.relative_to(ROOT).as_posix()]
        for path in sources
    )
    if changed:
        raise RuntimeError("source files changed during this acquisition")
    if comm.rank != 0:
        return
    ARCHIVES.mkdir(exist_ok=True, parents=True)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    stem = f"classical-rt{args.degree}-{args.nx}x{args.ny}" + ("-patch" if args.patch else "")
    stem += "-mumps" if args.factorization == "ldlt" else ""
    stem += "-pardiso" if args.factorization == "pypardiso-symmetric-matching" else ""
    archive = ARCHIVES / f"{stem}.npz"
    field.save(archive)
    record.update(
        archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_sha256=source_hashes[Path(__file__).relative_to(ROOT).as_posix()],
        source_hashes=source_hashes,
        source_changed=False,
        material_sha256=None
        if data is None
        else current_source_manifest(
            {
                name: hashlib.sha256(getattr(data, name).tobytes()).hexdigest()
                for name in ("young", "poisson", "density")
            }
        ),
    )
    (OUTPUT / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record, flush=True)


if __name__ == "__main__":
    main()
