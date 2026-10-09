# Guides

These guides explain practical execution and mesh workflows. They keep one
simple heterogeneous Darcy problem fixed while changing the environment;
the [method tutorials](../tutorials/overview.md) explain how to design the
variational formulation itself.

| Start here | What you learn |
| --- | --- |
| [One common Darcy problem](heterogeneous-darcy.md) | Physical data, local/global forms, approximation spaces and correctness measurements shared by every execution guide |
| [Serial, threads and processes](cpu.md) | Ordered assembly, spawn-safe providers, native thread budgets and bounded scheduling |
| [Distributed MPI](mpi.md) | Rank-owned macrocells, distributed global rows and additive boundary data |
| [CUDA and multiple GPUs](gpu.md) | CPU UFL assembly, sparse/batched GPU local solves, device ownership and solver restrictions |
| [Generate and exchange meshes](../meshing.md) | Gmsh/Netgen, meshio, planar/volume cells, material IDs, boundary IDs and round trips |
| [Materials and marked meshes](materials.md) | Build piecewise material fields from tags, preserve internal interfaces and apply boundary groups explicitly |
| [Visualize and export fields](../visualization.md) | Named fields, macro meshes, independent interface values and PyVista |
| [Solvers and repeated loads](../solvers.md) | LU, Krylov/AMG applicability, reusable factors and offline/online solves |

## Reproducible environments

Use the repository's checked-in lockfile for the exact development profiles.
For an installed package, select the corresponding native dependencies first
and install `pymhm` in that environment; see [installation](../installation.md).

| Workflow | Locked Pixi profile | Required native capabilities |
| --- | --- | --- |
| Serial/thread/spawn UFL assembly | `introduction` | DOLFINx, UFL, Basix and a C compiler |
| PARDISO local factors | `introduction-intel` | The same FEM stack plus MKL/PyPardiso |
| Distributed MPI with UFL | `introduction` | DOLFINx, MPI, PETSc and MUMPS |
| CUDA local factors/batches | `introduction-gpu` | The FEM stack, CUDA-capable hardware, CuPy and cuDSS |
| Mesh generators and exchange | `meshing` | Gmsh, Netgen and meshio |

CPU scheduling is portable. The PETSc/MUMPS MPI guide uses Unix native
dependencies. CUDA solving requires a supported driver and device; ordinary
CPU assembly does not become GPU assembly by selecting a CUDA factorization.
The [Windows guide](../windows.md) separates portable, native-FEM and CUDA
installation requirements.

## Learn from the notebook

The [heterogeneous execution notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/darcy/heterogeneous_execution.ipynb)
shows the actual UFL callback, builds the mesh/interface/problem, compares
serial/thread/process coefficients and plots the physical fields. Its MPI and
CUDA sections execute the same problem when their declared dependencies are
available. This small problem verifies execution contracts; performance
claims come from the [recorded 3D campaigns](../gallery/darcy.md#three-dimensional-cases).
