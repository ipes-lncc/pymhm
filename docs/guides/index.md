# Guides

Use these pages when you already have a problem definition and want to change
one part of its setup. Each guide explains a particular task: configuring
execution, supplying coefficients, exchanging meshes or selecting a solver.
For the complete variational workflow, start with the
[API overview](../tutorials/overview.md). For physical applications and measured
performance, explore the [Gallery](../gallery/index.md).

| Task | Guide |
| --- | --- |
| Run local problems sequentially or with CPU workers | [Serial, threads and processes](cpu.md) |
| Distribute local work and the global solve | [Distributed MPI](mpi.md) |
| Assign local algebra to CUDA devices | [CUDA and multiple GPUs](gpu.md) |
| Change the material coefficient | [Heterogeneous coefficients](heterogeneous-darcy.md) |
| Associate materials and boundaries with mesh labels | [Materials and marked meshes](materials.md) |
| Generate, import and export planar or volume meshes | [Mesh generation and exchange](../meshing.md) |
| Configure Cartesian local and interface meshes | [Cartesian meshes and local spaces](../quadrilateral.md) |
| Write and compile native weak forms | [UFL and FEniCSx](../fenics.md) |
| Supply local equations or use an external solver | [External and custom providers](providers.md) |
| Supply a different trace basis or numbering | [Custom interface spaces](custom-interface.md) |
| Choose LU, Krylov or AMG and reuse factors | [Linear solvers](../solvers.md) |
| Reuse multiscale responses for changing loads | [Execution and repeated solves](../execution.md) |
| Evaluate, visualize and export reconstructed fields | [Visualization](../visualization.md) |

## Choose an environment

The package's core works without native FEM, MPI or accelerator imports.
Install the dependencies needed by the selected task as described in
[installation](../installation.md). In a checkout, use the checked-in Pixi
lockfile:

| Task | Locked profile | Additional capabilities |
| --- | --- | --- |
| UFL local assembly and CPU workers | `introduction` | DOLFINx, UFL, Basix and a C compiler |
| Native local PARDISO | `introduction-intel` | The FEM stack and MKL/PyPardiso |
| Distributed MPI with UFL | `introduction` | MPI, PETSc and MUMPS |
| CUDA local factorization | `introduction-gpu` | CuPy, cuDSS and a supported CUDA device/driver |
| Mesh generation and exchange | `meshing` | Gmsh, Netgen and meshio |

The [Windows guide](../windows.md) describes its native FEM, solver and CUDA
requirements. Distributed PETSc/MUMPS uses the supported Unix environments.
Selecting a GPU solver changes numerical algebra; it does not move UFL assembly
to the device.
