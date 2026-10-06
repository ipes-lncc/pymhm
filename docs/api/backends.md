# Solvers and execution backends

Finite-element adapters, sparse solvers, parallel execution and separable operators.

[All API families](../api.md)

## Native spaces and interface forms

::: pymhm.backends.spaces
    options:
      show_source: false

::: pymhm.backends.traces
    options:
      show_source: false

## Direct form compilation

::: pymhm.backends.forms
    options:
      show_source: false

## Reusable local form workspaces

Compile a bundle of UFL forms once, then bind it to compatible native meshes,
spaces and coefficient data. A worker can retain its own bindings and assembly
buffers while updating geometry, Functions and Constants between local problems.
Each thread owns the mesh and coefficient data it mutates; sharing compiled code
uses explicit bindings to those separate native objects.
Each assembly returns independent arrays; material matrices, loads and solver
factors remain specific to each local problem. Native integration controls cover
single-mesh scalar, vector and mixed forms in 2D, tensor diffusion in 3D, and
constrained Neumann reconstruction in both dimensions.

::: pymhm.backends.workspace
    options:
      show_source: false

## Fixed hybrid local-form adapter

::: pymhm.backends.fenics
    options:
      show_source: false

## Solvers and execution

::: pymhm.linalg.linear
    options:
      show_source: false

::: pymhm.linalg.dynamics
    options:
      show_source: false

::: pymhm.execution.cpu
    options:
      show_source: false

::: pymhm.execution.mpi
    options:
      show_source: false

::: pymhm.execution.cuda
    options:
      show_source: false

::: pymhm.linalg.block
    options:
      show_source: false

::: pymhm._legacy.models.darcy.separable
    options:
      show_source: false

::: pymhm.linalg.separable
    options:
      show_source: false
