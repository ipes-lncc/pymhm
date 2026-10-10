# Tutorials

Learn to express a multiscale method through its mathematics: **define meshes → write local equations → write global equations → assemble → solve → inspect physical fields and errors**. Numbering and supported geometric orientation are handled by the bound API; the variational signs, physical modes and boundary conditions stay explicit.

- [Start here: feature and API overview](overview.md). Learn the common workflow and available extension points.
- [Choose a method or reconstruction strategy](methods/index.md). Every lesson pairs mathematical terms with the executable declarations and ends with qualified rates or its relevant physical invariant.
- [Explore applications](../gallery/index.md) with their data, implementation and results.
- [Choose an execution environment or mesh workflow](../guides/index.md) for serial, process, MPI and GPU calculations.

The complete English source notebooks are downloadable from each tutorial. Native UFL lessons require the compatible DOLFINx environment described in [Getting Started](../getting-started/index.md); portable Basix/SciPy forms run with the installed core.

Each method lesson teaches the same sequence: choose the meshes and spaces,
write the local variational equations, declare the global coupling, assemble,
solve and reconstruct the physical fields. The refinement study at its end
states the expected rate, the hypotheses under which it applies and the
measured asymptotic behavior. Recovery and adaptive strategies use the
corresponding reconstruction error, conservation or error-versus-work study.
