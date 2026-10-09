# Darcy

Darcy cases report pressure and the physical Darcy flux $q=-K\nabla p$.
Primal reconstructions provide a broken volume gradient; mixed H(div)
reconstructions also represent normal-flux moments. Their conservation
statements follow the actual approximation spaces on each case page.

## Two-dimensional cases

| Case | Method and question | Read the result |
| --- | --- | --- |
| Smooth multiscale tensor permeability | Primal MHM; separate macro, local and skeletal refinement | [Step-by-step formulation and fields](../tutorials/introduction/darcy_multiscale_convergence.md) |
| SPE10 Model 2, layer 36 | Heterogeneous Darcy; local refinement and a classical pressure/flux comparison | [Tutorial and permeability components](../tutorials/introduction/darcy_spe10_layer.md), [accepted initial increments](../cases/minimal-convergence.md#darcy-spe10) |
| Robin MH and three-field MH²M | Distinct Robin/conormal variables on analytical problems | [MH](../cases/minimal-convergence.md#scalar-mh), [MH²M](../cases/minimal-convergence.md#scalar-mh2m) |
| MsHHO | Cell/face pressure moments, five refinements and four contrasts | [Detailed fields and convergence](../cases/mshho.md) |
| Recursive MHM | Global problems whose local operators are themselves multiscale problems | [Independent leaf-system comparison and basis replay](../cases/nested.md) |
| Mixed tensor RT | Independently selected face and pressure degrees | [Face degree 1](../cases/minimal-convergence.md#scalar-tensor-rt-k1), [2](../cases/minimal-convergence.md#scalar-tensor-rt-k2), [3](../cases/minimal-convergence.md#scalar-tensor-rt-k3) |
| PGMHM | Enriched pressure and Darcy flux on a smooth problem | [Initial refinement study](../cases/minimal-convergence.md#darcy-pgmhm) |
| Unfitted Darcy | Trace refinement on a fixed macro/local mesh | [Accepted pressure and flux errors](../cases/minimal-convergence.md#darcy-unfitted) |
| Quarter-five spot and obstacle | Point wells and independent classical mixed references | [Physical fields and reference refinement](../cases/quarter-five-spot.md) |
| Periodic coefficient | Local sensitivity and independent conforming Q1 refinement | [Resolved and unresolved field diagnostics](../cases/minimal-convergence.md#periodic-local-reference) |

The SPE10 pressure increments decrease in the retained series, but its
Darcy-flux increments do not establish a resolved reference. The periodic
coefficient study similarly retains its unresolved H1 and local-sensitivity
diagnostics. These are visible results, without transferring smooth-case rates
to those heterogeneous cases.

## Three-dimensional cases

| Case | Method and question | Read the result |
| --- | --- | --- |
| MsHHO on affine tetrahedra and cubes | Executed local bases, original equations and physical fields | [Ten analytical controls](../cases/mshho3d.md) |
| Enriched mixed H(div) on tetrahedra and prisms | Normal degree, pressure/divergence degree and Piola conventions | [Twenty analytical controls](../cases/enriched-hdiv3d.md) |
| Mixed well | Annular geometry with mixed local spaces | [Two accepted resolution levels](../cases/minimal-convergence.md#darcy-mixedwell) |
| Multiscale anisotropic cube | Spawned workers, native workspace reuse, LU and AMG | [Strong/weak scaling and physical accuracy](../cases/darcy-3d-scalability.md) |
| CPU PARDISO and one/two GPUs | Complete costs, GPU AMG/direct solves and larger local grids | [Accelerator campaign](../cases/darcy-3d-accelerators.md) |

## Continue with a formulation

The [API overview](../tutorials/overview.md) starts from meshes and variational
equations. The [execution guides](../guides/index.md) keep one heterogeneous
Darcy operator fixed while changing CPU, MPI and GPU policies. The
[mesh guide](../meshing.md) explains supported geometry and material markers.

## References

- Christopher Harder, Diego Paredes and Frédéric Valentin (2013).
  *A family of Multiscale Hybrid-Mixed finite element methods for the Darcy
  equation with rough coefficients*. [DOI: 10.1016/j.jcp.2013.03.019](https://doi.org/10.1016/j.jcp.2013.03.019).
- Omar Durán, Philippe R. B. Devloo, Sônia M. Gomes and Frédéric Valentin (2019).
  *A multiscale hybrid method for Darcy’s problems using mixed finite element
  local solvers*. [DOI: 10.1016/j.cma.2019.05.013](https://doi.org/10.1016/j.cma.2019.05.013).
- Mike A. Christie and Martin J. Blunt (2001). *Tenth SPE Comparative Solution
  Project: A Comparison of Upscaling Techniques*.
  [DOI: 10.2118/72469-PA](https://doi.org/10.2118/72469-PA).
