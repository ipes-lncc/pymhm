# Learn PyMHM by method

Start with the [feature and API overview](../overview.md), then choose a method. Every lesson follows **meshes → local problem → global problem → assembly → solve → fields and errors**, with the mathematical formulation next to its code. Ready-made physical solvers appear only as optional conveniences after the definition.

A final convergence study uses the observable and assumptions of its cited literature. Polynomial patches, reconstruction moments, conservation, estimator effectivity and adaptive error decrease use their meaningful invariants instead of an artificial fitted slope.

## Local and global formulations

| Method | Unknowns |
| --- | --- |
| [Primal MHM](primal-mhm.md) | pressure and normal Darcy flux |
| [MHM with mixed H(div) locals](mixed-mhm.md) | Darcy flux, pressure and boundary pressure |
| [Robin multiscale hybrid method](robin-mh.md) | pressure and Robin multiplier |
| [MH²M: pressure, conormal and trace](mh2m.md) | pressure, private conormal and continuous pressure trace |
| [MsHHO: cell and face moments](mshho.md) | cell/face pressure moments and energy reconstruction |
| [Petrov–Galerkin MHM](pgmhm.md) | base and enriched pressure with normal flux |
| [MHM-USFEM](mhm-usfem.md) | scalar values, coarse moments and normal multiplier |
| [Unfitted MHM traces](unfitted.md) | pressure and independently segmented normal-flux traces |
| [Stokes–Brinkman MHM](stokes-brinkman.md) | velocity, pressure and pseudo-traction |
| [Oseen MHM](oseen.md) | velocity, pressure and half-advection pseudo-traction |
| [Primal elasticity MHM](primal-elasticity.md) | displacement and physical traction |
| [Displacement–pressure GaLS MHM](gals-elasticity.md) | displacement, pressure and negative physical Cauchy traction |
| [Mixed H(div) stress MHM](mixed-elasticity.md) | stress rows, displacement and weak rotation |
| [Transient transport MHM](transient-transport.md) | time-dependent scalar and interface multiplier |
| [Elastodynamic MHM](elastodynamics.md) | displacement, velocity, acceleration and traction |
| [Helmholtz MHM](helmholtz.md) | complex pressure and normal multiplier |
| [Maxwell MHM](maxwell.md) | electric, magnetic and tangential trace fields |

## Reconstruction, indicators and refinement

| Strategy | What you will verify |
| --- | --- |
| [Flux and potential reconstruction](flux-recovery.md) | fine-cell balance, normal moments and potential conformity |
| [Error indicators](error-indicators.md) | physical residuals, recovery contributions and estimator scope |
| [Adaptive spaces](adaptivity.md) | error decrease against work with independent scale controls |
| [Recursive MHM](recursive-mhm.md) | equivalent nested condensation, inherited boundary and gauge |

Explore physical problems in the [Gallery](../../gallery/index.md) or select an [execution environment](../../guides/index.md). All cited publications are collected in the [Bibliography](../../literature.md).
