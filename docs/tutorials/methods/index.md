# Learn PyMHM by method

Start with the [feature and API overview](../overview.md), then choose a method. Every lesson follows **meshes → local problem → global problem → assembly → solve → fields and errors**, with the mathematical formulation next to its code. Ready-made physical solvers appear only as optional conveniences after the definition.

A final convergence study identifies the physical observable, refinement variable,
approximation spaces and hypotheses of its cited estimate. A polynomial patch
checks the implementation; the refinement studies use solutions outside the chosen local spaces
and several terminal intervals to examine the asymptotic regime.

## Reading the convergence evidence

Each refinement figure preserves the coarse levels. Dashed curves show the
cited target powers anchored to the finest measured error. The successive-rate
panel uses the actual refinement ratios, and a third panel shows the error
amplitude divided by the target power:

$$
\begin{aligned}
r_i&=\frac{\log(E_{i-1}/E_i)}{\log(h_{i-1}/h_i)},\\
A_i&=\frac{E_i}{h_i^q}.
\end{aligned}
$$

Stable consecutive rates near $q$, together with a plateau in $A_i$, distinguish
the asymptotic regime from a single favorable slope. The shaded window always
contains the last four measured levels. The accompanying table reports its
three successive orders and the variation of its normalized amplitude.
Independent local-resolution, quadrature, boundary and physical-equation checks
remain necessary: a fitted exponent does not verify an error theorem.

An error estimate is an upper bound; a faster measured order can satisfy it
without producing a plateau at its guaranteed power. Each page distinguishes
proved estimates from target rates reported in a published numerical experiment.
The elastodynamic spatial targets, in particular, are published numerical rates
and are not presented as a proved dynamic error estimate.

Smooth spatial controls and heterogeneous applications appear separately. A
temporal study holds the spatial discretization fixed; it cannot establish the
multiscale spatial rate. Estimates for trace refinement use the trace segment
size rather than the fixed macro diameter. An order marked **observed** has no
separate theorem claim. The [convergence notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/convergence/method_rates.ipynb)
recomputes these diagnostics from the attributed records.

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
| [Flux and potential reconstruction](flux-recovery.md) | recovered-field convergence, normal continuity, projected divergence and continuous-test equilibrium |
| [Error indicators](error-indicators.md) | physical residuals, estimator effectivity and recovery contributions |
| [Adaptive spaces](adaptivity.md) | error decrease against work with independent scale controls |
| [Recursive MHM](recursive-mhm.md) | equivalent nested condensation, inherited boundary and gauge |

Explore physical problems in the [Gallery](../../gallery/index.md) or select an [execution environment](../../guides/index.md). All cited publications are collected in the [Bibliography](../../literature.md).
