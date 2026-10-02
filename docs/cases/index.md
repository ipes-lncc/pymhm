# Numerical evidence guide

The gallery presents representative evidence for the implemented MHM families.
Each case states its operator, material and boundary data, local and skeletal
spaces, norm conventions and scientific limits. The selected figures show
published comparisons, physical fields or convergence; numerical tables retain
the corresponding resolution and reference checks.

A **published comparison** measures a stated figure or table against its source.
A **matched setup** preserves the identified physical data and approximation
spaces, while unresolved historical choices remain explicit. An **independent
reference** checks the same problem using a separate assembly; without an exact
solution its classical reference also needs a refinement study. An **analytical
verification** tests an exact solution or a conditional identity. These forms of
evidence do not imply that every numerical experiment in a paper is reproduced.

## Darcy and multiscale constructions

Reference identifiers refer to the [literature catalog](../literature.md).

| Case and literature | Evidence and scientific limit | Public acquisition or rendering entry point |
| --- | --- | --- |
| [Darcy and Stokes publication curves](reproduction.md), L01/L02/L13/L16 | Published curves with digitization uncertainty and separate local-space/norm conventions. The complete historical Stokes stress diagnostic remains unresolved. | `examples/plot_reproductions.py` reads the comparison records; the case lists the acquisitions. |
| [Recursive MHM](nested.md), L03 | Full leaf-system field equivalence on five refinements; an algebraic construction, not a historical benchmark. | `examples/verify_nested.py` |
| [MHM–MsHHO](mshho.md), L06 | Five-level convergence and field equivalence under the stated source and space hypotheses. | `examples/verify_mshho.py` |
| [Quarter-five spot and square obstacle](quarter-five-spot.md), L01/L02 | Normalized wells, published elevation convention, recorded MSL/NeoPZ comparisons and a separately refined classical RT0 reference. The adopted square geometry and singular-load allocation are explicit. | `examples/plot_quarter_spot.py` acquires the homogeneous well study. Obstacle/reference galleries read comparison records; new obstacle and independent-reference acquisitions require separate drivers. |
| [Periodic permeability](periodic.md), L04/L07 | Separate macro/local/face refinement and Q1/Q3/Q5 reference increments. The historical error curve is not reproduced identically. | `examples/verify_periodic.py` |
| [Rectangular mixed Darcy](tensor-rt.md), L05 | Pressure/flux convergence and all 50 distinguishable published Figure 3 markers within the stated extraction uncertainty. | `examples/verify_tensor_rt.py` |
| [Tetrahedral and prismatic wells](mixed-well-geometries.md), L05 | Exact radial fields, compatible mixed spaces and executed NeoPZ/Basix comparisons on a common faceted domain; historical connectivity remains distinct. | `examples/solve_mixed_well_geometries.py`, `examples/plot_mixed_well_geometries.py` |
| [Oscillatory mixed well](mapped-well-oscillatory.md), L05 | Independent same-case fields and spatial/quadrature refinement of a classical mixed baseline; historical material/mesh qualifications apply. | `examples/solve_mapped_oscillatory_well.py`; acquisition and integration settings are on the case page. |
| [SPE10 Darcy and Brinkman](spe10.md), L07/L13/L16 | Identified Model 2 slices and method-specific resistance conventions; refined Taylor–Hood reference for Brinkman. Visual agreement alone does not identify original fields. | `examples/solve_spe10.py`; the case lists material download and classical acquisitions. |
| [SPE10 flux references](spe10-flux.md), L07 | Signed flux components and refinement of Q3, MSL P1 and NeoPZ RT0 references, each with its own spaces. | `examples/plot_spe10_flux.py` renders acquired comparison records. |
| [Adaptive SPE10](spe10-adaptive.md), L09 | Published P2/P0/RT2 spaces, seven adaptive states, independent MHM assembly and a refined classical RT2 baseline. The published flux figure remains quantitatively different. | `examples/solve_spe10_adaptive.py`, `examples/plot_spe10_adaptive.py` |
| [Unfitted interfaces](unfitted.md), L08/L10 | Material cuts, fitted local fields, separate trace/local/integration studies and independent physical-field comparisons. Historical connectivity is not inferred from matching degrees. | `examples/unfitted_campaign.py`, `examples/plot_unfitted_convergence.py` |
| [Reconstruction and estimation in 3D](reconstruction3d.md), L09 | Admissible-degree convergence, exact fields, independent full-saddle checks and one-sided profiles; error-estimate hypotheses require `k >= ell + d`. | Pixi tasks `reconstruction3d-uniform`, `reconstruction3d-adaptive`, `reconstruction3d-resolution` |
| [MH²M oscillatory medium](mh2m-heterogeneous.md), additional reference | Published data lineage, independent CG1/CG3 refinement and matched discrete controls. The historical curves, including their classical reference, remain different. | `examples/mh2m_heterogeneous.py`, `examples/mh2m_cg_reference.py` |

Further formulation details cover [triangular RT](https://github.com/volpatto/pymhm/blob/main/docs/cases/darcy-rt.md), [BDM](https://github.com/volpatto/pymhm/blob/main/docs/cases/darcy-bdm.md),
[Robin MH](https://github.com/volpatto/pymhm/blob/main/docs/cases/mh.md), [tetrahedral MH/MH²M](https://github.com/volpatto/pymhm/blob/main/docs/cases/mh3d.md), [PGMHM](https://github.com/volpatto/pymhm/blob/main/docs/cases/pgmhm.md),
[PGMHM inclusions](https://github.com/volpatto/pymhm/blob/main/docs/cases/pgmhm-inclusions.md), [PGMHM on SPE10](https://github.com/volpatto/pymhm/blob/main/docs/cases/pgmhm-spe10.md),
[Darcy reconstruction moments](https://github.com/volpatto/pymhm/blob/main/docs/cases/reconstruction-moments.md),
[weighted estimation](https://github.com/volpatto/pymhm/blob/main/docs/cases/weighted-estimator.md), [face jumps](https://github.com/volpatto/pymhm/blob/main/docs/cases/darcy-jump.md),
[three-dimensional Darcy](https://github.com/volpatto/pymhm/blob/main/docs/cases/darcy3d.md) and [planar transmission](https://github.com/volpatto/pymhm/blob/main/docs/cases/planar3d.md).

## Transport and flow

| Case and literature | Evidence and scientific limit | Public acquisition or rendering entry point |
| --- | --- | --- |
| [Adaptive RAD](adaptive-transport.md), L11 | Independent mixed-wall systems, local/face refinement and the literal ε=0.1 curve comparison. The separate ε=1 control does not resolve its historical discrepancy. | `examples/transport_mixed_campaign.py`, `examples/transport_coefficient_controls.py` |
| [Transient Darcy transport](transient-transport.md), L11 | Manufactured time-convergence and conservative coupling checks. The published random-field §5.4 problem requires its own complete acquisition and independent reference. | `examples/transport_campaign.py --collect` acquires the manufactured study; the historical case requires a separate driver/reference. |
| [MHM-USFEM reaction–diffusion](unusual.md), additional reference | Full residual stabilization, analytical layers and executed FreeFEM comparisons; local and skeletal errors are separate. | `examples/solve_unusual.py`, `examples/verify_unusual_resolution.py` |
| [Polygonal families](polygons.md), L12 | Five polygon families and five refinements with analytical fields; historical connectivity is not asserted. | `examples/verify_polygons.py` |
| [Stokes–Brinkman adaptation](stokes-adaptive.md), L14 | Published estimator terms and separate macro/face algorithms; classical Taylor–Hood cavity refinement. Constant and regularized lids have different corner behavior. | `examples/solve_stokes_adaptive.py`, `examples/solve_cavity_reference.py` |
| [Adaptive Oseen](oseen.md), L15 | Smooth and layer studies with full residuals and measured effectivity. The zero-reaction layer lies outside strict positive coercivity. | `examples/solve_oseen.py`, `examples/plot_oseen.py` |

Additional cases state [convex polyhedral](https://github.com/volpatto/pymhm/blob/main/docs/cases/polyhedral-rad.md) and
[nonconvex star-shaped](https://github.com/volpatto/pymhm/blob/main/docs/cases/star-polyhedra.md) geometry conditions,
[three-dimensional RAD](https://github.com/volpatto/pymhm/blob/main/docs/cases/rad3d.md), [hexagonal layers](https://github.com/volpatto/pymhm/blob/main/docs/cases/rad-layer.md),
[conditioning conventions](https://github.com/volpatto/pymhm/blob/main/docs/cases/rad-conditioning.md), [three-dimensional flow](https://github.com/volpatto/pymhm/blob/main/docs/cases/flow3d.md),
[high-order operators](https://github.com/volpatto/pymhm/blob/main/docs/cases/high-order.md) and [SPE10 unusual stabilization](https://github.com/volpatto/pymhm/blob/main/docs/cases/unusual-spe10.md).

## Elasticity and waves

| Case and literature | Evidence and scientific limit | Public acquisition or rendering entry point |
| --- | --- | --- |
| [Mixed stress families](mixed-families.md), L18 | Analytical convergence, weak symmetry, enrichment and finite/infinite bulk-modulus limits; historical exterior traces remain qualified. | `examples/solve_elasticity_families.py`, `examples/plot_elasticity_families.py` |
| [Mixed AFW in 3D](mixed-elasticity3d.md), L18 | Five-level analytical fields, Lamé sweeps and independent original-equation checks; this is original 3D verification. | `examples/solve_mixed_elasticity3d.py`, `examples/plot_mixed_elasticity3d.py` |
| [Displacement–pressure elasticity](elasticity.md), L19 | Locking/face-refinement studies, GaLS/Taylor–Hood and matching MSL fields. Printed amplitudes and unstated historical stabilization limit exact table reproduction. | `examples/plot_elasticity.py` separates acquisition from `--reuse-results`. |
| [HPC4e material](hpc4e.md), L18 | Published geological samples and spaces, physical stress profiles and independent classical RT refinement. | `examples/solve_hpc4e_mhm.py --download`, `examples/solve_hpc4e_reference.py` |
| [Helmholtz](helmholtz.md), additional reference | Analytical waves, published angular comparisons, local resolution and stability/refinement studies; historical angular ordinates remain different. | `examples/helmholtz_campaign.py`, `examples/plot_helmholtz.py` |
| [Marmousi](marmousi.md), additional reference | Selected primary SEG-Y inputs and independently refined P4 baseline; the 15-case MHM comparison and literal historical table remain incomplete. | `examples/marmousi_reference.py`, `examples/marmousi_trace_family.py`; large acquisition settings are stated in the case. |
| [Maxwell nanowaveguide](maxwell-nanoguide.md), additional reference | Recorded component fields and refinement relative to a classical DG baseline with selected incident-wave data; the complete independent device comparison remains a separate verification target. | `examples/maxwell_nanoguide.py` acquires the MHM candidate; `examples/maxwell_nanoguide_results.py` integrates and renders records. Independent DG acquisition is not distributed. |
| [Elastodynamics](elastodynamics.md), additional reference | Analytical Equation (53) spatial/time convergence and full-saddle trajectories. The heterogeneous 2017 source case still requires a complete acquisition, full trajectories and a refined classical baseline. | `examples/elastodynamics_campaign.py` and `examples/elastodynamics_results.py` cover Equation (53); the selected heterogeneous inputs require their own acquisition and independent-reference drivers. |

The [primal tensor](https://github.com/volpatto/pymhm/blob/main/docs/cases/primal-elasticity.md), [anisotropic stress](https://github.com/volpatto/pymhm/blob/main/docs/cases/anisotropic-stress.md),
[rectangular mixed](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity-tensor-rt.md), [GaLS3D](https://github.com/volpatto/pymhm/blob/main/docs/cases/gals3d.md) and
[general 3D elasticity](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity3d.md) pages retain their space-specific details.
The [Maxwell formulation](https://github.com/volpatto/pymhm/blob/main/docs/cases/maxwell.md) includes its analytical TM and 3D controls.
[Execution](../execution.md) and [performance](https://github.com/volpatto/pymhm/blob/main/docs/performance.md) state measured
MPI/GPU/solver behavior; historical cluster scaling is a separate target.

## Read and regenerate a case

Spatial panels mark actual macro boundaries; profile plots preserve one-sided
values and macroface intersections. Analytical/numerical panels share color limits,
while errors use separately labeled scales. Sampled image differences are distinct
from quadrature norms. Convergence guides apply only to their stated spaces and
regularity assumptions.

Use the selected case's acquisition command first, then its rendering command.
A plot-only reader requires its numerical records and does not acquire an
independent reference. Primary SPE10, SEG-Y and geological inputs have download
procedures on their case pages. A clean checkout needs those acquisitions before
regenerating research figures; large reference and endpoint campaigns are separate
from CI. Reference programs or comparison acquisitions that are not distributed
must be supplied or independently implemented with the same declared contract.

For example, these tasks acquire their own analytical studies:

```bash
pixi run -e notebooks verify-tensor-rt
pixi run -e notebooks verify-mshho
pixi run -e notebooks verify-polygons
pixi run -e docs docs-check
```

The [literature](../literature.md), [scope](../roadmap.md) and
[verification](../verification.md) pages distinguish formulation support,
executed evidence and historical targets still requiring comparison.
