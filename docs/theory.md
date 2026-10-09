# Theoretical Background

PyMHM separates a variational formulation from its local numerical solver.
A method is specified by its volume equations, skeleton spaces, retained
physical modes and global coupling equations. The common
[hybrid construction](theory/foundations.md) makes these choices explicit;
changing a linear solver backend does not change the method.

The [features overview](tutorials/overview.md) follows the complete
mesh → local problems → global problem → assembly → solution → postprocessing
workflow. The [Bibliography](literature.md) gives publication details;
the [Gallery](gallery/index.md) records the discretizations actually evaluated.

## Method families

| Family | Mathematical description | Step-by-step tutorials |
| --- | --- | --- |
| MHM with primal or mixed locals | [Normal-flux hybridization, local means and conservation](theory/elliptic.md) | [Primal MHM](tutorials/methods/primal-mhm.md), [mixed MHM](tutorials/methods/mixed-mhm.md), [unfitted interfaces](tutorials/methods/unfitted.md) |
| MH, MH²M, MsHHO and PGMHM | [Robin, three-field, moment and Petrov–Galerkin variants](theory/alternatives.md) | [MH](tutorials/methods/robin-mh.md), [MH²M](tutorials/methods/mh2m.md), [MsHHO](tutorials/methods/mshho.md), [PGMHM](tutorials/methods/pgmhm.md) |
| Stokes, Brinkman and Oseen MHM | [Mixed velocity/pressure and pseudotraction](theory/flow.md) | [Stokes–Brinkman](tutorials/methods/stokes-brinkman.md), [Oseen](tutorials/methods/oseen.md) |
| Elasticity MHM | [Rigid motions, GaLS and weak stress symmetry](theory/elasticity.md) | [Primal elasticity](tutorials/methods/primal-elasticity.md), [GaLS elasticity](tutorials/methods/gals-elasticity.md), [mixed elasticity](tutorials/methods/mixed-elasticity.md), [elastodynamics](tutorials/methods/elastodynamics.md) |
| Scalar transport and time stepping | [Conservative RAD, stabilized locals and mass terms](theory/transport.md) | [MHM-USFEM](tutorials/methods/mhm-usfem.md), [transient transport](tutorials/methods/transient-transport.md) |
| Recovery and adaptation | [Fluxes, conforming potentials, estimators and refinement](theory/recovery.md) | [Recovery](tutorials/methods/flux-recovery.md), [indicators](tutorials/methods/error-indicators.md), [adaptivity](tutorials/methods/adaptivity.md), [recursive levels](tutorials/methods/recursive-mhm.md) |
| Wave MHM | [Complex Helmholtz and time-domain Maxwell](theory/waves.md) | [Helmholtz](tutorials/methods/helmholtz.md), [Maxwell](tutorials/methods/maxwell.md) |

## Choose spaces before choosing a solver

Let $H$ denote the macrocell diameter, $h_\Gamma$ the skeletal subface
diameter, $h$ the local finite element diameter, $\ell$ the skeletal degree
and $k$ the local degree. These are independent controls. Continuous face
interpolation is continuous **within** a macroface; MH²M additionally has a
globally continuous pressure trace.

Every discretization needs local invertibility on its physical complement,
detection of every skeleton mode by the local test space, and coupling of
the retained coarse modes. Mixed methods also need their inf-sup or
stabilization conditions. A successful factorization does not establish
mesh-uniform stability.

| Construction | Essential degree or space requirements |
| --- | --- |
| Primal MHM | Local boundary traces must detect the normal-flux skeleton; constants remain macrocell pressure unknowns. |
| Mixed Darcy | $\operatorname{div}V_h=Q_h$ with compatible restricted normal traces. On triangles, RT$_r$ pairs with discontinuous $P_r$ and BDM$_r$ with $P_{r-1}$. |
| PGMHM | Published stability uses $k\ge\ell+d$, $\ell\ge1$ and sufficiently small stabilization. The $\ell=0$ numerical experiments are separate. |
| MH²M | The analyzed family is $\Gamma=P_{r+1}$, $\Lambda=P_r$, $V_h=P_{r+1}$ with independent partitions and the required Fortin conditions. |
| MsHHO | The energy reconstruction must represent cell $P_m$ and face $P_\ell$ moments. The source variant controls MHM equivalence; $m=-1$ is an exception. |
| Primal elasticity | Retain every rigid displacement. On one local triangle, a sufficient condition is $k\ge\ell+1$ for even $\ell$ and $k\ge\ell+2$ for odd $\ell$. |
| GaLS elasticity | Equal-order locals need the strain inverse bound and compatible traction refinement. The two-dimensional $\ell=1$ choices are $(k,r)=(1,4),(2,2),(3,1)$. |
| Mixed stress elasticity | Rigid displacement and rotation belong to the coupled kernel. The tetrahedral AFW realization uses BDM$_k$/P$_{k-1}$/P$_{k-1}$ with $k\ge2$. |
| Flow | Stable Taylor–Hood or consistent equal-order USFEM locals, compatible vector traces and a physical global pressure normalization when required. |
| Moment-based flux estimator | $k\ge\ell+d$, $\ell\le m\le k$ for RT$_m$ reconstruction; discontinuous subface tests are part of the stated estimator theorem. |

The method pages give original citations and geometry restrictions behind
these rules. In particular, a condition with $d=2$ cannot be reused for a
three-dimensional estimator.

## Interpret convergence rates

Two-level estimates contain skeleton, local and algebraic errors. Transient
problems also contain time error. Rates concern sufficiently regular data,
admissible meshes and resolved local problems; singular forcing, material
interfaces and unresolved layers can reduce them. Each tutorial identifies
the refinement variable, field norm and measured final interval rates.

| Method or family | Smooth-problem target | Hypotheses and interpretation |
| --- | --- | --- |
| Primal MHM, MH and PGMHM | Typically energy/physical flux $\ell+1$, pressure $\ell+2$ | Pressure gains need dual regularity; local errors must not dominate. PGMHM divergence is a separate norm. |
| MsHHO with cell degree $m=\ell$ | Energy $\ell+1$, pressure $\ell+2$ | Pressure gains need dual regularity and resolved local problems. Independent cell degrees and $m=-1$ have separate [degree and injectivity conditions](theory/alternatives.md). |
| MH²M: $\Gamma=P_{r+1}$, $\Lambda=P_r$, $V_h=P_{r+1}$ | Energy $r+1$ | Theorem 19 controls pressure/conormal trace and volume energy with independent mesh sizes. |
| Fitted faces on an unfitted macro mesh | Broken-gradient $\ell+3/2$ in the smooth face-refinement study | Fixed macro partition, physical-region regularity, fitted subfaces and accurate locals. |
| Flow | Velocity energy and pressure follow the mixed/stabilized order; velocity $L^2$ can gain one order | State both local and skeletal families; arbitrary pairs have no universal viscosity-independent rate. |
| Weak-symmetry BDM$_k$ elasticity | Stress, displacement and rotation $L^2$ order $k$ for the AFW family | Sufficient regularity and trace accuracy; symmetry is a moment condition. |
| Linear-trace elastodynamic study | Displacement/velocity $L^2$: 3; broken $H^1$: 2; stress broken $H(\mathrm{div})$: 1 | The published smooth spatial family; Newmark time order is 2 separately. |
| Helmholtz trace degree $\ell$ | Pressure $\ell+2$, broken gradient $\ell+1$ | Fixed frequency in the resolved nonresonant regime with accurate locals. |
| Maxwell TM trace degree $\ell=1,2$ | Combined $L^2$: $\ell+1$; broken curl: $\ell$; time: 2 | Published smooth cavity data; retain staggered field times and CFL restrictions. |
| Backward-Euler transport/diffusion | Time: 1 | Refine time until spatial and reference errors are smaller. |

A measured slope qualifies the stated sequence. It does not qualify every
material, geometry or backend. Method-specific sources and estimates follow
on the linked pages; the [Gallery](gallery/index.md) distinguishes analytical
verification, matched discrete comparisons and reproduction limits.

## Geometry and meshes

[Meshes and approximation spaces](theory/meshes.md) covers triangles,
quadrilaterals, polygons, tetrahedra, prisms, hexahedra and polyhedra, with
formulation-specific scope. The [mesh guide](meshing.md) explains generation,
reading, writing and material/boundary tags. Geometry support does not imply
that every formulation uses that geometry.

## Original analyses

- Elliptic MHM: [Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019), [Araya et al. (2013)](https://doi.org/10.1137/120888223), and [Durán et al. (2019)](https://doi.org/10.1016/j.cma.2019.05.013).
- Alternative global formulations: [Barrenechea et al. (2024), MH](https://doi.org/10.1137/22M1542556), [de Barros et al. (2026, v3), MH²M](https://arxiv.org/abs/2404.16978v3), [Chaumont-Frelet et al. (2022), MsHHO](https://doi.org/10.1051/m2an/2021082), and [Fernando et al. (2023), PGMHM](https://doi.org/10.1007/s40314-023-02304-y).
- Flow: [Araya et al. (2017), Stokes–Brinkman](https://doi.org/10.1016/j.cma.2017.05.027), [Araya et al. (2025), a priori analysis](https://doi.org/10.1137/24M1649368), and [Araya et al. (2021), Oseen](https://doi.org/10.1007/s10444-020-09833-8).
- Elasticity: [Harder et al. (2016), primal](https://doi.org/10.1051/m2an/2015046), [Devloo et al. (2021), mixed stress](https://doi.org/10.1051/m2an/2021013), [Arnold et al. (2007), AFW spaces](https://doi.org/10.1090/S0025-5718-07-01998-9), and [Gomes et al. (2024, v1), GaLS](https://arxiv.org/abs/2403.16890v1).
- Transport and stabilized locals: [Harder et al. (2015)](https://doi.org/10.1137/130938499), [Araya et al. (2024), generalized RAD](https://doi.org/10.1016/j.cma.2024.117089), and [Santiago et al. (CILAMCE 2025), MHM-USFEM](https://doi.org/10.55592/cilamce2025.v5i.14270).
- Dynamics and waves: [Gomes et al. (2017), elastodynamics](https://doi.org/10.20906/CPS/CILAMCE2017-0399), [Chaumont-Frelet and Valentin (2020), Helmholtz](https://doi.org/10.1137/19M1255616), and [Lanteri et al. (2018), Maxwell](https://doi.org/10.1137/16M110037X).
- Unfitted faces and recovery: [Chaumont-Frelet et al. (2026)](https://doi.org/10.1016/j.camwa.2026.01.016) and [Barrenechea et al. (2026)](https://doi.org/10.1137/24M1673073).
- Recursive hybrid structure: [Harder and Valentin (2016)](https://doi.org/10.1007/978-3-319-41640-3_13).

Full titles, publication versions and supporting numerical literature are in
the [Bibliography](literature.md). Each method page gives its own relevant
references, rather than assigning a single theorem to every entry.
