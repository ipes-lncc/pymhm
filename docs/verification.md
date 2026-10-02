# Numerical verification

## Numerical tests

The test suite checks both executable branches and the mathematical meaning of
results. Run `pixi run -e test test-cov` to measure the current source tree against
the independent 99% line and branch coverage gates. Optional-backend contract
tests check dispatch, error handling and resource cleanup; native integration
tests are separately marked and must run with their dependencies installed.

The current portable suite passed **3296 tests** on Python 3.13, with 270
optional cases skipped. The Python 3.13 coverage run covered
**16573 of 16585 executable lines (99.9276%)** and
**4729 of 4744 branches (99.6838%)**. Complete suites on Python 3.11 and
3.12 passed the same 3296 tests, with the same 270 skips. The installed-package
suite using the minimum supported NumPy 1.26.4 and SciPy 1.12.0 versions
also passed 3296 tests, with 270 skips. Skipped cases are not counted as passed.

Separate native runs passed 216 DOLFINx/PETSc integration checks, with two
optional cases skipped, including global AFW/BDM reference comparisons.
The current PyVista/VTK integration suite passed 95 checks. Earlier separate
native runs passed 19 meshing checks,
nine MPI checks, eight GPU checks, and the selected PARDISO and PyAMG
integrations. The solver selections
overlap in their PyAMG cases and should not be added as independent test counts.
Native FreeFEM/BAMG execution is verified
separately from the portable metric-transfer and marking tests.
Source notebooks retain no execution output; executed copies are stored separately.
Ruff, formatting, static typing and strict MkDocs checks passed.
Release validation checks wheel/sdist contents and the noarch Conda
import/dependency contract. These are release procedures; the ongoing numerical
campaigns and their notebook updates require separate acceptance before a final
release. Building artifacts does not publish a release.

Archived comparison notebooks read preserved numerical results; executing them
does not rerun an external reference program. CI is configured to exercise
Linux, Windows and macOS; the native executions recorded here were on Linux.

The reference-backend suite includes:

- Oriented mesh incidence, outward boundary normals, polynomial quadrature,
  Pk nodal identities, RT0/RT1/RT2 and BDM2 Piola/moment identities, and
  independent continuous/discontinuous face partitions.
- Condensed versus uncondensed block solutions, multi-mode local kernels,
  mean constraints, pure-Neumann compatibility and singular trace detection.
- Affine/anisotropic Darcy patches, manufactured convergence, high-contrast
  layered interfaces, macro balances, RT0 fine-cell balances and all BDM2/P1
  divergence moments.
- Taylor–Hood and USFEM velocity/pressure patches, Stokes/Brinkman limits,
  complete high-order residuals, tensor resistance, physical global gauges
  and an Oseen advection-sign patch.
- GaLS and Taylor–Hood elasticity, variable material residuals, finite and
  infinite Lamé limits, and BDM2 weak-symmetry stress with projected displacement.
- Variable-coefficient conservative RAD/SUPG, half-advection Robin data,
  tangential-advection gauges and high-order heat space-time patches.
- RT moment recovery with continuous-test conservation, distinct from
  minimum-energy RT0 equilibration with fine-cell balance constraints.
- Oswald fine-triangle nodal averaging, exact polynomial recovery and all four
  unit-diffusion estimator terms; rejection of incompatible meshes, boundary
  moments and coefficient assumptions.
- CPU parallel numerical equivalence, AMG kernel projection and CPU/GPU residuals.
- Cartesian Qk bases, physical gauges, exact material intersections and point-source
  angular allocation; SPE10 indexing, units, checksums and strict property parsing.
- Componentwise slip-wall traction and pressure gauges; independent constant and
  variable tensor USFEM operators under explicit residual-parameter conventions.
- Independent RT normal/interior enrichment, total-degree rotation spaces,
  mapped hexahedral Piola transformations and physical normal continuity.
- Affine tetrahedral 18-mode/P1 and 32-mode/P2 and prismatic 27-mode/W11
  mixed families, including independent Basix operators, physical Neumann
  gauges, anisotropic patches and normal-trace orientation.
- Original polygonal face spaces on star-shaped polyhedra, independent local
  tetrahedra, general three-dimensional Pk fields and six elasticity rigid modes.
- Independent native UFL checks of tetrahedral P5/P6 operators and AFW mixed
  stress elasticity, including weak symmetry and scaled incompressibility constraints.
- Newmark displacement/velocity updates, inertia, local subcycling and the
  discrete energy identity for a common undriven time step.
- MsHHO cell/face moment reconstruction, conditional MHM equivalence,
  recursive local condensation, offline factors and exact-matrix reuse.
- Material intersections, fitted subfaces, conforming red/green and longest-edge
  refinement, preserved parentage, local error control and published marking rules.
- The literal published Darcy recovery indicator, distinct from its weighted
  energy variant, and FreeFEM/BAMG metric remeshing with transferred coefficients.
- Rank-owned MPI assembly without gathering all local matrices, resident GPU
  batches, saddle block preconditioning and complete physical residual checks.
- Geological data orientation, dimensional elasticity scaling, polynomial
  field interchange and independently known L2/compliance integrals.

Analytical polynomial pressure patches have errors near floating-point roundoff.
Such a result is an algebraic/consistency check; it does not prove high-contrast
robustness or stability for arbitrary enriched spaces.

## Numerical checkpoints

Reusing a saved field requires matching mathematical settings, numerical sources
and acquired file hashes. Changes to frequency, approximation spaces, quadrature,
precision, material, estimator or boundary conditions start a new acquisition.
Worker counts and requested cases may change where the driver treats them as
execution controls.

Inspect a recorded campaign with:

```bash
pixi run -e test python -m examples.validate_campaign_checkpoint \
  --manifest examples/results/gals3d/campaign.json \
  --output build/validation/gals3d-checkpoint.json
```

This writes a separate report and a digest-named copy of the original manifest.
The report verifies acquired field hashes and finite JSON values. For a field
without an acquisition hash, it records the current bytes as an observation;
that observation does not establish the earlier bytes or permit continuation.
Physical accuracy and equivalence with current operators require the numerical
checks described on the corresponding case page.

Existing results retain their original acquisition metadata. To calculate with
changed sources or settings, select a fresh output directory:

```bash
pixi run -e test python -m examples.solve_gals3d \
  --output build/new-gals3d --levels 1 2 3 4 5
```

The structured and crisscross MH²M campaigns and their CG3 comparisons also
check reference identities, norm quadratures and complete finite norm records:

```bash
pixi run -e test python -m examples.validate_mh2m_campaign
```

This command preserves acquired manifests and writes separate validation
records. It links the existing independent CG1 verification without rerunning
that solver or attributing current source hashes to earlier acquisitions.
To recompute CG3 comparisons with current sources, select a new `--output`
JSON in `examples.compare_mh2m_cg3` or `examples.compare_mh2m_cg3_controls`.
For new fields, use a new `--output` directory in `examples.mh2m_heterogeneous`
or `examples.mh2m_crisscross_campaign`.

## Direct comparisons

The [published-result comparison](cases/reproduction.md) matches all four
Darcy Figure 5 curves of Harder et al. (2013) at five resolutions within the
1% digitization allowance; the largest difference is 0.20%. It explicitly
distinguishes primal P1 from classical RT0 and its quadratic potential.
The [analytical MHM case](https://github.com/volpatto/pymhm/blob/main/docs/cases/analytic.md) separately verifies equation (42)
with full source moments and a nonconstant-source Neumann reconstruction;
these give different coarse pressures for the same cosine forcing.
These are pyMHM calculations against digitized article curves, not an execution
of the historical Darcy program.

For Stokes, pyMHM's DOLFINx/UFL P2/P2 local formulation matches 24 published values across
six resolutions for velocity L2/H1, pressure L2 and standard stress H(div),
with a maximum discrepancy of 0.14%. The P3/P3 velocity and pressure comparison
has 18 values across six levels and a maximum discrepancy of
0.70%. The full stress L2 curves and the P3/P3 stress H(div) curve do **not**
match; both the numerical values and the norm definitions remain reported.

The [MSL implementation comparison](https://github.com/volpatto/pymhm/blob/main/docs/cases/reference-comparison.md) executes
`msl_mhm` with `msl_cg` local solves and `msl_core` geometry. It matches full
Darcy pressure and raw flux fields on five meshes, up to 65,536
fine triangles. The largest field L2 differences are approximately
\(5.1\times10^{-14}\) and \(4.5\times10^{-13}\), respectively.

For the [coarse cosine gallery case](https://github.com/volpatto/pymhm/blob/main/docs/cases/coarse-cosine.md), the exact
32-macrotriangle/512-fine-triangle comparison gives flux differences of
\(7.98\times10^{-15}\) against MSL primal assembly and
\(6.94\times10^{-14}\) against restricted NeoPZ RT0 assembly. The MSL
comparison includes the prescribed nonzero weak boundary functional.
Zero, constant, and affine pressure checks verify boundary assembly separately.
The 22.42%/21.87% analytical flux errors remain;
these are underresolved approximations, even though the codes agree.

The [Darcy verification](https://github.com/volpatto/pymhm/blob/main/docs/cases/darcy-audit.md) compares actual physical fields with
independent DOLFINx 0.9.0 mixed assembly and uncondensed primal UFL assembly.
The [NeoPZ comparison](https://github.com/volpatto/pymhm/blob/main/docs/cases/neopz.md) adds 46 independent RT0/P0 field
comparisons: five analytical cases, up to five refinement levels, and both
complete and restricted macro traces. The largest flux L2 difference is
\(3.94\times10^{-10}\), in a layered medium with contrast 1000 and flux norm
approximately \(1.414\times10^3\). A separate boundary-quadrature diagnostic
quantifies the coarse cosine differences. RT0 trace partitions must align with
the fine-edge partition under an absolute geometric tolerance.

The [Stokes pressure verification](https://github.com/volpatto/pymhm/blob/main/docs/cases/flow-audit.md) checks native local operators,
full versus condensed global systems, five macro and local refinements, trace
enrichment, and both one-sided pressure traces. Its low-order USFEM convergence
is not substituted for the paper's higher-order local-space experiment.
Its external finite element references use DOLFINx/UFL, while the uncondensed
check reuses the native pyMHM local matrices. The
[software provenance](literature.md#software-provenance) records MSL revisions
and distinguishes the auxiliary `msl_mfem` execution from the separately examined
`mhm-mfem` repository; neither supplies these Stokes reference curves.

The [GaLS elasticity comparison](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity-reference.md) uses native MSL
local assembly on five matching P1/P1 macro meshes and two higher-order checks.
The largest recorded L2 differences in displacement, pressure, displacement
gradient and Cauchy stress are respectively \(2.59\times10^{-13}\),
\(8.60\times10^{-12}\), \(1.68\times10^{-11}\) and
\(3.10\times10^{-11}\). These matched-space comparisons do not substitute for
the article's face-refinement tables or its missing stabilization parameter.

Independent global DOLFINx assemblies also check the
[BDM2/P1 Darcy](https://github.com/volpatto/pymhm/blob/main/docs/cases/darcy-bdm.md) and
[BDM2/P1/P1 weak-symmetry elasticity](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-elasticity.md) fields.
The latter checks stress, displacement and rotation for a non-exact solution,
in addition to portable polynomial projection and equilibrium identities.

The [unit-diffusion estimator](https://github.com/volpatto/pymhm/blob/main/docs/cases/estimator.md) is checked on two local/trace
pairs across five macro meshes. Its measured effectivity remains above one;
at the finest level it is 1.2895 for P2/P0 and 1.5652 for P3/P1, both using
RT2 recovery. Error integration is independent of the estimator quadrature.
This verifies the stated experiment and identities, not an adaptive algorithm
or a guaranteed bound for coefficients outside its identity-diffusion scope.

The [periodic-material study](cases/periodic.md) separates local and skeletal
refinement and uses a Q5 classical reference with 26.2 million unknowns. Its own
last spatial refinement changes the H1 field by 1.856%; the finest recorded MHM
field differs by 5.905%. A smaller difference against a Q1 reference reflects
shared discretization error and does not establish greater accuracy.

The [mixed-elasticity geological case](cases/hpc4e.md) uses the original HPC4e
material arrays and the published RT1/Q1/P1 and 16 × 8 macro partition. Digitized
stress-profile intervals and independently assembled classical mixed fields
provide distinct checks of historical agreement and approximation error.
Increasing the classical degree from RT1 to RT2 on 512 × 256 cells changes
stress by 1.96681% and the compliance norm by 4.79411%. Refining RT2 to
1024 × 512 cells changes these fields by 1.11913% and 3.11340%, respectively.
Against that finer reference, the finest MHM differences are 4.15940% in stress
and 7.33891% in compliance. The measured reference increments are not continuum
error bounds.

The [tetrahedral and prismatic well study](cases/mixed-well-geometries.md)
separates classical refinement from macro-trace restriction on a common
faceted domain. Its 18 calculations check analytical pressure, vector flux,
physical residuals, production and quadrature sensitivity; independent
NeoPZ basis evaluations and Basix operators verify the polynomial families.
Independent NeoPZ classical mixed solves and its native
`TPZMHMixedMeshControl` compare physical fields on matching cells and normal-trace
spaces. The persisted basis is part of the coefficient-data contract, with
replay checks across native thread counts. Fine-space approximation and skeletal
restriction errors are measured separately; agreement between implementations
does not remove the measured coarse-trace flux error.

The adaptive [SPE10](cases/spe10-adaptive.md) and
[Stokes–Brinkman cavity](cases/stokes-adaptive.md) campaigns compare estimator
histories with independently refined classical solutions. The constant cavity
lid has singular upper corners: pressure and velocity-gradient comparisons use
a fixed interior exclusion, while global velocity L2 remains meaningful.
An estimator decrease is not identified with per-step physical error contraction.
In the final recorded SPE10 adaptive state, pressure differs from the classical
480 × 1760 RT2 baseline by 1.89717%, while reconstructed flux differs by 65.90930%
in physical L2 norm. The last classical flux increment is itself 3.86719%.
A separate control on the same macro mesh combines material-fitted local
refinement with eight trace segments per face; its raw and reconstructed flux
differences are 11.70349% and 13.29238%. This enriched control does not replace
the article's local and trace spaces. These measurements do not establish
quantitative flux reproduction or a uniformly accurate reconstruction merely
from pressure-profile agreement.

## Vanishing reaction and drag

Small positive reaction or Brinkman drag produces nearly null local modes.
Direct inversion can amplify their contributions and cause cancellation between
source and trace lifts during reconstruction. The
[coarse elimination](theory.md#retaining-nearly-null-local-modes) retains these
modes explicitly while preserving the PDE and solver tolerances. The zero and
small-positive parameter regimes are tested separately. Tests include nonzero
pressure means, non-affine manufactured Stokes limits, scalar reaction
approaching zero, weak advection,
and backward Euler approaching steady diffusion. For drag
\(10^{-8},10^{-12},10^{-16}\), the affine pressure patch has velocity error below
\(7\times10^{-16}\) and pressure error below \(5.2\times10^{-15}\) for both
Taylor–Hood and USFEM. Scalar tests extend to reaction \(10^{-16}\) and time
increments \(10^{16}\).

Generic symmetric, indefinite and nonsymmetric local matrices are also compared
with independent full saddle solves. Integral constraints act on the complete
reconstruction, including pressure components generated by retained velocity
modes. These checks verify the elimination algebra and its limiting regimes;
they do not constitute a uniform parameter-robust stability theorem.

The surviving global modes are checked separately. With pure traction and
resistance `diag(1,0)`, a prescribed mean in the unresisted direction is
recovered with L2 error below \(9\times10^{-16}\). A declared rotated
translation kernel gives error below \(1.9\times10^{-15}\); a resisted mean
and a falsely declared null direction for `diag(1,1e-16)` are rejected.
For divergence-free tangential scalar advection, local P4 and cubic Robin
traces recover a constant with prescribed mean one to
\(4.6\times10^{-15}\). The trace degree is essential: the continuous constant
mode need not belong to a discrete method with an incompatible multiplier.

## Published analytical problems

`examples/verify.py` writes complete results to
`examples/results/verification.json`. These cases use analytical data appearing
in the literature, with explicitly different triangulations and local spaces.
They are **not reproductions of the published numerical tables**.

### Mixed-local Darcy problem

For `p=cos(pi*x)cos(pi*y)`, `K=I`, and `f=2*pi²*p`, prescribe exact boundary
pressure on the unit square. The analytical problem follows
[Duran et al. (2019)](https://doi.org/10.1016/j.cma.2019.05.013).
Use `n×n` macro squares split into two triangles, four subdivisions per local
edge, and a constant trace on each macroface. Assembly uses Duffy order 6;
error integration uses order 8.

| n | Primal pressure L2 | Mixed pressure L2 | Primal flux L2 | Mixed flux L2 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 8.0251e-2 | 1.0057e-1 | 8.8141e-1 | 8.6968e-1 |
| 4 | 2.4573e-2 | 4.0350e-2 | 4.9800e-1 | 4.8575e-1 |
| 8 | 6.4101e-3 | 1.7483e-2 | 2.5652e-1 | 2.4954e-1 |

P1 and P0 pressure have different asymptotic approximation orders. Mass balances
use the **assembled source quadrature** by default; passing a higher order to
`conservation_residuals(order=...)` diagnoses source integration error separately.

### Polynomial Stokes problem

Use the streamfunction and pressure of §3.1.1 in
[Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027):

$$
\psi=-128x^2(x-1)^2y^2(y-1)^2,\quad
u=(\psi_y,-\psi_x),\quad p=150(x-1/2)(y-1/2).
$$

Here the symbol `u` denotes velocity, viscosity is one and drag is zero.
Differentiate these expressions to generate `f=-Delta(u)+grad(p)`;
the streamfunction guarantees the analytical divergence is zero.
The computational grid has four local subdivisions and degree-one traces.
Duffy order 5 integrates the P2 source terms, and order 8 integrates the squared
velocity error exactly for these polynomials.

| Local method | n | Velocity L2 | Pressure L2 | Divergence L2 |
| --- | ---: | ---: | ---: | ---: |
| Taylor–Hood P2/P1 | 2 | 1.3720e-1 | 1.5056 | 2.3373e-1 |
| Taylor–Hood P2/P1 | 4 | 2.6339e-2 | 3.6839e-1 | 6.6679e-2 |
| USFEM P1/P1 | 2 | 1.5580e-1 | 1.7772 | 1.5241 |
| USFEM P1/P1 | 4 | 3.5607e-2 | 6.4850e-1 | 7.6174e-1 |

The divergence norm is reported rather than silently assumed to vanish. Weak
incompressibility and global pressure normalization do not imply pointwise
incompressibility of a Taylor–Hood or stabilized P1 velocity field.

![Measured pressure and velocity convergence for the analytical test problems](figures/convergence.svg)

The five-level curves are generated from `darcy-audit.json` and `flow-audit.json`
by `examples/plot_verification.py`. The tables above use `verification.json`;
all these values are numerical calculations rather than digitized paper figures.

## Native optional-backend evidence

Real DOLFINx tests assemble two independent macrotriangles, compare local
operators/moments with the portable backend, and reconstruct an affine pressure.
Additional real UFL tests cover RT/DG pressure data, Taylor–Hood, elasticity and
USFEM residual terms. Gmsh, Netgen and meshio tests generate meshes and verify
MSH/VTU round trips, physical markers and ownership of Gmsh sessions.

Native tests exercise PETSc/MUMPS with structurally absent diagonal entries,
repeated right-hand sides, Darcy primal/mixed local and global saddle solves,
and Taylor–Hood/USFEM flow with a global pressure gauge. Reconstructed fields
are compared with SciPy, including source-driven macro balances and affine
flow errors. These checks prevent an SPD-only smoke test from being treated
as evidence for the full MHM saddle system.

A set of 18 native PDE checks uses PETSc, PARDISO and cuDSS for both local
and global solves in near-null regimes: affine Brinkman rotation
and pressure with nonzero mean, scalar reaction \(10^{-16}\), and heat step
\(10^{16}\). Across these checks, the largest velocity L2 error was
\(1.33\times10^{-14}\), pressure error \(7.70\times10^{-14}\), and scalar error
below \(1.5\times10^{-15}\). These are separate native executions, not extra
portable-test counts.

Other real solver runs exercise PARDISO and NVIDIA GPU libraries where available.
See [performance](https://github.com/volpatto/pymhm/blob/main/docs/performance.md) for hardware, library versions, measured times
and residuals. A native integration skipped by pytest remains unverified in that
particular environment, even when its Python adapter has complete coverage.

## Reproducibility boundary

For the SPE10 flow extension, twelve independent DOLFINx/UFL P3/P3 assemblies
check constant and varying tensor resistance on two affine triangles for each
of the 2025, global-minimum 2017 and pointwise 2017 parameter conventions.
The largest relative matrix/load differences are 1.94×10⁻¹⁵/3.54×10⁻¹⁵;
identical physical quadrature points and independently computed inverse
constants are used. This checks the implemented formulas without identifying
one convention as the undocumented historical heterogeneous implementation.

An additional [SPE10 Brinkman control](cases/spe10.md) independently assembles
classical conforming P2/P1 Taylor–Hood elements with DOLFINx/UFL and solves them
with PETSc/MUMPS. It uses the same PDE, coefficient and physical boundary
conditions as the MHM case, with no residual stabilization. Successively refined
triangular meshes align with every material pixel; DG0 resistance and degree-four
quadrature therefore integrate the P2 reaction term exactly. This is a separate
classical reference, not the USFEM reference computation reported in the article.

Five fitted meshes reach 6,758,400 triangles and 30,435,203 unknowns. The last
reference refinement changes velocity and pressure by 0.3779% and 0.03572% in
relative L2, respectively. The fixed MHM fields differ from that finest reference
by 3.0299% and 1.0452%, using comparison quadrature order 32. The finest reference's
original free-equation residual is \(2.07\times10^{-12}\). These are measured
differences between computed fields, not certified errors against the exact
reservoir solution.

The inlet velocity is imposed strongly; vertical walls prescribe zero normal
velocity and natural tangential grad-grad traction. Zero total grad-grad traction
at the top fixes the additive pressure constant, so neither an extra mean gauge
nor a subsequent pressure shift is applied. Residual checks retain every free
velocity equation and every pressure equation from the original matrix. Exterior
flux balance and weak continuous-P1 incompressibility are reported separately;
they do not imply fine-cell or MHM macrocell conservation.

Two represented solutions verify these boundary conventions. Constant resistance
recovers \(u=(0,1)\), \(p=\gamma(2200-y)\), with zero forcing. A synthetic layered
patch uses \(\gamma=1\) below \(y=1100\), \(\gamma=3\) above, the same velocity,
and continuous pressure \(p=4400-y\) below and \(p=6600-3y\) above. On a
60-by-220 fitted grid, maximum nodal errors are \(8.35\times10^{-12}\) in
velocity and \(8.51\times10^{-11}\) in pressure; the original free-equation
relative residual is \(6.60\times10^{-12}\). This synthetic patch tests coefficient
assignment and boundary signs independently of the reservoir data. Its numerical
record is `examples/results/spe10/taylor-hood-layered-patch.json`.

Reference refinement differences use physical, area-weighted L2 integration.
MHM/reference comparisons preserve the broken MHM fields and repeat quadrature
on the nonmatching meshes. Successive reference differences, MHM distances and
display samples are distinct diagnostics; none is substituted for an exact
solution or a certified error bound.

The [SPE10 comparison](cases/spe10.md) matches the reported Darcy geometry,
spaces and boundary conditions, including the 9,738,625-unknown Q3 reference.
Its published pressure curve is compared at digitized coordinates with an
explicit raster uncertainty; original field coefficients and the historical
local Q1 refinement are unavailable. The separate
[Darcy flux study](cases/spe10-flux.md) integrates physical vector-field differences
against refined conforming Q3, native MSL_CG P1 and native NeoPZ RT0/P0 solutions.
It reports each reference's own refinement changes; these numerical fields are
not exact solutions. The published RT2 flux image uses different MHM spaces and
provides a qualitative comparison, not field-error data.
The [quarter-five-spot study](cases/quarter-five-spot.md)
checks point-well normalization against MSL and an analytical series, and checks
a separate obstacle problem against MSL and NeoPZ. The obstacle is an additional
benchmark, not a figure attributed to an article.

No exact-table claim is made for certified adaptive Oseen, unfitted
superconvergence or every mixed/locking-free elasticity table. Their acceptance
criteria remain in the literature catalog.
