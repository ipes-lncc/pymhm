# Contextual API qualification

The explicit equation API and mesh-associated contexts use the same numerical
assembly, condensation and solver owners. The controls in
[`equivalence.json`](equivalence.json) compare their main Darcy, SPE10, elasticity,
MsHHO and MH²M examples with unchanged meshes, coefficients, spaces and boundary
data. They identify the explicit notebook revision, current provider snapshots,
common numerical owners, tolerances and measured differences.

Native coefficients are compared after transport into a common physical basis.
For Darcy, SPE10 and elasticity, deterministic interior samples additionally
compare field values and gradients. These sampled differences are representation
controls, rather than integrated error estimates or literature convergence rates.
Both providers execute the current shared owners; this comparison does not run
the old installed distribution. Full current introductory notebooks separately
verify their equations, physical norms and reference-refinement studies.

[`layer-local-equivalence.json`](layer-local-equivalence.json) checks RAD and
Stokes–Brinkman in Galerkin and USFEM variants at boundary and interior
macrocells, with mild and small-diffusion/viscosity coefficients. The sixteen
controls compare the complete declared local blocks, source responses, trace
lifts and retained bases in their common coordinate convention. They establish
local representation agreement, rather than a new stability or convergence claim.

[`rad-physical-norms.json`](rad-physical-norms.json) compares the fresh current
notebook with previously archived physical norms on its twenty-two matching
cases. The largest relative change in scalar, flux and gradient norms is
2.13 × 10⁻¹⁵. Original report provenance remains explicit; this comparison is
result preservation and does not claim a fresh execution of the old package.

[`brinkman-physical-norms.json`](brinkman-physical-norms.json) compares
twenty-four common Stokes–Brinkman cases. Velocity, pressure, velocity-gradient,
divergence and pseudo-stress norms are compared separately where reported.
The largest MHM relative norm difference is 7.59 × 10⁻¹¹, corresponding to
a velocity L2 norm change of 5.53 × 10⁻¹⁴. The largest reference relative
difference is 6.67 × 10⁻¹⁰, a pressure L2 norm change of 8.21 × 10⁻¹⁵.
Gauge and macro-conservation diagnostic differences remain below
1.15 × 10⁻¹⁵. The bounded single-element family compares three common levels
per degree; the full original sequence remains an explicit notebook profile.

[`tutorial-execution.json`](tutorial-execution.json) identifies ten executed
introductions and the custom-interface example: 271 code cells, 62 figures
and no execution errors. Current cell sources match the executed copies.
[`figure-provenance.json`](figure-provenance.json) identifies the numerical
acquisition and convergence-plot source snapshots separately; the latter
uses the same acquired physical rows and persisted states.
The [RAD profile/publication receipt](rad-figure-provenance.json) similarly
identifies its numerical acquisition and final publication sources. Its ten
checksummed archives replay through the shared owner, preserving the physical
rows, persisted bases and profile image.

[`documentation.json`](documentation.json) records the strict documentation
build, generated-page verification and actual Chrome inspection. The markup
check covers 59 pages and 939 expressions; nineteen current API/tutorial and
performance pages render 464 expressions without MathJax errors, missing
containers or display overflow at 1440×1000. All 62 tutorial figures load,
and final viewport inspection checks macro meshes, one-sided profiles,
legends, labels and independent color scales.

[`route-cost.json`](route-cost.json) compares a warmed, serial 4×4 Darcy
macro mesh with Q2 fields and 4×4 fine cells per macroelement. The explicit
route assembles boundary blocks with the existing Basix owner; the contextual
route assembles independently declared UFL pairings. Three alternating
repetitions give median timed-operation costs of 0.979 s and 6.043 s,
respectively. Native form lookups/bindings account for 0.561 s and 5.186 s;
local condensation and global solution costs are essentially unchanged.
This is a measured 6.17× cost increase for this small workload. It changes
the boundary assembly route as well as the API, rather than isolating Python
abstraction overhead. The receipt states untimed common setup, nested timers,
native initialization, hardware and numerical equality checks. These samples
do not establish parallel scalability or the cost of large local problems.

The contextual API supports automatic planar polynomial trace integration and
the documented native Lagrange field descriptors. Custom interfaces, unsupported
face families and moment-based field coordinates supply explicit capabilities.
See the [API overview](../../../docs/tutorials/overview.md) and
[custom-interface tutorial](../../../docs/tutorials/custom-interface.md).
