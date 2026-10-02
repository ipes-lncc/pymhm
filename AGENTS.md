# Scientific development

Use Pixi environments and the checked-in lockfile. Keep the portable core
independent of optional FEM, CAD, MPI and accelerator imports. Public functions
and classes require docstrings, type annotations and explicit numerical
conventions.

Before claiming a formulation works, verify the operator, local kernel, trace
orientation, boundary convention, physical gauge and discretization compatibility.
Distinguish macro conservation from fine-cell conservation, raw gradients from
H(div) fluxes, and the physical flux from Robin or pseudo-traction multipliers.
Derive manufactured sources independently and choose adequate assembly/error
quadrature. A small algebraic residual does not establish uniqueness or inf-sup
stability.

When extending a formulation between dimensions, recheck every dimension-dependent
degree, trace, regularity and inverse-inequality hypothesis against the source
theorem. Centralize shared conditions and test the admissible boundary and the
immediately excluded cases in each dimension. Distinguish existence of an
algebraic reconstruction from the hypotheses of its error estimates. Audit all
callers, examples, notebooks and scientific claims when such a condition changes.

Run the relevant tests and the line/branch coverage gates (both >=99%). Do not add
coverage exclusions or relax solver tolerances to hide numerical failures. Use
native backend integrations in addition to optional API contract tests. Preserve
cross-platform spawn semantics and manage native resources explicitly.

Fix numerical defects in the shared implementation that owns the operation,
then reuse that implementation in every affected solver, example and comparison.
Do not leave case-specific workarounds or duplicate corrected formulas in drivers.
Add a regression check for the underlying invariant, audit related call sites,
and rerun affected cases, including homogeneous and nonhomogeneous boundary data
when relevant. Regenerate their numerical records, figures and notebooks from
the corrected code. Published cases contain the current correct results and
precise method provenance; obsolete incorrect outputs and debugging history do
not belong in the case gallery. Apply this workflow to reference adapters too,
while keeping private comparison tools outside release artifacts.

Treat a numerical basis as part of a persisted coefficient vector's data
contract. Fix otherwise arbitrary nullspace orientations through declared
moments, and archive the executed basis matrix with a digest. Replaying fields
must use that matrix consistently in orientation maps and evaluation. Verify
replay across BLAS thread counts and equivalent rotations of computed
nullspaces; matching dimensions or source hashes alone is insufficient.

Apply the same editorial rule throughout public documentation: describe the
current methods, supported inputs, reproducible procedures, results and scientific
limitations. Exclude development diaries, attempted fixes, debugging narratives
and accounts of defects found during implementation. Preserve useful mathematical
conditions as present-tense explanations; do not present obsolete behavior or
conversation-specific instructions as part of the user documentation.

Document verified capabilities separately from literature results and planned
extensions. A paper reproduction requires matching its discretization and data;
an analytical problem on a different mesh must be labeled accordingly. Performance
reports include setup, transfer, synchronization and reproducible provenance;
report regressions and absent speedups as measured.
Treat matched literature reproduction as the primary scientific acceptance
criterion. If investigation cannot reconcile a published result, record the
specific unresolved input or discrepancy and compare the same physical case
against an independently assembled reference implementation. Match geometry,
coefficients, boundary data, approximation spaces and field norms when claiming
discrete agreement. A small patch or a different analytical problem does not
replace that comparison. Verify the convergence rates stated in the literature
when their regularity and discretization hypotheses apply; do not transfer smooth
problem rates to singular or heterogeneous cases without justification.
Identify reference codes by project name, module, revision and verified source
URL. Distinguish code inspection from execution, and an original application
from an instrumented comparison driver. Record source-access limitations.
Keep external reference solver sources and comparison execution tools outside
the versioned repository and release artifacts. Publish attribution, revisions,
numerical results and plots separately. Preserve the package's own analytical
tests and native backend integration tests.
Spatial figures must highlight the actual macro mesh on analytical, numerical
and error panels. Mark macroface intersections on profile plots and preserve
independent one-sided values instead of smoothing reconstructed interfaces.
Use "velocity" and "velocity magnitude" in scientific labels, metadata and
explanations. Use "flux" for Darcy flux and its magnitude where appropriate.
Inspect final rendered figures at their intended viewing and publication size.
Reserve separate regions for fields, axes, titles, legends and color scales;
check every regenerated panel for overlap, clipping and illegible labels.
Repeated field names must not silently merge independent subplot colorbars.

Write display mathematics in standalone `$$` blocks with blank lines before and
after each block. In Markdown tables use explicit TeX commands for norm bars,
absolute values and restrictions so they cannot become column separators.
Run `docs-check` to validate generated mathematical markup, and inspect MathJax
rendering in the browser: every expression must render without error. Break long
equations into aligned lines rather than relying on horizontal scrolling at
ordinary desktop widths. A successful MkDocs build alone does not compile TeX.

For a problem without an exact solution, compute a classical conforming
reference on several sufficiently fine meshes with the same operator, material
and boundary conditions. Verify its own refinement error before using it as
a baseline; compare physical fields through stated norms and integration
checks. A fine numerical reference is not an exact solution. Preserve the
published MHM spaces in the comparison instead of changing them to fit a figure.

Use `pixi run -e test lint`, `format-check`, `typecheck`, `test-cov`, and
`pixi run -e docs docs-check` before declaring changes ready. Build and inspect
release artifacts with the package tasks. Keep release metadata synchronized.
