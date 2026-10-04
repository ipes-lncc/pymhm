# Development and release

Development uses Pixi, a `src` layout and Hatchling. Basix supplies native
reference-element tabulation as a standard dependency. It is loaded on demand;
the algebraic core requires no optional FEM compiler, MPI, CAD or accelerator
runtime. Functions and classes, including private and nested helpers, have
docstrings and type annotations. The package includes its typing marker and
typed root interface.

## Quality gates

- Ruff checks style, imports, common mistakes and documentation.
- The recursive source contract checks docstrings and annotations on every
  module and named definition, including implementation helpers.
- Mypy checks annotations against the resolved development environment.
- Pytest measures branch and line coverage, each with an independent **99% minimum**.
- Analytical patches, convergence, conservation, gauges and independent
  uncondensed comparisons check numerical meaning beyond coverage.
- Native integration tests run separately from optional-dependency contract tests.
- MkDocs builds strictly; notebooks execute with nbclient and bounded cell timeouts.
- Wheel and source distribution metadata are checked, and every runtime module,
  typing stub and marker must match the current source tree byte for byte.

```bash
pixi run -e test lint
pixi run -e test format-check
pixi run -e test typecheck
pixi run -e test test-cov
pixi run -e docs docs-check
pixi run -e test build
```

The coverage task uses two test workers and one native numerical thread per
worker. Each file's tests stay together; numerical tolerances and the two
coverage thresholds are identical to a serial run. Larger literature campaigns
run through their example commands, outside the CI test suite.

The CI matrix exercises the core on native Linux, Windows and macOS runners and
supported Python versions. Native FEniCS/PETSc and meshing jobs run on Linux;
PARDISO integration runs on Linux and Windows. The
manually dispatched GPU workflow requires a configured self-hosted Linux runner
with the `gpu` label and an NVIDIA device. A skipped integration test is not a
passed backend verification.

## Reproducible science

Every new paper comparison records formulation, coefficients, geometry, boundary
conditions, gauges, spaces, quadrature, refinement and solver tolerances. Separate
verification of an analytical problem from reproduction of a specific table.
Do not use rounded paper values as an exact algebraic oracle. Store numerical
outputs with environment and hardware provenance when measuring performance.

Identify each reference implementation by its project name, module, revision
and source URL when available. State whether the result comes from an unchanged
application, an instrumented driver, or an independently assembled restriction.
The [MSL comparison](https://github.com/volpatto/pymhm/blob/main/docs/cases/reference-comparison.md) identifies MSL_MHM,
MSL_CG and MSL_Core; the [NeoPZ comparison](https://github.com/volpatto/pymhm/blob/main/docs/cases/neopz.md) distinguishes its
RT0 driver from Labmec/MHM. Record source-access requirements when a repository
cannot be fetched anonymously. Matching a library name or polynomial degree
alone does not establish identical discrete equations.

Reference solver sources and comparison execution tools are maintained separately
from the public package. Public case studies retain the project names, revisions,
methods, numerical results and plots. The package's analytical verification and
native DOLFINx integration tests remain executable parts of its test suite.

## Distribution

The PyPI workflow uses trusted publishing and a version-tag check. It does not
upload on ordinary pushes. The Conda recipe builds a portable package and tests
its installed import/dependencies; conda-forge acceptance requires a feedstock
review. Release maintainers must validate optional runtime compatibility and
publish notes describing the actual scientific scope.
Inspect the source archive as well as the installed runtime: reference solver
sources and comparison runners are excluded from both version control and release
artifacts. Plotting archived numerical results does not require those runners.

Source distributions contain the Python implementation, tests, scientific
examples, compact numerical records and documentation sources. Large field
archives and the rendered figure gallery are available from the repository and
published documentation rather than included in the PyPI archives. Build the
complete documentation from a repository checkout. The package wheel contains
only the runtime modules, typing marker and distribution metadata.

See the repository `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff` and CI
workflows for the maintained commands and policies.
