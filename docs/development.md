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
- Native integration tests verify actual libraries and hardware independently
  of portable optional-dependency contract tests.
- MkDocs builds strictly; notebooks execute with nbclient and bounded cell timeouts.
- Wheel and source distribution metadata are checked, and every runtime module,
  typing stub and marker must match the current source tree byte for byte.

The complete `test` environment targets Linux CUDA hosts and includes the
FEniCS/PETSc/MPI, PARDISO, AMG, meshing, FreeFEM, visualization and accelerator
stacks. Two NVIDIA devices are required for the native multi-GPU tests. Install
its locked dependencies and build the pinned AmgX/PyAMGX integration before the
first complete run:

```bash
pixi install --locked -e test
pixi run --locked -e test test-setup-amgx
pixi run --locked -e test test-dependencies
pixi run --locked -e test lint
pixi run --locked -e test format-check
pixi run --locked -e test typecheck
pixi run --locked -e test test-cov
pixi run --locked -e docs docs-check
pixi run --locked -e test build
```

`test` and `test-cov` require the dependency check; a missing native library,
MPI launcher or CUDA device fails before collection. This profile does not turn
unavailable native integrations into successful backend validation. The
`test-core` environment provides the portable suite and development tools on
Linux, Windows and macOS. It omits optional native integrations; their capability
skips are separate from its line and branch coverage gates. `test-py311` and
`test-py312` provide portable checks for those Python versions. The package's
runtime dependencies remain independent of these full-test requirements.

The test tasks run two separate phases: tests without a `serial` mark use all
CPUs available to the process by default, and `@pytest.mark.serial` tests run
in a fresh pytest process after every worker has exited. The runner sets one
native numerical thread per process. The `loadscope` scheduler keeps module
fixtures on one worker so independent checks reuse scientific acquisitions
without sharing mutable data between processes. Serial marks are for shared
native resources, exclusive subprocess campaigns and other tests that cannot overlap safely; computational
cost alone does not require a serial mark.

```bash
pixi run --locked -e test test
pixi run --locked -e test test --workers 8 -- -k conservation
pixi run --locked -e test-core test --distribution worksteal -- -m "not fem"
```

`PYMHM_TEST_WORKERS` or `--workers` can set a smaller worker count for resource
planning; requested counts are capped to the CPUs available to the process,
including Linux affinity restrictions. Direct
`pytest -q` executes the whole selected suite sequentially. A direct xdist run
must select `-m "not serial"`; attempting to distribute selected serial tests is
an error. The phase filter respects user `-k` and `-m` selections. Requested
JUnit XML reports receive `-parallel` and `-serial` filename suffixes so both
results remain available.

Coverage starts from fresh, separate measurements for the two phases and combines
only successful phases. A failed parallel phase stops before the serial phase,
and an incomplete run does not publish coverage reports. The independent line
and branch thresholds remain **99%**, with unchanged numerical tolerances.
Larger literature campaigns run through the problem notebooks outside the CI
suite.

The CI matrix uses `test-core` on Linux, Windows and both macOS architectures,
with additional Python 3.11 and 3.12 jobs on Linux. Native FEniCS/PETSc, MPI,
meshing and FreeFEM jobs run in their CPU environments; PARDISO integration runs
on Linux and Windows. The manually dispatched full native workflow requires a
configured self-hosted Linux runner with the `gpu` label and two NVIDIA devices.
It prepares AmgX, requires the complete native stack and runs the whole suite
with the same parallel/serial phases and independent coverage gates. A skipped
integration test does not constitute passed backend verification.

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
