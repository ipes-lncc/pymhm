# Development and release

Development uses Pixi 0.76.2, a `src` layout and Hatchling. The workspace and
pinned AmgX toolchain require the same Pixi version. Basix supplies native
reference-element tabulation as a standard dependency. It is loaded on demand;
the algebraic core requires no optional FEM compiler, MPI, CAD or accelerator
runtime. Functions and classes, including private and nested helpers, have
docstrings and type annotations. The package includes its typing marker and
typed root interface.

Use the checked-in lockfiles for every environment. Validate workspace resolution
without installing packages or changing the lockfile before preparing an environment:

```bash
pixi --version
pixi list --locked --no-install -e test-core
pixi list --locked --no-install --manifest-path tools/amgx/pixi.toml
```

Dependency changes require matching manifest and lockfile updates. An environment
command with `--locked` fails if their resolutions disagree.
The `pixi run --locked -e packaging lock-check` task performs both checks; its
`amgx-lock-check` dependency validates the separate native build toolchain.

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

CI separates responsibilities into four workflows: [Tests](https://github.com/volpatto/pymhm/actions/workflows/tests.yml),
[Lint and Quality](https://github.com/volpatto/pymhm/actions/workflows/lint-and-quality.yml),
[Docs](https://github.com/volpatto/pymhm/actions/workflows/docs.yml) and
[Publish to PyPI](https://github.com/volpatto/pymhm/actions/workflows/publish-pypi.yml).
Tests, Quality and Docs run independently on pull requests and main-branch pushes.
They also expose `workflow_call` so release validation reuses the same checks.
The workflows use Pixi 0.76.2 and the checked-in lockfiles. Locked workspace
validation checks both the package and separate AmgX toolchain without installing
their environments.

The Tests portable matrix uses `test-core` on Linux, Windows and both macOS
architectures, with additional Python 3.11 and 3.12 jobs on Linux. The integration
matrix exercises FEniCS/PETSc, MPI, meshing and FreeFEM on Linux, PARDISO on Linux
and Windows, and PyVista/VTK on all four platforms. Every portable coverage job
enforces the independent 99% line and branch gates.

The complete CPU/GPU job is an explicit opt-in: dispatch Tests with `full_native`
enabled on a configured self-hosted Linux runner with the `gpu` label and two
NVIDIA devices. It prepares AmgX, requires the complete native stack and runs the
whole suite with the same parallel/serial phases and independent coverage gates.
It is not scheduled by an ordinary push, pull request or release tag. A skipped
integration test does not constitute passed backend verification. Scientific
acquisitions, notebook execution, field and figure acceptance, and browser-side
MathJax inspection remain separate from CI's documentation markup checks.

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

The `publish-pypi.yml` release workflow runs for a `v*` tag pushed to
`volpatto/pymhm`. It calls Tests and Lint and Quality in parallel for that revision.
The quality gate validates the version tag against release metadata and uploads
the checked distributions. Once both gates succeed, the release workflow calls
Docs with `publish=true` to build, validate and deploy the documentation artifact
to GitHub Pages through `github-pages`. The final PyPI job waits for Tests,
Quality and Docs, downloads the checked Python distributions and publishes them
through the configured `pypi` environment. The PyPI publishing job runs directly
in `publish-pypi.yml`.

Before a release, configure the PyPI trusted publisher with owner `volpatto`,
repository `pymhm`, workflow `publish-pypi.yml` and environment `pypi`. Enable Pages in
**Settings → Pages → Build and deployment → Source: GitHub Actions**.
Allow release tags in the `github-pages` environment's deployment rules, and
ensure the release jobs can run automatically under both environments' protection
rules. The Pages job declares `pages: write`
and `id-token: write`; the PyPI job declares `id-token: write`.

Ordinary branch pushes, pull requests and manual dispatches do not publish a
package or deploy documentation. The optional GPU job is qualified separately
and does not block tag releases. The Conda recipe builds a portable package and
tests its installed import/dependencies; conda-forge acceptance requires a
feedstock review. Release maintainers must validate optional runtime compatibility
and publish notes describing the actual scientific scope.
Inspect the source archive as well as the installed runtime: reference solver
sources and comparison runners are excluded from both version control and release
artifacts. Plotting archived numerical results does not require those runners.

The wheel contains every `pymhm` runtime module, typing stub and marker, plus
distribution metadata and the license. The source distribution contains only
`src/pymhm`, `pyproject.toml`, `README.md`, `LICENSE`, the backend-required `.gitignore`
and generated package metadata.
Build an installable wheel from that source archive without a repository checkout.
`check-dist` verifies allowed archive paths and exact runtime source bytes; its
source-archive rebuild check verifies identical wheel contents without repository
resources.

Tests, scripts, examples, benchmarks, documentation, notebooks, recipes, roadmap
and Pixi files are repository resources and are excluded from both release
archives. Use a checkout for those workflows, including scientific acquisitions,
field replay and documentation builds. The full runtime includes the optional
backend adapters; their native libraries are separate installation requirements.

The selected figures required by the published gallery are versioned in an
explicit `.gitignore` allowlist, so documentation builds need no field acquisition.
Regenerate them from accepted scientific records when a case changes and inspect
the rendered figures before updating the allowlist. Large field archives and
intermediate outputs remain outside Git. Documentation and publication figures
are excluded from Python and Conda installation artifacts.

See the repository `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff` and CI
workflows for the maintained commands and policies.
