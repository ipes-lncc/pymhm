# Development and release

Development uses Pixi 0.76.2, a `src` layout and Hatchling. The workspace and
pinned AmgX toolchain require the same Pixi version. Basix supplies native
reference-element tabulation as a standard dependency. It is loaded on demand;
the algebraic core requires no optional FEM compiler, MPI, CAD or accelerator
runtime. Functions and classes, including private and nested helpers, have
docstrings and type annotations. The package includes its typing marker and
typed root interface.
Pip installs the coefficient/Basix core. Symbolic `fenics-ufl` alone does not
provide native assembly; the [installation guide](installation.md#native-ufl-assembly)
describes DOLFINx environments for UFL applications.

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
- Combined Linux core and native FEM coverage has independent **99% minimums**
  for lines and branches.
- Analytical patches, convergence, conservation, gauges and independent
  uncondensed comparisons check numerical meaning beyond coverage.
- Native integration tests verify actual libraries and hardware independently
  of portable optional-dependency contract tests.
- MkDocs builds strictly; notebooks execute with nbclient and bounded cell timeouts.
- Wheel and source distribution metadata are checked, and every runtime module,
  typing stub and marker must match the current source tree byte for byte.

## Git hooks and editor commits

Install the Git hook once and run the checks with the portable environment:

```bash
pixi run --locked -e test-core pre-commit install
pixi run --locked -e test-core pre-commit run
```

The hooks come from `pre-commit-hooks` and the official `ruff-pre-commit`
repository, with pinned revisions. Pre-commit installs and manages their
isolated environments, so editor commits do not require Pixi on the GUI's
`PATH`. Ruff applies its safe lint fixes and formatting before a commit;
review and stage any resulting changes, then retry the commit. Its hook version
matches the portable `test-core` lockfile.

Release metadata validation remains available as
`pixi run --locked -e test-core metadata-check` and runs in CI.

## Complete native validation

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
Linux, Windows and macOS. It omits optional native integrations; its coverage
measurement is combined with native FEM coverage for qualification. `test-py311` and
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
results remain available. Workers that exit abnormally are not restarted: the
run fails and retains the failing test's report instead of retrying it.

### Coverage

`coverage-run` collects coverage without applying the package-wide gate.
`test-cov` collects coverage and checks both independent 99% thresholds; use it
with the complete native `test` environment. Portable Core runs and native FEM
runs each measure the package, and their combined Linux dataset qualifies
the package-wide coverage. Optional-backend skips remain explicit in each run.

From one checkout on Linux, collect both datasets into separate directories,
then combine them and check the thresholds:

```bash
pixi run --locked -e test-core coverage-run
pixi run --locked -e fem test-fem-cov
mkdir -p build/reports/qualified-coverage
pixi run --locked -e test-core python -m coverage combine --keep \
  --data-file=build/reports/qualified-coverage/.coverage \
  build/reports/coverage/.coverage build/reports/fem-coverage/.coverage
pixi run --locked -e test-core python -m coverage xml --fail-under=0 \
  --data-file=build/reports/qualified-coverage/.coverage \
  -o build/reports/qualified-coverage/coverage.xml
pixi run --locked -e test-core python -m coverage json --fail-under=0 \
  --data-file=build/reports/qualified-coverage/.coverage \
  -o build/reports/qualified-coverage/coverage.json
pixi run --locked -e test-core python scripts/check_coverage.py \
  build/reports/qualified-coverage/coverage.json
```

Both measurements must come from the same source revision and checkout paths.
`--keep` preserves the input datasets, and the combined output has its own
directory. The JSON checker enforces line and branch coverage separately.
CI combines only the Linux `test-core` and FEM datasets from the current
workflow run; Codecov receives their XML report after the gate passes.

Each runner starts from fresh, separate measurements for the two phases and combines
only successful phases. A failed parallel phase stops before the serial phase,
and an incomplete run does not publish coverage reports. The independent line
and branch thresholds remain **99%**. Numerical comparisons follow the
[scale and precision conventions](verification.md#numerical-tests); coverage
thresholds and scientific acceptance criteria are separate controls.
Larger literature campaigns run through the problem notebooks outside the CI
suite.

CI separates responsibilities into four workflows: [Tests](https://github.com/ipes-lncc/pymhm/actions/workflows/tests.yml),
[Lint and Quality](https://github.com/ipes-lncc/pymhm/actions/workflows/lint-and-quality.yml),
[Docs](https://github.com/ipes-lncc/pymhm/actions/workflows/docs.yml) and
[Publish to PyPI](https://github.com/ipes-lncc/pymhm/actions/workflows/publish-pypi.yml).
Tests, Quality and Docs run independently on pull requests and main-branch pushes.
They also expose `workflow_call` so release validation reuses the same checks.
The workflows use Pixi 0.76.2 and the checked-in lockfiles. Locked workspace
validation checks both the package and separate AmgX toolchain without installing
their environments.

Every job that installs Pixi environments enables the cache managed by
[`setup-pixi`](https://github.com/prefix-dev/setup-pixi/tree/v0.10.2#caching).
Its keys include the platform, requested environments, Pixi binary, lockfile
and environment paths. Changing these inputs creates a separate cache.
The workspace validation job installs no environments and disables this cache.
Release distributions and the documentation site pass between jobs as validated
artifacts.

The Tests portable matrix uses `test-core` on Linux x86-64, Windows x86-64 and
macOS Apple Silicon (ARM64), with additional Python 3.11 and 3.12 jobs on Linux.
The Integration matrix starts only after every Core matrix job succeeds. It
contains four native solver jobs: FEniCS/PETSc and MPI on Linux, and PARDISO
on Linux and Windows. Core jobs collect coverage on each tested platform. The
Linux FEM job collects native CPU coverage, and a separate Coverage job combines
the Linux `test-core` and FEM datasets and enforces both 99% gates after Core and
Integration succeed. `test-fem` selects CPU FEM tests without MPI;
`test-fem-cov` measures that selection in `build/reports/fem-coverage`,
leaving the core measurement in its own directory.
`test-mpi` selects CPU MPI tests, including runs with one, two and four ranks.
The FEM and MPI jobs upload separate parallel/serial JUnit reports even when a
test fails. The FEM job checks native imports before starting the suite and
leaves native standard error visible while capturing Python output. GPU
integrations run in the complete native suite.

Hosted integrations configure `UCX_TLS=tcp,sm,self` to use sockets, shared memory
and loopback communication. They require no RDMA hardware. This transport
selection applies only to the hosted Integration matrix; HPC and GPU runs use
their own runtime configuration. See the
[UCX transport conventions](https://openucx.readthedocs.io/en/master/faq.html#which-transports-does-ucx-use).

The complete CPU/GPU job is an explicit opt-in: dispatch Tests with `full_native`
enabled on a configured self-hosted Linux runner with the `gpu` label and two
NVIDIA devices. After all Core jobs succeed, it prepares AmgX, requires the
complete native stack and runs the whole suite with the same parallel/serial
phases and independent coverage gates. Native Gmsh/Netgen, FreeFEM/BAMG and
PyVista/VTK checks belong to this full Linux suite. FreeFEM provides optional
metric remeshing for adaptive studies; the PyMHM PDE solver remains independent
of it. Portable adapter contracts run in Core; native visualization on Windows
and macOS requires separate local qualification.
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

### Cite the literature on each page

Use an author–year citation beside the mathematical statement, theorem,
comparison or attributed figure. Link it to the publisher's DOI record or to
the exact version of an arXiv or HAL preprint. For example,
`[Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019)`
identifies the source without requiring a reader to interpret catalog codes.
Include a `## References` section on the same page with the full authors,
title, venue, year and persistent link for every work cited. Retain theorem,
equation and section numbers beside the claim they support.

The [literature catalog](literature.md) explains the scope of the methods; a
link to that catalog does not replace a page's bibliography. Distinguish a
preprint version from its journal publication, and distinguish an original
PyMHM application from a reproduction of an article's numerical experiment.
References in rendered introductory tutorials belong in their source
notebooks. Regenerate those pages with
`pixi run --locked -e introduction tutorials-render` after editing notebook
Markdown; preserve the recorded execution and numerical outputs when only
the exposition changes.

Identify each reference implementation by its project name, module, revision
and source URL when available. State whether the result comes from an unchanged
application, an instrumented driver, or an independently assembled restriction.
The [MSL comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/reference-comparison.md) identifies MSL_MHM,
MSL_CG and MSL_Core; the [NeoPZ comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/neopz.md) distinguishes its
RT0 driver from Labmec/MHM. Record source-access requirements when a repository
cannot be fetched anonymously. Matching a library name or polynomial degree
alone does not establish identical discrete equations.

Reference solver sources and comparison execution tools are maintained separately
from the public package. Public case studies retain the project names, revisions,
methods, numerical results and plots. The package's analytical verification and
native DOLFINx integration tests remain executable parts of its test suite.

## Distribution

### Prepare versions and release notes

The release tasks follow [SciAstro's release preparation strategy](https://github.com/volpatto/sciastro/blob/main/scripts/changelog.mjs).
Use the locked `release` environment, which provides Git and git-cliff 2.13.1:

```bash
pixi run --locked -e release release-fetch
pixi run --locked -e release changelog-preview
pixi run --locked -e release release-prepare 1.1.0
pixi run --locked -e release version-check
```

For the first release, before any canonical release tag exists on `origin/main`,
add `--initial` to both `changelog-preview` and `release-prepare`. For example,
`pixi run --locked -e release release-prepare 1.0.0 --initial` prepares an initial
`1.0.0` release. Subsequent releases require a version newer than the last
release. Official versions use `X.Y.Z`; their tags are exactly `vVERSION`.

Merge the changes to release into `main` before fetching and preparing. Preparation
uses the nearest canonical release tag on `origin/main`'s first-parent history
and generates notes from that exact commit to the fetched main tip. Tags on
other branches do not split the notes, and commits confined to the preparation
branch do not enter the range. Initial preparation uses the whole fetched main
history. A shallow checkout must fetch the full history first.

`cliff.toml` groups conventional commits into features, fixes, documentation,
tests, performance, refactoring and maintenance; other commits remain included.
Notes link to the source commits and retain breaking-change markers. Only the
marked generated block of the current release is replaced. Handwritten notes,
older releases and historical numerical records remain intact.

`release-prepare VERSION` synchronizes the versions in `pyproject.toml`,
`pixi.toml`, `src/pymhm/__init__.py`, `CITATION.cff`, the Conda recipe, the README
badge and text, and the documentation landing page. It updates `CHANGELOG.md`
in the same operation after validating every field. Failed writes restore the
original files. `version-set VERSION` is an alias for this full preparation;
`version-check` checks every current version and the release notes.

Review the generated diff, run the release gates, then commit the prepared files
and push `main` with its matching tag. The following example assumes preparation
on `main`. If preparation uses a separate branch, commit and merge it into `main`
before tagging the resulting main commit. The tasks create no commit or tag:

```bash
git add CHANGELOG.md pyproject.toml pixi.toml src/pymhm/__init__.py
git add CITATION.cff recipe/recipe.yaml README.md docs/index.md
git commit -m "chore(release): prepare 1.1.0"
git tag -a v1.1.0 -m "PyMHM 1.1.0"
git push origin main v1.1.0
```

### Automatic publication

The `publish-pypi.yml` workflow runs for a `v*` tag pushed to
`ipes-lncc/pymhm`. It validates synchronized versions, usable release notes and
the tagged commit's inclusion in `main`. Tests, Lint and Quality, and Docs
validation then run in parallel. Quality builds and checks the distributions;
after all checks pass, PyPI receives those same artifacts through trusted
publishing. The workflow then creates a GitHub Release using the matching
CHANGELOG section and checked wheel/sdist, and deploys the documentation through
`github-pages`. Official versions are published as regular GitHub Releases.
Push one release tag at a time; release workflows share a single concurrency
group and never cancel an active publication.

Ordinary pushes to `main` check versions and notes without publishing a package.
The PyPI job runs directly in `publish-pypi.yml`.

Both release jobs that call the reusable Docs workflow declare `contents: read`,
`pages: write` and `id-token: write`. GitHub validates the called workflow's
permission requirements before evaluating whether its deployment job will run.
The documentation build limits its own token to `contents: read`; the acceptance
call passes `publish=false`, and only the final publication call passes
`publish=true`. These job permissions are declared in the workflows and require
no change to the repository's default token permissions.

A workflow rerun uses the original run's commit and workflow revision. Pushing
corrected files to `main` does not change the workflow associated with an existing
release tag. Keep published release tags immutable.
See [GitHub's workflow rerun rules](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).

If a release failed before publishing any distributions or release artifacts,
the same version can be used again. Confirm that the version has no published
files on PyPI and no GitHub Release, then commit the correction on `main`.
Also check the project's upload history: [PyPI permanently reserves uploaded
distribution filenames](https://pypi.org/help/#file-name-reuse), even when their
files or project have been deleted. A public `404` does not establish that a
filename has never been used. A deleted distribution cannot be uploaded again
under the same filename; publishing the standard wheel and source archive then
requires a version whose filenames have not previously been uploaded.
Keep the prepared version metadata and CHANGELOG section; do not run
`release-prepare` again, because that command rejects an existing release tag.
For an unpublished `v1.1.0`, run these commands in Bash after committing:

```bash
pixi run --locked -e release version-check
previous_release_tag=$(git rev-parse refs/tags/v1.1.0)
git tag -f -a v1.1.0 -m "PyMHM 1.1.0" main
git push --atomic \
  --force-with-lease="refs/tags/v1.1.0:$previous_release_tag" \
  origin main refs/tags/v1.1.0
```

The explicit [Git lease](https://git-scm.com/docs/git-push) guards only the tag
against concurrent changes; `main` retains its normal fast-forward protection.
The atomic push updates both refs together or neither. The updated tag triggers
a fresh release run using the corrected workflow and the same package version.

Before a release, configure the [PyPI trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
with owner `ipes-lncc`, repository `pymhm`, workflow `publish-pypi.yml` and
environment `pypi`. The publisher's GitHub owner must match this organization;
an existing publisher registered to a different owner must be replaced.

Configure [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
in **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The documentation URL is **https://ipes-lncc.github.io/pymhm/**. In
**Settings → Environments → github-pages**, allow `main` and release tags
matching `v*` in the deployment rules. Ensure release jobs can run automatically
under both environments' protection rules. The Pages job declares `pages: write`
and `id-token: write`; the PyPI job declares `id-token: write`.

To publish documentation before the next release, open **Actions → Docs** and
select **Run workflow**, choose `main` or a `v*` release tag, and enable `publish`.
The workflow builds and checks the site before deploying it to GitHub Pages.
The manual input defaults to `false`; an ordinary manual run validates the site
without deploying.
Publication is restricted to `ipes-lncc/pymhm` and these refs.

Ordinary branch pushes and pull requests validate documentation without deploying.
Manual dispatches never publish a Python package. The optional GPU job is
qualified separately and does not block tag releases. The Conda recipe builds a
portable package and tests its installed import/dependencies; conda-forge acceptance requires a
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
