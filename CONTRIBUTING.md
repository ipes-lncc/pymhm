# Contributing

Use Pixi 0.76.2 and commit changes to both `pixi.toml` and `pixi.lock` when
dependencies change. The workspace and AmgX toolchain require this version.
The Pixi runtime constraints are the source of truth for the matching
PyPI metadata; `metadata-check` detects drift.

```bash
pixi --version
pixi list --locked --no-install -e test-core
pixi install --locked -e test-core
pixi run --locked -e packaging lock-check
pixi run --locked -e test-core metadata-check
pixi run --locked -e test-core lint
pixi run --locked -e test-core format-check
pixi run --locked -e test-core typecheck
pixi run --locked -e test-core test-cov
pixi run --locked -e test-core build
pixi run --locked -e test-core check-dist
```

CI runs these portable checks on Linux x86-64, Windows x86-64 and macOS Apple
Silicon (ARM64). Full native acceptance uses the Linux CUDA `test` environment,
including two-device GPU tests,
FEniCS/PETSc, PARDISO, meshing, visualization and FreeFEM integrations. Follow
[the development guide](docs/development.md) to prepare AmgX, check every required
dependency and run the complete suite. Both suites use all available CPU workers;
mark a test `serial` only when shared native resources require isolation.

After computing and accepting the scientific cases needed by a notebook or page,
run its selected notebook and `pixi run -e docs docs-check`. Review rendered
figures and MathJax as well as generated markup. CI checks documentation markup;
scientific acquisition, notebook execution and full gallery generation require
their separate numerical and rendering acceptance.

Organize numerical operations by responsibility in the packages described in
[the architecture guide](docs/architecture.md). Prefer free functions and explicit
delegation; store problem, configuration and solution data in small objects.
Write instructional examples as notebooks under `notebooks/<problem>/`, grouping
the methods for each physical problem. Update `notebooks/catalogue.json` and
`notebooks/README.md`. Keep reusable acquisition, archive-reading and plotting
helpers in support modules or private tools; importable worker callables preserve
spawn execution. Keep numerical algorithms in the package rather than copying
their formulas into notebook cells.
Import each operation from the module that owns its implementation. Keep the
root exports focused on the generic variational and numerical interfaces; do
not introduce module aliases or files that only re-export another module.

Document every class and function, including private and nested helpers, with mathematical conventions,
array shapes, coefficient assumptions, units where applicable, and failure
conditions. The `source-check` dependency of `lint` validates these contracts
and annotations recursively. Include reproducible tests for any numerical change. Coverage must
remain at least 99% with branches enabled; coverage alone does not establish
numerical correctness. Tests should include exact solutions, conservation,
residuals, nullspaces, and convergence rates when relevant.

Keep optional runtimes behind explicit backend selection. CPU installation must
remain usable without MPI, FEniCS, a proprietary solver, or GPU drivers. Tests of
optional backends must distinguish dependency availability from successful
execution on actual hardware.

Use `pixi run -e fem test-fem` to exercise DOLFINx integration on supported Unix
platforms. Contributions to accelerated backends should report hardware,
software versions, thread counts, warmup, transfer costs, and correctness checks.

Add documentation and executable problem notebooks for new formulations. Identify
whether a case is a manufactured verification, an independently reproduced
published result, or an exploratory calculation. Provide citations and explicit
tolerances; do not infer validation from a visually plausible field.

Place readable author–year citations beside the statements or figures they
support, linked to a DOI or an explicitly versioned preprint. Add full
bibliographic entries under `## References` on each page that cites literature.
Keep theorem and equation numbers with their claims. For introductory
tutorials, edit the source notebook's Markdown and regenerate its rendered
documentation; see the [documentation citation conventions](docs/development.md#cite-the-literature-on-each-page).

## Releases

Merge the changes being released into `main`, then prepare the version and notes
from its fetched history. Use the locked `release` environment, which supplies
Git and git-cliff; preparation creates no commit, tag or publication:

```bash
pixi run --locked -e release release-fetch
pixi run --locked -e release changelog-preview
pixi run --locked -e release release-prepare 1.1.0
pixi run --locked -e release version-check
```

Replace `1.1.0` with the intended version. Official releases use canonical
`X.Y.Z` versions and the tag must be exactly `vVERSION`.
For the first release, before a canonical
release tag exists on `origin/main`, use `--initial` for both commands:

```bash
pixi run --locked -e release changelog-preview --initial
pixi run --locked -e release release-prepare 1.0.0 --initial
```

Notes cover the nearest canonical release on the first-parent history of
`origin/main` through its fetched tip. Commits confined to a preparation branch
are excluded. If `main` advances, fetch and incorporate it before regenerating
the pending release. Subsequent versions must be newer than the previous release;
never rewrite a published release tag.

`release-prepare` synchronizes seven files: `pyproject.toml`, `pixi.toml`,
`src/pymhm/__init__.py`, `CITATION.cff`, `recipe/recipe.yaml`, `README.md` and
`docs/index.md`. It also updates the matching `CHANGELOG.md` section.
Regenerating a pending version replaces its marked generated block while
preserving handwritten notes and older releases. Add scientific scope,
limitations and migration instructions outside that block. `version-set` is an
alias for the same preparation; use these tasks instead of editing versions
individually.

Run the portable quality, coverage and distribution checks above, then validate
the documentation, workflows and Conda artifact:

```bash
pixi run --locked -e release version-check
pixi run --locked -e docs docs-check
pixi run --locked -e packaging ci-check
pixi run --locked -e packaging conda-build
```

For native-backend changes, qualify the complete suite on the supported Linux
CUDA runner, with AmgX prepared and all dependencies available, using
`pixi run --locked -e test test-cov`. Portable `test-core` acceptance covers the
portable implementation; its optional-backend skips do not validate those
backends. Both line and branch coverage must remain at least 99%.

Review the eight prepared files and commit them on the preparation branch:

```bash
git add CHANGELOG.md pyproject.toml pixi.toml src/pymhm/__init__.py
git add CITATION.cff recipe/recipe.yaml README.md docs/index.md
git commit -m "chore(release): prepare 1.1.0"
```

Merge that branch into `main` before tagging. If preparation was performed
directly on `main`, the reviewed commit is already there. From the synchronized
`main` checkout, check the version again and push its annotated tag:

```bash
git switch main
git pull --ff-only origin main
pixi run --locked -e release version-check
git tag -a v1.1.0 -m "PyMHM 1.1.0"
git push origin main v1.1.0
```

Adjust the version consistently in these commands. Push one release tag at a
time. A `v*` tag pushed to `ipes-lncc/pymhm` triggers validation of synchronized
versions, usable changelog notes and inclusion in `main`. Tests, Lint and
Quality, and Docs checks then run in parallel. On success, the workflow publishes
the checked wheel and sdist to PyPI, creates a GitHub Release with their notes and
artifacts, and deploys the documentation to GitHub Pages. Official releases are
published as regular GitHub Releases. Ordinary pushes to `main` validate the
changes without publishing.

Rerunning a workflow uses its original tagged commit and workflow revision.
Pushing a fix to `main` does not update an existing tag's workflow. If a release
failed before publishing any distributions or release artifacts, its version can
remain unchanged: commit the correction on `main`, then update the unpublished
tag with the guarded push described in the
[distribution guide](docs/development.md#automatic-publication).
Keep published release tags immutable. The release workflow already declares
the token permissions needed by its reusable workflows and publishing jobs.
PyPI permanently reserves previously uploaded filenames, including deleted
files. A missing public release is insufficient to establish that its filenames
are available; consult the upload history before retrying the same version.

Before the first publication, configure the PyPI trusted publisher with owner
`ipes-lncc`, repository `pymhm`, workflow `publish-pypi.yml` and environment
`pypi`. Set GitHub Pages' source to **GitHub Actions** and allow `main` and tags
matching `v*` in the `github-pages` environment. The
[distribution guide](docs/development.md#distribution) documents these settings
and the optional manual Docs run with `publish=true`.

The complete CPU/two-GPU job runs only on manual dispatch with `full_native`
enabled; it does not run for ordinary pushes or tags.

Keep release artifacts limited to installation and build requirements. The wheel
contains every runtime module, typing file and distribution metadata. The source
archive contains `src/pymhm`, `pyproject.toml`, `README.md`, `LICENSE`, the
backend-required `.gitignore` and generated package metadata. `check-dist` validates
the allowed paths and their source bytes;
tests, scripts, examples, benchmarks, notebooks, documentation, recipes, roadmap
and Pixi files must remain outside both artifacts. Optional backend integrations
ship as runtime code, with their dependencies installed separately.

Run tests, scientific acquisitions and documentation builds from a repository
checkout. Archived campaign and notebook checks require locally generated fields
and any additional figures, as described in the installation guide. Git contains
source code, JSON records, compact input layers and the selected figures required
by the published documentation. Large computed fields and intermediate outputs
remain outside Git. [ROADMAP.md](ROADMAP.md) defines implementation and
validation priorities, dependencies and acceptance criteria. Do not add checkpoints,
comparison tools or native environments to commits.
Check both archive contents and installation from the built distribution.

The local Conda recipe is built with `pixi run -e packaging conda-build` and
produces a `noarch: python` core package. Before a conda-forge submission, replace
its local source with the released source archive URL and verified SHA-256, then
follow the conda-forge staged-recipes review. The recipe is preparation for that
process; it does not imply that a feedstock or published Conda package exists.

Automated CI runs source tests and packaging checks without computed scientific
outputs. Run a case's public acquisition command, verify its fields, norms,
reference and convergence criteria, then run its plotting command and inspect
the rendered figure. A plotting command that reads an existing JSON record does
not recompute the physical field or verify a previous result.

Before a selected notebook execution,
`pixi run -e notebooks python scripts/notebook_data.py --check` reports whether
its required fields exist. Missing inputs must be generated by their corresponding
public calculation; no automatic result download is configured. Documentation
builds use the selected publication figures in Git. Regenerate and inspect a
changed case's accepted figures before including them in the published gallery.
An analytical verification must remain labeled as such; it does not replace a
matched published benchmark. Source/coverage gates alone do not establish that
all literature capabilities or reproductions are complete.
