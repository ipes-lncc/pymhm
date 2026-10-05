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

For releases, update the version in `pixi.toml`, `pyproject.toml`,
`src/pymhm/__init__.py`, `CITATION.cff`, and `recipe/recipe.yaml`; update the changelog, regenerate the
lock, and run the checks above. Tests, Lint and Quality, and Docs have dedicated
workflows that run independently on pull requests and main-branch pushes and
provide the same checks as reusable release gates. On a pushed `vVERSION` tag,
`publish-pypi.yml` runs Tests and Quality in parallel, then builds and deploys the
validated documentation, then publishes the checked Python distributions.
The quality gate checks tag/version consistency before building artifacts.
Maintainers must configure the `pypi` environment and PyPI trusted publisher for
`publish-pypi.yml`, set Pages' source to GitHub Actions, and permit release tags
in the `github-pages` environment. See the development guide for the release procedure.

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
