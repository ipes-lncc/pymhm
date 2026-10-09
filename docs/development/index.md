# Development

Contributions should preserve the numerical meaning of the formulations as
well as their public interfaces. Use the checked-in Pixi environments to make
the dependency profile and verification reproducible.

## Start contributing

Read the [contributor workflow](contributing.md) for environment setup, Git
hooks, checks and release preparation, and the
[repository contribution guide](https://github.com/ipes-lncc/pymhm/blob/main/CONTRIBUTING.md)
for contribution conventions.

The [architecture](../architecture.md) describes the owners of mesh operations,
finite-element kernels, local/global equations, execution and post-processing.
Place shared numerical operations in those owners and call them from notebooks
and acquisition helpers. Tutorials teach users how to define their equations;
they should not depend on a hidden application solver.

## Verify a change

Run tests for the affected operators and callers in their locked dependency
profiles. Optional-backend changes also need portable collection and failure
contracts. Native assembly, plotting and accelerator paths require actual
execution when they are affected. Use the
[verification criteria](../verification.md) to distinguish algebraic residuals,
physical conservation and approximation error.

Document the current inputs, mathematical assumptions and numerical conventions.
Publish figures and convergence records with executed-source identities, and
check the rendered mathematics before delivery. The
[documentation checks](contributing.md#quality-gates) and independent coverage
gates are part of the contributor workflow.

## Prepare a release

The [release procedure](contributing.md#distribution) synchronizes
versions, generates the changelog and validates the distributions before tag
publication. Scientific data, notebooks and documentation remain separate from
the runtime wheel and source distribution.

The [roadmap](https://github.com/ipes-lncc/pymhm/blob/main/ROADMAP.md) records
remaining implementation and scientific-validation work.
