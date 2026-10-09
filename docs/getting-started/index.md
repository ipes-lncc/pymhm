# Getting started

PyMHM separates the mathematical problem from the finite-element implementation
and the execution environment. Begin with one small calculation, then choose a
multiscale method and an execution backend independently.

## 1. Install the capabilities you need

The portable package supplies the mesh, coefficient, Basix, multiscale assembly
and reconstruction interfaces:

```bash
python -m pip install pymhm
```

For UFL forms, install DOLFINx in the same environment before installing PyMHM.
UFL describes a form; DOLFINx supplies its native finite-element assembly. Follow
[Installation](../installation.md) for the tested Conda/Pixi environments and
[Windows](../windows.md) for platform-specific capabilities.

## 2. Learn the calculation workflow

The [feature overview](../tutorials/overview.md) introduces the sequence used
throughout the tutorials:

1. Define a macro mesh and its local meshes.
2. Choose the local approximation and interface spaces.
3. Write local variational equations and identify their physical kernels.
4. Write the global coupling and boundary conditions.
5. Assemble, solve and reconstruct the physical fields.
6. Plot fields, check conservation and measure approximation errors.

Built-in interface spaces handle numbering and orientation. The
[custom interface tutorial](../tutorials/custom-interface.md) explains the
explicit contracts available when you supply another representation.

## 3. Choose a method through its formulation

Use [Theoretical Background](../theory.md) to compare assumptions and spaces,
then select a [method tutorial](../tutorials/index.md). Each tutorial connects
the variational equations to the corresponding code and identifies the norms
used in its convergence study. Start with primal MHM before studying mixed
local spaces or independent cell/face constructions.

## 4. Choose an environment and a physical example

[Guides](../guides/index.md) explain mesh exchange, material markers, parallel
execution and accelerators. [Gallery](../gallery/index.md) lets you explore
the same numerical evidence by physical problem or by dimension.

The [download catalogue](../data.md) provides source notebooks and their
declared inputs. Numerical datasets are separate from the installed Python
package; use the stated checksums and provenance when reproducing a case.
