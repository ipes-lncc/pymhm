# Verification studies

These reports document analytical verification, reference refinement and
independent assembly checks for specific discretizations. They complement the
[method lessons](../tutorials/methods/index.md) and the
[application Gallery](../gallery/index.md).

The [refinement catalogue](minimal-convergence.md) retains the separate
physical-field errors or successive-solution increments for its recorded
configurations. Exact errors, numerical-reference differences and successive
increments are identified on each report.

The selected detailed pages cover the following checks:

| Case | Verified scope | Scientific limit |
| --- | --- | --- |
| [RAD and Brinkman layers](introduction-layers.md) | Declared local/global UFL forms, resolution studies and independent native hybrid-system checks | Scalar layer resolution and Brinkman polynomial-family rates are distinct from algebraic residuals; historical mesh connectivity is not recovered |
| [Recursive MHM](nested.md) | Ten analytical cases; equivalence to their complete leaf systems, independent assembly and literal basis replay | The recursive construction is not a reproduction of a historical mesh |
| [MsHHO](mshho.md) | Five refinements and four contrasts; physical fields and original equations in the stated local spaces | Equivalence with MHM depends on source and space hypotheses; finite cases do not prove uniform inf-sup stability |
| [Quarter-five spot and obstacle](quarter-five-spot.md) | Six point-well cases, independent mixed assemblies, and four classical obstacle refinements | The obstacle reference remains a numerical approximation with a nonzero refinement increment |
| [MsHHO in 3D](mshho3d.md) | Ten affine tetrahedral/cubic analytical cases; both executed bases, original equations and physical field comparisons | This unit-cube study is original analytical verification |
| [Mixed H(div) in 3D](enriched-hdiv3d.md) | Twenty tetrahedral/prismatic analytical cases; independent Basix assembly, physical pressure/flux/divergence and basis replay | Normal degree and pressure degree are independent; the selected cases do not establish a uniform stability theorem |

Each numerical record identifies the actual operator, geometry, material,
boundary convention, approximation spaces, gauge and integration rules. Darcy
flux, raw gradient and a Robin or pseudo-traction multiplier are distinct fields.
The [literature catalog](../literature.md) identifies the primary sources, and
the [scope matrix](https://github.com/ipes-lncc/pymhm/blob/main/ROADMAP.md#scientific-scope-and-acceptance-criteria) distinguishes implemented capabilities from
literature results and remaining comparisons.

Detailed application and reference reports are linked from their Gallery,
theory and verification pages and open within this site. Each report identifies
its available figures, numerical records and any inputs needed for field replay.
A recorded error norm does not reconstruct a missing coefficient vector.

## Reproduce a case

Download the case notebook and open it with the installed `pymhm` and the
declared optional dependencies. Its first cell acquires the verified companion
sources and selected inputs into a writable directory; generated fields remain
there. Examples, notebooks, documentation and datasets are separate from the
library distribution. The [notebook guide](../tutorials/notebooks.md#execute-downloaded-notebooks)
explains this workflow, and [data downloads](../data.md) list available inputs.

Commands beginning with `pixi run` on individual case pages are developer
reproduction commands for a repository checkout and its checked-in lockfile.
