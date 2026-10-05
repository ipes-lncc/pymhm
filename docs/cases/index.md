# Numerical evidence guide

The [initial convergence studies](minimal-convergence.md) collect short refinement
series with separate physical field norms, original-equation diagnostics,
quadrature controls and executed source identities. Analytical errors and
increments between numerical solutions have distinct labels. A decreasing
increment alone does not establish the accuracy of a numerical reference.

The selected detailed pages provide additional evidence:

| Case | Verified scope | Scientific limit |
| --- | --- | --- |
| [RAD and Brinkman layers](introduction-layers.md) | Declared local/global UFL forms, resolution studies and independent native hybrid-system checks | Scalar layer resolution and Brinkman polynomial-family rates are distinct from algebraic residuals; historical mesh connectivity is not recovered |
| [Recursive MHM](nested.md) | Ten analytical cases; equivalence to their complete leaf systems, independent assembly and literal basis replay | The recursive construction is not a reproduction of a historical mesh |
| [MHM–MsHHO](mshho.md) | Five refinements and four contrasts; physical fields and original equations in the stated local spaces | Equivalence depends on source and space hypotheses; finite cases do not prove uniform inf-sup stability |
| [Quarter-five spot and obstacle](quarter-five-spot.md) | Six point-well cases, independent mixed assemblies, and four classical obstacle refinements | The obstacle reference remains a numerical approximation with a nonzero refinement increment |
| [MsHHO in 3D](mshho3d.md) | Ten affine tetrahedral/cubic analytical cases; both executed bases, original equations and physical field comparisons | This unit-cube study is original analytical verification |
| [Mixed H(div) in 3D](enriched-hdiv3d.md) | Twenty tetrahedral/prismatic analytical cases; independent Basix assembly, physical pressure/flux/divergence and basis replay | Normal degree and pressure degree are independent; the selected cases do not establish a uniform stability theorem |

Each numerical record identifies the actual operator, geometry, material,
boundary convention, approximation spaces, gauge and integration rules. Darcy
flux, raw gradient and a Robin or pseudo-traction multiplier are distinct fields.
The [literature catalog](../literature.md) identifies the primary sources, and
the [scope matrix](../roadmap.md) distinguishes implemented capabilities from
literature results and remaining comparisons.

The full campaign profile is retained in `docs/publication-full.json`. Its
parameter sweeps, fine-reference studies and supplementary case sources are
separate from the selected current evidence. Missing or unresolved campaign
results are not substituted with short-series values.
