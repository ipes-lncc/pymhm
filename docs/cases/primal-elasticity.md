# General-tensor primal elasticity

The primal solver accepts a spatially varying, symmetric positive-definite
constitutive operator on symmetric strains. It solves

$$
-\operatorname{div}\sigma=f,\qquad \sigma=\mathcal A\varepsilon(u),
\qquad \varepsilon(u)=\tfrac12(\nabla u+\nabla u^T).
$$

This is the displacement formulation analysed in
[Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046).
Its coefficient-dependent estimates do not establish uniform robustness as
Poisson ratio approaches one half. Use the displacement-pressure or
weak-symmetry mixed formulations for the incompressible limit.

## Constitutive and space contracts

`solve_elasticity(..., formulation="primal")` and
`solve_primal_elasticity` accept either a constant/callable Kelvin matrix
`constitutive` of shape `(3,3)` or a Cartesian fourth-order tensor of shape
`(2,2,2,2)`. Callbacks receive physical coordinates and prepend a point axis.
The Kelvin coordinates are
\((\varepsilon_{xx},\varepsilon_{yy},\sqrt2\varepsilon_{xy})\), so ordinary
Euclidean products equal Frobenius tensor products. Cartesian tensors must
have both minor symmetries and major symmetry. Positivity is checked at
assembly points; these samples do not certify a global bound for a callback.

With no general tensor, finite Lamé data define plane-strain isotropic
elasticity. A `CartesianCellField` supplied as the constitutive operator uses
quadrature on exact fine-triangle/material-pixel intersections. This resolves
integration across a discontinuity without enriching displacement across it.
Raw stress remains symmetric but is generally neither H(div)-conforming nor
fine-cell equilibrated. The skeletal equations enforce macro force and moment
balance.

Continuous Pk displacement is available locally. On a single local triangle,
Lemma 6.6 requires local degree \(k\geq\ell+1\) for even trace degree \(\ell\)
and \(k\geq\ell+2\) for odd \(\ell\). The separate global assumption
\(\Lambda_{rm}\subset\Lambda_h\) requires traces of rigid motions: P1 on every
face is sufficient in two dimensions. Scalar trace injectivity alone does not
establish this global property. The original P0 default remains a primal
baseline outside that hypothesis of [Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046); the campaigns below explicitly use P1.

For even `degree=k`, unsplit P(k−1) traces and one local triangle,
`minimal_enrichment=True` adds exactly one P(k+1) polynomial per displacement
component. Its nonzero functional detects the scalar trace mode invisible to
Pk. This implements the enriched-space construction underlying Lemma 6.8;
elimination is algebraically equivalent to its additional local operators.
The enrichment is **not** a zero-boundary bubble. Returned displacement values
use P(k+1) nodal coordinates, while `hybrid.fields` retain only
`2*(dim(Pk)+1)` local coordinates. For P2 this gives 14 coordinates, compared
with 20 for full P3.

The example helpers below are repository sources, supplied separately from the
installed library. Open the corresponding notebook to acquire its verified
companion, as described in the [notebook guide](../tutorials.md#execute-downloaded-notebooks).

```python
import numpy as np
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from examples.formulations.application import elasticity as solve_elasticity

mesh = TriangleMesh.unit_square(4)
trace = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
solution = solve_elasticity(
    mesh, formulation="primal", degree=2, minimal_enrichment=True,
    local_refinement=1, skeleton=trace,
    constitutive=np.array([[5., 1., .4], [1., 4., .3], [.4, .3, 2.]]),
    source=(1., 0.),
)
```

Traction data mean physical outward Cauchy traction. Pure traction specifies
three integrated displacement moments against the two translations and the
rotation about the domain centroid. No pointwise pin or independent rotation
gauge is introduced.

## Face indicator and the one-level hypothesis

`estimate_primal_elasticity_error` implements equations (5.4)–(5.6):

$$
\begin{aligned}
r_F&=\begin{cases}
\tfrac12[u_h],&\text{inside},\\
g-u_h,&\text{on the boundary},
\end{cases}\\
\eta_F^2&=\frac{c_{min}^2}{H_F}\|r_F\|_F^2,
\qquad\eta^2=\sum_K\sum_{F\subset\partial K}\eta_F^2.
\end{aligned}
$$

Interior faces count twice; the denominator is the original macroface length,
including when the trace is segmented. The caller supplies a positive global
bound satisfying \(\mathcal A\epsilon:\epsilon\geq c_{min}^2|\epsilon|^2\)
and explicitly declares full Dirichlet data. P1-containing skeletal spaces
are required. The bound is checked at face integration points, with the same
sampling limitation as the constitutive validation.

The reliability theorem assumes **exact local solves**. With finite local
spaces, the returned quantity is only a face indicator. It does not measure
all local approximation error and has no declared unit reliability constant.
A zero-boundary local bubble can have zero indicator and nonzero displacement
error; a CI test verifies this distinction. Local refinement must be assessed
separately. The paper by [Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046) supplies analysis rather than a published numerical table for
this general-tensor case.

## Independent assembly and five-level evidence

Six DOLFINx/UFL tests independently assemble the full stiffness and load for
P1–P4 and minimally enriched P2/P4, including spatially varying anisotropy and
Kelvin shear coupling. Additional tests verify physical traction signs, rigid
moments, exact affine/quadratic patches, material-interface integration and the
one-mode trace construction on a translated small cell.

`examples/solve_primal_elasticity.py` uses the original analytical field
\(u=(\sin(\pi x)\sin(\pi y),\sin(2\pi x)\sin(\pi y))\). The tensor is the
matrix in the example above, multiplied either by one or by \(1+x+2y\).
The analytical load includes derivatives of that multiplier. Each configuration
uses \(n=1,2,4,8,16\), P1 macro traces and homogeneous displacement data.
P1 uses local refinement four; minimally enriched P2 and full P3 use one
local triangle. Thus the local dimensions are 30, 14 and 20 respectively.
Assembly/error quadrature orders are 10/12.

![Variable-tensor convergence and face indicator](../figures/primal-elasticity/variable-convergence.png)

![Constant-tensor convergence and face indicator](../figures/primal-elasticity/constant-convergence.png)

![Variable-tensor enriched P2 displacement and raw stress](../figures/primal-elasticity/variable-p2-fields.png)

![Variable-tensor P1 displacement and raw stress](../figures/primal-elasticity/variable-p1-fields.png)

```bash
pixi run --locked -e test-core python -m examples.solve_primal_elasticity --degree 2
pixi run --locked -e test-core python -m examples.solve_primal_elasticity --degree 2 --constant
pixi run --locked -e notebooks python -m examples.plot_elasticity_extensions
```

## References

- Christopher Harder, Alexandre L. Madureira, and Frédéric Valentin (2016). *A hybrid-mixed method for elasticity*, ESAIM: M2AN 50, 311–336. [DOI: 10.1051/m2an/2015046](https://doi.org/10.1051/m2an/2015046).
