# Independent MSL GaLS elasticity comparison

MSL_MHM + MSL_CG (GaLS) and pyMHM were run with the same local spaces,
macrotriangles, linear traction traces, material, forcing and boundary data.
Seven archived comparisons cover five P1/P1 mesh levels and additional P2/P2
and P3/P3 local spaces. The largest absolute field differences across these
cases are $2.59\times10^{-13}$ in displacement, $8.60\times10^{-12}$ in
Herrmann pressure, $1.68\times10^{-11}$ in displacement gradient and
$3.10\times10^{-11}$ in full Cauchy stress, measured in $L^2$.

These are direct comparisons with an independently assembled reference
operator. They validate the stated discrete configurations; they do not
identify the data that generated every table in the
[locking-free elasticity paper](https://arxiv.org/abs/2403.16890v1).
The separate [elasticity gallery](elasticity.md) studies material limits and
skeletal refinement against exact fields.

## Matched problem and spaces

The domain is the unit square, with $\mu=1$, $\lambda=4999$ and
$\nu=0.4999$. The displacement vanishes on the entire boundary. The consistent
trigonometric [analytical family](elasticity.md#exact-family-and-printed-benchmark-conventions)
uses amplitude one, as in the MSL manufactured input:

$$
u=u_0+(1-2\nu)\sin(\pi x)\sin(\pi y)(1,1)^T,
\qquad p=-2\pi\nu\sin(\pi(x+y)),
$$

with $p=-\lambda\operatorname{div}u$ and
$\sigma=2\mu\varepsilon(u)-pI$. The source is derived from
$-\operatorname{div}\sigma=f$; the exact pressure mean is zero.

The macro mesh is a uniform crisscross grid: each Cartesian square is split
into four triangles through its center. With $n$ squares along each axis,
there are $4n^2$ macrotriangles and their diameter is $H=1/n$.
Every macroface carries one discontinuous linear vector traction segment.
The skeleton represents $-\sigma n$ and local problems retain all three
rigid displacement modes.

| Local displacement/pressure | Macro resolutions $n$ | Fine subdivisions per macro edge | Stabilization $\alpha$ |
| --- | --- | --- | --- |
| P1/P1 | 1, 2, 4, 8, 16 | 4 | 0.1 |
| P2/P2 | 4 | 2 | 0.001 |
| P3/P3 | 4 | 1 | 0.001 |

These pairs satisfy the stated linear-trace refinement condition and the
computed inverse-inequality bound. Both implementations include all GaLS
residual and forcing terms. In particular, the P2/P3 cases include
second derivatives of displacement, which vanish only in the P1 case.
The coefficients above are explicit experiment inputs, not estimates fitted
to the resulting errors.

## Convergence and agreement

The P1/P1 series refines the macro mesh and its associated fine meshes.
The following errors are from the archived pyMHM fields; MSL values coincide
at the displayed precision. The gradient norm is the full broken $H^1$
seminorm and the stress norm includes every tensor component.

| $n$ | Macrotriangles | $\lVert u-u_h\rVert_0$ | $\lVert \nabla(u-u_h)\rVert_0$ | $\lVert p-p_h\rVert_0$ | $\lVert \sigma-\sigma_h\rVert_0$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.24152365 | 4.18209317 | 4.36532990 | 8.38044980 |
| 2 | 16 | 0.08116602 | 2.12584063 | 1.78284264 | 3.70293418 |
| 4 | 64 | 0.01981371 | 1.05754389 | 0.91628264 | 1.81555192 |
| 8 | 256 | 0.00491662 | 0.51565665 | 0.43585575 | 0.86668932 |
| 16 | 1024 | 0.00122784 | 0.25546743 | 0.21368265 | 0.42634049 |

![Five matched MSL and pyMHM GaLS mesh levels](../figures/elasticity-reference/convergence.svg)

The finer levels approach second-order displacement convergence and
first-order gradient, pressure and stress convergence for this fixed
P1/P1 local choice. Agreement between implementations and accuracy relative
to the exact solution are separate quantities.

![Absolute differences between the reconstructed fields](../figures/elasticity-reference/field-differences.svg)

The difference curves measure the numerical scale of agreement; their slopes
are not discretization convergence rates. Both fields retain their computed
pressure means and use the same physical quadrature for comparison, with
requested order 12. Assembly uses
requested order 10; the native integration rule and pyMHM's quadrature need
not have the same number of points.

At $n=4$, the additional local-space checks give:

| Local pair | $\lVert u-u_h\rVert_0$ | $\lVert \nabla(u-u_h)\rVert_0$ | $\lVert p-p_h\rVert_0$ | $\lVert \sigma-\sigma_h\rVert_0$ |
| --- | ---: | ---: | ---: | ---: |
| P2/P2, refinement 2 | 0.01334115 | 0.48088736 | 0.46755687 | 0.92203326 |
| P3/P3, refinement 1 | 0.01385936 | 0.49858893 | 0.40800194 | 0.84610484 |

These two rows change the local mesh as well as the polynomial degree and
use the same linear macro trace. They are not a fixed-mesh polynomial-rate
study, and a monotone decrease in every norm is not implied.

## Exact, reference and reconstructed fields

The maps show the P3/P3 case with $n=4$. The top row uses a common scale for
the analytical, MSL and pyMHM fields. The first two error panels also share
a scale; the implementation difference has its own explicitly labelled
scale. Every panel shows the actual macro boundaries. Polynomial fields are
evaluated separately on each fine triangle, preserving one-sided values
rather than averaging stresses across interfaces.

![Displacement component with exact, MSL, pyMHM and error maps](../figures/elasticity-reference/u1.svg)

![Herrmann pressure with exact, MSL, pyMHM and error maps](../figures/elasticity-reference/pressure.svg)

![Cauchy stress component with exact, MSL, pyMHM and error maps](../figures/elasticity-reference/stress11.svg)

The GaLS stress reconstructed from displacement and pressure is symmetric,
but these plots and comparisons do not assert that it is globally H(div)
conforming. The [mixed stress formulation](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mixed-elasticity.md) enforces a
different set of continuity and equilibrium conditions.

## Provenance and retained data

The reference consists of MSL_MHM at
`4cb8cf81518284313b680b13fd586ee619f08b99`, MSL_CG at
`afb76d14c1baf50f0b9e69f7bcac675749ef4458` and MSL_CORE at
`7f15f455717173d29080d411a7e732c72c1e87f8`.
The comparison uses the native GaLS element assembly and numerical libraries
with a mixed-field MHM adapter. The adapter applies rigid-mode source projection
to displacement rows and reconstructs pressure with the same native skeletal
coefficients as displacement. The results record the adapter digests and
reference revisions, identifying the complete computational configuration.

`examples/results/elasticity-reference.json` contains the parameters, norms,
reference revisions and archive checksums. Seven NPZ archives retain the
actual macro geometry, fine-element connectivity, nodal coordinates, and
both implementations' displacement and pressure coefficients.

The public plotter reads those records and evaluates the archived fields;
it does not rerun the comparison:

```bash
pixi run -e notebooks python examples/plot_elasticity_reference.py
```

The source paper's tables fix 32 macrotriangles and refine skeletal and
local meshes together, while the main series here changes the macro mesh.
The stabilization coefficient used to generate the paper's tables is not
specified, and the printed exact-data coefficients require the consistency
choices documented in the literature catalog. Consequently the present
agreement establishes a direct code comparison, not reproduction of those
six tables.
