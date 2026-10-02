# Higher-order local fields and variable operators

These experiments exercise the native polynomial assemblers through complete MHM
solves. Their references are analytical fields differentiated independently of the
linear solver. The six refinement points use `s = 1, 2, 3, 4, 6, 8` segments per
macroface, with the macrotriangulation fixed. Local refinement grows with `s`;
these are joint local/skeleton refinements, not macro-mesh refinement curves.

All map panels show the actual macrotriangles, including analytical and error
panels. Local polynomial fields are evaluated on separate display subdivisions;
macroface values are never merged. Color maps interpolate those samples for
rendering. Error norms use independent triangle quadrature, not image samples.

## Darcy pressure P1 through P4

On the unit square, choose unit permeability and

$$
p_\star=\cos(\pi x)\cos(\pi y),\qquad
\boldsymbol q_\star=-\nabla p_\star,\qquad
f=2\pi^2p_\star.
$$

The entire boundary has the corresponding nonzero Dirichlet pressure. Eight
macrotriangles are fixed. Degree-`k` local pressure uses degree-`k-1` traces and
`2*s` local subdivisions. Assembly uses Duffy order 8 and errors order 10.

![Six-point primal Darcy pressure and physical flux errors for P1 through P4](../figures/high-order/darcy-convergence.svg)

| Segments `s` | P1 pressure L2 | P2 pressure L2 | P3 pressure L2 | P4 pressure L2 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.6544e-02 | 8.9994e-03 | 5.6403e-04 | 3.6404e-05 |
| 2 | 1.7397e-02 | 7.5339e-04 | 2.3581e-05 | 9.1255e-07 |
| 3 | 7.5475e-03 | 2.0207e-04 | 4.2069e-06 | 1.1379e-07 |
| 4 | 4.2007e-03 | 8.1142e-05 | 1.2758e-06 | 2.6321e-08 |
| 6 | 1.8601e-03 | 2.2839e-05 | 2.4351e-07 | 3.3805e-09 |
| 8 | 1.0450e-03 | 9.3791e-06 | 7.6005e-08 | 7.9234e-10 |

At `s=8`, raw-flux L2 errors are **0.10926**, **2.1486e-3**,
**2.5877e-5** and **2.9000e-7**, respectively. The maximum absolute macro
balance over all 24 calculations is below **2e-15**.

The flux norm evaluates the raw physical field `-grad(p_h)`. Primal pressure
reconstruction does not make this gradient H(div)-conforming. Macro conservation
is checked independently using multiplier fluxes and source integrals. A small
macro balance therefore does not imply zero fine-cell divergence error.

![Exact pressure, P4 pressure, and pointwise error on the actual macro mesh](../figures/high-order/darcy-p4.svg)

The representative map uses P4 pressure, degree-three traces, `s=2` and four local
subdivisions. Its error panel has its own scale.

![Exact and P4 physical flux magnitudes with pointwise vector error](../figures/high-order/darcy-p4-flux.svg)

The flux map evaluates the P4 gradient inside each fine triangle. The third
panel is the vector error norm, not a difference of flux magnitudes.

## Variable tensor Brinkman resistance

Set viscosity one and use

$$
\Gamma(x,y)=(1+x+2y)
\begin{pmatrix}20&3\\3&1\end{pmatrix},\qquad
-\Delta\boldsymbol u+\Gamma\boldsymbol u+\nabla p=\boldsymbol f,
\qquad \nabla\cdot\boldsymbol u=0.
$$

The constant matrix has positive determinant 11 and positive diagonal entries;
the scalar factor is positive throughout the unit square. Thus the resistance is
symmetric positive definite, spatially varying, and couples velocity components.
The velocity is the curl of
`psi = -128*x**2*(x-1)**2*y**2*(y-1)**2`, with zero boundary velocity, and
`p = 150*(x-.5)*(y-.5)`. Its mean pressure is zero. The independently differentiated
source is the Stokes source plus `Gamma @ u`.

Taylor–Hood P2/P1 and USFEM P2/P2 use the same 32 macrotriangles, degree-one
vector traces, `2*s` local subdivisions and assembly/error Duffy order 8.
USFEM includes the full residual `-Delta(u) + Gamma @ u + grad(p)` in both
stabilization and source terms; the tensor is not replaced by its diagonal or
by a scalar when assembling these terms.

![Six-point errors for Taylor–Hood and USFEM with variable tensor resistance](../figures/high-order/flow-convergence.svg)

| `s` | TH velocity L2 | TH pressure L2 | USFEM velocity L2 | USFEM pressure L2 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.5173e-02 | 3.9888e-01 | 2.4611e-02 | 5.1235e-01 |
| 2 | 2.0259e-03 | 6.6104e-02 | 1.9925e-03 | 1.3524e-01 |
| 3 | 5.1502e-04 | 2.5551e-02 | 5.0823e-04 | 7.1276e-02 |
| 4 | 1.9437e-04 | 1.3259e-02 | 1.9261e-04 | 4.4826e-02 |
| 6 | 4.9596e-05 | 5.3648e-03 | 4.9391e-05 | 2.2666e-02 |
| 8 | 1.8938e-05 | 2.8614e-03 | 1.8904e-05 | 1.3676e-02 |

Both velocity errors decrease through the six points. At `s=8`, the two
velocity errors are close, while the Taylor–Hood pressure error is smaller.
Local pressure degree alone does not determine the error of the hybrid solve.

![Exact and USFEM velocity magnitudes with the pointwise vector error](../figures/high-order/tensor-brinkman.svg)

The field map uses `s=2`. Its third panel displays
`norm(u_h-u_exact)`, not the difference between velocity magnitudes. The flow
operator uses the gradient–gradient viscous form. Exterior traction data use
`(grad(u)-p*I) @ n`; the skeletal multiplier uses the opposite
pseudo-stress convention `(-grad(u)+p*I) @ n`. These differ from the
symmetric-gradient Cauchy traction used in elasticity.

![Exact pressure and USFEM pressure with pointwise error](../figures/high-order/tensor-brinkman-pressure.svg)

## Conservative variable reaction–advection–diffusion

For the conservative equation

$$
-\nabla\cdot(K\nabla u)+\nabla\cdot(\boldsymbol\beta u)+c u=f,
$$

prescribe zero Dirichlet data and choose

$$
\begin{aligned}
u_\star&=\sin(\pi x)\sin(\pi y),\\
K&=0.05(1+x+y)\begin{pmatrix}2&0.3\\0.3&1\end{pmatrix},\\
\boldsymbol\beta&=(1+x,\tfrac12+y),\qquad c=\tfrac12+xy.
\end{aligned}
$$

The explicit coefficient derivatives are
`diffusion_divergence=(0.115,0.065)` and `velocity_divergence=2`. Differentiation
gives

$$
\begin{aligned}
f={}&0.05(1+x+y)\pi^2
\left[3u_\star-0.6\cos(\pi x)\cos(\pi y)\right]\\
&+(\boldsymbol\beta-(0.115,0.065))\cdot\nabla u_\star
 +(2.5+xy)u_\star.
\end{aligned}
$$

The conservative strong operator includes `div(beta)*u`. The SUPG residual
also includes `-div(K) @ grad(u)` and the polynomial Hessian contraction
`-K : Hessian(u)`.
Complex-step differentiation of the full physical flux checks these identities
in the analytical-data tests.

Eight macrotriangles, local P2 fields, degree-one traces and `2*s` local
subdivisions are used for both Galerkin and SUPG. Assembly uses order 8 and
error integration order 10. These smooth solutions test consistency and
convergence; they do not establish an advantage for SUPG in a resolved regime.

![Six-point variable conservative RAD errors](../figures/high-order/rad-convergence.svg)

| `s` | Galerkin P2 L2 | SUPG P2 L2 |
| ---: | ---: | ---: |
| 1 | 1.1787e-02 | 2.3200e-02 |
| 2 | 1.3119e-03 | 4.7529e-03 |
| 3 | 3.6037e-04 | 1.3675e-03 |
| 4 | 1.4217e-04 | 5.0825e-04 |
| 6 | 3.7780e-05 | 1.1520e-04 |
| 8 | 1.4687e-05 | 3.8883e-05 |

Galerkin has the smaller measured error at every point of this smooth test.
The SUPG errors also decrease; stabilization is not presented as a general
accuracy improvement.

![Exact and SUPG P2 fields with their pointwise error](../figures/high-order/variable-rad.svg)

The multiplier is the Robin flux
`(-K @ grad(u) + beta*u/2) @ n`, associated with the skew advection form.
The full conservative physical flux is `-K @ grad(u) + beta*u`.

## Two outflow layers and measurable oscillations

For `epsilon = 0.02` and `0.005`, solve

$$
-\epsilon\Delta u+\partial_xu=1,\qquad
u_\star(x,y)=x-
\frac{e^{(x-1)/\epsilon}-e^{-1/\epsilon}}
     {1-e^{-1/\epsilon}}.
$$

The exact field is independent of `y`. It vanishes at `x=0,1`, while the
horizontal boundaries carry its nonzero values. Those values are prescribed
on the **whole boundary**, using the same analytical function. Homogeneous data
on the horizontal edges would describe a different problem.

The exact maximum is evaluated at

$$
x_\star=1+\epsilon\log\!\left(\epsilon(1-e^{-1/\epsilon})\right).
$$

This gives `0 <= u_exact <= u_exact_max < 1`. Each Galerkin/SUPG calculation uses
32 macrotriangles, local P1 fields, degree-one traces and `4*s` subdivisions.
Assembly uses Duffy order 20 and error integration order 24. The comparatively
high assembly order resolves the exponential boundary moments on coarse faces;
a low-order approximation of these data can dominate a thin-layer comparison.

![L2 error, negative undershoot, and excess above the exact maximum](../figures/high-order/layer-errors.svg)

Extrema are computed from every local P1 coefficient. Since a linear function
attains its extrema at triangle vertices, these are the extrema of the broken
reconstructed field, including weak boundary deviations. Undershoot is
`max(0,-min(u_h))`; overshoot is `max(0,max(u_h)-u_exact_max)`. Neither quantity
is inferred from a plotted profile.

| ε | Method | `s` | L2 error | Undershoot | Overshoot |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.02 | galerkin | 1 | 5.5446e-02 | 2.9583e-01 | 4.3921e-01 |
| 0.02 | galerkin | 8 | 1.0516e-03 | 4.9477e-03 | 1.4665e-03 |
| 0.02 | supg | 1 | 1.0014e-01 | 1.5073e-01 | 8.1084e-02 |
| 0.02 | supg | 8 | 5.0364e-03 | 4.4505e-03 | 3.5045e-04 |
| 0.005 | galerkin | 1 | 1.6299e-01 | 1.1418e+00 | 1.3320e+00 |
| 0.005 | galerkin | 8 | 7.9746e-03 | 9.7058e-02 | 4.5024e-02 |
| 0.005 | supg | 1 | 1.4002e-01 | 6.2332e-01 | 2.1296e-01 |
| 0.005 | supg | 8 | 2.5150e-02 | 5.6461e-02 | 2.3436e-02 |

SUPG reduces both extrema in these endpoint comparisons, but substantial
negative values remain for the thin layer even at `s=8`. For `epsilon=0.02`,
Galerkin has the smaller L2 error at both endpoints. For `epsilon=0.005`, SUPG
improves the first coarse-point L2 error, while Galerkin is more accurate at
`s=8`. Neither method is positivity-preserving in this experiment.

![One-sided Galerkin and SUPG profiles with every actual macroface crossing](../figures/high-order/layer-profiles.svg)

![Thin-layer SUPG solution at the first refinement point](../figures/high-order/layer-0.005-supg-1.svg)

SUPG adds streamline stabilization. It does not impose a discrete maximum
principle, positivity constraint or crosswind shock-capturing term. The measured
extrema and profiles must therefore be considered together with the L2 errors;
a smaller integrated error does not by itself establish absence of oscillations.

## Higher-order transient heat

Take unit diffusivity, zero boundary data and

$$
u_\star(x,y,t)=(1+t)\sin(\pi x)\sin(\pi y),\qquad
f=(1+2\pi^2(1+t))\sin(\pi x)\sin(\pi y).
$$

Eight macrotriangles are fixed. The local fields are P2 or P3 with traces of
degree `k-1`, `2*s` local subdivisions, and four backward-Euler steps to `t=0.1`.
Assembly uses order 8 and error integration order 10. Initial values are nodal
interpolants of the exact initial field.

![Six-point spatial refinement of the time-dependent P2 and P3 heat fields](../figures/high-order/heat-convergence.svg)

| `s` | P2 heat L2 | P3 heat L2 |
| ---: | ---: | ---: |
| 1 | 8.6593e-03 | 6.2492e-04 |
| 2 | 8.1222e-04 | 2.5923e-05 |
| 3 | 2.2005e-04 | 4.6349e-06 |
| 4 | 8.8685e-05 | 1.4069e-06 |
| 6 | 2.5035e-05 | 2.6872e-07 |
| 8 | 1.0294e-05 | 8.3896e-08 |

![Exact heat field and local P3 reconstruction at t=0.1](../figures/high-order/heat-p3.svg)

The exact solution is linear in time, so its backward-Euler difference quotient
has no temporal truncation error. These curves assess the spatial approximation
and initial interpolation in a transient solve. They do not measure a higher
order of the time integrator. The [decaying-mode example](https://github.com/volpatto/pymhm/blob/main/docs/cases/scalars.md) examines
actual temporal discretization error separately.

## Reproduce and inspect

The analytical data are defined in `examples/native_extension_data.py`;
`examples/plot_native_extensions.py` records discretizations, norms, field hashes
and numerical-source provenance. The archive contains 84 parameter configurations, with four
time steps within each of the 12 heat configurations. Results are in
`examples/results/native-extensions/report.json`, with separate compressed
broken-field arrays. Notebook `19_native_extensions.ipynb` demonstrates the
operators and inspects the archived six-point series.

```bash
pixi run -e notebooks python examples/plot_native_extensions.py --workers 4
pixi run -e notebooks python examples/plot_native_extensions.py --reuse-results
```

Use `--sections darcy flow rad layer heat` with any subset to recompute selected
families. Every completed case is saved immediately. The displayed figures are
analytical verifications of these native paths; they are not external solver
comparisons or reproductions of a published error table.
