# Darcy-coupled transient transport

`solve_transient_transport` advances

$$
\rho\,\partial_t u-\nabla\!\cdot(D\nabla u)
       +\nabla\!\cdot(\beta u)+cu=f
$$

with backward Euler, continuous local Pk fields, and the conservative hybrid
Robin trace. Capacity is positive, and the spatial coefficients are stationary;
sources and boundary data may depend on time. Both Galerkin and consistent SUPG
testing include the time residual. In particular, the previous solution enters
through the same weighted mass form as the current time derivative.

`OfflineHybridSystem` prepares local and global factors once per distinct time
increment. Changes in sources and boundary values only rebuild loads. The
reported `operator_builds` counts these prepared time-step operators; it is not
a performance speedup measurement.

For long trajectories, `output_steps` selects the one-based step indices retained
in the result. Every time step is computed; `integration_times` records the entire
grid, while `times`, `solutions` and balance records refer to the selected outputs.
The optional `on_step(step, time, solution, balance)` callback receives read-only
numerical arrays after each computed step, including steps omitted from the
returned history. This permits checkpointing without keeping every state in RAM.
Callback exceptions close the prepared local and global factorizations.

## Conservative Darcy velocity and dispersion

Section 5.4 of [Harder, Paredes and Valentin (2015)](https://doi.org/10.1137/130938499)
couples a stationary Darcy field to transport with hydrodynamic dispersion.
`solve_darcy_transport` accepts locally H(div)-conforming RT and BDM Darcy
solutions, including RT moment reconstructions. It evaluates their actual
physical vector polynomials without changing the Darcy approximation:

$$
\beta=q_h,\qquad
 D(q_h)=(\alpha_m+\alpha_t|q_h|)I+
     (\alpha_l-\alpha_t)\frac{q_h\otimes q_h}{|q_h|}.
$$

At zero flux the tensor equals \(\alpha_m I\). The coefficients satisfy
\(\alpha_m>0\) and \(\alpha_l\ge\alpha_t\ge0\). Within a two-dimensional RT0
triangle, \(\nabla q_h=bI\), where \(b=\nabla\!\cdot q_h/2\), so its broken
tensor divergence away from zero flux is

$$
\nabla\!\cdot D=(2\alpha_l-\alpha_t)b\frac{q_h}{|q_h|}.
$$

This derivative is included in SUPG. At isolated zero-flux points the formula is
understood almost everywhere; no derivative across a material interface is
inserted into a fine-element residual.

The transport fine triangulation matches the Darcy fine triangulation.
The coupling passes the actual Darcy partitions to every transport operator;
matching cell counts alone is insufficient. An explicit mesh override must
preserve their points and connectivity. Standalone heat and transient transport
also accept validated `local_meshes`, which take precedence over uniform
refinement and define both the old-state interpolation and the new-state forms.
`MacroCoefficient` and `RT0DarcyVelocity` preserve the macro side when evaluating
broken coefficients. A barycentric membership check validates each fine-cell
location; nearest-centroid selection alone is insufficient. Raw primal Darcy
gradients are not silently substituted for a conservative velocity.

With `velocity_representation="primal"`, the coupling instead retains the raw
Darcy volume vector \(q_h=-K\nabla p_h\) and the Darcy skeletal multiplier
\(\lambda_D\) as numerical normal flux. It integrates the Galerkin advection
form directly:

$$
a_{\mathrm{adv},K}(u,w)
=-\int_K q_hu\cdot\nabla w
+\frac12\int_{\partial K}\lambda_Duw.
$$

This pair preserves separate volume and skeleton approximations. It is not
asserted to be H(div)-conforming: replacing its interface distributions by a
broken divergence would give a different operator. The material's broken tensor
derivative must be supplied explicitly as `permeability_gradient`; use `0.0`
for a piecewise constant field. Darcy and transport face partitions and degrees
may differ. Strong-residual stabilization is supported for the H(div) path and
is rejected for the primal volume/numerical-normal pair.

Exterior `diffusive_flux` data prescribe \((-D\nabla u)\cdot n\). This permits
strong inflow concentration together with homogeneous diffusive outflow and
wall conditions, as in the published transport configuration. The package also
supports explicitly prescribed Robin data through `neumann`.

## A five-level temporal verification

The recorded test is an original manufactured problem using the article's
coupling and dispersion law. It does not reproduce the article's unspecified
random permeability realization. On \([0,3]\times[0,1]\), set

$$
K(y)=\begin{cases}1,&y<1/2,\\10,&y\ge1/2,\end{cases}\qquad
 p=3-x,\qquad q=(K(y),0),
$$

and

$$
\begin{aligned}
u&=e^{-t}b(x),\qquad b(x)=1+\frac{x(3-x)}9,\\
f&=e^{-t}\left[-b(x)+\frac{2(\alpha_m+\alpha_lK)}9
                          +\frac{K(3-2x)}9\right].
\end{aligned}
$$

The concentration is prescribed strongly on \(x=0,3\); the remaining exterior
sides have zero diffusive flux. Initial data are \(b\). The layer interface
aligns with the fine and macro meshes, and normal advective and diffusive fluxes
across it are zero. The parameters are \(\alpha_m=10^{-6}\),
\(\alpha_l=10^{-2}\), \(\alpha_t=10^{-3}\), unit capacity, and zero reaction.

Darcy uses RT0/P0 on sixteen fine triangles per macrotriangle. Transport uses
local P3 and degree-two macroface multipliers, with SUPG. Five uniform time grids
contain 4, 8, 16, 32 and 64 steps up to \(t=1\). The exact concentration and
independent error quadratures distinguish temporal truncation from algebraic
residuals.

| Steps | Time increment | L² error at t=1 | Prepared operators |
|---:|---:|---:|---:|
| 4 | 1/4 | 0.08350081 | 1 |
| 8 | 1/8 | 0.04469750 | 1 |
| 16 | 1/16 | 0.02328481 | 1 |
| 32 | 1/32 | 0.01190523 | 1 |
| 64 | 1/64 | 0.00602688 | 1 |

The last doubling gives an observed temporal order of 0.982. The computed Darcy
flux has L² error \(4.79\times10^{-15}\) for this exactly representable field.
The largest discrete balance residual is \(2.19\times10^{-14}\). Error
quadratures of orders 10 and 12 differ by less than \(7\times10^{-17}\).

The [current-source replay](../figures/transport/local-mesh-verification.json)
identifies the campaign and field archive by SHA-256 and reproduces all five
temporal records and all seven saved field arrays bitwise. Its source hashes
describe this replay, separately from the original acquisition. The same
record documents the custom-partition regressions and independent UFL checks
of the backward-Euler Galerkin and SUPG operators.

![Darcy-coupled concentration at t=1](../figures/transport/darcy-transient.png)
![Temporal and spatial refinement records](../figures/transport/refinement.png)

`total_mass()` integrates \(\rho u_h\) using the capacity-weighted mass moments.
`balance_residuals` sums the original local scalar equations, including reaction
forces from strongly imposed data. It is a discrete weak balance diagnostic;
it is not a claim of fine-element conservation, an independent physical error,
or a positivity certificate.

The source is available as `examples/transport_campaign.py`. Run
`pixi run -e notebooks python -m examples.transport_campaign --collect` to recompute
all levels, or omit `--collect` to render the archived results. The lightweight
test suite separately checks a linear-in-time variable-coefficient patch,
mass preservation with closed boundaries, the heat limit, exact RT0 evaluation,
and the dispersion derivative.
