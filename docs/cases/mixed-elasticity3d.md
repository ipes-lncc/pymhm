# Weak-symmetry mixed elasticity in 3D

`solve_elasticity_mixed_3d` uses a row-wise $H(\mathrm{div})$ Cauchy stress,
discontinuous displacement and discontinuous axial rotation on affine
tetrahedra. The local family is

$$
\Sigma_h=[\mathrm{BDM}_k]^3,\qquad
V_h=[P_{k-1}]^3,\qquad Q_h=[P_{k-1}]^3.
$$

The classical three-dimensional family is established by
[Arnold, Falk and Winther (2007), Eq. (7.1) and Theorems 7.1–7.2](https://arxiv.org/abs/math/0701506v1).
This implementation uses $k\ge2$ so that all six local rigid displacements
are represented exactly. The skeletal MHM coupling and the analytical
campaign below are original three-dimensional constructions. They are not
identified with the two-dimensional enriched-family experiments of
[Devloo et al. (2021)](https://doi.org/10.1051/m2an/2021013).
Classical AFW stability does not by itself establish stability of an arbitrary
MHM trace restriction; independent coupling and local-kernel checks remain
necessary.

## Constitutive law, signs and gauges

The physical convention is

$$
\begin{aligned}
-\mathrm{div}\,\sigma&=f,\\
A\sigma&=\varepsilon(u),\\
A\tau&=\frac{\mathrm{dev}\,\tau}{2\mu}
 +\frac{\mathrm{tr}\,\tau}{3(2\mu+3\lambda)}I.
\end{aligned}
$$

The deviatoric expression avoids subtracting nearly equal spherical terms
as $\lambda\to\infty$. Lamé coefficients can be spatial fields; $\mu$ must
be positive and $\lambda$ nonnegative or infinite. An explicit anisotropic
compliance is a self-adjoint positive-definite $9\times9$ Cartesian tensor
operator, including its extension to skew tensors. A Kelvin $6\times6$
operator alone does not define that extension. Material fields are integrated
on the local tetrahedra with the selected positive quadrature. Discontinuous
materials need an aligned local mesh or a separately justified integration
rule; this solver does not silently fit interfaces or add material cuts.

Rotation is represented by

$$
r=\frac12\left(
\partial_z u_y-\partial_y u_z,
\partial_x u_z-\partial_z u_x,
\partial_y u_x-\partial_x u_y\right).
$$

Its multiplier pairs with
$(\sigma_{yz}-\sigma_{zy},\sigma_{zx}-\sigma_{xz},
\sigma_{xy}-\sigma_{yx})$. Stress symmetry is imposed in all local
$P_{k-1}$ moments, rather than by replacing the computed tensor with its
symmetric part. The physical stress is therefore kept unsymmetrized in
errors, plots and archives.

The skeletal multiplier is $-\sigma n$ in the canonical normal orientation.
The `traction` argument prescribes the outward physical $\sigma n$; remaining
exterior faces carry displacement moments. The local Neumann kernel contains
three translations and three rotations, with their corresponding axial
rotation values. Pure traction uses six integrated displacement moments and
has no independent rotation gauge.

Full displacement data impose the physical identity

$$
\int_\Omega\mathrm{tr}(A\sigma)
=\int_{\partial\Omega}g\cdot n.
$$

At infinite $\lambda$, the right-hand side must vanish. Compatibility is
checked relative to the uncancelled assembled physical moments, independent
of displacement amplitude. The hydrostatic indeterminacy then uses the
prescribed mean of $-\mathrm{tr}\,\sigma/3$. At finite $\lambda$ the
constitutive law determines this mean.

```python
import numpy as np
from pymhm.meshes.mixed import AffineMixedMesh
from examples.formulations.application import weak_stress_elasticity as solve_elasticity_mixed_3d

solution = solve_elasticity_mixed_3d(
    AffineMixedMesh.unit_cube(2),
    stress_degree=2,
    trace_degree=1,
    local_refinement=2,
    lame_lambda=np.inf,
    source=(1.0, 0.0, 0.0),
)
```

Trace subdivisions must align with local faces. The global trace degree may
not exceed the local normal degree. Local factorizations, global solves,
serial/thread/process execution and explicit extended residual refinement use
the shared package infrastructure.

## Analytical deformation without bulk-dependent force

On the unit cube, let

$$
\psi=\prod_{i=1}^3\sin^2(\pi x_i),\qquad
u=\nabla\times(0,0,\psi).
$$

Then $\mathrm{div}\,u=0$ and $u=0$ on the full boundary. With $\mu=1$, the
same stress $\sigma=2\varepsilon(u)$, zero pressure and body force
$f=-\Delta u$ apply at every finite $\lambda$ and at incompressibility.
The force does not grow with the bulk modulus. Independent complex-step
derivatives check both the analytical gradient and $-\mathrm{div}\,\sigma$.

The campaign has five macro resolutions for BDM2/P1/P1 with local refinement
two and skeletal $P_1$, and five for BDM3/P2/P2 with local refinement one and
skeletal $P_2$. Both sequences use $\lambda=\infty$. A separate fixed-space
study spans $\lambda/\mu=0,1,10^2,10^4,10^6,10^8,\infty$. These data measure
approximation and bulk-modulus sensitivity; they do not establish a universal
locking-free theorem for all traces or anisotropic compliance choices.

The exact norms for $\mu=1$ also have closed forms:

$$
\|u\|_{L^2}=\frac{3\pi}{8},\qquad
\|\sigma\|_{L^2}=\pi^2\sqrt{\frac{15}{8}},\qquad
\|r\|_{L^2}=\frac{\pi^2\sqrt{15}}{8}.
$$

Relative errors divide the physical displacement, Frobenius stress and axial
rotation $L^2$ errors by these respective norms. Positive error rules
are recomputed at higher order, independently of the analytical tensor-Gauss
norm calculation. Force balance checks every DG displacement moment;
weak symmetry checks all three DG skew moments. Traction checks use the
canonical integral normal moments on every fine boundary face.

## Measured spatial errors

The table reports relative physical errors in percent at infinite bulk modulus.
The two families have different local refinements and trace degrees, as specified
above; their curves are not a comparison at equal computational cost.

| Macrocells | BDM2 displacement | BDM2 stress | BDM2 rotation | BDM3 displacement | BDM3 stress | BDM3 rotation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 81.5205 | 86.8050 | 138.4961 | 61.5394 | 62.8113 | 57.7163 |
| 48 | 24.6947 | 45.8737 | 71.4885 | 16.9549 | 16.0883 | 17.2572 |
| 162 | 10.2874 | 25.6480 | 40.9170 | 5.4637 | 6.6023 | 7.4718 |
| 384 | 5.0665 | 15.6142 | 24.4499 | 2.4685 | 3.0629 | 3.5630 |
| 750 | 3.0119 | 10.5233 | 16.6716 | 1.3062 | 1.6410 | 1.9391 |

All three errors decrease along both sequences. Displacement alone is an
incomplete accuracy measure: at the final BDM2 resolution its error is about
3%, whereas stress and rotation errors remain about 11% and 17%. The final
BDM3 fields have errors between 1.3% and 2.0% in the stated norms. These are
measured errors against the analytical solution, not an extrapolated rate or
a uniform bound over arbitrary material and trace choices.

![Five spatial resolutions for both AFW families](../figures/mixed-elasticity3d/convergence.png)

The sections at $z=0.37$ evaluate the archived polynomial fields inside each
fine tetrahedron. Common signed scales compare exact and numerical values;
differences have their own zero-centered scales. Tangential stress components,
displacement and rotation may jump. No interface averaging or stress
symmetrization is applied, and actual macro intersections remain visible.


![BDM3 displacement, stress and rotation against exact fields](../figures/mixed-elasticity3d/bdm3-fields.png)

At fixed 48-macro BDM2/P1/P1 resolution, the bulk-modulus control gives the
following relative errors in percent. The analytical fields and load are
unchanged throughout this table.

| $\lambda/\mu$ | Displacement | Stress | Axial rotation |
| ---: | ---: | ---: | ---: |
| 0 | 32.3691 | 42.8959 | 95.8714 |
| 1 | 28.1970 | 43.4532 | 82.2530 |
| $10^2$ | 24.7586 | 45.7760 | 71.6850 |
| $10^4$ | 24.6953 | 45.8726 | 71.4905 |
| $10^6$ | 24.6947 | 45.8736 | 71.4885 |
| $10^8$ | 24.6947 | 45.8737 | 71.4885 |
| $\infty$ | 24.6947 | 45.8737 | 71.4885 |

These errors approach finite limits as the bulk modulus increases. The
coarse spatial approximation remains substantial, especially in rotation;
bounded bulk sensitivity does not imply an accurate coarse field. The
incompressible solve uses its physical hydrostatic mean and is not a finite
penalty approximation.

![Fixed-space bulk-modulus sensitivity](../figures/mixed-elasticity3d/incompressibility.png)

## Independent operator and solution checks

Native DOLFINx/UFL tests independently assemble complete BDM2 and BDM3
constitutive, divergence, rotation and force matrices on a sheared tetrahedron,
including variable Lamé coefficients. Two additional native classical mixed
systems on the cube verify nonaffine displacement, full stress and rotation
when the complete fine normal space is retained. They compare physical fields
and original-system residuals rather than matching basis coefficients.

Portable tests cover affine and quadratic deformation, nonzero body force,
Dirichlet/mixed/traction boundaries, all six rigid modes, anisotropic Cartesian
compliance, variable materials, incompressibility, incompatible volume flux,
physical moment balances, the complementary-energy/body-work identity and
process-spawn parity.

```bash
pixi run --locked -e notebooks python -m examples.solve_mixed_elasticity3d --workers 4
pixi run --locked -e notebooks python -m examples.plot_mixed_elasticity3d
```

Archives retain local stress/displacement/rotation coefficients, fine geometry,
trace moments and the executed H(div) basis matrix with its SHA256 digest.
Every figure reuses that matrix for orientation and evaluation. Section
polynomials are evaluated separately in each fine cell; no interface averaging
or stress symmetrization is applied. Notebook 68 checks the recorded fields
and displays the campaign.

## References

- Douglas N. Arnold, Richard S. Falk, and Ragnar Winther (2007). *Mixed finite element methods for linear elasticity with weakly imposed symmetry*. Mathematics of Computation 76, 1699–1723. [DOI: 10.1090/S0025-5718-07-01998-9](https://doi.org/10.1090/S0025-5718-07-01998-9). [Preprint: arXiv:math/0701506v1](https://arxiv.org/abs/math/0701506v1).

- Philippe R. B. Devloo, Agnaldo M. Farias, Sônia M. Gomes, Weslley Pereira, Antonio J. B. dos Santos, and Frédéric Valentin (2021). *New H(div)-conforming multiscale hybrid-mixed methods for the elasticity problem on polygonal meshes*, ESAIM: M2AN 55, 1005–1037. [DOI: 10.1051/m2an/2021013](https://doi.org/10.1051/m2an/2021013).
