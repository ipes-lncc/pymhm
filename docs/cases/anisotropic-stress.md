# Anisotropic stress on triangles, rectangles and polygons

The mixed stress formulation accepts a full anisotropic compliance, including
spatially varying coefficients. This analytical study checks that capability on
triangular, rectangular and nonconvex polygonal macro meshes. It extends the
weak-symmetry formulation of
[the 2021 MHM elasticity paper](https://doi.org/10.1051/m2an/2021013);
the material and manufactured solution below define a reproducible verification
problem rather than a reproduction of a figure from that paper.

## Material, equations and boundary data

On the unit square, prescribe displacement on the whole boundary and use

$$
u(x,y)=\begin{pmatrix}x^2y^2\\x^3y\end{pmatrix},\qquad
-\operatorname{div}\sigma=f,\qquad
\mathcal A\sigma=\varepsilon(u).
$$

In the orthonormal Kelvin coordinates
\((\varepsilon_{xx},\varepsilon_{yy},\sqrt2\varepsilon_{xy})\), the stiffness is

$$
C=\begin{pmatrix}6&2&0.7\\2&4&-0.4\\0.7&-0.4&3\end{pmatrix}.
$$

The symmetric compliance is \(C^{-1}\). Its explicit extension on the skew
subspace is multiplication by \(1/2\); it preserves symmetric tensors and is
positive definite on the full Cartesian tensor space. The API receives the
corresponding \(4\times4\) compliance in the order \((xx,xy,yx,yy)\).
The source is obtained by analytically differentiating the constitutive stress.
The independent exact rotation is
\((\partial_yu_x-\partial_xu_y)/2=-x^2y/2\).

Weak symmetry is imposed through rotation moments. Reported stress errors use
the full Frobenius norm, including the numerical skew part. For the prescribed
displacement boundary, the physical hydrostatic condition is

$$
\int_\Omega\operatorname{tr}(\mathcal A\sigma)
=\int_{\partial\Omega}u\cdot n.
$$

This condition retains the anisotropic compliance; an isotropic bulk-modulus
formula would define a different problem.

## Spaces and convergence

The five resolutions are \(n=1,2,4,8,16\), with \(H=1/n\).
Each triangular macrocell uses BDM2/P1/P1 for stress rows, displacement and
rotation. Rectangles use RT1/Q1/P1. The polygonal mesh tiles every square by one
L-shaped macrocell and its complementary small square; BDM2/P1/P1 is assembled
on the local triangulation while the original polygon edges remain the skeleton.
The number of macro unknowns and the local mesh differ between families, so this
is not an equal-cost comparison.

Interior macro tractions have degree one. Exterior tractions retain the complete
normal space of their local family. The assembly uses quadrature order eight;
independent orders nine and ten agree in all reported norms to about
\(10^{-15}\). Automated checks also cover anisotropic pure-traction patches,
physical rigid-motion moments and independent DOLFINx/UFL constitutive operators.

![Five-level errors in displacement, full stress, rotation and divergence](../figures/core-extensions/elasticity-convergence.png)

The final absolute errors and rates between the last two resolutions are:

| Macro family | Displacement error / rate | Stress error / rate | Rotation error / rate | Divergence error / rate |
|---|---:|---:|---:|---:|
| Triangles | 5.3752e-4 / 1.997 | 3.4994e-3 / 1.935 | 8.0155e-4 / 1.589 | 6.8513e-3 / 2.000 |
| Rectangles | 3.4440e-4 / 1.999 | 2.9815e-3 / 2.003 | 2.0875e-4 / 2.014 | 5.3079e-3 / 2.000 |
| Nonconvex polygons | 1.0157e-4 / 1.986 | 4.8736e-4 / 1.962 | 5.2268e-5 / 1.962 | 2.1011e-3 / 2.000 |

Displacement and divergence exhibit the expected second-order behavior for these
spaces. The triangular rotation rate is lower on the measured sequence; it is
reported independently rather than inferred from displacement convergence.
The largest integrated macro force/moment residual is below \(1.52\times10^{-13}\).
Those equilibrium residuals are algebraic checks, distinct from the field errors.

## Physical components

Exact and numerical panels use identical color scales; the signed differences
have their own labeled scales. Every panel shows the original macro boundaries.
The polynomial fields are evaluated on a separate display triangulation inside
each fine cell. Coincident vertices retain their independent one-sided values;
neither displacement nor stress is averaged across an element boundary. Each
fine edge is divided into six display intervals. The convergence norms integrate
the complete fields using independent quadrature, not the display samples.

### Displacement approximation and the local projection

Let \(P_1u\) be the independent cellwise \(L^2\) projection of the exact
displacement onto the local displacement space. Orthogonality gives

$$
\lVert u_h-u\rVert_{L^2}^2
=\lVert u_h-P_1u\rVert_{L^2}^2
+\lVert P_1u-u\rVert_{L^2}^2.
$$

For the finest polygonal mesh, the projection error is \(1.0156771\times10^{-4}\),
the computed displacement differs from that projection by
\(3.9880042\times10^{-7}\), and the total error is
\(1.0156849\times10^{-4}\). Thus the projection accounts for 99.9985% of the
squared displacement error. Repeated triangle shapes produce a repeated
approximation-error pattern even for this smooth solution. The measured
projection error decreases with order approximately two, while the difference
between the solution and its projection decreases with order approximately
three on the four tested resolutions \(n=2,4,8,16\). These measured properties
describe this case; they are not a general superconvergence assertion.

An independent [FEniCS/DOLFINx 0.9.0](https://github.com/FEniCS/dolfinx/tree/6443e3b27d29aec04ce83d8424b57c339e35f865)
and Basix 0.9 assembly solves the full UFL saddle system on
the same fine meshes for \(n=1,2,4\), using PETSc/MUMPS. It imposes the same
interior P1 traction restriction through quadratic normal-moment constraints,
retains unrestricted exterior P2 traction and uses the nonhomogeneous analytical
displacement. DOLFINx assembles the conforming fine-space operator; the comparison
application supplies the macroface constraints defining the same MHM space.
Maximum relative differences from PyMHM are
\(2.10\times10^{-14}\) for stress, \(1.35\times10^{-14}\) for displacement and
\(2.13\times10^{-13}\) for rotation. The largest independent saddle residual is
\(9.86\times10^{-15}\). The numerical record is
`examples/results/core-extensions/polygon-ufl-verification.json`.
The associated archives retain the native system, coefficient vectors, cell and
DOF ordering, PyMHM fields and the executed BDM moment basis for field replay.

![Triangular displacement components](../figures/core-extensions/triangle-bdm2-displacement.png)
![Triangular full stress components](../figures/core-extensions/triangle-bdm2-stress.png)
![Rectangular displacement components](../figures/core-extensions/rectangle-rt1-displacement.png)
![Rectangular full stress components](../figures/core-extensions/rectangle-rt1-stress.png)
![Nonconvex polygon displacement components](../figures/core-extensions/polygon-bdm2-displacement.png)
![Nonconvex polygon full stress components](../figures/core-extensions/polygon-bdm2-stress.png)

## Profiles through both coordinate directions

The cuts \(x=0.43\) and \(y=0.37\) avoid lying on a macroface. Each curve segment
evaluates its own fine-cell polynomial, including its one-sided endpoint values.
Separate segments preserve displacement and tangential-stress jumps. The thin
vertical dotted lines locate the actual macroface intersections; lower panels
show the signed error on its physical scale.

![Triangular displacement profiles and signed errors](../figures/core-extensions/triangle-bdm2-displacement-profiles.png)
![Rectangular displacement profiles and signed errors](../figures/core-extensions/rectangle-rt1-displacement-profiles.png)
![Polygonal displacement profiles and signed errors](../figures/core-extensions/polygon-bdm2-displacement-profiles.png)
![Triangular full stress profiles](../figures/core-extensions/triangle-bdm2-stress-profiles.png)
![Rectangular full stress profiles](../figures/core-extensions/rectangle-rt1-stress-profiles.png)
![Polygonal full stress profiles](../figures/core-extensions/polygon-bdm2-stress-profiles.png)

## Reproduce

```bash
pixi run -e notebooks python -m examples.solve_core_extensions elasticity
pixi run -e notebooks python -m examples.sample_core_elasticity
pixi run -e notebooks python -m examples.sample_core_elasticity --projection-study
pixi run -e notebooks python -m examples.plot_core_extensions elasticity
```

The acquisition stores physical errors, quadrature comparisons, source hashes
and field arrays in `examples/results/core-extensions/elasticity.json` and the
associated NPZ archives. `examples/core_extension_data.py` contains the exact
constitutive data and the independently differentiated source.
The independently reevaluated polynomial display fields and projection norms
are recorded in `elasticity-field-sampling.json` in the same directory.
The four-level projection decomposition is saved in `elasticity-projection.json`.
