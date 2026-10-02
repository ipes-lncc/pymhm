# PGMHM with 729 square-annulus inclusions

This case evaluates the heterogeneous experiment of
[Fernando et al. (2023), §6.2, Figures 8–10](https://doi.org/10.1007/s40314-023-02304-y).
MHM and PGMHM use the same local meshes, polynomial spaces, material and
boundary data. A separately assembled DOLFINx/UFL conforming CG2 sequence
provides a physical-field reference, with its own refinement increments.
The experiment specifies the geometry and local resolution below; the
historical coefficient file, local triangulations and stabilization parameter
are unavailable, so identical historical output is not asserted.

## Physical geometry and data

On the unit square, the equation and boundary condition are

$$
-\nabla\cdot(K\nabla p)=1,\qquad p\big|_{\partial\Omega}=0,
\qquad q=-K\nabla p.
$$

There are 27 periods in each direction. The inclusions are **square annuli**:
their inner square cores have the background coefficient. They are neither
filled squares nor circles. For each period with center \(c\),

$$
\begin{aligned}
L&=1/27, & r_{\mathrm{in}}&=3L/32, & r_{\mathrm{out}}&=3L/16,\\
K(x)&=10^5
&&\text{if }r_{\mathrm{in}}<\lVert x-c\rVert_\infty<r_{\mathrm{out}},\\
K(x)&=1&&\text{otherwise}.
\end{aligned}
$$

The union of the high-permeability annuli occupies exactly **10.546875%** of
the domain. The source integral is one. The coefficient is scalar and acts
isotropically; no rescaling or clipping is applied.

The radius-to-period ratios scale the three-by-three geometry of
[Barrenechea, Jaillet, Paredes and Valentin (2020), §5.2, Figure 7](https://doi.org/10.1007/s00211-020-01103-5),
which is reference [14] of the PGMHM paper. That source states \(L=1/3\),
\(r_{\mathrm{in}}=1/32\) and \(r_{\mathrm{out}}=1/16\).
The 27-by-27 image in PGMHM Figure 9 is consistent with these scaled widths.
PGMHM Figure 8 instead prints \(L=1/4\) beside a three-by-three drawing;
that caption does not uniquely determine a periodic 27-by-27 input.
The ratios above are therefore a declared, source-supported geometry,
not a fitted choice based on the resulting pressure.

## Matched MHM and PGMHM spaces

The macro mesh contains 32 nonconvex L-shaped polygons and 148 original
macrofaces. Each macroface carries two discontinuous P0 segments, giving
296 multipliers and 32 retained mean-pressure modes: **328 global unknowns**.
The multiplier remains on the original polygon boundary; interior edges of
its triangulation are not new macrofaces.

Local pressure spaces are continuous P2 on independently represented
triangular partitions. Every material interface is a fine-cell edge.
Successive local-resolution factors one, two and four subdivide the same
coordinate intervals; the macro partition and its 328 unknowns remain fixed.
Both methods use exactly these spaces. The selected PGMHM parameter is
\(\alpha=0.1\), with the published weight

$$
\tau_E=\frac{\alpha K_{\min}}{2H_E},\qquad K_{\min}=1.
$$

Here \(H_E\) is the original macroface length, independent of its two
segments. The base and residual-enriched PGMHM pressures are retained
separately. Their raw fluxes are computed as \(-K\nabla p\); they are not
H(div) reconstructions. Local fitting resolves integration and permits
independent normal derivatives across material interfaces. It does not
add macroface moments or establish that the fixed trace space is accurate.

The paper reports 328 unknowns and a refined reference with 328,192 unknowns,
but it does not provide an executable local mesh or the heterogeneous-case
value of \(\alpha\). Matching the global count alone does not identify those
missing choices. This comparison uses a **classical conforming CG2 reference**;
it does not relabel that method as the paper's fine P0-trace MHM reference.

## Independent reference and arithmetic contract

DOLFINx/UFL assembles the native CG2 diffusion operator and constant source
on meshes aligned with every square-annulus interface. Exterior pressure is
imposed strongly. PETSc/MUMPS supplies the factorization. The original free
equations retain the relative residual criterion \(10^{-10}\).

For these high-contrast matrices, correction coefficients are explicitly
accumulated in two binary64 components. Compensated products and row sums
check the unchanged CSR equations against both components. The archive
stores them separately, and replay evaluates the two P2 polynomials before
combining their physical fields. The arithmetic is tested against independent
80-digit decimal sums and a native heterogeneous affine patch. It is not an
arbitrary-precision factorization. Physical energy and norm reductions use
NumPy's extended real type; no 106-bit claim is made for those reductions.

For the uniform material-aligned reference sequence, the final original-matrix
residual is \(4.07\times10^{-14}\). Independent 80-digit decimal products on
1,000 deterministic rows agree with the compensated residual to an absolute
\(5.98\times10^{-28}\). The assembled-matrix energy and the energy integrated
from physical gradients are distinct numerical checks: at the finest uniform
level their difference is \(1.92\times10^{-9}\). For the two consecutive
nested refinements, the physical Galerkin identity has absolute defects
\(1.42\times10^{-9}\) and \(5.04\times10^{-9}\). These are respectively
\(1.61\times10^{-6}\) and \(1.82\times10^{-5}\) of the squared increment.
The coefficient precision contract does not remove rounding already present
in the native binary64 matrix assembly.

A second classical sequence grades every material coordinate interval in the
ratios $1:2:4:8:4:2:1$, concentrating cells on both sides of every material
interface. Isotropic bisection then gives 2,331,729 and 9,320,809 CG2 unknowns.
These two triangulations are nested, but the graded and piecewise-uniform
sequences are not nested with one another. The
[independent canonical polynomial-moment check](../figures/pgmhm-inclusions/classical-energy-verification.json)
keeps physical energy and native-matrix energy separate. At the finer graded
level, their values are $0.0258472457988$ and $0.0258470920731$, respectively:
an absolute difference of $1.54\times10^{-7}$, or $5.95\times10^{-6}$ relative
to the physical energy. The original CSR residual is $1.04\times10^{-11}$.
The physical nested-energy identity has a defect of $2.75\times10^{-7}$,
or 1.98% of the much smaller squared refinement increment. These measured
differences limit an interpretation based on an exact discrete energy identity;
they do not alter the reported physical-field norms or their denominators.

The [native constant-mode energy verification](../figures/pgmhm-inclusions/classical-csr-energy-verification.json)
separates this assembly contribution from the solved-system defect. For the
original matrix $A$, complete coefficient vector $p$, and row and column sums
$r_i$ and $c_i$, respectively, the exact algebraic identity is

$$
p^{\mathsf T}Ap
=-\frac12\sum_{i,j}A_{ij}(p_i-p_j)^2
 +\frac12\sum_i p_i^2(r_i+c_i).
$$

It does not assume exact symmetry, zero row sums, or a nonnegative sign for
each entry contribution. At 9,320,809 unknowns, the first term agrees with the
independently integrated physical energy within $3.13\times10^{-17}$.
The second term is $-1.53725\times10^{-7}$ and accounts for the measured
physical-to-CSR energy difference, with a decomposition remainder of
$7.67\times10^{-13}$. By comparison, the coefficient-weighted algebraic
residual $p^{\mathsf T}(Ap-b)$ is only $-2.24\times10^{-14}$.
The maximum row-sum defect relative to absolute row action is
$8.18\times10^{-16}$. Its aggregate effect is measurable for this
high-contrast system. The original assembled matrix and both archived
coefficient components are preserved in this check.

A third graded reference contains 37,271,025 unknowns. Its consecutive
physical increments are $0.01828\%$ in pressure, $1.74632\%$ in raw flux,
and $1.30742\%$ in weighted flux. Its integrated energy minus source work
is $3.99\times10^{-7}$, or $1.54\times10^{-5}$ relative to its physical
energy. Between the second and third levels, the physical energy increase
is $4.90937\times10^{-6}$ and the integrated squared difference is
$4.41906\times10^{-6}$. Their gap, $4.90\times10^{-7}$, is 11.10% of
the latter increment. The explicit CSR decomposition above applies to the
second graded level; it is not silently extrapolated to the third.

Pressure, raw flux and weighted flux differences are integrated on exact
intersections of the MHM fine cells and the classical triangles:

$$
\begin{aligned}
E_p^2&=\int_\Omega (p_h-p_{\mathrm{ref}})^2,\\
E_q^2&=\int_\Omega \lvert q_h-q_{\mathrm{ref}}\rvert^2,\\
E_A^2&=\int_\Omega K^{-1}\lvert q_h-q_{\mathrm{ref}}\rvert^2.
\end{aligned}
$$

Each relative quantity uses the corresponding norm of the stated reference.
Duffy orders three and four integrate the same piecewise-polynomial expressions
independently. Macro pressure jumps remain one-sided, without averaging.
Classical refinement increments remain numerical differences, not exact-error
bounds.

The [independent geometry and polynomial-replay check](../figures/pgmhm-inclusions/graded-geometry-verification.json)
verifies all 729 annuli, their analytic area and the material of every cell in
an interface-graded mesh. The grading preserves each material interface and
has bitwise nested coordinate axes. Prolonging a quadratic finite-element
field to the next isotropically refined triangulation changes its pressure
and flux norms by less than $1.3\times10^{-15}$ relatively. This check establishes
the geometry and field-evaluation contracts; it does not measure the accuracy
of a PDE reference solution.

## Resolved fields and remaining reference uncertainty

The three local resolutions contain 30,752, 123,008 and 492,032 triangles,
respectively, with the same 328 global unknowns. Their original hybrid residuals
are below $2.6\times10^{-16}$. The piecewise-uniform CG2 sequence contains
47,961, 190,969 and 762,129 unknowns; its last flux increment is 13.4037%.
The separate interface-graded sequence contains 2,331,729 and 9,320,809 unknowns.
Its last refinement changes pressure by **0.05513%**, raw flux by **2.93166%**
and weighted flux by **2.32107%**.

All entries below use the same **9,320,809-unknown graded CG2 field** as their
reference and norm denominator. The MHM and PGMHM fields share the geometry
and local spaces. Residual enrichment is archived separately; it does not
change the local polynomial degree.

| Local factor | MHM pressure $L^2$ | PGMHM pressure $L^2$ | MHM raw flux $L^2$ | PGMHM raw flux $L^2$ | PGMHM weighted flux |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.96405% | 2.96013% | 28.02284% | 28.02262% | 21.87699% |
| 2 | 0.48700% | 0.48422% | 18.59574% | 18.59535% | 15.04859% |
| 4 | 0.58627% | 0.58957% | 13.79221% | 13.79162% | 12.00097% |

Raw flux differences decrease under local refinement at fixed macro trace.
The pressure difference is not monotone against this numerical reference.
The finest PGMHM/CG2 flux difference is **13.7916%**; the reference's own
**2.9317%** increment is about 21% of that measured distance. The refinement
check helps distinguish the local-resolution trend, but is not a bound on the
remaining reference error or a claim of a converged reference. The enriched
PGMHM raw flux has a 13.8144% difference at the finest local level; the residual
correction does not guarantee improvement in every physical-field norm.
The two norm quadratures agree within $1.5\times10^{-9}$ relatively in flux
and $4.3\times10^{-14}$ in weighted flux across all three local resolutions.

The finest piecewise-uniform and graded references have **0.64186% pressure**,
**11.28329% flux** and **7.97284% weighted-flux** differences, normalized by the
graded field. These are integrated over the exact intersections of their
nonnested triangulations, with separate polynomial-exact quadratures. The
[reference comparison](../figures/pgmhm-inclusions/uniform-graded-comparison.json)
therefore measures reference-resolution sensitivity directly; it is not a
change in the MHM or PGMHM fields. Percentages using different reference
norms are not added or interpreted as an orthogonal decomposition.

![Square-annulus material, actual macro partition and independent pressure profiles](../figures/pgmhm-inclusions/material-profile.png)

The diagonal profile retains separate polynomial pieces on every crossed fine
cell and independent values on either side of a macroface. Vertical markers
identify macroface crossings. Neither method's pressure is smoothed.

![Classical and PGMHM pressure fields with their signed difference](../figures/pgmhm-inclusions/pressure-fields.png)

All three spatial panels retain the actual 32-polygon macro partition. The two
pressure panels share one numerical range; the signed difference has its own
symmetric range.

![Independent classical refinement and matched MHM/PGMHM physical differences](../figures/pgmhm-inclusions/physical-norms.png)

These measurements verify the stated implementation and provide a resolution
study. They do not establish equality with the historical local meshes or
reference of Figures 9–10.

## Reproducible acquisitions

The following original application commands keep the geometry, coefficient,
stabilization and field-replay coordinates explicit:

```bash
pixi run -e fem python -m examples.solve_pgmhm_inclusions_reference --factors 1 2 4
pixi run -e fem python -m examples.solve_pgmhm_inclusions_reference --graded --factors 1 2
pixi run -e test python -m examples.solve_pgmhm_inclusions --factors 1 2 4 --workers 8
pixi run -e test python -m examples.compare_pgmhm_inclusions \
  examples/results/pgmhm-inclusions/mhm-pgmhm-factor4-s2.npz \
  examples/results/pgmhm-inclusions/classical-cg2-graded2.npz --workers 8 \
  --output examples/results/pgmhm-inclusions/mhm-pgmhm-factor4-s2-graded2-comparison.json
```

The original analytical tests and native UFL integration tests are distributed;
external reference-code comparison runners are not required by these commands.
