# Selective RAD kernels and matrix conditioning

The generalized MHM construction of
[Araya et al. (2024), sections 4–5](https://doi.org/10.1016/j.cma.2024.117089)
allows a different constant space on each macrocell. With zero reaction,
divergence-free advection tangent to the local boundary leaves a constant
nullspace. A nonzero advective field does not by itself remove that nullspace.
Positive reaction or a nonzero Robin normal component can instead give an
invertible local problem.

`solve_transport(..., coarse_space="kernel")` and `solve_rad_3d` select the
corresponding mixed/primal construction. The default `"constants"` retains all
constants in a complementary formulation, useful near singular parameter limits.
Eliminating its non-null coarse amplitudes gives the selective system. Tests
compare reconstructed fields, traces and the mixed/primal partition in two and
three dimensions. Physical diffusive-flux faces in the selective variant require
tangent advection; general diffusive data use retained constants.

## Five published coefficient cases

The unit square contains sixteen square macrocells. The source is one and
Dirichlet data are zero. Diffusion is $\epsilon I$. On a cell mapped to
$(s,t)\in[0,1]^2$, the circulation is

$$
\widehat\beta=((s^2-s)(1-2t),(1-2s)(t-t^2)).
$$

Alternating signs make the field continuous across the square partition.
The advection is $\alpha=\chi\beta+(\delta(y),0)$, where $\delta=\omega$
below $y=1/2$, interpolates linearly to $\omega_0$ at $y=3/4$, and equals
$\omega_0$ above that height. Reaction is $\nu_1$ for $x\leq1/2$ and
$\nu_0$ otherwise.

| Case | $\chi$ | $\omega_0$ | $\nu_0$ | $\nu_1$ | Constant null modes at positive $\omega$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 75 | 0 | 0 | 0 | 4 |
| 2 | 1 | 0 | 0 | 1 | 2 |
| 3 | 1 | 0.5 | 0 | 0 | 0 |
| 4 | 1 | 0 | 1 | 10 | 0 |
| 5 | 25 | 1 | 1 | 0 | 0 |

Local continuous P2 elements use four center-fan triangles in each square;
the skeleton has constant moments. Distinct macrofaces have distinct adjacent
triangles, as in assumption (A1). The local matrix is taken on
$[1/4,1/2]^2$, the red square in Figure 7. The Euclidean spectral condition
number uses the physical nodal/constant bases, before solver equilibration.
The selective global matrix is also obtained by exact elimination of the
non-null constant amplitudes from the complementary system. Its condition
number is not that of the better-conditioned auxiliary system used to solve.

![Diffusion-parameter sweep beside published markers](../figures/rad-conditioning/epsilon.png)

![Advection-parameter sweep beside published markers](../figures/rad-conditioning/omega.png)

Open markers are extracted from the vector paths of Figures 8–9; lines are
PyMHM values. The CSV includes one PDF point of ordinate uncertainty on the
logarithmic axes. The expected trends include quadratic growth of global
conditioning with large advection, inverse-quadratic growth with small
advection for cases 1, 2, 3 and 5, and bounded small-advection conditioning
for case 4. These are parameter-dependent observations, not uniform stability
of every floating-point system.

At $\epsilon=\omega=1$, the five global condition numbers are approximately
7350, 4672, 6153, 408 and 1867. The digitized values are respectively
7308, 4572, 6151, 408 and 1875. The local ordinates have larger differences:
for example, case 3 gives 7123 versus approximately 39082 in the publication.
Local nodal conditioning depends on the actual fine connectivity and basis;
the recorded center-fan realization does not establish reproduction of those
local ordinates. Both comparisons are shown, without rescaling either curve.

The local and global solvers explicitly retain extended-precision correction
digits on platforms with a wider NumPy `longdouble`; factors remain double
precision. The most ill-conditioned parameter combinations can fail the unchanged physical
residual criterion. Such records carry `status="not_certified"` and are not
reported as verified field solutions. A plotted local matrix condition number
alone does not certify its solve. The raw numerical record includes each
parameter, local/global dimension, status and residual where available.

## Independent systems for the five coefficient fields

[DOLFINx/UFL 0.9.0](https://github.com/FEniCS/dolfinx/tree/6443e3b27d29aec04ce83d8424b57c339e35f865)
supplies the triangular Lagrange P2 volume matrices in a separately written
complete broken-P2/P0 MHM comparison. This is the same MHM discretization
as the condensed PyMHM system. The same sixteen square macrocells use
center-fan triangulations; the squares are macroelement groups, not DOLFINx
finite-element cells. Local nodal indices are separate across macrofaces.
Symbolic UFL expressions
define the circulation, drift and reaction of each row in the table above.
The comparison application separately assembles face moments using Basix basis evaluations
and P0 coupling, including weak homogeneous Dirichlet conditions. SciPy
SuperLU solves this full saddle system without using PyMHM element matrices,
local inverses or selective condensation.

All five coefficient cases are compared at $\omega=1$ with both
$\epsilon=1$ and $\epsilon=0.01$. Across these ten systems, the largest
relative difference in scalar, gradient or physical-flux L2 norms is
$2.96\times10^{-13}$; the largest normalized residual of the original
native equations is $6.42\times10^{-15}$. Native integration degrees 16
and 20 provide separate norm checks. These are comparisons of the variable
coefficient problems themselves, including cases with and without local
constant modes.

The [native numerical records](../figures/rad-conditioning/native-discrete-verification.json)
retain field archives, approximation spaces, build/source provenance and
each physical comparison. They establish agreement for the stated
discretization. They do not recover the historical local connectivity or
assert agreement throughout the extreme parameter ranges where the physical
residual criterion rejects a floating-point solve.

## Separate local and skeletal resolution

![Separate local and skeletal studies](../figures/rad-conditioning/mesh.png)

With fixed P0 macrofaces, case 2 approaches a global condition number of about
4573 as its local mesh is refined. Refining face segments instead grows the
global condition number approximately in proportion to their count, consistent
with the $H^{-1}$ behavior reported in Figure 10. These are distinct refinements:
local resolution changes the accuracy of the lifts; skeletal resolution changes
the global approximation space.

The original parameter plots are included for direct inspection:

![Published Figure 8](../figures/rad-conditioning/published-figure8.png)

*[Araya et al. (2024)](https://doi.org/10.1016/j.cma.2024.117089), Figure 8. Local and global conditioning versus diffusion.*

![Published Figure 9](../figures/rad-conditioning/published-figure9.png)

*[Araya et al. (2024)](https://doi.org/10.1016/j.cma.2024.117089), Figure 9. Local and global conditioning versus advection.*

Run `pixi run -e notebooks verify-rad-conditioning`, followed by
`pixi run -e notebooks python examples/plot_rad_conditioning.py`.
Results and digitized markers are under `examples/results`; the parameter sweeps
are research jobs, separate from the compact automated nullspace tests.

## References

- Rodolfo Araya, Fabrice Jaillet, Diego Paredes, and Frédéric Valentin (2024). *Generalizing the multiscale hybrid-mixed method for reactive-advective-diffusive equations*, Computer Methods in Applied Mechanics and Engineering 428, 117089. [DOI: 10.1016/j.cma.2024.117089](https://doi.org/10.1016/j.cma.2024.117089).
