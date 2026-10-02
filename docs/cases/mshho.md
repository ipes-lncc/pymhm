# MHM and multiscale HHO

`solve_mshho` constructs multiscale HHO cell and face unknowns from constrained
local energy minimization. It implements the projected-source formulation (4.6)
and reconstructed-source formulation (5.1) of
[Chaumont-Frelet, Ern, Lemaire and Valentin (2022)](https://doi.org/10.1051/m2an/2021082).
The article establishes an equivalence theorem; it does not provide a numerical
benchmark table. The comparisons here verify that theorem against the independent
MHM construction in PyMHM.

## Spaces and source convention

The macrocell unknowns are moments against $P_m(K)$, while each macroface has
independent polynomial moments. The reconstructed field minimizes
$\int_K A\nabla v\cdot\nabla v$ subject to those cell and face moments.
Its implementation uses a conforming local $P_k$ discretization of that
minimization. Cell moments are eliminated before assembling the global face
operator. This operator is symmetric positive definite after Dirichlet elimination.

For `source_variant="projected"`, the load uses the cellwise $L^2$ projection
of the source onto $P_m(K)$. For `source_variant="reconstructed"`, the original
load acts on the reconstructed test function. Polynomial sources in $P_m(K)$
give identical fields. A general source requires the corresponding projection
when applying the article's MHM equivalence theorem. The face-only setting
`cell_degree=-1` uses the reconstructed-source variant, as in Remark 5.4;
it is not the same method as MHM with a source lifting.

```python
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_mshho

mesh = TriangleMesh.unit_square(4)
faces = tuple(FaceSpace.uniform(1) for _ in mesh.faces)
solution = solve_mshho(
    mesh, skeleton=SkeletonSpace(mesh, faces), cell_degree=1,
    degree=3, local_refinement=2, source=4.0,
)
```

Polygonal macrocells use `solve_mshho_polygons` with the same moment spaces and
local energy minimization. Nonconvex L-shaped cells are included in the field
equivalence tests.

## Five-level analytical convergence

The unit-square problem uses $A=I$, zero Dirichlet data,
$u=\sin(\pi x)\sin(\pi y)$ and $f=2\pi^2u$. Both configurations use local
$P_3$ elements and two subdivisions of every macrotriangle edge. Macro spacing
ranges from $1$ to $1/16$. Error quadrature is independent of assembly.

![Five-level pressure and physical flux convergence](../figures/mshho/convergence.png)

For $m=k_F=0$, the expected leading orders are two for pressure and one for
the raw physical flux. For $m=k_F=1$, they are three and two. The final measured
orders are approximately **2.00/1.00** and **3.08/2.01**, respectively. Local
discretization remains part of the measured error.

![Analytical, reconstructed and error fields with the macro mesh](../figures/mshho/fields.png)

The panels use separate broken local fields, without smoothing macro interfaces.
The error panel has its own symmetric scale. The plotted raw flux is not asserted
to be globally $H(\mathrm{div})$ conforming.

## Equivalence and numerical precision

The polynomial-source comparison uses $A=\mathrm{diag}(\kappa,1)$, $f=4$,
nonzero affine boundary data, local $P_3$ and degree-one face/cell moments.
The records include field differences and residuals for each $\kappa$.
Agreement is verified through $\kappa=10^6$. The relative maximum field
differences for $\kappa=1,10^3,10^4,10^6$ are respectively
$1.38\times10^{-14}$, $1.29\times10^{-12}$, $2.78\times10^{-11}$ and
$1.65\times10^{-8}$. The MHM local factors remain in double precision, while
iterative corrections accumulate in extended precision; the physical residual
criterion is unchanged. Both complete global residuals are below
$1.5\times10^{-16}$ in this campaign. This precision option requires a platform
whose `longdouble` has more mantissa bits than `float64`; an unsupported platform
reports an availability error. These measured differences delimit the numerical
equivalence check and do not imply a contrast-independent floating-point error.

Run `pixi run -e notebooks verify-mshho` to acquire and render the study.
The numerical record is `examples/results/mshho.json`; compact algebraic,
source-convention and polynomial-patch checks run in the test suite.
