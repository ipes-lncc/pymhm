# Vector primal, displacement–pressure and H(div) stress tutorials

The example imports below refer to the verified local companion downloaded
by the notebook’s first cell; they are separate from the installed library.

Run 17 small, executable analytical patches: primal elasticity, Herrmann
displacement–pressure elasticity, weak-symmetry H(div) stress families,
incompressible flow and a vector Maxwell trajectory. Every stationary variant
uses two macrocells with explicit affine physical fields. The default boundary
data are nonhomogeneous; `homogeneous=True` supplies zero fields and loads.
These controls demonstrate the stated discrete formulations. Convergence
studies and literature comparisons are documented separately in the case gallery.

The primal elasticity construction follows
[Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046),
the displacement–pressure GaLS construction follows
[Gomes, Pereira and Valentin (2024, preprint v1)](https://arxiv.org/abs/2403.16890v1),
and the two-dimensional weak-symmetry mixed construction follows
[Devloo et al. (2021)](https://doi.org/10.1051/m2an/2021013).
The separate three-dimensional BDM stress family uses the classical
[Arnold, Falk and Winther (2007)](https://doi.org/10.1090/S0025-5718-07-01998-9)
spaces; it does not transfer a two-dimensional enrichment theorem to tetrahedra.

The primary elasticity and Brinkman cells declare their UFL equations through
the generic variational interface and require the compatible native DOLFINx/UFL backend. Their
method-family controls use editable providers in `examples.formulations` that
declare the actual coefficient spaces, operators, trace signs and physical
moment rows before calling `assemble`.
The primary Maxwell trajectory declares coefficient forms for each mass/curl
stage and composes them with free time-integration functions in the example.
Scalar, vector and mixed user-defined forms share the
[variational interface](../variational.md); its space and stability choices
remain explicit.

Choose the [problem notebook](notebooks.md) and edit `selected_methods`.
The elasticity, flow and Maxwell notebooks expose their own supported variants.

```bash
jupyter lab introductory_methods.ipynb
jupyter lab introductory_methods.ipynb
jupyter lab introductory_methods.ipynb
```

The notebooks report each physical field's error separately, its original-equation
residual where exposed, the local spaces, boundary convention and gauge. One
native BLAS/OpenMP thread is used per worker. The Maxwell stepper has its own
serial trajectory interface; select a stationary elasticity/flow variant for
`backend="thread"` or `backend="process"`. Its factors are closed by a context
manager. All variants are available through `selected_methods`; no reference solver
or external project is loaded by this tutorial.

## Choose the unknowns and spaces

| Notebook variants | Local unknowns and approximation | Macroface data |
|---|---|---|
| `elasticity-primal-2d`, `elasticity-primal-3d` | H1 Lagrange P3 displacement; raw symmetric stress is evaluated from strain | P1 vector traction coordinates; prescribed displacement |
| `elasticity-gals-2d`, `elasticity-gals-3d` | H1 P3 displacement and P3 Herrmann pressure, with GaLS stabilization | P1 vector Cauchy-traction coordinates; prescribed displacement |
| `elasticity-bdm-2d` | Row-wise BDM2 stress; DG P1 displacement and weak rotation | Interior P1 traction; exterior fine-edge P2 traction |
| `elasticity-bdm-plus-2d`, `elasticity-bdm-plusplus-2d` | Normal degree 2; BDM3 or BDM4 zero-normal interior enrichment; DG P2 or P3 displacement/rotation | The normal trace degree remains 2 |
| `elasticity-bdm-3d` | Tetrahedral BDM2 stress rows; DG P1 displacement and three axial rotation coordinates | P1 vector triangular traction |
| `elasticity-rt-2d`, `elasticity-rt-plus-2d` | Tensor RT normal degree 1 and interior order 1 or 2; displacement Q1 or Q2; rotation P1 or P2 | Interior P1; exterior independent fine-edge P1 traction |
| `flow-taylor-hood-2d`, `flow-taylor-hood-3d` | H1 velocity P3 and pressure P2 | P1 vector pseudotraction; prescribed velocity |
| `flow-usfem-2d`, `flow-usfem-3d` | H1 velocity/pressure P3/P3, with the stated minimum-2017 residual stabilization | P1 vector pseudotraction; prescribed velocity |
| `flow-oseen-2d`, `flow-oseen-3d` | H1 velocity/pressure P3/P3, with Oseen-2021 stabilization | P1 skew-transport pseudotraction; prescribed velocity |
| `maxwell-vector-3d` | Broken vector DG P2 electric and magnetic fields on refined tetrahedra | Tangential P1 traces and prescribed impedance data |

H1 in this table describes the **local** conforming Lagrange space. These
solvers perform multiscale hybrid assembly, with independent fields on the two
macrodomains. They do not produce a globally conforming classical H1 finite
element solve. Similarly, raw primal stress and flow pseudostress are distinct
from the explicitly H(div)-conforming mixed stress.

The BDM variants retain all zero-normal interior bubbles of the declared higher
degree, while preserving the normal trace degree. The tensor RT implementation
uses its full stated tensor-product spaces. These choices are specified in the
[mixed-family](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mixed-families.md) and
[tensor RT](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/elasticity-tensor-rt.md) pages. An instrumented comparison
against a restricted Unicamp implementation has its own discretization and
provenance; it is not identified as the pure BDM or tensor RT tutorial solver.
The public 3D mixed-elasticity solver presently supplies the tetrahedral BDM
family. A 3D tensor RT/enriched-stress elasticity solver is not implied by the
availability of scalar H(div) bases on other cell types.

## Local vector blocks use the same provider contract

The unknown of a local problem can be displacement, `(u,p)`, or
`(sigma,u,rotation)`. Its coefficient order is declared by that local assembler.
The generic provider declares both its local equations and global balance

$$
\begin{aligned}
A_K w_K+B_K\lambda_K&=f_K,\\
C_K w_K+D_K\lambda_K&=g_K.
\end{aligned}
$$

`LocalEquations` carries independent blocks, trace maps, literal retained
bases and physical moment columns. Its `metadata` supplies reconstruction data.
An ordinary callable provider returns this record, and `MultiscaleProblem`
combines it with an additional global `Equation` and ordered item specifications. The
shared assembler reduces each contribution in cell order, regardless of which
local spaces generated it. It does not infer the spaces, trace signs, gauge or
essential constraints from the number of vector components.

For a user-defined H1 velocity–pressure volume form, a Taylor–Hood UFL space and
the **grad-grad** operator used by the flow tutorials can be written explicitly:

```python
import basix.ufl
import ufl
from dolfinx import fem

element = basix.ufl.mixed_element([
    basix.ufl.element("Lagrange", local_mesh.basix_cell(), 3, shape=(dimension,)),
    basix.ufl.element("Lagrange", local_mesh.basix_cell(), 2),
])
V = fem.functionspace(local_mesh, element)
u, p = ufl.TrialFunctions(V)
v, q = ufl.TestFunctions(V)
dx = ufl.dx(domain=local_mesh)
a = (
    ufl.inner(ufl.grad(u), ufl.grad(v)) + 2 * ufl.inner(u, v)
    - p * ufl.div(v) - q * ufl.div(u)
) * dx
L = ufl.inner(force, v) * dx
```

Here `local_mesh` is owned by the provider on `MPI.COMM_SELF`, and `force` is
the explicitly supplied physical source. Signed trace forms, retained
translation coefficients and moment forms complete `LocalEquations`, with
independently declared trial/test trace pairings. See the
[variational guide](../variational.md).
With the positive reaction in this example, translations are retained modes
rather than exact null vectors; use `coarse_basis` when retaining them.
This volume form supplies neither USFEM/Oseen stabilization nor the interface
conditions by itself. `pymhm.backends.fenics.brinkman_forms` instead defines symmetric
strain diffusion; it has a different natural traction and rigid-motion kernel.

The vector-Laplacian MHM construction and USFEM conventions follow
[Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027).
Stable mixed local spaces and their additional hypotheses are analyzed by
[Araya et al. (2025)](https://doi.org/10.1137/24M1649368).
The Oseen extension follows
[Araya et al. (2021)](https://doi.org/10.1007/s10444-020-09833-8).

An H(div) stress formulation also needs its stress/displacement/rotation spaces,
weak-symmetry pairing and normal-stress treatment. Normal H(div) constraints are
essential data in the corresponding local variational formulation. Attaching
ordinary mixed volume forms or boundary-pressure terms does not construct the
MHM flux/stress interface conditions. The mixed providers declare the appropriate
augmented local equations explicitly; the same generic assembler consumes them.

For example, the Taylor–Hood provider composes public velocity energy,
divergence and pressure moments with a separately declared trace coupling:

```python
from pymhm import TriangleMesh, assemble
from examples.formulations.vector import define_flow, flow_constraints, recover_vector

definition = define_flow(
    TriangleMesh.unit_square(), formulation="taylor-hood",
    degree=2, local_refinement=2,
    viscosity=1.0, drag=2.0, source=(0.0, 0.0),
    dirichlet=(0.0, 0.0), mean_pressure=0.0,
)
system = assemble(definition.problem)
coefficients = system.solve(constraints=flow_constraints(definition, system))
solution = recover_vector(definition, system, coefficients)
print(coefficients.field("velocity")[0].evaluate([[0.1, 0.1]]))
```

Changing the local bilinear form occurs in the provider's `LocalEquations`,
while boundary values and gauges remain explicit application data. Named fields
carry the executed basis and reconstruction matrix independently of the optional
physical result record. Mixed stress providers additionally declare `stress`
and `stress_divergence`, using the same archived row-wise Piola coordinates.

## Physical data, pressure and rigid moments

Elasticity uses the independently specified affine displacement `u(x)=Gx+b`,
constant moduli `lambda=mu=1`, and zero volume force. Its physical stress is

$$
\sigma=G+G^{\mathsf T}+\operatorname{tr}(G)I.
$$

The displacement–pressure formulation uses the Herrmann convention

$$
p=-\lambda\operatorname{div}u,
\qquad \sigma=2\mu\varepsilon(u)-pI.
$$

At finite lambda, the boundary compressibility identity determines pressure;
an arbitrary pressure mean is not imposed. `incompressible=True` applies to the
mixed-elasticity variants, chooses a solenoidal affine displacement, and
prescribes the physical mean pressure 2.3. Its stress includes `-2.3 I`.
Primal elasticity requires finite lambda. The weak rotation is an independent
field, with its 2D scalar or 3D axial sign convention stated by the solver.

Full prescribed displacement removes global rigid ambiguity. Pure-traction
problems instead require three integrated rigid moments in 2D or six in 3D,
including rotations about the declared centroid. Those physical integrals are
the `rigid_moments` inputs; a point pin or mean translation alone does not supply
the complete gauge. The [elasticity API](../api/elasticity.md) gives each method's
traction and moment conventions.

Flow uses a trace-free affine velocity, resistance 2 and a linear pressure with
explicit mean 0.7 about the actual patch volume centroid. The body source is

$$
f=2u+G\beta+\nabla p.
$$

Taylor–Hood and USFEM use `beta=0`; Oseen uses the declared constant transport
vector. The operator is `-Delta u+2u+beta.grad(u)+grad(p)` with `div(u)=0`.
Its natural interface quantity is the grad-grad/skew-transport pseudotraction,
not the symmetric Cauchy traction of elasticity or `brinkman_forms`. P1 traces
represent this patch's affine pseudotraction. The pressure mean is an explicit
physical gauge under the fully prescribed velocity boundary.

## A short vector Maxwell trajectory

The tangential hybrid construction follows
[Lanteri et al. (2018)](https://doi.org/10.1137/16M110037X).
The stationary trajectory below is an original orientation and time-update
control, rather than a reproduction of a published wave experiment.

`maxwell-vector-3d` advances four steps with `dt=0.001`, unit permittivity and
permeability, constant `E=(1.2,-0.3,0.7)` and `H=(0.4,0.8,-0.2)`. The impedance
boundary supplies `E_tan-(H cross n)` with coefficient one. These stationary
nonzero fields have zero curl and verify tangential orientation on the shared
macroface while exercising the time updates and reused factors.

The final magnetic time is 0.004 and the electric time is 0.0045. Their field
errors and the cross-time energy balance are reported separately. This is a
stationary analytical trajectory, with no temporal convergence or wave
resolution claim. The stepper checks its original tangential equations at the
shared tolerance; its result does not expose a scalar algebraic residual, which
the tutorial records as `null`. The broken DG curl is not an H(curl)-conforming
reconstruction. Complex scalar Helmholtz is covered in the scalar tutorial.

## References

- Christopher Harder, Alexandre L. Madureira, and Frédéric Valentin (2016). *A hybrid-mixed method for elasticity*, ESAIM: M2AN 50, 311–336. [DOI: 10.1051/m2an/2015046](https://doi.org/10.1051/m2an/2015046).

- Antônio Tadeu Azevedo Gomes, Weslley da Silva Pereira, and Frédéric Valentin (2024). *A low-order locking-free multiscale finite element method for isotropic elasticity*, arXiv preprint, version 1, 25 March 2024. [arXiv: 2403.16890v1](https://arxiv.org/abs/2403.16890v1).

- Philippe R. B. Devloo, Agnaldo M. Farias, Sônia M. Gomes, Weslley Pereira, Antonio J. B. dos Santos, and Frédéric Valentin (2021). *New H(div)-conforming multiscale hybrid-mixed methods for the elasticity problem on polygonal meshes*, ESAIM: M2AN 55, 1005–1037. [DOI: 10.1051/m2an/2021013](https://doi.org/10.1051/m2an/2021013).

- Douglas N. Arnold, Richard S. Falk, and Ragnar Winther (2007). *Mixed finite element methods for linear elasticity with weakly imposed symmetry*. Mathematics of Computation 76, 1699–1723. [DOI: 10.1090/S0025-5718-07-01998-9](https://doi.org/10.1090/S0025-5718-07-01998-9). [Preprint: arXiv:math/0701506v1](https://arxiv.org/abs/math/0701506v1).

- Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2017). *Multiscale hybrid-mixed method for the Stokes and Brinkman equations—The method*, Computer Methods in Applied Mechanics and Engineering 324, 29–53. [DOI: 10.1016/j.cma.2017.05.027](https://doi.org/10.1016/j.cma.2017.05.027).

- Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2025). *Multiscale Hybrid-Mixed Methods for the Stokes and Brinkman Equations—A Priori Analysis*, SIAM Journal on Numerical Analysis 63(2), 588–618. [DOI: 10.1137/24M1649368](https://doi.org/10.1137/24M1649368).

- Rodolfo Araya, Cristian Cárcamo, Abner H. Poza, and Frédéric Valentin (2021). *An adaptive multiscale hybrid-mixed method for the Oseen equations*, Advances in Computational Mathematics 47, article 15. [DOI: 10.1007/s10444-020-09833-8](https://doi.org/10.1007/s10444-020-09833-8).

- Stéphane Lanteri, Diego Paredes, Claire Scheid, and Frédéric Valentin (2018). *The Multiscale Hybrid-Mixed method for the Maxwell Equations in Heterogeneous Media*. Multiscale Modeling & Simulation 16(4) 1648-1683. [DOI: 10.1137/16M110037X](https://doi.org/10.1137/16M110037X).
