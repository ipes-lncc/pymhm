# Vector primal, displacement–pressure and H(div) stress tutorials

Run 17 small, executable analytical patches: primal elasticity, Herrmann
displacement–pressure elasticity, weak-symmetry H(div) stress families,
incompressible flow and a vector Maxwell trajectory. Every stationary variant
uses two macrocells with explicit affine physical fields. The default boundary
data are nonhomogeneous; `--homogeneous` supplies zero fields and loads.
These controls demonstrate the stated discrete formulations. Convergence
studies and literature comparisons are documented separately in the case gallery.

```bash
pixi run --locked -e test python -m examples.tutorial_vector_variants \
  --variant all
pixi run --locked -e test python -m examples.tutorial_vector_variants \
  --variant elasticity-bdm-plus-2d --backend process --workers 2
pixi run --locked -e test python -m examples.tutorial_vector_variants \
  --variant elasticity-gals-3d --incompressible
pixi run --locked -e test python -m examples.tutorial_vector_variants \
  --variant flow-usfem-3d --homogeneous
```

The script prints each physical field's error separately, its original-equation
residual where exposed, the local spaces, boundary convention and gauge. One
native BLAS/OpenMP thread is used per worker. The Maxwell stepper has its own
serial trajectory interface; select a stationary elasticity/flow variant for
`--backend thread` or `--backend process`. Its factors are closed by a context
manager. All variants are available through `--variant`; no reference solver
or external project is loaded by this tutorial.

## Choose the unknowns and spaces

| CLI variants | Local unknowns and approximation | Macroface data |
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
[mixed-family](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-families.md) and
[tensor RT](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity-tensor-rt.md) pages. An instrumented comparison
against a restricted Unicamp implementation has its own discretization and
provenance; it is not identified as the pure BDM or tensor RT tutorial solver.
The public 3D mixed-elasticity solver presently supplies the tetrahedral BDM
family. A 3D tensor RT/enriched-stress elasticity solver is not implied by the
availability of scalar H(div) bases on other cell types.

## Local vector blocks use the same provider contract

The unknown of a local problem can be displacement, `(u,p)`, or
`(sigma,u,rotation)`. Its coefficient order is declared by that local assembler.
Every method supplies the same numerical record

$$
A_K w_K+B_K\lambda_K=f_K.
$$

`LocalProblem` carries the original operator, signed trace map, literal retained
basis and physical moment columns. `LocalAssembly` adds reconstruction metadata.
An ordinary callable provider returns one of these records, and `HybridProblem`
combines it with a `GlobalForm` and the ordered macrocell specifications. The
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
translation coefficients and moment forms complete the `LocalForm`. See the
[complete local/global provider example](../fenics.md#local-provider-and-global-hybrid-form).
This volume form supplies neither USFEM/Oseen stabilization nor the interface
conditions by itself. `pymhm.fenics.brinkman_forms` instead defines symmetric
strain diffusion; it has a different natural traction and rigid-motion kernel.

An H(div) stress formulation also needs its stress/displacement/rotation spaces,
weak-symmetry pairing and normal-stress treatment. Normal H(div) constraints are
essential data in the corresponding local variational formulation. Attaching
ordinary mixed volume forms or boundary-pressure terms does not construct the
MHM flux/stress interface conditions. Use the supported high-level mixed
assemblers, or supply the appropriate augmented/eliminated local equations.

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
an arbitrary pressure mean is not imposed. `--incompressible` applies to the
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
