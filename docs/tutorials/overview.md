# Overview: define a multiscale problem from its equations

PyMHM connects independently defined local problems to a global interface
problem. You choose the spaces, physical coefficients, weak forms and boundary
conditions. The package compiles the forms, eliminates local coordinates,
assembles the shared interface equation and reconstructs the local fields.

This tutorial introduces that workflow through a small, runnable example. It
then shows how the same interface accommodates UFL, scalar and vector fields,
alternative local solvers, parallel execution and several scales. The
[introductory physical tutorials](../tutorials.md#self-contained-introductory-course)
develop those ideas with heterogeneous materials, field plots and reference
comparisons.

The decomposition into local kernels, complementary responses and interface
equations is developed by [Harder and Valentin (2016)](https://doi.org/10.1007/978-3-319-41640-3_13).
This tutorial demonstrates PyMHM's equation interface; its first interval example
is a conforming Galerkin system rather than a literature reproduction.

## Install and choose the level of description

The example below uses the portable core:

```bash
python -m pip install pymhm
```

It runs on the supported Linux, macOS and Windows Python environments without
DOLFINx, PETSc or MPI. UFL/DOLFINx compilation is an optional native integration;
the [installation guide](../installation.md) describes its environments.

Three objects describe the equations:

| Object | Mathematical responsibility |
| --- | --- |
| `LocalEquations` | Local volume equation and its contribution to the interface balance. |
| `Equation` | Additional global bilinear form and right-hand side. |
| `MultiscaleProblem` | Ordered local items, their provider, global coordinates, retained modes and constraints. |

### Start with the two scales

Think of a coarse partition of the domain into macroelements $K$, with scale
$H$. Inside each $K$, choose a mesh or another approximation space fine enough
to represent its material and local response; its scale is $h$. These local
coordinates are eliminated before the global solve. The global coordinates
live on macroelement interfaces, with any retained local modes added explicitly.
Refining a local mesh therefore changes the local response without creating
more macroelements.

There are three decisions before writing code:

1. Choose the local field and its trial and test spaces: scalar potential,
   vector displacement, or a mixed velocity–pressure field, for example.
2. Choose the interface variable and its space, including its orientation
   and the map from each macroelement to global coefficients.
3. Write both the local equation and the interface balance. Include any local
   kernel, physical normalization and global boundary condition.

### Give the two equations their API names

For an item $K$, the local provider declares

$$
\begin{aligned}
a_K(u_K,v_K)+b_K(\lambda_K,v_K)&=L_K(v_K),\\
c_K(u_K,\mu_K)+d_K(\lambda_K,\mu_K)&=g_K(\mu_K).
\end{aligned}
$$

The first equation determines the local response to a source and interface
coefficients. The second contributes to the global balance. In assembled
coordinates these equations read

$$
\begin{aligned}
A_Ku_K+B_K\lambda_K&=f_K,\\
C_Ku_K+D_K\lambda_K&=g_K.
\end{aligned}
$$

Rows correspond to test coordinates and columns to trial coordinates. You
declare the signs of $B_K$ and $C_K$ explicitly. The physical meaning of
$\lambda$ is also part of your formulation: it can represent a potential,
normal flux, traction or another interface variable.

| Mathematical term | Provider declaration | Meaning |
| --- | --- | --- |
| $a_K(u_K,v_K)$ or $A_Ku_K$ | `a=...` | The local volume operator. |
| $L_K(v_K)$ or $f_K$ | `L=...` | The local source. |
| $b_K(\lambda_K,v_K)$ or $B_K\lambda_K$ | `b=...` | How interface coordinates act on local test functions. |
| $c_K(u_K,\mu_K)$ or $C_Ku_K$ | `c=...` | How the local response enters global test equations. |
| $d_K(\lambda_K,\mu_K)$ or $D_K\lambda_K$ | `d=...` | A direct local contribution to the interface operator. |
| $g_K(\mu_K)$ or $g_K$ | `g=...` | A direct local contribution to the global load. |
| Local-to-global trace map | `dofs=...` | Which global coordinates the local trial trace uses. |
| Additional global $a_G$ and $L_G$ | `Equation(a_G, L_G)` | Terms assembled in the complete reduced coordinate order. |

Trial and test trace maps can differ; supply `test_dofs` when they do. For
the first example their maps coincide, so `dofs` suffices.

## A complete first example

Consider the one-dimensional problem

$$
-u''=1\quad\text{in }(0,1),\qquad u(0)=u(1)=0.
$$

Its analytical solution is $u(x)=x(1-x)/2$. Divide the interval into two
macrointervals, each containing two linear finite elements. The five nodes
are $0,1/4,1/2,3/4,1$. Prescribe the two exterior values and keep the value at
$x=1/2$ as the shared interface coordinate $\lambda$.

Each macrointerval has one remaining interior value $u_K$. A linear element
of length $h=1/4$ has stiffness and constant-source load

$$
A_e=\frac{1}{h}\begin{bmatrix}1&-1\\-1&1\end{bmatrix},\qquad
f_e=\frac{h}{2}\begin{bmatrix}1\\1\end{bmatrix}.
$$

After imposing the exterior value, each local contribution is therefore

$$
\begin{aligned}
8u_K-4\lambda&=\frac14,\\
-4u_K+4\lambda&=\frac18.
\end{aligned}
$$

The second rows are summed at the shared node. They are local contributions
to one global equation, rather than independent conditions on each cell.
This example is a conforming Galerkin system expressed through hybrid
elimination. The physical tutorials use the same records to declare their
MHM, MsHHO and MH²M constructions.

### 1. Write the local provider

A provider is an ordinary Python callable. Its argument identifies a local
item; its return value contains that item's equations and coordinate map.

```python
import numpy as np

from pymhm import Equation, LocalEquations, MultiscaleProblem, assemble


def local_equations(cell: int) -> LocalEquations:
    """Declare one macrointerval's interior and shared-node equations."""
    return LocalEquations(
        a=[[8.0]],
        L=[0.25],
        b=[[-4.0]],
        c=[[-4.0]],
        d=[[4.0]],
        g=[0.125],
        dofs=[0],
        metadata={"interior_x": 0.25 + 0.5 * cell},
    )
```

Both items use `dofs=[0]` because their interface contributions belong to the
same global coordinate. In a mesh, this map comes from its skeleton incidence
and the declared trace basis. The example's matrices have already been
assembled; they require no form compiler or case-specific solver.

### 2. Declare the global problem

```python
problem = MultiscaleProblem(
    global_equation=Equation(a=0, L=0),
    local_provider=local_equations,
    items=(0, 1),
    trace_size=1,
    coarse_sizes=(0, 0),
)
```

`trace_size=1` declares the shared coordinate. `coarse_sizes=(0, 0)` declares
that neither item retains an additional local mode. `Equation(0, 0)` adds
zero global terms; the local second equations already provide the entire
interface balance.

An independent global form can add boundary terms, loads or couplings between
items. Its coordinate order is all trace coefficients followed by the retained
coefficients of each item, in `items` order. Exterior Dirichlet values in this
example have already been eliminated in the declared local blocks. In a
different formulation, `fixed` can prescribe represented trace coefficients
and `constraints` can impose explicit physical integral conditions.

### 3. Assemble, solve and reconstruct

```python
system = assemble(problem)
solution = system.solve()

print("Interface value:", solution.trace)
print("Interior values:", [float(field[0]) for field in solution.fields])
print("Reduced matrix:", system.matrix.toarray())
print("Reduced load:", system.rhs)
```

The output is

```text
Interface value: [0.125]
Interior values: [0.09375, 0.09375]
Reduced matrix: [[4.]]
Reduced load: [0.5]
```

For an invertible local operator, elimination forms

$$
\begin{aligned}
S_K&=D_K-C_KA_K^{-1}B_K,\\
r_K&=g_K-C_KA_K^{-1}f_K.
\end{aligned}
$$

Here every item contributes $S_K=2$ and $r_K=1/4$. Their sum gives
$4\lambda=1/2$. PyMHM computes the local source and interface responses
using the same local factorization, assembles the global equation and recovers
the interior coordinates after the global solve.

### 4. Check what the coefficients represent

```python
nodes = np.linspace(0.0, 1.0, 5)
nodal_values = np.array([
    0.0,
    solution.fields[0][0],
    solution.trace[0],
    solution.fields[1][0],
    0.0,
])
analytical_values = nodes * (1.0 - nodes) / 2.0
np.testing.assert_allclose(
    nodal_values, analytical_values, rtol=1e-11, atol=1e-12,
)
```

The finite-element nodal values agree with the analytical solution for this
constant-source patch. The reconstructed field is piecewise linear; it is
an approximation to the quadratic solution between nodes. `solution.fields`
contains coefficients in each provider's local basis. Physical evaluation,
field names and units follow that basis and belong to the problem definition.

Use `solve(problem)` when you want assembly and solution in one call. Keeping
the assembled `system` is useful when inspecting the reduced equation,
reconstructing given coordinates or applying a compatible global load update.
The [variational guide](../variational.md) explains these operations and
their source restrictions.

## Replace assembled blocks with UFL forms

For a spatial problem, a provider can create a local DOLFINx mesh and space,
then express the weak operator directly with UFL. For a scalar diffusion
problem with a Neumann-local MHM construction, the volume equation is

$$
\begin{aligned}
\int_K (\kappa\nabla u)\cdot\nabla v\,dx
&+\int_{\partial K}\lambda v\,ds\\
&=\int_K f v\,dx.
\end{aligned}
$$

In this convention, $\lambda$ is outward normal physical flux, with
$q=-\kappa\nabla u$. Each local pairing includes its orientation relative
to the globally declared normal. Once the provider has created `V`, its
measures, material and signed trace functions, it can return

```python
import ufl

from pymhm import LocalEquations, columns, rows

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

forms = LocalEquations(
    a=ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * dx,
    L=source * v * dx,
    b=columns(*(phi * v * ds for phi in signed_trace_functions)),
    c=rows(*(-phi * u * ds for phi in signed_trace_functions)),
    dofs=trace_dofs,
    kernel=constant_coefficients,
    moments=columns(v * dx),
)
```

This is a contextual form declaration: the provider supplies the local space,
quadrature, face support, orientation and coefficient arrays. `columns`
assembles one source-like form per interface trial coordinate; `rows`
assembles one pairing per interface test coordinate. No trace basis on another
mesh is identified implicitly.

The Neumann diffusion operator has a constant kernel. Declare its coefficients
as an $n$-by-1 array and its physical mean through `moments`; retain one mode
per item through `coarse_sizes`. Global pressure boundary data or a physical
mean gauge complete the problem. These choices are developed in the
[Darcy convergence tutorial](introduction/darcy_multiscale_convergence.md).

The primal Neumann-local construction follows
[Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019).
UFL's mathematical notation and symbolic form representation are described by
[Alnæs et al. (2014)](https://doi.org/10.1145/2566630).

The native adapter currently compiles real linear and bilinear forms on
single-rank DOLFINx domains, normally `MPI.COMM_SELF`. The eliminated local
operator is square. Cross-mesh integration needs explicit entity maps; gauges,
essential conditions and stable space choices remain explicit. See the
[native compilation contract](../variational.md#native-compilation-limits).

### Follow the interface balance as well as the volume form

For the UFL declaration above, let $\phi_{Kj}$ denote a trace basis function
with its local orientation included. Then

$$
\begin{aligned}
b_K(\lambda_K,v)&=
\sum_j\lambda_j\int_{\partial K}\phi_{Kj}v\,ds,\\
c_K(u_K,\mu_K)&=
-\sum_j\mu_j\int_{\partial K}\phi_{Kj}u_K\,ds.
\end{aligned}
$$

`columns(phi * v * ds)` constructs the corresponding $B_K$ column;
`rows(-phi * u * ds)` constructs the corresponding $C_K$ row. Summing those
rows over adjacent macroelements supplies weak potential continuity with the
declared normal convention. On exterior faces with homogeneous prescribed
potential, that balance has zero load. Nonhomogeneous data supply the matching
signed boundary pairing.

Thus a user-written diffusion provider is attached to the global problem with
the same construction as the portable example:

```python
diffusion_problem = MultiscaleProblem(
    global_equation=Equation(0, 0),
    local_provider=diffusion_provider,
    items=range(len(macro_mesh.cells)),
    trace_size=skeleton.size,
    coarse_sizes=(1,) * len(macro_mesh.cells),
)
```

Here `diffusion_provider` creates the local space and returns the displayed
forms for each macroelement. `macro_mesh` and `skeleton` supply the geometry
and trace numbering. One retained constant per item accounts for the local
Neumann kernel. The [complete Darcy tutorial](introduction/darcy_multiscale_convergence.md)
defines these objects, the physical data and evaluation functions step by
step; it also shows how to choose compatible local and skeletal resolutions.

## Scalar, vector and mixed unknowns use the same records

`LocalEquations` describes coordinates and forms rather than selecting a
physical model. A vector trial function makes a vector equation; a mixed space
combines several fields in the same local operator.

For example, on a stable mixed velocity–pressure space `W`, a symmetric
Stokes–Brinkman volume form can be written as

```python
velocity, pressure = ufl.TrialFunctions(W)
test_velocity, test_pressure = ufl.TestFunctions(W)

a = (
    2 * viscosity * ufl.inner(
        ufl.sym(ufl.grad(velocity)), ufl.sym(ufl.grad(test_velocity)),
    )
    + ufl.inner(resistance * velocity, test_velocity)
    - pressure * ufl.div(test_velocity)
    - test_pressure * ufl.div(velocity)
) * dx
L = ufl.inner(body_force, test_velocity) * dx
```

The trace pairings, pressure gauge and local constraints complete that
construction. A volume form alone does not select Taylor–Hood spaces or
equal-order stabilization. The [Stokes–Brinkman tutorial](introduction/stokes_brinkman_boundary_layer.md)
declares its executed operator, spaces and boundary conventions in full.

Mixed $H(\mathrm{div})$ problems similarly declare flux/stress coordinates,
divergence pairings, pressure/displacement spaces and normal traces. Basix
supplies reference elements and entity transformations; the physical mapping,
restricted moments and global orientation remain part of the construction.
The [scalar](scalar.md) and [vector](vector.md) guides list the verified
space variants. Constants, rigid motions and other retained modes require
their own declared bases and physical moments.

## Change local execution independently of the equations

The first example also runs with bounded thread batches:

```python
from pymhm import ExecutionConfig

threaded_system = assemble(
    problem,
    execution=ExecutionConfig(
        backend="thread", workers=2, batch_size=2, native_threads=1,
    ),
)
threaded_solution = threaded_system.solve()
np.testing.assert_allclose(
    threaded_solution.trace, solution.trace, rtol=1e-11, atol=1e-12,
)
```

Serial execution constructs and contributes one item at a time. Thread and
process execution construct and eliminate independent items in bounded
batches. The coordinator reduces contributions in item order, including
contributions to shared faces. `native_threads=1` controls supported native
BLAS/OpenMP thread pools within that worker budget.

Use `backend="process"` for spawned workers. Put providers in an importable
module and protect a script's execution with `if __name__ == "__main__":`.
Create meshes, solver objects and communicators inside their owning worker;
return owned arrays and portable metadata. Live native resources do not
cross process boundaries. The [process tutorial](introduction/darcy_process_scalability.md)
shows an importable worker definition from notebook cells.

Threads require independent mutable workspaces and thread-safe native calls.
Parallelism is a scheduling choice, and its speed depends on local work,
startup, transfer and global solve costs. The
[2D](introduction/darcy_parallel_scalability.md) and
[3D performance tutorials](introduction/darcy_3d_parallel_scalability.md)
report complete timings and physical field checks. The
[execution guide](../execution.md) separately describes MPI assembly,
resident GPU factors and explicit multi-GPU condensation.

## Supply another local solver or numerical backend

There are two extension points:

| Extension | What you provide |
| --- | --- |
| Local provider | `LocalEquations` or `CompiledLocalEquations`, built with your FEM package, discretization or response construction. |
| Local numerical solver | All constrained source, trace and retained-mode response columns for the supplied operator. |

For the runnable example, a small custom numerical solver can be

```python
from numpy.typing import NDArray
from scipy.sparse import csc_matrix

from pymhm import SolverConfig


def dense_response(
    matrix: csc_matrix, rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve every supplied local response column in the declared coordinates."""
    return np.linalg.solve(matrix.toarray(), rhs)


custom_system = assemble(
    problem, solvers=SolverConfig(local_solver=dense_response),
)
custom_solution = custom_system.solve()
np.testing.assert_allclose(
    custom_solution.trace, solution.trace, rtol=1e-11, atol=1e-12,
)
```

A custom solver receives owned copies of the constrained operator and all
right-hand sides. PyMHM checks each returned column against the original
equations. An external simulation package or learned response model must
provide the same coordinates and complete response contract. Its physical
accuracy and stability require separate evidence.

`SolverConfig` also selects named local and global solvers. The
[solver guide](../solvers.md) describes SciPy, PARDISO, PETSc/MUMPS and
optional accelerator adapters, including which operators admit elliptic AMG.
The local solver and global solver can differ. Reusing kernels or solver
infrastructure does not imply that different material matrices share a
factorization.

## Express different methods and finer scales

The form interface accommodates several constructions by changing the declared
spaces and pairings:

| Construction | Distinguishing data | Next tutorial |
| --- | --- | --- |
| Neumann-local MHM | Flux or traction trace, local kernel, physical moments and compatibility rows. | [Darcy](introduction/darcy_multiscale_convergence.md), [elasticity](introduction/multiscale_elasticity.md) |
| Mixed local MHM | Flux/stress and potential/displacement spaces with compatible divergence and normal trace. | [Scalar and mixed variants](scalar.md) |
| MHM-USFEM | Explicit residual stabilization and matching stabilized source and interface terms. | [Reaction–diffusion layers](introduction/mhm_usfem_rad.md), [velocity–pressure layers](introduction/stokes_brinkman_boundary_layer.md) |
| MsHHO | Cell/face moments and constrained energy reconstruction before condensation. | [MsHHO](introduction/mshho_multiscale.md) |
| Three-field MH²M | Independent conormal and pressure traces, local normalization and source lift. | [MH²M](introduction/mh2m_multiscale.md) |
| Robin-local MH and residual Petrov–Galerkin MHM | Matching local/global Robin terms, or independent trial/test pairings and explicit facet terms. | [Variational constructions](../variational.md#represent-distinct-method-constructions) |

These are distinct methods with distinct compatibility and approximation
hypotheses. Their algebra is expressible through the shared interface; their
scientific qualification applies to the spaces and data stated in each study.

For their mathematical constructions, consult
[Durán et al. (2019)](https://doi.org/10.1016/j.cma.2019.05.013) for mixed local Darcy,
[Santiago, Valentin and Martins (2025)](https://doi.org/10.55592/cilamce2025.v5i.14270)
for scalar MHM-USFEM and
[Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027) for its
Stokes–Brinkman counterpart. The MsHHO and three-field constructions follow
[Chaumont-Frelet et al. (2022)](https://doi.org/10.1051/m2an/2021082) and
[de Barros, Madureira and Valentin (2026, version 3)](https://arxiv.org/abs/2404.16978v3),
respectively. Robin-local MH follows
[Barrenechea, Gomes and Paredes (2024)](https://doi.org/10.1137/22M1542556), and
residual Petrov–Galerkin MHM follows
[Fernando et al. (2023)](https://doi.org/10.1007/s40314-023-02304-y).

The [literature guide](../literature.md) and
[verification pages](../verification.md) separate that evidence from
literature predictions and planned extensions.

For several scales, a local operator can itself be a `MultiscaleProblem`.
Its reduced coordinates become the parent's local coordinates. `NestedEquations`
additionally declares a child boundary restriction and its orientation map.
Children assemble serially inside their owning worker; the outer execution
policy controls the worker budget. Children leave physical gauges and
prescribed trace values to the parent.

The returned solution preserves this hierarchy through `solution.children`.
Leaf field reconstruction uses the executed local bases. A physical gauge
must integrate the physical leaf fields, including source offsets, rather
than take a mean of unrelated coarse coefficients. The
[recursive formulation guide](../variational.md#reuse-the-interface-at-finer-scales)
provides the full coordinate contract and executable notebook links.

Continue with [multiscale Darcy and convergence](introduction/darcy_multiscale_convergence.md)
for the first complete spatial workflow, or choose another problem in the
[tutorial catalogue](../tutorials.md).

## References

- Christopher Harder and Frédéric Valentin (2016). *Foundations of the MHM Method*, in *Building Bridges: Connections and Challenges in Modern Approaches to Numerical Partial Differential Equations*, Lecture Notes in Computational Science and Engineering 114, Springer. [DOI: 10.1007/978-3-319-41640-3_13](https://doi.org/10.1007/978-3-319-41640-3_13).

- Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *A family of Multiscale Hybrid-Mixed finite element methods for the Darcy equation with rough coefficients*, Journal of Computational Physics 245, 107–130. [DOI: 10.1016/j.jcp.2013.03.019](https://doi.org/10.1016/j.jcp.2013.03.019).

- Martin S. Alnæs, Anders Logg, Kristian B. Ølgaard, Marie E. Rognes, and Garth N. Wells (2014). *Unified Form Language: A domain-specific language for weak formulations of partial differential equations*. ACM Transactions on Mathematical Software 40(2), article 9, 1–37. [DOI: 10.1145/2566630](https://doi.org/10.1145/2566630). [Author preprint: arXiv:1211.4047v2](https://arxiv.org/abs/1211.4047v2).

- Omar Durán, Philippe R. B. Devloo, Sônia M. Gomes, and Frédéric Valentin (2019). *A multiscale hybrid method for Darcy’s problems using mixed finite element local solvers*, Computer Methods in Applied Mechanics and Engineering 354, 213–244. [DOI: 10.1016/j.cma.2019.05.013](https://doi.org/10.1016/j.cma.2019.05.013).

- Juan Felipe Pacazuca Santiago, Frédéric Valentin, and Larissa Martins (2025). *A Multiscale Hybrid-Mixed Method with Local Stabilization*. Proceedings of the Ibero-Latin American Congress on Computational Methods in Engineering, CILAMCE 2025, volume 5, article 14270; published online 18 March 2026. [DOI: 10.55592/cilamce2025.v5i.14270](https://doi.org/10.55592/cilamce2025.v5i.14270).

- Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2017). *Multiscale hybrid-mixed method for the Stokes and Brinkman equations—The method*, Computer Methods in Applied Mechanics and Engineering 324, 29–53. [DOI: 10.1016/j.cma.2017.05.027](https://doi.org/10.1016/j.cma.2017.05.027).

- Théophile Chaumont-Frelet, Alexandre Ern, Simon Lemaire, and Frédéric Valentin (2022). *Bridging the Multiscale Hybrid-Mixed and Multiscale Hybrid High-Order Methods*, ESAIM: M2AN 56, 261–285. [DOI: 10.1051/m2an/2021082](https://doi.org/10.1051/m2an/2021082).

- Franklin de Barros, Alexandre L. Madureira, and Frédéric Valentin (2026). *A three-field Multiscale Method*. arXiv preprint, version 3, 5 August 2026; first submitted 25 April 2024. [arXiv: 2404.16978v3](https://arxiv.org/abs/2404.16978v3).

- Gabriel R. Barrenechea, Antonio Tadeu A. Gomes, and Diego Paredes (2024). *A Multiscale Hybrid Method*. SIAM Journal on Scientific Computing 46(3), A1628–A1657. [DOI: 10.1137/22M1542556](https://doi.org/10.1137/22M1542556).

- Honório Fernando, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2023). *A Petrov–Galerkin multiscale hybrid-mixed method for the Darcy equation on polytopes*. Computational and Applied Mathematics 42, article 173. [DOI: 10.1007/s40314-023-02304-y](https://doi.org/10.1007/s40314-023-02304-y).
