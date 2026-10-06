# Local providers, forms and execution

This tutorial declares primal Galerkin and mixed $H(\mathrm{div})$ Darcy
problems through `LocalEquations`, `Equation` and `MultiscaleProblem`.
A provider is an ordinary callable. It supplies the physical equations and
coefficient maps; `assemble` supplies compilation, local elimination and
ordered shared-face reduction. Changing the assembly backend does not require
subclassing the solver or selecting a PDE-specific runtime entry point.

The local/global decomposition is developed by
[Harder and Valentin (2016)](https://doi.org/10.1007/978-3-319-41640-3_13).
The primal and mixed Darcy constructions below follow
[Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019)
and [Durán et al. (2019)](https://doi.org/10.1016/j.cma.2019.05.013), respectively.
Their analytical patch data are defined for this tutorial.

Open the [local/global provider notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/local_global_providers.ipynb)
and edit the forms, boundary, local solver and execution cells. The
[UFL notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/ufl_provider.ipynb)
contains user-written UFL forms and reports whether DOLFINx executed.

```bash
pixi run --locked -e notebooks notebooks-run foundations/operators/local_global_providers.ipynb
pixi run --locked -e notebooks notebooks-run foundations/operators/ufl_provider.ipynb
```

The first example uses two macrotriangles and two subdivisions per local
edge. Its independently differentiated analytical data are

$$
p=1+x^2+y^2,\qquad q=-\nabla p=-2(x,y),\qquad f=\nabla\cdot q=-4.
$$

The primal local space is continuous $P_2$ pressure; the mixed space is RT0
Darcy flux with $P_0$ pressure and local boundary-pressure multipliers.
Both use a $P_0$ normal-flux skeleton. Exact normal flux is constant on each
straight edge. Primal pressure and flux are exact up to floating-point errors
on this patch; mixed pressure has a nonzero approximation error because it
is piecewise constant. This patch checks the declared equations. It does not
replace a literature reproduction or an inf-sup convergence study.

## Declare the local and global forms

The two equations are explicit:

$$
\begin{aligned}
a(u,v)+b(\lambda,v)&=L(v),\\
c(u,\mu)+d(\lambda,\mu)&=g(\mu).
\end{aligned}
$$

The notebook's `TutorialProvider` integrates its stated local spaces with
shared FEM operations. It returns `LocalEquations` with $B$ and the explicitly
chosen $C=-B^T$, literal kernel coefficients and physical pressure moments.
It does not call a packaged Darcy solver. `columns` and `rows` preserve the
ordered trace pairings and their signs.

```python
import numpy as np
from pymhm import Equation, MultiscaleProblem, SkeletonSpace, TriangleMesh, assemble
from pymhm.fem.scalar.operators import boundary_data
from examples.tutorial_local_provider import TutorialProvider, exact_pressure

mesh = TriangleMesh.unit_square()
skeleton = SkeletonSpace(mesh)
provider = TutorialProvider(mesh, skeleton, formulation="primal", local_refinement=2)
boundary, fixed = boundary_data(skeleton, exact_pressure, order=4)
problem = MultiscaleProblem(
    global_equation=Equation(0, -np.r_[boundary, np.zeros(len(mesh.cells))]),
    local_provider=provider,
    items=range(len(mesh.cells)),
    trace_size=skeleton.size,
    coarse_sizes=(1,) * len(mesh.cells),
    fixed=fixed,
)
system = assemble(problem)
solution = system.solve()
```

The global load is declared in trace-first, retained-cell-second coordinates.
The primal Dirichlet pairing is `-boundary`; it follows the chosen global
rows $C=-B^T$. An additional global bilinear form could supply other
couplings. `Equation(0, L)` supplies no such addition. This interface accepts
actual numerical or UFL global forms; the fixed hybrid `GlobalForm` record only
specifies the fixed hybrid construction's layout and boundary convention.
See the [variational guide](../variational.md) for independent trial/test maps,
direct local global terms and compilation limits.

## Keep mixed fields and boundary variables explicit

For the mixed provider, local coefficients are flux $q_h$, pressure $p_h$,
then boundary-pressure multipliers $\eta_h$. The original local equations are

$$
\begin{bmatrix}
M&-D^T&S\\
-D&0&0\\
S^T&0&0
\end{bmatrix}
\begin{bmatrix}q_h\\p_h\\\eta_h\end{bmatrix}
+\begin{bmatrix}0\\0\\-F\end{bmatrix}\lambda
=\begin{bmatrix}0\\-f_h\\0\end{bmatrix}.
$$

$M$ is the inverse-permeability RT0 mass matrix; $D$ is integrated divergence;
$S$ selects fine boundary normal-flux coefficients; $F$ maps oriented skeletal
flux density to those integrated coefficients. The joint kernel shifts $p_h$
and $\eta_h$ together. Physical moments integrate only $p_h$ and give zero
weight to auxiliary boundary multipliers.

This is flux-prescribing mixed MHM: the skeleton is physical normal Darcy
flux, and the local boundary variable is pressure. Its declared global
Dirichlet load is `boundary`, consistently with its actual coupling signs.
RT0 needs degree-zero trace segments aligned with fine boundary edges.
Primal volume flux is the raw gradient field $-\nabla p_h$; mixed RT0 flux is
$H(\mathrm{div})$ conforming within each local mesh. Macro conservation alone
does not imply fine-cell conservation of a general primal reconstruction.

The notebook also constructs a pure-Neumann problem with compatible outward
flux and physical pressure integral $5/3$. For that problem, form the gauge
from the provider's executed physical weights:

```python
from examples.tutorial_local_provider import build_problem

problem = build_problem(formulation="mixed", boundary="neumann")
system = assemble(problem)
weights = [record[1] for record in system.local_metadata]
gauge = system.mean_constraint(weights, value=5 / 3)
solution = system.solve(constraints=[gauge])
```

The metadata is `(fine_mesh, physical_pressure_weights)` in the executed
coefficient order. The notebook reports physical field errors and original
algebraic row norms separately by field block. Assembly uses four Gauss points
per Duffy coordinate and error integration uses eight.

## Replace the numerical local solver

`SolverConfig.local_solver` accepts a callable on owned copies of the
constrained sparse operator and all source, trace and retained-mode
right-hand-side columns. It returns every response column, including constraint
multipliers, in that coefficient basis.

```python
import numpy as np
from pymhm import SolverConfig

def dense_response(matrix, rhs):
    return np.linalg.solve(matrix.toarray(), rhs)

system = assemble(problem, solvers=SolverConfig(local_solver=dense_response))
```

An external solver or response model can implement this contract. The shared
solver checks the original augmented operator and each returned column at
its unchanged relative residual criterion. These checks preserve the equations;
they do not establish approximation accuracy or a mesh-independent inf-sup
constant. Release native resources before returning coefficient arrays.

## Choose bounded execution independently

```python
from pymhm import ExecutionConfig

system = assemble(
    problem,
    execution=ExecutionConfig(
        backend="process", workers=2, native_threads=1, batch_size=1,
    ),
)
```

Serial execution builds and contributes one cell before taking the next.
Thread and process execution use bounded batches; the coordinator sums shared
trace entries in cell order. With `batch_size=None`, the effective worker
count bounds the parallel batch. Assembly owns early-exit cleanup.

Processes use spawn. Providers, compilers, callbacks and items must be
picklable, and executable process entry points need a `__main__` guard.
Construct native meshes and forms within the worker; return owned numerical
operators and portable reconstruction metadata. Closures can be used in
serial and thread execution. Recursive children use this same form interface
but assemble serially inside their owning worker, so the outer executor controls
the worker budget.

## Write native UFL forms directly

UFL represents weak forms in mathematical notation, as described by
[Alnæs et al. (2014)](https://doi.org/10.1145/2566630).

The UFL notebook supplies a native context inside each worker and defines
both pairings as forms. Its affine solution is $p=1+x+2y$ on the same two
macrotriangles, with continuous local $P_1$ pressure and a $P_0$ normal-flux
skeleton. On each local space `V`, the essential declaration is

```python
import ufl
from pymhm import LocalEquations, columns, rows

p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
forms = LocalEquations(
    a=ufl.inner(ufl.grad(p), ufl.grad(v)) * dx,
    L=0.0 * v * dx,
    b=columns(*(sign * v * ds(i + 1) for i, sign in enumerate(signs))),
    c=rows(*(-sign * p * ds(i + 1) for i, sign in enumerate(signs))),
    dofs=trace_dofs,
    kernel=constant_coefficients,
    moments=columns(v * dx),
)
```

The provider supplies `V`, its local measures, marked macrofaces, orientation
signs and literal constant coefficients. No physical form helper selects the
operator. `pymhm.backends.forms` compiles real linear and bilinear UFL forms;
linear trial-side argument number one is supported. The local pivot is square,
and domains use a single-rank communicator. Cross-mesh integration requires
explicit entity maps. These native restrictions are distinct from the more
fixed hybrid `LocalForm` adapter described in the [FEniCS page](../fenics.md).

Basix is a required runtime dependency, loaded when element or polynomial
operations are used. It supplies nodal bases, RT/BDM elements and entity
transformations; PyMHM supplies its declared coefficient order, physical maps,
moment restrictions and trace incidence. Native Basix RT degree `m+1` denotes
mathematical $\mathrm{RT}_m$; simplicial BDM degree `m` denotes
$\mathrm{BDM}_m$. Persisted coefficients also need their actual executed basis,
digest and orientation maps. See the [reference-element API](../api/elements.md).

## References

- Christopher Harder and Frédéric Valentin (2016). *Foundations of the MHM Method*, in *Building Bridges: Connections and Challenges in Modern Approaches to Numerical Partial Differential Equations*, Lecture Notes in Computational Science and Engineering 114, Springer. [DOI: 10.1007/978-3-319-41640-3_13](https://doi.org/10.1007/978-3-319-41640-3_13).

- Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *A family of Multiscale Hybrid-Mixed finite element methods for the Darcy equation with rough coefficients*, Journal of Computational Physics 245, 107–130. [DOI: 10.1016/j.jcp.2013.03.019](https://doi.org/10.1016/j.jcp.2013.03.019).

- Omar Durán, Philippe R. B. Devloo, Sônia M. Gomes, and Frédéric Valentin (2019). *A multiscale hybrid method for Darcy’s problems using mixed finite element local solvers*, Computer Methods in Applied Mechanics and Engineering 354, 213–244. [DOI: 10.1016/j.cma.2019.05.013](https://doi.org/10.1016/j.cma.2019.05.013).

- Martin S. Alnæs, Anders Logg, Kristian B. Ølgaard, Marie E. Rognes, and Garth N. Wells (2014). *Unified Form Language: A domain-specific language for weak formulations of partial differential equations*. ACM Transactions on Mathematical Software 40(2), article 9, 1–37. [DOI: 10.1145/2566630](https://doi.org/10.1145/2566630). [Author preprint: arXiv:1211.4047v2](https://arxiv.org/abs/1211.4047v2).
