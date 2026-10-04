# Local providers, forms and execution

This tutorial constructs the same Darcy problem with primal Galerkin and
mixed $H(\mathrm{div})$ local spaces through `LocalForm`, `GlobalForm` and
`HybridProblem`. A local provider is an ordinary callable; changing its
assembly backend does not require subclassing the hybrid solver. The runnable
example delegates physical assembly to the public `darcy_local_provider`.

```bash
pixi run --locked -e test python -m examples.tutorial_local_provider
pixi run --locked -e test python -m examples.tutorial_local_provider \
  --formulation mixed --boundary neumann --local-solver external \
  --backend process --workers 2 --batch-size 1
pixi run --locked -e fem python -m examples.tutorial_local_provider \
  --element-backend basix
```

The example uses two macrotriangles in the unit square and two subdivisions
per local edge. Its independently differentiated analytical fields are

$$
p=1+x^2+y^2,\qquad q=-\nabla p=-2(x,y),\qquad f=\nabla\cdot q=-4.
$$

The primal local space is continuous $P_2$ pressure. The mixed local space is
RT0 Darcy flux with $P_0$ pressure and auxiliary boundary-pressure multipliers.
Both use a $P_0$ normal-flux skeleton. The exact normal flux is constant on each
straight edge, so this skeletal choice represents its boundary data. Primal
pressure and flux are exact up to floating-point errors on this patch; mixed
pressure has a nonzero approximation error because the quadratic pressure
does not belong to $P_0$. This patch checks the declared construction; it is
not a literature reproduction or an inf-sup convergence proof.

## Declare the local and global contracts

`LocalForm` describes signed local forms in a literal coefficient basis:

$$
a(u,v)+\sum_j\lambda_j b_j(v)=L(v).
$$

Its `trace_forms` and `trace_dofs` have the same order. `kernel` contains the
actual nullspace coefficients; `moment_forms` specifies one physical moment
per retained mode. `coarse_basis` can instead retain modes that are not a
kernel. The compiler returns a validated `LocalProblem` without changing
these maps or rotating the declared basis.

```python
from functools import partial
from pymhm import (
    GlobalForm, HybridProblem, SkeletonSpace, TriangleMesh,
    assemble_hybrid, darcy_local_provider, solve_hybrid_system,
)
from examples.tutorial_local_provider import declared_local, exact_pressure, source
from pymhm.elements import boundary_data

mesh = TriangleMesh.unit_square()
skeleton = SkeletonSpace(mesh)
provider = darcy_local_provider(
    mesh, skeleton=skeleton, formulation="primal", degree=2,
    local_refinement=2, permeability=1.0, source=source,
    quadrature_order=4,
)
load, fixed = boundary_data(skeleton, exact_pressure, order=4)
form = GlobalForm(
    trace_size=skeleton.size,
    coarse_sizes=(1,) * len(mesh.cells),
    boundary_load=load,
    fixed_trace=fixed,
)
problem = HybridProblem(
    form, partial(declared_local, provider=provider), range(len(mesh.cells))
)
system = assemble_hybrid(problem)
solution = solve_hybrid_system(system, fixed=fixed)
```

The example's `declared_local` function exposes the provider's assembled
matrix, source vector, signed trace columns and pressure moments as a
`LocalForm`. Its small numeric compiler passes these coefficients to
`LocalProblem`; it does not reimplement Darcy integration. A UFL compiler
can consume the same record with different form expressions.

`GlobalForm` declares the trace size, retained-mode sizes, boundary load,
fixed trace coordinates and optional linear constraints. It represents the
supported algebraic hybrid form. It does not compile arbitrary global UFL
forms or infer their geometry. `HybridProblem` combines that record with a
provider and an ordered iterable of cell specifications. `LocalAssembly`
adds evaluation metadata without changing the local equations.

## Keep the mixed flux prescription explicit

For the mixed provider, the local coefficient order is flux $q_h$, pressure
$p_h$, then boundary-pressure multipliers $\eta_h$. Its original equations are

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

$M$ is the inverse-permeability RT0 mass matrix, $D$ is integrated divergence,
$S$ selects fine boundary normal-flux coefficients, and $F$ maps signed
skeletal flux density to those integrated coefficients. The joint pressure
kernel shifts $p_h$ and $\eta_h$ together. Physical pressure moments integrate
only $p_h$; they give zero weight to the auxiliary multipliers.

This is flux-prescribing mixed MHM. Its skeleton is physical normal Darcy
flux, and its auxiliary boundary variable is pressure. The mixed global
boundary load is `-load`, whereas the primal load is `load`, consistently
with their actual coupling signs. RT0 requires degree-zero skeleton segments
aligned with fine boundary edges. Here the legacy provider parameter
`degree=1` selects RT0; use the high-level mixed solvers for other RT/BDM
orders and enriched families.

The `--boundary neumann` option prescribes the compatible outward flux on
every exterior face and fixes the physical pressure integral to $5/3$:

```python
from pymhm import hybrid_mean_constraint

physical_weights = [record[1] for record in system.local_metadata]
gauge = hybrid_mean_constraint(system, physical_weights, value=5 / 3)
solution = solve_hybrid_system(
    system, fixed=fixed, constraints=[gauge]
)
```

Use this gauge with the Neumann problem built by `build_problem` or
`run_tutorial`, rather than adding it to the preceding Dirichlet example.
The returned metadata is `(fine_mesh, physical_pressure_mean_weights)` in
the executed local coefficient order. The tutorial reports separate physical
field errors and original algebraic row norms by field block. Its assembly
quadrature uses four Gauss points per Duffy coordinate; field errors use eight.
Primal volume flux is the raw gradient field $-\nabla p_h$; RT0 flux is
$H(\mathrm{div})$ conforming inside each local mesh. Macro conservation alone
does not imply fine-cell conservation for a general primal reconstruction.

## Replace the numerical local solver

`SolverConfig.local_solver` accepts a callable as well as a solver name.
The callback receives owned copies of the constrained sparse local operator
and every source, trace and retained-mode right-hand-side column. It returns
the complete response array, including constraint multipliers, in that basis.

```python
import numpy as np
from pymhm import SolverConfig

def dense_response(matrix, rhs):
    return np.linalg.solve(matrix.toarray(), rhs)

system = assemble_hybrid(
    problem, solvers=SolverConfig(local_solver=dense_response)
)
```

An external solver or response model can implement this contract. PyMHM
checks the original constrained operator's rank and accepts each returned
column only at the existing relative residual tolerance $10^{-10}$. This
check preserves the declared equations; it does not establish approximation
accuracy or a mesh-independent inf-sup constant. The callback releases its
native resources before returning arrays.

## Choose bounded execution independently

```python
from pymhm import ExecutionConfig

system = assemble_hybrid(
    problem,
    execution=ExecutionConfig(
        backend="process", workers=2, native_threads=1, batch_size=1
    ),
)
```

Serial execution constructs and contributes one cell before requesting the
next. Thread and process execution consume strictly bounded batches, return
results in cell order and accumulate trace entries in that same order. With
`batch_size=None`, the effective parallel worker count bounds the batch.
The standalone `iter_local` generator must be closed when iteration stops
early; `assemble_hybrid` owns that cleanup. Process execution uses spawn, so
providers, callbacks and items must be picklable and executable entry points
need the usual `if __name__ == "__main__"` guard. Construct optional native
objects inside the worker instead of serializing them. Closures remain
available for serial and thread execution.

## Compile UFL forms or reuse Basix elements

The optional native adapter accepts continuous primal or stable mixed volume
forms. For a primal local space `V`, the declaration has this structure:

```python
from pymhm import LocalForm
from pymhm.fenics import assemble_local_forms, primal_darcy_forms

a, L = primal_darcy_forms(V, permeability, source_expression)
forms = LocalForm(
    a=a, L=L, trace_forms=tuple(signed_trace_forms), trace_dofs=trace_dofs,
    kernel=constant_coefficients, moment_forms=(pressure_integral_form,),
)
local = assemble_local_forms(forms)
```

`V`, the signed trace forms, literal kernel coefficients and physical moment
form belong to the local provider. The fully runnable
`examples.variational_darcy` example supplies
that geometry and native assembly. `mixed_darcy_forms(W, K_inverse, f)`
defines the symmetric $H(\mathrm{div})\times L^2$ **volume** form and source;
the provider must still supply its boundary prescription, augmented variables,
trace coupling and compatible kernel. Choosing RT/DG volume spaces alone does
not construct flux-prescribing MHM.

Basix is the default element implementation for primal pressure and mixed RT0.
It preserves PyMHM's nodal order, physical derivatives and declared RT0 moments.
`--element-backend portable` is a compatibility spelling for the same library.
More general reference elements can be obtained
without importing a FEM framework:

```python
from pymhm import ReferenceElementSpec, create_reference_element, tabulate_reference

element = create_reference_element(ReferenceElementSpec("RT", "triangle", 1))
values = tabulate_reference(element, reference_points, nderiv=1)
```

Native Basix RT degree `m+1` represents mathematical $\mathrm{RT}_m$; native
simplex BDM degree `m` represents $\mathrm{BDM}_m$. Restricted/enriched
BDM$(k,n)$ is a separate construction. The reference provider exposes the
executed coefficient matrix and native entity transformations; physical Piola
maps and globally conforming orientation remain the consuming assembler's
responsibility. A persisted coefficient vector also needs its actual basis,
digest and orientation maps. Supported native versions are tested separately
from the declared PyMHM node order, following the
[Basix reference API](https://docs.fenicsproject.org/basix/v0.11.0/python/_autosummary/basix.finite_element.html).
