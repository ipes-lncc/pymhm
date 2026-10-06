# Overview: write the local and global variational problems

The usual PyMHM workflow is **meshes → spaces → local equations → global
balance → assembly → solution → physical fields and errors**. You choose the
mathematical formulation. PyMHM derives the associated numbering and geometric
coordinate maps, eliminates local unknowns and assembles shared interface
contributions in a deterministic order.

This tutorial writes a complete two-dimensional problem with UFL. It uses
ordinary functions for both equations, without a named physical-model
constructor. The [introductory course](../tutorials.md) develops the same steps
for heterogeneous Darcy, elasticity, MsHHO, MH²M, reaction–diffusion and
Stokes–Brinkman, including fields, classical references and convergence studies.

The local/global decomposition follows
[Harder and Valentin (2016)](https://doi.org/10.1007/978-3-319-41640-3_13).
The analytical problem here checks the selected discretization; it is not a
reproduction of a published application.

## Install and select the backend

```bash
python -m pip install pymhm
```

The mesh-associated binding API used in this course is available in the current
source checkout and will be included in the next release. The published PyPI
1.0.0 provides the explicit `Equation`/`MultiscaleProblem` interface. Use the
[source installation](../installation.md#from-a-checkout) for this workflow.

The portable package works independently of DOLFINx, PETSc, MPI and GPU
libraries. This UFL example uses the optional native DOLFINx backend. In a
checkout, run it in the locked `introduction` environment; the
[installation guide](../installation.md) describes available backends and
native Windows support.

## 1. Choose the macro mesh, local meshes and interface space

A macro mesh controls the global problem. Each macroelement has its own fine
mesh for the local response. Refining a local mesh changes its approximation
without creating more macroelements.

```python
import numpy as np
import ufl
import basix
import basix.ufl

from pymhm import (
    CartesianMacroMesh, Equation, FaceSpace, LocalContext, MeshHierarchy,
    SkeletonSpace, assemble, bind_interface, bind_problem, columns, solve,
)

macro = CartesianMacroMesh(2, 2)
hierarchy = MeshHierarchy(
    macro,
    tuple(macro.submesh(cell, 4) for cell in range(len(macro.cells))),
)
skeleton = SkeletonSpace(
    macro, tuple(FaceSpace.uniform(1) for _ in macro.faces)
)
interface = bind_interface(skeleton, convention="normal")
local_element = basix.ufl.element(
    "Lagrange", "quadrilateral", 2,
    lagrange_variant=basix.LagrangeVariant.equispaced,
)
```

`MeshHierarchy` associates macroelements and local meshes. For large problems,
its second argument can instead be an importable `cell -> mesh` callable;
workers then create their own meshes as needed. `SkeletonSpace` declares the
face approximation. `bind_interface` supplies local/global coefficient maps
and outward-normal incidence signs from the existing topology.

The interface variable below represents Darcy flux in the canonical face
normal. That physical choice is explicit. The library's geometric sign
conversion does not choose a sign in a variational form.

## 2. Translate the local mathematical problem into UFL

Take identity permeability and the exact pressure
$p_*(x,y)=x^2+2y^2$. Independently differentiating gives the volume source
$f=-6$ and physical flux $q_*=-\nabla p_*=(-2x,-4y)$. Prescribe $p_*$ on the
exterior boundary.

On each macroelement, write

$$
\begin{aligned}
(\nabla p_K,\nabla v)_K+\langle\lambda_K,v\rangle_{\partial K}
&=(-6,v)_K,\\
-\langle p_K,\mu_K\rangle_{\partial K}
&=g_K(\mu_K).
\end{aligned}
$$

The first line is the local volume equation. The second contributes to the
global balance. The normal signs in $\lambda_K$ come from the interface
binding. The minus sign in the second line belongs to the formulation and is
written in the callback.

```python
def local_equations(local: LocalContext):
    """Write the volume operator and both interface pairings independently."""
    space = local.native_space(local_element)
    p, v = ufl.TrialFunction(space.space), ufl.TestFunction(space.space)
    dx = ufl.Measure("dx", domain=space.mesh,
                     metadata={"quadrature_degree": 8})

    a = ufl.inner(ufl.grad(p), ufl.grad(v)) * dx
    L = -6.0 * v * dx
    b = local.trace_pairings(lambda phi, ds: phi * v * ds)
    c = local.trace_pairings(lambda phi, ds: -phi * p * ds, axis="rows")

    constant = np.ones((space.size, 1))
    local.field("pressure", space)
    return local.equations(
        a=a, L=L, b=b, c=c,
        kernel=constant, moments=columns(v * dx),
    )
```

`native_space` binds the chosen element to the local geometry and owns its
native coefficient convention. `trace_pairings` represents each declared
trace basis function on its integration support. `equations` applies the
local/global maps once. The provider needs neither face DOF indices nor
native/portable permutations.

The Neumann volume operator has a constant kernel. Its explicit `kernel` and
physical integral `moments` select the local complement. We retain one
pressure-mean coordinate per macroelement. PyMHM checks these declared data;
it does not infer kernels or establish inf-sup stability from field names.

## 3. Declare the global equation and boundary data

The global balance imposes exterior pressure weakly:

$$
-\sum_K\langle p_K,\mu_K\rangle_{\partial K}
=-\langle p_*,\mu\rangle_{\partial\Omega}.
$$

The local `c` blocks supply the left-hand side. The additional global
`Equation` supplies the boundary load:

```python
def exact_pressure(points: np.ndarray) -> np.ndarray:
    """Evaluate the independently declared analytical pressure."""
    return points[:, 0] ** 2 + 2 * points[:, 1] ** 2


def global_equation(global_problem):
    """Write exterior pressure moments in the interface's declared coordinates."""
    boundary, fixed = global_problem.boundary_data(exact_pressure, order=6)
    assert not fixed  # This formulation imposes pressure weakly.
    return Equation(0, global_problem.trace_load(-boundary))


problem = bind_problem(
    hierarchy, interface, local_equations,
    global_equation=global_equation, retained=1,
)
```

`bind_problem` derives the global interface size, local maps and retained
coordinate offsets. `retained=1` declares the mathematical number of retained
modes; it is not a request for automatic nullspace discovery. Prescribed
Dirichlet pressure removes the global constant-pressure ambiguity in this
example. Other formulations must declare their boundary and physical gauge.

## 4. Assemble and solve

```python
system = assemble(problem)
solution = solve(system)
print("Global unknowns:", system.matrix.shape[0])
print("Original-equation residual:", solution.raw_residual)
```

These free functions reuse the shared condensation, ordered contribution
reduction and linear solver owners. A small residual accompanies the field
check; it does not replace that check or establish uniqueness for an arbitrary
formulation.

## 5. Evaluate physical fields without reconstructing index maps

Register each `local.field(...)` before returning `local.equations(...)`: the
equation snapshots the registered field definitions. The solution then carries
each field's mesh, executed basis and coefficient map:

```python
pressure = solution.field("pressure")
for cell, field in enumerate(pressure):
    point = macro.points[macro.cells[cell]].mean(axis=0, keepdims=True)
    value, gradient = field.values_and_gradient(point)
    np.testing.assert_allclose(value, exact_pressure(point), rtol=1e-8, atol=1e-8)
    exact_gradient = np.column_stack((2 * point[:, 0], 4 * point[:, 1]))
    np.testing.assert_allclose(gradient, exact_gradient, rtol=1e-8, atol=1e-8)
```

`field.gradient(points)` returns raw physical derivatives;
`field.values_and_gradient(points)` obtains both in one sampling pass. Scalar
gradient axes are `(point, derivative)` and vector gradient axes are
`(point, component, derivative)`. `field.portable_coefficients` supplies the
owned nodal ordering for these supported Lagrange fields while preserving
the executed basis; it does not replace an archived basis by a new one.

Tuple entries follow the selected-item order of `MeshHierarchy.items`; each
entry belongs to one macroelement. For discontinuous fields,
explicit fine-cell owners can be supplied through `field.evaluate(points,
cells=...)`; independent one-sided values are retained. `field.coefficients`
and `field.basis_digest` expose the executed representation for advanced
postprocessing and reproducibility.

`solution.local_trace(index)` reads the selected global coefficients in the
executed local interface basis. `index` is the deterministic assembled item
position, matching the field tuple; `test=True` selects independent test maps.
The normal convention includes outward-normal incidence signs, while a value
convention does not. Saved fields require the executed basis and maps when
replayed; a bare coefficient vector does not identify its representation.

The physical tutorials show field plots, physical norms, conservation checks
and classical reference refinement. A normal interface multiplier is not
automatically a pointwise physical flux or an H(div) reconstruction.

## 6. Select execution and solvers without changing the equations

```python
from pymhm import ExecutionConfig, SolverConfig

system = assemble(
    problem,
    execution=ExecutionConfig("thread", workers=4, batch_size=4, native_threads=1),
    solvers=SolverConfig(local_solver="scipy", global_solver="scipy"),
)
solution = solve(system)
```

Local workers return independent contributions. The coordinator accumulates
shared faces in macroelement order. Process execution additionally requires
importable, picklable providers and mesh factories; native resources are
created and released in their owning workers. The
[process tutorial](introduction/darcy_process_scalability.md) demonstrates
that workflow. A solver or learned response provider can use the same
[local contracts](providers.md), with their physical acceptance checks.

## More general formulations use the same contracts

- **Vector and mixed fields:** declare a vector or mixed Basix/UFL element and
  write its trial/test forms. The
  [elasticity](introduction/multiscale_elasticity.md) and
  [Brinkman](introduction/stokes_brinkman_boundary_layer.md) tutorials declare
  rigid modes, velocity–pressure spaces and physical gauges explicitly.
- **Independent pairings:** `b`, `c`, `d` and `g` are independent. Petrov–Galerkin,
  Robin and nonsymmetric systems do not require an inferred transpose.
  A formulation can also declare an independent local boundary space through
  `local.trace_pairings(expression, interface=other_interface)`.
  `local.interface_pairing(other_interface)` supplies the unsigned boundary
  mass matrix against the primary interface, accumulating shared vertices and
  integrating unequal face partitions. The [MH²M tutorial](introduction/mh2m_multiscale.md)
  uses these operations for its private conormal and shared pressure trace.
- **Additional global UFL forms:** `global_problem.interface_equation(builder)`
  binds supported planar face spaces. For example, a user-selected term
  $\alpha\int_\Sigma\lambda\mu$ can be written as
  `builder=lambda lam, mu, dx: Equation(alpha * lam * mu * dx, 0 * mu * dx)`.
  That term changes the formulation; the package does not add it implicitly.
- **Several scales:** a local provider can bind a child problem using the same
  abstractions and return `local.nested(child_problem)`. Automatic restrictions
  currently require compatible bound planar normal spaces. Arbitrary
  restrictions remain explicit, and the documented recursive/MPI and child
  boundary limitations still apply.
- **Custom interfaces and full manual control:** provide your own
  `InterfaceSpace` and `TraceBinding`, or use `LocalEquations` and
  `MultiscaleProblem` directly. The
  [custom-space tutorial](custom-interface.md) demonstrates dense basis changes,
  arbitrary numbering and explicitly declared orientations while preserving
  physical fields.

## Supported bindings and explicit limits

The native mesh binding supports existing triangles, tetrahedra, Cartesian
quadrilaterals and `HexMesh` hexahedra. Native assembly accepts user-selected
supported elements. Automatic portable coefficient maps and named-field
descriptors currently cover supported equispaced nodal Lagrange fields and mixed
components on triangles, tetrahedra and quadrilaterals. Hexahedral native assembly
and direct native evaluation are available, but automatic portable hex nodal
maps and `local.field(name, hex_space)` descriptors are not; declare a custom
evaluator and its executed basis for a named hexahedral field. Moment-based
H(div)/H(curl) fields keep their native coefficient conventions unless an
adapter declares their evaluator and basis contract.

Automatic native local trace pairings currently cover planar scalar/vector
piecewise polynomial `SkeletonSpace` bases whose breakpoints align with local
boundary facets. Automatic global interface UFL forms use supported planar
face spaces. Custom and three-dimensional interfaces can supply their own
capabilities or assembled blocks. Arbitrary unrelated cross-mesh UFL forms,
complex native blocks, nonlinear solvers and unrestricted recursive MPI
execution are not implied by these bindings. See the
[variational guide](../variational.md) for exact contracts.
