# Writing local equations with UFL and FEniCSx

Use UFL to express the local weak form, and DOLFINx to compile it on the local
mesh. The bound problem API supplies coefficient maps and geometric orientation;
your provider supplies the operator, the two interface pairings, retained modes
and physical moments. Solver and execution settings are independent of this form.

Install the native stack following [installation](installation.md#native-ufl-assembly).
In a checkout, the locked `fem` profile supplies DOLFINx, UFL, Basix, MPI and
compiler dependencies. The adapter supports DOLFINx 0.9 and 0.10; it assembles
native CSR matrices and vectors without requiring PETSc. Each local mesh uses
a single-rank communicator such as `MPI.COMM_SELF`.

## Write the operator and pairings explicitly

The following provider fragment declares a scalar Neumann diffusion local
problem. Its load is zero for this configuration example:

$$
\begin{aligned}
a_K(u_K,v)+b_K(\lambda,v)&=0,\\
a_K(u,v)&=(\nabla u,\nabla v)_K,\\
b_K(\lambda,v)&=\langle\lambda_K,v\rangle_{\partial K},\\
c_K(u,\mu)&=\langle u,\mu_K\rangle_{\partial K}.
\end{aligned}
$$

```python
import numpy as np
import ufl
from pymhm import LocalContext, LocalEquations, columns


def local_equations(local: LocalContext) -> LocalEquations:
    """Declare scalar diffusion in the local native coefficient basis."""
    binding = local.native_space(degree=1)
    u, v = ufl.TrialFunction(binding.space), ufl.TestFunction(binding.space)
    dx = ufl.Measure("dx", domain=binding.mesh, metadata={"quadrature_degree": 4})
    local.field("pressure", binding)
    return local.equations(
        a=ufl.inner(ufl.grad(u), ufl.grad(v)) * dx,
        L=0.0 * v * dx,
        b=local.trace_pairings(lambda phi, ds: phi * v * ds),
        c=local.trace_pairings(lambda phi, ds: phi * u * ds, axis="rows"),
        kernel=np.ones((binding.size, 1)),
        moments=columns(v * dx),
    )
```

The scalar P1 nodal constant is the exact local kernel. Its physical moment
sets the constrained response's integral to zero, while its retained amplitude
belongs to the global unknowns. For the normal-flux convention, the interface
binding transports canonical face coordinates to outward local normals once.
The sign of `c` is still a mathematical choice: the global boundary load must
follow the same chosen pairing.

The global equation for this chosen pairing is

$$
\sum_K c_K(u_K,\mu)=0
$$

for all skeleton test functions, with zero prescribed exterior pressure.
Interior oriented pairings express pressure continuity; exterior pairings impose
that pressure data weakly. Bind the callback to your hierarchy and interface:

```python
from pymhm import Equation, bind_problem

problem = bind_problem(
    hierarchy,
    interface,
    local_equations,
    global_equation=Equation(0, 0),
    retained=1,
)
```

This zero additional global equation is appropriate when the local `c` pairings
supply the balance and exterior pressure moments are zero. Supply actual
boundary moments and any additional global forms for your problem.
The [API overview](tutorials/overview.md) constructs every mesh, interface,
boundary condition and global equation in a complete worked workflow.

## Change the element or coefficient

`local.native_space(degree=k)` creates the supported equispaced Lagrange space.
For vector fields, pass `shape=(dimension,)`. To choose another element, pass a
Basix/UFL element, including a mixed element, to `local.native_space(element)`.
The adapter assembles any admissible native space; portable replay and field
conversion have their own restrictions below.

For heterogeneous scalar or tensor material, define `K` on `binding.mesh` and
use `ufl.inner(K * ufl.grad(u), ufl.grad(v)) * dx`. Scalar multiplication and
tensor contraction use the same expression. Follow
[heterogeneous coefficients](guides/heterogeneous-darcy.md) for material grids,
physical coordinates and integration across jumps.

A positive reaction removes the constant from the exact nullspace. To retain it
as a nearly null mode, pass `coarse_basis=constant_coefficients` instead of
`kernel=constant_coefficients`, with the corresponding physical moment. Exact
kernels, near-null modes and their test spaces remain explicit formulation data;
the compiler does not infer them from the PDE name.

## Register fields before returning equations

`local.field("pressure", binding)` snapshots the executed basis and mesh when
`local.equations` returns. After solving, obtain `solution.field("pressure")`.
Its views follow the hierarchy's selected item order. Evaluation supports
physical points and explicit one-sided fine-cell owners:

```python
pressure = solution.field("pressure")
values = pressure[local_index].evaluate(points, cells=fine_cell_owners)
gradients = pressure[local_index].gradient(points, cells=fine_cell_owners)
```

Scalar gradients have axes `(point, derivative)`; vector gradients have axes
`(point, component, derivative)`. Darcy flux is a separately defined physical
field $-K\nabla p$; a raw gradient is not an H(div) reconstruction. For a mixed
space, register supported nodal components using `component=...` or declare an
explicit evaluator for a moment-based component.

Native geometry bindings accept triangles, tetrahedra, Cartesian quadrilaterals
and hexahedra. Automatic nodal coefficient maps and named-field descriptors
cover supported equispaced Lagrange fields on triangles, tetrahedra and
quadrilaterals. Hexahedral assembly and direct native evaluation remain
available; hexahedral portable descriptors require a custom evaluator and basis
contract. H(div)/H(curl) coefficients are native moments, not nodal values.

Built-in boundary UFL adapters support the declared polynomial edge and
triangular-face spaces. Trace partition interfaces must resolve the supported
boundary facets. Custom spaces can provide `trace_pairings(...)` and
`interface_pairing(...)`, or supply explicit assembled blocks. See
[custom interfaces](guides/custom-interface.md) and
[compilation limits](variational.md#native-compilation-limits).

## Integrate an existing assembled or native formulation

Use `LocalEquations` directly when you already own the coefficient ordering and
trace maps. `columns` and `rows` express independent rectangular pairings;
`pymhm.backends.forms.assemble_form` compiles real linear and bilinear UFL forms.
Cross-mesh forms require explicit `entity_maps`: DOLFINx 0.9 uses mesh-to-index
mappings, while 0.10 uses native `EntityMap` sequences.

The lower-level `LocalForm` adapter supplies a fixed hybrid construction:
`assemble_local_forms(LocalForm(...))` returns a `LocalProblem`. In that manual
path, `trace_forms` already include physical signs and geometric orientation,
`trace_dofs` declares global numbering, and `moment_forms` supplies constraints.
The bound context path derives geometric maps instead. Already globally
oriented blocks must use `local.equations(coordinates="global", ...)` to avoid
applying that map twice.

For an external simulation library or learned response, use the focused
[provider guide](guides/providers.md). The library's native convention, numerical
basis and physical boundary conditions remain part of the adapter contract.

## Keep native resources in their execution owner

Create meshes, spaces and forms inside the provider invocation or a worker-owned
workspace. No live DOLFINx mesh, PETSc factor or communicator crosses a spawn
boundary. Use importable callbacks and worker cleanup as described in
[CPU execution](guides/cpu.md). UFL compilation caches can reuse the same form
kernel across cells; distinct coefficient arrays still define distinct matrices.

Select algebra independently:

```python
from pymhm import SolverConfig, assemble

system = assemble(
    problem,
    solvers=SolverConfig(local_solver="scipy", global_solver="scipy"),
)
solution = system.solve()
```

SciPy is portable; PARDISO, PETSc/MUMPS and CUDA require their corresponding
runtimes and admissible operators. [Linear solvers](solvers.md) describes these
choices. [Windows FEM requirements](windows.md#native-fem-scope) explains its
MPI and native C compiler setup.

## References

- Martin S. Alnæs, Anders Logg, Kristian B. Ølgaard, Marie E. Rognes and
  Garth N. Wells (2014). *Unified Form Language: A domain-specific language
  for weak formulations of partial differential equations*.
  [DOI: 10.1145/2566630](https://doi.org/10.1145/2566630).
- [DOLFINx documentation](https://docs.fenicsproject.org/dolfinx/): native
  spaces, form assembly and mesh/entity maps.
- [Basix documentation](https://docs.fenicsproject.org/basix/): elements,
  interpolation and entity transformations.
