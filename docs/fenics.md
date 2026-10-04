# Defining local problems with FEniCSx

`pymhm.fenics.from_ufl` assembles local forms using DOLFINx and returns an
ordinary `LocalProblem`. Global skeleton numbering and condensation remain
independent of DOLFINx. The same adapter accepts scalar, vector, mixed and
H(div) spaces, provided the supplied local equations and constraints are
well posed.

Here FEniCSx means [DOLFINx](https://docs.fenicsproject.org/dolfinx/) assembly
of forms expressed in [UFL](https://docs.fenicsproject.org/ufl/). The
[Darcy reference comparison](https://github.com/volpatto/pymhm/blob/main/docs/cases/darcy-audit.md) records DOLFINx 0.9.0. The
examples assemble their local forms directly through UFL and DOLFINx.

Use the optional environment:

```bash
pixi run -e fem pytest -m fem
```

The adapter uses the native DOLFINx CSR/vector assembly interface and requires
DOLFINx 0.9 or 0.10. It does not need PETSc for assembly. Meshes must have a
single-rank communicator; `MPI.COMM_SELF` makes that choice explicit. Global
MPI-distributed assembly of a local mesh is not implemented by this adapter.
The assembled arrays can be condensed using the serial, thread or spawn-process
execution modes and any compatible linear-solver backend. GPU solving does not
move finite-element assembly to the GPU.

## Local form contract

The adapter implements

$$
A_K u_K+B_K\lambda_K=f_K.
$$

`a` defines \(A_K\), `load` defines \(f_K\), and every entry of `trace_forms`
defines one column of \(B_K\). `trace_dofs` maps those columns to the global
skeleton. A trace form must contain its orientation sign already. For primal
Darcy, a face basis \(\psi_j\) produces
\(B_{K,j}(v)=s_{K,F}\int_F\psi_jv\), where \(s_{K,F}\) relates the outward
normal of the local cell to the fixed global face normal.

`kernel` is an explicit coefficient array with one nullspace vector per column.
It is checked against both sides of the local matrix. `constraint_forms` defines
physical moments of the constrained local lifts. For scalar Neumann diffusion,
`kernel=np.ones((V.dofmap.index_map.size_local, 1))` and
`constraint_forms=[v * dx]` enforce zero integral. The default constraint uses
coefficient-space orthogonality, which generally differs from a zero physical
mean. Mixed-space kernels must follow DOLFINx's mixed coefficient ordering;
use collapsed-subspace maps when constructing them.

For small positive reaction or drag, use `coarse_basis=Z` in place of `kernel=Z`
to retain nearly null modes without declaring them exact null vectors. The same
`constraint_forms` defines one physical moment per retained column. For example,
scalar reaction–diffusion retains the constant coefficient vector with `[v * dx]`.
The adapter passes these choices to the exact
[coarse elimination](theory.md#retaining-nearly-null-local-modes); it does not
infer which modes are needed. Native tests compare a nonsymmetric UFL problem
near the Poisson limit with an independent uncondensed saddle solve.

The following assembles a local scalar Neumann response with four separately
marked boundary traces. The local field remains free on its boundary.

```python
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from pymhm.fenics import from_ufl, primal_darcy_forms

local_mesh = mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
V = fem.functionspace(local_mesh, ("Lagrange", 1))
v = ufl.TestFunction(V)
a, load = primal_darcy_forms(V, permeability=1.0, source=0.0)

predicates = [
    lambda x: np.isclose(x[0], 0.0),
    lambda x: np.isclose(x[0], 1.0),
    lambda x: np.isclose(x[1], 0.0),
    lambda x: np.isclose(x[1], 1.0),
]
facets, values = [], []
for marker, predicate in enumerate(predicates, start=1):
    found = mesh.locate_entities_boundary(local_mesh, 1, predicate)
    facets.extend(found)
    values.extend([marker] * len(found))
order = np.argsort(facets)
tags = mesh.meshtags(
    local_mesh, 1,
    np.asarray(facets, dtype=np.int32)[order],
    np.asarray(values, dtype=np.int32)[order],
)
ds = ufl.Measure("ds", domain=local_mesh, subdomain_data=tags)

# These signs assume every global face normal is locally outward.
# Interior macrofaces must use opposite signs in their two adjacent cells.
trace_forms = [v * ds(marker) for marker in range(1, 5)]
local = from_ufl(
    a, load, trace_forms, np.arange(4),
    kernel=np.ones((V.dofmap.index_map.size_local, 1)),
    constraint_forms=[v * ufl.dx],
)
response = local.condense()
```

The integration test `test_two_subdomain_darcy_affine_reconstruction` provides a
complete two-macrotriangle problem: it creates independent fine meshes, assembles
signed interface traces, compares local matrices and moments with the native P1
implementation, solves the common skeleton system, and recovers the exact affine
pressure and globally oriented flux.

## Volume-form helpers

| Helper | Unknowns and equation | Boundary/nullspace responsibility |
|---|---|---|
| `primal_darcy_forms` | Pressure, \(-\nabla\cdot(K\nabla p)=f\) | Normal Darcy flux multiplies the pressure test; one constant local mode for pure diffusion. |
| `mixed_darcy_forms` | H(div) velocity and L2 pressure, \(K^{-1}u+\nabla p=0\), \(\nabla\cdot u=f\) | Pressure is natural data, normal velocity is essential; RT/DG stability and essential constraints must be supplied. |
| `brinkman_forms` | H1 velocity and pressure, \(-\nabla\cdot(2\mu\varepsilon(u))+Ru+\nabla p=f\) | Select a stable pair and supply traction, the actual local kernel and any required global pressure gauge. Rigid motions form the kernel in the zero-drag Stokes limit; positive-definite drag removes them. |
| `elasticity_forms` | Displacement, stress \(2\mu\varepsilon(u)+\lambda\operatorname{div}(u)I\) | Supply rigid motions and signed traction forms. The 2D default interpretation is plane strain. |
| `usfem_brinkman_forms` | Equal-order velocity/pressure with vector-Laplacian pseudostress and residual stabilization | Use compatible pseudotractions, the stated inverse estimate and the appropriate global pressure constraint. |

The mixed Darcy and incompressible helpers negate the pressure test equation to
produce a symmetric saddle operator. Accordingly, a divergence source \(g\)
contributes \(-\int gq\). For mixed Darcy, imposed pressure contributes
\(-\int_{\partial K}p_D(v\cdot n_K)\). The native integration test checks this
sign by recovering constant pressure and zero RT velocity.

The adapter does not infer essential normal-velocity constraints for H(div)
spaces. An arbitrary mixed volume form plus a boundary pressure coupling is not
a replacement for the flux-multiplier MHM construction: choose the intended
hybrid formulation and enforce its essential conditions through elimination or
an appropriate augmented local system.

## USFEM Stokes–Brinkman formulation

`usfem_brinkman_forms` implements equations (41)–(42) of the 2017 construction
with the pressure test sign changed consistently. Define

$$
R(u,p)=-\nu\Delta u+\Theta u+\nabla p.
$$

The bilinear form and source are

$$
\begin{aligned}
B((u,p),(v,q))={}&(\nu\nabla u,\nabla v)+(\Theta u,v)
\\
&-(p,\nabla\cdot v)-(q,\nabla\cdot u)\\
&-\sum_\tau\kappa_\tau(R(u,p),R(v,q))_\tau,\\
L(v,q)={}&(f,v)-(g,q)-\sum_\tau\kappa_\tau(f,R(v,q))_\tau.
\end{aligned}
$$

The stabilization has a **negative** sign. With
\(m_\tau=\min(1/3,C_\tau)\), the implementation uses the equivalent expression

$$
\kappa_\tau=
\frac{h_\tau^2}{\max(\gamma_{\min}h_\tau^2,4\nu/m_\tau)+4\nu/m_\tau}.
$$

Here \(\gamma_{\min}\) is the smallest eigenvalue of the resistance tensor.
For Stokes and a P1 pair this reduces to \(h_\tau^2/(24\nu)\); an integration
test checks the assembled pressure block against that exact coefficient.
Viscosity must be constant per cell, and a higher-order pair requires a justified
inverse estimate. Both viscosity and resistance cannot vanish simultaneously.
The helper does not validate coercivity of user-provided coefficient fields.

This form uses \(\nu\nabla u:\nabla v\), so its natural boundary quantity is
pseudostress. The physical symmetric-stress form in `brinkman_forms` has different
boundary terms and, in the Stokes limit, different rigid-motion modes. They must
not be interchanged while keeping the same boundary and kernel definitions.
See [the published construction](https://doi.org/10.1016/j.cma.2017.05.027) and
[the literature map](literature.md).
The [high-order reproduction](https://github.com/volpatto/pymhm/blob/main/docs/cases/reproduction.md#stokes-2017-equal-order-local-spaces-and-direct-figure-comparison)
used P2/P2 and P3/P3 local spaces through DOLFINx with a computed inverse
estimate, then compared their numerical errors with digitized article curves.
These calculations use pyMHM with DOLFINx local assembly; they do not execute
the authors' Stokes program.

## Validation boundary

Core tests exercise real UFL expression construction and explicitly simulated
DOLFINx API contracts. The `fem` test suite separately executes actual DOLFINx
assembly: scalar MHM affine reconstruction, RT/DG pressure-boundary signs,
Taylor–Hood and elasticity symmetry, and the signed USFEM pressure block.
An additional comparison maps DOLFINx mixed DOFs to the portable ordering and
checks the complete USFEM matrices and loads for scalar drag 0, 3 and 1000,
including both branches of the stabilization parameter and the drag cross-terms.
These checks establish the implemented local form and coupling contracts. They
do not establish all stability estimates or reproduce every benchmark in the
literature. DOLFINx's assembly APIs are documented in the
[official finite-element reference](https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html).
