# MH²M with multiscale permeability: pressure, conormal and trace

Follow the numbered steps: state the variational problem, choose the local and trace spaces, declare the local and global equations, solve, and inspect the physical fields.

The main workflow is **meshes → spaces → local equations → global balance → assemble → solve → fields and errors**. `MeshHierarchy` associates macro and local meshes; `bind_interface` and `LocalContext` own supported numbering, geometric orientation and native coordinate conversions. The mathematical forms, physical trace meaning, local modes and gauge remain explicit in the cells below. Fully manual/custom spaces use the same numerical owners; see the [custom-interface notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/custom_interface.ipynb).

The **Multiscale-Hybrid-Hybrid Method** uses three fields: local pressure, local boundary conormal and a shared global pressure trace. We declare all three equations explicitly and let `LocalEquations` and `MultiscaleProblem` eliminate the local fields.

Our two-dimensional P1/P0/P1 family follows [de Barros, Madureira and Valentin (2026, version 3)](https://arxiv.org/abs/2404.16978v3), equations (7), (28)–(29) and section 6.2. The oscillatory material and meshes below define an original introductory case, not a reproduction of the article's heterogeneous figures.

All problem data, boundary conditions, provider, classical baseline, reconstruction, norms and plots are defined in notebook cells. PyMHM supplies only generic finite-element and algebraic operations. Execute with the checked-in lockfile:

```bash
pixi run --locked -e introduction python scripts/run_notebooks.py notebooks/introduction/mh2m_multiscale.ipynb
```

Execution is serial; process workers need importable provider callables.



```python

from pathlib import Path
import sys

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pixi.toml").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.collections import LineCollection
from scipy import sparse
from threadpoolctl import threadpool_limits

from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.multiscale import MultiscaleProblem, assemble
from pymhm.fem.scalar.triangle import (
    nodal_space, reference_basis, tabulate,
)
from pymhm.fem.scalar.operators import p1_geometry, triangle_quadrature
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.linalg.linear import solve_linear
from pymhm.meshes.triangle import TriangleMesh

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

from pymhm.methods.three_field import PressureTraceSpace

from pymhm import MeshHierarchy, LocalContext, bind_interface, bind_problem, solve
from pymhm.backends.spaces import bind_space
from pymhm.backends.forms import assemble_pairing

```

### 1. The physical problem and the conormal sign

Use the same scalar Darcy problem as the MsHHO introduction:



$$
\begin{aligned}
 -\nabla\cdot(a_\varepsilon\nabla p)&=1 &&\text{in }(0,1)^2,\\
 p&=0 &&\text{on its boundary},\\
 a_\varepsilon(x,y)&=2+\sin(2\pi x/\varepsilon)
                           \sin(2\pi y/\varepsilon),
 &\varepsilon&=1/8,\\
 q&=-a_\varepsilon\nabla p,
 &\eta_K&=a_\varepsilon\nabla p\cdot n_K=-q\cdot n_K.
\end{aligned}
$$



The article's **conormal** $\eta_K$ has the opposite sign to physical outward flux. Each macrocell has its private copy on a shared face; pressure trace $\rho$ is global. The coefficient has ellipticity bounds 1 and 3.

Our 32 macrotriangles do not resolve eight oscillations per direction. Each local mesh has 16 subdivisions per macroedge, keeping the material inside assembly and physical flux norms. This is fine local resolution on a simple macro mesh, not replacement of the coefficient by its average.



```python
epsilon = 1 / 8

def permeability(points: np.ndarray) -> np.ndarray:
    """Evaluate aε I through its positive scalar factor, with ε=1/8."""
    phase = 2 * np.pi * points / epsilon
    return 2 + np.sin(phase[:, 0]) * np.sin(phase[:, 1])

macro = TriangleMesh.unit_square(4)
source = 1.0
local_refinement = 16
assembly_order = 8
print({"macro_triangles": len(macro.cells), "epsilon": epsilon,
       "H_max": float(macro.lengths.max()), "h_max": float(macro.lengths.max() / local_refinement),
       "ellipticity_bounds": [1.0, 3.0], "source": source})

```

```text
{'macro_triangles': 32, 'epsilon': 0.125, 'H_max': 0.3535533905932738, 'h_max': 0.02209708691207961, 'ellipticity_bounds': [1.0, 3.0], 'source': 1.0}
```


```python
preview = TriangleMesh.unit_square(64)
fig_material, axis = plt.subplots(figsize=(5.4, 4.2), layout="constrained")
artist = axis.tripcolor(Triangulation(*preview.points.T, preview.cells),
                       permeability(preview.points), shading="gouraud", vmin=1, vmax=3)
axis.add_collection(LineCollection(macro.points[macro.faces], colors="white", linewidths=1.1))
axis.add_collection(LineCollection(macro.points[macro.faces], colors="#24343c", linewidths=.45))
axis.set(title="Multiscale permeability and actual macro mesh", xlabel="x", ylabel="y", aspect="equal")
fig_material.colorbar(artist, ax=axis, label="Permeability aε")
plt.show()

```



[![Figure 1 — MH²M with multiscale permeability: pressure, conormal and trace](../../assets/tutorials/mh2m_multiscale/figure_4_0.png)](../../assets/tutorials/mh2m_multiscale/figure_4_0.png)


### 2. Choose compatible spaces before assembling operators

Use $k=0$ in the two-dimensional family of section 6.2:



$$
\begin{aligned}
 \Gamma_{H_\Gamma}&=\text{continuous P1 pressure trace},\\
 \Lambda_{H_\Lambda}(K)&=\text{discontinuous P0 on }\partial K,\\
 V_h(K)&=\text{continuous P1 on the local triangulation},\\
 H_\Gamma&\sim H/4,
 &H_\Lambda&\sim H/8,
 &h&\sim H/16.
\end{aligned}
$$



Each pressure segment contains **two** conormal segments (M2), and each conormal segment contains **two** local boundary edges (M1). The uniform partitions match between incident macrocells. These mesh conditions accompany the chosen degrees; matrix dimensions alone do not establish compatibility.

`PressureTraceSpace` shares pressure values at macrovertices and numbers interior face nodes. `SkeletonSpace` supplies the broken conormal basis, with a private copy per macrocell. Only `gamma` coordinates are global.



```python
local_degree = 1
gamma_segments, conormal_segments = 4, 8
gamma = PressureTraceSpace.uniform(macro, degree=1, segments=gamma_segments)
conormal = SkeletonSpace(
    macro, tuple(FaceSpace.uniform(0, conormal_segments) for _ in macro.faces),
)
assert conormal_segments == 2 * gamma_segments
assert local_refinement == 2 * conormal_segments
pressure_interface = bind_interface(gamma, convention="value")
conormal_interface = bind_interface(conormal, convention="value")
print({"global_pressure_trace_coordinates": gamma.size,
       "local_conormal_unknowns_per_macrocell": 3 * conormal_segments,
       "local_P1_nodes_per_macrocell": (local_refinement + 1) * (local_refinement + 2) // 2})

```

```text
{'global_pressure_trace_coordinates': 193, 'local_conormal_unknowns_per_macrocell': 24, 'local_P1_nodes_per_macrocell': 153}
```

### 3. Write the three variational equations

For each macrocell $K$,



$$
\begin{aligned}
 (a_\varepsilon\nabla p_K,\nabla v)_K
  -\langle\eta_K,v\rangle_{\partial K}&=(1,v)_K,\\
 -\langle\mu,p_K\rangle_{\partial K}
  +\langle\mu,\rho\rangle_{\partial K}&=0,\\
 \sum_K\langle\eta_K,\xi\rangle_{\partial K}&=0.
\end{aligned}
$$



The second equation matches **trace moments** against P0; it does not impose pointwise equality. The third equation balances conormal against continuous pressure trace tests.

With $A_{ij}=a_K(\phi_j,\phi_i)$, $B_{ij}=\int_{\partial K}\phi_i\mu_j$ and $P_{j\ell}=\int_{\partial K}\mu_j\psi_\ell$,



$$
\begin{aligned}
 \begin{bmatrix}A&-B\\-B^T&0\end{bmatrix}
 \begin{bmatrix}p_K\\\eta_K\end{bmatrix}
 +\begin{bmatrix}0\\P\end{bmatrix}\rho_K
 &=\begin{bmatrix}f_K\\0\end{bmatrix},\\
 \sum_K P^T\eta_K&=0.
\end{aligned}
$$



Write the boundary pairing $B$ as `phi * v * ds` in UFL, selecting the private conormal space as its interface. `local.interface_pairing(conormal_interface)` integrates $P$ on the union of the pressure and conormal partitions and accumulates shared pressure vertices. Both are unsigned local boundary pairings: their mathematical signs are the explicit **$-B$ and $-B^T$** in the saddle. The context owns face ordering and coefficient transport.


### Write the local differential operator in UFL

Define the trial pressure `p` and test pressure `v` on the local Pk space. The UFL expressions below are the executable weak form, mass pairing and source. Changing this expression defines a new local operator; the multiscale contracts do not select a physical model.



$$
a_K(p,v)=\int_K a_\varepsilon\nabla p\cdot\nabla v,
\qquad \ell_K(v)=\int_K v.
$$



`local.native_space` binds the declared fine mesh and equispaced element. The local saddle keeps its native pressure coordinates; `compile_form` assembles the expressions and releases native assembly resources. The same weak form can return portable nodal matrices for the independent classical baseline. The following section declares the two boundary spaces and their pairings.




```python
import basix.ufl
import dolfinx
import ufl
from mpi4py import MPI
from pymhm.core.equations import compile_form


def native_scalar_space(fine: TriangleMesh, degree: int) -> tuple[Any, np.ndarray]:
    """Bind the declared nodal element with shared topology and coefficient maps."""
    element = basix.ufl.element(
        "Lagrange", "triangle", degree,
        lagrange_variant=basix.LagrangeVariant.equispaced,
    )
    binding = bind_space(fine, element)
    return binding.space, binding.mapping


def user_volume_forms(
    fine: TriangleMesh, degree: int, *, binding: Any = None, portable: bool = True,
) -> tuple[Any, Any, np.ndarray]:
    """Assemble the user weak form in native or declared portable nodal coordinates."""
    if binding is None:
        V, order = native_scalar_space(fine, degree)
    else:
        V, order = binding.space, binding.mapping
    p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(V.mesh)
    a_epsilon = 2 + ufl.sin(2 * np.pi * x[0] / epsilon) * ufl.sin(2 * np.pi * x[1] / epsilon)
    dx = ufl.Measure("dx", domain=V.mesh, metadata={"quadrature_degree": 2 * assembly_order})
    a = a_epsilon * ufl.inner(ufl.grad(p), ufl.grad(v)) * dx
    mass = p * v * dx
    L = source * v * dx
    A, M, F = compile_form(a), compile_form(mass), compile_form(L)
    if portable:
        return A[order][:, order].tocsc(), M[order][:, order].tocsc(), F[order]
    return A, M, F

```


```python
first_fine = macro.submesh(0, local_refinement)
A_ufl, M_ufl, F_ufl = user_volume_forms(first_fine, local_degree)
print({"operator": "user-written UFL", "local_nodal_unknowns": A_ufl.shape[0],
       "constant_kernel_relative": float(np.linalg.norm(A_ufl @ np.ones(A_ufl.shape[0])) / sparse.linalg.norm(A_ufl))})
form_equivalence = {}  # Filled by the optional convenience check at the end.

```

```text
{'operator': 'user-written UFL', 'local_nodal_unknowns': 153, 'constant_kernel_relative': 1.3562500818233753e-16}
```


```python
def local_three_field_equations(local: LocalContext) -> LocalEquations:
    """Declare native pressure/conormal equations and the shared pressure balance."""
    fine = local.mesh
    element = basix.ufl.element(
        "Lagrange", "triangle", local_degree,
        lagrange_variant=basix.LagrangeVariant.equispaced,
    )
    pressure_space = local.native_space(element)
    stiffness, mass, force = user_volume_forms(
        fine, local_degree, binding=pressure_space, portable=False,
    )
    v = ufl.TestFunction(pressure_space.space)
    boundary_forms = local.trace_pairings(
        lambda phi, ds: phi * v * ds, interface=conormal_interface,
    )
    B = assemble_pairing(boundary_forms.forms)
    P = local.interface_pairing(conormal_interface, order=4)

    # These block dimensions express the three-field formulation itself.
    npressure, nconormal = stiffness.shape[0], B.shape[1]
    local_saddle = sparse.bmat([[stiffness, -B], [-B.T, None]], format="csc")
    coupling = np.vstack((np.zeros((npressure, P.shape[1])), P))
    pressure_reconstruction = np.hstack((np.eye(npressure), np.zeros((npressure, nconormal))))
    local.field("pressure", pressure_space, reconstruction=pressure_reconstruction)
    return local.equations(
        a=local_saddle, L=np.r_[force, np.zeros(nconormal)],
        b=coupling, c=coupling.T,
        metadata={"mesh": fine, "A": stiffness, "F": force, "B": B, "P": P,
                  "pressure_count": npressure, "volume_moments": np.asarray(mass.sum(axis=1)).ravel()},
    )

```

### 4. Declare the shared trace equation and recover pressure

The stiffness $A_K$ alone has a constant kernel. For the compatible spaces above, the complete local saddle is invertible: boundary pairings fix the constant and trace moments. We neither invent a kernel for that saddle nor pin an arbitrary pressure node.

The local unknown of `LocalEquations` is $(p_K,\eta_K)$. The coupling `b` is $[0;P]$, and the global balance `c` is its transpose. Generic assembly eliminates both local fields, leaving $\rho$. Dirichlet fixes its exterior nodes, so no global pressure gauge remains.

The article's Neumann maps use a **boundary-mean-zero** decomposition. The full saddle below retains the complete pressure and does not select any nodal or volume-mean complement. It expresses the same three-field equations directly. The field record below associates the pressure coefficients with their executed P1 basis; it contains no formulation or solver.



```python
@dataclass(frozen=True)
class ScalarField:
    """A physical Pk nodal coefficient vector on one existing triangular mesh."""
    mesh: TriangleMesh
    degree: int
    values: np.ndarray

    @cached_property
    def dofs(self) -> np.ndarray:
        """Use the package's declared nodal ordering without changing its basis."""
        return nodal_space(self.mesh, self.degree)[0]

    @cached_property
    def locator(self) -> Any:
        """Find triangles from original connectivity, without retriangulating nodes."""
        return Triangulation(*self.mesh.points.T, self.mesh.cells).get_trifinder()

    @cached_property
    def geometry(self) -> np.ndarray:
        """Return the package's physical barycentric gradients."""
        return p1_geometry(self.mesh)[0]


def evaluate_scalar(field: ScalarField, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate pressure and raw gradient; choose one incident fine cell on edges."""
    selected = np.asarray(field.locator(*points.T), dtype=int)
    if np.any(selected < 0):
        raise ValueError("evaluation point lies outside the declared field mesh")
    displacement = points - field.mesh.points[field.mesh.cells[selected, 0]]
    bary = np.einsum("qia,qa->qi", field.geometry[selected], displacement)
    bary[:, 0] += 1
    basis, derivative, _ = reference_basis(field.degree, bary)
    nodal = field.values[field.dofs[selected]]
    pressure = np.einsum("qi,qi->q", basis, nodal)
    gradient = np.einsum("qib,qba,qi->qa", derivative, field.geometry[selected], nodal)
    return pressure, gradient

```


```python
hierarchy = MeshHierarchy(
    macro, tuple(macro.submesh(cell, local_refinement) for cell in range(len(macro.cells)))
)
problem = bind_problem(
    hierarchy, pressure_interface, local_three_field_equations,
    global_equation=Equation(0, 0), retained=0,
    fixed=lambda global_context: global_context.fix_faces(macro.boundary_faces, 0),
)
with threadpool_limits(1):
    system = assemble(problem)
    solution = solve(system)
pressure_fields = solution.field("pressure")
physical_fields = tuple(
    ScalarField(field.mesh, local_degree, field.portable_coefficients)
    for field in pressure_fields
)
global_free_unknowns = gamma.size - len(problem.fixed)
conormal_fields = tuple(
    mixed[data["pressure_count"]:] for data, mixed in zip(system.local_metadata, solution.fields, strict=True)
)
print({"global_pressure_trace_free": global_free_unknowns,
       "global_residual": solution.residual})

```

```text
{'global_pressure_trace_free': 129, 'global_residual': 1.087033410919318e-16}
```

### 5. Check injectivity, trace moments and macro conservation

The constant test $v=1$ gives



$$
-\int_{\partial K}\eta_K=\int_K1=\lvert K\rvert.
$$



Thus physical outward flux $q\cdot n_K=-\eta_K$ conserves on each macrocell. Check this balance, $B^Tp_K=P\rho_K$, the constant stiffness kernel and the original pressure equation. A rank check on $B$ confirms that no conormal mode is invisible on the first cell.

These are execution diagnostics. Uniform stability depends on the compatible spaces and M1/M2 conditions, not merely a small algebraic residual. Raw $-a_\varepsilon\nabla p_h$ need not equal $-\eta_K$ pointwise on a boundary and is not claimed to be an $H(\mathrm{div})$ field on the fine mesh.

The physical reconstruction reads `solution.local_trace(cell)`: the bound space supplies the executed local face coefficients, including any declared basis changes. The value convention used here has no outward-normal sign. The method's reconstruction matrix remains an explicit part of its mathematical definition.



```python
diagnostics = {"constant_kernel_relative": 0.0, "trace_moment_max": 0.0,
               "macro_conservation_max": 0.0, "original_local_relative_residual": 0.0}
basis_digests = []
archive = {"macro_points": macro.points, "macro_cells": macro.cells,
           "pressure_trace": solution.trace, "gamma_nodes": gamma.nodes}
for cell, (data, field, eta) in enumerate(
    zip(system.local_metadata, physical_fields, conormal_fields, strict=True)
):
    A, F, B, P = data["A"], data["F"], data["B"], data["P"]
    executed_pressure = pressure_fields[cell].coefficients
    rho = solution.local_trace(cell)
    diagnostics["constant_kernel_relative"] = max(
        diagnostics["constant_kernel_relative"],
        float(np.linalg.norm(A @ np.ones(A.shape[0])) / np.linalg.norm(A.data)),
    )
    diagnostics["trace_moment_max"] = max(
        diagnostics["trace_moment_max"], float(np.max(abs(B.T @ executed_pressure - P @ rho))),
    )
    diagnostics["macro_conservation_max"] = max(
        diagnostics["macro_conservation_max"], float(abs(B.sum(axis=0) @ eta + F.sum())),
    )
    defect = A @ executed_pressure - B @ eta - F
    scale = max(np.linalg.norm(F), np.linalg.norm(A @ executed_pressure), np.linalg.norm(B @ eta))
    diagnostics["original_local_relative_residual"] = max(
        diagnostics["original_local_relative_residual"], float(np.linalg.norm(defect) / scale),
    )
    digest = hashlib.sha256(np.ascontiguousarray(B).tobytes())
    digest.update(np.ascontiguousarray(P).tobytes())
    basis_digests.append(digest.hexdigest())
    descriptor = pressure_fields[cell].definition.descriptor
    binding = solution.trace_bindings[cell]
    archive.update({f"executed_pressure_{cell}": executed_pressure,
                    f"executed_pressure_mapping_{cell}": descriptor.mapping,
                    f"executed_pressure_basis_{cell}": descriptor.basis_matrix,
                    f"executed_pressure_basis_points_{cell}": descriptor.basis_points,
                    f"pressure_basis_digest_{cell}": np.array(pressure_fields[cell].basis_digest),
                    f"trace_trial_map_{cell}": binding.trial_map,
                    f"points_{cell}": field.mesh.points, f"cells_{cell}": field.mesh.cells,
                    f"pressure_{cell}": field.values, f"conormal_{cell}": eta,
                    f"boundary_pairing_{cell}": B, f"pressure_trace_pairing_{cell}": P,
                    f"gamma_dofs_{cell}": binding.dofs})
assert np.linalg.matrix_rank(system.local_metadata[0]["B"]) == 3 * conormal_segments
assert max(diagnostics.values()) < 1e-10
print(diagnostics)

```

```text
{'constant_kernel_relative': 2.1836088566646393e-16, 'trace_moment_max': 6.505213034913027e-19, 'macro_conservation_max': 9.124645483638005e-16, 'original_local_relative_residual': 8.20673263589874e-14}
```

### 6. A classical primal Galerkin baseline, with its own refinement check

On a fine **globally conforming** mesh, the baseline solves the same operator, material, source and boundary conditions:



$$
\begin{aligned}
 p_{\mathrm{CG}}&\in V_{\mathrm{CG}}\subset H_0^1(\Omega),\\
 (a_\varepsilon\nabla p_{\mathrm{CG}},\nabla v)_\Omega
   &=(1,v)_\Omega,\qquad v\in V_{\mathrm{CG}}.
\end{aligned}
$$



Use P2 on 32, 64 and 128 squares per direction, each split into two triangles. Assembly and boundary identification are visible below; `solve_linear` owns the free algebraic solve. This classical global assembly is separate from multiscale condensation. Both use DOLFINx/UFL with the same weak form; their global assembly procedures and boundary spaces are separate.

The finest classical field is a **numerical reference**, not an exact solution. First compare 32→64 and 64→128 in physical norms. The plotted difference rate is a reference-resolution indicator; two consecutive differences do not constitute a rigorous reference-error bound.



```python
def dirichlet_solve(
    matrix: Any, load: np.ndarray, dofs: np.ndarray, values: np.ndarray,
) -> np.ndarray:
    """Lift prescribed nodal values and delegate the free linear solve to PyMHM."""
    pressure = np.zeros(len(load))
    pressure[dofs] = values
    free = np.setdiff1d(np.arange(len(load)), dofs)
    pressure[free] = solve_linear(matrix[free][:, free], (load - matrix @ pressure)[free])
    return pressure

```


```python
references = []
reference_dimensions = []
for resolution in (32, 64, 128):
    fine_global = TriangleMesh.unit_square(resolution)
    A_ref, _, F_ref = user_volume_forms(fine_global, 2)
    _, xy = nodal_space(fine_global, 2)
    boundary = np.flatnonzero(np.any(np.isclose(xy, 0) | np.isclose(xy, 1), axis=1))
    p_ref = dirichlet_solve(A_ref, F_ref, boundary, np.zeros(len(boundary)))
    references.append(ScalarField(fine_global, 2, p_ref))
    reference_dimensions.append({"squares_per_direction": resolution,
                                 "triangles": len(fine_global.cells),
                                 "free_unknowns": len(p_ref) - len(boundary)})
    print(reference_dimensions[-1])
reference = references[-1]

```

```text
{'squares_per_direction': 32, 'triangles': 2048, 'free_unknowns': 3969}
```

```text
{'squares_per_direction': 64, 'triangles': 8192, 'free_unknowns': 16129}
```

```text
{'squares_per_direction': 128, 'triangles': 32768, 'free_unknowns': 65025}
```


```python
def compare_scalar_fields(
    fields: Sequence[ScalarField], reference: ScalarField,
    coefficient: Callable[[np.ndarray], np.ndarray], *, order: int = 8,
) -> dict[str, float]:
    """Integrate on the finest reference partition resolving all compared meshes.

    This notebook uses nested uniform triangular grids. Gaussian points lie
    strictly inside each reference triangle; no interface averaging is used.
    The scalar coefficient multiplies the gradient in physical flux norms.
    """
    bary, weights = triangle_quadrature(order)
    basis, derivative, _ = reference_basis(reference.degree, bary)
    errors, norms = np.zeros(3), np.zeros(3)
    for start in range(0, len(reference.mesh.cells), 256):
        stop = min(start + 256, len(reference.mesh.cells))
        triangles = reference.mesh.cells[start:stop]
        points = np.einsum("qi,tia->tqa", bary, reference.mesh.points[triangles])
        flat = points.reshape(-1, 2)
        pressure, gradient = np.empty(len(flat)), np.empty((len(flat), 2))
        assigned = np.zeros(len(flat), dtype=bool)
        for field in fields:
            inside = np.asarray(field.locator(*flat.T)) >= 0
            if np.any(inside):
                pressure[inside], gradient[inside] = evaluate_scalar(field, flat[inside])
                assigned[inside] = True
        if not np.all(assigned):
            raise ValueError("compared fields do not cover the physical integration domain")
        nodal = reference.values[reference.dofs[start:stop]]
        pref = (nodal @ basis.T).ravel()
        gref = np.einsum(
            "qib,tba,ti->tqa", derivative, reference.geometry[start:stop], nodal,
        ).reshape(-1, 2)
        a = coefficient(flat)
        measure = (reference.mesh.areas[start:stop, None] * weights).ravel()
        dp, dg = pressure - pref, gradient - gref
        dq, qref = -a[:, None] * dg, -a[:, None] * gref
        errors += np.array([measure @ dp**2, measure @ np.sum(dq**2, axis=1),
                            measure @ (a * np.sum(dg**2, axis=1))])
        norms += np.array([measure @ pref**2, measure @ np.sum(qref**2, axis=1),
                           measure @ (a * np.sum(gref**2, axis=1))])
    if np.any(norms <= 0):
        raise ValueError("relative norms require a nonzero reference field")
    absolute, relative = np.sqrt(errors), np.sqrt(errors / norms)
    return dict(pressure_L2=float(absolute[0]), flux_L2=float(absolute[1]),
                flux_energy=float(absolute[2]), pressure_relative=float(relative[0]),
                flux_relative=float(relative[1]), energy_relative=float(relative[2]))

```


```python
from matplotlib.ticker import NullFormatter

reference_refinement = [
    compare_scalar_fields((coarser,), finer, permeability, order=8)
    for coarser, finer in zip(references[:-1], references[1:])
]
for interval, norms in zip(("32 → 64", "64 → 128"), reference_refinement):
    print(interval, norms)

# These rates concern differences between successive reference meshes.
# They are indicators, not exact-solution error rates.
refinement_h = np.array([1 / 64, 1 / 128])
refinement_figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
for ax, key, label in zip(axes, ("pressure_relative", "flux_relative"),
                          ("Pressure difference / reference norm", "Flux difference / reference norm")):
    difference = np.array([row[key] for row in reference_refinement])
    rate = float(np.log(difference[0] / difference[1]) / np.log(2))
    ax.loglog(refinement_h, difference, "o-", label=f"observed difference rate = {rate:.2f}")
    ax.set_xticks(refinement_h, labels=[f"{h:.3g}" for h in refinement_h])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel="Finer classical mesh spacing", ylabel=label)
    ax.grid(True, which="both", alpha=.25)
    ax.legend(fontsize=9)
plt.show()

# A reference must resolve its own physical fields before serving as a baseline.
assert reference_refinement[-1]["pressure_relative"] < reference_refinement[0]["pressure_relative"]
assert reference_refinement[-1]["flux_relative"] < reference_refinement[0]["flux_relative"]

```

```text
32 → 64 {'pressure_L2': 3.823696684608936e-05, 'flux_L2': 0.007334647810617816, 'flux_energy': 0.005272480435871398, 'pressure_relative': 0.0018001151790877515, 'flux_relative': 0.03838323140026158, 'energy_relative': 0.03921980440020734}
64 → 128 {'pressure_L2': 3.7669518113256023e-06, 'flux_L2': 0.0021132796164602483, 'flux_energy': 0.00154056976620498, 'pressure_relative': 0.00017731712073356501, 'flux_relative': 0.011059633117206046, 'energy_relative': 0.01145890983794155}
```



[![Figure 2 — MH²M with multiscale permeability: pressure, conormal and trace](../../assets/tutorials/mh2m_multiscale/figure_21_1.png)](../../assets/tutorials/mh2m_multiscale/figure_21_1.png)


### 7. Compare pressure and physical flux on a common fine partition

The reference mesh resolves every local mesh interface in this nested, uniformly refined case. Integrate on its triangles and use its field norms as denominators:



$$
\begin{aligned}
 e_p&=\frac{\lVert p_h-p_{\mathrm{ref}}\rVert_{L^2(\Omega)}}
                {\lVert p_{\mathrm{ref}}\rVert_{L^2(\Omega)}},\\
 e_q&=\frac{\lVert q_h-q_{\mathrm{ref}}\rVert_{L^2(\Omega)}}
                {\lVert q_{\mathrm{ref}}\rVert_{L^2(\Omega)}},\\
 e_E&=\frac{\lVert a_\varepsilon^{-1/2}(q_h-q_{\mathrm{ref}})\rVert_{L^2(\Omega)}}
                {\lVert a_\varepsilon^{-1/2}q_{\mathrm{ref}}\rVert_{L^2(\Omega)}}.
\end{aligned}
$$



The comparison code keeps the coefficient inside flux and energy norms. Quadrature orders 8 and 10 independently check norm integration. At volume quadrature points, evaluate the original coefficients directly; do not compare images or interpolated color maps.

For plotting, sample every fine triangle separately. Duplicate interface points retain separate incident values. Pressure and flux-magnitude panels share scales with their reference panels; each difference has its own colorbar. All panels show the actual macro mesh.



```python
def sample_field(fields: Sequence[ScalarField], refinement: int = 2) -> dict[str, np.ndarray]:
    """Sample each fine triangle separately, preserving both sides of interfaces."""
    template = TriangleMesh(np.array([[0., 0.], [1., 0.], [0., 1.]]),
                            np.array([[0, 1, 2]])).submesh(0, refinement)
    bary = np.column_stack((1 - template.points.sum(axis=1), template.points))
    points, cells, pressure, gradients = [], [], [], []
    offset = 0
    for field in fields:
        dofs, _, basis, gradient, _ = tabulate(field.mesh, field.degree, bary)
        coordinates = np.einsum("qi,tia->tqa", bary, field.mesh.points[field.mesh.cells])
        values = field.values[dofs]
        points.append(coordinates.reshape(-1, 2))
        cells.append((template.cells[None] + len(bary) * np.arange(len(dofs))[:, None, None]
                      + offset).reshape(-1, 3))
        pressure.append((values @ basis.T).ravel())
        gradients.append(np.einsum("tqia,ti->tqa", gradient, values).reshape(-1, 2))
        offset += len(coordinates) * len(bary)
    return dict(points=np.concatenate(points), cells=np.concatenate(cells),
                values=np.concatenate(pressure), gradient=np.concatenate(gradients))


def plot_field_panels(
    macro_mesh: TriangleMesh,
    panels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Any:
    """Draw independent field/colorbar panels and the actual macro mesh."""
    count = len(panels)
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.8 * rows),
                                squeeze=False, layout="constrained")
    # Share scales between numerical and reference panels of the same field.
    pressure_max = max((float(np.max(v[2])) for k, v in panels.items()
                        if k.startswith("Pressure:")), default=1.)
    flux_max = max((float(np.max(v[2])) for k, v in panels.items()
                   if k.startswith("Flux magnitude:")), default=1.)
    for axis, (label, (points, triangles, values)) in zip(axes.flat, panels.items()):
        settings = {}
        if label.startswith("Pressure:"):
            settings = dict(vmin=0., vmax=pressure_max)
        elif label.startswith("Flux magnitude:"):
            settings = dict(vmin=0., vmax=flux_max)
        elif label == "Signed pressure difference":
            bound = max(float(np.max(abs(values))), np.finfo(float).tiny)
            settings = dict(vmin=-bound, vmax=bound, cmap="RdBu_r")
        artist = axis.tripcolor(Triangulation(*points.T, triangles), values,
                               shading="gouraud", rasterized=True, **settings)
        axis.add_collection(LineCollection(macro_mesh.points[macro_mesh.faces],
                                           colors="white", linewidths=1.0, alpha=.85))
        axis.add_collection(LineCollection(macro_mesh.points[macro_mesh.faces],
                                           colors="#24343c", linewidths=.45))
        axis.set(title=label, xlabel="x", ylabel="y", aspect="equal")
        figure.colorbar(artist, ax=axis, shrink=.85, pad=.025)
    for axis in list(axes.flat)[count:]:
        axis.set_visible(False)
    return figure

```


```python
errors_q8 = compare_scalar_fields(physical_fields, reference, permeability, order=8)
errors_q10 = compare_scalar_fields(physical_fields, reference, permeability, order=10)
quadrature_difference = max(
    abs(errors_q8[key] - errors_q10[key]) / max(abs(errors_q10[key]), np.finfo(float).tiny)
    for key in ("pressure_L2", "flux_L2", "flux_energy")
)
print({"physical_relative_errors": errors_q10,
       "relative_norm_change_q8_q10": quadrature_difference})
assert quadrature_difference < 1e-5

samples = sample_field(physical_fields, refinement=2)
points, triangles = samples["points"], samples["cells"]
p_h, grad_h = samples["values"], samples["gradient"]
p_reference, grad_reference = evaluate_scalar(reference, points)
a = permeability(points)
q_h, q_reference = -a[:, None] * grad_h, -a[:, None] * grad_reference
panels = {
    "Pressure: classical P2": (points, triangles, p_reference),
    "Pressure: MH²M": (points, triangles, p_h),
    "Signed pressure difference": (points, triangles, p_h - p_reference),
    "Flux magnitude: classical P2": (points, triangles, np.linalg.norm(q_reference, axis=1)),
    "Flux magnitude: MH²M": (points, triangles, np.linalg.norm(q_h, axis=1)),
    "Flux difference magnitude": (points, triangles, np.linalg.norm(q_h - q_reference, axis=1)),
}
figure = plot_field_panels(macro, panels)
plt.show()

```

```text
{'physical_relative_errors': {'pressure_L2': 0.00019495407271238573, 'flux_L2': 0.019449591904094187, 'flux_energy': 0.01341267956291061, 'pressure_relative': 0.009176834899960531, 'flux_relative': 0.10178745352163333, 'energy_relative': 0.09976483322479031}, 'relative_norm_change_q8_q10': 2.085495363217584e-15}
```



[![Figure 3 — MH²M with multiscale permeability: pressure, conormal and trace](../../assets/tutorials/mh2m_multiscale/figure_24_1.png)](../../assets/tutorials/mh2m_multiscale/figure_24_1.png)


### 8. Interpret and archive the experiment

Only the shared pressure trace is solved globally (129 free coordinates). Conormal refinement changes local operators without directly adding global unknowns. Fine coefficient resolution still requires local assembly and factorization work.

Read multiscale differences together with the classical reference-refinement indicator. We do not assign homogeneous-problem macro convergence orders to this oscillatory material study. Such a study must vary $H$, $H_\Gamma$, $H_\Lambda$ and $h$ under the relevant compatibility and regularity hypotheses.

If changing a face partition, preserve M1/M2 or justify a different compatible family. Refining only $\Lambda$ does not necessarily improve a coarse $\Gamma$ approximation. Save complete fields and executed boundary/pressure pairing matrices with their basis digests; never reinterpret archived coefficients using a different trace basis.


## Optional: use a prepared operator for this weak form

For the scalar diffusion form written above, `scalar_operators` provides a
portable Basix assembly convenience. Defining a new physical operator does not
require this function: the main computation already assembles the UFL expression
written by the user. The following check verifies agreement in the same local
finite-element space and basis before substituting this convenience.



```python
from pymhm.fem.scalar.triangle import scalar_operators
A_prepared, M_prepared, F_prepared = scalar_operators(
    first_fine, local_degree, diffusion=permeability, source=source, order=assembly_order,
)
form_equivalence = {
    "stiffness_relative": float(sparse.linalg.norm(A_ufl - A_prepared) / sparse.linalg.norm(A_prepared)),
    "mass_relative": float(sparse.linalg.norm(M_ufl - M_prepared) / sparse.linalg.norm(M_prepared)),
    "source_relative": float(np.linalg.norm(F_ufl - F_prepared) / np.linalg.norm(F_prepared)),
}
assert max(form_equivalence.values()) < 1e-10
print(form_equivalence)

```

```text
{'stiffness_relative': 2.681979681896503e-16, 'mass_relative': 6.514095339453426e-16, 'source_relative': 4.464810611149017e-16}
```


```python
output = ROOT / "build" / "introduction" / "mh2m_multiscale"
output.mkdir(parents=True, exist_ok=True)
figure.savefig(output / "physical-fields.png", dpi=160)
fig_material.savefig(output / "permeability.png", dpi=160)
refinement_figure.savefig(output / "reference-refinement.png", dpi=160)
record = {
    "method": "MH²M", "physical_problem": "-div(aε grad(p))=1, p|boundary=0",
    "epsilon": epsilon, "macro_triangles": len(macro.cells),
    "local_refinement": local_refinement, "local_degree": local_degree,
    "assembly_order": assembly_order, "error_orders": [8, 10],
    "global_free_unknowns": global_free_unknowns,
    "reference_dimensions": reference_dimensions,
    "reference_refinement": reference_refinement,
    "physical_errors": errors_q10,
    "quadrature_relative_difference": quadrature_difference,
    "global_residual": solution.residual, "diagnostics": diagnostics,
    "basis_digests": basis_digests, "form_equivalence": form_equivalence, "operator_language": "executed UFL",
    "native_backend": {"dolfinx": dolfinx.__version__, "basix": basix.__version__, "ufl": ufl.__version__},
    "notebook_sha256": hashlib.sha256((ROOT / "notebooks/introduction/mh2m_multiscale.ipynb").read_bytes()).hexdigest(),
    "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted((ROOT / "src" / "pymhm").rglob("*.py"))},
    "pixi_lock_sha256": hashlib.sha256((ROOT / "pixi.lock").read_bytes()).hexdigest(),
    "scope": "Original introductory case; finite local Galerkin realization; numerical classical baseline",
}
(output / "report.json").write_text(json.dumps(record, indent=2) + "\n")
np.savez_compressed(output / "executed-fields.npz", **archive)
print(output)

```

```text
./build/introduction/mh2m_multiscale
```

## References

- Franklin de Barros, Alexandre L. Madureira, and Frédéric Valentin (2026). *A three-field Multiscale Method*. arXiv preprint, version 3, 5 August 2026; first submitted 25 April 2024. [arXiv: 2404.16978v3](https://arxiv.org/abs/2404.16978v3).

## Reproduce this tutorial

[View the source notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/introduction/mh2m_multiscale.ipynb) or [download the notebook](https://raw.githubusercontent.com/ipes-lncc/pymhm/main/notebooks/introduction/mh2m_multiscale.ipynb). Run its cells interactively, or execute the notebook from the repository root with the checked-in Pixi lockfile:

```bash
pixi install --locked -e introduction
pixi run --locked -e introduction notebooks-run introduction/mh2m_multiscale.ipynb --timeout 3600
```

The runner writes the executed copy to `build/notebooks/introduction/`. The figures and numerical outputs on this page come from that execution. Timings describe the recorded hardware and solver settings; rerun performance examples on an idle machine to measure your own environment.
