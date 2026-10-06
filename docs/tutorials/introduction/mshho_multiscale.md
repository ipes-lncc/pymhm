# MsHHO with multiscale permeability: from the problem to the API

Follow the numbered steps: state the variational problem, choose the local and trace spaces, declare the local and global equations, solve, and inspect the physical fields.

We build a simple **Multiscale Hybrid High-Order** Darcy problem and explain every local and global block. All case definitions, equations, boundary conditions, reference solves, norms and plots are visible in this notebook. The weak operator is written in UFL. PyMHM provides generic compilation, trace pairings, constrained reconstruction and linear algebra.

The method family follows [Chaumont-Frelet, Ern, Lemaire and Valentin (2022)](https://doi.org/10.1051/m2an/2021082), sections 4–5. Their local spaces are defined by exactly solved PDEs. We use a finite local Galerkin realization on declared fine meshes. This original introductory problem is not a reproduction of an article figure or table.

From the repository root, run with the locked introduction environment:

```bash
pixi run --locked -e introduction python scripts/run_notebooks.py notebooks/introduction/mshho_multiscale.ipynb
```

Prerequisites: a weak Poisson formulation, basic NumPy and Jupyter. Cell-defined providers are used in serial execution; process workers require importable callables.



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

from pymhm.core.moments import energy_reconstruction

```

### 1. The physical problem and its two scales

On $\Omega=(0,1)^2$, solve



$$
\begin{aligned}
 -\nabla\cdot(a_\varepsilon\nabla p)&=1 &&\text{in }\Omega,\\
 p&=0 &&\text{on }\partial\Omega,\\
 a_\varepsilon(x,y)&=2+
 \sin(2\pi x/\varepsilon)\sin(2\pi y/\varepsilon),
 &\varepsilon&=1/8,\\
 q&=-a_\varepsilon\nabla p.
\end{aligned}
$$



The bounds $1\le a_\varepsilon\le3$ give uniform ellipticity. Eight material oscillations fit in each direction. Our **macro mesh** has four squares per direction, split into 32 triangles. Each macrocell contains material variation that its geometry cannot resolve. Its independent local problem uses 16 subdivisions per macroedge. This reduces the global system while retaining fine local work.

`TriangleMesh` defines geometry. Below, we write and execute the form $\int_K a_\varepsilon\nabla u\cdot\nabla v$ and load $\int_K v$ in UFL before introducing optional assembly conveniences.



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



[![Figure 1 — MsHHO with multiscale permeability: from the problem to the API](../../assets/tutorials/mshho_multiscale/figure_4_0.png)](../../assets/tutorials/mshho_multiscale/figure_4_0.png)


### 2. The unknowns are pressure moments

Choose $m=k_F=1$: three P1 volume moments per macrotriangle and two P1 moments per macroface. We store **integrals** rather than polynomial projection coefficients. The two coordinate systems are related by invertible polynomial mass matrices:



$$
\begin{aligned}
 z_{K,i}&=\int_K p_h r_{K,i},
 &r_K&=(1,\widehat x,\widehat y),\\
 z_{F,j}&=\int_F p_h\psi_{F,j},
 &\psi_F&=(1,2t-1).
\end{aligned}
$$



Here $\widehat x,\widehat y$ are Cartesian coordinates scaled to the macrocell bounding box. They span P1 on each triangle. Both incident macrocells share $z_F$; their complete fine nodal fields remain independent. Exterior face moments are zero by Dirichlet data.

`SkeletonSpace` numbers these face coordinates. Its name does not determine their physical meaning: **these are pressure moments**, not flux coefficients. Our local finite space is continuous P2; the moment constraints must be independent in that space.



```python
local_degree = 2
skeleton = SkeletonSpace(macro, tuple(FaceSpace.uniform(1) for _ in macro.faces))
fixed = {int(dof): 0.0 for face in macro.boundary_faces for dof in skeleton.dofs(int(face))}
global_free_unknowns = skeleton.size - len(fixed)
print({"face_moments_total": skeleton.size,
       "face_moments_free": global_free_unknowns,
       "cell_moments_per_macrocell": 3,
       "local_polynomial_degree": local_degree})

```

```text
{'face_moments_total': 112, 'face_moments_free': 80, 'cell_moments_per_macrocell': 3, 'local_polynomial_degree': 2}
```

### 3. Construct the local basis by constrained energy minimization

Let $A_K$ be the fine stiffness and $C_K$ the matrix of volume and face functionals. The reconstruction $R_K$ gives the minimum-energy field with prescribed moments:



$$
\begin{aligned}
 \begin{bmatrix}A_K&C_K\\C_K^T&0\end{bmatrix}
 \begin{bmatrix}R_K\\M_K\end{bmatrix}
 &=\begin{bmatrix}0\\I\end{bmatrix},\\
 C_K^TR_K&=I,
 &E_K&=R_K^TA_KR_K.
\end{aligned}
$$



The Neumann stiffness has the constant kernel. Independent moments fix it; the stiffness must also be coercive on the moment-zero space. A small residual alone does not prove these properties.

The P1 cell tests are represented exactly at P2 nodes, so `mass @ cell_tests` integrates their volume moments. `trace_coupling` includes the mesh normal orientation. **Pressure** moments are unsigned: we remove that orientation by multiplying each face block by `macro.signs`.

`energy_reconstruction` solves the generic $A/C$ saddle above. It receives no physical-model or method selector. The constant source belongs to P1: in integral-moment coordinates the projected-source functional is simply $z_{K,0}$, hence load $(1,0,0)$ on cell coordinates and zero on face coordinates.


### Write the local differential operator in UFL

Define the trial pressure `p` and test pressure `v` on the local Pk space. The UFL expressions below are the executable weak form, mass pairing and source. Changing this expression defines a new local operator; the multiscale contracts do not select a physical model.



$$
a_K(p,v)=\int_K a_\varepsilon\nabla p\cdot\nabla v,
\qquad \ell_K(v)=\int_K v.
$$



The small geometry adapter keeps the same fine triangles and equispaced nodal basis. It maps native coefficients by their physical coordinates; it contains no PDE. `compile_form` assembles the expressions and releases native assembly resources. The interface pairing is introduced separately, with its declared normal convention.




```python
import basix.ufl
import dolfinx
import ufl
from mpi4py import MPI
from scipy.spatial import cKDTree
from pymhm.core.equations import compile_form
from pymhm.fem.scalar.triangle import trace_coupling


def native_scalar_space(fine: TriangleMesh, degree: int) -> tuple[Any, np.ndarray]:
    """Use the existing triangles and bijectively map equispaced nodal coordinates."""
    geometry = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, fine.cells, fine.points, geometry)
    element = basix.ufl.element("Lagrange", "triangle", degree,
                                lagrange_variant=basix.LagrangeVariant.equispaced)
    V = dolfinx.fem.functionspace(domain, element)
    _, nodes = nodal_space(fine, degree)
    distance, permutation = cKDTree(V.tabulate_dof_coordinates()[:, :2]).query(nodes)
    assert distance.max() < 1e-12
    assert len(np.unique(permutation)) == len(nodes)
    return V, permutation


def user_volume_forms(fine: TriangleMesh, degree: int) -> tuple[Any, Any, np.ndarray]:
    """Execute the user-written weak operator, mass and source in the stated basis."""
    V, order = native_scalar_space(fine, degree)
    p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(V.mesh)
    a_epsilon = 2 + ufl.sin(2 * np.pi * x[0] / epsilon) * ufl.sin(2 * np.pi * x[1] / epsilon)
    dx = ufl.Measure("dx", domain=V.mesh, metadata={"quadrature_degree": 2 * assembly_order})
    a = a_epsilon * ufl.inner(ufl.grad(p), ufl.grad(v)) * dx
    mass = p * v * dx
    L = source * v * dx
    A, M, F = compile_form(a), compile_form(mass), compile_form(L)
    return A[order][:, order].tocsc(), M[order][:, order].tocsc(), F[order]

```


```python
first_fine = macro.submesh(0, local_refinement)
A_ufl, M_ufl, F_ufl = user_volume_forms(first_fine, local_degree)
print({"operator": "user-written UFL", "local_nodal_unknowns": A_ufl.shape[0],
       "constant_kernel_relative": float(np.linalg.norm(A_ufl @ np.ones(A_ufl.shape[0])) / sparse.linalg.norm(A_ufl))})
form_equivalence = {}  # Filled by the optional convenience check at the end.

```

```text
{'operator': 'user-written UFL', 'local_nodal_unknowns': 561, 'constant_kernel_relative': 3.6207041887734765e-16}
```


```python
def local_moment_equations(cell: int) -> LocalEquations:
    """Declare MsHHO P1 cell/face moments on a local conforming P2 space."""
    fine = macro.submesh(cell, local_refinement)
    stiffness, mass, force = user_volume_forms(fine, local_degree)
    _, nodes = nodal_space(fine, local_degree)
    vertices = macro.points[macro.cells[cell]]
    scaled_xy = 2 * (nodes - vertices.min(axis=0)) / np.ptp(vertices, axis=0) - 1
    cell_tests = np.column_stack((np.ones(len(nodes)), scaled_xy))
    volume_moments = mass @ cell_tests
    face_moments = trace_coupling(macro, cell, fine, skeleton, local_degree)
    offset = 0
    for side, face in enumerate(macro.cell_faces[cell]):
        width = skeleton.faces[face].size
        face_moments[:, offset:offset + width] *= macro.signs[cell, side]
        offset += width
    moments = np.column_stack((volume_moments, face_moments))
    reconstruction, energy = energy_reconstruction(stiffness, moments)

    # The source is the constant 1. In integral-moment coordinates the
    # projected-source load is (1, 0, 0) in the cell block, zero on faces.
    moment_load = np.r_[1.0, 0.0, 0.0, np.zeros(face_moments.shape[1])]
    ncell = volume_moments.shape[1]
    return LocalEquations(
        a=energy[:ncell, :ncell], L=moment_load[:ncell],
        b=energy[:ncell, ncell:], c=energy[ncell:, :ncell],
        dofs=skeleton.cell_dofs(cell),
        d=energy[ncell:, ncell:], g=moment_load[ncell:],
        metadata={"mesh": fine, "R": reconstruction, "C": moments,
                  "A": stiffness, "F": force, "energy": energy},
    )

```

### 4. The global problem contains face moments

Partition $E_K$ into cell ($c$) and face ($F$) blocks. The projected-source formulation becomes



$$
\begin{aligned}
 E_{cc}z_K+E_{cF}z_{\partial K}&=(1,0,0)^T,\\
 \sum_K\big(E_{Fc}z_K+E_{FF}z_{\partial K}\big)&=0.
\end{aligned}
$$



These are exactly `LocalEquations`: `a,L,b` define the first equation; `c,d,g` contribute the second. `assemble` eliminates the three cell moments and adds shared face contributions.

`Equation(0, zeros)` supplies no additional global term. `coarse_sizes=(0,...)` retains no extra modes: the fine stiffness kernel has already been fixed in the constrained reconstruction, and cell moments are ordinary eliminated local unknowns. Prescribed exterior pressure moments remove the global pressure gauge.

After solving, `solution.fields` contains **cell moments**. Multiplying the complete local moment vector by the executed $R_K$ recovers physical nodal pressure. The small field record below only associates coefficients with their declared mesh and Pk basis; it is not a solver.



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
problem = MultiscaleProblem(
    global_equation=Equation(0, np.zeros(skeleton.size)),
    local_provider=local_moment_equations,
    items=range(len(macro.cells)),
    trace_size=skeleton.size, coarse_sizes=(0,) * len(macro.cells), fixed=fixed,
)
with threadpool_limits(1):
    system = assemble(problem)
    solution = system.solve()

# solution.fields contains cell moments, not physical nodal pressures.
physical_fields = tuple(
    ScalarField(data["mesh"], local_degree,
                data["R"] @ np.r_[cell_moments, solution.trace[skeleton.cell_dofs(cell)]])
    for cell, (data, cell_moments) in enumerate(zip(system.local_metadata, solution.fields, strict=True))
)
print({"global_free_unknowns": global_free_unknowns,
       "global_residual": solution.residual,
       "local_nodal_unknowns_per_macrocell": len(physical_fields[0].values)})

```

```text
{'global_free_unknowns': 80, 'global_residual': 6.462215755491108e-17, 'local_nodal_unknowns_per_macrocell': 561}
```

### 5. Check the kernel and the executed moment basis

Check $A_K1=0$, $C_K^TR_K=I$, and the moments of the recovered pressure. Full pressure may jump across a macroface; its P1 face moments agree.

The finite local Galerkin realization does not automatically inherit the ideal article space's $H(\mathrm{div})$ flux property. We report the raw physical flux $-a_\varepsilon\nabla p_h$ inside each fine element and do not assert fine-cell conservation.

Archive each executed $R_K$ with its SHA256 digest and the complete moment vector. Field replay uses that same matrix; persisted moment coefficients must not be applied to a numerically changed basis.



```python
diagnostics = {"constant_kernel_relative": 0.0, "moment_identity_max": 0.0,
               "recovered_moment_max": 0.0}
basis_digests = []
archive = {"macro_points": macro.points, "macro_cells": macro.cells,
           "face_moments": solution.trace}
for cell, (data, field, cell_moments) in enumerate(
    zip(system.local_metadata, physical_fields, solution.fields, strict=True)
):
    A, C, R = data["A"], data["C"], data["R"]
    target = np.r_[cell_moments, solution.trace[skeleton.cell_dofs(cell)]]
    diagnostics["constant_kernel_relative"] = max(
        diagnostics["constant_kernel_relative"],
        float(np.linalg.norm(A @ np.ones(A.shape[0])) / np.linalg.norm(A.data)),
    )
    diagnostics["moment_identity_max"] = max(
        diagnostics["moment_identity_max"], float(np.max(abs(C.T @ R - np.eye(C.shape[1])))),
    )
    diagnostics["recovered_moment_max"] = max(
        diagnostics["recovered_moment_max"], float(np.max(abs(C.T @ field.values - target))),
    )
    basis_digests.append(hashlib.sha256(np.ascontiguousarray(R).tobytes()).hexdigest())
    archive.update({f"points_{cell}": field.mesh.points, f"cells_{cell}": field.mesh.cells,
                    f"pressure_{cell}": field.values, f"reconstruction_{cell}": R,
                    f"moments_{cell}": C, f"moment_coordinates_{cell}": target})
assert max(diagnostics.values()) < 1e-10
print(diagnostics)

```

```text
{'constant_kernel_relative': 3.644021074115437e-16, 'moment_identity_max': 3.552713678800501e-15, 'recovered_moment_max': 5.204170427930421e-18}
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



[![Figure 2 — MsHHO with multiscale permeability: from the problem to the API](../../assets/tutorials/mshho_multiscale/figure_21_1.png)](../../assets/tutorials/mshho_multiscale/figure_21_1.png)


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
    "Pressure: MsHHO": (points, triangles, p_h),
    "Signed pressure difference": (points, triangles, p_h - p_reference),
    "Flux magnitude: classical P2": (points, triangles, np.linalg.norm(q_reference, axis=1)),
    "Flux magnitude: MsHHO": (points, triangles, np.linalg.norm(q_h, axis=1)),
    "Flux difference magnitude": (points, triangles, np.linalg.norm(q_h - q_reference, axis=1)),
}
figure = plot_field_panels(macro, panels)
plt.show()

```

```text
{'physical_relative_errors': {'pressure_L2': 0.0003496968018570189, 'flux_L2': 0.022527859850014497, 'flux_energy': 0.015758818791433912, 'pressure_relative': 0.01646085034817636, 'flux_relative': 0.11789725454046812, 'energy_relative': 0.11721564816135269}, 'relative_norm_change_q8_q10': 1.3951828413288198e-15}
```



[![Figure 3 — MsHHO with multiscale permeability: from the problem to the API](../../assets/tutorials/mshho_multiscale/figure_24_1.png)](../../assets/tutorials/mshho_multiscale/figure_24_1.png)


### 8. Interpret and archive the experiment

Compare the 80 free global face coordinates with the refined Galerkin dimension. Inspect the flux oscillations **inside** the coarse triangles. Measured errors include macro approximation and finite local resolution; fewer global unknowns alone do not establish lower total computational cost.

To change the problem, alter the coefficient, source and boundary moments first, then rerun the moment and reference-refinement checks. To resolve local material more accurately, increase `local_refinement`. To enrich the macro approximation, change cell and face moment tests consistently. The ideal equivalence and convergence results of the article require their own local-space hypotheses; this finite Galerkin example does not claim them automatically.

Save the numerical record, physical coefficients, moment vectors and executed reconstruction matrices. Source notebooks remain output-free; their executed versions are written under `build/notebooks/introduction/`.


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
{'stiffness_relative': 1.6755633221035159e-15, 'mass_relative': 1.410965141920348e-15, 'source_relative': 1.0931848206312767e-15}
```


```python
output = ROOT / "build" / "introduction" / "mshho_multiscale"
output.mkdir(parents=True, exist_ok=True)
figure.savefig(output / "physical-fields.png", dpi=160)
fig_material.savefig(output / "permeability.png", dpi=160)
refinement_figure.savefig(output / "reference-refinement.png", dpi=160)
record = {
    "method": "MsHHO", "physical_problem": "-div(aε grad(p))=1, p|boundary=0",
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
    "notebook_sha256": hashlib.sha256((ROOT / "notebooks/introduction/mshho_multiscale.ipynb").read_bytes()).hexdigest(),
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
./build/introduction/mshho_multiscale
```

## References

- Théophile Chaumont-Frelet, Alexandre Ern, Simon Lemaire, and Frédéric Valentin (2022). *Bridging the Multiscale Hybrid-Mixed and Multiscale Hybrid High-Order Methods*, ESAIM: M2AN 56, 261–285. [DOI: 10.1051/m2an/2021082](https://doi.org/10.1051/m2an/2021082).

## Reproduce this tutorial

[View the source notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/introduction/mshho_multiscale.ipynb) or [download the notebook](https://raw.githubusercontent.com/ipes-lncc/pymhm/main/notebooks/introduction/mshho_multiscale.ipynb). Run its cells interactively, or execute the notebook from the repository root with the checked-in Pixi lockfile:

```bash
pixi install --locked -e introduction
pixi run --locked -e introduction notebooks-run introduction/mshho_multiscale.ipynb --timeout 3600
```

The runner writes the executed copy to `build/notebooks/introduction/`. The figures and numerical outputs on this page come from that execution. Timings describe the recorded hardware and solver settings; rerun performance examples on an idle machine to measure your own environment.
