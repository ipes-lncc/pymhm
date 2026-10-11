# Primal MHM

The original MHM is hybrid because it couples broken local pressure fields
through a skeleton multiplier. Its global problem is mixed: the unknowns
are normal Darcy-flux coefficients and one pressure mean per macroelement.
The local finite element method here is primal. Local Neumann responses
contain the full permeability field; no homogenized permeability is
substituted. See [Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019).

This tutorial uses **two-level MHM**. The first level discretizes the
skeletal normal flux and retains one pressure mean per macroelement.
The second level approximates the local Neumann responses in fine-mesh
$Q_2$ spaces. With exact local responses, the skeletal discretization
defines the one-level method; see
[Araya et al. (2013)](https://doi.org/10.1137/120888223).

[Open the step-by-step notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/introduction/darcy_multiscale_convergence.ipynb) · [Download notebook](https://raw.githubusercontent.com/ipes-lncc/pymhm/main/notebooks/introduction/darcy_multiscale_convergence.ipynb)

Follow the numbered steps: state the variational problem, choose the local and trace spaces, declare the local and global equations, solve, and inspect the physical fields.

The main workflow is **meshes → spaces → local equations → global balance → assemble → solve → fields and errors**. `MeshHierarchy` associates macro and local meshes; `bind_interface` and `LocalContext` own supported numbering, geometric orientation and native coordinate conversions. The mathematical forms, physical trace meaning, local modes and gauge remain explicit in the cells below. Fully manual/custom spaces use the same numerical owners; see the [custom-interface notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/custom_interface.ipynb).

This notebook builds primal MHM step by step: bind the macro and local meshes, declare the spaces, write local UFL equations and the global balance, then assemble and solve. A small macro mesh controls the global problem; independent local fine meshes resolve the material oscillations. The baseline is classical conforming Galerkin assembled on several much finer global meshes.

Install `pymhm[notebooks,visualization]` and the compatible native DOLFINx/UFL backend, then open this notebook with `jupyter lab` in a writable directory. Run the cells in order. Physical data and equations are declared here; downloaded companion helpers provide evaluation, norms, plots and archives.

The manufactured solution distinguishes the finite-reference error from the multiscale error. This is an analytical control devised for this tutorial, rather than a matched paper reproduction.

The local Neumann responses, skeletal flux unknowns and retained cell constants follow [Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019). The oscillatory manufactured data and refinement sequence are defined for this tutorial.

Field evaluation, norms, plots and executed-array archives use the importable
[supporting scalar helpers](https://github.com/ipes-lncc/pymhm/blob/main/examples/introduction/scalar.py).
The physical data and local/global variational equations remain explicit below.


The PyMHM distribution contains only the library. The first cell explicitly
downloads a checksum-verified companion archive and prepares the declared data
using its local Python helpers. You can inspect these support files in `ROOT`.
Complete studies and historical replay retain their existing opt-in flags.



```python
from pathlib import Path
import os
import sys
from pymhm.io.workspace import workspace_from_archive

# Download verified support files; this operation does not execute them.
COMPANION_URL = "https://ipes-lncc.github.io/pymhm/downloads/3473ec3bc17a7c7dc8fdf16c6b2dd1057bdbbd6ddbddc6e6cf4b2c9b83b2f027/darcy_multiscale_convergence-companion.zip"
COMPANION_SHA256 = "3473ec3bc17a7c7dc8fdf16c6b2dd1057bdbbd6ddbddc6e6cf4b2c9b83b2f027"
WORKSPACE = Path(
    os.environ.get("PYMHM_WORKSPACE", Path.cwd() / ".pymhm-companions" / COMPANION_SHA256)
)
ROOT = workspace_from_archive(COMPANION_URL, sha256=COMPANION_SHA256, directory=WORKSPACE)
os.environ["PYMHM_WORKSPACE"] = str(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Explicitly prepare declared data with the downloaded Python helpers.
from scripts.notebook_reproduction import notebook_workspace

ROOT = notebook_workspace("introduction/darcy_multiscale_convergence.ipynb", directory=ROOT)
root = ROOT
print("Workspace:", ROOT)


import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import partial
from typing import Any
from pymhm import Equation, LocalEquations, assemble, columns
from pymhm import MeshHierarchy, LocalContext, bind_interface, bind_problem
from pymhm.core.equations import compile_form
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from examples.introduction.scalar import (
    Array,
    dirichlet_solve,
    evaluate_bound_pressure,
    evaluate_qk,
    grid_quadrature,
    native_scalar_space,
    observed_rates,
    physical_errors,
    plot_convergence,
    plot_field_panels,
    rectangle_panel,
)
import ufl
from pymhm import ExecutionConfig, CartesianMacroMesh
from pymhm.fem.scalar.quadrilateral import qk_space, quadrilateral_operators
from pymhm.fem.scalar.operators import boundary_data

```

```text
Workspace: .
```

## 1. Physical problem and an independently derived source

On $\Omega=(0,1)^2$, Darcy's law and conservation are



$$
\boldsymbol q=-K\nabla p,\qquad\nabla\cdot\boldsymbol q=f,
\qquad p=0\quad\text{on }\partial\Omega.
$$



Choose $\varepsilon=1/8$, $a=1.5$ and



$$
\begin{aligned}
K(x,y)&=\mathrm{diag}\!\left(e^{a\sin(\omega x)},e^{a\cos(\omega y)}\right),
\qquad\omega=2\pi/\varepsilon,\\
p_\star(x,y)&=\sin(\pi x)\sin(\pi y).
\end{aligned}
$$



The uniform spectral contrast is $e^{2a}\simeq20.1$. Eight oscillations fit in each direction, even when only four macroelements fit in that direction. Differentiate **before** discretizing:



$$
f=\pi^2(K_{xx}+K_{yy})p_\star
 -(\partial_xK_{xx})\partial_xp_\star
 -(\partial_yK_{yy})\partial_yp_\star.
$$



The data class below implements these expressions directly. Its source never applies an assembled matrix to the exact coefficients. The coefficient is smooth, but its derivatives depend on $\varepsilon$; our convergence measurements keep $\varepsilon$ fixed.


```python
@dataclass(frozen=True)
class OscillatoryDarcyData:
    """Smooth anisotropic diffusion with declared wavelength and contrast.

    ``Kxx=exp(a*sin(2*pi*x/epsilon))`` and
    ``Kyy=exp(a*cos(2*pi*y/epsilon))``; cross entries vanish. Thus the
    uniform spectral contrast is ``exp(2*a)``, independently of the mesh.
    """

    epsilon: float = 0.125
    amplitude: float = 1.5

    def permeability(self, points: Array) -> Array:
        """Return the physical SPD tensor at XY points, with no averaging."""
        phase = 2 * np.pi * points / self.epsilon
        tensors = np.zeros((len(points), 2, 2))
        tensors[:, 0, 0] = np.exp(self.amplitude * np.sin(phase[:, 0]))
        tensors[:, 1, 1] = np.exp(self.amplitude * np.cos(phase[:, 1]))
        return tensors

    def pressure(self, points: Array) -> Array:
        """Return ``sin(pi*x)*sin(pi*y)`` on the unit square."""
        return np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])

    def gradient(self, points: Array) -> Array:
        """Return the independently differentiated exact pressure gradient."""
        x, y = np.pi * points.T
        return np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))

    def flux(self, points: Array) -> Array:
        """Return exact physical Darcy flux ``-K grad(p)``."""
        return -np.einsum("qab,qb->qa", self.permeability(points), self.gradient(points))

    def source(self, points: Array) -> Array:
        """Return independently differentiated ``-div(K grad(p))``.

        This expression uses analytic derivatives of both K and p, never the
        assembled operator or a discretely manufactured right-hand side.
        """
        omega = 2 * np.pi / self.epsilon
        k = self.permeability(points)
        derivative_x = self.amplitude * omega * np.cos(omega * points[:, 0]) * k[:, 0, 0]
        derivative_y = -self.amplitude * omega * np.sin(omega * points[:, 1]) * k[:, 1, 1]
        gradient = self.gradient(points)
        return (
            np.pi**2 * (k[:, 0, 0] + k[:, 1, 1]) * self.pressure(points)
            - derivative_x * gradient[:, 0]
            - derivative_y * gradient[:, 1]
        )

```

## 2. Choose the macro mesh, local meshes and trace space

Choose the discretization of the global and local problems:

- **Macro mesh:** `macro` partitions the unit square into macroelements. Here $H$ denotes the side length of each square.
- **Local meshes and spaces:** `local_refinement=r` divides each macroelement into $r\times r$ fine squares, with side length $h=H/r$. `local_degree=k` sets the local pressure space $Q_k$.
- **Global trace space:** the trace degree $\ell$ and `trace_segments` select the normal-flux approximation on each macroface. Segmentation refines this global space.

These choices configure the two-level discretization. The coefficient period $\varepsilon=1/8$ is a physical parameter of the permeability and remains fixed throughout the refinement study.



```python
data = OscillatoryDarcyData(epsilon=1 / 8, amplitude=1.5)
exact = lambda points: (data.pressure(points), data.gradient(points))
macro = CartesianMacroMesh(4, 4)
# H=1/4; h_local=H/r=1/64; degree k=2; trace degree ell=1.
local_degree, local_refinement, trace_segments = 2, 16, 2
# The oscillatory, nonpolynomial material requires independently qualified integration.
assembly_order = 26  # Gauss points per axis; UFL quadrature degree 51.
skeleton = SkeletonSpace(
    macro, tuple(FaceSpace.uniform(1, trace_segments, continuous=True) for _ in macro.faces)
)
print(
    {
        "macro_cells": len(macro.cells),
        "H": 1 / 4,
        "local_h": 1 / 64,
        "epsilon": data.epsilon,
        "spectral_contrast": float(np.exp(2 * data.amplitude)),
    }
)

```

```text
{'macro_cells': 16, 'H': 0.25, 'local_h': 0.015625, 'epsilon': 0.125, 'spectral_contrast': 20.085536923187668}
```

## 3. Translate the weak formulation into local and global objects

On each macroelement $T$, pressure uses a continuous $Q_k$ fine space. The skeletal multiplier $\lambda$ is the physical normal Darcy flux in the unique orientation $\boldsymbol n_F$ of face $F$. The normal interface binding derives $s_{TF}=\boldsymbol n_T\cdot\boldsymbol n_F$ from the macroface topology. Write the trace integrals in UFL; `LocalContext` supplies their basis supports and coefficient maps.



$$
\begin{aligned}
a_T(p_T,v)+\langle s_{TF}\lambda,v\rangle_{\partial T}&=(f,v)_T,\\
a_T(p_T,v)&=\int_T K\nabla p_T\cdot\nabla v.
\end{aligned}
$$



| Mathematical term | Code declaration |
| --- | --- |
| Local energy and source | `a=inner(K*grad(p),grad(v))*dx`, `L=f*v*dx` in UFL |
| Oriented normal-flux pairing | `local.trace_pairings(lambda phi, ds: phi*v*ds)` |
| Constant kernel and physical average | `kernel`, `moments` in `LocalEquations` |
| Weak pressure continuity | `local.trace_pairings(lambda phi, ds: -phi*p*ds, axis="rows")` |
| Macroface coordinates | `bind_interface(skeleton, convention="normal")` supplies local/global maps |

The local Neumann kernel is constant. The pressure decomposition is

$$
p_T=c_T+T_T\lambda+\widehat T_Tf,
\qquad\int_TT_T\lambda=\int_T\widehat T_Tf=0.
$$

The source response and each interface response are solved on the
mean-zero complement. The constant $c_T$ is a global physical unknown.
The retained compatibility equation supplies the missing constant test,

$$
\int_{\partial T}s_{TF}\lambda\,ds=\int_Tf\,dx.
$$

Together with the pressure-coupling moments, this is the original MHM
global problem. It is not imposed by setting all local pressure means
to zero.

Interior faces have zero pressure-jump moments; Dirichlet faces have prescribed pressure moments. With the declared sign $C=-B^T$, the additional global right-hand side is



$$
-\sum_T\langle s_{TF}\mu,p_T\rangle_{\partial T}
=-\langle\mu,p_D\rangle_{\Gamma_D}.
$$



`Equation(0, ...)` stores this additional global term. One constant mode per macroelement gives the compatibility equation imposing **macro conservation**. The Darcy flux is evaluated as $q_h=-K\nabla p_h$.

All local solves use 26 Gauss points per coordinate direction, declared in UFL
as `quadrature_degree=51`. This integrates the oscillatory coefficient and
manufactured source on the coarsest local mesh as well as the finer meshes.
Assembly quadrature is an independent numerical choice: increasing it changes
neither the local polynomial degree nor the skeleton space. Field errors use a
separate, fixed integration grid, so changes in the discretization choices
can be compared consistently.

The provider below declares the executable UFL weak forms and each coupling sign. `local.native_space` owns the native geometry and coefficient convention, while `local.trace_pairings` binds both interface forms independently. The provider supplies no face DOF indices or nodal permutation. The generic `assemble` operation owns elimination, assembly and reconstruction; no PDE-specific problem constructor is used.


```python
def ufl_coefficient_source(domain: Any, physical_data: OscillatoryDarcyData) -> tuple[Any, Any]:
    """Declare the same smooth K and independently differentiated manufactured f in UFL."""
    x = ufl.SpatialCoordinate(domain)
    a, omega = physical_data.amplitude, 2 * np.pi / physical_data.epsilon
    kxx = ufl.exp(a * ufl.sin(omega * x[0]))
    kyy = ufl.exp(a * ufl.cos(omega * x[1]))
    tensor = ufl.as_matrix(((kxx, 0.0), (0.0, kyy)))
    exact_pressure = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    pressure_x = np.pi * ufl.cos(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    pressure_y = np.pi * ufl.sin(np.pi * x[0]) * ufl.cos(np.pi * x[1])
    kxx_x = a * omega * ufl.cos(omega * x[0]) * kxx
    kyy_y = -a * omega * ufl.sin(omega * x[1]) * kyy
    force = np.pi**2 * (kxx + kyy) * exact_pressure - kxx_x * pressure_x - kyy_y * pressure_y
    return tensor, force


@dataclass(frozen=True)
class DarcyLocalProvider:
    """Declare primal Qk UFL energy, physical flux coupling and pressure moments."""

    macro: CartesianMacroMesh
    skeleton: SkeletonSpace
    data: object
    degree: int
    refinement: int
    quadrature_order: int

    def __call__(self, local: LocalContext) -> LocalEquations:
        """Declare K grad(p)·grad(v), f v, oriented normal flux and the constant average."""
        cell, fine = local.cell, local.mesh
        import basix
        import basix.ufl

        element = basix.ufl.element(
            "Lagrange",
            "quadrilateral",
            self.degree,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        )
        binding = local.native_space(element)
        domain, space = binding.mesh, binding.space
        mapping = binding.mapping
        p, v = ufl.TrialFunction(space), ufl.TestFunction(space)
        K, f = ufl_coefficient_source(domain, self.data)
        dx = ufl.Measure(
            "dx", domain=domain, metadata={"quadrature_degree": 2 * self.quadrature_order - 1}
        )
        # These executed UFL forms are the local weak formulation, directly.
        a = ufl.inner(K * ufl.grad(p), ufl.grad(v)) * dx
        load = f * v * dx
        local.field("pressure", binding)
        b = local.trace_pairings(lambda phi, ds: phi * v * ds)
        c = local.trace_pairings(lambda phi, ds: -phi * p * ds, axis="rows")
        area = float(fine.areas.sum())
        return local.equations(
            a=a,
            L=load,
            b=b,
            c=c,
            kernel=np.ones((len(mapping), 1)),
            moments=columns((v / area) * dx),
            metadata=(fine, mapping),
        )

```

## 4. Construct, assemble and solve the global problem

`retained=1` retains one pressure-average coordinate per macroelement; `bind_problem` derives its global coordinate layout. The global equation imposes pressure through boundary moments, without fixing local pressure nodes. Dirichlet data remove the global constant-pressure ambiguity, so no additional physical gauge is required.

The local problems compute source responses and responses to trace basis functions. Their coefficients are recovered after solving for the macroface and retained coordinates. They do not create a large conforming global mesh.

### Evaluate pressure in its executed coefficient basis

Before reconstructing fields, define how the x-fastest Qk coefficients represent pressure and physical gradients. The second function selects a macrocell and evaluates its independent local coefficients. This is field evaluation, not another solve; it preserves the distinction between broken pressure and physical Darcy flux.


```python
provider = DarcyLocalProvider(macro, skeleton, data, local_degree, local_refinement, assembly_order)
boundary, fixed = boundary_data(skeleton, data.pressure, order=7)
hierarchy = MeshHierarchy(
    macro, tuple(macro.submesh(cell, local_refinement) for cell in range(len(macro.cells)))
)
interface = bind_interface(skeleton, convention="normal")
problem = bind_problem(
    hierarchy,
    interface,
    provider,
    global_equation=lambda global_problem: Equation(0, global_problem.trace_load(-boundary)),
    retained=1,
    fixed=fixed,
)
system = assemble(problem, execution=ExecutionConfig("serial", native_threads=1))
solution = system.solve()
pressure_fields = solution.field("pressure")
local_meshes = tuple(field.mesh for field in pressure_fields)
mhm_fields = tuple(field.portable_coefficients for field in pressure_fields)
mhm_evaluator = partial(evaluate_bound_pressure, macro, pressure_fields)
print(
    {
        "global_unknowns": system.matrix.shape[0],
        "largest_local_unknowns": max(len(v) for v in solution.fields),
        "original_equations_relative_residual": solution.raw_residual,
    }
)
# Named fields carry their mesh and executed basis; no index map is needed to evaluate.
first_point = macro.points[macro.cells[0]].mean(axis=0, keepdims=True)
print("First macrocell pressure at its center:", pressure_fields[0].evaluate(first_point))

```

```text
{'global_unknowns': 136, 'largest_local_unknowns': 1089, 'original_equations_relative_residual': 1.1110151320839327e-16}
First macrocell pressure at its center: [0.14646015]
```

### Optional numerical equivalence after the UFL definition

The main formulation above is the executed UFL weak form. Once its operator and coordinate conventions are explicit, `quadrilateral_operators` provides an alternative assembled representation. The following control verifies equivalence on one local mesh; it does not select the global/local problem or hide its forms. The classical baseline below remains an explicit UFL assembly.


```python
declared = problem.local_provider(0)
fine, mapping = declared.metadata
ufl_a = compile_form(declared.a)
ufl_load = compile_form(declared.L, (len(mapping),))
ready_a, _, ready_load = quadrilateral_operators(
    fine,
    local_degree,
    permeability=data.permeability,
    source=data.source,
    order=provider.quadrature_order,
)
operator_defect = np.max(np.abs((ufl_a[mapping][:, mapping] - ready_a).data), initial=0.0)
load_defect = float(np.max(np.abs(ufl_load[mapping] - ready_load), initial=0.0))
print({"UFL_vs_ready_operator_max": float(operator_defect), "UFL_vs_ready_source_max": load_defect})
assert operator_defect < 1e-10 * max(1.0, float(np.max(np.abs(ready_a.data))))
assert load_defect < 1e-10 * max(1.0, float(np.max(np.abs(ready_load))))

```

```text
{'UFL_vs_ready_operator_max': 7.105427357601002e-14, 'UFL_vs_ready_source_max': 2.1510571102112408e-16}
```

## 5. Classical primal Galerkin on independently refined global meshes

Now assemble **one global continuous matrix**, without an MHM skeleton or local condensation. The operator, permeability, source and boundary values are identical. Strong Dirichlet values eliminate boundary nodes; `dirichlet_solve`, defined below, performs only that generic algebraic elimination.



$$
A^{\rm CG}_{ij}=\int_\Omega K\nabla\phi_j\cdot\nabla\phi_i,
\qquad F^{\rm CG}_i=\int_\Omega f\phi_i.
$$



Evaluate pressure and raw gradients directly from the executed nodal basis. Integrate on a common grid resolving every compared finite-element mesh, with physical Gauss weights. The field measurements are pressure $L^2$, vector Darcy flux $L^2$, and flux energy



$$
\left(\int_\Omega\boldsymbol e_q^T K^{-1}\boldsymbol e_q\right)^{1/2}.
$$



The baseline is assembled independently as a different classical discretization, while sharing PyMHM basis and integration kernels. It is not an independent external-code comparison. Successive references are compared, and the analytical solution separately verifies their errors.


```python
reference_rows, reference_evaluators = [], []
error_points, error_weights = grid_quadrature(macro.bounds, (128, 128), order=7)
for n in (32, 64, 128):
    fine = CartesianMacroMesh(n, n)
    domain, space, mapping = native_scalar_space(fine, 2)
    p, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    K, f = ufl_coefficient_source(domain, data)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * assembly_order - 1})
    a_cg = compile_form(ufl.inner(K * ufl.grad(p), ufl.grad(v)) * dx)
    load_cg = compile_form(f * v * dx)
    _, nodes = qk_space(fine, 2)
    exterior = np.flatnonzero(np.any(np.isclose(nodes, 0) | np.isclose(nodes, 1), axis=1))
    coefficients = dirichlet_solve(
        a_cg, load_cg, mapping[exterior], data.pressure(nodes[exterior])
    )[mapping]
    evaluator = partial(evaluate_qk, fine, 2, coefficients)
    reference_evaluators.append(evaluator)
    errors = physical_errors(evaluator, exact, data.permeability, error_points, error_weights)
    reference_rows.append({"n": n, "unknowns": len(nodes), **errors})
reference_rows

```





```text
[{'n': 32,
  'unknowns': 4225,
  'pressure_L2': 1.6311682549303613e-05,
  'flux_L2': 0.001500019626951807,
  'flux_energy': 0.0009521702517930258,
  'pressure_relative': 3.2623365098607226e-05,
  'flux_relative': 0.0003056447149562913,
  'energy_relative': 0.0003340176631603194},
 {'n': 64,
  'unknowns': 16641,
  'pressure_L2': 1.2191023086918222e-06,
  'flux_L2': 0.00042476133764538903,
  'flux_energy': 0.0002514193228832227,
  'pressure_relative': 2.4382046173836444e-06,
  'flux_relative': 8.654957284318855e-05,
  'energy_relative': 8.819693174058369e-05},
 {'n': 128,
  'unknowns': 66049,
  'pressure_L2': 9.40083804181791e-08,
  'flux_L2': 0.00010915481610239898,
  'flux_energy': 6.37048253925932e-05,
  'pressure_relative': 1.880167608363582e-07,
  'flux_relative': 2.2241437414735915e-05,
  'energy_relative': 2.2347407797713358e-05}]
```





```python
reference_refinement = [
    physical_errors(first, second, data.permeability, error_points, error_weights)
    for first, second in zip(reference_evaluators[:-1], reference_evaluators[1:])
]
print("Successive reference differences:", reference_refinement)
reference = reference_evaluators[-1]
comparison = physical_errors(
    mhm_evaluator, reference, data.permeability, error_points, error_weights
)
exact_errors = physical_errors(mhm_evaluator, exact, data.permeability, error_points, error_weights)
print("MHM versus fine CG:", comparison)
print("MHM versus exact solution:", exact_errors)
# The exact solution keeps this conclusion independent of reference uncertainty.
assert reference_rows[-1]["flux_L2"] < reference_rows[0]["flux_L2"]
assert reference_rows[-1]["pressure_L2"] < reference_rows[0]["pressure_L2"]

```

```text
Successive reference differences: [{'pressure_L2': 1.5227554215844082e-05, 'flux_L2': 0.0014259460346024602, 'flux_energy': 0.0009183771081969466, 'pressure_relative': 3.045504084239697e-05, 'flux_relative': 0.0002905517837260761, 'energy_relative': 0.000322163159577337}, {'pressure_L2': 1.1518248263488742e-06, 'flux_L2': 0.0004109899554135434, 'flux_energy': 0.0002432146605356715, 'pressure_relative': 2.30364932261365e-06, 'flux_relative': 8.374351590097702e-05, 'energy_relative': 8.531876775781793e-05}]
```

```text
MHM versus fine CG: {'pressure_L2': 0.00021110534460575797, 'flux_L2': 0.019092751630456706, 'flux_energy': 0.012636675043150416, 'pressure_relative': 0.00042221062871402036, 'flux_relative': 0.0038903484834555825, 'energy_relative': 0.0044328970172397815}
MHM versus exact solution: {'pressure_L2': 0.00021110587827486888, 'flux_L2': 0.01909305299115884, 'flux_energy': 0.01263683572847488, 'pressure_relative': 0.00042221175654973776, 'flux_relative': 0.0038904095881644074, 'energy_relative': 0.004432953383932771}
```

## 6. Plot the coefficient, pressure and physical flux

The overlay is the actual macro mesh used by MHM, on every analytical, numerical and reference panel. Duplicate interface coordinates belong to separate local triangulations, preserving both one-sided values. Neither the broken pressure nor its raw flux is smoothed across macrofaces.

The permeability tensor has two distinct diagonal components: $K_{xx}$ controls transport in the x direction and $K_{yy}$ in the y direction. Plot both with the same physical color limits to compare their oscillations. The field panels then place the analytical pressure and Darcy flux beside MHM and the refined classical reference.


```python
permeability_panels = {
    "Permeability $K_{xx}$": rectangle_panel(
        macro, exact, quantity="permeability_xx", permeability=data.permeability
    ),
    "Permeability $K_{yy}$": rectangle_panel(
        macro, exact, quantity="permeability_yy", permeability=data.permeability
    ),
}
permeability_bounds = (np.exp(-data.amplitude), np.exp(data.amplitude))
plot_field_panels(
    macro,
    permeability_panels,
    figsize=(12, 4.8),
    color_limits={label: permeability_bounds for label in permeability_panels},
)
plt.show()

panels = {
    "Analytical pressure": rectangle_panel(macro, exact),
    "MHM pressure": rectangle_panel(macro, mhm_evaluator),
    "Fine CG pressure": rectangle_panel(macro, reference),
    "Analytical Darcy flux magnitude": rectangle_panel(
        macro, exact, quantity="flux_magnitude", permeability=data.permeability
    ),
    "MHM Darcy flux magnitude": rectangle_panel(
        macro, mhm_evaluator, quantity="flux_magnitude", permeability=data.permeability
    ),
    "Fine CG Darcy flux magnitude": rectangle_panel(
        macro, reference, quantity="flux_magnitude", permeability=data.permeability
    ),
}
plot_field_panels(macro, panels, figsize=(15, 9))
plt.show()

```



[![Figure 1 — Multiscale Darcy: a two-level primal MHM workflow](../../assets/tutorials/darcy_multiscale_convergence/figure_17_0.png)](../../assets/tutorials/darcy_multiscale_convergence/figure_17_0.png)




[![Figure 2 — Multiscale Darcy: a two-level primal MHM workflow](../../assets/tutorials/darcy_multiscale_convergence/figure_17_1.png)](../../assets/tutorials/darcy_multiscale_convergence/figure_17_1.png)


## 7. Separate macro size, local size and macroface resolution

Configure the macro partition, local meshes and skeleton space separately to study their effects on the error. Keep $k=2$ and $\ell=1$ and run:

1. $H=1/2,1/4,1/8$ with fixed $h=1/64$ and two segments per face;
2. $h=1/8,1/16,1/32,1/64$ with fixed $H=1/4$ and two segments per face;
3. one, two and four segments per face with fixed $H=1/4$ and $h=1/64$.

Each configuration uses the same qualified assembly quadrature (26 Gauss
points per axis), including the coarsest local mesh. The error integration
remains fixed independently of these refinements.

With $s$ segments per macroface, each segment has length $H/s$. In the first experiment, changing $H$ therefore also changes the physical trace-segment length, even though the segment count stays fixed. In the second experiment, the macro mesh and trace space stay fixed; in the third, both volume meshes stay fixed.

The driver below repeats the same visible local/global declarations. Successive observed rates use $\log(e_i/e_{i+1})/\log(\eta_i/\eta_{i+1})$, where $\eta$ is $H$, $h$, or segment length. A plateau can reflect an error contribution from a discretization choice held fixed. Section 8 studies the asymptotic rates for its stated smooth problem and approximation spaces.


```python
def run_discretization(
    n: int, refinement: int, segments: int, *, quadrature_order: int = assembly_order
) -> dict:
    """Repeat the declared local/global equations for one explicit resolution."""
    grid = CartesianMacroMesh(n, n)
    trace = SkeletonSpace(
        grid, tuple(FaceSpace.uniform(1, segments, continuous=True) for _ in grid.faces)
    )
    local = DarcyLocalProvider(grid, trace, data, 2, refinement, quadrature_order)
    boundary, fixed = boundary_data(trace, data.pressure, order=7)
    hierarchy = MeshHierarchy(
        grid, tuple(grid.submesh(cell, refinement) for cell in range(len(grid.cells)))
    )
    declared = bind_problem(
        hierarchy,
        bind_interface(trace, convention="normal"),
        local,
        global_equation=lambda global_problem: Equation(0, global_problem.trace_load(-boundary)),
        retained=1,
        fixed=fixed,
    )
    assembled = assemble(declared, execution=ExecutionConfig("serial", native_threads=1))
    resolved = assembled.solve()
    evaluate = partial(evaluate_bound_pressure, grid, resolved.field("pressure"))
    return {
        "H": 1 / n,
        "h": 1 / (n * refinement),
        "segments": segments,
        "assembly_gauss_points_per_axis": quadrature_order,
        "trace_segment_length": 1 / (n * segments),
        "global_unknowns": assembled.matrix.shape[0],
        "residual": resolved.raw_residual,
        **physical_errors(evaluate, exact, data.permeability, error_points, error_weights),
    }


macro_rows = [run_discretization(n, 64 // n, 2) for n in (2, 4, 8)]
local_rows = [run_discretization(4, r, 2) for r in (2, 4, 8, 16)]
trace_rows = [run_discretization(4, 16, s) for s in (1, 2, 4)]
for name, rows, variable in (
    ("H", macro_rows, "H"),
    ("h", local_rows, "h"),
    ("macroface segment", trace_rows, "trace_segment_length"),
):
    scales = [row[variable] for row in rows]
    print(name, rows)
    print("Observed pressure rates:", observed_rates(scales, [r["pressure_L2"] for r in rows]))
    print("Observed flux rates:", observed_rates(scales, [r["flux_L2"] for r in rows]))
    figure = plot_convergence(
        scales,
        {
            "pressure L2": [r["pressure_L2"] for r in rows],
            "Darcy flux L2": [r["flux_L2"] for r in rows],
        },
    )
    for axis in figure.axes:
        axis.set_xlabel(name)
    plt.show()

```

```text
H [{'H': 0.5, 'h': 0.015625, 'segments': 2, 'assembly_gauss_points_per_axis': 26, 'trace_segment_length': 0.25, 'global_unknowns': 40, 'residual': 3.6872751243354284e-16, 'pressure_L2': 0.0038440419909965243, 'flux_L2': 0.07258266496358691, 'flux_energy': 0.07565780837939495, 'pressure_relative': 0.007688083981993049, 'flux_relative': 0.01478947844745518, 'energy_relative': 0.026540468269334178}, {'H': 0.25, 'h': 0.015625, 'segments': 2, 'assembly_gauss_points_per_axis': 26, 'trace_segment_length': 0.125, 'global_unknowns': 136, 'residual': 1.1110151320839327e-16, 'pressure_L2': 0.00021110587827486888, 'flux_L2': 0.01909305299115884, 'flux_energy': 0.01263683572847488, 'pressure_relative': 0.00042221175654973776, 'flux_relative': 0.0038904095881644074, 'energy_relative': 0.004432953383932771}, {'H': 0.125, 'h': 0.015625, 'segments': 2, 'assembly_gauss_points_per_axis': 26, 'trace_segment_length': 0.0625, 'global_unknowns': 496, 'residual': 1.0512929247960976e-16, 'pressure_L2': 3.1368102919560713e-05, 'flux_L2': 0.0046017888463211086, 'flux_energy': 0.0031211773986935957, 'pressure_relative': 6.273620583912143e-05, 'flux_relative': 0.0009376626911749365, 'energy_relative': 0.001094897030291863}]
Observed pressure rates: [4.18658544 2.75059657]
Observed flux rates: [1.92657722 2.05278112]
```



[![Figure 3 — Multiscale Darcy: a two-level primal MHM workflow](../../assets/tutorials/darcy_multiscale_convergence/figure_19_1.png)](../../assets/tutorials/darcy_multiscale_convergence/figure_19_1.png)


```text
h [{'H': 0.25, 'h': 0.125, 'segments': 2, 'assembly_gauss_points_per_axis': 26, 'trace_segment_length': 0.125, 'global_unknowns': 136, 'residual': 1.2538891685613543e-16, 'pressure_L2': 0.0014175186973032102, 'flux_L2': 0.032107165227115826, 'flux_energy': 0.019148357106354187, 'pressure_relative': 0.0028350373946064203, 'flux_relative': 0.006542171307343601, 'energy_relative': 0.006717170045986694}, {'H': 0.25, 'h': 0.0625, 'segments': 2, 'assembly_gauss_points_per_axis': 26, 'trace_segment_length': 0.125, 'global_unknowns': 136, 'residual': 1.249712816668689e-16, 'pressure_L2': 0.0002134928004798865, 'flux_L2': 0.019233707237595186, 'flux_energy': 0.011695147063781344, 'pressure_relative': 0.000426985600959773, 'flux_relative': 0.003919069364534661, 'energy_relative': 0.004102612621224435}, {'H': 0.25, 'h': 0.03125, 'segments': 2, 'assembly_gauss_points_per_axis': 26, 'trace_segment_length': 0.125, 'global_unknowns': 136, 'residual': 1.246386877072097e-16, 'pressure_L2': 0.0002070063686187589, 'flux_L2': 0.019169355242091866, 'flux_energy': 0.012587647033400743, 'pressure_relative': 0.0004140127372375178, 'flux_relative': 0.003905956971223881, 'energy_relative': 0.004415698178835121}, {'H': 0.25, 'h': 0.015625, 'segments': 2, 'assembly_gauss_points_per_axis': 26, 'trace_segment_length': 0.125, 'global_unknowns': 136, 'residual': 1.1110151320839327e-16, 'pressure_L2': 0.00021110587827486888, 'flux_L2': 0.01909305299115884, 'flux_energy': 0.01263683572847488, 'pressure_relative': 0.00042221175654973776, 'flux_relative': 0.0038904095881644074, 'energy_relative': 0.004432953383932771}]
Observed pressure rates: [ 2.73110844  0.04451227 -0.0282916 ]
Observed flux rates: [0.73925843 0.00483505 0.005754  ]
```



[![Figure 4 — Multiscale Darcy: a two-level primal MHM workflow](../../assets/tutorials/darcy_multiscale_convergence/figure_19_3.png)](../../assets/tutorials/darcy_multiscale_convergence/figure_19_3.png)


```text
macroface segment [{'H': 0.25, 'h': 0.015625, 'segments': 1, 'assembly_gauss_points_per_axis': 26, 'trace_segment_length': 0.25, 'global_unknowns': 96, 'residual': 9.543184134267058e-17, 'pressure_L2': 0.003837047167948131, 'flux_L2': 0.09746442298587585, 'flux_energy': 0.09155965342167202, 'pressure_relative': 0.007674094335896262, 'flux_relative': 0.019859397334975343, 'energy_relative': 0.03211877436633401}, {'H': 0.25, 'h': 0.015625, 'segments': 2, 'assembly_gauss_points_per_axis': 26, 'trace_segment_length': 0.125, 'global_unknowns': 136, 'residual': 1.1110151320839327e-16, 'pressure_L2': 0.00021110587827486888, 'flux_L2': 0.01909305299115884, 'flux_energy': 0.01263683572847488, 'pressure_relative': 0.00042221175654973776, 'flux_relative': 0.0038904095881644074, 'energy_relative': 0.004432953383932771}, {'H': 0.25, 'h': 0.015625, 'segments': 4, 'assembly_gauss_points_per_axis': 26, 'trace_segment_length': 0.0625, 'global_unknowns': 216, 'residual': 1.4103711785446846e-16, 'pressure_L2': 2.4324400273386223e-05, 'flux_L2': 0.003250083737124539, 'flux_energy': 0.0022417239886965796, 'pressure_relative': 4.864880054677245e-05, 'flux_relative': 0.0006622386131281087, 'energy_relative': 0.0007863881556316715}]
Observed pressure rates: [4.18395784 3.11749061]
Observed flux rates: [2.35182789 2.55449901]
```



[![Figure 5 — Multiscale Darcy: a two-level primal MHM workflow](../../assets/tutorials/darcy_multiscale_convergence/figure_19_5.png)](../../assets/tutorials/darcy_multiscale_convergence/figure_19_5.png)


## 8. Qualify the smooth asymptotic rate separately

The oscillatory study in section 7 varies macro, local and trace scales for its chosen material. Its measured orders do not verify a smooth asymptotic estimate. A separate analytical control uses the data of section 5.1 in [Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019) on the unit square:

$$
\begin{aligned}
K&=I,& p(x,y)&=\cos(2\pi x)\cos(2\pi y),\\
f&=8\pi^2p,& q&=-\nabla p,\\
q\cdot n&=0\quad\text{on }\partial\Omega,& \int_\Omega p&=0.
\end{aligned}
$$

Use local P1 pressure on each macrotriangle and unsegmented P0 normal-flux traces. The zero physical pressure integral fixes the Neumann gauge. Refine the declared diagonal triangular mesh through $n=4,8,16,32,64$, with macro diameter $H=\sqrt{2}/n$.

Theorem 4.1 of [Araya et al. (2013)](https://doi.org/10.1137/120888223) gives one-level energy order one and pressure order two under its solution and dual-regularity hypotheses, with exact local PDE responses. The finite P1/P0 family here also contains local finite-element error. Its first-order physical-flux and second-order pressure behavior are measured observations; the latter is also reported in the JCP paper. This control uses its declared mesh and does not claim reproduction of the original mesh connectivity. Its smooth rates do not transfer to a different coefficient, unresolved local solve or singular domain.

### Recompute every measured order

For successive physical errors $E_{i-1},E_i$ at the actual macro diameters, compute

$$
r_i=\frac{\log(E_{i-1}/E_i)}{\log(H_{i-1}/H_i)}.
$$

The cell reads the attributed numerical record, displays every measured level and recomputes its orders. It does not execute a PDE acquisition. The fitted terminal slope uses the final four levels and their three intervals. For each declared target $q$, the amplitude ratio is the maximum divided by the minimum of $E/H^q$ over those same levels. A value near one indicates stabilization of the compensated error; an observed L2 order is kept separate from a proved energy estimate.

```python
import json
from IPython.display import Markdown, display
from examples.tutorial_convergence import refinement_series, asymptotic_summary, plot_method_series
refinement_record = ROOT / 'examples/results/darcy_2013_comparison.json'
refinement_data = json.loads(refinement_record.read_text(encoding='utf-8'))
refinement_rows = refinement_data['results']
refinement_error_keys = {'pressure L2': 'pressure_l2', 'physical Darcy flux L2': 'flux_l2'}
refinement_targets = {'physical Darcy flux L2': 1.0}
refinement_rows = sorted(refinement_rows, key=lambda refinement_row: refinement_row['macro_diameter'], reverse=True)
refinement_sizes = np.asarray([refinement_row['macro_diameter'] for refinement_row in refinement_rows])
refinement_field_errors = {refinement_field: np.asarray([refinement_row[refinement_key] for refinement_row in refinement_rows]) for refinement_field, refinement_key in refinement_error_keys.items()}
refinement_orders = {refinement_field: np.log(refinement_values[:-1] / refinement_values[1:]) / np.log(refinement_sizes[:-1] / refinement_sizes[1:]) for refinement_field, refinement_values in refinement_field_errors.items()}
refinement_header = ['Refinement size'] + [refinement_column for refinement_field in refinement_error_keys for refinement_column in (refinement_field, 'Order')]
refinement_lines = [' | '.join(refinement_header), ' | '.join(['---'] * len(refinement_header))]
for refinement_index, refinement_size in enumerate(refinement_sizes):
    refinement_values = [f'{refinement_size:.6g}']
    for refinement_field in refinement_error_keys:
        refinement_values.extend([f'{refinement_field_errors[refinement_field][refinement_index]:.6e}', '—' if refinement_index == 0 else f'{refinement_orders[refinement_field][refinement_index - 1]:.3f}'])
    refinement_lines.append(' | '.join(refinement_values))
display(Markdown('\n'.join(refinement_lines)))
refinement_series_data = refinement_series(refinement_record, refinement_rows, 'macro_diameter', refinement_error_keys, refinement_targets, method='primal-mhm', spaces='P1 local pressure; P0 normal-flux trace', rate_provenance='Harder, Paredes and Valentin (2013): declared finite P1/P0 study; energy order one and pressure order two observed. Araya et al. (2013), Theorem 4.1 analyzes exact local responses; finite locals add local approximation error.', root=ROOT)
print({'record': refinement_series_data['record'], 'sha256': refinement_series_data['sha256']})
for refinement_field, refinement_summary in asymptotic_summary(refinement_series_data).items():
    print(refinement_field, {'last_three_orders': [round(refinement_order, 3) for refinement_order in refinement_summary['orders']], 'fitted_order': round(refinement_summary['fitted_order'], 3), 'target': refinement_summary['target'], 'normalized_amplitude_ratio': refinement_summary.get('amplitude_ratio')})
plot_method_series(refinement_series_data)
plt.show()

```

| Refinement size | pressure L2 | Order | physical Darcy flux L2 | Order |
| --- | --- | --- | --- | --- |
| 0.353553 | 1.114293e-01 | — | 2.419077e+00 | — |
| 0.176777 | 3.020032e-02 | 1.883 | 1.276715e+00 | 0.922 |
| 0.0883883 | 7.721936e-03 | 1.968 | 6.472200e-01 | 0.980 |
| 0.0441942 | 1.941659e-03 | 1.992 | 3.247330e-01 | 0.995 |
| 0.0220971 | 4.861202e-04 | 1.998 | 1.625073e-01 | 0.999 |

```text
{'record': 'examples/results/darcy_2013_comparison.json', 'sha256': '81ccaf3b9e11a385ed92c01547881d18a5ad02ef694ea0cfd3aa9fa699011155'}
pressure L2 {'last_three_orders': [1.968, 1.992, 1.998], 'fitted_order': 1.986, 'target': None, 'normalized_amplitude_ratio': None}
physical Darcy flux L2 {'last_three_orders': [0.98, 0.995, 0.999], 'fitted_order': 0.992, 'target': 1.0, 'normalized_amplitude_ratio': 1.0182843961826575}
```

![All physical errors, successive orders and target-normalized amplitudes](../../assets/tutorials/methods/primal-mhm-convergence.png)

Download the figure as [SVG](../../assets/tutorials/methods/primal-mhm-convergence.svg) or [PDF](../../assets/tutorials/methods/primal-mhm-convergence.pdf). Shading covers the same final four levels in every panel; dashed curves are target powers anchored to the finest measured error.

The smooth convergence figure uses the [attributed numerical record](https://github.com/ipes-lncc/pymhm/blob/main/examples/results/darcy_2013_comparison.json) (SHA256 `81ccaf3b9e11a385ed92c01547881d18a5ad02ef694ea0cfd3aa9fa699011155`). The [shared rate notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/convergence/method_rates.ipynb) compares the method sequences.

## Reproduce this tutorial

[View the source notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/introduction/darcy_multiscale_convergence.ipynb) or [download the notebook](https://raw.githubusercontent.com/ipes-lncc/pymhm/main/notebooks/introduction/darcy_multiscale_convergence.ipynb), then open it:

```bash
python -m pip install 'pymhm[notebooks,visualization]'
jupyter lab darcy_multiscale_convergence.ipynb
```

The first cell downloads a SHA256-verified companion containing inspectable support files. Downloading does not execute them. Native UFL forms require the compatible backend described in the [installation guide](../../installation.md).

For batch execution, extract the same companion, change to its workspace and run:

```bash
python -m scripts.run_notebooks /path/to/darcy_multiscale_convergence.ipynb --timeout 7200
```

The runner uses the active interpreter and writes an executed copy and receipt under `build/notebooks/introduction/`. Larger field archives have [documented downloads](../../data.md).

Figures and outputs of sections 1–7 use notebook SHA256 `c68c829464b85ab26563022a2ddb45c64e5a7127303feb75438aa4bbb3d6a529` in the [publication manifest](notebook-manifest.json). The smooth convergence figure and tables use the separate numerical records cited in section 8. Running the current source produces a receipt for its actual notebook, support files and environment.

## References

- Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *A family of Multiscale Hybrid-Mixed finite element methods for the Darcy equation with rough coefficients*, Journal of Computational Physics 245, 107–130. [DOI: 10.1016/j.jcp.2013.03.019](https://doi.org/10.1016/j.jcp.2013.03.019).

- Rodolfo Araya, Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *Multiscale Hybrid-Mixed Method*, SIAM Journal on Numerical Analysis 51(6), 3505–3531. [DOI: 10.1137/120888223](https://doi.org/10.1137/120888223).
