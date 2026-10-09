# Residual Petrov–Galerkin MHM

[Open the step-by-step notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/darcy/pgmhm_workflow.ipynb) · [Download notebook](https://raw.githubusercontent.com/ipes-lncc/pymhm/main/notebooks/darcy/pgmhm_workflow.ipynb) · [Theory and degree conditions](../../theory/alternatives.md)


We declare the primal local equations, add a user-written global residual form, and lift its boundary residual through the same local solver owner. The physical pressure is $p=\sin(2\pi x)\sin(2\pi y)$, with identity diffusion and independently derived source $8\pi^2p$. Exterior pressure is zero.

The construction follows [Fernando, Martins, Pereira and Valentin (2023)](https://doi.org/10.1007/s40314-023-02304-y). The trace multiplier of the base field differs from the enriched conservative normal flux. Neither volume gradient is automatically H(div)-conforming.

```python
import numpy as np
import matplotlib.pyplot as plt
import ufl
from pymhm import (
    Equation, FaceSpace, LocalContext, MeshHierarchy, SkeletonSpace, TriangleMesh,
    assemble, bind_interface, bind_problem, columns, solve,
)
from threadpoolctl import threadpool_limits


def exact_pressure(points: np.ndarray) -> np.ndarray:
    """Analytical pressure; it vanishes on every exterior edge."""
    return np.sin(2 * np.pi * points[:, 0]) * np.sin(2 * np.pi * points[:, 1])


def exact_gradient(points: np.ndarray) -> np.ndarray:
    """Independently differentiated physical pressure gradient."""
    x, y = 2 * np.pi * points.T
    return 2 * np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))


def source(points: np.ndarray) -> np.ndarray:
    """Negative Laplacian computed independently of the discrete operator."""
    return 8 * np.pi**2 * exact_pressure(points)
from scipy import sparse
from pymhm import with_global_equation
from pymhm.fem.scalar.triangle import scalar_operators, trace_coupling
from pymhm.fem.traces.jump import face_jump_form
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.fields import DiscreteField
from pymhm.core.reconstruction import reconstruct_local
from pymhm.postprocessing.solutions import PGMHMSolution
```

## 1. Define spaces

Choose trace degree one and local degree three. In two dimensions this meets the energy-estimate restriction $k\ge\ell+d$ and $\ell\ge1$. Select $\alpha=0.1$ and certified diffusion lower bound one.

```python
macro = TriangleMesh.unit_square(2)
local_meshes = tuple(macro.submesh(cell, 2) for cell in range(len(macro.cells)))
hierarchy = MeshHierarchy(macro, local_meshes)
skeleton = SkeletonSpace(macro, tuple(FaceSpace.uniform(1) for _ in macro.faces))
interface = bind_interface(skeleton, convention="normal")
degree, alpha, lower = 3, 0.1, 1.0
```

## 2. Define the primal volume and independent trace forms

$$
A_T(p,v)=(\nabla p,\nabla v)_T,\qquad
B_T(\lambda,v)=\langle\lambda,v\rangle_{\partial T},\qquad
C_T(p,\mu)=-\langle p,\mu\rangle_{\partial T}.
$$

For reference, the volume integrand is `ufl.inner(ufl.grad(p), ufl.grad(v)) * dx`. Here the shared Basix `scalar_operators` operation integrates this declared volume form in the portable nodal basis used by the common jump operator. It supplies element tabulation and quadrature; it does not select the multiscale method. The block signs, kernel, physical moment and global equation below are explicit. `trace_coupling` already applies canonical incidence, hence `coordinates="global"`.

```python
def local_equations(local: LocalContext):
    """Declare volume diffusion, physical mean and signed scalar trace forms."""
    fine = local.mesh
    A, mass, force = scalar_operators(fine, degree, diffusion=1.0, source=source, order=12)
    B = trace_coupling(macro, local.cell, fine, skeleton, degree)
    constant = np.ones((A.shape[0], 1))
    field = nodal_field("pressure", fine, degree)
    local.field("pressure", evaluator=field.evaluator,
                gradient_evaluator=field.gradient_evaluator, basis_id=field.basis_id)
    return local.equations(a=A, L=force, b=B, c=-B.T,
                          kernel=constant, moments=mass @ constant,
                          coordinates="global", metadata=(fine,))


def global_equation(global_context):
    """Homogeneous weak Dirichlet pressure in the independent trace test."""
    boundary, _ = global_context.boundary_data(0.0, order=8)
    return Equation(0, global_context.trace_load(-boundary))


problem = bind_problem(hierarchy, interface, local_equations,
                       global_equation=global_equation, retained=1)
with threadpool_limits(1):
    base_system = assemble(problem)
```

## 3. Write the global residual jump equation

Let $J_F z+j_F^f$ be the one-sided pressure jump reconstructed from reduced coordinates $z$, and set $\tau_F=\alpha/(2H_F)$. The additional global terms are

$$
\begin{aligned}
D_{\mathrm{jump}}&=\sum_F J_F^T W_F\tau_FJ_F,\\
g_{\mathrm{jump}}&=\sum_FJ_F^TW_F\tau_F(p_D-j_F^f).
\end{aligned}
$$

`face_jump_form` supplies executed response maps, common quadrature and one-sided trace evaluation. The multiplication and global accumulation below implement the two displayed forms. Coincident global entries are summed by sparse COO assembly.

```python
penalties = tuple(face_jump_form(base_system, skeleton, face, degree, 12,
                                 alpha, lower, 0.0)
                  for face in range(len(macro.faces)))
rows, columns_, entries = [], [], []
load = np.zeros_like(base_system.rhs)
for data in penalties:
    weighted = data.weights * data.coefficient
    block = data.jump.T @ (weighted[:, None] * data.jump)
    forcing = data.jump.T @ (weighted * (data.prescribed - data.source_jump))
    rows.extend(np.repeat(data.indices, len(data.indices)))
    columns_.extend(np.tile(data.indices, len(data.indices)))
    entries.extend(block.ravel())
    np.add.at(load, data.indices, forcing)
D_jump = sparse.coo_matrix((entries, (rows, columns_)),
                          shape=base_system.matrix.shape).tocsc()
system = with_global_equation(base_system, Equation(D_jump, load))
with threadpool_limits(1):
    coefficients = solve(system)
```

## 4. Define and solve the enrichment problem

The normal correction is $\delta\lambda_F=-\tau_F([p]-p_D)$. Its local boundary load is assembled from the declared one-sided pairing, and the correction has zero physical volume mean:

$$
A_T\delta p_T=-B_T\delta\lambda_T,\qquad \int_T\delta p_T=0.
$$

The same `reconstruct_local` owner solves this additional local load. No factorization or basis algorithm is copied into the notebook. The final `PGMHMSolution` is a physical data record for norms and conservation; it does not construct or choose the problem.

```python
coordinates = np.r_[coefficients.trace, *coefficients.coarse]
extra = [np.zeros(len(response.problem.load)) for response in system.responses]
for data in penalties:
    jump = data.jump @ coordinates[data.indices] + data.source_jump - data.prescribed
    normal_correction = -data.coefficient * jump
    for cell, sign, evaluation in data.sides:
        extra[cell] += sign * (evaluation.T @ (data.weights * normal_correction))
enriched = []
for response, base, boundary_load in zip(system.responses, coefficients.fields, extra, strict=True):
    local_problem = response.problem.with_load(-boundary_load)
    correction = reconstruct_local(local_problem, np.zeros(local_problem.coupling.shape[1]),
                                   np.zeros(local_problem.coarse_basis.shape[1]))
    enriched.append(base + correction)
solution = PGMHMSolution(skeleton, local_meshes, coefficients.fields, tuple(enriched),
                        coefficients, system, 1.0, source, degree, 12,
                        alpha, lower, penalties, tuple(extra))
pressure_fields = tuple(DiscreteField(nodal_field("pressure", fine, degree), values)
                        for fine, values in zip(local_meshes, enriched, strict=True))
print({"pressure_L2": solution.l2_error(exact_pressure, order=12, enriched=True),
       "Darcy_flux_L2": solution.flux_l2_error(lambda x: -exact_gradient(x), order=12, enriched=True),
       "enriched_macro_balance": float(np.max(abs(solution.conservation_residuals())))})
assert np.max(abs(solution.conservation_residuals())) < 1e-10
```

```text
{'pressure_L2': 0.12576942534393837, 'Darcy_flux_L2': 1.8288671828046679, 'enriched_macro_balance': 3.497202527569243e-15}
```

## 5. Plot the enriched physical pressure

The macro mesh is shown on every analytical, numerical and error panel.

```python
fig, axes = plt.subplots(2, 2, figsize=(8.5, 7), layout="constrained")
all_exact, all_values, panels = [], [], []
for fine, field in zip(local_meshes, pressure_fields, strict=True):
    points = fine.points[fine.cells].mean(axis=1)
    exact, values = exact_pressure(points), field.evaluate(points)
    flux_magnitude = np.linalg.norm(-field.gradient(points), axis=1)
    panels.append((fine, exact, values, flux_magnitude))
    all_exact.extend(exact)
    all_values.extend(values)
common = max(np.max(np.abs(all_exact)), np.max(np.abs(all_values)))
error_max = max(np.max(abs(values - exact)) for _, exact, values, _ in panels)
flux_max = max(np.max(flux) for _, _, _, flux in panels)
titles = ("Analytical pressure", "Hybrid pressure", "Pressure error", "Hybrid Darcy flux magnitude")
for index, (ax, title) in enumerate(zip(axes.flat, titles, strict=True)):
    for fine, exact, values, flux in panels:
        value = (exact, values, values - exact, flux)[index]
        limit = (common, common, error_max, flux_max)[index]
        artist = ax.tripcolor(*fine.points.T, fine.cells, facecolors=value,
                             vmin=0.0 if index == 3 else -limit,
                             vmax=limit, cmap="viridis" if index == 3 else "coolwarm")
    for edge in macro.faces:
        ax.plot(*macro.points[edge].T, color="black", lw=0.5, alpha=0.7)
    ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(artist, ax=ax)
plt.show()
```

![Pressure and signed error sampled at fine-cell centroids](../../assets/tutorials/methods/pgmhm-field-00.png)

The field panels use one sample at each fine triangle’s centroid $x_t$, drawn as a constant triangle color. The analytical panel shows $p(x_t)$; the numerical panel shows $p_h^\star(x_t)$, and the error panel shows the **signed** difference $p_h^\star(x_t)-p(x_t)$. These are display samples of the continuous analytical and enriched local polynomial fields. The physical error norms above use independent volume quadrature. Macro edges are overlaid and values from opposite sides of an interface remain separate. The Darcy flux magnitude panel samples $\lVert-\nabla p_h^\star(x_t)\rVert$, with identity permeability.

## Independently refined classical primal Galerkin baseline

The classical baseline uses one globally conforming P2 space on independent 16×16, 32×32 and 64×64 triangular grids, with the same physical coefficient, source and boundary values:

$$
(\nabla p_h^{\mathrm{CG}},\nabla v_h)=(f,v_h),\qquad
p_h^{\mathrm{CG}}\big\vert_{\partial\Omega}=p_D.
$$

The shared volume operator integrates this already stated form. The explicit essential lifting $A_{ff}p_f=f_f-A_{fb}p_D$ selects its classical boundary values. Check this reference's own errors before comparing fields. The multiscale solve above retains its simpler macro mesh and separate local spaces; it does not use this refined global grid. An exactly representable polynomial has roundoff-level classical errors on all three grids.

```python
from threadpoolctl import threadpool_limits
from pymhm.fem.scalar.triangle import scalar_operators, nodal_space, tabulate
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.linalg.linear import solve_linear

classical_rows = []
bary, weights = triangle_quadrature(12)
for resolution in (16, 32, 64):
    reference_mesh = TriangleMesh.unit_square(resolution)
    A, _, forcing = scalar_operators(reference_mesh, 2, diffusion=1.0, source=source, order=12)
    dofs, nodes = nodal_space(reference_mesh, 2)
    boundary = np.flatnonzero(np.any(np.isclose(nodes, 0.0) | np.isclose(nodes, 1.0), axis=1))
    free = np.setdiff1d(np.arange(len(nodes)), boundary)
    reference_pressure = np.zeros(len(nodes))
    reference_pressure[boundary] = exact_pressure(nodes[boundary])
    rhs = forcing[free] - A[free][:, boundary] @ reference_pressure[boundary]
    with threadpool_limits(1):
        reference_pressure[free] = solve_linear(A[free][:, free], rhs)
    _, _, basis, gradient, _ = tabulate(reference_mesh, 2, bary)
    points = np.einsum("qi,tia->tqa", bary, reference_mesh.points[reference_mesh.cells])
    difference = reference_pressure[dofs] @ basis.T - exact_pressure(points.reshape(-1, 2)).reshape(points.shape[:2])
    error = float(np.sqrt(reference_mesh.areas @ (difference**2 @ weights)))
    classical_rows.append({"resolution": resolution, "elements": len(reference_mesh.cells),
                           "pressure_L2": error})
for row in classical_rows:
    print(row)
# Polynomial patches can be exactly represented, in which case all levels are
# at roundoff. Otherwise verify the classical reference's own refinement.
assert (max(row["pressure_L2"] for row in classical_rows) < 1e-10
        or all(b["pressure_L2"] < a["pressure_L2"] for a, b in zip(classical_rows, classical_rows[1:])))
centers = reference_mesh.points[reference_mesh.cells].mean(axis=1)
_, _, center_basis, center_gradient, _ = tabulate(reference_mesh, 2, np.full((1, 3), 1/3))
reference_values = (reference_pressure[dofs] @ center_basis.T)[:, 0]
reference_flux = -np.einsum("ti,tqia->tqa", reference_pressure[dofs], center_gradient)[:, 0]
fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
for ax, values, title in zip(axes, (reference_values, np.linalg.norm(reference_flux, axis=1)),
                             ("Fine classical P2 pressure", "Fine classical Darcy flux magnitude"), strict=True):
    artist = ax.tripcolor(*reference_mesh.points.T, reference_mesh.cells, facecolors=values, cmap="viridis")
    for edge in macro.faces:
        ax.plot(*macro.points[edge].T, color="black", lw=0.5, alpha=0.7)
    ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(artist, ax=ax)
plt.show()
```

```text
{'resolution': 16, 'elements': 512, 'pressure_L2': 0.0005479034106938733}
{'resolution': 32, 'elements': 2048, 'pressure_L2': 6.873255317566801e-05}
{'resolution': 64, 'elements': 8192, 'pressure_L2': 8.600387223616975e-06}
```

![Classical pressure and Darcy flux magnitude at fine-cell centroids](../../assets/tutorials/methods/pgmhm-field-01.png)

The classical panels also use fine-cell centroid samples for display. Its pressure and gradient are evaluated from the conforming P2 coefficients; the reference-refinement errors are integrated independently of these plotting samples. The overlaid macro mesh belongs to the multiscale comparison.

## 6. Reach the smooth asymptotic regime

The method page ends with five macro refinements, independent physical pressure/flux errors and the expected second-order enriched flux rate. It retains the published P1 trace/P3 local spaces and independently controlled local discretization. The small workflow here is a teaching solve, not that full acquisition.

## Smooth refinement and the asymptotic regime

The following independent qualification study uses **P3 locals; P1 trace; residual enrichment; alpha=0.1** and refines the **macro diameter**. It states its own geometry, data and spaces; its smooth rates are not transferred to the oscillatory teaching case or to singular material interfaces. Successive orders use the independently integrated physical errors:

$$
r_i=\frac{\log(E_{i-1}/E_i)}{\log(h_{i-1}/h_i)}.
$$

| Physical observable | Literature order under its hypotheses | Last three measured orders |
| --- | ---: | --- |
| enriched pressure L2 | reported observation | 3.118 / 3.001 / 2.988 |
| enriched Darcy flux L2 | 2 | 2.229 / 2.078 / 2.019 |

![Physical errors and successive rates](../../assets/tutorials/methods/pgmhm-convergence.png)

Download the figure as [SVG](../../assets/tutorials/methods/pgmhm-convergence.svg) or [PDF](../../assets/tutorials/methods/pgmhm-convergence.pdf).

Fernando et al. (2023), theorem 6: smooth energy; ell>=1, k>=ell+d; pressure order three observed. The [source numerical record](https://github.com/ipes-lncc/pymhm/blob/main/examples/results/pgmhm/comparison.json) has SHA256 `abb1407a479551db45a43b66ddf6b37830679121a2cceef612cfca6929da1206`. Its execution attribution remains in that record. The [rate notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/convergence/method_rates.ipynb) recomputes the orders; reading existing measurements does not execute the underlying PDE.

## References for this method

- Honório Fernando, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2023). *A Petrov–Galerkin multiscale hybrid-mixed method for the Darcy equation on polytopes*. Computational and Applied Mathematics 42, article 173. [DOI: 10.1007/s40314-023-02304-y](https://doi.org/10.1007/s40314-023-02304-y).
