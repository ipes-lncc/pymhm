# User-defined variational problems

`Equation`, `LocalEquations` and `MultiscaleProblem` describe forms and their
coefficient maps. `assemble` and `solve` supply compilation, constrained local
elimination, ordered global assembly and reconstruction. Scalar, vector and
mixed fields use the same records. The physical operator, boundary convention,
trace space and retained modes belong to the problem definition.

For the abstract MHM decomposition and local/global interpretation, see
[Harder and Valentin (2016)](https://doi.org/10.1007/978-3-319-41640-3_13).
The four-block API below also accepts independently declared trial/test forms.

The current contract is a linear system with real coefficients. Nonlinear
iterations and time integration can compose these systems; their update rules
and scientific qualification belong to the application.

The [provider notebooks](tutorials/providers.md) give executable scalar and
mixed examples. The [API](api/hybrid.md) gives the complete signatures.
The case gallery also documents predefined physical formulations imported from
their implementation owners. Their numerical qualification applies to the
stated data and spaces, rather than arbitrary variational forms.

## Declare both equations

A provider returns the following four blocks for each local item:

$$
\begin{aligned}
a_K(u_K,v_K)+b_K(\lambda_K,v_K)&=L_K(v_K),\\
c_K(u_K,\mu_K)+d_K(\lambda_K,\mu_K)&=g_K(\mu_K).
\end{aligned}
$$

Their coefficient representation is

$$
\begin{aligned}
A_Ku_K+B_K\lambda_K&=f_K,\\
C_Ku_K+D_K\lambda_K&=g_K.
\end{aligned}
$$

Rows are test coordinates and columns are trial coordinates. `b` and `c`
are independent: the package does not infer an adjoint, sign or physical
boundary variable. `dofs` maps the columns of $B_K$ to global trace
coordinates; `test_dofs` maps the rows of $C_K$. The maps can differ in order
and length. `d` has shape `(len(test_dofs), len(dofs))` and `g` has length
`len(test_dofs)`. Omitting `test_dofs` uses `dofs`.

```python
from pymhm import Equation, LocalEquations, MultiscaleProblem, solve

def local(cell):
    return LocalEquations(
        a=[[2.0]], L=[1.0], b=[[1.0]], c=[[-1.0]],
        d=[[1.0]], g=[0.0], dofs=[0],
    )

problem = MultiscaleProblem(
    global_equation=Equation(0, 0),
    local_provider=local,
    items=[0],
    trace_size=1,
    coarse_sizes=(0,),
)
solution = solve(problem)
assert abs(solution.trace[0] - 1 / 3) < 1e-14
assert abs(solution.fields[0][0] - 1 / 3) < 1e-14
```

This is an algebraic illustration, not a discretization study. For invertible
$A_K$, local elimination contributes

$$
\begin{aligned}
S_K&=D_K-C_KA_K^{-1}B_K,\\
r_K&=g_K-C_KA_K^{-1}f_K.
\end{aligned}
$$

The implementation factors the local operator once for the source and coupling
columns. It uses the same checked condensation owner as the established
formulations. A global `Equation(a_G, L_G)` adds an independent global operator
and load after all local contributions. It can contain cross-cell couplings,
boundary terms or an already assembled facet operator.

If $P_K$ gathers a cell's trial coordinates and $Q_K$ gathers its test
coordinates, the reduced global equation is

$$
a_G(x,y)+\sum_K(Q_Ky)^TS_K(P_Kx)
=L_G(y)+\sum_K(Q_Ky)^Tr_K.
$$

The additional global matrix and vector use the complete reduced coordinate
order: all trace coefficients first, then each item's retained coefficients
in `items` order. They must have that full shape. `Equation(0, 0)` denotes
zero additions with dimensions supplied by the problem.

The assembled system is a data object. `solve_multiscale_system(system)`
solves its declared boundary and gauge problem;
`reconstruct_multiscale(system, coordinates)` recovers fields from executed
coefficients without another global solve. Its `solve` and `reconstruct`
methods delegate to those free functions. `with_global_load(system, rhs)`
reuses the local responses for a new global balance load while preserving
retained source-compatibility rows. It does not change a local volume source.

## Use UFL or assembled blocks

`a`, `L`, `b`, `c`, `d`, `g` and the global `Equation` accept assembled
arrays or sparse operators. `compile_form` also accepts real linear and
bilinear UFL forms through the optional DOLFINx adapter. A scalar zero obtains
its dimensions from the surrounding block; nonzero scalars are not broadcast.

For example, on a declared scalar space `V`, a reaction–diffusion volume form
can be written directly:

```python
import ufl
from pymhm import LocalEquations, columns, rows

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
dx = ufl.Measure("dx", domain=local_mesh, metadata={"quadrature_degree": 6})
ds = ufl.Measure("ds", domain=local_mesh)
a = (ufl.inner(K * ufl.grad(u), ufl.grad(v)) + reaction * u * v) * dx
L = source * v * dx

local = LocalEquations(
    a=a,
    L=L,
    b=columns(*(phi * v * ds for phi in signed_trace_functions)),
    c=rows(*(-phi * u * ds for phi in signed_trace_functions)),
    dofs=trace_dofs,
)
```

`V`, coefficients, measures, integration regions and the signed trace
functions are supplied by the provider. Each `phi` represents one explicitly
chosen trace basis function on the local integration mesh, including its
support and orientation. `columns` assembles test-side linear forms as $B_K$;
`rows` assembles trial-side linear forms as $C_K$. There is no automatic
identification of basis functions on another mesh.

The scalar example has no declared kernel. If `reaction=0` and the local
volume operator is a Neumann diffusion operator, provide its constant
coefficients and physical mean moments as described below.

Vector and mixed UFL forms follow the same contract. For a mixed velocity and
pressure space `W`, one possible symmetric Brinkman volume form is

```python
u, p = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)
a = (
    2 * viscosity * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v)))
    + ufl.inner(resistance * u, v)
    - p * ufl.div(v) - q * ufl.div(u)
) * dx
L = ufl.inner(body_force, v) * dx
```

This declares the volume operator. The trace may represent physical traction
or a method-specific multiplier according to the separately declared boundary
forms. Stable velocity/pressure spaces, local kernels, compatible data and the
physical pressure gauge still have to be supplied. The API does not choose
Taylor–Hood, stabilization or a boundary prescription from the field names.

Mixed $H(\mathrm{div})\times L^2$ volume forms can similarly use
`inner(K_inverse * flux, test_flux)`, divergence pairings and a pressure source.
Flux-prescribing MHM also needs its boundary variables and normal-trace
constraints. A mixed volume form alone does not define that construction.

### Native compilation limits

`pymhm.backends.forms.assemble_form` accepts real rank-one and rank-two UFL
forms on single-rank DOLFINx domains, normally `MPI.COMM_SELF`. Bilinear
forms may have distinct test and trial spaces and rectangular shapes; the
chosen local pivot in `LocalEquations` must nevertheless be square. A declared
0-by-0 pivot has no eliminated field coordinates: its `d` and `g` contribute
directly without a local factorization. Linear forms with UFL argument number
zero or one are supported.
The complete reduced global operator is also square.

The adapter assembles binary64 coefficients. It infers no essential boundary
elimination, gauge, normal orientation, quadrature or transpose relation.
Cross-mesh integration requires explicit DOLFINx `entity_maps`; use
`assemble_form` or a custom compiler to pass these maps and compilation options.
Zero forms whose arguments were simplified away require an explicit shape.
Complex-valued blocks, nonzero rank-zero functionals and higher-rank forms are
outside this compiler contract. Declare complex formulations in explicit real
and imaginary coordinates, as in the
[Helmholtz notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/waves/helmholtz/introductory_methods.ipynb);
the compiler assembles the resulting real blocks.

## Retain local modes and physical constraints

For a singular local operator, declare `kernel=Z` and its physical column
pairings `moments=C`. For distinct left modes, supply `left_kernel`,
`test_moments` and the explicit test-side pairing. These arrays use the actual
local coefficient basis. They are checked without rediscovering or rotating
the basis. A nonsingular moment pairing is required for the chosen modes.

`coarse_basis` can retain modes that need not be exact null vectors, with an
independent `test_basis` when needed. This uses the same complementary
elimination as the [shared hybrid contract](architecture.md#local-operators).
`kernel` and `coarse_basis` are mutually exclusive. `coarse_sizes` declares
the retained width of every item, including zero.

`fixed` prescribes explicit trace coefficients. `constraints` contains global
physical rows and values. With `assemble(problem)`, the returned system also
provides `mean_constraint` for physical integral weights in reconstructed
fields; then pass that row to `system.solve(constraints=[...])`. Integrals
must include the correct physical fields and exclude auxiliary multipliers.
No mean pressure, rigid-motion constraint or boundary datum is inferred.

These checks establish consistency with the represented algebraic contract.
They do not establish mesh-independent inf-sup stability, uniqueness of a
continuous PDE or a convergence theorem. Keep physical field errors and
original equation residuals separated by their field blocks in scientific
verification.

## Assemble serially or in bounded batches

```python
from pymhm import ExecutionConfig, assemble

system = assemble(
    problem,
    execution=ExecutionConfig(
        backend="process", workers=10, batch_size=10, native_threads=1,
    ),
)
solution = system.solve()
```

Serial execution builds, eliminates and contributes one item before taking the
next. Thread or spawn-process execution uses bounded batches. Only the
coordinator reduces shared global coordinates, in item order, so two cells
contributing to the same face do not concurrently mutate a global matrix.
Items, providers, compilers and callbacks must be picklable for spawn.
Executable process entry points use the usual `__main__` guard.

Create native meshes and forms inside the provider invocation. Compile and
release them there before returning `CompiledLocalEquations` with owned arrays
and portable metadata. Live native meshes, matrices and communicators must not
cross a worker boundary. `SolverConfig.local_solver` can use an external
numerical solver or response model, subject to the unchanged original-equation
checks. A learned field without the required operator, coordinates and response
columns does not satisfy that solver contract.

## Reuse the interface at finer scales

Set `LocalEquations.a` to another `MultiscaleProblem` when its condensed global
operator is the local operator at the parent scale. Its reduced coordinates
become the parent's local coefficient basis; `L` adds a parent-scale source
to the child's assembled load. Parent trace pairings and moment arrays use
those same coordinates.

Child assembly runs serially inside its owning worker, avoiding nested process
pools. The child leaves prescribed trace coefficients and physical gauges to
its parent: child `fixed` and `constraints` must be empty. Express required
local essential constraints as explicit local blocks. Parent coupling and
additional loads must leave the child's retained compatibility rows unchanged.

`NestedEquations` provides the equality-constrained boundary construction for
a child hybrid problem. With child equation $A x=f$, boundary selector $E$
and an explicit oriented restriction $M$, its local system is

$$
\begin{bmatrix}-A&E\\E^T&0\end{bmatrix}
\begin{bmatrix}x\\\mu\end{bmatrix}
+\begin{bmatrix}0\\-M\end{bmatrix}\lambda
=\begin{bmatrix}-f\\0\end{bmatrix}.
$$

`boundary_dofs` selects distinct child trace coordinates, excluding retained
cell coordinates. `trace_map` must be injective on the parent trace space and
contain every orientation and basis restriction. Declare kernels and moments
in the child's reduced coefficient basis; the shared nesting operation lifts
their reaction coordinates and extends moments by zero reaction weights.
The physical nesting moment is $-M^T\mu$. The shared hybrid saddle negates
that pairing in its reduced trace equations, so the four-block convention on
this page has $C[x;\mu]=+M^T\mu$. Global loads use that same row convention.
Reconstruction checks the child's actual load $f+E\mu$.
This construction also requires empty child `fixed` and `constraints`.

`MultiscaleSolution.children[cell]` contains a recovered child solution, or
`None` at a leaf. The hierarchy reconstructs against the actual parent-induced
load without solving the child's global system again. This is recursive
algebraic elimination; an error estimate for a particular multilevel method
requires its own space, regularity and compatibility hypotheses.

`leaf_moment(system, weights)` represents a physical integral in finest-scale
fields as `row @ coordinates + offset`. The callback supplies integral weights
in each leaf's executed coefficient basis. The operation includes source
offsets and pads boundary-reaction coordinates by zero. A prescribed physical
moment `value` therefore uses the global constraint `(row, value - offset)`.
It does not replace that integral by a mean of coarse coefficients or include
auxiliary reactions in a physical pressure gauge.

The [three-level coefficient notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/variational_hierarchy.ipynb)
checks operator recursion against an independently assembled full matrix.
The [spatial recursive MHM notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/darcy/36_recursive_mhm.ipynb)
declares Q2 leaf forms, P1 trace restrictions and `NestedEquations`, with a
physical leaf-integral gauge for Neumann data. Its live analytical checks are
separate from the archived convergence acquisitions and their provenance.

## Represent distinct method constructions

The interface does not select a method by name. These recipes identify the
blocks and data needed to express the published constructions; the existing
method implementations provide their separately verified cases.

| Construction | Required form and space choices |
| --- | --- |
| Neumann-local MHM — [Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019) | Local operator on the complement of its declared kernel; normal-flux trace, source lift, physical moments and retained compatibility equations. |
| Robin-local MH — [Barrenechea, Gomes and Paredes (2024)](https://doi.org/10.1137/22M1542556) | Robin terms inside the local operator and their matching global balance; the multiplier must be distinguished from physical flux. |
| Three-field MH²M — [de Barros, Madureira and Valentin (2026, v3)](https://arxiv.org/abs/2404.16978v3) | Local solution and duplicated conormal coefficients as mixed local fields; continuous pressure trace globally; complete source lifting and a boundary-mean local normalization. |
| MsHHO — [Chaumont-Frelet, Ern, Lemaire and Valentin (2022)](https://doi.org/10.1051/m2an/2021082) | Constrained energy reconstruction with declared cell and face moments; a local moment saddle and subsequent cell/face condensation, with the selected source projection. |
| Residual Petrov–Galerkin MHM — [Fernando, Martins, Pereira and Valentin (2023)](https://doi.org/10.1007/s40314-023-02304-y) | Independently declared trial/test pairings, enriched local responses and cross-cell macroface terms, with the published stabilization and reconstruction. |

For MsHHO, the reconstruction uses the mixed block

$$
\begin{bmatrix}A_K&C_K\\C_K^T&0\end{bmatrix}
\begin{bmatrix}u_K\\\eta_K\end{bmatrix}
=\begin{bmatrix}0\\m_K\end{bmatrix}.
$$

Here $C_K$ denotes the declared moment columns and $m_K$ the prescribed cell
and face moments, independently of the balance block notation above. The
energy matrix and subsequent elimination are part of this method's
construction. They are not obtained merely by changing the name of a flux
trace. The free function `energy_reconstruction(A, C)` supplies the moment
reconstruction $R$ and its reduction $R^T A R$ through the shared original
moment saddle. Its energy-minimization interpretation requires symmetry and
coercivity on the moment-zero space; nonsymmetric input preserves the full
bilinear reduction. `with_global_equation(system, equation)` adds forms after
the executed local response maps are available, so an explicitly assembled
jump of reconstructed fields can supply a Petrov facet term. It preserves
those response maps, bases and children in a new system. Automatic coupling
of reconstructed fields on different meshes, jump signs and penalties are
not inferred by `Equation`.

The literature imposes additional conditions. The MHM–MsHHO equivalence in
[Chaumont-Frelet, Ern, Lemaire and Valentin (2022)](https://doi.org/10.1051/m2an/2021082)
needs exact local solves and the prescribed polynomial source space; the
source-projected variant has a different equivalence statement. The original
equivalence also excludes the face-only $m=-1$ variant.

The three-field analysis of
[de Barros, Madureira and Valentin (2026, v3)](https://arxiv.org/abs/2404.16978v3)
requires its local injectivity and uniform norm bounds, not just an observed
full-rank matrix.

The residual Petrov–Galerkin error estimates of
[Fernando, Martins, Pereira and Valentin (2023)](https://doi.org/10.1007/s40314-023-02304-y)
include dimension-dependent enrichment and stabilization bounds: the stated
theorem uses trace degree $\ell\geq1$, local degree $k\geq\ell+d$ in dimension
$d$, and $0<\alpha\leq\alpha_0$. The admissible upper constant is part of the
analysis, not a numerical value inferred from a successful solve.

Mixed $H(\mathrm{div})$ methods require compatible divergence, pressure and
normal-trace spaces; see
[Durán, Devloo, Gomes and Valentin (2019)](https://doi.org/10.1016/j.cma.2019.05.013)
for the mixed-local multiscale construction. The
[primary literature and scope](literature.md) give the specific hypotheses and
the [case gallery](cases/index.md) gives executed evidence. None of these
theorems follows from the ability to assemble four blocks.

## References

- Christopher Harder and Frédéric Valentin (2016). *Foundations of the MHM Method*, in *Building Bridges: Connections and Challenges in Modern Approaches to Numerical Partial Differential Equations*, Lecture Notes in Computational Science and Engineering 114, Springer. [DOI: 10.1007/978-3-319-41640-3_13](https://doi.org/10.1007/978-3-319-41640-3_13).

- Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *A family of Multiscale Hybrid-Mixed finite element methods for the Darcy equation with rough coefficients*, Journal of Computational Physics 245, 107–130. [DOI: 10.1016/j.jcp.2013.03.019](https://doi.org/10.1016/j.jcp.2013.03.019).

- Gabriel R. Barrenechea, Antonio Tadeu A. Gomes, and Diego Paredes (2024). *A Multiscale Hybrid Method*. SIAM Journal on Scientific Computing 46(3), A1628–A1657. [DOI: 10.1137/22M1542556](https://doi.org/10.1137/22M1542556).

- Franklin de Barros, Alexandre L. Madureira, and Frédéric Valentin (2026). *A three-field Multiscale Method*. arXiv preprint, version 3, 5 August 2026; first submitted 25 April 2024. [arXiv: 2404.16978v3](https://arxiv.org/abs/2404.16978v3).

- Théophile Chaumont-Frelet, Alexandre Ern, Simon Lemaire, and Frédéric Valentin (2022). *Bridging the Multiscale Hybrid-Mixed and Multiscale Hybrid High-Order Methods*, ESAIM: M2AN 56, 261–285. [DOI: 10.1051/m2an/2021082](https://doi.org/10.1051/m2an/2021082).

- Honório Fernando, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2023). *A Petrov–Galerkin multiscale hybrid-mixed method for the Darcy equation on polytopes*. Computational and Applied Mathematics 42, article 173. [DOI: 10.1007/s40314-023-02304-y](https://doi.org/10.1007/s40314-023-02304-y).

- Omar Durán, Philippe R. B. Devloo, Sônia M. Gomes, and Frédéric Valentin (2019). *A multiscale hybrid method for Darcy’s problems using mixed finite element local solvers*, Computer Methods in Applied Mechanics and Engineering 354, 213–244. [DOI: 10.1016/j.cma.2019.05.013](https://doi.org/10.1016/j.cma.2019.05.013).
