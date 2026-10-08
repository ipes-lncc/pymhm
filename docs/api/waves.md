# Wave and transient equations

The wave and transport tutorials declare the mathematical operators through
`Equation`, `LocalEquations` and `MultiscaleProblem`. Their time loops use
generic assembly, solution, field recovery and explicit factor lifetimes.
The importable providers are application examples whose formulas can be
inspected and changed by a user; the package primitives select no physical
method or time integrator.

| Executable application | Provider | Declared spaces and inputs |
| --- | --- | --- |
| Frequency-domain acoustics | `examples/tutorial_helmholtz_equations.py` | Continuous triangular Pk or Cartesian Qk pressure; polynomial or oscillatory normal traces; complex source and boundary data, Dirichlet, physical Neumann, impedance, Cartesian PML and shared discrete point sources |
| Maxwell leapfrog | `examples/tutorial_maxwell_equations.py` | DG scalar electric/vector magnetic fields for 2D TM, vector fields for 3D; continuous local material integration, canonical tangential traces, PEC or impedance moments, time-dependent forcing and boundary data |
| Elastic Newmark | `examples/tutorial_elastodynamic_equations.py` | Continuous Pk displacement on triangles or tetrahedra; scalar density, Lamé or Kelvin/cartesian constitutive data, displacement and physical traction moments, independently declared local substep counts |
| Backward-Euler heat and conservative transport | `examples/formulations/transient.py` | Triangular continuous Pk fields; positive capacity, tensor diffusion, conservative variable velocity, Galerkin or SUPG, weak/strong Dirichlet, Robin or diffusive flux, macro-local coefficients and supplied conforming fine partitions |
| Darcy-driven passive transport | `examples/formulations/darcy_transport.py` | The executed RT0, RT/BDM or reconstructed vector field, or an explicitly selected raw primal field with numerical boundary normals; hydrodynamic dispersion and the preceding time equations on the same fine partitions |

These providers retain the approximation spaces and physical conventions of
the executed applications. Arbitrary polynomial degree does not by itself
establish trace compatibility, uniqueness, stability or a convergence rate.
The demonstrated analytical patches and original coefficient comparisons are
separate from matched literature reproductions.

## Frequency-domain acoustics

For complex pressure, normal-flux coordinates and volume forcing, a cell
declares the literal coefficient equations

$$
\begin{aligned}
(A_K-\omega^2M_K-\mathrm{i}\omega R_K)p_K+B_K\lambda_K&=f_K,\\
-B_K^{\mathsf T}p_K&=-G_K.
\end{aligned}
$$

`volume_forms`, `edge_rule` and the executed trace basis define these blocks.
The physical flux is minus density-inverse times the pressure gradient.
Dirichlet data supply weak pressure moments in the second equation; physical
Neumann data fix normal trace coefficients. Impedance contributes to the first
equation and leaves zero exterior trace placeholders. PML changes the volume
operator and does not imply positive definiteness. Two-dimensional Dirac
sources are discrete point functionals and do not imply a finite continuum
energy norm.

`realify_operator`, `realify_vector` and `complexify_vector` are generic exact
coordinate maps. Real and imaginary components are interleaved. Rectangular,
complex and nonsymmetric operators retain their literal action; the caller
supplies any conjugation needed by a sesquilinear form.

## Maxwell stages

With electric mass matrix, magnetic mass matrix, the declared discrete curl
and tangential pairing, a magnetic update and an electric midpoint update are

$$
\begin{aligned}
M_HH^{n+1}&=M_HH^n+\Delta t\,CE^{n+1/2},\\
M_Ew+\frac{\Delta t}{2}Q\lambda
&=M_EE^{n+1/2}+\frac{\Delta t}{2}(F-C^{\mathsf T}H^{n+1}),\\
-Q^{\mathsf T}w+Z\lambda&=-b,\\
E^{n+3/2}&=2w-E^{n+1/2}.
\end{aligned}
$$

The initial electric kick uses half the time increment. The provider declares
all four local/global blocks independently: the midpoint local coupling is
scaled by time, while its global test pairing is the unscaled negative
transpose of Q. `TangentialTraceSpace` supplies the actual normal and tangent
orientation. PEC and impedance are constraints on the declared tangential
moments; they are not pointwise constraints on every DG boundary value.

`EquationLeapfrog` in the example owns the generic factor caches for these
unchanged equations. Energy uses the explicit staggered cross term. Its
positivity depends on the chosen time increment and spatial spectrum; a small
stage residual alone does not establish wave accuracy.

## Newmark displacement and velocity

`newmark_step` is a generic mass/stiffness operation. The application declares
its endpoint displacement response and global displacement moments as
`LocalEquations`. The recovered velocity uses its separate endpoint response.
Each cell can take its specified number of local substeps, and forcing is
integrated at each actual substep endpoint. A replay of the original local
Newmark equations checks the endpoint reconstruction with the computed physical
traction. Nonmatching local substeps do not inherit a uniform-step energy
identity automatically.

Density-weighted mass projection initializes displacement and velocity.
`stress_from_gradient` interprets symmetric strain using the stated 2D or 3D
Kelvin convention. The actual constitutive tensor and density determine the
operators; raw displacement gradients and physical traction remain distinct
from skeletal coordinates.

## Capacity and conservative time residuals

Backward Euler declares a step coefficient and a source pairing

$$
\begin{aligned}
c_{\Delta t}&=c+\frac{\rho}{\Delta t},\\
L_{\Delta t}(v)&=(f,v)+\frac{1}{\Delta t}(\rho u_{\mathrm{old}},v).
\end{aligned}
$$

The conservative SUPG application tests the complete residual with

$$
\psi_i=\phi_i+\tau\,\beta\mathbin{\cdot}\nabla\phi_i.
$$

The same executed tests integrate both forcing and the previous-state capacity
term. Tau uses the full step reaction including capacity divided by the time
increment. The public `assemble_load` and `assemble_mass` integrate literal
sampled test/trial arrays with supplied physical weights and integer maps.
They support different test/trial counts, complex coefficients and repeated
coordinates, without assuming positivity or selecting a time scheme.

The conservative multiplier is the normal component of
minus the diffusion tensor times the gradient plus half the advective field.
Physical diffusive-flux faces add the corresponding half-advection boundary
mass. Strong Dirichlet data add exact local nodal constraints and local boundary
reactions. Physical capacity integrals use the original trial functions rather
than stabilized tests. Their balance is a discrete macrocell weak balance,
not a claim of fine-cell conservation or a maximum principle.

`MacroCoefficient` associates one-sided coefficient callbacks with the actual
macro partition. `PolynomialVectorField` and `RT0VectorField` evaluate the
supplied physical vector polynomial without smoothing independent sides.
`HydrodynamicDispersion` supplies the 2D material tensor and its broken
derivative. The raw primal/numerical-normal pair is supported with Galerkin
transport; it is not asserted to be H(div) and is excluded from strong-residual
stabilization.

## Source reuse and direct coefficient checks

`OfflineMultiscaleSystem` reuses the literal operators and executed lifts for
new volume and interface loads. `with_loads` produces an ordinary updated
system; global factors are owned explicitly and passed through the generic
`factorization` option. Returned coefficient systems remain reconstructible
after caches close.

`assemble_original_blocks` independently scatters the declared A/B/C/D rows and
loads into the full original coefficient matrix. It uses no harmonic lift or
Schur complement. Its present layout covers leaf equations with no retained
modes; recursive layouts and retained-mode gauges require an explicitly
declared original layout and are rejected. The wave notebooks compare the
condensed result with this direct original solve on identical operators,
spaces, data and quadrature. This verifies elimination and reconstruction;
it is not a classical reference on a separately refined mesh.
