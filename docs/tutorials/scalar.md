# Scalar methods and local element choices

This tutorial solves small analytical problems through the public high-level
interfaces. Select a method and a local element family with `--variant`; each
choice assembles its declared spaces and measures the scalar and available flux
errors separately. The examples exercise boundary conventions and physical
fields. They are analytical patches, not convergence studies or reproductions
of a paper's meshes and data.

```bash
pixi run --locked -e test python -m examples.tutorial_scalar_variants --variant all
pixi run --locked -e test python -m examples.tutorial_scalar_variants --variant bdm-plus
pixi run --locked -e test python -m examples.tutorial_scalar_variants \
  --variant primal-neumann --output build/tutorials/neumann.json
```

The default patches use two macrotriangles, two macrorectangles, six
macrotetrahedra or two macroprisms on a unit square/cube. `--refinement` changes
the local edge subdivisions independently of the unsplit macroface space.
Tetrahedral and mixed 3D subdivisions must be powers of two. For
`classical-rt1`, this argument instead subdivides the globally conforming mesh;
there is no MHM skeleton. The defaults keep every example small.

## Start with pressure and physical Darcy flux

For identity permeability, the two-dimensional patch is

$$
p(x,y)=1+x+2y,\qquad q=-\nabla p=(-1,-2),\qquad \nabla\cdot q=0.
$$

In three dimensions it is

$$
p(x,y,z)=1+x+2y+3z,\qquad q=(-1,-2,-3),\qquad \nabla\cdot q=0.
$$

The boundary pressure is nonhomogeneous. The `primal-neumann` choice instead
prescribes the outward physical density $q\cdot n$ on every exterior face and
fixes the volume mean of pressure to $2.5$. Dirichlet problems require no added
pressure gauge. The `primal-quadratic` choice tests a nonzero independently
differentiated source:

$$
p=1+x^2+2y^2,\qquad q=(-2x,-4y),\qquad f=-6.
$$

A direct public solve contains the same information as the corresponding
tutorial choice:

```python
from pymhm import TriangleMesh, solve_darcy_bdm
from examples.tutorial_scalar_variants import affine_pressure, affine_flux

solution = solve_darcy_bdm(
    TriangleMesh.unit_square(),
    degree=1,
    enrichment=1,
    local_refinement=1,
    quadrature_order=6,
    dirichlet=affine_pressure,
    source=0.0,
)
print(solution.l2_error(affine_pressure, order=8))
print(solution.flux_l2_error(affine_flux, order=8))
```

Here `degree` controls the fine-edge normal degree; `enrichment` changes
interior bubbles and the pressure/divergence space. It does not increase the
normal degree. Macroface segmentation and polynomial degree are independent
parameters, supplied through a scalar `SkeletonSpace`.

## Primal Galerkin and mixed H(div) locals

$P_k$ denotes complete total-degree polynomials; $Q_k$ denotes tensor-product
polynomials. Continuity of a primal local pressure applies inside one
macrocell. Different macrocells retain independent reconstructed values.

| CLI choice | Public solver and local space | Macroface variable |
| --- | --- | --- |
| `primal`, `primal-neumann` | `solve_darcy`, continuous $P_2$ pressure inside each macrocell | $P_0$ physical normal flux |
| `primal-quadratic` | `solve_darcy`, continuous $P_2$ pressure | $P_1$ physical normal flux |
| `primal-rectangle` | `solve_darcy_quadrilateral`, continuous $Q_2$ pressure on Cartesian rectangles | $P_0$ physical normal flux |
| `primal3d` | `solve_darcy_3d`, continuous tetrahedral $P_2$ pressure | $P_0$ physical normal flux |
| `rt0`, `rt1` | `solve_darcy_rt`, $\mathrm{RT}_m/P_m$, $m=0,1$ | $P_m$ physical normal flux |
| `classical-rt1` | `solve_darcy_rt_conforming`, globally conforming $\mathrm{RT}_1$ flux and discontinuous $P_1$ pressure | No MHM skeleton |
| `bdm` | `solve_darcy_bdm`, $\mathrm{BDM}_2/P_1$ | $P_1$ physical normal flux |
| `bdm-plus` | Normal degree $k=1$, all zero-normal $\mathrm{BDM}_2$ bubbles, pressure $P_1$ | $P_1$ physical normal flux |
| `bdm-double-plus` | Normal degree $k=1$, all zero-normal $\mathrm{BDM}_3$ bubbles, pressure $P_2$ | $P_1$ physical normal flux |
| `tensor-rt` | `solve_darcy_tensor_rt`, rectangular $\mathrm{RT}_1/Q_1$ | $P_1$ physical normal flux |
| `tensor-rt-plus` | Rectangular normal degree $k=0$, all zero-normal $\mathrm{RT}_1$ bubbles, pressure $Q_1$ | $P_0$ physical normal flux |

The returned primal volume flux is the raw field $-K\nabla p_h$; it generally
does not belong to $H(\mathrm{div})$. RT/BDM/tensor RT volume fluxes are
conforming in $H(\mathrm{div})$ inside each local mesh. The macro skeleton
controls their exterior normal moments. Complete fine-face normal moments
recover the classical conforming mixed discretization; a coarser skeleton is
a declared restriction. A classical fine solution still requires its own
refinement checks before serving as a numerical reference.

The affine flux belongs to every listed mixed flux space. RT0 pressure is
piecewise constant, so its scalar error is nonzero: for the default two
triangles it is $\sqrt{7/18}$. The field stores the exact cell averages, while
the physical constant flux is exact. The other default pressure spaces contain
the affine pressure.

The mixed-local constructions follow the face/interior decomposition of
[Durán, Devloo, Gomes and Valentin (2019)](https://doi.org/10.1016/j.cma.2019.05.013),
sections 4.1–4.2: retain the specified face modes, retain every selected
zero-normal bubble, and match the pressure space to the divergence image.
Aligned, affine, shape-regular fine partitions and the stable-pair hypotheses
remain part of the corresponding approximation theory. A passing patch or
rank check does not establish a uniform inf-sup constant. Mathematical
$\mathrm{RT}_m$ uses native Basix RT degree `m+1`; native and executed bases
also require an explicit moment/orientation transformation.

## Restricted tetrahedral and prismatic mixed families

These choices use `solve_darcy_hdiv3d` on `AffineMixedMesh.unit_cube`.
They expose the concrete spaces used by the mixed-local framework associated
with the Unicamp references; a degree label alone does not identify a space.

| CLI choice | Executed fine-cell flux | Pressure/divergence | Normal trace |
| --- | --- | --- | --- |
| `hdiv-tetra` | $\{v\in[P_2]^3:v\cdot n\vert_F\in P_1(F)\}$, dimension 18 | $P_1$ | $P_1$ on triangles |
| `hdiv-tetra-plus` | $P_1$ face modes and all zero-normal $[P_3]^3$ bubbles, dimension 32 | $P_2$ | $P_1$ on triangles |
| `hdiv-prism` | Restricted horizontal/vertical tensor construction, dimension 27 | $W_{1,1}=P_1(x,y)\otimes P_1(z)$ | $P_1$ on triangles; $Q_1$ on rectangles |

The tetrahedral spaces are restricted/enriched constructions rather than the
full native BDM parents. The prism construction is not the larger space
defined only by normal and divergence constraints: that larger space contains
one additional solenoidal bubble. These examples check the implemented spaces;
they do not reproduce the historical well geometry or resolve ambiguous BDFM
order labels in the paper. The
[mixed-well documentation](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-well-geometries.md)
gives the explicit tensor factors, Piola convention and physical mean gauge.

```python
from pymhm import AffineMixedMesh, solve_darcy_hdiv3d
from examples.tutorial_scalar_variants import affine_pressure, affine_flux

solution = solve_darcy_hdiv3d(
    AffineMixedMesh.unit_cube(kind="tetrahedron"),
    pressure_degree=2,
    normal_degree=1,
    trace_degree=1,
    subdivisions=1,
    local_refinement=1,
    dirichlet=affine_pressure,
)
print(solution.errors(affine_pressure, affine_flux))
```

## Other global scalar formulations

The following methods retain their distinct variational forms and unknowns.
Their shared use of local linear algebra does not make their skeleton variables
interchangeable.

| CLI choice | Local approximation and convention |
| --- | --- |
| `mh-robin`, `mh-robin3d` | $P_2$ Robin locals with $\nu=0.1$ and $\sigma=\nu(x-x_0)/d$; the $P_1$ multiplier is $(q-p\sigma)\cdot n$, rather than physical $q\cdot n$ |
| `mh2m`, `mh2m3d` | $P_2$ locals, continuous pressure trace $\Gamma=P_1$ and independent outward conormal $\Lambda=P_0$; conormal is $K\nabla p\cdot n=-q\cdot n$ |
| `mshho`, `mshho3d` | $P_2$ energy reconstructions, cell moments $m=0$, $P_0$ face-pressure moments and projected source |
| `mshho-face` | Cell degree $m=-1$ and reconstructed source, as required by the face-only formulation |
| `pgmhm` | Petrov-Galerkin $P_3/P_1$ configuration with $\alpha=0.1$, reporting the enriched pressure and conservative face flux |

Robin MH uses its explicit coercivity bound on $\nu$; the selected unit-cell
value is admissible. The 3D option is a documented dimensional extension.
MH2M requires compatible fine/trace partitions and injective local Neumann
maps; numerical rank acceptance does not replace the source theorem's mesh
conditions. MsHHO requires independent represented cell/face moments and
uses its own moment assembly. PGMHM here satisfies $k\geq\ell+2$ in two
dimensions with $k=3$, $\ell=1$; the theorem's sufficiently-small-$\alpha$
condition is not a universal numerical threshold. Its raw volume gradients
remain distinct from its conservatively enriched face flux.

The dedicated pages describe
[MH](https://github.com/volpatto/pymhm/blob/main/docs/cases/mh.md),
[MH2M](https://github.com/volpatto/pymhm/blob/main/docs/cases/mh2m.md),
[MsHHO](../cases/mshho.md) and
[PGMHM](https://github.com/volpatto/pymhm/blob/main/docs/cases/pgmhm.md),
including their additional hypotheses.

## Reaction, advection and complex acoustics

`rad` and `rad-supg` solve the conservative scalar operator

$$
-\Delta u+\nabla\cdot(\beta u)+cu=f,\qquad
\beta=(1/4,-1/2),\quad c=1/2,\quad
f=-3/4+\frac12(1+x+2y).
$$

The solution is $u=1+x+2y$. Both use $P_2$ local fields and the $P_1$
half-advection Robin multiplier $(-\nabla u+\beta u/2)\cdot n$.
`rad-unusual` uses zero advection, $c=1/2$ and $f=cu$, with the
reaction-diffusion UNUSUAL stabilization. The tutorial reports scalar L2 errors
for these three choices; it does not infer physical flux balance from their
algebraic residual. See the [transport conventions](../api/transport.md).

`helmholtz` uses $\rho=\kappa=1$, $\omega=1/2$, no absorbing faces and
complex pressure $p=(1+i)(1+x+2y)$. Its independently differentiated source is
$f=-\omega^2p$. The skeleton has two real components per complex mode; it
uses the package's interleaved real convention. Positive frequency introduces
no pressure gauge. At unit density, the measured complex gradient error equals
the physical flux error. A successful low-frequency patch does not certify
wave-resolution or resonance conditions for another frequency. See
[Helmholtz](https://github.com/volpatto/pymhm/blob/main/docs/cases/helmholtz.md).

## Read the output and move to providers

`scalar_l2` and `flux_l2` are physical field errors against the analytical data.
Assembly uses quadrature order 6 and these measurements use order 8. Separate
keys report absolute macro balances, fine pressure-test moments, normal-flux
moments and original local equation defects when the public result supplies
those diagnostics. The 3D mixed result also reports separate relative
constitutive, divergence and prescribed-normal-flux residuals.
`algebraic_relative_residual` is a reduced-system diagnostic, not a combined
L2 field residual or a stability certificate. Missing diagnostics are omitted.

Changing `--variant` switches the supported high-level local/method choice.
For a common explicit provider interface, continue with
[local providers and execution](providers.md): `darcy_local_provider` feeds
primal or RT0 `LocalAssembly` records into the same
`HybridProblem(GlobalForm(...), provider, items)` contract. The provider owns
the actual spaces, local kernel, physical moments and oriented trace maps;
the shared assembler owns ordered global reduction. MH2M and MsHHO retain
their dedicated global trace/moment contracts. Native element replacement
must preserve the executed basis and its coefficient mapping, including the
basis digest when coefficient vectors are persisted.
