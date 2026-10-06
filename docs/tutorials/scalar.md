# Scalar methods and local element choices

This tutorial solves small analytical problems through the public high-level
interfaces. Select a method and a local element family in its notebook cell; each
choice assembles its declared spaces and measures the scalar and available flux
errors separately. The examples exercise boundary conventions and physical
fields. They are analytical patches, not convergence studies or reproductions
of a paper's meshes and data.

The primal Darcy construction follows
[Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019).
The analytical patches below use explicitly declared data and discretizations.

The primary notebook cells declare local and global equations through the
generic variational interface. Their method-family comparisons retain the
predefined solvers. The [provider tutorial](providers.md) and
[variational guide](../variational.md) explain the form contract independently
of those comparisons.

Open the [problem notebook catalogue](../tutorials.md). Darcy examples are
`notebooks/darcy/primal_galerkin.ipynb`, `mixed_hdiv.ipynb` and
`hybrid_methods.ipynb`; transport and Helmholtz have their own problem folders.
The notebooks expose the same analytical patch choices through `selected_methods`.

```bash
pixi run --locked -e notebooks notebooks-run darcy/primal_galerkin.ipynb
pixi run --locked -e notebooks notebooks-run darcy/mixed_hdiv.ipynb
pixi run --locked -e notebooks notebooks-run darcy/hybrid_methods.ipynb
```

The default patches use two macrotriangles, two macrorectangles, six
macrotetrahedra or two macroprisms on a unit square/cube. The `refinement` argument changes
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
from pymhm import TriangleMesh
from pymhm._legacy.models.darcy.mixed_bdm import solve_darcy_bdm
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

| Notebook choice | Public solver and local space | Macroface variable |
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

| Notebook choice | Executed fine-cell flux | Pressure/divergence | Normal trace |
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
[mixed-well documentation](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mixed-well-geometries.md)
gives the explicit tensor factors, Piola convention and physical mean gauge.

```python
from pymhm import AffineMixedMesh
from pymhm._legacy.models.darcy.hdiv_3d import solve_darcy_hdiv3d
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

| Notebook choice | Local approximation and convention |
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
[MH](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mh.md),
[MH2M](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mh2m.md),
[MsHHO](../cases/mshho.md) and
[PGMHM](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/pgmhm.md),
including their additional hypotheses.

The corresponding constructions are described by
[Barrenechea, Gomes and Paredes (2024)](https://doi.org/10.1137/22M1542556) for MH,
[de Barros, Madureira and Valentin (2026, version 3)](https://arxiv.org/abs/2404.16978v3)
for MH²M,
[Chaumont-Frelet et al. (2022)](https://doi.org/10.1051/m2an/2021082) for MsHHO,
and [Fernando et al. (2023)](https://doi.org/10.1007/s40314-023-02304-y) for PGMHM.

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

The conservative RAD construction follows
[Harder, Paredes and Valentin (2015)](https://doi.org/10.1137/130938499), while
the scalar negative-residual stabilization follows
[Santiago, Valentin and Martins (2025)](https://doi.org/10.55592/cilamce2025.v5i.14270).

`helmholtz` uses $\rho=\kappa=1$, $\omega=1/2$, no absorbing faces and
complex pressure $p=(1+i)(1+x+2y)$. Its independently differentiated source is
$f=-\omega^2p$. The skeleton has two real components per complex mode; it
uses the package's interleaved real convention. Positive frequency introduces
no pressure gauge. At unit density, the measured complex gradient error equals
the physical flux error. A successful low-frequency patch does not certify
wave-resolution or resonance conditions for another frequency. See
[Helmholtz](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/helmholtz.md).

The acoustic formulation is developed by
[Chaumont-Frelet and Valentin (2020)](https://doi.org/10.1137/19M1255616);
the affine low-frequency patch is this tutorial's separate analytical control.

## Read the output and move to providers

`scalar_l2` and `flux_l2` are physical field errors against the analytical data.
Assembly uses quadrature order 6 and these measurements use order 8. Separate
keys report absolute macro balances, fine pressure-test moments, normal-flux
moments and original local equation defects when the public result supplies
those diagnostics. The 3D mixed result also reports separate relative
constitutive, divergence and prescribed-normal-flux residuals.
`algebraic_relative_residual` is a reduced-system diagnostic, not a combined
L2 field residual or a stability certificate. Missing diagnostics are omitted.

Selecting a notebook method chooses its stated predefined discretization.
For a common explicit provider interface, continue with
[local providers and execution](providers.md): the notebook declares primal
or RT0 `LocalEquations` and an additional global `Equation`, then uses
`MultiscaleProblem` and `assemble`. The provider owns
the actual spaces, local kernel, physical moments and oriented trace maps;
the shared assembler owns ordered global reduction. MH2M and MsHHO retain
their dedicated global trace/moment contracts. Native element replacement
must preserve the executed basis and its coefficient mapping, including the
basis digest when coefficient vectors are persisted.

## References

- Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *A family of Multiscale Hybrid-Mixed finite element methods for the Darcy equation with rough coefficients*, Journal of Computational Physics 245, 107–130. [DOI: 10.1016/j.jcp.2013.03.019](https://doi.org/10.1016/j.jcp.2013.03.019).

- Omar Durán, Philippe R. B. Devloo, Sônia M. Gomes, and Frédéric Valentin (2019). *A multiscale hybrid method for Darcy’s problems using mixed finite element local solvers*, Computer Methods in Applied Mechanics and Engineering 354, 213–244. [DOI: 10.1016/j.cma.2019.05.013](https://doi.org/10.1016/j.cma.2019.05.013).

- Gabriel R. Barrenechea, Antonio Tadeu A. Gomes, and Diego Paredes (2024). *A Multiscale Hybrid Method*. SIAM Journal on Scientific Computing 46(3), A1628–A1657. [DOI: 10.1137/22M1542556](https://doi.org/10.1137/22M1542556).

- Franklin de Barros, Alexandre L. Madureira, and Frédéric Valentin (2026). *A three-field Multiscale Method*. arXiv preprint, version 3, 5 August 2026; first submitted 25 April 2024. [arXiv: 2404.16978v3](https://arxiv.org/abs/2404.16978v3).

- Théophile Chaumont-Frelet, Alexandre Ern, Simon Lemaire, and Frédéric Valentin (2022). *Bridging the Multiscale Hybrid-Mixed and Multiscale Hybrid High-Order Methods*, ESAIM: M2AN 56, 261–285. [DOI: 10.1051/m2an/2021082](https://doi.org/10.1051/m2an/2021082).

- Honório Fernando, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2023). *A Petrov–Galerkin multiscale hybrid-mixed method for the Darcy equation on polytopes*. Computational and Applied Mathematics 42, article 173. [DOI: 10.1007/s40314-023-02304-y](https://doi.org/10.1007/s40314-023-02304-y).

- Christopher Harder, Diego Paredes, and Frédéric Valentin (2015). *On a Multiscale Hybrid-Mixed Method for Advective-Reactive Dominated Problems with Heterogeneous Coefficients*, Multiscale Modeling & Simulation 13(2), 491–518. [DOI: 10.1137/130938499](https://doi.org/10.1137/130938499).

- Juan Felipe Pacazuca Santiago, Frédéric Valentin, and Larissa Martins (2025). *A Multiscale Hybrid-Mixed Method with Local Stabilization*. Proceedings of the Ibero-Latin American Congress on Computational Methods in Engineering, CILAMCE 2025, volume 5, article 14270; published online 18 March 2026. [DOI: 10.55592/cilamce2025.v5i.14270](https://doi.org/10.55592/cilamce2025.v5i.14270).

- Théophile Chaumont-Frelet, and Frédéric Valentin (2020). *A Multiscale Hybrid-Mixed Method for the Helmholtz Equation in Heterogeneous Domains*. SIAM Journal on Numerical Analysis 58(2) 1029-1067. [DOI: 10.1137/19M1255616](https://doi.org/10.1137/19M1255616).
