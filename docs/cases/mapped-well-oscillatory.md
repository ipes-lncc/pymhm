# Oscillatory anisotropic flow around a producing well

This case uses the three-dimensional hexahedral RT1 MHM spaces of
[L05, §7.4, Problem 5](../literature.md). The reservoir, producer pressure data,
physical units and explicitly graded polygonal geometry are those described in
[the analytical well case](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mapped-well.md). The prescribed pressure trace is
continuous, while the discrete local mixed pressure is discontinuous.
Fluxes use the contravariant Piola map and share normal moments.

## Material and boundary conventions

With \(\delta=\pi/25\), the physical permeability is

$$
K=s\,\operatorname{diag}(\kappa_x,\kappa_y,\kappa_z),\qquad s=10^{-11}\;\mathrm{m}^2,
$$

$$
\begin{aligned}
\kappa_x&=10^{-1}\frac{2+1.8\sin(\delta xy)}{2+1.8\sin(\delta y)},\\
\kappa_y&=10^{-3}\frac{2+1.8\sin(\delta xy)}{2+1.8\sin(\delta x)},\\
\kappa_z&=10^{-3}\frac{2+1.8\sin(\delta z^2)}{2+1.8\sin(\delta z)}.
\end{aligned}
$$

Every trigonometric factor lies between 0.2 and 3.8. Thus each ratio lies in
\([1/19,19]\), and the tensor is uniformly positive definite without a numerical
floor. Its inverse is bounded and integrable. For the declared scale,
\(K_x\) lies between \(5.263\times10^{-14}\) and \(1.9\times10^{-11}\) m²;
\(K_y,K_z\) lie between \(5.263\times10^{-16}\) and
\(1.9\times10^{-13}\) m².


The factor \(10^{-11}\) matches the ranges shown in Figure 16. The printed
formula instead states \(10^{-3}\). This discrepancy is retained as an input
ambiguity, rather than treating a chosen convention as a unique reconstruction
of the historical calculation. At fixed pressure Dirichlet data and zero source,
multiplying the entire tensor by a constant leaves pressure unchanged and
multiplies flux by that constant. A direct discrete test verifies this identity;
it separates the scale ambiguity from the spatial heterogeneity.

The viscosity is 0.001 Pa·s, so the constitutive operator uses \(K/\eta\).
Analytical *constant-permeability* producer pressures are imposed at both
lateral walls, using the positive extraction convention stated in the analytical
case. The top and bottom have zero outward flux, and the volume source is zero.
The heterogeneous total well rate is an output of this boundary-value problem;
it is not constrained to the rate parameter used to define the pressure data.

## Spatial and quadrature comparisons

Three MHM macro levels share an F8 fine grid and retain the published RT1/Q1
local spaces. An additional F16 control holds the finest macro partition and
its coarse normal-trace space fixed while refining the local RT1/Q1 spaces.
A conforming classical PyMHM reference releases every fine normal
moment. Local condensation of larger blocks is an exact algebraic elimination
and does not impose a coarser trace restriction in that reference. Its own
refinement comparison is separate from the MHM-to-reference difference.

| MHM macro level | Macro hexahedra | Fine hexahedra | Trace coefficients | Coarse pressure coefficients |
|---:|---:|---:|---:|---:|
| 0 | 32 | 16,384 | 544 | 32 |
| 1 | 256 | 16,384 | 3,712 | 256 |
| 2 | 2,048 | 16,384 | 27,136 | 2,048 |
| 2, local F16 control | 2,048 | 131,072 | 27,136 | 2,048 |

These counts include prescribed boundary trace coefficients. They describe
the discrete spaces, not the number of unconstrained equations after boundary
elimination.

For these particular data, the exact solution is independent of height:
the horizontal tensor and lateral pressure data depend only on \(x,y\), the
vertical coefficient depends only on \(z\), and the caps are impermeable.
The unique coercive pressure problem therefore admits \(\partial_zp=0\) and
\(q_z=0\). The refined classical study retains the complete three-dimensional
RT1/Q1 operator while refining its reference grid as \((F,F,1)\). A separate
comparison with vertical refinement verifies this invariance. The larger
classical references use its exact invariant subspace: twelve horizontal RT1
modes and four height-independent Q1 pressure modes per cell. Their full
three-dimensional Piola map and volume measure remain unchanged. The removed
vertical and height-dependent equations vanish by orthogonality for these
particular coefficients and boundary conditions. This reduction does not apply
to arbitrary three-dimensional permeability or boundary data. At F2 and F8,
the reduced and full three-dimensional fields agree within
\(1.5\times10^{-13}\) in relative flux L2 norm.

A full-dimensional F64 verification retains 2,099,200 trace coefficients
and 131,072 coarse pressure coefficients. With an explicit global target
of \(10^{-17}\) and extended residual corrections, its largest physical
block residual is \(3.21\times10^{-12}\), below the unchanged
\(10^{-10}\) acceptance criterion. The maximum sampled vertical flux is
\(5.46\times10^{-16}\) m/s. These checks verify the full-dimensional
equations and symmetry; they do not estimate spatial error. This auxiliary
verification requires a long-double type wider than float64. The distributed
field archives used for the reference comparisons remain float64.

The invariant classical operator is assembled directly, without a macro trace
restriction. A positive diagonal congruence balances its flux and pressure
units; PETSc/MUMPS solves that equivalent saddle system with explicit extended
residual refinement. Both the original physical equations and every fine-cell
divergence moment are checked, including after archival conversion to float64.
The MHM study retains its three-dimensional isotropic fine grid and macro
partitions. These are native PyMHM references, not an independent code
comparison or exact solutions.

All reported pressure and flux differences are physical volume L2 norms, with
the corresponding reference norm in the denominator. Integration respects every
fine element and macro interface. The reference is numerical, not exact; its
refinement differences quantify its remaining discretization sensitivity.
The principal pressure comparison uses \(\|p_{\rm ref}-p_e\|_{L^2}\)
as its denominator, so the common 25 MPa datum does not hide differences in
the pressure drop. The records also retain the total-pressure normalization
and the absolute differences. The three MHM macro levels share a fixed F8
fine grid. Comparisons with the classical F8 field measure the effect of
restricting the normal trace at that fine resolution. Comparisons with F128
also contain the effect of the unresolved fine approximation space.
The common integration hierarchy resolves both fields, including every vertical
MHM cell. Horizontal orders six and eight are compared; three Gauss points in
the vertical direction integrate the squared RT1/Q1 fields exactly on this
extruded geometry, without discarding their computed vertical components.

Oscillations of \(\delta xy\) are appreciable inside the exterior fine cells.
Consequently, a low-order polynomial rule is insufficient merely because the
finite-element degree is one. The assembly uses separate Gauss orders in the
horizontal and vertical reference directions. Bounded quadrature batches retain
the same points, weights and operator. Material-integration sensitivity is
measured independently of macro restriction and local spatial approximation.

For the F16 classical grid, increasing horizontal assembly order from 20 to
40 changes the flux by 0.07483% and the pressure increment by 0.03790%.
For the F32 classical grid, increasing horizontal assembly order from 10 to
20 changes the flux L2 norm by 0.1268% and the pressure-increment norm by
0.02494%. On the 32-macro MHM grid with F8 local resolution, orders 28 to 40
change those quantities by 0.02671% and 0.03492%. These checks separate
material quadrature from spatial error. For the additional MHM F16 control,
orders 28 to 40 change flux by 0.005907% and pressure increment by 0.006971%.
On F64, increasing the assembly order
from 10 to 14 changes the flux by 0.007556% and the pressure increment by
0.0008360%. Divergence moments use a separately
recorded polynomial rule: the Piola determinant cancels, and RT1 divergence
times a Q1 pressure test is integrated exactly by two Gauss points per axis
when the volume source is zero.

The classical sequence contains five independently solved fine resolutions.
The following differences use the finer field as denominator:

| Fine factors | Flux L2 difference | Pressure-increment L2 difference |
|---:|---:|---:|
| 8 → 16 | 15.8517% | 6.13199% |
| 16 → 32 | 18.7424% | 11.2420% |
| 32 → 64 | 11.5023% | 5.19362% |
| 64 → 128 | 5.24788% | 1.02117% |

The horizontal assembly orders are respectively 40, 40, 20, 14 and 8;
material-quadrature checks are recorded separately. Orders six and eight for
the physical norms agree within 1.9e−15 in absolute normalized differences.
The last flux increment is about one sixth of the 32.10% difference between
the finest MHM macro space and F128. Thus the baseline is substantially more
resolved for this comparison, but its remaining sensitivity is not negligible.
A difference between two reference levels is not a guaranteed error bound;
F128 is not asserted to be an exact or fully converged solution.

The direct classical F8–F128 comparison gives **28.3482% in flux** and
**22.8686% in pressure increment**, with F128 in the denominator. Its norm
quadratures six and eight agree within \(8.4\times10^{-17}\) in absolute
normalized difference. Thus keeping F8 fixed leaves substantial fine-grid
error even when all its normal moments are released. Macro refinement alone
does not remove this limitation.

## Trace restriction at fixed fine resolution

The following comparison uses the classical F8 field as reference. Both
methods retain RT1/Q1 on the same horizontal fine partition. The classical
field uses one vertical subdivision; agreement with the unrestricted,
vertically refined F8 field is independently verified within
\(4.8\times10^{-12}\) in relative flux norm. Integration against MHM retains
all eight vertical subdivisions and the computed vertical flux components.

| Macro hexahedra | Flux L2 difference to F8 | Pressure-increment L2 difference to F8 | Total-pressure L2 difference to F8 |
|---:|---:|---:|---:|
| 32 | 37.2872% | 41.3800% | 1.51935% |
| 256 | 20.8571% | 9.19376% | 0.337567% |
| 2,048 | 11.6196% | 2.84815% | 0.104576% |

The maximum relative change of these normalized norms between horizontal
quadrature orders six and eight is \(5.92\times10^{-12}\). These differences
decrease as the macro partition releases more fine normal moments. They
quantify the remaining trace restriction at F8; they do not demonstrate that
F8 resolves the heterogeneous problem.

The denominators in this table are the F8 norms. The direct classical
F8–F128 comparison and the table in [Physical field differences](#physical-field-differences)
use F128 norms. These percentages
must not be added or subtracted as an error decomposition. No orthogonality
between fine-grid and macro-trace errors is asserted.

The independent classical F16–F128 difference is **22.5039% in flux** and
**17.6648% in pressure increment**, using F128 norms. Refining the local
approximation therefore addresses a substantial source of discrepancy, even
before considering restrictions on the macro interfaces. These values describe
the two numerical fields and do not constitute an estimate with a known
reliability constant.

## Independent assembly of the same heterogeneous case

An independent **DOLFINx 0.9.0/UFL** application solves this same annular problem:
identical permeability, pressure data, cap conditions, geometry and RT1/Q1
spaces. Basix supplies the Raviart–Thomas basis and its orientation maps;
UFL assembles the mixed weak form, and PETSc/MUMPS solves the complete
conforming system. The Basix degree parameter is two for the family called
RT1 here. PyMHM supplies the declared mesh coordinates and evaluates its own
archived target fields; its element matrices, condensation and global solver
are not used in the independent assembly.

The classical F2, F8 and F16 comparisons release all fine normal moments.
A separate full three-dimensional F2 solve verifies the height-invariant
reduction: its largest sampled vertical flux is
$1.10\times10^{-16}$ m/s. The MHM comparisons restrict the independently
assembled conforming system to the same piecewise-$P_1$ macroface normal
space, retaining every interior fine flux and pressure coefficient. Basix
interpolation defines that restriction. This is a global restricted mixed
formulation, not a native DOLFINx MHM controller. The accepted residual is
that of the restricted equations; unconstrained fine-face tests belong to a
larger space and need not have zero residual.

| PyMHM field | Independent field | Relative flux L2 difference | Relative pressure-increment L2 difference |
|---|---|---:|---:|
| Classical F2 | Full 3D RT1/Q1 | $8.64\times10^{-13}$ | $1.73\times10^{-13}$ |
| Classical F2 | Height-invariant RT1/Q1 | $1.22\times10^{-14}$ | $3.20\times10^{-14}$ |
| Classical F8 | Height-invariant RT1/Q1 | $6.42\times10^{-14}$ | $2.12\times10^{-14}$ |
| Classical F16 | Height-invariant RT1/Q1 | $1.16\times10^{-13}$ | $3.19\times10^{-14}$ |
| MHM F8, 32 macros | Restricted RT1/Q1 | $5.49\times10^{-12}$ | $7.88\times10^{-13}$ |
| MHM F8, 256 macros | Restricted RT1/Q1 | $3.62\times10^{-12}$ | $1.76\times10^{-13}$ |
| MHM F8, 2,048 macros | Restricted RT1/Q1 | $3.25\times10^{-11}$ | $9.00\times10^{-13}$ |

These are full physical-volume comparisons, with the independent field in
the denominator. Horizontal integration orders six and eight are recorded
separately. Every vertical PyMHM cell and its computed $q_z$ are retained;
the largest sampled MHM $q_z$ is $1.05\times10^{-14}$ m/s. The pressure
increment uses the same 25 MPa datum as the spatial study. Agreement checks
the finite discretizations of the declared heterogeneous case. It does not
remove their approximation error or resolve the historical material-scale
and mesh ambiguities.

![Independent UFL and PyMHM pressure and signed flux components on the same annular macros](../figures/mapped-well-oscillatory/native-comparison.png)

The reference panels represent the same MHM spaces through monolithic
RT1/Q1 Darcy assembly in DOLFINx/UFL and a separate macroface normal-moment
restriction. This restriction is part
of the comparison application, not a built-in DOLFINx MHM controller.

The figure displays independent values at 36 Gauss points per horizontal fine
cell on $z=-4.375$ m. Each value occupies its own midpoint-bounded subregion
inside that cell; no interpolation crosses an element interface. The actual
macro intersections appear on every panel. Native and PyMHM fields share
color limits, while differences retain their own numerical scales. Signed
flux and difference colors use an asinh normalization with linear width
one ten-thousandth of their respective maximum absolute value. The much smaller
difference scales express finite-precision
agreement; they are not spatial-error estimates.

The [numerical comparison records](../figures/mapped-well-oscillatory/native-discrete-verification.json)
identify all seven accepted fields, physical norms, archive digests, executed
solver and installed native build. The verified upstream source is
[DOLFINx v0.9.0, commit `6443e3b`](https://github.com/FEniCS/dolfinx/tree/6443e3b27d29aec04ce83d8424b57c339e35f865).
The original plotting application `examples/plot_mapped_well_independent.py`
renders the published numerical samples.

## Local refinement with the coarse traces fixed

The additional control keeps all 2,048 macro hexahedra and 27,136 normal-trace
coefficients unchanged. Doubling the local resolution from F8 to F16 increases
the number of fine hexahedra eightfold. It changes the computed flux by
10.9546% and the pressure increment by 5.26678%, with the F16 MHM norms in
the denominators. Assembly-order sensitivity is below 0.007% for both fields,
so this comparison measures spatial resolution rather than material quadrature.

Each MHM field can also be compared with the unrestricted classical field
on its own fine partition, using matching horizontal assembly order 40:

| Fine resolution | Fine hexahedra | Flux difference to matching classical field | Pressure-increment difference to matching classical field |
|---:|---:|---:|---:|
| F8 | 16,384 | 11.6196% | 2.84815% |
| F16 | 131,072 | 15.8663% | 3.50097% |

The larger relative restriction difference at F16 is compatible with a richer
classical fine space and an unchanged coarse trace space. Local refinement
does not add normal moments on macro interfaces. Consequently, it cannot be
treated as a substitute for trace enrichment. The corresponding fine-reference
norm changes between the two rows; these percentages are not an additive
decomposition of the differences from F128.

The F16 MHM production rate is 0.01203972026 m³/s. Its maximum physical block
residual is \(2.93\times10^{-13}\), and its largest pressure-tested divergence
moment is \(1.47\times10^{-18}\). Inner and outer rates balance, and the cap
fluxes vanish. These checks establish the discrete equations and conservation;
accuracy is assessed separately through the physical field differences.


## Physical field differences

The finest classical reference has 524,288 hexahedra and 6,293,504 mixed
unknowns. Its integrated production rate is 0.01317757 m³/s, balanced by
the opposite exterior-wall rate; the cap fluxes vanish. The corresponding
MHM production rates are 0.01005014, 0.01130883 and 0.01177187 m³/s. These
are outputs under the same prescribed pressures, without adjusting the
heterogeneous tensor to impose a selected rate. All four MHM configurations are
compared with this same F128 field. The first three differences include the fixed
F8 approximation and the remaining macro-trace restriction; the final row
uses the refined F16 local space:

| Macro hexahedra | Fine resolution | Flux L2 difference | Pressure-increment L2 difference | Total-pressure L2 difference |
|---:|---:|---:|---:|---:|
| 32 | F8 | 49.5554% | 55.5926% | 1.71429% |
| 256 | F8 | 37.8916% | 22.5462% | 0.695251% |
| 2,048 | F8 | 32.0979% | 23.0199% | 0.709857% |
| 2,048 | F16 | 28.5379% | 18.9954% | 0.585755% |

Horizontal norm quadratures six and eight differ by at most
\(2.3\times10^{-16}\) in absolute value between the normalized norms
in this table. This checks
the integration of the archived fields; it does not reduce their spatial
approximation error. In particular, pressure improvement is not monotone
between the last two macro spaces at fixed fine resolution.
The additional local refinement improves the difference from F128 but leaves a
substantial discrepancy. Its two controls identify separate limitations: even
classical F16 differs from F128 by 22.50% in flux, and the fixed MHM trace
differs from classical F16 by 15.87%. Neither increasing the local quadrature
nor refining only the local mesh resolves both approximation spaces.

Across the eight additional norm acquisitions, the largest relative change
between orders six and eight is \(5.54\times10^{-14}\). The remaining
discrepancies are much larger than these integration changes. The F128
reference's own 5.25% final flux increment still prevents interpretation of
any MHM-to-F128 difference as an exact error or certified bound.


The spatial panels use the upper-side value at \(z=0\), common reference/MHM
color limits and the actual macro edges. Error panels have separate scales.
The profiles preserve independent element values and show signed radial flux,
positive away from the well, alongside pressure and flux magnitude. Negative
radial values near the well indicate production with the stated pressure data.

The complementary logarithmic map resolves the flux dynamic range away
from the well. Reference and MHM magnitudes share the same color limits;
the vector-difference norm has its own logarithmic scale. The linear maps
above retain the absolute peak contrast, and neither display clips or
interpolates the physical field. Both F8 and F16 logarithmic maps are identified
below; the diagonal profiles retain the three F8 macro configurations.


![One-sided pressure, flux magnitude and signed radial flux profiles](../figures/mapped-well-oscillatory/profiles.png)

![Separate fixed-fine trace restriction, MHM to F128, classical refinement and material-quadrature comparisons](../figures/mapped-well-oscillatory/convergence.png)

## Reproducibility

`examples/solve_mapped_oscillatory_well.py` acquires the native fields and records
source digests, physical block residuals, pressure-tested mass balances and
integrated boundary rates. The command is invoked as a module, for example:

```sh
pixi run --locked -e test-core python -m examples.solve_mapped_oscillatory_well \
  --fine-factor 8 --macro-factor 1 --quadrature-xy 40 --quadrature-z 10
pixi run --locked -e test-core python -m examples.solve_mapped_oscillatory_well \
  --fine-factor 16 --macro-factor 4 --quadrature-xy 40 --quadrature-z 10
```

The shared three-dimensional solver accepts `global_rtol` and
`global_refinement_precision="extended"` as explicit options for the retained
saddle solve. The latter requires a NumPy long-double type wider than float64.
A stricter global target can trigger additional residual corrections; it does
not relax the physical block criterion or guarantee that every conditioned
system reaches it. The acquisition drivers expose the corresponding
`--global-rtol` and `--global-refinement-precision` flags and give nondefault
runs distinct names. Their numerical records retain the actual source digests.

The refined invariant classical references are acquired with:

```sh
pixi run -e fem python -m examples.solve_mapped_well_invariant \
  --fine-factor 128 --quadrature-xy 8 --solver petsc
pixi run -e fem python -m examples.solve_mapped_well_invariant \
  --fine-factor 16 --quadrature-xy 40 --solver petsc
```

The common field reader `examples/mapped_well_fields.py` verifies archive digests
and uses the exact integer refinement hierarchy. It neither matches nodes by a
spatial tolerance nor averages discontinuous fields. Tests independently verify
nonzero analytical L2 norms, nested-grid equivalence, physical producer signs
and the global tensor-scale identity. Geometry and coefficient-scale ambiguities
prevent an assertion of identical historical Figures 16–18.

The complete norm table can be regenerated from the field archives with:

```sh
pixi run --locked -e test-core python -m examples.compare_mapped_well --workers 8
```

Each published comparison retains its field digests, numerical acquisition
source digests and analysis-record digest. The aggregate distinguishes the
three fixed-F8 comparisons, three MHM-to-F128 comparisons, the direct
classical F8–F128 difference, consecutive classical refinements and the
separate quadrature and vertical-invariance checks. The additional F16 control
has its own same-fine trace comparison, F128 comparison, local-refinement
difference and material-quadrature check, each retaining its actual acquisition
and analysis source digests.
