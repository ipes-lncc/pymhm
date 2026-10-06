# MH²M with an oscillatory permeability

This study uses the scalar coefficient in equation (61) of
[de Barros, Madureira and Valentin (2026)](https://arxiv.org/abs/2404.16978v3):

$$
\begin{aligned}
K(x,y)
 &=\frac{2+\gamma\sin(2\pi x/\varepsilon)}
         {2+\gamma\cos(2\pi y/\varepsilon)}
   +\frac{2+\sin(2\pi y/\varepsilon)}
         {2+\gamma\sin(2\pi x/\varepsilon)},\\
\varepsilon&=\frac1{14},\qquad \gamma=1.8,\\
-\nabla\cdot(K\nabla p)&=f,\qquad q=-K\nabla p,\\
f(x,y)&=-2x(x-1)-2y(y-1),\\
p&=0\quad\text{on }\partial(0,1)^2.
\end{aligned}
$$

The article explicitly gives $\varepsilon=1/14$, homogeneous pressure on the
unit square, and local/interface degrees $P_1/P_0/P_1$. Section 8.2 does not
specify $\gamma$ or restate the source. The preceding
[de Barros (2022)](https://www.lncc.br/~alm/students/frankdissert.pdf)
does specify both: Section 4.2.2.1, Equation (4.4), printed page 62
(PDF page 64), gives $\gamma=1.8$ and uses the forcing of Equation (4.1),
printed page 54 (PDF page 56). These are exactly the coefficient parameter
and forcing written above. The dissertation's oscillatory study uses
$\varepsilon=1/17$, whereas the version-3 article specifies $1/14$.
Its Figure 24, printed page 63 (PDF page 65), also specifies a classical
$P_1$ reference with 130 subdivisions per coordinate; article Figure 5
specifies 128. These are distinct reference discretizations and periods.
This primary source establishes the data-family lineage; it does not by
itself identify the historical meshes or arrays behind every version-3
figure. No permeability or forcing scale is adjusted to match plotted
pressure.

The [source-archive provenance](../figures/mh2m-heterogeneous/crisscross/source-archive-provenance.json)
identifies the arXiv version-1, version-2 and version-3 source packages by
URL and SHA-256. They contain manuscript and figure files, without numerical
solver scripts or mesh/field arrays. The ten figure images common to versions
2 and 3 have identical RGB pixels. The source packages do not supply an
additional value of $\gamma$ or a different forcing for the oscillatory case.

## Spaces and comparison scope

The native [MH²M formulation](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mh2m.md) uses a continuous pressure trace
$\Gamma$, a broken conormal space $\Lambda$, and continuous local $P_1$
pressures. Its conormal is $K\nabla p\cdot n$, the negative of the physical
outward Darcy flux. The local Neumann maps include the full source correction
in equation (29).

The diagonal-mesh control series splits every macro square along its
southwest-to-northeast diagonal.
For $n$ squares per coordinate direction there are $2n^2$ triangular
macroelements and maximum diameter $H=\sqrt2/n$. Local refinement $r$
produces $r^2$ fine triangles in each macroelement. Trace subdivisions are
independent inputs.

The diagonal-mesh spatial control associated with **Figure 5** uses the stated
$n=31$, local $r=4$, and two segments per macroedge for both $\Gamma$ and
$\Lambda$. The classical $P_1$ reference at $n=128$ has the stated reference
diameter $\sqrt2/128$. A separately refined classical series checks that
this reference is itself an approximation. The source and $\gamma$ agree with the predecessor dissertation. The
recorded diagonal local mesh is an explicitly identified discretization;
it is not inferred to be the historical crisscross mesh.

The global-refinement and fixed-$\Gamma$ studies examine the mechanism
illustrated in **Figures 6–8**. They retain continuous $P_1$ pressure traces,
broken $P_0$ conormals and local $P_1$ pressures. Refining $\Lambda$ increases
the local Neumann-map dimension while keeping the global pressure-trace
dimension fixed. Total global dimensions include Dirichlet trace nodes;
the number of free unknowns is reported separately. The figure labels such
as $289^2$ denote the size of a square matrix, whose unknown count is 289.

## Compatibility of the local spaces

Assumption A requires the Neumann lifting to be injective on the
zero-boundary-mean conormal space. On a uniformly refined triangle, placing
one constant conormal unknown on every fine boundary edge can violate this
condition: an even boundary cycle admits alternating edge values whose
integrals against every continuous $P_1$ nodal boundary function vanish.
Their total conormal integral also vanishes. Such a mode lies in the
local Neumann-map kernel.

The combinations $r=s_\Lambda=2,4,8$ therefore have an insufficient local
pair on this triangulation. They are rejected rather than inverted through
a pseudoinverse. The enrichment study uses $r=16$ and $r=32$ with
$s_\Lambda=1,2,4,8$, resolving each conormal segment with at least two fine
edges. This preserves the prescribed global $\Gamma$ while satisfying the
checked local injectivity condition. It differs from taking the starred
$h=H/8$, $H_\Lambda=H/8$ caption literally on a uniform triangular boundary.
The predecessor dissertation supplies an alternative fine connectivity in
Figure 4: each Cartesian fine square is split across both diagonals.
The section below examines that recovered geometry separately.

A nonsingular finite matrix is a discrete check, not a mesh-uniform inf-sup
proof. The sufficient hypotheses and conclusions of the article remain
distinct from this verification.

## Recovered crisscross geometry

Figure 4 of the [de Barros (2022)](https://www.lncc.br/~alm/students/frankdissert.pdf),
printed page 54 (PDF page 56), shows northwest-to-southeast macro diagonals
uniformly throughout the square, and a Cartesian fine grid split across both
diagonals. Definition 4.2.1, printed page 55, partitions **each** macroface
into the same number $N$ of subfaces. Section 4.2.1.3, printed page 57,
uses that definition for both $\Lambda$ and $\Gamma$. Thus the interface
partition is independent of the number of fine edges on a macroface;
it is not automatically inherited from the local triangulation.
The explicit
`crisscross_submesh` construction has $2r^2$ fine triangles per macro,
with $r,r,2r$ fine boundary edges on its three sides. The local boundary
pairing with $r$ constant conormal segments per macroedge is injective in
the checked cases. This differs from the alternating null mode on a
uniform $r^2$ triangular subdivision.

The [geometric admissibility record](../figures/mh2m-heterogeneous/crisscross-n4-r8-g1.json)
uses $n=4$, $r=8$, one continuous $P_1$ pressure-trace segment and eight
constant conormal segments. It has 32 macroelements, 128 fine triangles
per macro and 25 total / 9 free global unknowns. The minimum local Neumann
energy eigenvalue is $9.05\times10^{-7}$; the global residual is
$9.59\times10^{-17}$ and maximum macro balance defect
$3.82\times10^{-17}$. These are discrete compatibility and equation checks,
not an approximation-error certificate.

The geometric lengths are recorded independently:

$$
H_{\max}=\frac{\sqrt2}{n},\qquad
h_{\max}=\frac1{nr},\qquad
H_{\rm Cartesian}=\frac1n.
$$

The dissertation calls the Cartesian spacing $H$ in its mesh figure.
Consequently, its pictured construction does not by itself identify the
meaning of every $h/H$ ratio in the version-3 article. In particular,
$r=4$ gives only 16 local boundary pressure nodes and cannot support 24
independent conormal coefficients ($s_\Lambda=8$ on three sides).
The zero-mean conormal space then has dimension 23, larger than the
15-dimensional zero-mean pressure trace. This dimension obstruction remains
regardless of an invertible global pressure-trace matrix.

The [boundary-pairing record](../figures/mh2m-heterogeneous/crisscross/space-compatibility.json)
checks these dimensions directly. For $r=4$, the 23-column zero-mean
conormal pairing has rank 15. For $r=8$, it has rank 23, with 32 boundary
pressure nodes. These are checks of the discrete lifting spaces, independent
of a permeability scale or a plotted pressure amplitude.
Inheriting all $8,8,16$ fine edges instead would give 31 zero-mean conormal
coordinates but rank 30: the alternating boundary mode remains invisible
to continuous $P_1$ tests. That inherited partition is therefore neither
the uniform-per-macroface definition above nor an admissible replacement
for this local space.

The recovered-geometry Figure-5 configuration uses $n=31$, $r=4$,
and two segments for each of $\Gamma$ and $\Lambda$. It has 1,922 macro
triangles, 32 local fine triangles per macro, and 3,969 total / 3,721 free
pressure-trace unknowns. Its archived pressure range is
$[-2.70194\times10^{-4},\,1.72712\times10^{-2}]$; the weak boundary
formulation does not impose a pointwise positivity principle.
The maximum macro balance defect is $1.08\times10^{-16}$.
The profiles retain the classical $P_1$, $n=128$ reference resolution
printed in Figure 5. Their additional reference, spatial differences and
volume norms use the independently refined $P_3$, $n=512$ field below. Their proximity is
a field comparison, not an identification of every historical mesh convention.


The additional flux view uses a signed asinh color normalization. Each
reference/MH²M pair shares a symmetric range containing every displayed
sample, with a linear width equal to 3% of that range's positive endpoint.
The difference has its own complete sampled range. Colorbar ticks remain physical flux values;
neither fields nor integrated norms are transformed or clipped. The
[display record](../figures/mh2m-heterogeneous/crisscross/fields-asinh-display.json)
lists every range and width. Pressure remains on the linear scale.


![Recovered crisscross pressure profile with independent incident values](../figures/mh2m-heterogeneous/crisscross/profiles.png)

### Figure-6 configurations and global dimensions

The upper-row configurations use $r=4$ and one segment per trace. The
following panels compare MHM at 336 and 4,560 global unknowns with MH²M at
289 and 4,356 total pressure-trace unknowns. The starred eight-segment
conormal variant is excluded for this local resolution by the injectivity
condition above.


The lower-row caption assigns $i=4$ and hence eight segments through
$H_\Gamma=H_\Lambda=H/2^{i-1}$. Its printed dimensions instead correspond
to four uniform subfaces: MHM gives 390 and 1,206 unknowns at $n=5,9$,
while continuous piecewise-linear $\Gamma$ gives 291 and 1,081 at
$n=5,10$. The dimension-matched controls therefore use four segments,
and compare four versus eight conormal segments at the same $\Gamma$.
The local mesh is $r=8$ in every lower-row control. These alternatives are
identified explicitly rather than treating the caption and matrix
dimensions as the same discrete space.


Both figures preserve the two incident traces at $y=1/2$. Dotted vertical
lines mark the union of the actual macroface intersections of the compared
meshes. The reference is the independently refined classical $P_3$, $n=512$ field;
these panels compare declared discrete configurations and do not infer
unavailable historical coefficients from visual proximity.

### Direct comparison with the printed profiles

The [published-profile record](../figures/mh2m-heterogeneous/crisscross/published-profiles.json)
extracts the visible line colours from the embedded rasters of Figures 7
and 8 in the version-3 article. Their RGB pixels are identical to the
original PNG figures in the article's arXiv source package; no PDF rendering
resamples these data. The original excerpts are available as
[Figure 7](../figures/mh2m-heterogeneous/crisscross/published-figure-7.png)
and [Figure 8](../figures/mh2m-heterogeneous/crisscross/published-figure-8.png).
Axis calibration uses the printed ticks. The record retains line thickness,
missing columns where another curve occludes the colour, and a one-pixel
coordinate uncertainty. One vertical pixel represents $6.01\times10^{-5}$
in Figure 7 and $4.01\times10^{-5}$ in Figure 8. No numerical field is used
to select the raster ordinates.

The following comparisons retain both incident numerical traces at $y=1/2$,
mark their actual macroface intersections and apply no pressure rescaling.
The grey bands contain the visible raster span plus one vertical pixel;
gaps remain where a printed colour is occluded. In the first panel, the blue
curve is the classical $P_1$, $n=128$ reference.
The available support differs between printed curves: Figure 8 provides
643 visible columns for $n=16$, but only 17 for $n=80$. The panel titles
report these counts. Their RMS differences therefore are not a convergence
series on a common sampling set.

![MHM crisscross profiles compared directly with article Figure 7](../figures/mh2m-heterogeneous/crisscross/published-figure-7-comparison.png)

![MH²M crisscross profiles compared directly with article Figure 8](../figures/mh2m-heterogeneous/crisscross/published-figure-8-comparison.png)

The [profile-comparison record](../figures/mh2m-heterogeneous/crisscross/published-profile-comparison.json)
reports differences on visible raster columns, separately from the physical
volume norms below. The recovered connectivity and the declared data do not
give quantitative agreement with every printed curve. For example, Figure 8
at $n=16$ has a printed central pressure of approximately $0.01936$,
whereas the computed field has maximum $0.01513$. The discrepancy exceeds
the raster uncertainty. Matching global dimensions and satisfying the discrete
equations therefore do not establish reproduction of those profiles.

The classical curve also provides a control independent of the multiscale
formulation. The [CG geometry comparison](../figures/mh2m-heterogeneous/crisscross/reference-geometry/comparison.json)
keeps $n=128$, the coefficient, source and homogeneous boundary data fixed,
and compares the two uniform diagonal orientations and a Cartesian
crisscross mesh. The following RMS differences use only visible raster
columns; they are absolute pressure differences, not relative volume errors.

| Classical connectivity | Pressure unknowns | Figure 7 profile RMS | Figure 8 profile RMS |
|---|---:|---:|---:|
| Southwest–northeast | 16,641 | $3.13344\times10^{-4}$ | $1.78109\times10^{-4}$ |
| Northwest–southeast | 16,641 | $3.13609\times10^{-4}$ | $1.77289\times10^{-4}$ |
| Cartesian crisscross | 33,025 | $2.86214\times10^{-4}$ | $1.85666\times10^{-4}$ |

The diagonal meshes have maximum diameter $\sqrt2/128$; crisscross has
maximum diameter $1/128$ and twice as many triangles. Thus the latter
changes connectivity and finite-element resolution together. Assembly
orders 10 and 14 change its pressure by $6.15\times10^{-14}$ in relative
$L^2$; the corresponding diagonal control changes by
$8.78\times10^{-13}$. Both geometries retain discrepancies beyond the
raster envelope. These controls do not identify the historical reference
mesh or replace the independently refined reference series.

The [independent P1 verification](../figures/mh2m-heterogeneous/cg1-native-verification.json)
assembles the same six $n=32,\ldots,1024$ operators with DOLFINx/UFL,
including the physical Duffy-8 coefficient integration. The maximum
absolute pressure-coefficient difference is $8.67\times10^{-15}$;
the maximum absolute gradient-component difference is
$7.46\times10^{-14}$. A nonhomogeneous affine patch also
checks the boundary convention. Separately, the documented combinations
$n=128,130$ and $\varepsilon=1/14,1/17$, together with nine fixed integration
rules, do not place the whole classical profile within the printed line
width plus one-pixel envelope. The literal $n=128$, $\varepsilon=1/14$
profile lies within that envelope on 197 of 466 visible columns in
Figure 7 and 110 of 314 in Figure 8; the dissertation's
$n=130$, $\varepsilon=1/17$ gives 93 and 38, respectively.
These are comparisons of specified data and integration choices, not
identifications of the historical field or a reason to change the
article's coefficient.

### Same-case native MH²M verification

A separate DOLFINx/UFL application assembles the complete three-field system
for six of these heterogeneous cases, retaining the archived macro mesh, fine
crisscross connectivity, source, coefficient and homogeneous boundary data.
It uses local continuous $P_1$ pressures, the stated continuous $P_1$ pressure
trace and broken $P_0$ conormal partitions. PETSc/MUMPS solves the uncondensed
system, without invoking the PyMHM multiscale condensation.

The native stiffness uses the original physical Duffy-10 cell average of the
coefficient. This retains the same $P_1$ operator because its local gradients
are constant; physical flux differences instead integrate the pointwise
coefficient on each fine triangle. The comparison covers complete physical
fields, including the lower-row alternatives of Figure 6 and the discrepant
Figure-8 $n=16$ configuration.

| Configuration | Macrotriangles | Native full-system unknowns | Relative pressure L2 difference | Relative physical-flux L2 difference |
|---|---:|---:|---:|---:|
| Figure 6, n5, sΛ=4 | 50 | 4,861 | 1.976e-15 | 1.274e-14 |
| Figure 6, n5, sΛ=8 | 50 | 5,461 | 6.722e-15 | 4.145e-14 |
| Figure 6, n10, sΛ=4 | 200 | 19,521 | 1.323e-14 | 3.545e-14 |
| Figure 6, n10, sΛ=8 | 200 | 21,921 | 5.436e-14 | 9.994e-14 |
| Figure 5 | 1,922 | 63,303 | 2.333e-14 | 6.666e-14 |
| Figure 8, n16 | 512 | 53,985 | 1.411e-14 | 5.034e-14 |

All relative differences use the norm of the corresponding archived PyMHM
field. The largest broken-gradient difference is 9.668e-14; the largest
relative residual of the complete native saddle system is 3.213e-12.
Pressure differences use exact $P_1$ triangle mass integrals. No field is
smoothed across macrofaces or rescaled to match a published ordinate.

The [six-case native-system record](../figures/mh2m-heterogeneous/crisscross/native-system-verification.json)
identifies every candidate field, executed source, discretization, norm and
[FEniCS/DOLFINx release](https://github.com/FEniCS/dolfinx/tree/6443e3b27d29aec04ce83d8424b57c339e35f865).
This discrete agreement supports the selected full heterogeneous cases,
including their pressure-profile discrepancy with the publication. It does
not recover unspecified historical inputs or establish equality with every
printed curve. The independently refined classical reference below assesses
approximation error, while the [smooth MH²M sequence](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mh2m.md)
checks the published gradient rates under smooth-data assumptions.

## Physical norms and reference refinement

All comparisons integrate on a common subdivision resolving both sets of
fine triangle edges, including their diagonals. They do not evaluate an
interpolated difference on only one mesh. The scalar permeability is
evaluated at the common physical quadrature points.

The reported quantities are

$$
\begin{aligned}
e_p&=\frac{\lVert p_h-p_{\rm ref}\rVert_{L^2}}
            {\lVert p_{\rm ref}\rVert_{L^2}},\\
e_{\nabla p}&=\frac{\lVert\nabla_h p_h-\nabla p_{\rm ref}\rVert_{L^2}}
                    {\lVert\nabla p_{\rm ref}\rVert_{L^2}},\\
e_q&=\frac{\lVert-K\nabla_h p_h+K\nabla p_{\rm ref}\rVert_{L^2}}
            {\lVert K\nabla p_{\rm ref}\rVert_{L^2}},\\
e_E&=\frac{\lVert K^{1/2}(\nabla_h p_h-\nabla p_{\rm ref})\rVert_{L^2}}
            {\lVert K^{1/2}\nabla p_{\rm ref}\rVert_{L^2}}.
\end{aligned}
$$

The broken $H^1$ comparison is the gradient seminorm used in the article.
The physical flux and energy comparisons are distinct additional diagnostics.
The raw flux is not identified with the conormal multiplier, nor with an
$H(\mathrm{div})$ reconstruction. Every denominator belongs to the stated
reference field; consecutive reference differences use the finer reference.

The overlay routine is verified independently on nonnested $2\times2$ and
$3\times3$ grids using affine fields, discontinuous fields and variable
permeability with known nonzero pressure, flux and energy integrals.
The original $P_1$ comparisons store error orders 6 and 8; the independent
$P_3$ comparisons below store orders 8 and 10.
Spawned norm integration preserves the same geometry batches, quadrature
points and ordered reductions; [field-based equivalence checks](../figures/mh2m-heterogeneous/norm-equivalence.json)
agree bitwise with serial evaluation at both orders.
The [common-refinement geometry verification](../figures/mh2m-heterogeneous/cg3/geometry-integration-verification.json)
also compares the ordered clipping batches bitwise on six independent
mesh pairs and verifies finitude, containment, determinants and total
domain area for all six large production overlays. These are integration
and data-integrity checks, not approximation-error estimates.

### Independently assembled P3 reference

A separate DOLFINx/UFL application uses continuous $P_3$ pressures on five
uniform SW–NE meshes, with the same coefficient, forcing and homogeneous
Dirichlet condition. Its independently measured refinement is:

| Fine resolution | P3 unknowns before boundary elimination | Pressure | Gradient | Physical flux | Energy |
|---:|---:|---:|---:|---:|---:|
| $32\to64$ | 37,249 | 1.410684% | 10.42242% | 12.72037% | 11.78373% |
| $64\to128$ | 148,225 | 0.09725054% | 2.516183% | 3.343996% | 3.052977% |
| $128\to256$ | 591,361 | 0.003259652% | 0.4139197% | 0.5774536% | 0.5242915% |
| $256\to512$ | 2,362,369 | 0.0001237549% | 0.05629337% | 0.07837009% | 0.071857% |

Each denominator belongs to the finer $P_3$ field. The last flux increment
is 0.0783701%, compared with 3.25733% for the last $P_1$ increment below.
The direct $P_1$ $n=1024$ versus $P_3$ $n=512$ comparison gives 0.0329737%
in pressure, 1.57617% in gradient, 1.89077% in physical flux and 1.81440%
in energy. These are separately measured differences, not additive error
components or certified bounds on the exact-solution error.

The [P3 refinement and comparison record](../figures/mh2m-heterogeneous/cg3/comparison.json)
retains both error quadratures, orders 8 and 10. Changing the finest
assembly order from 12 to 16 changes the flux by $2.39\times10^{-12}$
relative and the pressure by $5.18\times10^{-13}$. Its original-equation
residual, evaluated with extended accumulation on the stored binary64
coefficients, is $6.01\times10^{-11}$.

An [independent native UFL energy integral](../figures/mh2m-heterogeneous/cg3/native-norm-verification.json)
agrees with the persisted-field evaluator and shared physical norm to
$1.99\times10^{-13}$ relative. This checks replay and integration; it does
not identify the remaining discretization error with zero. The
[acquisition environment](../figures/mh2m-heterogeneous/cg3/acquisition-environment.json)
records the floating-point formats and numerical library versions.

For the recovered crisscross Figure-5 configuration, the relative differences
against this same $P_3$ field are 0.882929% in pressure,
18.4582% in broken gradient, 20.5388% in physical flux
and 20.027% in energy. The flux difference is substantially
larger than the final reference increment; the independently checked
classical refinement therefore does not explain the remaining multiscale
approximation error. All 24 crisscross cases and 18 diagonal controls
use this same reference for volume norms and numerical-error figures.

For the recovered crisscross sequence associated with Figure 8, the
physical differences are:

| Macro subdivisions $n$ | Pressure | Broken gradient | Physical flux | Energy |
|---:|---:|---:|---:|---:|
| 16 | 12.7005% | 34.6277% | 38.2061% | 35.8157% |
| 34 | 10.0132% | 29.9456% | 33.7507% | 31.5694% |
| 65 | 5.07402% | 20.6101% | 23.579% | 22.4609% |
| 80 | 3.69282% | 17.382% | 20.0422% | 19.1586% |
| 100 | 2.56454% | 14.3068% | 16.6177% | 15.9487% |

The finest field therefore retains a 16.6177% flux difference despite the
much smaller reference increment. This is an approximation limitation of
the stated local and trace resolutions. The table measures whole-domain
norms; it is distinct from the visible-column raster comparisons above.

### Separate P1 reference-resolution control

The continuous $P_1$ control is refined independently through six meshes.
The following percentages compare consecutive levels using the finer field
as the denominator in each norm:

| Fine resolution | P1 unknowns before boundary elimination | Pressure | Gradient | Physical flux | Energy |
|---:|---:|---:|---:|---:|---:|
| $32\to64$ | 4,225 | 8.35416% | 28.8831% | 28.8080% | 28.7981% |
| $64\to128$ | 16,641 | 4.35282% | 19.5535% | 21.0693% | 20.7936% |
| $128\to256$ | 66,049 | 1.41344% | 10.6135% | 12.1984% | 11.8582% |
| $256\to512$ | 263,169 | 0.384634% | 5.41787% | 6.42669% | 6.18614% |
| $512\to1024$ | 1,050,625 | 0.0983507% | 2.72241% | 3.25733% | 3.12808% |

These $P_1$ fields remain a separate reference-resolution control. Their
last flux increment is 3.26%; the physical multiscale comparisons below use
the $P_3$, $n=512$ field with its own 0.0783701% flux increment. Neither
measurement is a rigorous exact-error bound. The $P_1$, $n=128$ curve in
the historical profiles retains the reference resolution stated in Figure 5.

## Diagonal-mesh control: comparable global dimensions

Both methods use local $P_1$ pressures with $r=16$. MH²M uses one continuous
$P_1$ pressure-trace segment and eight broken $P_0$ conormal segments per
macroedge; primal MHM uses one $P_0$ flux segment per macroedge. The paired
rows have similar global dimensions but different macro meshes and local
fine resolutions. They do not isolate a method change at identical spaces.
All percentages below use the same independently assembled $P_3$, $n=512$ field.

| Method | $n$ | Global total/free | Pressure | Broken gradient | Physical flux | Energy |
|---|---:|---:|---:|---:|---:|---:|
| MH²M | 4 | 25 / 9 | 19.536% | 44.8492% | 45.8888% | 45.5005% |
| MHM | 2 | 24 / 24 | 19.052% | 53.5457% | 64.477% | 57.9873% |
| MH²M | 8 | 81 / 49 | 9.6528% | 31.8566% | 32.9249% | 32.4292% |
| MHM | 4 | 88 / 88 | 7.12284% | 39.1251% | 40.0642% | 39.7538% |
| MH²M | 16 | 289 / 225 | 11.9155% | 33.9508% | 36.4749% | 34.8302% |
| MHM | 8 | 336 / 336 | 9.99028% | 39.8203% | 33.9435% | 36.9524% |
| MH²M | 32 | 1,089 / 961 | 10.2378% | 30.4549% | 34.1015% | 31.9583% |
| MHM | 16 | 1,312 / 1,312 | 19.9965% | 51.0266% | 41.9914% | 45.8025% |
| MH²M | 64 | 4,225 / 3,969 | 5.09467% | 20.7658% | 23.5161% | 22.5074% |
| MHM | 32 | 5,184 / 5,184 | 16.6933% | 43.2401% | 39.7771% | 41.1149% |

Neither series is monotone over all five macro meshes. The permeability
oscillation is fixed throughout: changing $H$ changes how its wavelength is
resolved by the macro trace and the local finite-element mesh. The finest
MH²M result has a 5.09467% pressure difference and a 23.5161% raw-flux
difference, well above the last 0.0783701% classical flux increment. The small algebraic and
conservation defects do not remove this approximation error. These data
verify the declared experiment and its independent traces; they do not
establish quantitative reproduction of Figures 6–8 with the historical
local connectivity.


## Diagonal-mesh control: fixed global pressure trace

The macro grid is fixed at $n=16$: 512 triangles, 289 total pressure-trace
unknowns and 225 free unknowns. Each row changes only the stated local
refinement and number of broken conormal segments. The pressure trace
remains continuous $P_1$ with one segment on each macroedge.

| Local $r$ | Conormal segments | Pressure | Broken gradient | Physical flux | Energy |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 12.6391% | 46.4411% | 41.0249% | 43.4636% |
| 16 | 2 | 1.84705% | 36.0102% | 36.2142% | 36.4933% |
| 16 | 4 | 8.88479% | 34.1233% | 34.5946% | 34.4848% |
| 16 | 8 | 11.9155% | 33.9508% | 36.4749% | 34.8302% |
| 32 | 1 | 13.4293% | 46.7957% | 40.9781% | 43.5162% |
| 32 | 2 | 1.88364% | 36.1878% | 35.9567% | 36.4569% |
| 32 | 4 | 8.12468% | 33.9203% | 33.9843% | 34.2028% |
| 32 | 8 | 11.3172% | 33.5143% | 35.4489% | 34.2384% |

Moving from one to two conormal segments reduces the gradient and flux
differences in both local spaces. Further enrichment has a smaller benefit
for the gradient and does not monotonically reduce pressure or physical-flux
error. Doubling the local resolution also does not remove the approximately
34% energy difference at four or eight segments. These comparisons demonstrate
independent local enrichment at unchanged global dimension; they do not prove
uniform accuracy for this fixed pressure-trace approximation. The final
classical $P_3$ energy increment is 0.071857%, a separate reference-resolution diagnostic.


## Diagonal-mesh control: Figure-5 parameters

With $n=31$, $r=4$ and two segments in each trace space, the pressure-trace
system has 3,969 total and 3,721 free unknowns. Against the $P_3$, $n=512$
classical reference, its relative differences are 0.760848% in pressure,
18.1908% in the broken gradient, 20.6137% in physical flux and 19.8963%
in energy. Error quadrature orders 8 and 10 agree to within
$5\times10^{-16}$ relative in these four differences. The algebraic residual
is $9.64\times10^{-17}$ and the maximum macro balance defect is
$1.41\times10^{-16}$; these quantities test the discrete equations and
do not certify the approximation error.

The small pressure difference therefore coexists with a substantial gradient
and flux difference. The profile retains both the article's $n=128$ reference
resolution and the additional $P_3$, $n=512$ reference. The coefficient amplitude and source agree with the dissertation, while
the recorded local mesh is the stated diagonal refinement. These results
do not assert equality to the historical solution on the recovered crisscross
connectivity.


Profile segments retain independent incident values instead of joining
discontinuous endpoint values. Field panels show the actual macro partition.
Pressure differences are sampled at the plotted fine vertices; signed $q_x$
and $q_y$ maps use fine-cell centroids. These display samples do not define the volume
norms, which use the common overlay above.

## Reproduce

The [numerical record](../figures/mh2m-heterogeneous/comparison.json) stores
the explicit data convention, every space and mesh, field-archive digests,
executed source hashes, quadrature checks and physical residuals.

```bash
pixi run -e notebooks mh2m-heterogeneous-campaign \
  --references 32 64 128 256 512 1024 --norm-workers 8
pixi run -e fem python -m examples.mh2m_cg_reference --sizes 32 64 128 256 512 --order 16
pixi run -e fem python -m examples.mh2m_cg_reference --sizes 512 --order 12
pixi run -e notebooks mh2m-cg3-comparison
pixi run -e notebooks mh2m-cg3-controls
pixi run -e notebooks gallery-mh2m-cg3
pixi run -e notebooks python scripts/run_notebooks.py notebooks/darcy/70_mh2m_heterogeneous.ipynb
```

The $P_1$ controls use a separate global conforming assembly with the
package's shared finite-element kernels. The additional $P_3$ reference is
assembled independently with DOLFINx/UFL. Both are numerical references,
not exact solutions.

## References

- Franklin de Barros, Alexandre L. Madureira, and Frédéric Valentin (2026). *A three-field Multiscale Method*. arXiv preprint, version 3, 5 August 2026; first submitted 25 April 2024. [arXiv: 2404.16978v3](https://arxiv.org/abs/2404.16978v3).

- Franklin da Conceição de Barros (2022). *The Multiscale Hybrid-Hybrid-Mixed Method*. Master’s dissertation in Computational Modeling, Laboratório Nacional de Computação Científica, Petrópolis, Brazil, 77 pages. [Institutional dissertation](https://www.lncc.br/~alm/students/frankdissert.pdf).
