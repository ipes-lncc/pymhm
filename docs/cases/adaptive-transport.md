# Conservative transport and face adaptation

The scalar operator is

$$
-\nabla\!\cdot(K\nabla u)+\nabla\!\cdot(\beta u)+cu=f.
$$

The local skew form uses the effective reaction
\(c+\tfrac12\nabla\!\cdot\beta\). Consequently, a negative reaction is admissible
when this combined coefficient is nonnegative. For example, \(\beta=x\) in
two dimensions and \(c=-1/2\) give effective reaction \(1/2\).
`solve_transport` checks this condition at its integration points; the caller
remains responsible for a coefficient bound throughout each element.

The multiplier on an interior macroface represents
\((-K\nabla u+\beta u/2)\cdot n\). It is neither the diffusive flux nor the
total conservative flux. Exterior data are explicit: `neumann` prescribes this
Robin multiplier, while `diffusive_flux` prescribes \((-K\nabla u)\cdot n\).
The latter changes the exterior local bilinear form by
\(+\tfrac12\langle\beta\cdot n\,u,v\rangle\).

Variable velocity requires its analytical `velocity_divergence`. SUPG with
variable diffusion additionally requires `diffusion_divergence`. The strong
residual contains both \((c+\nabla\!\cdot\beta)u\) and
\(-\nabla\!\cdot K\cdot\nabla u\). Neither derivative is replaced by zero.
SUPG is a consistent streamline residual method; it does not establish a
discrete maximum principle.

## Published analytical transport problem

Section 5.1 of [Harder, Paredes and Valentin (2015)](https://doi.org/10.1137/130938499)
uses the unit square, \(K=\epsilon I\), \(\beta=(a,0)\), \(f=1\), zero Dirichlet
data at \(x=0,1\), and homogeneous diffusive flux at \(y=0,1\).
For \(a=1\), \(c=0\), the exact solution is

$$
u(x,y)=x-\frac{e^{(x-1)/\epsilon}-e^{-1/\epsilon}}
                   {1-e^{-1/\epsilon}}.
$$

The recorded five-level campaign uses \(\epsilon=0.02\), macro resolutions
\(n=1,2,4,8,16\), continuous local P3 fields on sixteen fine triangles per
macrotriangle, and one constant multiplier per macroface. Local problems use
SUPG and strong nodal Dirichlet conditions. These are specified discretizations
of the published PDE; the campaign does not identify its curves with a table or
figure generated on different meshes in the article.

| Macro subdivisions | Macrotriangles | L² error |
|---:|---:|---:|
| 1 | 2 | 0.28375032 |
| 2 | 8 | 0.17683636 |
| 4 | 32 | 0.08221265 |
| 8 | 128 | 0.03637914 |
| 16 | 512 | 0.02179689 |

Errors use order-12 integration; the largest difference from order 10 is
\(1.86\times10^{-8}\), on the coarsest mesh.


## A face residual and a closed adaptive loop

`estimate_transport_faces` implements equations (4.3)–(4.4) of the same article:

$$
\begin{aligned}
R_F&=-\tfrac12[u_h],\qquad
\eta_{\widehat F}=\frac{C}{\sqrt{H_F}}\|R_F\|_{L^2(\widehat F)},\\
C&=\max\{\sqrt{K_{\max}},d_\Omega\|\beta\|_\infty,
              d_\Omega^2\|c+\tfrac12\nabla\!\cdot\beta\|_\infty\}.
\end{aligned}
$$

Here \(H_F\) is the length of the original macroface, including after repeated
subdivision into segments \(\widehat F\). The coefficient bounds are declared
with `TransportBounds`; maxima sampled at quadrature points do not certify
these bounds. Integration splits at the fine partitions on both sides, preserving
the separate traces of the broken solution.

Dirichlet conditions are imposed strongly on the essential boundary. Prescribed
natural faces are supported as well. The multiplier variations vanish on those
faces, so their exterior jump indicators are zero and their prescribed data are
unchanged by adaptation. This is the mixed-boundary construction of section
2.1.2, equations (2.14)–(2.19). Equation (4.3) itself lists only interior and
Dirichlet faces; the zero contribution on a prescribed natural face is the
extension obtained by removing its multiplier variations. On the horizontal walls of section 5.1,
\(\beta\cdot n=0\), so the physical diffusive flux and the half-advection Robin
multiplier coincide. On other faces they generally differ; the solver retains
the boundary bilinear term associated with the selected convention.

The adaptive loop marks segments satisfying
\(\eta_{\widehat F}\ge\theta\max\eta\), with default \(\theta=3/4\), and bisects
only those segments. Macrogeometry and local meshes remain fixed. Polynomial
degree, continuity within a macroface, and scalar/vector component counts are
preserved by the reusable `refine_skeleton_faces` operation.

## Mixed-wall adaptation with the Figure 12 physical data

The current mixed-wall campaign uses \(\epsilon=0.1\), \(a=1\), \(c=0\), and
\(f=1\), as in Figure 12. It preserves zero Dirichlet data at \(x=0,1\) and zero
**diffusive** flux at \(y=0,1\) throughout all adaptive states. The local spaces
are continuous P1 with Galerkin assembly, and the skeletal spaces are
discontinuous P0 on each segment. This is the \(\Lambda_0\)/P1 member of the
family specified in section 5; this campaign does not add SUPG. The Figure 12
caption and its discussion do not restate the polynomial degree or the local
mesh resolution.

The initial macro mesh contains sixteen crisscross triangles from a \(2\times2\)
square grid. Each macrotriangle has \(r^2=256\) fine triangles. Eight solves use
the same macro and local meshes, with \(\theta=3/4\). The global count below is
the number of **free interior multipliers**, excluding prescribed exterior
coordinates. Selective retention introduces no extra coarse amplitudes for
these advective local operators.

| State | Free trace DOFs | Face indicator | Absolute L² error | Broken H¹ seminorm error |
|---:|---:|---:|---:|---:|
| 0 | 20 | 0.42238474 | 0.05366185 | 0.72481413 |
| 1 | 24 | 0.20719280 | 0.02768635 | 0.38715575 |
| 2 | 32 | 0.11041036 | 0.01545116 | 0.28491666 |
| 3 | 40 | 0.04245884 | 0.00365079 | 0.18034753 |
| 4 | 44 | 0.03019193 | 0.00262073 | 0.16377096 |
| 5 | 60 | 0.01949550 | 0.00190055 | 0.14681987 |
| 6 | 64 | 0.01449228 | 0.00154093 | 0.14340390 |
| 7 | 72 | 0.00807510 | 0.00082631 | 0.13879015 |

The two errors are

$$
\begin{aligned}
E_0^2&=\sum_K\|u-u_h\|_{L^2(K)}^2,\\
E_1^2&=\sum_K\|\nabla u-\nabla u_h\|_{L^2(K)}^2.
\end{aligned}
$$

They are absolute norms against the analytical solution, without material
weighting or a relative denominator. Orders 8 and 12 differ by at most
\(1.12\times10^{-16}\) over the eighteen recorded solves. The maximum normalized
continuity moment is \(1.42\times10^{-14}\), the largest original hybrid-equation
residual is \(1.01\times10^{-16}\), and the prescribed wall multipliers are zero.
Continuity of the tested moments is distinct from pointwise continuity or
fine-cell conservation of the raw gradient.


Uniform macro refinement uses the same P1/Galerkin/P0 family and \(r=16\).
Its eight meshes contain 4, 16, 36, 64, 144, 256, 576 and 1024 macrotriangles,
with 4, 20, 48, 88, 204, 368, 840 and 1504 free multipliers.
At 88 free multipliers its L² error is 0.0182393,
whereas face adaptation reaches 0.000826309 with 72. The local and macro meshes
are explicitly specified here; the historical Figure 12 does not supply a
complete initial adaptive connectivity and local-refinement sequence. Matching
the physical problem and space family therefore does not establish equality
with its historical numerical points.

The [Figure 12 graphical data](../figures/transport/published-figure-12.json)
contain the original vector-marker coordinates, logarithmic-axis calibration
and independent graphical intervals. Nineteen markers are visible. The final
L² mesh-curve endpoint lies under the original legend; it is identified as a
recoverable vector endpoint rather than a visible marker. No numerical result
from PyMHM is used to locate a point or calibrate an axis.

![Absolute errors compared with the original Figure 12 curves and their graphical intervals](../figures/transport/mixed-published-comparison.png)

The first published mesh point has \(N_1\simeq367.56\), compatible with the
368 free moments of the \(n=8\) crisscross mesh. Its L² error is approximately
\(4.43519\times10^{-4}\), whereas the nominal \(\epsilon=0.1\) calculation
gives \(5.22937\times10^{-3}\). The corresponding broken gradient errors are
0.0177968 and 0.239493. This is a quantitative disagreement, not a difference
hidden by relative normalization.

An independent DOLFINx/UFL assembly of all 39,168 broken local P1 coordinates
and 368 P0 moments verifies the implemented discrete problem. It uses direct
geometric face integration, strong boundary elimination and a complete saddle
solve, independently of the local condensation. Its nodal field differs from
the archived PyMHM field by at most \(3.91\times10^{-13}\). This verifies the
specified discrete system; it does not establish the historical inputs.

For the nominal fixed $\epsilon=0.1$, the spatial sequence
$n=1,2,3,4,6,8,12,16$ approaches the expected smooth-solution orders two
in L2 and one in the broken H1 seminorm. The last $n=12\to16$ pair gives
1.9298 and 0.9573, using
$\log(E_{12}/E_{16})/\log(16/12)$ and unchanged P1/P0/r16 spaces.
The exact layer is smooth for fixed positive diffusion; these rates concern
the resolved-layer regime. The
[nominal native and rate record](../figures/transport/nominal-native-rates.json)
verifies all eight field hashes and preserves the earlier complete UFL
comparison separately from the current provenance audit. The measured
orders do not reconcile the absolute Figure 12 errors or specify its
historical second-level mesh.

At the same \(n=8\) macro mesh and the same 368 interior moments, local
refinement gives:

| Local subdivision r | Absolute L² error | Broken H¹ seminorm error |
|---:|---:|---:|
| 16 | 0.0052293720 | 0.23949266 |
| 32 | 0.0052247705 | 0.23754144 |
| 64 | 0.0052236508 | 0.23704724 |

The changes from \(r=16\) to \(r=64\) are 0.109% and 1.021%, respectively.
Thus the measured local-resolution dependence in this sequence does not
explain the much larger disagreement with the first Figure 12 marker. The
[fixed-macro local controls](../figures/transport/mixed-nominal-local-resolution.json)
retain the same diffusion, advection, boundary data and skeletal moments.

### Explicit coefficient-regime controls

Equation (5.1) specifies \(K=\epsilon I\), and Figure 12 prints
\(\epsilon=10^{-1}\), not \(\epsilon^2\). A separate control uses
\(\epsilon=1\), the regime explicitly identified in Figure 7 of the same
article. Every other physical datum, the P1/Galerkin/P0 family, the crisscross
construction and the error definitions remain unchanged. These are different
physical problems; no error or axis is rescaled between them.

Five uniform macro refinements with \(r=16\) give:

| Macro subdivisions n | Free interior multipliers | Absolute L² error | Broken H¹ seminorm error |
|---:|---:|---:|---:|
| 8 | 368 | 4.5215994e−4 | 0.018068755 |
| 16 | 1504 | 1.1399103e−4 | 0.0090841830 |
| 32 | 6080 | 2.8616243e−5 | 0.0045545280 |
| 64 | 24448 | 7.1688499e−6 | 0.0022803719 |
| 128 | 98048 | 1.7940637e−6 | 0.0011409628 |

All ten error values lie within the independently extracted intervals of the
Figure 12 **mesh** curve. These comprise nine visible markers and the final
L² endpoint beneath the legend. Their departures from the central graphical
ordinates range from −0.169% to +1.948%. The
[unscaled comparison record](../figures/transport/mixed-published-comparison.json)
preserves both abscissa and ordinate intervals, every field digest and the
nominal \(\epsilon=0.1\) comparison. This agreement supports compatibility of
the mesh curve with the separately published \(\epsilon=1\) regime; it does
not establish the inputs used by the authors or the cause of the incompatible
Figure 12 caption.

At fixed \(n=8\), increasing local subdivision from 16 to 32 changes the L²
error by +0.125% and the gradient error by −0.178%. This finite-local control
is distinct from the graphical reading uncertainty.

The independent complete DOLFINx/UFL saddle assembly also verifies
\(\epsilon=1\), \(n=8\), \(r=16\). Its maximum nodal difference from the
archived field is \(6.64\times10^{-14}\), with a maximum tested continuity
moment of \(1.64\times10^{-16}\). The comparison includes the original local
equations and the global multiplier coupling rather than only a local matrix.

The coefficient controls retain local constants as algebraic coordinates.
Their elimination preserves the physical approximation space; on the
\(n=8\) control, the fields agree with selective retention within
\(6.41\times10^{-14}\). The abscissae count only the free interior
multipliers. The retained-coordinate count is recorded separately and is not
added to the published first-level count. Compatibility of these counts with
the graphical abscissae does not identify the historical mesh connectivity.

### Face subdivision and finite local approximation

The maximum-indicator rule and uniform subdivision of every macroface are
distinct controls. The literal rule uses \(\theta=3/4\) and the individual
segment indicators; it produces 368, 528, 608, 624, 880, 1120, 1136 and 1408
free moments in the recorded \(\epsilon=1\), \(n=8\), \(r=16\) sequence.
Uniform subdivision uses \(s=1,2,4,8,16\), hence \(368s\) free moments.
The article refers to indicator-based refinement; the geometric progression
of the graphical DOF counts alone does not establish uniform refinement.


For every continuous local P1 field, its gradient is constant in each fine
triangle. Therefore an independent lower bound on its gradient error is

$$
\begin{aligned}
E_1^2&\ge B_r^2,
& B_r^2&=\sum_{T\in\mathcal T_h}
  \int_T\left\|\nabla u-\overline{\nabla u}_T\right\|^2,\\
\overline{\nabla u}_T&=\frac{1}{|T|}\int_T\nabla u.
\end{aligned}
$$

This bound uses the larger, unconstrained DG0 gradient space; its projection
need not be the gradient of an admissible solution. The reported integrals use
two quadratures and the squared difference directly. Their agreement is an
integration check, not a certified enclosure. The bound is independent of the
skeletal moments, so increasing their number cannot remove this P1 local
approximation floor. It characterizes the declared meshes, not an unknown
historical local resolution or an error in the publication.

Keeping the final maximum-marked space fixed at 1408 free moments gives:

| Local subdivision r | Absolute L² error | Broken H¹ seminorm error | DG0 gradient lower bound |
|---:|---:|---:|---:|
| 16 | 4.6379323e−6 | 0.0021796063 | 0.0013539605 |
| 32 | 3.2034087e−6 | 0.0014060548 | 0.0006769809 |
| 64 | 3.3772141e−6 | 0.0011570594 | 0.0003384905 |
| 128 | 3.4659365e−6 | 0.0010927951 | 0.0001692453 |

The macrogeometry and all skeletal breakpoints are identical in these four
calculations. Local refinement reduces the gradient error, with a 5.55% change
from the last two resolutions, while the L² error is not monotone. The indicator
does not include this finite local approximation error. In particular, even
the DG0 lower bound at \(r=128\) exceeds the final published space-curve
gradient error of approximately \(1.18913\times10^{-4}\). That endpoint cannot
be reached by refining only the trace of these fixed P1 meshes. This conclusion
does not determine the local discretization used for the original figure.

For comparison, fixing the uniform skeletal space at 5888 free moments gives:

| Local subdivision r | Absolute L² error | Broken H¹ seminorm error |
|---:|---:|---:|
| 16 | 4.2701738e−6 | 0.0020287414 |
| 32 | 1.0677572e−6 | 0.0010156042 |
| 64 | 2.7016811e−7 | 0.00051826458 |
| 128 | 1.0656683e−7 | 0.00028549271 |
| 256 | 9.6542188e−8 | 0.00018940387 |

The two fixed skeletal spaces have different approximation errors. The
uniform-space gradient error decreases by 33.66% from \(r=128\) to
\(r=256\), while the L² error decreases by 9.41%. At \(r=256\), the
DG0 lower bound is \(8.46226\times10^{-5}\), below the final published
gradient ordinate. The computed gradient error nevertheless remains 59.28%
above that ordinate, and the L² error remains 5.91% above its marker. Seven
of the ten uniform-space error values lie inside the independent graphical
intervals. This is partial agreement for the separate \(\epsilon=1\)
control, not reproduction of the complete curve or its printed coefficient.
A small L² error alone does not establish local gradient accuracy.

![Local approximation errors on two fixed skeletal spaces, with the independent DG0 gradient lower bound](../figures/transport/mixed-local-resolution.png)

The difference between successive **fields**, rather than the difference
between their scalar error norms, is also integrated on their nested fine
triangles. For the fixed 5888-multiplier space:

| Local subdivisions | Scalar field increment in L² | Gradient field increment in L² |
|---:|---:|---:|
| 64 → 128 | 2.1284427e−7 | 4.4597080e−4 |
| 128 → 256 | 5.4462520e−8 | 2.2553158e−4 |

The last gradient increment is 119.07% of the finest field's exact-gradient
error. Thus local resolution still materially affects this endpoint, despite
the smaller difference between successive error norms. The
[field-increment record](../figures/transport/mixed-local-field-increments.json)
contains both increments for all five fixed skeletal spaces. Each fine
triangle is checked to lie inside its actual coarse parent; affine P1
prolongation preserves independent macro-local fields. Exact P1 mass
integration agrees with quadrature orders eight and twelve. These increments
measure resolution sensitivity and are not bounds on an uncomputed limit.


![Analytical, initial and final fields with the actual macro mesh; red dots mark added trace breakpoints](../figures/transport/mixed-fields.png)


For the \(\epsilon=0.1\) adaptive sequence, the final 72-DOF skeletal space is
also solved on three local resolutions:

| Local subdivision r | Fine triangles per macro | Face indicator | Absolute L² error | Broken H¹ seminorm error |
|---:|---:|---:|---:|---:|
| 8 | 64 | 0.00619788 | 0.00270344 | 0.26800464 |
| 16 | 256 | 0.00807510 | 0.00082631 | 0.13879015 |
| 32 | 1024 | 0.00971317 | 0.00058272 | 0.08053943 |

This control isolates local approximation from the fixed trace restriction.
The gradient error still depends substantially on local resolution. Here the
jump indicator increases while both errors decrease, confirming that it does
not measure the local approximation contribution. In particular, Figure 12's
discussion of exactly solved second-level problems must not be applied to
these finite P1 local meshes as an established accuracy guarantee.

The [numerical record](../figures/transport/mixed-campaign.json) identifies every
field archive and source digest. The [verification record](../figures/transport/mixed-verification.json)
reports the mixed-boundary regressions and independent DOLFINx/UFL matrix and
load comparisons. The latter include nonzero normal advection and nonhomogeneous
natural data, so they distinguish the Robin and physical-flux conventions.
The [trace replay record](../figures/transport/scalar-trace-replay.json) verifies
the archived continuity moments and segment indicators with prepared,
immutable face geometry. It preserves independent one-sided traces and the
declared endpoint ownership; no averaging is introduced at macrointerfaces.

## Separate full-Dirichlet control

The following analytical adaptive control uses \(\epsilon=0.02\) and prescribes
the exact field on the **entire boundary**. Its top and bottom data differ from
section 5.1 and from the mixed-wall campaign above. It uses 32 macrotriangles,
64 local triangles per macro, P3/SUPG local fields, and five solves with face
adaptation. Its exact-field error is evaluated independently of the indicator.

| Iteration | Trace DOFs | Face indicator | L² error |
|---:|---:|---:|---:|
| 0 | 56 | 0.72617664 | 0.05200088 |
| 1 | 59 | 0.58817082 | 0.04415063 |
| 2 | 61 | 0.59617350 | 0.04784078 |
| 3 | 62 | 0.56536858 | 0.04538293 |
| 4 | 78 | 0.21204561 | 0.01958710 |

The final error is smaller, while the intermediate sequence is nonmonotone.
This distinction matters: marking by a jump indicator is not a proof of error
contraction at every iteration.


In both studies, the displayed indicator is a refinement criterion. It does not
include local-discretization error or approximation error in nonrepresentable
boundary data. The implementation does not
assert a computable constant-one upper bound, monotonic error reduction for all
inputs, or a maximum principle. Trace enrichment must remain compatible with
the local trial space; incompatible spaces are rejected by the common numerical
rank checks. No hidden local refinement or tolerance adjustment is made.

## Reproduce the measurements

```bash
pixi run -e notebooks python -m examples.transport_mixed_campaign
pixi run -e notebooks python -m examples.transport_mixed_campaign \
  --append-spatial 8 12 16
pixi run -e notebooks python -m examples.transport_coefficient_controls \
  --resolutions 8 16 32 64 128 --epsilon 1 --local-refinement 16 --local-workers 4
pixi run -e notebooks python -m examples.transport_face_controls \
  --strategy adaptive --iterations 8 --local-refinement 16
for r in 16 32 64 128 256; do
  pixi run -e notebooks python -m examples.transport_face_resolution \
    --local-refinement "$r" --workers 2
done
for r in 32 64 128; do
  pixi run -e notebooks python -m examples.transport_adaptive_resolution \
    --local-refinement "$r" --workers 2
done
pixi run -e notebooks python -m examples.verify_transport_published
pixi run -e notebooks python -m examples.plot_transport_mixed
pixi run -e notebooks python -m examples.transport_campaign --collect
```

The mixed campaign acquires the nominal Figure-12 physical data and its
separate controls. The coefficient and face commands acquire the explicitly
different \(\epsilon=1\) controls, including fixed-skeleton local refinement.
These multilevel acquisitions are separate from the lightweight CI tests.
`verify_transport_published` checks field digests and compares unscaled errors
with the independently extracted graphical intervals; the plotting command
only reads archives and redraws the figures.

The last command acquires the separate P3 stationary/adaptive and transient
controls. Without `--collect`, `examples.transport_campaign` only regenerates figures from
`examples/results/transport/campaign.json` and its field archives. The record
contains two independent error quadrature orders, all five spatial resolutions,
all adaptive states, and the temporal study described in
[Darcy-coupled transient transport](transient-transport.md).
