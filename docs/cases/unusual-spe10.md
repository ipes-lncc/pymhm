# SPE10 reaction layers with MHM-UNUSUAL

This case evaluates the scalar stabilized method of
[Santiago, Valentin and Martins (2025), §4.2](https://doi.org/10.55592/cilamce2025.v5i.14270)
on the stated SPE10 layer and compares physical fields against a separately
refined **DOLFINx/UFL conforming CG2** solution. The nominal macro partition,
local degree and skeleton degree follow the paper. Material fitting and the
additional resolution controls are explicit choices. The source is set to zero;
the heterogeneous subsection does not state its value. Consequently, this is
a fully specified comparison of the published formulation, not a claim of
identical historical input files or Figure 3 reproduction.

## Physical problem and discrete spaces

The domain is the original numerical rectangle, in source-coordinate units:

$$
\begin{aligned}
\Omega&=(0,1200)\times(0,2200),\\
-\nabla\cdot(K\nabla p)+p&=0,\qquad q=-K\nabla p,\\
p(x,0)&=1,\qquad p(x,2200)=0,\\
q\cdot n&=0\quad\text{on the vertical sides}.
\end{aligned}
$$

Here \(K=K_x I\) uses **SPE10 Model 2 layer 36**, in both coordinate
directions. Coordinates retain the original foot-valued numbers and permeability
retains the original millidarcy-valued numbers. The reaction coefficient is one
in this numerical convention; there is no implicit conversion to SI units.

The macro grid has 128 SW–NE triangles, from eight rectangles per coordinate
direction. Each local space is continuous P1; each macroface is divided into
\(s\) discontinuous P0 pieces. The paper states \(H=\mathcal H/8\) and
\(h=H\), giving the nominal \(r=s=8\) control. The local mesh is then fitted
to the permeability pixels. This adds local triangles without moving the macro
partition or adding skeleton moments. The actual fine-cell count is therefore
larger than \(128r^2\).

Fitting is substantive for the strong residual. On each fitted triangle,
\(K\) is constant and \(p_h\) is affine, so
\(\nabla\cdot(K\nabla p_h)=0\) inside that triangle. Across a permeability
jump, the distributional operator cannot instead be treated as an ordinary
volume function on an unresolved P1 cell. The [UNUSUAL operator page](unusual.md)
states the complete formulation and admissibility conditions.

The element parameter is retained without adjustment:

$$
\tau_T=
\frac{h_T^2/3}{\max\{h_T^2/3,2K_T\}+2K_T}.
$$

For the piecewise-constant material and affine local functions, the element
bilinear form reduces exactly to

$$
a_T^{\mathrm{UN}}(p_h,v_h)
=\int_T K_T\nabla p_h\cdot\nabla v_h
 +(1-\tau_T)p_hv_h.
$$

This is the unchanged published residual stabilization, not an independently
selected reaction coefficient. When $h_T$ greatly exceeds $\sqrt{K_T}$,
$\tau_T$ approaches one, so the discrete reaction contribution is small.
For the layer-resolution criterion $h_T\leq c\sqrt{K_T}$, with
$c\leq\sqrt{6}$, the same formula gives

$$
\tau_T=\frac{h_T^2}{12K_T}\leq\frac{c^2}{12}.
$$

Thus $c=0.5$ and $c=0.25$ bound this local modification by $1/48$ and
$1/192$, respectively, in the refined layer.
The estimate explains a spatial-resolution requirement of the fixed method;
it is neither an error estimate nor a discrete maximum-principle guarantee.

All MHM results on this page impose the exterior pressure **weakly through
skeleton moments**, as in the hybrid equations (17)–(19). Side fluxes use the
outward physical convention. An algebraic residual verifies these discrete
equations; it does not establish a maximum principle or physical accuracy.

## Independently refined classical reference

The reference is an independently assembled, globally continuous triangular
P2 Galerkin solution with the same material, reaction, load and mixed boundary
conditions. DOLFINx 0.9.0/UFL assembles the forms and PETSc/MUMPS solves the
Dirichlet-eliminated SPD system. Its coordinate axes include every permeability
interface. Additional grading near the inlet and material transitions resolves
the short reaction length \(\ell_K=\sqrt{K}\).

The final reference has 1,434,240 triangles and 2,872,357 unknowns. The last
720×249 → 1440×498 refinement gives the following relative increments, each
normalized by the finer field:

| Physical norm | Last reference increment |
|---|---:|
| Pressure \(L^2\) | 0.11554% |
| Flux \(L^2\) | 0.31754% |
| Diffusion–reaction energy | 0.28015% |

The assembled-matrix free-equation residual is \(1.036\times10^{-15}\).
An independent evaluation of the elementwise diffusion–reaction form on the
reloaded coefficients gives \(4.316\times10^{-15}\), including every
unprescribed equation on the natural lateral boundaries. It evaluates diffusion
through nodal differences and reaction through exact P2 mass moments.
The [physical-form verification](../figures/unusual-spe10/physical-form-verification.json)
also records an energy/work relative defect of \(1.70\times10^{-14}\) and
an absolute defect of \(1.14\times10^{-11}\) in outward boundary flux plus
the integrated reaction term. All eight archived classical fields pass this
independent check, with maximum relative residual \(6.12\times10^{-15}\).
For consecutive nested meshes, the independently integrated squared energy
difference also agrees with the difference of the two discrete energies.
These checks do not make the reference exact; its remaining refinement
increment accompanies all MHM comparisons.

Uniform CG2 meshes are retained as resolution controls. Even 240×880 rectangles
give a 46.84% flux increment from the preceding uniform mesh. They are therefore
not used as the final denominator. The final graded reference has pressure
minimum \(-7.63\times10^{-6}\) and maximum one; no values are clipped.

## Norms and separated resolution controls

For the stated numerical reference \(p_{\rm ref}\), let
\(e=p_h-p_{\rm ref}\). The recorded physical norms are

$$
\begin{aligned}
E_p^2&=\int_\Omega e^2,\\
E_q^2&=\int_\Omega\lVert-K\nabla p_h-q_{\rm ref}\rVert^2,\\
E_A^2&=\int_\Omega K\lVert\nabla e\rVert^2+e^2.
\end{aligned}
$$

Every local MHM triangle is intersected with the reference rectangles and
their diagonals before integration. Both meshes are material-fitted. Positive
Duffy rules of orders three and four integrate the resulting polynomial
expressions and provide an independent quadrature check. Relative quantities
divide by the corresponding physical norm of the same CG2 reference. Pixel
samples and plotted profiles are not used as substitutes for these integrals.

Local refinement \(r=8,16,32\) at fixed \(s=8\) and trace refinement
\(s=8,16,32\) at fixed \(r=32\) are recorded separately. A further local
control refines triangles intersecting \(y<6\ell_K\) until
\(h_T\leq0.5\ell_K\), using conforming longest-edge refinement. It preserves
P1, the material, the stabilization formula and the selected skeleton. The
combined layer/\(r32,s32\) case changes both resolutions and is labeled as
such, rather than attributing its difference to one approximation space.
An additional trace control retains that exact fine mesh and adds P0 segments
at material crossings of the macrofaces. Each face then has 39–66 segments,
depending on its geometry, instead of uniformly 32. The local P1 functions and
stabilization parameters remain identical.

The nominal \(r8,s8\) field differs from the final reference by **355.20%**
in pressure, **88.24%** in raw flux and **259.31%** in energy. Its pressure
range is \([-0.27181,3.06421]\). Thus the nominal spaces in this explicit
zero-source problem do not resolve the physical reaction layer. This conclusion
uses integrated norms and the independently refined field; it is not inferred
from a residual or a visual overshoot alone.

The following controls retain the stabilization and physical data. Percentages
are differences from the same final CG2 field, with its own increments reported
above. “Layer” denotes the explicit \(h_T/\ell_K\leq0.5\) local criterion,
unless another value of $c$ is specified.

| Local control | Trace segments | Fine triangles | Pressure \(L^2\) | Raw flux \(L^2\) | Energy |
|---|---:|---:|---:|---:|---:|
| \(r=8\) | 8 | 182,976 | 355.195% | 88.239% | 259.315% |
| \(r=16\) | 8 | 338,752 | 356.879% | 89.646% | 260.654% |
| \(r=32\) | 8 | 711,808 | 351.934% | 91.088% | 257.501% |
| \(r=32\) | 16 | 711,808 | 330.964% | 90.549% | 243.082% |
| \(r=32\) | 32 | 711,808 | 326.695% | 90.153% | 240.073% |
| Layer, \(r=8\) | 8 | 862,481 | 22.105% | 23.700% | 23.468% |
| Layer, \(r=32\) | 32 | 1,424,837 | 9.820% | 14.749% | 12.529% |
| Layer, \(r=32\), material-fitted trace | 39–66 | 1,424,837 | 4.335% | 9.262% | 7.304% |
| $c=0.25$, $r=32$, same material-fitted trace | 39–66 | 3,422,735 | 4.194% | 5.392% | 4.899% |

Uniform local and skeletal refinement over these ranges leave the short
reaction length unresolved. Resolving that length locally produces a much
larger reduction than those nominal refinements. The final control still has
substantial physical-field differences, so it is not a converged approximation.
In particular, neither the stabilization nor a small discrete residual enforces
a maximum principle on these spaces. The layer-control pressure ranges are
\([-0.02823,2.63593]\), \([-0.06712,2.07226]\) and
\([-0.07161,1.43001]\), respectively. Adding material-crossing trace moments
at the last fixed fine mesh reduces the flux difference from **14.7485%** to
**9.2618%**. This is a separate skeletal-space comparison.

At this fixed skeletal space, reducing $c$ from $0.5$ to $0.25$ changes the
flux difference from **9.2618%** to **5.3916%** and the physical energy difference
from **7.3044%** to **4.8988%**. This is a local-resolution control; the
stabilization formula and its parameter remain prescribed, while $h_T$ and
therefore the elementwise value of $\tau_T$ follow the refined geometry.
The $c=0.25$ hybrid residual is $5.69\times10^{-16}$ and its two norm rules
agree within $4.7\times10^{-16}$ relatively. Its pressure range remains
$[-0.071375,1.435222]$. These checks accompany, rather than replace, the
remaining discretization difference.

The [physical trace-moment verification](../figures/unusual-spe10/trace-moment-verification.json)
measures those extrema on the independently represented macro sides. At
$(0,0)$, the incident values are $1.435222$ and $0.014286$, while the prescribed
boundary pressure is one. At the interior diagonal point
$(1053.515625,6.4453125)$, the two incident values are $-0.071375$ and
$0.075175$. These values are retained without averaging. The bottom weak
Dirichlet moments nevertheless have maximum absolute defect
$8.0\times10^{-15}$, and the integrated boundary difference is

$$
\frac{\lVert p_h-1\rVert_{L^2(y=0)}}{\sqrt{1200}}
=1.3365\%.
$$

A finite set of boundary and continuity moments does not prescribe endpoint
values or remove every jump between those moments. These measurements
identify unresolved traces in addition to the local reaction-layer error;
they do not establish a maximum principle. Skeletal enrichment must also
remain compatible with the local trace spaces: face-supported multiplier
modes that annihilate both incident local traces make the hybrid operator
singular.

The two $c=0.5$ layer-resolved $r32$ fields also satisfy the nested-trace Galerkin
identity in their unchanged stabilized form:

$$
a^{\mathrm{UN}}(p_f-p_c,p_f-p_c)
=a^{\mathrm{UN}}(p_f,p_f)-a^{\mathrm{UN}}(p_c,p_c).
$$

The [independently integrated check](../figures/unusual-spe10/trace-energy-verification.json)
gives energies $488.1224210$ and $493.2077812$, with squared difference
$5.0853601$ and relative identity defect $1.04\times10^{-14}$.
This verifies the change of skeletal constraints on identical local spaces.
Its stabilized energy is distinct from the physical diffusion–reaction norm
used against the CG2 reference above.

![Separated local, skeletal and reaction-layer resolution controls](../figures/unusual-spe10/resolution.png)

The full profile and two inlet enlargements retain independent macro-side
values. The legend occupies a separate region below the profiles. Vertical
dotted lines mark actual macroface crossings. Every crossed fine edge is
included, so the plotted P1 restriction preserves its breakpoints.

![Complete pressure profile and two inlet scales](../figures/unusual-spe10/profiles.png)

The field panels show the bottom 20 coordinate units and have an expanded
vertical display scale. The pressure uses common limits, and the signed
vertical flux uses common symmetric limits. Macro boundaries remain visible.
Neither the pressure extrema nor the raw fluxes are clipped; the displayed
cell fluxes are one-sided values, not smoothed interpolants.

![Pressure and signed vertical flux in the inlet layer](../figures/unusual-spe10/boundary-layer.png)

The complementary first-foot view uses the same field samples and common
linear pressure and flux color limits as the 20 ft view. Its expanded vertical
scale shows variation across the reaction layer without clipping the nominal field or
rescaling individual methods. Actual macro edges remain visible.

![Pressure and signed vertical flux in the first foot](../figures/unusual-spe10/reaction-layer.png)

The physical flux shown here is \(-K\nabla p_h\). It is a broken raw-gradient
field, not an H(div) reconstruction. The stabilized discrete constant-test
equation is also distinct from an unstabilized fine-cell conservation identity.

## Reproducible records

The original acquisition and replay programs are:

```bash
pixi run --locked -e test-core python -m examples.solve_unusual_spe10 --refinement 8 16 32 --segments 8
pixi run --locked -e test-core python -m examples.solve_unusual_spe10 --refinement 32 --segments 16 32
pixi run --locked -e test-core python -m examples.solve_unusual_spe10 --refinement 32 --segments 32 --layer-resolution 0.5
pixi run --locked -e test-core python -m examples.solve_unusual_spe10 --refinement 32 --segments 32 --layer-resolution 0.5 --trace-fitted
pixi run --locked -e test-core python -m examples.solve_unusual_spe10 --refinement 32 --segments 32 --layer-resolution 0.25 --trace-fitted --max-local-cells 1000000
```

`examples/solve_unusual_spe10_reference.py` acquires the independent classical
reference. `examples/compare_unusual_spe10.py` integrates saved fields on their
exact geometric overlay. Records in `examples/results/unusual-spe10` identify
the executed source and field digests separately for PDE acquisitions and norm
evaluation. Their coefficient arrays retain each macro's independent local
mesh and one-sided values.

The [field contract](../figures/unusual-spe10/field-sampling-verification.json)
identifies the coefficient arrays, material and axes independently of display
samples. Flux profiles retain both incident triangle limits at their common
endpoint; a single point sample uses the lower triangle and the right/up
rectangle at an exact interface. Volume norms integrate the geometric
intersections and do not select sides through displaced sampling points.
The [consumer verification](../figures/unusual-spe10/consumer-replay-verification.json)
links all nine MHM comparisons and five classical refinement differences to
their executed inputs and the current field archives. The
[ordered-reduction check](../figures/unusual-spe10/norm-reduction-verification.json)
compares cell blocks and whole macroelement tasks for the complete r8/s8 field:
both quadrature orders reproduce all nine physical norms bit for bit.

A complete independent DOLFINx/UFL comparison assembles the declared SPE10
layer-36 problem with the same 128 macrotriangles, fitted r8/s8 P1/P0 spaces,
original UNUSUAL stabilization, source zero and weak boundary moments.
Its uncondensed PETSc/MUMPS system contains 97,440 pressure coordinates and
1,536 free trace coordinates. Relative physical L2 differences against the
archived MHM field are $2.20\times10^{-14}$ in pressure and
$6.77\times10^{-14}$ in Darcy flux; integration orders three and five agree.
The original-system residual is $1.45\times10^{-16}$.
The [same-case numerical record](../figures/unusual-spe10/native-discrete-verification.json)
identifies the full field, material, comparison-module digest and verified
DOLFINx 0.9.0 source provenance. This establishes discrete agreement for the
explicitly declared source; the historical volume source remains unspecified.
Smooth-problem convergence rates are not assigned to this discontinuous
reservoir coefficient.

The supplied **MHMUN-RAD_Parallel / FreeFem++** application is independently
verified on the analytical configurations described on the operator page.
It does not contain an executable SPE10 fixture with the paper's complete
input data. That analytical code comparison is therefore not presented as an
execution of the heterogeneous figure.

## References

- Juan Felipe Pacazuca Santiago, Frédéric Valentin, and Larissa Martins (2025). *A Multiscale Hybrid-Mixed Method with Local Stabilization*. Proceedings of the Ibero-Latin American Congress on Computational Methods in Engineering, CILAMCE 2025, volume 5, article 14270; published online 18 March 2026. [DOI: 10.55592/cilamce2025.v5i.14270](https://doi.org/10.55592/cilamce2025.v5i.14270).
