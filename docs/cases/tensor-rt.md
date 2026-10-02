# Rectangular RT spaces with interior enrichment

This case reproduces the mixed-space configuration of Figure 3 in
[Duran et al. (2019)](https://doi.org/10.1016/j.cma.2019.05.013).
The macro mesh consists of uniform squares, each containing four fine rectangles.
Normal traces have degree $k=1,2,3$. The interior order is independently selected
as $s=k+n$, with $n=0,1,2$, and pressure belongs to $Q_s$ on each fine rectangle.

## Problem and reference convention

The unit-square problem has unit permeability and exact Dirichlet data from

$$
p=\cos(2\pi x)\cos(2\pi y),\qquad f=8\pi^2p,\qquad q=-\nabla p.
$$

The reference code is **NeoPZ, labmec/MHM**, function
[`TLaplaceExampleSmooth::uxy`, revision `f978f29`](https://github.com/labmec/MHM/blob/f978f29d657d28fe58bcea20fabee68953093482/meshgen.cpp#L479).
Its frequency is $2\pi$. This identifies the data used here: the displayed
formula in the article uses $\pi$, whereas its reference implementation and the
numerical values in Figure 3 correspond to $2\pi$. The physical fields follow
the figure's axis labels: $u$ denotes pressure and $\sigma$ denotes flux.

The reference-source attribution above records code inspection. The numerical
comparison below is to the published figure; it is not an execution of the NeoPZ
MHM controller. The element spaces are additionally checked against independent
Basix RT elements, after an explicit change of basis.

## Formulation and conservation

The unenhanced rectangular space is
$Q_{k+1,k}\times Q_{k,k+1}$. Enrichment retains its degree-$k$ normal traces
and adds zero-normal-trace interior functions from
$Q_{s+1,s}\times Q_{s,s+1}$. The divergence spans $Q_s$.
Integral Legendre normal moments and contravariant Piola transformations
preserve physical normal flux on anisotropic rectangles.

The local mixed problem uses $\int K^{-1}q\cdot v-\int p\,\mathrm{div}v$,
prescribes the represented skeletal normal flux, and retains the physical
pressure mean. Tests check every pressure equilibrium moment, every restricted
boundary flux moment, anisotropic patches, and a zero-normal polynomial bubble
that requires interior enrichment. Material pixels cut through fine rectangles
are integrated over their exact intersections.

```python
from pymhm import CartesianMacroMesh, solve_darcy_tensor_rt

solution = solve_darcy_tensor_rt(
    CartesianMacroMesh(4), degree=2, enrichment=1,
    local_refinement=2, source=1.0, quadrature_order=6,
)
```

## Six-level comparison

![Pressure and flux errors against the published markers](../figures/tensor-rt/convergence.png)

The experiment uses $N=2,4,8,16,32,64$ squares per direction. The five
published levels are $N=4$ through $64$. Solid curves are PyMHM measurements;
crosses and error bars are independently digitized article markers. Errors
are physical $L^2$ norms from independent quadrature, not nodal sample norms.

The raster comparison uses a conservative three-pixel vertical uncertainty
with logarithmic-axis calibration. The dark-blue MHM-H(div) circles are
identified by their enclosed white interiors, which distinguish them from
the overlapping lighter H1 symbols and connecting lines. Dark-red squares
are isolated by color; the declared uncertainty also covers partial occlusion.
Only distinguishable markers are retained. The numerical record preserves
each center, extraction convention, uncertainty and measured difference.

All **50 distinguishable published markers lie within the stated digitization
uncertainty**; the largest difference is 0.559 uncertainty units. At
$k=3$, $n=2$, $N=64$, the pressure error is
$2.67505\times10^{-11}$ in PyMHM and approximately
$2.70908\times10^{-11}$ at the center of the published circle. Their difference
is 0.120 uncertainty units. These comparisons support reproduction at the
resolution of the figure, not agreement to floating-point precision with
unavailable original field coefficients.

[The marker verification record](../figures/tensor-rt/marker-verification.json)
contains the raster digest, extraction digest and coordinate checks.
[The enlarged symbol panel](../figures/tensor-rt/marker-centers.png) shows every
selected center, with rows ordered by degree and columns by refinement.
Marker selection uses only the raster and its legend; it does not use PyMHM
values or alter the physical solutions.

![Original Figure 3 from Duran and collaborators](../figures/tensor-rt/duran-2019-figure3.jpeg)

The expected pressure orders are approximately $k+1$ without enrichment and
$k+2$ after interior enrichment; the skeletal approximation limits further
improvement. The physical flux order remains approximately $k+1$. Refining only
the interior therefore improves pressure substantially but cannot remove the
normal-trace approximation error.

Run `pixi run -e notebooks verify-tensor-rt` for the research campaign.
Use `--plot-only` to render stored results. The numerical, published and comparison
records are `examples/results/tensor-rt.json`, `tensor-rt-published.json` and
`tensor-rt-comparison.json`. The full campaign is separate from the lightweight
element, conservation and polynomial-patch tests used in CI.
