# Hexagonal MHM for an analytical boundary layer

This case uses the operator, data and P3/P1 approximation degrees of
[Araya et al. (2024), section 5.2.2, Figures 5–6](https://doi.org/10.1016/j.cma.2024.117089).
On the unit square,

$$
-\epsilon\Delta u+\partial_x u=1,\qquad \epsilon=10^{-2},
$$

with zero Dirichlet data at $x=0,1$ and zero diffusive normal flux at
$y=0,1$. The exact solution is

$$
u(x,y)=x-\frac{e^{(x-1)/\epsilon}-e^{-1/\epsilon}}
                       {1-e^{-1/\epsilon}}.
$$

The boundary layer at $x=1$ has characteristic width $\epsilon$. Standard
continuous P2 Galerkin can oscillate when its mesh does not resolve that
width. The MHM local P3 solves resolve the layer inside hexagonal macrocells;
independent P1 polynomials couple the macrofaces. There is no SUPG term in
either approximation. The selective local construction has no constant
nullspace on these cells because the advective normal trace is nonzero.

The macro meshes are explicitly generated clipped staggered Voronoi partitions.
Boundary-separated centroid fans define conforming local triangulations. These
are recorded original meshes: the article does not provide its historical
connectivity. The equation, approximation spaces and profile location match the
publication; equality with every historical numerical ordinate is not asserted.

## Fields and the published profile

![Classical, MHM and exact scalar fields](../figures/rad-layer/fields.png)

All spatial panels show the actual MHM macro boundaries and share color limits.
The classical P2 mesh has a different connectivity. Local cubic fields are
evaluated independently on each fine triangle, preserving broken macro traces.

![Exact field and profile at the published height](../figures/rad-layer/profile.png)

The cut is $y=0.4375$, as in Figure 5. Vertical markers locate its intersections
with macrofaces. The classical curve uses $n=8$; the MHM curve uses the same
macro partition parameter and eight local subdivisions. These have different
numbers of global and local unknowns, recorded separately.

![Matched elevation views](../figures/rad-layer/elevation.png)

These views use common height and color scales. Macro edges are projected onto
the base plane so that the actual partition remains visible without concealing
the boundary layer.

![Published boundary-layer comparison](../figures/rad-layer/published-figure5.png)

*[Araya et al. (2024)](https://doi.org/10.1016/j.cma.2024.117089), Figure 5: classical P2, MHM P3/P1 and the exact profile.*

## Independent refinements and integrated errors

![Macro refinement and classical P2 errors](../figures/rad-layer/convergence.png)

Five macro resolutions use fixed P3/P1 degrees and two local subdivisions.
The finest MHM mesh has 1184 macrocells and 6830 free global unknowns. Its scalar
$L^2$ error is $2.6622\times10^{-4}$; its diffusion-weighted gradient error is
$1.1120\times10^{-2}$. The corresponding classical P2 values are
$7.6547\times10^{-3}$ and $1.7392\times10^{-1}$.

The recorded norms distinguish the diffusion-weighted seminorm from the
article's unweighted $V$ norm:

$$
\|e\|_E^2=\epsilon\sum_K\|\nabla e\|_{0,K}^2,\qquad
\|e\|_V^2=\sum_K\|\nabla e\|_{0,K}^2+\tfrac12\|e\|_0^2.
$$

The factor $1/2$ is $1/\operatorname{diam}(\Omega)^2$. Error norms are volume
integrals, independent of the visualization samples. Higher quadrature orders
check the integration error. Refining local meshes at fixed macrofaces can
reach a skeletal-error floor; it is not equivalent to enriching the skeleton.

![Published norm and independent local refinement](../figures/rad-layer/published-norm.png)

At a fixed twenty-cell partition, increasing the number of P1 segments per
macroface through 1, 2, 4, 8 and 16 reduces the $V$ error from 3.0715 to 0.011940.
Local subdivisions increase with the segments to preserve the trace/local
compatibility. The last state has 1376 free global unknowns (1952 total trace coefficients); changing the error
quadrature from order 20 to 28 changes its norms by at most $1.4\times10^{-12}$
relatively. In contrast, increasing only the local resolution at fixed 410
free global unknowns approaches a nonzero error floor, as expected from a fixed
skeletal approximation.

![Published convergence comparison](../figures/rad-layer/published-figure6.png)

*[Araya et al. (2024)](https://doi.org/10.1016/j.cma.2024.117089), Figure 6. The publication reports the $V$ norm and separates
macro refinement from skeletal refinement. Diffusion-weighted errors must not
be compared directly to these ordinates.*

## Independent complete-system comparison

The reference is an independent implementation of the **same MHM
discretization**, solved as a complete saddle system. PyMHM solves its
statically condensed counterpart.

[DOLFINx/UFL 0.9.0](https://github.com/FEniCS/dolfinx/tree/6443e3b27d29aec04ce83d8424b57c339e35f865)
assembles standard Lagrange P3 elements on the fine **triangles** inside
each hexagonal macroelement. The fine meshes have separate vertex and
degree-of-freedom indices across macroelement boundaries: the scalar field
is continuous within each macroelement and may jump between macroelements.
The hexagons drawn in the figures are groups of triangular elements,
not polygonal finite elements supplied by FEniCS.

A separately written comparison application assembles the P1 macroface
coupling from basis evaluations provided by Basix and geometrically oriented
Legendre moments. Basix supplies basis tabulations; the application assembles
the global MHM coupling. Strong homogeneous
Dirichlet data at
$x=0,1$ and zero diffusive flux at $y=0,1$ match the MHM problem above.
The comparison application solves the complete scalar/multiplier saddle
system with SciPy SuperLU. DOLFINx/UFL supplies its independently assembled
volume forms; the application supplies the interface blocks. Neither block
uses PyMHM's local operator matrices or its condensation.

Three configurations are compared: $(n,r)=(2,2),(4,2),(8,8)$, with one P1
segment per macroface. The last configuration is the field/profile example
above: 80 macrocells, 28,544 fine triangles and 133,790 native saddle unknowns.
Its relative differences are $1.944\times10^{-13}$ for the scalar field,
$2.594\times10^{-13}$ for its gradient and $1.955\times10^{-13}$ for the
physical flux $q=-\epsilon\nabla u+(u,0)$. Independent UFL quadrature degrees
16 and 20 verify these physical norms. The normalized native original-equation
residual is $5.43\times10^{-13}$.

![Independent scalar and physical-flux comparison](../figures/rad-layer/native-comparison.png)

The scalar and streamwise-flux panels share scales between codes. Difference
panels use their own symmetric scales. The exact transverse flux vanishes,
whereas both discrete solutions have nonzero transverse components near the
outflow layer. These components agree between codes: the code differences,
not the transverse fields themselves, are at floating-point roundoff scale.
The visualization preserves
separate fine-cell polynomial pieces and draws the actual macro edges.

The [native comparison record](../figures/rad-layer/native-discrete-verification.json)
identifies source and build revisions, all three systems, physical norms and
archive digests. Saved nodal coordinates, fine connectivity and basis ordering
support replay. This comparison verifies the declared discrete realization;
it does not identify the article's unavailable mesh connectivity.

Run `pixi run -e notebooks verify-rad-layer`. Numerical records are in
`examples/results/rad-layer.json`. The research campaign is separate from the
small kernel, boundary and polynomial checks run in CI.

## References

- Rodolfo Araya, Fabrice Jaillet, Diego Paredes, and Frédéric Valentin (2024). *Generalizing the multiscale hybrid-mixed method for reactive-advective-diffusive equations*, Computer Methods in Applied Mechanics and Engineering 428, 117089. [DOI: 10.1016/j.cma.2024.117089](https://doi.org/10.1016/j.cma.2024.117089).
