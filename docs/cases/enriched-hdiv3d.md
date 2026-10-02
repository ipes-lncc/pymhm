# Independent normal and interior orders in three dimensions

`HDiv3DFamily` and `solve_darcy_hdiv3d` separate the normal polynomial degree
from the complete interior divergence space. Adding zero-normal bubbles changes
local approximation without adding macroface unknowns. Enriching a normal trace
changes a different part of the discrete problem. The [well comparisons](mixed-well-geometries.md)
include independent NeoPZ verification for their stated lower-order spaces.
The higher orders below are checked against exact fields and native Basix operators.

## Local spaces

For tetrahedra, pressure degree \(p\) and retained normal degree \(k\), with
\(0\leq k\leq p+1\), define

$$
\begin{aligned}
V_{p,k}(T)&=\{v\in[P_{p+1}(T)]^3:
                    v\cdot n_F\in P_k(F)\text{ on every face }F\},\\
\operatorname{div}V_{p,k}(T)&=P_p(T).
\end{aligned}
$$

On a prism, horizontal components belong to
\([P_{p+1}(\triangle)\otimes P_p(I)]^2\), and the vertical component belongs to
\(P_p(\triangle)\otimes P_{p+1}(I)\). Retained normals are \(P_k\) on triangular
faces and \(Q_k\) on rectangular faces, with \(0\leq k\leq p\). Pressure and
divergence use the complete space \(W_{p,p}=P_p(\triangle)\otimes P_p(I)\).

Every zero-normal bubble of the declared parent space is retained. Physical
Piola maps, face orientation and normal moments are applied before local
condensation. A macro trace can further restrict these local normal spaces.
The geometry is affine; this construction does not assert support for curved
prisms or pyramids. Full matrix rank is not evidence of a uniform inf-sup
constant as polynomial orders increase.

## Five-level analytical comparison

The unit cube has \(K=I\), zero prescribed pressure, and

$$
p=\sin(\pi x)\sin(\pi y)\sin(\pi z),\qquad
q=-\nabla p,\qquad f=3\pi^2p.
$$

Each macrocell is one affine fine cell. Resolutions are \(n=1,2,3,4,5\), with
\(6n^3\) tetrahedra or \(2n^3\) prisms. All macroface normal modes of the stated
family are retained. Assembly uses order 12. Independent error rules 12 and 13
agree within \(2.7\times10^{-10}\) in the coarsest case and to floating-point
precision on finer meshes. These are analytical verification cases rather than
historical figure reproductions.

![Pressure and physical flux convergence](../figures/core-extensions/hdiv3d-convergence.png)

| Geometry and orders | Pressure L2 error at n=5 | Final rate | Flux L2 error at n=5 | Final rate |
|---|---:|---:|---:|---:|
| Tetrahedron, p=1, k=1 | 1.11701e-2 | 1.951 | 3.48926e-2 | 2.003 |
| Tetrahedron, p=2, k=2 | 1.26533e-3 | 2.946 | 2.70823e-3 | 3.051 |
| Tetrahedron, p=3, k=1 | 5.78561e-4 | 3.058 | 2.85690e-2 | 1.979 |
| Prism, p=2, k=2 | 8.35752e-4 | 2.969 | 2.88195e-3 | 2.988 |

The tetrahedral rows separate the roles of these spaces. Increasing interior
order to \(p=3\) improves pressure, but keeping normal degree one retains a
second-order flux limitation in this sequence. Increasing both to degree two
yields third-order pressure and flux. An inaccurate coarse normal space cannot
be repaired solely by adding local bubbles.

Fine-cell equilibrium moments are below \(1.2\times10^{-13}\), and original
physical block residuals are below \(8.2\times10^{-14}\). These conservation
and algebraic checks accompany the field errors; they do not replace them.

## Pressure and flux components

Horizontal sections are at \(z=0.37\). The full pressure and physical flux
polynomials are evaluated on six subdivisions of each triangular fan of a
fine-cell section. Exact and numerical fields use the same display vertices.
Each fine cell owns its vertices, so display interpolation never averages
independent traces across an interface. Dark lines show the original macro mesh
intersections. Differences have separate symmetric scales. The volume norms
are independently reintegrated from the archived coefficients and executed
basis before the section fields are exported.

![Tetrahedral order-two pressure and physical flux components](../figures/core-extensions/tetra-p2-k2-fields.png)
![Interior enrichment with fixed normal degree one](../figures/core-extensions/tetra-p3-k1-fields.png)
![Prismatic order-two pressure and physical flux components](../figures/core-extensions/prism-p2-k2-fields.png)

## Verification and reproduction

Portable tests check normal duality, divergence range, orientation, polynomial
patches and complete interior bubbles. Native Basix tests independently compare
tetrahedral polynomial operators and prismatic tensor-product spaces, including
pressure orders two and three. Persisted fields include the executed basis
matrix alongside coefficients, so evaluation does not depend on a new nullspace
orientation.

```bash
pixi run -e notebooks python -m examples.solve_core_extensions hdiv3d
pixi run -e notebooks python -m examples.sample_core_sections hdiv3d
pixi run -e notebooks python -m examples.plot_core_extensions hdiv3d
```

The 20-case record is `examples/results/core-extensions/hdiv3d.json`, accompanied
by field archives, source hashes and quadrature checks.
The display replay and reintegrated volume norms are recorded separately in
`examples/results/core-extensions/hdiv3d-field-sampling.json`.
