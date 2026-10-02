# Multiscale HHO in three dimensions

`solve_mshho_3d` constructs cell and face moments on tetrahedral or convex
polyhedral macro meshes. The local field minimizes physical diffusion energy
subject to these moments. Local volume moments are condensed before the global
face solve. It uses the same projected-source and reconstructed-source
conventions as the [two-dimensional implementation](mshho.md), following
[Chaumont-Frelet, Ern, Lemaire and Valentin (2022)](https://doi.org/10.1051/m2an/2021082).
The article proves a dimension-independent equivalence rather than publishing
a numerical benchmark table; the cases here verify the formulation with exact
fields and independent MHM constructions.

## Geometry, moments and boundary conditions

Each tetrahedral face carries independent \(P_\ell\) moments, optionally on
an aligned triangular partition. Polyhedral faces retain moments on the original
polygon, rather than independent traces on the triangles used to integrate it.
Volume moments use \(P_m(K)\). The energy lift uses a conforming tetrahedral
\(P_k\) local mesh, which must have enough independent degrees of freedom to
represent all moment constraints.

Prescribed pressure is imposed through exterior face moments. `neumann`
prescribes physical outward Darcy flux on selected faces. For pure Neumann data,
compatibility is checked and `mean_pressure` selects the physical volume mean.
The global mean is not a coordinate-dependent choice of one nodal value.

```python
from pymhm import TetraMesh, solve_mshho_3d

solution = solve_mshho_3d(
    TetraMesh.unit_cube(2), source=1.0, degree=2,
    cell_degree=0, local_refinement=2,
)
```

`source_variant="projected"` uses the cellwise \(L^2\) source projection.
`source_variant="reconstructed"` applies the original source to reconstructed
test functions. Their fields coincide for represented polynomial sources.
The face-only choice `cell_degree=-1` requires the reconstructed-source variant.
Tests cover both conventions, mixed and pure Neumann boundaries, tensor
diffusion and source compatibility.

## Five-level unit-cube study

The material is \(K=I\), pressure is zero on the boundary, and

$$
p=\sin(\pi x)\sin(\pi y)\sin(\pi z),\qquad
q=-\nabla p,\qquad f=3\pi^2p.
$$

Both families use constant cell/face moments, local \(P_2\), and the projected
source. Tetrahedral macrocells use local refinement two; cubic macrocells use
their conforming tetrahedral decomposition at refinement one. These are
different local meshes, so their errors are not an equal-cost comparison.
The macro resolutions are \(n=1,2,3,4,5\). Assembly order nine and independent
error rules ten and eleven give a maximum norm difference \(4.4\times10^{-10}\)
in the coarsest case and approximately machine precision on finer meshes.

![Five-level three-dimensional MsHHO convergence](../figures/core-extensions/mshho3d-convergence.png)

| Macro geometry | Pressure L2 error at n=5 | Final rate | Raw flux L2 error at n=5 | Final rate |
|---|---:|---:|---:|---:|
| Tetrahedra | 1.43723e-2 | 1.942 | 3.98453e-1 | 0.972 |
| Cubes | 2.28545e-2 | 1.954 | 4.90777e-1 | 0.984 |

The measured rates approach second order for pressure and first order for
flux, as expected for these constant macro moments. The flux is the physical
raw field \(-K\nabla p_h\); this reconstruction does not make it globally
\(H(\mathrm{div})\) conforming. The largest condensed backward residual is
below \(1.0\times10^{-16}\), separately from the approximation errors above.

## Sections and components

The section \(z=0.37\) displays the complete local pressure polynomial and
every physical flux component. Exact and numerical samples share scales,
while differences have independent symmetric scales. Original macroface
intersections are overlaid. Each triangular fan of a fine-cell section has
six display subdivisions; vertices remain private to their fine cell so that
interpolation cannot smooth independent traces across interfaces. Both finest
states are replayed and their volume error norms checked against the campaign
before all local coefficients and section samples are archived.

![Tetrahedral MsHHO pressure and flux components](../figures/core-extensions/tetra-p0-fields.png)
![Cubic MsHHO pressure and flux components](../figures/core-extensions/cube-p0-fields.png)

```bash
pixi run -e notebooks python -m examples.solve_core_extensions mshho3d
pixi run -e notebooks python -m examples.sample_core_sections mshho3d
pixi run -e notebooks python -m examples.plot_core_extensions mshho3d
```

The ten-case record is `examples/results/core-extensions/mshho3d.json`, with
source hashes, error quadrature checks and section archives.
The coefficient replay and display hashes are recorded separately in
`examples/results/core-extensions/mshho3d-field-sampling.json`.
