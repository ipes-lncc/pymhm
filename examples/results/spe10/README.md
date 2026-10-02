# SPE10 Model 2 slices and numerical records

The `layer-1.npz`, `layer-36.npz` and `layer-85.npz` property arrays are subsets
of the SPE10 reservoir data attributed to Mike Christie and Martin Blunt, obtained from
[OPM/opm-data](https://github.com/OPM/opm-data/tree/eaa2261683a97027e057c2bc49612ad1c86390b3/spe10model2).
`dataset.json` records the pinned revision, SHA-256 source checksums, array
ordering, layer numbers, material ranges and units. The source include-file
headers state that the property data are public domain. The OPM repository
distributes its dataset/decks under ODbL 1.0, with individual contents under
DbCL 1.0 unless otherwise specified; both notices accompany these subsets.
They are separate from the package's LGPL license.

Layers are one based. Arrays use `(x,y,component)` for permeability and `(x,y)`
for porosity, with unchanged Kx, Ky and Kz in mD and cell sizes 20×10 ft.
The OPM porosity values are preserved, including its documented replacement
of most zero entries by 1e-7. No inactive cells are deleted.

`pixi run -e notebooks python examples/plot_spe10_data.py --download`
explicitly downloads and validates the full model into the build cache and
regenerates these subsets and PyVista figures. The full property includes and
full-volume VTI are not included in the distribution.

The `darcy-*.npz` files contain PyMHM solutions, not imported SPE10 solution
data. Their JSON records distinguish local Q1 refinement, continuous-P1 face
partition, material alignment, quadrature, residuals and archive checksums.
The aligned 32-segment family uses local refinements 40, 60, 80, 100 and 120;
the pixel-cut grids form a separate refinement study. The article does not
specify its MHM local refinement.

The `reference-q3-*.npz` files are separately assembled conforming Q3 solutions
using PyMHM's basis and integration kernels, not an independent code. The
768-by-1408 grid has the article's 9,738,625 coefficients. Exact integration
over material-pixel intersections does not enrich the polynomial space at
permeability jumps. `published-profile.json` instead contains coordinates
digitized from the published curve, with its citation and raster uncertainty.

Computed Darcy fields retain numerical permeability in mD, coordinates in ft
and the prescribed pressure values 1 and 0. Flux follows `q = -K grad(p)` in
that declared numerical convention; it is not reported as an SI velocity.
Profile arrays retain separate values at both sides of macro interfaces.

To redraw the distributed data and numerical results without downloading the
full model or rerunning the solves:

```bash
pixi run -e notebooks python examples/plot_spe10_data.py --layers-only
pixi run -e notebooks gallery-spe10-darcy
```
