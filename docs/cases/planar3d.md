# Oblique material interfaces and nonuniform 3D faces

`PlanarMaterial` describes an SPD background and intersections of affine
halfspaces. `fit_planar_material` creates conforming local tetrahedra at their
interfaces, while `planar_face_partitions` supplies the matching triangulation
of each original macroface. The original macro mesh is retained. Local
refinement and skeletal enrichment are separate choices.

## Exact transmission problem

On the unit cube, define

$$
t=n\cdot x-0.63,\qquad n=(1,0.4,0.2)^T,\qquad
A=\begin{pmatrix}2&0.3&0.2\\0.3&1&0.1\\0.2&0.1&1.5\end{pmatrix}.
$$

The material is $K=kA$, with $k=25$ in $t\leq0$ and $k=1$ outside. The exact
pressure, physical Darcy flux and source are

$$
\begin{aligned}
p(x)&=\frac{t+0.3t^2}{k},\\
q(x)&=-K\nabla p=-(1+0.6t)An,\\
f(x)&=\nabla\cdot q=-0.6n^TAn.
\end{aligned}
$$

Pressure and normal flux are continuous on the interface. The pressure
gradient has the material jump; the source has no surface distribution.
Independent finite differences verify the pressure gradient and flux divergence
on both sides. This is an original exact transmission test, not a historical
paper reproduction.

![Material and original macro mesh](../figures/planar3d/geometry.png)

## Geometric and algebraic contracts

Each face partition contains barycentric vertex triples in the original order
`mesh.faces[face]`. Its triangles have positive, generally unequal area
fractions. Conformity, containment and complete coverage are checked; equality
of total area or volume alone is insufficient. The local meshes must resolve
every active skeletal subtriangle. Their global topology is used by the local
finite-element assembly, including nodes on the material interface.

The skeletal integral uses these physical fractions, never the arithmetic mean
of all subtriangle coefficients. Pressure data enter through area-weighted weak
moments. Normal-flux data are projected in the individual face space. Material
values in RT face moments are one-sided traces selected by the incident fine
cell, without moving quadrature points or averaging the permeability.

```python
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from examples.formulations.application import tetrahedral_darcy as solve_darcy_3d
from pymhm.meshes.fitting import fit_planar_material, planar_face_partitions

parts = planar_face_partitions(macro, material)
skeleton = TriangularSkeleton(macro, degree=1, face_partitions=parts)
locals_ = tuple(
    fit_planar_material(macro.submesh(cell, 1), material).mesh
    for cell in range(len(macro.cells))
)
solution = solve_darcy_3d(
    macro, skeleton=skeleton, local_meshes=locals_, degree=4,
    permeability=material, source=source,
    dirichlet=exact_pressure, neumann=outward_flux_data,
)
```

The example uses continuous local P4 and discontinuous P1 modes on each fitted
face subtriangle. Numerical rank is checked independently of geometry: a
conforming partition does not guarantee a stable choice of local and skeletal
spaces. For this partition, the P4 boundary space resolves the selected trace.

## Physical checks and field sections

The campaign uses macro subdivisions $n=1,2,3$, each with six tetrahedra per cube,
and three boundary configurations. Full Dirichlet prescribes exact pressure
weakly everywhere. The mixed case prescribes exact outward flux on $x=1$ and
pressure on the remaining exterior. Pure Neumann prescribes exact flux on all
exterior faces and the physical volume mean of the exact pressure.

Assembly uses positive order-six quadrature, exact for these fitted polynomial
integrands. Error norms use separate order-six/seven rules. The recorded errors
are absolute physical volume norms; macro balance is outward skeletal flux minus
integrated source. Canonical RT1 reconstruction is checked against the same
physical flux. These are polynomial transmission patches, so their roundoff
errors do not measure convergence rates or establish robustness for arbitrary
contrasts, geometries or nonpolynomial data.

The table reports the largest absolute error across the three boundary
configurations at each resolution. All nine cases preserve their source and
field digests. The largest macro balance defect is $1.45\times10^{-15}$;
the largest change between the two error quadratures is $2.98\times10^{-17}$.

| $n$ | Macro / fine tetrahedra | Pressure $L^2$ error | Raw flux $L^2$ error | RT1 flux $L^2$ error |
|---:|---:|---:|---:|---:|
| 1 | 6 / 80 | $1.38\times10^{-14}$ | $7.91\times10^{-13}$ | $1.84\times10^{-12}$ |
| 2 | 48 / 528 | $1.22\times10^{-14}$ | $7.71\times10^{-13}$ | $1.44\times10^{-12}$ |
| 3 | 162 / 1,432 | $8.56\times10^{-15}$ | $6.04\times10^{-13}$ | $1.09\times10^{-12}$ |

An independent DOLFINx/UFL check assembles the P2 and P4 stiffness, mass and load
on the fitted tetrahedra, with the tensor represented by cellwise constant data.
It compares every matrix and load entry after identifying physical nodal
coordinates. Separate regression tests cover the one-sided material values in
both two- and three-dimensional RT reconstruction.

The field figures show the mixed $n=2$ case on $z=0.37$. Colors are flat values
at centroids of actual cut fine-cell polygons, preserving separate one-sided
fields. Exact and numerical panels share limits; each error has its own scale.
Black/white lines mark the original macroface intersections.

![Pressure and physical flux](../figures/planar3d/fields.png)

![Signed physical flux components](../figures/planar3d/components.png)

## Reproduction

```bash
pixi run --locked -e test-core python -m examples.solve_planar3d
pixi run --locked -e notebooks python -m examples.plot_planar3d
pixi run --locked -e test-core pytest tests/test_planar3d_example.py tests/test_planar_reconstruction.py
pixi run --locked -e fem pytest tests/test_planar3d_fenics.py
```

The [campaign JSON](../figures/planar3d/campaign.json) records geometry counts,
area ranges, boundary conventions, source and field hashes, norms and residuals.
[Notebook 63](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/darcy/63_planar3d.ipynb)
executes a light mixed-boundary patch and replays the accepted records and figures.
