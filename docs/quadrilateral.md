# Quadrilateral Darcy MHM

`CartesianMacroMesh` represents axis-aligned rectangles with shared, oriented
faces. `solve_darcy_quadrilateral` uses continuous tensor-product Qk pressure
spaces inside each macrocell and the same normal-flux skeleton as triangular
Darcy. Both the source response and the face responses are assembled and
condensed inside the selected local worker.

The primal Neumann-local construction follows
[Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019).
The Cartesian geometry, local Qk spaces and independent face partitions used
here are specified below.

```python
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm._legacy.models.darcy.cartesian import solve_darcy_quadrilateral

mesh = CartesianMacroMesh(6, 11, bounds=(0, 1200, 0, 2200))
skeleton = SkeletonSpace(
    mesh,
    tuple(FaceSpace.uniform(1, 32, continuous=True) for _ in mesh.faces),
)
no_flow = {
    int(face): 0.0
    for face in mesh.boundary_faces
    if abs(mesh.normals[face, 0]) > 0.5
}

# Supply a permeability callback with an explicit physical-unit convention.
solution = solve_darcy_quadrilateral(
    mesh,
    permeability=1.0,
    dirichlet=lambda x: 1.0 - x[:, 1] / 2200.0,
    neumann=no_flow,
    skeleton=skeleton,
    degree=1,
    local_refinement=(80, 80),
)
```

This configuration has 127 free macrofaces, 33 coefficients per continuous-P1
face, and 66 retained cell means: 4257 global unknowns after the prescribed
flux coefficients are eliminated. A constant coefficient produces the affine
pressure in this example; the SPE10 case uses a separate, piecewise constant
permeability callback.

## Local equations and independent resolutions

The local bilinear form is

$$
a_T(p,v)=\int_T K\nabla p\cdot\nabla v,
\qquad
B_T(v,\lambda)=\int_{\partial T}\lambda_T v.
$$

The constant pressure spans the local kernel. The constrained local response
has zero integral, and its retained amplitude is assembled globally. The
multiplier is the outward physical normal flux \(q\cdot n_T\), with
\(q=-K\nabla p\). Dirichlet pressure enters through its skeleton moments;
it is imposed weakly. Pure Neumann data require compatibility and use a global
mean-pressure constraint.

`degree` selects Qk. `local_refinement=(rx, ry)` selects the number of fine
rectangles in each macrocell. `FaceSpace` independently selects the polynomial
degree and partition of each face, with optional continuity within that
macroface. Face coupling integrates over the union of the fine-edge and
skeleton breakpoints, so these two partitions need not coincide. They must
still form a stable discrete pair: an over-enriched skeleton can contain
invisible modes and is rejected by the global rank diagnostic.

Volume quadrature is a tensor-product Gauss rule with at least `degree + 1`
points per direction. It is exact for the relevant constant-coefficient
polynomial forms. A `CartesianCellField` permeability automatically splits
each integration rectangle at material-pixel boundaries. This integrates
the discontinuous coefficient without adding pressure degrees of freedom
or making the approximation space conform to the interface. An aligned
grid uses a faster uniform rule. Other callback discontinuities and an
independently partitioned source still require explicit resolution.

For SPE10 pixels of 20 by 10 units and macrocells of 200 by 200 units,
equal local refinements that are multiples of 20 align every material
interface. Exact pixel intersections also handle the intervening refinement
levels. The degrees of freedom, assembly quadrature, integration strategy
and material alignment are recorded separately in the executable case.

## Field evaluation and verification

`solution.evaluate(cell, points)` returns pressure and physical flux on one
specified macrocell. It preserves its local polynomial and does not average
values across a macroface. On fine-cell interfaces it chooses the fine cell
on the positive coordinate side, except at an upper macro boundary, where
it chooses the interior cell. A discontinuous material is sampled from the
same side as that polynomial gradient.

`l2_error`, `flux_l2_error` and `conservation_residuals` provide independent
quadrature diagnostics. Conservation refers to the integrated skeleton flux
per macrocell. Raw Qk pressure gradients generally do not form an H(div)
field and do not imply fine-cell conservation or a discrete maximum principle.

The local matrix is sparse; its factorization is reused for all source and
trace responses. `backend="process"` uses portable spawned workers and
requires picklable coefficient functions plus a guarded executable entry
point. The assembled global skeleton solve remains in the parent process.

Verification includes independent Q1 tensor-product stiffness and mass
matrices, exact non-affine Q2 solutions, anisotropic Q1–Q3 patches,
contrast-1000 interfaces, nonmatching face partitions, Neumann gauges,
serial/thread/process equality, and native Basix Q1–Q6 basis and derivative
comparisons. These checks establish the tested Cartesian implementation;
they do not extend it to warped or curved quadrilaterals.

The 66-cell geometry and continuous-face construction correspond to
[Paredes, Valentin and Versieux (2024)](https://doi.org/10.1016/j.cam.2023.115415), §5.2.
Its MHM local refinement count is not reported. The local-refinement campaign
therefore states each numerical choice and measures its sensitivity. The
article's reference grid does not specify the MHM local grid.

The archived, pixel-aligned local-refinement family uses 40, 60, 80, 100 and
120 subdivisions in each direction with 32 continuous-P1 face segments:

```bash
pixi run python -m examples.solve_spe10
```

The separate conforming Q3 assembly uses the same tensor-product basis and
operators; it is not a comparison against an independent code. Its largest
reference has the article's 768-by-1408 grid and 9,738,625 coefficients:

```bash
pixi run -e intel python -m examples.solve_spe10_reference --shape 768 1408 --order 5
```

The MHM command reuses archived cases after checking their checksums. The Q3
command assembles and solves the requested reference, requiring substantially
more memory than the smaller 240-by-440 default reference.

## References

- Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *A family of Multiscale Hybrid-Mixed finite element methods for the Darcy equation with rough coefficients*, Journal of Computational Physics 245, 107–130. [DOI: 10.1016/j.jcp.2013.03.019](https://doi.org/10.1016/j.jcp.2013.03.019).

- Diego Paredes, Frédéric Valentin, and Henrique M. Versieux (2024). *Revisiting the robustness of the multiscale hybrid-mixed method: The face-based strategy*, Journal of Computational and Applied Mathematics 436, 115415. [DOI: 10.1016/j.cam.2023.115415](https://doi.org/10.1016/j.cam.2023.115415).
