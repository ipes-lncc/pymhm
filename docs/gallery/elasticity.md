# Elasticity

Solid-mechanics cases distinguish displacement, physical Cauchy stress,
Herrmann pressure, weak rotation and stress divergence. Two-dimensional
constitutive data identify plane strain or plane stress explicitly.

| Dimension | Case | Numerical evidence |
| --- | --- | --- |
| 2D | Anisotropic weak-symmetry elasticity | [Triangle, polygon and rectangular mixed local families](../cases/minimal-convergence.md#initial-elasticity2d) |
| 2D | Oscillatory elasticity | [Original oscillatory geometry/data, fixed macro mesh and trace refinement](../cases/minimal-convergence.md#elasticity-l18-oscillatory) |
| 2D | Heterogeneous elasticity tutorial | [Material, displacement and stress fields against classical refinement](../tutorials/introduction/multiscale_elasticity.md) |
| 3D | Nearly incompressible GaLS elasticity | [Variable shear, lambda=1e8, GaLS and Taylor–Hood families](../cases/minimal-convergence.md#gals3d) |
| 3D | Anisotropic primal elasticity | [Displacement and raw symmetric-stress errors](../cases/minimal-convergence.md#primal-elasticity3d) |
| 3D | AFW weak-symmetry mixed elasticity | [Displacement, Cauchy stress, axial weak rotation and stress divergence](../cases/minimal-convergence.md#mixed-elasticity3d) |

These initial studies preserve separate physical field norms and their executed
spaces. A primal raw stress is distinct from an H(div)-conforming mixed stress.
The time-dependent examples are in [elastodynamics](elastodynamics.md).

## References

- Christopher Harder, Alexandre L. Madureira and Frédéric Valentin (2016).
  *A hybrid-mixed method for elasticity*.
  [DOI: 10.1051/m2an/2015046](https://doi.org/10.1051/m2an/2015046).
- Philippe R. B. Devloo, Agnaldo M. Farias, Sônia M. Gomes, Weslley Pereira,
  Antonio J. B. dos Santos and Frédéric Valentin (2021).
  *New H(div)-conforming multiscale hybrid-mixed methods for the elasticity
  problem on polygonal meshes*. [DOI: 10.1051/m2an/2021013](https://doi.org/10.1051/m2an/2021013).
