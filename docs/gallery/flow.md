# Stokes–Brinkman and Oseen

These cases compare **velocity**, physical pressure, velocity gradient and
incompressibility. The symmetric-stress and vector-Laplacian pseudostress
formulations have distinct boundary variables and local modes; each page
identifies which one is executed.

| Dimension | Case | Numerical evidence |
| --- | --- | --- |
| 2D | Brinkman analytical boundary layer | [Velocity/pressure fields, one-sided profiles, polynomial-family convergence and independently refined Taylor–Hood reference](../cases/introduction-layers.md#incompressible-brinkman-flow) |
| 2D | Stabilized Stokes | [Separate field errors and divergence](../cases/minimal-convergence.md#stokes2d) |
| 2D | Oseen with P1 skeleton | [Smooth analytical refinement](../cases/minimal-convergence.md#oseen2d-trace) |
| 2D | Oseen at two viscosities | [Viscosities 1 and 0.01](../cases/minimal-convergence.md#oseen2d-viscosity) |
| 2D | Boundary layer, internal layer and variable advection | [Field errors and the internal-layer divergence limitation](../cases/minimal-convergence.md#oseen2d-data) |
| 3D | Analytical Stokes, Brinkman and Oseen | [Four fixed-formulation refinement series](../cases/minimal-convergence.md#flow3d) |

The layer example has an analytical solution. The detailed page states the
layer width, local/trace compatibility, pressure convention and observed rates;
it also preserves the independently assembled native hybrid-system checks.
The Oseen internal-layer divergence increases on its initial meshes, so this
series does not establish incompressibility convergence.

Follow the [Brinkman introductory tutorial](../tutorials/introduction/stokes_brinkman_boundary_layer.md)
to write the local and global forms before assembling the same problem.

## References

- Rodolfo Araya, Christopher Harder, Abner H. Poza and Frédéric Valentin (2017).
  *Multiscale hybrid-mixed method for the Stokes and Brinkman equations—The
  method*. [DOI: 10.1016/j.cma.2017.05.027](https://doi.org/10.1016/j.cma.2017.05.027).
- Rodolfo Araya, Christopher Harder, Abner H. Poza and Frédéric Valentin (2025).
  *Multiscale Hybrid-Mixed Methods for the Stokes and Brinkman Equations—A
  Priori Analysis*. [DOI: 10.1137/24M1649368](https://doi.org/10.1137/24M1649368).
