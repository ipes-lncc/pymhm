# Three-dimensional gallery

Volume cases retain their actual tetrahedral, prismatic or hexahedral local
spaces and face degrees. A 2D degree or trace condition is not transferred to
3D without its stated dimensional hypotheses.

| Problem | Cases |
| --- | --- |
| [Darcy](darcy.md#three-dimensional-cases) | MsHHO on tetrahedra/cubes, enriched H(div) on tetrahedra/prisms, mixed well and complete CPU/GPU performance campaigns |
| Stokes–Brinkman and Oseen | [Four analytical fixed-space refinement series](../cases/minimal-convergence.md#flow3d) |
| Nearly incompressible elasticity | [GaLS and Taylor–Hood local families](../cases/minimal-convergence.md#gals3d) |
| Anisotropic primal elasticity | [Displacement and raw stress refinement](../cases/minimal-convergence.md#primal-elasticity3d) |
| Weak-symmetry mixed elasticity | [AFW displacement, stress, rotation and stress-divergence refinement](../cases/minimal-convergence.md#mixed-elasticity3d) |

Three-dimensional transport and wave APIs have their own declared geometry and
space contracts. This gallery does not assign them a completed 3D convergence
campaign without corresponding current accepted records. The
[mesh guide](../meshing.md#three-dimensional-generation-and-exchange) documents
volume generation, exchange and material/interface tags.
