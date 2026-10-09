# Steady transport

Reaction–advection–diffusion examples state whether the advective operator is
conservative and which Robin quantity couples the local fields. A stabilized
layer profile is assessed through physical field errors and extrema as well as
the original discrete equations.

| Dimension | Case | Numerical evidence |
| --- | --- | --- |
| 2D | Polygonal advection–diffusion | [Initial macro refinement with fixed local/trace choices](../cases/minimal-convergence.md#scalar-polygons) |
| 2D | SUPG transport layer | [Analytical scalar and physical diffusion-flux errors](../cases/minimal-convergence.md#scalar-transport-layer) |
| 2D | Reaction–diffusion layer | [Initial refinement](../cases/minimal-convergence.md#scalar-rad-layer) |
| 2D | MHM-USFEM negative-residual stabilization | [Smooth-case refinement](../cases/minimal-convergence.md#darcy-unusual), [resolved and unresolved boundary-layer fields](../cases/introduction-layers.md#scalar-reactiondiffusion) |

The detailed scalar layer page compares MHM-Galerkin and MHM-USFEM for the
same operator and boundary data. USFEM reduces the reported nodal overshoot
while its scalar L2 error can be larger. An unresolved physical layer is not a
test of the smooth asymptotic rate.

Use the [MHM-USFEM tutorial](../tutorials/introduction/mhm_usfem_rad.md) to connect
the signed stabilization term directly to its UFL form. The
[transient transport gallery](transient-transport.md) treats time evolution
separately.

## References

- Christopher Harder, Diego Paredes and Frédéric Valentin (2015).
  *On a Multiscale Hybrid-Mixed Method for Advective-Reactive Dominated Problems
  with Heterogeneous Coefficients*. [DOI: 10.1137/130938499](https://doi.org/10.1137/130938499).
- Juan Felipe Pacazuca Santiago, Frédéric Valentin and Larissa Martins (2025).
  *A Multiscale Hybrid-Mixed Method with Local Stabilization*.
  [DOI: 10.55592/cilamce2025.v5i.14270](https://doi.org/10.55592/cilamce2025.v5i.14270).
