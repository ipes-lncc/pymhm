# Elastodynamics

Time-dependent solid mechanics measures displacement, **velocity** and physical
stress separately. Spatial refinement fixes the time discretization; temporal
refinement fixes the spatial discretization.

| Dimension | Case | Numerical evidence |
| --- | --- | --- |
| 2D | Analytical elastic wave | [Separate displacement, velocity and stress errors](../cases/minimal-convergence.md#wave-elastic-wave) |
| 2D | Three-layer medium | [Conforming temporal increments on a fixed spatial mesh](../cases/minimal-convergence.md#wave-three-layer) |

The analytical series uses three spatial levels, vector P3 local fields, P1
traction traces, time step 0.005 and final time 0.025. Its explicitly shortened
time interval is stated in the record. The three-layer series uses a conforming
temporal reference, so it does not qualify an MHM spatial rate on its own.

For static stresses and displacement, use the [elasticity gallery](elasticity.md).
**Related** wave examples appear in [acoustics](acoustics.md) and
[Maxwell](maxwell.md), with their distinct physical fields.

## References

- Antonio Tadeu Gomes, Diego Paredes, Weslley Pereira, Roberto Souto and
  Frederic Valentin (2017). *A Multiscale Hybrid-Mixed Method for the
  Elastodynamic Model with Rough Coefficients*.
  [DOI: 10.20906/CPS/CILAMCE2017-0399](https://doi.org/10.20906/CPS/CILAMCE2017-0399).
