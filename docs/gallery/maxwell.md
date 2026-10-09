# Maxwell

Electromagnetic cases report electric and magnetic fields in their actual
H(curl) or DG spaces. Tangential trace conventions are distinct from Darcy
normal-flux and solid-mechanics traction conventions.

| Dimension | Case | Numerical evidence |
| --- | --- | --- |
| 2D | Complete nanoguide | [Independent DG electric/magnetic increments](../cases/minimal-convergence.md#wave-nanoguide) |

This accepted device refinement uses independent DG Q2 fields on grids
16, 32 and 64. The electric and magnetic field times differ according to its
staggered scheme. Its increments decrease, but this DG reference control is
not relabelled as an MHM convergence study. The method tutorial identifies
the supported MHM tangential spaces separately.

## References

- Stéphane Lanteri, Diego Paredes, Claire Scheid and Frédéric Valentin (2018).
  *The Multiscale Hybrid-Mixed method for the Maxwell Equations in
  Heterogeneous Media*. [DOI: 10.1137/16M110037X](https://doi.org/10.1137/16M110037X).
