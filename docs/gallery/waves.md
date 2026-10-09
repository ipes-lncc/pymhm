# Waves and elastodynamics

Wave examples report their physical field, frequency or final time, spatial
resolution and temporal scheme. A scalar Helmholtz pressure and an electric
field are different unknowns, even when their reduced algebra uses the same
multiscale assembly owners.

## Helmholtz and acoustics

| Dimension | Case | Numerical evidence |
| --- | --- | --- |
| 2D | Analytical Helmholtz plane wave | [Separate complex pressure and gradient errors](../cases/minimal-convergence.md#wave-helmholtz) |
| 2D | Marmousi II | [Explicit 160×80 m primary-data crop, fixed fine mesh and macro refinement](../cases/minimal-convergence.md#wave-marmousi) |

The Marmousi comparison level is numerical, and the crop is stated explicitly;
it is not the complete published seismic geometry.

## Elastodynamics

| Dimension | Case | Numerical evidence |
| --- | --- | --- |
| 2D | Analytical elastic wave | [Displacement, velocity and stress on three spatial levels](../cases/minimal-convergence.md#wave-elastic-wave) |
| 2D | Three-layer medium | [Conforming temporal increments on a fixed spatial mesh](../cases/minimal-convergence.md#wave-three-layer) |

The three-layer temporal control is a conforming reference study. It does not
by itself qualify a multiscale spatial approximation.

## Maxwell

| Dimension | Case | Numerical evidence |
| --- | --- | --- |
| 2D | Nanoguide | [Independent DG electric/magnetic refinement on the complete device](../cases/minimal-convergence.md#wave-nanoguide) |

The recorded nanoguide series uses the independent DG method. Its decreasing
electric/magnetic increments are not relabelled as MHM convergence evidence.

## References

- Théophile Chaumont-Frelet and Frédéric Valentin (2020).
  *A Multiscale Hybrid-Mixed Method for the Helmholtz Equation in Heterogeneous
  Domains*. [DOI: 10.1137/19M1255616](https://doi.org/10.1137/19M1255616).
- Antonio Tadeu Gomes, Diego Paredes, Weslley Pereira, Roberto Souto and
  Frederic Valentin (2017). *A Multiscale Hybrid-Mixed Method for the
  Elastodynamic Model with Rough Coefficients*.
  [DOI: 10.20906/CPS/CILAMCE2017-0399](https://doi.org/10.20906/CPS/CILAMCE2017-0399).
- Stéphane Lanteri, Diego Paredes, Claire Scheid and Frédéric Valentin (2018).
  *The Multiscale Hybrid-Mixed method for the Maxwell Equations in
  Heterogeneous Media*. [DOI: 10.1137/16M110037X](https://doi.org/10.1137/16M110037X).
