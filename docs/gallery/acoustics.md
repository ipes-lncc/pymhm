# Helmholtz and acoustics

Acoustic cases distinguish complex pressure, its physical gradient/flux and
the frequency/resolution assumptions of the Helmholtz operator.

| Dimension | Case | Numerical evidence |
| --- | --- | --- |
| 2D | Analytical plane wave | [Separate pressure and gradient errors on three spatial levels](../cases/minimal-convergence.md#wave-helmholtz) |
| 2D | Marmousi II material | [Explicit 160×80 m primary-data crop and macro refinement](../cases/minimal-convergence.md#wave-marmousi) |

The crop pilot fixes its fine mesh and reports differences to a numerical
comparison level. It is not the complete published Marmousi geometry or an
exact solution. Source data, units and coordinates remain explicit in the
[data catalogue](../data.md).

## References

- Théophile Chaumont-Frelet and Frédéric Valentin (2020).
  *A Multiscale Hybrid-Mixed Method for the Helmholtz Equation in Heterogeneous
  Domains*. [DOI: 10.1137/19M1255616](https://doi.org/10.1137/19M1255616).
