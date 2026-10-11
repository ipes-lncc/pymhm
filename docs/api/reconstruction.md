# Flux reconstruction and error estimation

Moment reconstruction, conforming potentials and dimension-dependent estimator hypotheses.

The moment construction accepts represented skeletal normal traces and
continuous local tests contained in the primal space. Its algebraic
conservation identity is distinct from an approximation theorem.
[Barrenechea et al. (2026), Theorem 4.7 and Remark 5.1](https://doi.org/10.1137/24M1673073)
require $k\ge\ell+d$, $\ell\le m\le k$ for $d=2,3$, compatible
shape-regular partitions, the stated solution/source Sobolev regularity,
and diffusion polynomial on each fine simplex. In particular the minimum
local degrees for constant/linear skeletal traces are P2/P3 in 2D and
P3/P4 in 3D. Material callbacks do not certify the theorem's regularity.
The same publication states the separate reliability and efficiency
hypotheses. Fine-cell balance is an independent implementation diagnostic.

[All API families](../api.md)

::: pymhm.recovery.equilibrated
    options:
      show_source: false

::: pymhm.recovery.moments
    options:
      show_source: false

::: pymhm.recovery.moments_3d
    options:
      show_source: false

::: pymhm.estimators.darcy
    options:
      show_source: false

::: pymhm.estimators.darcy_energy
    options:
      show_source: false

::: pymhm.estimators.darcy_3d
    options:
      show_source: false

## References

- Gabriel R. Barrenechea, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2026). *An H(div; Ω)-Conforming Flux Reconstruction for the Multiscale Hybrid-Mixed Method*, Multiscale Modeling & Simulation 24(2), 399–428. [DOI: 10.1137/24M1673073](https://doi.org/10.1137/24M1673073).
