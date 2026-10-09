# Adaptive scalar and elasticity indicators

Darcy refinement policies, local error controls, metric remeshing and the primal elasticity indicator.

Darcy and transport controllers accept `solve_step` to assemble each state
from user-defined equations. The callable receives the current mesh and
skeleton with the problem's keyword options; it returns the physical solution
required by the chosen estimator. The default formulation remains available.
See the [variational guide](../variational.md#formulations-composed-from-the-same-api)
for composition with mathematical definition factories.

[All API families](../api.md)

::: pymhm.adaptivity.darcy
    options:
      show_source: false

::: pymhm.adaptivity.darcy_balanced
    options:
      show_source: false

::: pymhm.adaptivity.darcy_budget
    options:
      show_source: false

::: pymhm.adaptivity.darcy_3d
    options:
      show_source: false

::: pymhm.estimators.darcy_jump
    options:
      show_source: false

::: pymhm.estimators.darcy_local
    options:
      show_source: false

::: pymhm.adaptivity.metric
    options:
      show_source: false

::: pymhm.estimators.elasticity
    options:
      show_source: false

## References

- Rodolfo Araya, Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *Multiscale Hybrid-Mixed Method*, SIAM Journal on Numerical Analysis 51(6), 3505–3531. [DOI: 10.1137/120888223](https://doi.org/10.1137/120888223).

- Gabriel R. Barrenechea, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2026). *An H(div; Ω)-Conforming Flux Reconstruction for the Multiscale Hybrid-Mixed Method*, Multiscale Modeling & Simulation 24(2), 399–428. [DOI: 10.1137/24M1673073](https://doi.org/10.1137/24M1673073).

- Christopher Harder, Alexandre L. Madureira, and Frédéric Valentin (2016). *A hybrid-mixed method for elasticity*, ESAIM: M2AN 50, 311–336. [DOI: 10.1051/m2an/2015046](https://doi.org/10.1051/m2an/2015046).

## Source modules

::: pymhm.adaptivity.flow_local_mesh
    options:
      show_source: false
