# Stokes, Brinkman and Oseen

Incompressible-flow formulations, residual estimators and adaptive policies.

[All API families](../api.md)

User-defined formulations compose these public numerical owners with
`LocalEquations` and `Equation`; their editable providers are listed in the
[variational guide](../variational.md#formulations-composed-from-the-same-api).
Physical field records provide optional interpretation of the resulting
coefficients. Named `FieldDefinition` views require no physical solver object.

The volume operator is the vector Laplacian. The normal-oriented hybrid
multiplier is $\lambda_K=(-\nu\nabla u+pI+u\otimes\beta/2)n_K$;
the `traction` convenience arguments prescribe its negative. With zero
convection the half-advection term vanishes. Velocity/pressure coefficient
arrays require their executed bases to recover these physical fields.
The API below documents its kernels, pressure gauge and residual signs.
For the analytical hypotheses, consult
[Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027) and
[Araya et al. (2025)](https://doi.org/10.1137/24M1649368).
Successful matrix assembly does not establish those hypotheses.

::: pymhm.fem.vector.operators
    options:
      show_source: false

::: pymhm.fem.vector.flow
    options:
      show_source: false

::: pymhm.fem.vector.flow_3d
    options:
      show_source: false

::: pymhm.materials.resistance
    options:
      show_source: false

::: pymhm.estimators.flow
    options:
      show_source: false

::: pymhm.adaptivity.flow
    options:
      show_source: false

::: pymhm.adaptivity.flow_macro
    options:
      show_source: false

::: pymhm.postprocessing.solutions.Flow3DSolution
    options:
      show_source: false

## References

- Rodolfo Araya, Ramiro Rebolledo, and Frédéric Valentin (2021). *On a multiscale a posteriori error estimator for the Stokes and Brinkman equations*, IMA Journal of Numerical Analysis 41(1), 344–380. [DOI: 10.1093/imanum/drz053](https://doi.org/10.1093/imanum/drz053). An earlier version is [HAL: hal-01945934v1](https://hal.science/hal-01945934v1).

- Rodolfo Araya, Cristian Cárcamo, Abner H. Poza, and Frédéric Valentin (2021). *An adaptive multiscale hybrid-mixed method for the Oseen equations*, Advances in Computational Mathematics 47, article 15. [DOI: 10.1007/s10444-020-09833-8](https://doi.org/10.1007/s10444-020-09833-8).
