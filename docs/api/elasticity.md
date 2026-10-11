# Elasticity

Displacement, displacement–pressure and weakly symmetric stress formulations.

[All API families](../api.md)

User-defined formulations compose these public numerical owners with
`LocalEquations` and `Equation`; their editable providers are listed in the
[variational guide](../variational.md#formulations-composed-from-the-same-api).
Physical field records provide optional interpretation of the resulting
coefficients. Named `FieldDefinition` views require no physical solver object.

Primal displacement and GaLS reconstruction return a symmetric raw Cauchy
stress from the declared constitutive law; its macro traction moments do not
make that stress globally $H(\operatorname{div})$ conforming. The mixed stress
owners use row-wise $H(\operatorname{div})$ spaces and enforce symmetry in the
rotation test moments. Their physical displacement/rotation kernel is coupled.
Every local static cell retains $d(d+1)/2$ rigid modes.

Herrmann pressure is $p=-\lambda_L\operatorname{div}u$. The mixed stress
hydrostatic gauge uses $-\operatorname{tr}\sigma/d$; these quantities generally
differ at finite bulk modulus. Traction inputs mean physical outward
$\sigma n$, while the MHM skeletal convention is $-\sigma n$.
The API below documents source responses, global coupling and gauges.
Consult the original publications listed in References for each
formulation's stabilization bounds, admissible spaces and dimensional scope.

::: pymhm.fem.vector.elasticity
    options:
      show_source: false

::: pymhm.fem.vector.elasticity_3d
    options:
      show_source: false

::: pymhm.fem.vector.primal
    options:
      show_source: false

::: pymhm.fem.vector.primal_3d
    options:
      show_source: false

::: pymhm.fem.vector.quadrilateral
    options:
      show_source: false

::: pymhm.fem.vector.pressure
    options:
      show_source: false

::: pymhm.fem.vector.pressure_3d
    options:
      show_source: false

::: pymhm.fem.vector.stress
    options:
      show_source: false

::: pymhm.fem.vector.stress_tensor
    options:
      show_source: false

::: pymhm.fem.vector.stress_3d
    options:
      show_source: false

::: pymhm.fem.traces.traction_3d
    options:
      show_source: false

::: pymhm.materials.elasticity
    options:
      show_source: false

::: pymhm.materials.compliance
    options:
      show_source: false

::: pymhm.postprocessing.solutions.ElasticitySolution
    options:
      show_source: false

::: pymhm.postprocessing.primal_elasticity
    options:
      show_source: false

::: pymhm.postprocessing.solutions.GaLS3DSolution
    options:
      show_source: false

::: pymhm.postprocessing.dynamics
    options:
      show_source: false

::: pymhm.postprocessing.stress
    options:
      show_source: false

::: pymhm.postprocessing.stress_tensor
    options:
      show_source: false

::: pymhm.postprocessing.stress_3d
    options:
      show_source: false

## References

- Christopher Harder, Alexandre L. Madureira, and Frédéric Valentin (2016). *A hybrid-mixed method for elasticity*, ESAIM: M2AN 50, 311–336. [DOI: 10.1051/m2an/2015046](https://doi.org/10.1051/m2an/2015046).
- Philippe R. B. Devloo et al. (2021). *New H(div)-conforming multiscale hybrid-mixed methods for the elasticity problem on polygonal meshes*. [DOI: 10.1051/m2an/2021013](https://doi.org/10.1051/m2an/2021013).
- Antônio Tadeu Azevedo Gomes, Weslley da Silva Pereira and Frédéric Valentin (2024). *A low-order locking-free multiscale finite element method for isotropic elasticity*. [arXiv: 2403.16890v1](https://arxiv.org/abs/2403.16890v1).
- Douglas N. Arnold, Richard S. Falk and Ragnar Winther (2007). *Mixed finite element methods for linear elasticity with weakly imposed symmetry*. [DOI: 10.1090/S0025-5718-07-01998-9](https://doi.org/10.1090/S0025-5718-07-01998-9).
