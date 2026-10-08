# Elasticity

Displacement, displacement–pressure and weakly symmetric stress formulations.

[All API families](../api.md)

User-defined formulations compose these public numerical owners with
`LocalEquations` and `Equation`; their editable providers are listed in the
[variational guide](../variational.md#formulations-composed-from-the-same-api).
Physical field records provide optional interpretation of the resulting
coefficients. Named `FieldDefinition` views require no physical solver object.

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
