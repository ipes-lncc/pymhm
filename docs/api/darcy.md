# Darcy in two dimensions

Primal, mixed, analytical and residual-enriched Darcy discretizations.

[All API families](../api.md)

User-defined formulations compose these public numerical owners with
`LocalEquations` and `Equation`; their editable providers are listed in the
[variational guide](../variational.md#formulations-composed-from-the-same-api).
Physical field records provide optional interpretation of the resulting
coefficients. Named `FieldDefinition` views require no physical solver object.

::: pymhm.fem.scalar.operators
    options:
      show_source: false

::: pymhm.fem.scalar.triangle
    options:
      show_source: false

::: pymhm.fem.scalar.quadrilateral
    options:
      show_source: false

::: pymhm.fem.scalar.metric
    options:
      show_source: false

::: pymhm.fem.scalar.separable
    options:
      show_source: false

::: pymhm.materials.separable
    options:
      show_source: false

::: pymhm.fem.traces.conforming
    options:
      show_source: false

::: pymhm.postprocessing.analytic
    options:
      show_source: false

::: pymhm.postprocessing.conforming
    options:
      show_source: false

::: pymhm.fem.hdiv.mixed
    options:
      show_source: false

::: pymhm.fem.hdiv.rt_trace
    options:
      show_source: false

::: pymhm.fem.hdiv.tensor_rt
    options:
      show_source: false

::: pymhm.postprocessing.solutions.DarcySolution
    options:
      show_source: false

::: pymhm.postprocessing.solutions.RTDarcySolution
    options:
      show_source: false

::: pymhm.postprocessing.solutions.BDMDarcySolution
    options:
      show_source: false

## Repeated physical RT evaluation

`RTField(mesh, coefficients, degree)` stores an immutable copy of canonical
Raviart–Thomas coefficients and validates it once. `evaluate(points, cell_indices)`
uses the prescribed incident cell at each physical point; it does not average
traces across interfaces. The flux and divergence use the same Piola map and
moment conventions as `rt_evaluate_points`. Point and cell-membership checks
remain active for every batch. Serialization preserves immutable coefficient
storage and its numerical precision. Its complete API is documented with the
[RT element family](elements.md#pymhm.fem.hdiv.rt.RTField).
