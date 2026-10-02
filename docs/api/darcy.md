# Darcy in two dimensions

Primal, mixed, analytical and residual-enriched Darcy discretizations.

[All API families](../api.md)

::: pymhm.darcy
    options:
      show_source: false

::: pymhm.darcy_mixed
    options:
      show_source: false

::: pymhm.darcy_rt
    options:
      show_source: false

::: pymhm.conforming
    options:
      show_source: false

::: pymhm.analytic
    options:
      show_source: false

::: pymhm.quadrilateral
    options:
      show_source: false

::: pymhm.tensor_rt
    options:
      show_source: false

::: pymhm.pgmhm
    options:
      show_source: false

## Repeated physical RT evaluation

`RTField(mesh, coefficients, degree)` stores an immutable copy of canonical
Raviart–Thomas coefficients and validates it once. `evaluate(points, cell_indices)`
uses the prescribed incident cell at each physical point; it does not average
traces across interfaces. The flux and divergence use the same Piola map and
moment conventions as `rt_evaluate_points`. Point and cell-membership checks
remain active for every batch. Serialization preserves immutable coefficient
storage and its numerical precision.

::: pymhm.rt.RTField
    options:
      show_source: false
