# Darcy in two dimensions

Primal, mixed, analytical and residual-enriched Darcy discretizations.

[All API families](../api.md)

::: pymhm._legacy.models.darcy.primal
    options:
      show_source: false

::: pymhm._legacy.models.darcy.mixed_bdm
    options:
      show_source: false

::: pymhm._legacy.models.darcy.mixed_rt
    options:
      show_source: false

::: pymhm._legacy.models.darcy.conforming
    options:
      show_source: false

::: pymhm._legacy.models.darcy.analytic
    options:
      show_source: false

::: pymhm._legacy.models.darcy.cartesian
    options:
      show_source: false


::: pymhm.methods.petrov_galerkin
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
