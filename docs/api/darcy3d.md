# Darcy in three dimensions

Tetrahedral, prismatic and mapped mixed formulations with their explicit geometry and degree contracts.

[All API families](../api.md)

For primal tetrahedral MHM, `TriangularSkeleton(..., continuous=True)` selects
continuous piecewise polynomials within each macroface. A boolean sequence
mixes continuous and discontinuous faces. Face degrees and partitions remain
independent of the local finite-element degree. Continuous faces accept every
positive polynomial degree, subject to the
[trace compatibility conditions](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/darcy3d.md).

::: pymhm.fem.traces.triangle_3d.TriangularSkeleton
    options:
      show_source: false

::: pymhm._legacy.models.darcy.primal_3d
    options:
      show_source: false

::: pymhm._legacy.models.darcy.mapped
    options:
      show_source: false



::: pymhm._legacy.models.darcy.hdiv_3d
    options:
      show_source: false
