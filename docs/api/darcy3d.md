# Darcy in three dimensions

Tetrahedral, prismatic and mapped mixed formulations with their explicit geometry and degree contracts.

[All API families](../api.md)

For primal tetrahedral MHM, `TriangularSkeleton(..., continuous=True)` selects
continuous piecewise polynomials within each macroface. A boolean sequence
mixes continuous and discontinuous faces. Face degrees and partitions remain
independent of the local finite-element degree. Continuous faces accept every
positive polynomial degree, subject to the
[trace compatibility conditions](../cases/darcy3d.md).

User-defined formulations compose these public numerical owners with
`LocalEquations` and `Equation`; their editable providers are listed in the
[variational guide](../variational.md#formulations-composed-from-the-same-api).
Physical field records provide optional interpretation of the resulting
coefficients. Named `FieldDefinition` views require no physical solver object.

::: pymhm.fem.scalar.tetrahedron
    options:
      show_source: false

::: pymhm.fem.hdiv.mixed_3d
    options:
      show_source: false

Affine H(div) families, quadrature and their moment conventions are documented
in the [finite-element API](elements.md).

::: pymhm.fem.hdiv.mapped
    options:
      show_source: false

::: pymhm.fem.traces.polygon_3d
    options:
      show_source: false

::: pymhm.fem.traces.triangle_3d
    options:
      show_source: false

::: pymhm.fem.traces.pressure_3d
    options:
      show_source: false

::: pymhm.postprocessing.solutions.Darcy3DSolution
    options:
      show_source: false

::: pymhm.postprocessing.solutions.Mixed3DDarcySolution
    options:
      show_source: false
