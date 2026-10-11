# Robin MH and three-field MH²M

Native two- and three-dimensional local maps, physical boundary conventions and independent trace spaces.

The Robin multiplier is $(q-p\sigma)\cdot n$, while MH²M's local
conormal is $A\nabla p\cdot n=-q\cdot n$ and its global trace represents
pressure. MH²M uses a boundary-mean-zero local pressure complement.
The reconstruction and gauge conventions are documented in the source
API below; the original formulations are given by
[Barrenechea, Gomes and Paredes (2024)](https://doi.org/10.1137/22M1542556)
and [de Barros, Madureira and Valentin (2026)](https://arxiv.org/abs/2404.16978v3).

Barrenechea et al. (2024) prove their Robin finite-element results in
two dimensions and indicate the three-dimensional extension. De Barros
et al. (2026, version 3) define their functional maps for dimensions two
and three, but their sufficient polynomial-space Fortin constructions
are two-dimensional. The implemented tetrahedral paths have separate
analytical/native verification; passing an algebraic rank check does
not transfer those mesh-uniform stability theorems.

[All API families](../api.md)

::: pymhm.methods.three_field
    options:
      show_source: false

::: pymhm.methods.robin
    options:
      show_source: false

::: pymhm.methods.robin_3d
    options:
      show_source: false

::: pymhm.methods.three_field_3d
    options:
      show_source: false

::: pymhm.fem.traces.pressure_3d
    options:
      show_source: false
