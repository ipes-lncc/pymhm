# Transport, reaction and diffusion

Stationary and transient scalar problems, conservative transport and stabilization.

[All API families](../api.md)

User-defined formulations compose these public numerical owners with
`LocalEquations` and `Equation`; their editable providers are listed in the
[variational guide](../variational.md#formulations-composed-from-the-same-api).
Physical field records provide optional interpretation of the resulting
coefficients. Named `FieldDefinition` views require no physical solver object.

Conservative RAD uses the Robin multiplier
$\lambda_K=(-K\nabla u+\beta u/2)\cdot n_K$, while its physical flux is
$j=-K\nabla u+\beta u$. The low-level `p1_operators` function uses the
conventional advective volume form; `triangle_transport_operators` and
`tetra_transport_operators` supply the conservative skew form. Scalar UNUSUAL
subtracts the complete reaction–diffusion residual pairing and its source
term; SUPG instead tests the conservative residual along the velocity.
The original local/global equations and analytical assumptions are given by
[Harder, Paredes and Valentin (2015)](https://doi.org/10.1137/130938499)
and [Araya et al. (2024)](https://doi.org/10.1016/j.cma.2024.117089).
The API below documents the implemented operators and boundary conventions.

Three-dimensional tetrahedral and polyhedral examples share the editable
`examples/formulations/scalar_3d.py` four-block provider. The importable
`define_transport_3d` composition selects only geometry and trace integration;
`scalar_constraints_3d` builds the complete physical integral when the declared
boundary/operator admits a constant gauge. The [tetrahedral notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/transport/41_rad3d.ipynb)
and [polyhedral notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/transport/49_polyhedral_rad.ipynb) show assembly and recovery
without a prepared method solver. Selective local counts come from the actual
coefficient samples and boundary tangency, without coordinator operator assembly
or numerical nullspace thresholds.

::: pymhm.fem.scalar.operators
    options:
      show_source: false

::: pymhm.fem.scalar.stabilization
    options:
      show_source: false

::: pymhm.fem.scalar.transport
    options:
      show_source: false

::: pymhm.fem.scalar.unusual
    options:
      show_source: false

::: pymhm.fem.scalar.transport_3d
    options:
      show_source: false

::: pymhm.materials.macro
    options:
      show_source: false

::: pymhm.materials.dispersion
    options:
      show_source: false

::: pymhm.postprocessing.velocity
    options:
      show_source: false

::: pymhm.postprocessing.transport
    options:
      show_source: false

::: pymhm.fem.traces.scalar
    options:
      show_source: false

::: pymhm.fem.traces.jump
    options:
      show_source: false

::: pymhm.adaptivity.transport
    options:
      show_source: false

::: pymhm.postprocessing.solutions.ScalarSolution
    options:
      show_source: false

::: pymhm.postprocessing.solutions.RAD3DSolution
    options:
      show_source: false

## References

- Christopher Harder, Diego Paredes, and Frédéric Valentin (2015). *On a Multiscale Hybrid-Mixed Method for Advective-Reactive Dominated Problems with Heterogeneous Coefficients*, Multiscale Modeling & Simulation 13(2), 491–518. [DOI: 10.1137/130938499](https://doi.org/10.1137/130938499).
