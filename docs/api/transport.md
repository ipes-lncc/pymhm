# Transport, reaction and diffusion

Stationary and transient scalar problems, conservative transport and stabilization.

[All API families](../api.md)

User-defined formulations compose these public numerical owners with
`LocalEquations` and `Equation`; their editable providers are listed in the
[variational guide](../variational.md#formulations-composed-from-the-same-api).
Physical field records provide optional interpretation of the resulting
coefficients. Named `FieldDefinition` views require no physical solver object.

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
