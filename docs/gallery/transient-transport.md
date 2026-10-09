# Transient transport

Time-dependent scalar transport reports concentration and gradient increments
on an explicitly fixed spatial discretization. A temporal study and a spatial
study answer different approximation questions.

| Dimension | Case | Numerical evidence |
| --- | --- | --- |
| 2D | Random-coefficient transport | [Concentration L2 and gradient increments for three time steps](../cases/minimal-convergence.md#transport-random-temporal) |

The accepted series fixes 2048 triangles and continuous P3 fields, with
time steps 0.001, 0.0005 and 0.00025 to final time 0.005. The increments decrease
under time-step halving. These conforming temporal controls do not establish
a spatial MHM convergence rate or a matched transient literature reproduction.

The [transport notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/transport/26_adaptive_transient_transport.ipynb)
identifies its data, acquisition scope and optional full study. Repeated solves
with an unchanged operator can use the [offline/online execution guide](../execution.md#repeated-sources-and-boundary-values);
changing the time-step-dependent operator requires new preparation.

## Native MHM backward-Euler teaching case

The [transient transport tutorial](../tutorials/methods/transient-transport.md)
writes capacity, conservative advection, diffusion and interface UFL forms
explicitly. Its two-dimensional analytical concentration study uses P2 local
fields, segmented P1 traces, fixed spatial spaces and five time increments.
The final measured backward-Euler concentration order is 0.9996. Independent
conforming P2 references are refined on 16, 32 and 64 global intervals before
the finest one is used as a numerical baseline. This native MHM temporal
control is separate from the conforming operator study above and from spatial
MHM convergence claims.

## References

- Christopher Harder, Diego Paredes and Frédéric Valentin (2015).
  *On a Multiscale Hybrid-Mixed Method for Advective-Reactive Dominated Problems
  with Heterogeneous Coefficients*. [DOI: 10.1137/130938499](https://doi.org/10.1137/130938499).
