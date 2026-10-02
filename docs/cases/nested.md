# Recursive MHM local problems

An MHM discretization can itself supply a local problem at the next scale.
This is the multilevel interpretation described by Harder, Paredes and Valentin
([2013, Remarks 7 and 10](https://doi.org/10.1016/j.jcp.2013.03.019)) and the
local variational decomposition of Harder and Valentin
([2016, sections 2 and 4.1](https://doi.org/10.1007/978-3-319-41640-3_13)).
The adapter below is an original algebraic realization of that decomposition.

## Local equations and physical moments

Let \(Ax=b\) be an inner hybrid system, including its retained local modes.
The columns of \(E\) select the inner boundary trace. An injective map \(M\)
restricts the parent trace coefficients to that boundary, with the normal
orientations included. The parent local problem is

$$
\begin{bmatrix}-A&E\\E^T&0\end{bmatrix}
\begin{bmatrix}x\\\mu\end{bmatrix}
+\begin{bmatrix}0\\-M\end{bmatrix}\lambda
=\begin{bmatrix}-b\\0\end{bmatrix}.
$$

Thus \(E^Tx=M\lambda\). The reaction \(\mu\) is minus the inner potential
moment, so the parent pairing \(-M^T\mu\) has the original continuity sign.
This system is condensed using the same `LocalProblem` implementation as other
formulations. The supplied physical kernel and moment constraints survive the
second elimination. Left and right kernels can differ for Petrov–Galerkin
operators.

`nested_trace_map` checks exact representability on each inner boundary face.
A coarse jump cannot be silently projected into an unsplit constant fine trace.
`NestedLocalProblem.moment` returns both the linear moment row and its source
offset; both are needed when prescribing a physical average. After the parent
solve, `reconstruct` recovers the inner unknowns and each inner volume field.
The construction can be repeated at further levels.

## Field equivalence and convergence

The test problem is
\(p=\sin(\pi x)\sin(\pi y)\), \(K=I\),
\(f=2\pi^2p\) on the unit square, with homogeneous Dirichlet data.
Each outer macrorectangle contains \(2\times2\) inner macrorectangles;
each inner macrorectangle contains \(2\times2\) Q2 fine cells. The inner
faces carry P1 traces, and every outer face carries two independent P1 segments.
Consequently the recursive and ordinary one-level MHM discretizations have the
same leaf approximation and trace spaces.

| Outer divisions | Outer unknowns | One-level unknowns | Pressure L2 error | Flux L2 error | Relative difference of leaf coefficients |
|---:|---:|---:|---:|---:|---:|
| 4 | 176 | 352 | 9.329e-5 | 7.317e-3 | 2.230e-15 |
| 8 | 640 | 1344 | 1.160e-5 | 1.825e-3 | 1.653e-15 |
| 16 | 2432 | 5248 | 1.448e-6 | 4.560e-4 | 3.824e-15 |

All five measured levels, including divisions 1 and 2, are in
`examples/results/nested.json`. Pressure and flux approach third and second
order, respectively. The reduction in outer unknowns measures condensation;
it does not by itself demonstrate a runtime or memory advantage.

![Recursive and ordinary MHM convergence](../figures/nested/convergence.png)

![Pressure and signed flux components on both macro levels](../figures/nested/fields.png)

Dark lines mark the outer macro mesh; thin lines mark the inner macro mesh.
The pressure and both signed physical flux components retain independent local
values at interfaces. No averaging between inner macrorectangles is applied.

The lightweight tests also use nonhomogeneous boundary data, constant source,
pure-Neumann compatibility and a prescribed physical mean. Each recursive
field is compared with the ordinary discretization and an exact polynomial.
The research campaign uses integrated errors with sixth-order tensor Gauss rules.

```sh
pixi run -e notebooks verify-nested
```

See also `notebooks/36_recursive_mhm.ipynb`,
the [API](../api/hybrid.md#pymhm.nested), and
[operator reuse](../execution.md). `restrict_response` reuses prepared harmonic
lifts on an exactly embedded skeletal subspace without refactoring a local matrix.
