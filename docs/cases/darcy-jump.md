# Darcy face-jump indicator

`estimate_darcy_jumps` evaluates the face residuals in equations (5.1)–(5.3) of
[Araya et al. (2013)](https://doi.org/10.1137/120888223).
The indicator measures mismatches of the broken pressure traces. Its reliability
theorem concerns exact local solution operators; finite-dimensional local solves
introduce an additional error that this indicator alone does not control.

## Definition and boundary conditions

For an interior macroface, let \([p_h]\) denote the difference between its two
one-sided traces. The face residual and calibrated indicator are

$$
R_F=\begin{cases}
-\tfrac12[p_h],&\text{interior},\\
g_D-p_h,&\text{Dirichlet},\\
0,&\text{prescribed Neumann}.
\end{cases}
$$

$$
\begin{aligned}
\eta_F^2&=c_\ell^2\alpha_{\min}H_F^{-1}\|R_F\|_{L^2(F)}^2,\\
\eta_K^2&=\sum_{F\subset\partial K}\eta_F^2,
\qquad\eta^2=\sum_K\eta_K^2.
\end{aligned}
$$

Here \(H_F\) is the entire original macroface length, even if its trace is
segmented, and \(\alpha_{\min}\) is a certified lower eigenvalue bound for the
permeability. An interior face contributes to both incident macro indicators.
Integration splits at all fine-edge and skeletal breakpoints; it preserves
independent one-sided values. The paper uses calibrations \(c_\ell=3,7,18,50\)
for degrees \(\ell=0,1,2,3\) in its numerical experiments. The API default is
one and does not imply a known reliability constant.

Polynomial boundary representability, regularity and exact local lifts are
assumptions of the published theorem. For numerically solved local problems,
the [weighted reconstruction estimator](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/adaptive-darcy.md) accounts separately
for flux, nonconformity, divergence and data terms. Neither a pressure jump of
zero nor a small algebraic residual establishes a small PDE error.

## Five-level analytical study

The problem is the cosine example of §6.1:

$$
\begin{aligned}
p&=\cos(2\pi x)\cos(2\pi y),\qquad K=I,\qquad f=8\pi^2p,\\
q\cdot n&=0,\qquad\int_\Omega p=0,\qquad\Omega=(0,1)^2.
\end{aligned}
$$

Each of \(n^2\) squares is divided along its southwest–northeast diagonal,
giving \(2n^2\) macrotriangles and \(H=\sqrt2/n\). Each macro has four fine
triangles. The two sequences use \((\ell,k)=(0,2)\) and \((3,5)\), with one
polynomial trace per macroface and continuous local \(P_k\) pressures.
These explicit local discretizations and connectivity are part of this
verification; equality with the historical numerical implementation is not
asserted.

The measured error uses the sum plotted in the paper,

$$
E=\|\nabla(p-p_h)\|_{\mathrm{div},H}
   +H^{-1}\|p-p_h\|+\|\nabla(p-p_h)\|,
$$

$$
\|\nabla e\|_{\mathrm{div},H}^2
 =\sum_K\bigl(\|\nabla e\|_K^2+H_K^2\|\Delta e\|_K^2\bigr).
$$

All derivatives are evaluated within the fine triangles. Thus the divergence
term is broken, without distributional interface jumps. Assembly and error
integration use Duffy order 10; order 12 independently checks the finest levels.

| \(n\) | \(E\), \(\ell=0\) | \(\eta\), \(c_0=3\) | \(\eta/E\) | \(E\), \(\ell=3\) | \(\eta\), \(c_3=50\) | \(\eta/E\) |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 6.51466 | 3.83921 | 0.589318 | 1.87640e−2 | 3.12937e−2 | 1.66775 |
| 8 | 2.56004 | 2.07700 | 0.811315 | 1.16330e−3 | 1.95826e−3 | 1.68337 |
| 16 | 1.15776 | 1.06074 | 0.916201 | 7.26816e−5 | 1.22396e−4 | 1.68401 |
| 32 | 0.561233 | 0.533243 | 0.950128 | 4.54315e−6 | 7.65006e−6 | 1.68387 |
| 64 | 0.278309 | 0.266983 | 0.959305 | 2.83980e−7 | 4.78138e−7 | 1.68370 |

![Measured error sums, calibrated indicators and their ratios](../figures/darcy-jump/refinement.png)

The final error rates are approximately 1.01 and 4.00. The constant-trace
indicator is below the recorded error throughout this finite-local sequence;
it is not presented as a certified upper bound. The cubic-trace ratio approaches
1.684 for the stated calibration. Changing the calibration scales that ratio
without changing the numerical solution.

## Reproduction

```bash
pixi run --locked -e notebooks python -m examples.darcy_jump_campaign --collect
pixi run --locked -e notebooks python -m examples.darcy_jump_campaign
```

The ten numerical records and macro-indicator arrays are in
`examples/results/darcy-jump`. [Notebook 42](../tutorials/notebooks.md) executes a small
analytical case and displays the archived campaign. Light tests verify face
multiplicity, permeability scaling, Neumann classification, affine consistency
and input restrictions. The multilevel experiment runs separately from CI.

## References

- Rodolfo Araya, Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *Multiscale Hybrid-Mixed Method*, SIAM Journal on Numerical Analysis 51(6), 3505–3531. [DOI: 10.1137/120888223](https://doi.org/10.1137/120888223).
