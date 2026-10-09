# Scalar MHM with UNUSUAL local stabilization

PyMHM implements the scalar MHM-UNUSUAL formulation of
[Santiago, Valentin and Martins (2025)](https://doi.org/10.55592/cilamce2025.v5i.14270),
Eqs. (14)–(19), for two-dimensional reaction–diffusion. The numerical evidence
here includes independent DOLFINx/UFL operators, an executed comparison with
**MHMUN-RAD_Parallel / FreeFem++**, and analytical tests over five spatial
resolutions. These establish the stated discrete formulation and its measured
accuracy; they do not constitute reproduction of the paper's SPE10 figure.

## Operator and boundary conventions

The physical equation and flux are

$$
\begin{aligned}
\sigma u-\nabla\cdot(A\nabla u)&=f,\\
q&=-A\nabla u.
\end{aligned}
$$

Here \(A\) is symmetric positive definite and \(\sigma\geq0\). Each local fine
triangle supports continuous Lagrange polynomials. The global unknown represents
oriented normal flux moments on independently specified macroface spaces.
Dirichlet data may be imposed weakly through the global equations or strongly
on exterior local nodes. In this page's campaigns, Dirichlet data are imposed
**strongly**, as in the supplied FreeFem application. `neumann` prescribes the
outward physical flux \(q\cdot n\); it is not the opposite-sign conormal
\(A\nabla u\cdot n\).

Set \(Lv=\sigma v-\nabla\cdot(A\nabla v)\). The local bilinear form and load are

$$
\begin{aligned}
a_{\rm UN}(u,v)
&=(A\nabla u,\nabla v)+(\sigma u,v)
  -\sum_{T\subset K}\tau_T(Lu,Lv)_T,\\
\ell_{\rm UN}(v)
&=(f,v)-\sum_{T\subset K}\tau_T(f,Lv)_T.
\end{aligned}
$$

The boundary functional retains the standard MHM sign and orientation. The
stabilization is **negative and symmetric**, and acts on the complete strong
residual. Scalar SUPG is a different formulation: with zero advection, its
streamline term vanishes. `stabilization="unusual"` therefore has its own
operator and requires zero advection.

The parameter is the algebraically equivalent, division-safe form of Eq. (15):

$$
\tau_T=
\frac{m_T h_T^2}
{\max\{\sigma_{\max,T}m_T h_T^2,\,2A_{\min,T}\}+2A_{\min,T}}.
$$

The length \(h_T\) is the physical triangle diameter. The declared material
bounds must hold throughout the fine cell. With \(\sigma=0\), the continuous
limit is \(\tau_T=m_T h_T^2/(4A_{\min,T})\); no reaction floor is introduced.

## Inverse constants and admissibility

For constant diffusion and P1 locals, the implementation uses the published
\(m_T=1/3\). For higher degrees or variable diffusion, the automatic choice
satisfies the sufficient discrete inverse bound

$$
m_T h_T^2\,\lVert\nabla\cdot(A\nabla v_h)\rVert_T^2
\leq A_{\min,T}^2\,\lVert\nabla v_h\rVert_T^2,
\qquad 0<m_T\leq\tfrac13.
$$

A generalized eigenproblem computes this bound on the actual physical local
polynomial space. The constant-gradient null mode is removed exactly. This
choice specifies the otherwise unspecified higher-degree constants in the
conference paper; it is not presented as a recovered historical parameter.

An explicitly supplied `inverse_constant` is instead checked directly against
local coercivity. On the complement of the constant kernel when necessary,
PyMHM requires

$$
\sup_{v_h\neq0}
\frac{\tau_T\lVert Lv_h\rVert_T^2}
{(A\nabla v_h,\nabla v_h)_T+(\sigma v_h,v_h)_T}<1.
$$

This accepts a stable explicit constant even when it exceeds the conservative
automatic bound. The check uses the same material, basis and quadrature as the
operator. It proves positivity of that discrete element form, not a
parameter-uniform approximation theorem or an inf-sup condition for arbitrary
trace spaces. Adequate quadrature and bounds valid between samples remain
necessary.

The spatial families used here satisfy the degree/refinement configurations in
Theorem 1: P1/P0 with one red local refinement, P2/P1 with one red refinement,
and P2/P0 or P3/P1 with an unrefined local triangle for the native comparison.
Increasing the local degree alone does not remove error from a fixed macro
trace space.

## Using variable materials

The example helpers below are repository sources, supplied separately from the
installed library. Open the corresponding notebook to acquire its verified
companion, as described in the [notebook guide](../tutorials.md#execute-downloaded-notebooks).

```python
import numpy as np
from pymhm import TriangleMesh
from examples.formulations.application import transport as solve_transport
from pymhm.fem.scalar.unusual import UnusualParameters

mesh = TriangleMesh.unit_square(4)
A0 = np.array([[2.0, 0.3], [0.3, 1.0]])
solution = solve_transport(
    mesh,
    diffusion=lambda x: (1 + x.sum(axis=1))[:, None, None] * A0,
    diffusion_divergence=A0 @ np.ones(2),
    reaction=lambda x: 3 + x[:, 0],
    source=1.0,
    degree=2,
    stabilization="unusual",
    unusual_parameters=UnusualParameters(
        diffusion_lower=np.linalg.eigvalsh(A0)[0],
        reaction_upper=4.0,
    ),
)
```

For a tensor callback, `diffusion_divergence` supplies
\((\operatorname{div}A)_j=\sum_i\partial_i A_{ij}\). The local operator contains
both \(A:D^2v\) and \((\operatorname{div}A)\cdot\nabla v\). Constant tensors need
no derivative input. A bound callback receives fine-cell centroids, but must
return a bound valid on the **whole cell**, not just the center.

Discontinuous Cartesian materials are supported when their interfaces coincide
with local fine-cell edges. Crossing an unresolved material jump is rejected:
volume quadrature cuts alone do not make the distributional strong residual an
\(L^2\) function. This restriction is specific to the residual formulation;
Galerkin integration with material cuts has a different contract.

## Analytical campaign

The mixed-boundary problem in §4.1 has \(f=\sigma=1\),
\(A=\epsilon I\), zero Dirichlet values on \(x=0,1\), and zero normal flux on
\(y=0,1\). The exact solution can be evaluated without growing exponentials:

$$
u(x,y)=1-
\frac{e^{-x/\sqrt\epsilon}+e^{-(1-x)/\sqrt\epsilon}}
{1+e^{-1/\sqrt\epsilon}}.
$$

Five macro grids use \(n=2,4,8,16,32\) divisions per side, two SW–NE triangles
per square, P1 locals on one red refinement, and unsplit P0 macro traces.
Both MHM-Galerkin and MHM-UNUSUAL use identical spaces and data. The smooth
case has \(\epsilon=1\); the layer case has \(\epsilon=10^{-3}\).

A separate manufactured test uses

$$
\begin{aligned}
A(x,y)&=(1+x+y)
\begin{pmatrix}2&0.3\\0.3&1\end{pmatrix},\\
\sigma(x,y)&=3+x,\qquad u(x,y)=\sin(\pi x)\sin(\pi y),\\
f&=\sigma u-\nabla\cdot(A\nabla u).
\end{aligned}
$$

This test has P2 locals, P1 macro traces, one red refinement, and homogeneous
Dirichlet values. It exercises the material derivatives and off-diagonal terms
on five levels, using the automatic inverse constant.

![Absolute scalar and physical flux errors for five macro resolutions](../figures/unusual/convergence.png)

At the finest level, the MHM-UNUSUAL results are:

| Test and local/trace spaces | Relative scalar error | Relative flux error |
|---|---:|---:|
| Smooth, P1/P0 | 0.11464% | 3.3114% |
| Reaction layer, P1/P0 | 1.1685% | 23.2268% |
| Variable tensor, P2/P1 | 0.00075305% | 0.039767% |

The P1/P0 layer flux remains underresolved: a small scalar error and an
algebraic residual below \(4\times10^{-16}\) do not imply an accurate gradient.
The last refinement approximately halves the smooth P1 flux error, while the
P2 tensor flux error decreases by a factor close to four. Increasing error
quadrature from 12 to 20 points per Duffy coordinate changes the recorded
finest-level norms by less than \(5\times10^{-13}\) relatively.

The plotted norms are absolute broken physical norms, integrated over every
fine triangle. The raw flux is \(-A\nabla u_h\), with no smoothing or
H(div) reconstruction. The diffusion-energy error is
\(\lVert A^{1/2}\nabla(u-u_h)\rVert\); it is distinct from the flux error
\(\lVert A\nabla(u-u_h)\rVert\). Relative values in the records divide by the
corresponding exact field norm.


At fixed \(n=16\), the diffusion sweep uses
\(\epsilon=1,10^{-2},10^{-3},10^{-4},10^{-5}\). The maximum nodal error measures
the quantity used in Fig. 2 of the paper; the profile retains separate
one-sided values at every macro crossing. This is the same analytical PDE,
with the explicit spaces and connectivity above, rather than a claim that the
historical figure's local mesh has been reconstructed.


At \(\epsilon=10^{-5}\), both fixed-space solutions remain inaccurate.
UNUSUAL reduces the maximum nodal error from 0.90012 to 0.69859 and the maximum
value from 1.44063 to 1.35376, while the exact solution remains below one.
Its relative scalar \(L^2\) error is **11.3504%**, versus **8.9354%** for
Galerkin; both relative flux errors are approximately **88.21%**. Thus this
experiment does not establish a discrete maximum principle or uniformly
smaller errors. The quadrature check uses orders 32 and 40 and preserves these
conclusions. Resolving the layer and enriching the trace remain necessary.

### Resolution control with enriched local and trace spaces

The additional P3/P2 control retains the same physical layer
(\(\epsilon=10^{-3}\)), macro triangulation and one red local refinement. It
changes the approximation spaces explicitly; it is not relabeled as the P1/P0
experiment or as a recovered historical configuration.

| Macro divisions \(n\) | Relative scalar error | Relative flux error |
|---:|---:|---:|
| 8 | 0.24067% | 3.73282% |
| 16 | 0.029073% | 0.793912% |
| 32 | 0.0023422% | 0.121856% |

The substantially smaller flux error demonstrates the role of resolution and
trace richness. Stabilization alone does not provide the missing approximation
modes. These norms use orders 12 and 20, agreeing within
\(2\times10^{-13}\) relatively. The field below uses \(n=16\), and the profile
above includes its independent one-sided values.

![Exact and P3/P2 physical flux component with signed error and macro boundaries](../figures/unusual/resolved-layer-flux.png)

## Independent verification

The DOLFINx/UFL integration tests differentiate the strong residual independently
and compare complete element matrices and loads for P1, P2 and P3, with constant
and variable anisotropic tensors. The portable tests also cover closed-form P1
moments, nonhomogeneous mixed-boundary polynomial patches, pure-Neumann gauges,
the zero-reaction limit, inadmissible parameters and material interfaces.

The executed **MHMUN-RAD_Parallel** application uses FreeFem++ 4.13 and archived
revision `27bd1ecdba684354c6f147be3ceb0021f3cb394a`. Its source was supplied
privately; a public source URL is unavailable. The comparison leaves its
operator unchanged and adds only output precision and native polynomial/gradient
sampling. It covers five grids, from 2 to 512 macro triangles, for each of
P2/P0 and P3/P1. The exact test is
\(u=\sin(2\pi y)\cos(\pi x)\), with \(A=I\), \(\sigma=1\), zero Dirichlet data
on the horizontal sides and zero normal flux on the vertical sides. Explicit
constants are \(1/48\) and \(1/149.1\), respectively, as in that application.

![PyMHM versus MHMUN-RAD_Parallel and FreeFem++, separately for scalar and flux fields](../figures/unusual/native-comparison.png)

Across the ten matched runs, the maximum relative scalar difference is
\(1.004\times10^{-12}\) and the maximum relative flux difference is
\(8.861\times10^{-13}\). The two implementations use independent assembly and
solvers. Their integrated
field agreement validates the matching discrete problems; its small magnitude
must not be confused with the approximation error relative to the exact field.
The analytical conference experiment and this supplied trigonometric example
are different data sets.

## Conservation and scope

Testing a stabilized local equation with one gives, where that test is admissible,

$$
\int_K\sigma u_h+\int_{\partial K}\lambda_h-\int_K f
=\sum_{T\subset K}\tau_T(\sigma,Lu_h-f)_T.
$$

Consequently, the skeleton multiplier does not satisfy the unmodified
reaction balance unless the residual term on the right vanishes. The computed
raw gradient flux is not claimed to be H(div)-conforming or fine-cell
conservative. Pure diffusion removes this particular reaction correction.

The implementation covers triangular two-dimensional meshes, arbitrary scalar
local polynomial degrees, compatible macroface degrees/subdivisions, mixed
boundary data, and the coefficient contracts stated above. It does not identify
this method with Oseen/Stokes USFEM, scalar SUPG, or an advection extension.
The SPE10 experiment in §4.2 is not numerically reproduced on this page;
its historical local mesh, degree and reference-field data are not established
by the comparisons reported here.

## Reproducing the package's analytical evidence

```sh
pixi run --locked -e test-core python -m examples.solve_unusual
pixi run --locked -e test-core python -m examples.verify_unusual_resolution
pixi run --locked -e notebooks python -m examples.plot_unusual
pixi run --locked -e fem python -m pytest -q tests/test_unusual_fenics.py
```

The producer records mesh sizes, quadrature checks, original-equation residuals,
source digests, exact-field norms and field-archive checksums. The plotter only
replays those archives. Numerical records are available as
[analytical results](../figures/unusual/analytical.json) and
[native comparison](../figures/unusual/native-comparison.json), and
[resolution control](../figures/unusual/resolution-control.json).

## References

- Juan Felipe Pacazuca Santiago, Frédéric Valentin, and Larissa Martins (2025). *A Multiscale Hybrid-Mixed Method with Local Stabilization*. Proceedings of the Ibero-Latin American Congress on Computational Methods in Engineering, CILAMCE 2025, volume 5, article 14270; published online 18 March 2026. [DOI: 10.55592/cilamce2025.v5i.14270](https://doi.org/10.55592/cilamce2025.v5i.14270).
