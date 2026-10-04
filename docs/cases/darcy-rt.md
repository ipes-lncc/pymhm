# RT0, RT1 and RT2 mixed Darcy

`solve_darcy_rt` uses a Raviart–Thomas flux of mathematical order
\(m=0,1,2\) and discontinuous \(\mathbb P_m\) pressure on every fine triangle.
The normal component has degree \(m\) on each fine edge, while the divergence
belongs to \(\mathbb P_m\). This degree convention corresponds to Basix/DOLFINx
RT element degree \(m+1\).

The local equations use inverse-permeability flux mass, the pressure/divergence
pairing, and fine-boundary pressure multipliers that enforce the macro normal
flux moments. One joint constant-pressure/boundary-pressure null mode is retained
per macrocell. The MHM multiplier is the physical flux \(q\cdot n\), not a
pressure trace. The formulation enforces every discontinuous \(\mathbb P_m\)
moment of \(\nabla\cdot q-f\) in each fine cell. This differs from testing a
reconstructed flux only against continuous macro-local pressure functions.

```python
from pymhm import TriangleMesh
from pymhm._legacy.models.darcy.mixed_rt import solve_darcy_rt, solve_darcy_rt_conforming

macro = TriangleMesh.unit_square(4)
solution = solve_darcy_rt(
    macro, degree=2, local_refinement=2, source=1.0, dirichlet=0.0
)
reference = solve_darcy_rt_conforming(
    TriangleMesh.unit_square(8), degree=2, source=1.0, dirichlet=0.0
)
```

The first call uses a macro skeleton; the second assembles a classical globally
conforming RT2/P2 mixed system directly on the fine mesh. It does not impose a
macroface polynomial restriction. Both use the same material and physical
boundary conventions. A classical discrete solution remains a numerical
reference whose refinement must be checked.

## Trace spaces, data and diagnostics

The default skeleton has one polynomial of degree \(m\) per macroface. Lower
degrees, segmented faces and within-face continuous polynomial traces are also
accepted. Every segment boundary must align with a fine boundary edge, and its
degree may not exceed \(m\). Orientation includes both normal signs and reversal
of odd Legendre moments. Fine-cell interior RT moments remain independent of
the skeleton degree.

Unspecified exterior faces receive weak pressure data. `neumann` maps exterior
face indices to outward physical normal flux, projected into its full fine-edge
moment space for the classical solve, or the selected skeletal space for MHM.
Pure Neumann data require source/flux compatibility and use one physical
pressure mean. No artificial pressure penalty replaces that gauge.

Scalar, symmetric positive-definite tensor and callback permeabilities are
supported. `CartesianCellField` invokes exact geometric material-intersection
quadrature, so quadrature points sample every positive-area material piece.
This integrates discontinuous coefficients without enriching the polynomial
approximation across an interface that cuts a fine triangle.

`l2_error`, `flux_l2_error` and `divergence_l2_error` integrate the complete
pressure, flux and divergence. `fine_equilibrium_residuals` reports every
DG-Pm moment; summing these moments gives the fine-cell conservation defect.
`normal_flux_residuals` compares local exterior RT moments with the MHM trace.
For nonpolynomial sources, moment conservation does not imply pointwise equality
between divergence and source.

The local-factory interface assembles and condenses independent macro problems
in serial, threads or spawned processes. Direct local/global backends remain
selectable; an SPD AMG method is not applied directly to the mixed local saddle.
The [block preconditioner](../solvers.md#amg-block-preconditioners-for-mixed-saddle-systems)
requires its explicitly verified block structure.

## Five-level analytical convergence

The campaign uses the unit square, \(K=I\),
\(p=\sin(2\pi x)\sin(2\pi y)\), \(f=8\pi^2p\), and homogeneous pressure
on the entire boundary. Macro grids have \(n=1,2,4,8,16\) subdivisions per
coordinate, two fine edge divisions per macrotriangle, and an unsegmented
skeletal polynomial of degree \(m\). The classical mixed references use the
same uniform fine spacing. Assembly uses positive quadrature of order six;
error norms use order eight.

![RT-family MHM and classical mixed convergence](../figures/darcy-rt/convergence.png)

[Convergence figure as SVG](../figures/darcy-rt/convergence.svg).
Solid curves are MHM; dashed curves are the classical mixed discretization.
Their errors are measured against the exact solution, independently for every
order and grid. The classical curves quantify their own discretization error,
and are not substituted for the analytical field.

On the finest grid (512 macrotriangles and 2,048 fine triangles), the errors are:

| RT order | MHM pressure L2 | MHM flux L2 | Classical pressure L2 | Classical flux L2 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3.33993e-2 | 5.02455e-1 | 3.27026e-2 | 2.51846e-1 |
| 1 | 1.24844e-3 | 1.82349e-2 | 1.24269e-3 | 7.04283e-3 |
| 2 | 3.45343e-5 | 4.72130e-4 | 3.44687e-5 | 1.53645e-4 |

The macroface restriction remains visible in the flux error even when both
methods share the same local RT order and fine spacing. The last refinement
reduces the MHM flux error by factors 1.986, 3.987 and 7.934 for RT0, RT1 and RT2,
respectively. These observations concern this smooth benchmark; they are not
uniform accuracy bounds for heterogeneous coefficients.


![Exact and MHM RT2 pressure and physical flux magnitude](../figures/darcy-rt/fields.png)

[Field figure as SVG](../figures/darcy-rt/fields.svg).
The finest RT2 panels display individual fine-cell centroid samples, with the
actual macro edges overlaid. No interface averaging smooths the broken pressure.
The exact and numerical panels use shared limits, and difference scales are
centered at zero. Volume norms evaluate complete polynomials, not these samples.

## Independent checks and reproduction

Portable tests compare RT0 with the separately assembled existing RT0 method,
verify exactly represented polynomial fluxes for all three orders, and check
anisotropic affine data, mixed/Neumann conditions, physical means, fine moments,
normal traces, material interfaces and process execution. Native optional tests
assemble classical RT0/P0, RT1/P1 and RT2/P2 systems independently in
DOLFINx/UFL and compare pressure and physical flux evaluations on matching cells.

```bash
pixi run -e test python examples/solve_darcy_rt.py
pixi run -e notebooks python examples/plot_darcy_rt.py
pixi run -e fem pytest tests/test_darcy_rt.py -m fem
```

[Acquisition record](https://github.com/volpatto/pymhm/blob/main/examples/results/darcy-rt.json)
and notebook `33_darcy_rt.ipynb` identify the executed spaces, norms, source hashes
and archived fields. This campaign validates the native RT family; it does not
claim that every heterogeneous published RT2 reservoir study has been reproduced.
