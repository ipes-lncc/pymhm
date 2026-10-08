# Reaction–advection–diffusion and heat

These cases have explicit analytical solutions. The reference panels are computed
from those expressions, independently of the linear solver. All field images
retain the broken local mesh: opposite sides of a macroface are not merged for
plotting. Dark contours with a light halo identify the actual macrotriangles
on every field map, including the analytical and error panels. Analytical
references are sampled at local mesh vertices and
interpolated for rendering. The reported L2 errors use triangle quadrature; the error images show
errors at local P1 nodes and their interpolation, not a second error norm.

Both calculations use pyMHM's native NumPy/SciPy backend. Their reference data
are the analytical expressions below; no external MSL, MFEM, or NeoPZ run is
used for these figures.

## A smooth transported field

Solve the conservative equation

$$
\nabla\cdot(-0.2\nabla u+\boldsymbol\beta u)+2u=f,
\qquad \boldsymbol\beta=(1,1/2),
$$

on the unit square with zero Dirichlet data. Choose
\(u_\star=\sin(\pi x)\sin(\pi y)\) and differentiate it to obtain

$$
f=(0.4\pi^2+2)u_\star
+\pi\cos(\pi x)\sin(\pi y)
+\frac{\pi}{2}\sin(\pi x)\cos(\pi y).
$$

The expected solution is a smooth, positive central peak, symmetric in its
analytical value even though the prescribed advection is directional: the
manufactured forcing supplies the corresponding asymmetric terms. This is a
resolved, moderate-advection test, not a boundary-layer or high-Péclet benchmark.

![Exact, numerical and nodal error fields for reaction-advection-diffusion](../figures/scalar-elasticity/rad-fields.svg)

The calculation uses 32 macrotriangles, four subdivisions per macro edge, local
P1 fields and degree-one traces. Assembly uses Duffy quadrature order 4 and
error integration uses order 8. The measured L2 error is **3.7877e-3**. Exact and
numerical panels use the same color limits, including any small numerical
undershoot. The much smaller third-panel scale must not be confused with the
solution scale.

![Numerical and exact scalar profile through y=0.37](../figures/scalar-elasticity/rad-profile.svg)

The dotted vertical markers show where the line at \(y=0.37\) intersects the
actual macrofaces, including diagonal edges. Small reconstructed field jumps are
permitted because continuity is enforced through skeletal moments. At a point
exactly on a macroface, the sampling routine selects the first containing cell;
the two traces remain distinct.

The multiplier here is the Robin flux
\((-0.2\nabla u+\boldsymbol\beta u/2)\cdot n\), rather than the full physical
flux \((-0.2\nabla u+\boldsymbol\beta u)\cdot n\). See the
[RAD formulation](../theory.md) for the boundary convention.

## Heat decay with an exact eigenmode

For \(u_t-\Delta u=0\), homogeneous boundary data and initial value
\(u_0=\sin(\pi x)\sin(\pi y)\), the exact solution is

$$
u_\star(x,y,t)=e^{-2\pi^2t}\sin(\pi x)\sin(\pi y),\qquad
\|u_\star(t)\|_{L^2}^2=\tfrac14 e^{-4\pi^2t}.
$$

The shape remains the same while its amplitude decays. Backward Euler has the
time-only amplitude \((1+2\pi^2\Delta t)^{-N}\), with \(t=N\Delta t\), if the
spatial eigenfunction is represented exactly. This predicts a larger amplitude
than the exact exponential at a fixed positive time. Spatial discretization
adds its own error; that time-only curve is a reference, not the exact answer
of the discrete MHM system.

![Exact heat field and two time step sizes at t=0.1 with a nodal error map](../figures/scalar-elasticity/heat-fields.svg)

All three solution panels share a color scale and show the same macro mesh.
At \(t=0.1\), the larger step
retains more heat than the reference; halving the time step approaches the exact
field. The spatial discretization is fixed at 32 macrotriangles, local refinement
four and degree-one traces.

| Time step | Steps to t=0.1 | Measured L2 error | Observed combined-error rate |
| ---: | ---: | ---: | ---: |
| 0.025 | 4 | 2.9197e-2 | — |
| 0.0125 | 8 | 1.4426e-2 | 1.017 |
| 0.00625 | 16 | 6.5510e-3 | 1.139 |
| 0.003125 | 32 | 2.5020e-3 | 1.389 |
| 0.0015625 | 64 | 5.7473e-4 | 2.122 |

![Monotone heat energy decay and temporal error compared with exact-space backward Euler](../figures/scalar-elasticity/heat-decay.svg)

The measured squared L2 norm decreases at every computed step. The coarser steps
follow the first-order temporal trend, while the final pair gives an apparent
rate of 2.122 for the **combined spatial and temporal error**. This does not
establish second-order accuracy of backward Euler: spatial and temporal errors
can partially cancel, and the exact-space backward Euler reference remains
first order. Refining time indefinitely on this fixed mesh would not remove
the spatial error floor. Rates in the table are computed as
\(\log_2(E_{\Delta t}/E_{\Delta t/2})\).

## Reproduce and inspect

```bash
pixi run --locked -e notebooks python -m examples.plot_scalar_elasticity
```

The script checks decreasing heat energy and refinement error, writes the
measurements to `examples/results/scalar-elasticity.json`, and exports each
figure as SVG and PNG. Notebook `08_transport_and_heat.ipynb` additionally checks
an exactly representable space-time patch; these smooth fields provide a
nonzero-error visual complement to that patch test.
