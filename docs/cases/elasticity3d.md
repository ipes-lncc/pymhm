# General-tensor elasticity in three dimensions

`solve_elasticity_3d` implements the primal tetrahedral MHM construction of
[L17](../literature.md), including all six rigid motions. It accepts continuous
local P1–P4 displacement fields, triangular vector traces containing P1, mixed
physical boundary conditions and pure traction with six prescribed displacement
moments. General stiffness is supplied as a symmetric positive-definite 6×6
Kelvin matrix, or as a fourth-order Cartesian tensor with the major and minor
symmetries. Coefficients may vary with physical position.

This is a displacement formulation. Its finite-dimensional stress is symmetric
but is not generally H(div)-conforming. The method is not claimed to be uniformly
locking-free, and the first Lamé modulus must remain finite. Positive definiteness
is checked at the integration points; material interfaces require a mesh and
integration rule that resolve them.

## Formulation and verification

The constitutive convention is

$$
-\operatorname{div}\sigma=f,\qquad
\sigma=C:\varepsilon(u),\qquad
\varepsilon(u)=\tfrac12(\nabla u+\nabla u^T).
$$

The orthonormal Kelvin order is
\((xx,yy,zz,\sqrt2 yz,\sqrt2 xz,\sqrt2 xy)\). The six local null modes are three
translations and \(e_i\times(x-x_K)\). Their coefficients are fixed through
physical L2 moments, rather than independent nodal pins. The skeleton variable
uses \(\lambda=-\sigma n\), while the public `traction` data are the physical
outward traction. Force and moment balance are checked separately.

Native DOLFINx/UFL assembly tests compare complete P1–P4 matrices and loads on
a sheared tetrahedron with a spatially varying anisotropic stiffness. Additional
patches verify variable coefficients, component ordering, traction signs, six
rigid gauges and segmented traces. These checks complement the following
original numerical cases; L17 itself does not specify a numerical benchmark.

## Spatial convergence with variable anisotropy

On the unit cube, define

$$
\psi=\sin^2(\pi x)\sin^2(\pi y)\sin^2(\pi z),\qquad
u=(\partial_y\psi,-\partial_x\psi,0).
$$

Thus the exact displacement vanishes on the boundary and has zero divergence.
The stiffness is

$$
C(x)=(1+x+2y+3z)
\left[\operatorname{diag}(5,6,7,2,3,4)+0.1\mathbf1\mathbf1^T\right].
$$

The force is obtained by differentiating the full stress, including derivatives
of the coefficient. Five macro grids have \(6n^3\) tetrahedra, \(n=1,2,3,4,5\).
The two local choices are P2 with two subdivisions per edge and P3 on one local
tetrahedron; both use vector P1 traces. Assembly and error quadrature orders are
7 and 9, respectively.

![Three-dimensional general-tensor convergence](../figures/elasticity3d/convergence.svg)

At the finest grid, the displacement/stress L2 errors are 0.0287114/11.9882 for
P2 and 0.0308053/10.9126 for P3. These are absolute physical volume norms. The
largest algebraic residual in the spatial studies is below \(2.4\times10^{-16}\).
Small residuals establish agreement with the assembled equations, independently
of the discretization errors shown above.

![P2 displacement and stress slice](../figures/elasticity3d/fields-p2.svg)

![P3 displacement and stress slice](../figures/elasticity3d/fields-p3.svg)

The slice is at \(z=3/8\), with actual macro intersections highlighted. Each
fine-tetrahedron intersection has a separate triangular display grid with six
subdivisions. The complete local displacement polynomial and its physical
gradient are evaluated at the display vertices, including the variable
constitutive tensor in the stress. Coincident points remain separate between
fine cells; color interpolation is confined to each display triangle. These
samples are distinct from the quadrature used for the reported volume L2 norms.
Exact and numerical panels share color scales; the final column shows the norm
of the vector/tensor difference.

The component panels use symmetric signed scales for the displacement and
Cauchy stress, with an independent symmetric scale for each difference.

![P2 displacement and stress components](../figures/elasticity3d/components-p2.svg)

![P3 displacement and stress components](../figures/elasticity3d/components-p3.svg)

## Finite Lamé sweep

The same solenoidal displacement is used with isotropic stiffness, \(\mu=1\),
and \(\lambda=1,10,10^2,10^3,10^4\), on the fixed \(n=2\) macro grid. The exact
stress and force are independent of \(\lambda\). This avoids a parameter-dependent
forcing amplitude and permits a direct comparison of the observed errors.

![Finite Lamé study for the primal 3D formulation](../figures/elasticity3d/lame-sweep.svg)

Errors remain bounded for these two discretizations and this specific
solenoidal field. This experiment does not establish parameter-uniform
stability, eliminate locking for general loads, or extend the solver to
\(\lambda=\infty\). The [mixed stress](mixed-elasticity3d.md) and
[displacement-pressure](https://github.com/volpatto/pymhm/blob/main/docs/cases/gals3d.md) formulations provide separate
three-dimensional incompressibility contracts.

The original acquisition driver is `examples/solve_elasticity3d.py`; numerical
records and field hashes are stored in `examples/results/elasticity3d/`.
`examples/plot_elasticity3d.py` replays the six figures and records the archive digests and display conventions
in `docs/figures/elasticity3d/field-sampling.json`. Notebook
`35_elasticity3d.ipynb` presents the archived results without rerunning the
numerical campaign.
