# Defining heterogeneous coefficients

A coefficient describes the material in physical coordinates. It is independent
of the execution policy and the macro mesh. Supply the same coefficient, units
and integration convention to every local provider and reference discretization.
This page changes only that input; the [method tutorials](../tutorials/overview.md)
explain how to construct the complete local and global equations.

## Define a smooth coefficient with UFL

Inside a provider that has created `binding = local.native_space(degree=1)`,
define a scalar isotropic permeability directly on its native mesh:

```python
import numpy as np
import ufl

x, y = ufl.SpatialCoordinate(binding.mesh)
period = 0.25
K = ufl.exp(ufl.sin(2 * np.pi * x / period) * ufl.sin(2 * np.pi * y / period))
p, v = ufl.TrialFunction(binding.space), ufl.TestFunction(binding.space)
dx = ufl.Measure("dx", domain=binding.mesh, metadata={"quadrature_degree": 6})
a = K * ufl.inner(ufl.grad(p), ufl.grad(v)) * dx
```

Here `period` is a spatial length in the mesh's coordinate units. The coefficient
is positive with contrast $e^2$. Increase quadrature and local resolution when
its variation is not adequately resolved; a fixed quadrature degree is not a
resolution guarantee.

For tensor permeability, write the tensor and contraction explicitly:

```python
K_tensor = ufl.as_matrix(((K, 0), (0, 2 * K)))
a = ufl.inner(K_tensor * ufl.grad(p), ufl.grad(v)) * dx
```

The tensor acts on the physical gradient. Its symmetry and positive definiteness
are physical input requirements for elliptic Darcy diffusion.

## Define a sampled material grid

`CartesianCellField` represents scalar or tensor values that are constant inside
material pixels. Its first axes are spatial: `(x, y)` or `(x, y, z)`.

```python
from pymhm.materials.cartesian import CartesianCellField

values = np.array([[1.0, 10.0], [100.0, 1000.0]])
material = CartesianCellField(values, spacing=(0.5, 0.5), origin=(0.0, 0.0))
samples = material(np.array([[0.25, 0.25], [0.75, 0.25]]))
np.testing.assert_array_equal(samples, [1.0, 100.0])
```

`spacing` contains physical pixel widths, and `origin` is the lower grid corner.
An internal grid line takes the positive-side value; the upper external boundary
takes the last interior cell. This convention preserves jumps without averaging.
It does not integrate them: assembly must resolve the actual material interfaces.

A tensor array has shape `(nx, ny, 2, 2)` in 2D or `(nx, ny, nz, 3, 3)` in 3D.
A final axis of length three alone does not imply a three-dimensional mesh or a
vector-valued PDE field.

## Bring a cellwise field into UFL

When each native fine cell lies inside one material pixel, interpolate into a
piecewise-constant coefficient space:

```python
from dolfinx import fem

Q = fem.functionspace(binding.mesh, ("DG", 0))
k = fem.Function(Q)
k.interpolate(lambda x: material(x[:2].T))
a = k * ufl.inner(ufl.grad(p), ufl.grad(v)) * dx
```

If a fine cell crosses a material jump, this interpolation assigns one sampled
value to the whole cell and therefore changes the coefficient. Instead align
its integration mesh with the material grid or use an adapter that integrates
the pixel intersections. PyMHM's Cartesian numerical quadrature splits
`CartesianCellField` integration at pixel boundaries; arbitrary UFL forms need
their own corresponding integration construction.

## Use region labels instead of pixels

For material regions described by integer cell tags, define a mapping from IDs
to coefficients and preserve those IDs during mesh refinement. Follow
[materials and marked meshes](materials.md). A tag does not automatically assign
a coefficient, and material interfaces inside a macrocell must remain represented
in its local integration.

The [SPE10 application](../gallery/applications/spe10.md) demonstrates a measured
reservoir permeability field. Its data, geometry, boundary conditions and
reference solutions belong to that application; these coefficient interfaces
also apply to user-defined materials.
