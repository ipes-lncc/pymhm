# Cartesian meshes and local spaces

`CartesianMacroMesh` represents axis-aligned rectangles with shared faces and
canonical orientations. Use it to configure global geometry, local refinement
and face spaces independently. The [method tutorials](tutorials/overview.md)
then define the local and global variational equations on those choices.

## Select the global and local geometry

```python
from pymhm.meshes.cartesian import CartesianMacroMesh

macro = CartesianMacroMesh(4, 4, bounds=(0.0, 1.0, 0.0, 1.0))
fine = macro.submesh(0, (8, 8))
```

The first pair specifies macrocell counts. `bounds` contains physical
`(xmin, xmax, ymin, ymax)`. Local subdivisions specify fine rectangles inside
a selected macrocell; they do not change the global skeleton size.

For a bound hierarchy, supply an importable local mesh factory:

```python
from functools import partial
from pymhm import MeshHierarchy

hierarchy = MeshHierarchy(macro, partial(macro.submesh, subdivisions=(8, 8)))
```

The portable geometry is created in the requesting worker. No native FEM object
crosses a process boundary. When using another local mesh constructor, keep its
macroface geometry consistent with the selected interface adapter.

## Configure face degree, partition and continuity

```python
from pymhm import bind_interface
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace

faces = tuple(FaceSpace.uniform(1, 4, continuous=True) for _ in macro.faces)
skeleton = SkeletonSpace(macro, faces)
interface = bind_interface(skeleton, convention="normal")
```

This gives a continuous P1 basis on four segments of each complete macroface.
Continuity within a face does not impose continuity between unrelated faces.
The `normal` convention declares a physical normal multiplier and supplies
canonical-to-outward incidence signs. Use the value convention only when it
matches the formulation you write.

The built-in Cartesian numerical coupling integrates over the union of local
fine-edge and face-space breakpoints. These partitions need not coincide,
but they must form a stable discrete pair. An excessively enriched skeleton can
have invisible local modes; a small algebraic residual is not evidence of
stability. Native UFL trace integration follows its supported facet-alignment
contract in the [FEniCSx guide](fenics.md).

## Choose the volume element and material integration

A scalar `local.native_space(degree=k)` on a quadrilateral mesh provides the
supported equispaced tensor-product Lagrange Qk pressure space. State the retained
modes and physical moments in your provider; changing the cell family does not
supply a PDE or select a pressure gauge.

Numerical Qk operators use tensor-product Gauss quadrature. A discontinuous
`CartesianCellField` coefficient requires integration across its actual pixel
interfaces. The shared Cartesian numerical owner splits integration rectangles
at those interfaces; an aligned grid uses its uniform path. UFL forms instead
need a corresponding native integration mesh or explicit subdomain integration.
See [heterogeneous coefficients](guides/heterogeneous-darcy.md).

## Evaluate with the executed coefficient order

For a bound nodal field, register it in the provider and access its local views
through `solution.field(name)`. Each view retains the executed basis and returns
its independent one-sided values. Numerical Qk evaluators follow their own
recorded nodal order; do not reuse native coefficient vectors without the
appropriate mapping.

Darcy flux is $-K\nabla p$. A raw Qk gradient field is generally not H(div)
conforming and does not imply fine-cell conservation. The integrated skeleton
flux defines macro conservation. Use the [recovery tutorials](tutorials/index.md)
when a conforming recovered flux is required.

Cartesian geometry is axis aligned. Warped or curved quadrilateral cells need
a different geometry/element adapter; this constructor does not silently
approximate them as rectangles.
