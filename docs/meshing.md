# Mesh generation and exchange

The package has separate geometry contracts for triangles, quadrilaterals,
polygons, tetrahedra, affine prisms, trilinear hexahedra and star-shaped polyhedra.
The two-dimensional adapters below use triangles in the XY plane; the
[volume adapters](#three-dimensional-generation-and-exchange) retain supported
three-dimensional cell families. Gmsh and Netgen generate external meshes;
meshio imports and exports them. Optional dependencies load only when used.

```bash
pip install '.[meshing]'
# Reproducible development environment:
pixi run -e meshing test-meshing
```

## Generate a mesh

```python
from pymhm.meshing import unit_square_gmsh, unit_square_netgen

gmsh_data = unit_square_gmsh(size=0.2)
netgen_data = unit_square_netgen(maxh=0.2)

mesh = gmsh_data.mesh
print(mesh.areas.sum())  # approximately 1
print(gmsh_data.physical_names)
```

Both helpers return `MeshData`. Its `mesh` is the same `TriangleMesh` accepted by
`SkeletonSpace` and the portable finite element builders. The helpers mark the
four sides and the material region. Mesh size is a generator request rather than
an exact edge-length constraint. The resulting triangulations can differ across
generator versions and are not required to match each other.

`unit_square_gmsh` creates and removes a temporary model. If the caller already
initialized Gmsh, the session remains active and its previous current model is
restored. Otherwise the helper owns initialization and finalization. Gmsh has
process-global state: concurrent threads must not call this helper or manipulate
the same Gmsh session.

## Use a custom geometry

Build any supported geometry with the generator's native interface, then convert
the resulting mesh:

```python
import gmsh
from pymhm.meshing import from_gmsh

gmsh.initialize()
try:
    gmsh.open("domain.geo")
    gmsh.model.mesh.generate(2)
    data = from_gmsh()
finally:
    gmsh.finalize()
```

`from_gmsh(model)` also accepts an explicit Gmsh model object. The converter reads
the existing mesh; it does not generate, renumber, or modify it. Sparse Gmsh node
tags are remapped to the contiguous indices of `TriangleMesh`.

```python
from netgen.geom2d import SplineGeometry
from pymhm.meshing import from_netgen

geometry = SplineGeometry()
geometry.AddRectangle((0, 0), (2, 1), bcs=["bottom", "right", "top", "left"])
data = from_netgen(geometry.GenerateMesh(maxh=0.2))
```

Consult the [Gmsh API manual](https://gmsh.info/doc/texinfo/) and the
[Netgen geometry tutorial](https://docu.ngsolve.org/v6.2.2501/netgen_tutorials/define_2d_geometries.html)
for geometry construction and generator-specific refinement controls.

## Physical groups and fields

`MeshData` stores the following information:

| Attribute | Meaning |
| --- | --- |
| `mesh` | Triangle geometry, topology, and consistent face orientations |
| `face_tags` | Mapping from physical curve ID to mesh face indices, including interior interfaces |
| `boundary_tags` | The same groups restricted to external boundary faces |
| `cell_tags` | One nonnegative material ID per triangle; zero means unmarked; `None` means absent |
| `physical_names` | Names keyed by `(dimension, physical_id)` |
| `point_data` | Numeric scalar/vector arrays with one entry per vertex |
| `cell_data` | Numeric scalar/vector arrays with one entry per triangle |

Physical IDs are labels, not array offsets. Dimensions distinguish a curve group
and a material group that share the same ID. The trace's physical sign follows
`mesh.normals` and `mesh.signs`, independently of the input line ordering.

The converter preserves straight interior material interfaces when they coincide
with triangle faces. A line that does not coincide with any triangle face raises
an error. Multiple physical curve groups can share a face in memory, but export
rejects them because a single scalar physical ID cannot represent that overlap.
Overlapping physical surface groups are rejected during Gmsh conversion.

## Import and export

```python
from pymhm.meshing import read_mesh, write_mesh

data = read_mesh("domain.msh")
write_mesh("domain.vtu", data)
write_mesh(
    "diagnostics.vtu",
    data,
    point_data={"x_coordinate": data.mesh.points[:, 0]},
    cell_data={"cell_area": data.mesh.areas},
)
restored = read_mesh("diagnostics.vtu")
```

VTU exports preserve physical IDs, physical names, and numeric point/triangle
fields. Gmsh 2.2 preserves groups, names, and point fields. Its fields must have
one, three, or nine components. With meshio 5.3, Gmsh 2.2 cell fields
are unsupported for meshes containing both triangle and line blocks because
the importer splits those blocks inconsistently. Export rejects that combination;
use VTU when exporting such cell fields. Cell fields remain available for a
triangles-only Gmsh mesh. `.msh` output defaults to Gmsh 2.2 because it does not
require recreating the CAD entity hierarchy. VTU stores named-group metadata in
additional arrays understood by the importer. Other meshio formats are available
for geometry-only output; requesting metadata with such a format raises an error
instead of silently discarding it. An explicit `file_format` may be supplied to
`read_mesh` and `write_mesh`.

The `point_data` and `cell_data` keyword mappings extend stored fields and replace
fields with the same name. Triangle fields are padded with zeros on exported line
elements, then recovered on triangles at import. General fields attached only to
input line or point elements are outside this triangular field model; use meshio
directly when those fields are needed. See the
[meshio project documentation](https://github.com/nschloe/meshio) for available
file formats.

## Geometry and numerical limitations

The two-dimensional adapters accept first-order triangles and line elements. Point
elements do not contribute to the triangular topology. Quadrilaterals,
higher-order/curved elements, volume meshes, and nonzero Z coordinates are
rejected by those adapters. Three-dimensional geometry uses the separate APIs
below; none of these converters implicitly linearizes curved elements.

Imported triangles are oriented and validated by `TriangleMesh`. Physical groups
do not automatically prescribe PDE boundary conditions: select the appropriate
face groups when setting up the problem. Generator boundary segments define the
macro mesh; MHM trace subfaces and polynomial degrees are then configured
independently with `FaceSpace` and `SkeletonSpace`.

Tests exercise portable adapter contracts and real Gmsh/Netgen meshes. Native
round trips compare cell geometry, area, group membership, names, and point
fields for both `.msh` and `.vtu`; VTU also checks triangle fields. The tests
verify that an existing Gmsh model survives generation.

## Tetrahedral exchange and other macro geometries

`pymhm.meshing3d.read_tetra_mesh` and `write_tetra_mesh` separately exchange
linear tetrahedra and integer volume/boundary tags through meshio. Their
`TetraMeshData` container keeps volume cells distinct from boundary triangles.
Use the [three-dimensional Darcy](https://github.com/volpatto/pymhm/blob/main/docs/cases/darcy3d.md),
[RAD](https://github.com/volpatto/pymhm/blob/main/docs/cases/rad3d.md) and [elasticity](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity3d.md) paths with these
tetrahedral meshes.

`PolygonMesh` retains original polygonal edges, `PolyhedralMesh` retains original
polygonal faces of star-shaped cells (including nonconvex cells), and `HexMesh` supplies trilinear hexahedral
maps. These native geometry constructors have their own validation contracts;
they do not silently convert arbitrary CAD or curved high-order elements.
See [polyhedral RAD](https://github.com/volpatto/pymhm/blob/main/docs/cases/polyhedral-rad.md) and the
[mapped RT well](https://github.com/volpatto/pymhm/blob/main/docs/cases/mapped-well.md) for concrete constructions.

## Conforming adaptive triangles

`refine_triangles(mesh, marked)` provides red–green subdivision.
`refine_longest_edge(mesh, marked)` provides longest-edge bisection with
conforming propagation. Both return the refined mesh and exact topological
cell/face ancestors, which allow transfer of material tags, local resolutions
and skeletal partitions.

The longest-edge policy propagates to a boundary edge or an edge longest in
both incident cells before bisecting its neighbors. It preserves conformity
without repeatedly refining green templates. In exact arithmetic, the
two-dimensional construction bounds the resulting minimum angle by half the
initial minimum angle; floating-point geometry still undergoes the ordinary
mesh checks. This is the refinement of
[Rivara (1984)](https://doi.org/10.1002/nme.1620200412).
The red–green entry point does not assert that bound under arbitrary repeated
marking. Adaptive studies state their selected policy explicitly.

## Isotropic residual-metric remeshing

`residual_mesh_size(mesh, local_squared, coefficient=1)` constructs a continuous
P1 size field from cellwise squared indicators. It uses incident-edge length
averages for the current nodal size and an area-weighted lumped projection of
\(\eta_K\) for the nodal indicator. With
\(\eta_* = c\,\operatorname{mean}_K\eta_K\), the requested size is

$$
h_{\mathrm{new}}(x_i)=h(x_i)\big/
 \operatorname{clip}(\eta_{P1}(x_i)/\eta_*,1/3,3).
$$

Both refinement and coarsening are possible. This is the mathematical metric
in the [FreeFEM residual-adaptation documentation](https://doc.freefem.org/models/static-problems.html#adaptation-using-residual-error-indicator);
the lower factor is exactly \(1/3\), while its illustrative script uses the
rounded decimal `0.3333`.

```python
from pymhm import residual_mesh_size, remesh_freefem

metric = residual_mesh_size(mesh, local_squared, coefficient=1)
new_mesh = remesh_freefem(mesh, metric.requested, executable="FreeFem++-nw")
```

`remesh_freefem` invokes the separately installed FreeFEM/BAMG mesh generator
with a P1 physical size field, `IsMetric=1`, `splitpbedge=1` and a configurable
vertex limit. It solves no PDE. Version 4.13 from conda-forge was used for the
recorded native validation. FreeFEM is available through its
[official installation instructions](https://doc.freefem.org/introduction/installation.html)
and the conda-forge `freefem` package; it is not a Python runtime dependency.
Pass an executable path when it is outside `PATH`. The locked Linux environment
and native integration check are available with

```bash
pixi run -e remeshing pytest -q tests/test_metric_freefem.py tests/test_metric_adapt.py
```


The returned `TriangleMesh` need not be nested. The adapter preserves the
polygonal geometry and checks its area; it does not transfer physical labels,
material interfaces, solution coefficients or cell ancestry. Reapply boundary
and material data as physical functions on the new mesh. This differs from
`refine_longest_edge`, which provides exact parent maps, and from
`refine_darcy_budget`, which enlarges an indicator-ranked bulk set to meet an
explicit complexity target.

## Three-dimensional generation and exchange

`pymhm.mesh_exchange` provides `VolumeMeshData`, `from_meshio`, `to_meshio`,
`read_volume_mesh` and `write_volume_mesh`. A volume mesh contains one supported
family; a mixture of tetrahedra and prisms is not implicitly split into unrelated
problems. The finite-element degree is independent of the geometric cell order.

| meshio cell | Native geometry | Connectivity and geometric requirements |
| --- | --- | --- |
| `tetra` | `TetraMesh` | Four corners; positive tetrahedral volume |
| `wedge` | `AffineMixedMesh(kind="prism")` | Two corresponding triangular triples; affine prism |
| `hexahedron` | `HexMesh` | Eight VTK-order corners are permuted into native lexicographic `(x,y,z)` order; valid trilinear map |
| `polyhedronN` | `PolyhedralMesh` | Cyclic simple planar polygonal faces and certified star-shaped cells; concave original macrofaces remain intact |

Wedge connectivity follows meshio's convention. Its VTK reader already performs
VTK's wedge permutation, so the adapter does not apply that permutation twice.
For hexahedra the native-to-meshio corner permutation is
`[0,4,6,2,1,5,7,3]`; the inverse is used at import. These are geometry conventions,
not transformations of finite-element coefficient vectors. See meshio's
[VTK connectivity implementation](https://github.com/nschloe/meshio/blob/main/src/meshio/_vtk_common.py).

`VolumeMeshData` uses one nonnegative integer `cell_tags` value per cell and one
`face_tags` value per geometric face, including interior material interfaces.
Zero denotes an unmarked entity. Scalar or vector `point_data` and `cell_data`
retain their association with vertices and cells. Physical-name strings and
multiple overlapping nonzero IDs on the same face are outside this contract.
Tags do not prescribe a PDE boundary condition automatically.

```python
from pymhm.hdiv3d_mesh import AffineMixedMesh
from pymhm.mesh_exchange import VolumeMeshData, read_volume_mesh, write_volume_mesh

mesh = AffineMixedMesh.unit_cube(kind="prism")
data = VolumeMeshData(mesh, cell_data={"volume": mesh.volumes})
write_volume_mesh("prisms.vtu", data)
restored = read_volume_mesh("prisms.vtu")
assert restored.mesh.kind == "prism"
```

Lossless volume export uses VTU. The integer arrays `physical_tag` and
`pymhm_face_tags` encode cell IDs and local face IDs. The latter is padded with
zeros when polyhedra have different face counts. Polyhedron blocks are emitted
in increasing vertex-count order, keeping geometry and cell fields paired in
meshio 5.3. VTU input with noncanonical polyhedron block order is rejected because
that reader can associate the field blocks inconsistently. `from_meshio` accepts
any correctly paired in-memory block order. Cell order can change when cells are
grouped by type; use physical IDs or explicit cell fields to identify cells.

`pymhm.meshing_native3d` supplies optional native generators and converters:

```python
from pymhm.meshing_native3d import unit_cube_gmsh, unit_cube_netgen

gmsh_volume = unit_cube_gmsh(size=0.35)
netgen_volume = unit_cube_netgen(maxh=0.35)
assert abs(gmsh_volume.mesh.volumes.sum() - 1.0) < 1e-12
assert abs(netgen_volume.mesh.volumes.sum() - 1.0) < 1e-12
```

Both helpers generate first-order tetrahedra. Gmsh labels the sides
`x=0, x=1, y=0, y=1, z=0, z=1` with IDs 1 through 6 and the volume with ID 1;
Netgen's existing boundary/material IDs are preserved. `from_gmsh_3d(model)` also
imports a live pure wedge or hexahedral volume mesh. Sparse node tags are mapped
to contiguous indices. `from_netgen_3d(mesh)` imports tetrahedra and reads the
boundary-condition value from each Netgen face descriptor. Higher-order volume
or boundary geometry is rejected. Gmsh session ownership and restoration follow
the same contract as the two-dimensional generator; calls must be serialized
within a process.

The native tests generate both unit-cube meshes, verify their volume and six
boundary groups, and solve an affine Darcy patch on each. VTU tests exercise
actual meshio writes and reads for all four families, including adjacent
polyhedra with different vertex and face counts. These checks validate exchange
and solver interoperability; they do not establish mesh-quality bounds or
support for every CAD topology.

```bash
pixi run -e meshing pytest tests/test_mesh_exchange.py tests/test_meshing_native3d.py
```

Notebook `60_mesh_exchange.ipynb` demonstrates the four in-memory geometries and
real meshio round trips. For generator-specific CAD construction, consult the
[Gmsh API](https://gmsh.info/doc/texinfo/) and
[Netgen three-dimensional CSG tutorial](https://docu.ngsolve.org/latest/netgen_tutorials/csg_3d.html).
