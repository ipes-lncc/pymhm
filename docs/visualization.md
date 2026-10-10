# Visualizing and exporting reconstructed fields

Install `pymhm[visualization]`, or use the locked `visualization` and `notebooks`
Pixi profiles. `pymhm.postprocessing.visualization` loads PyVista only when a conversion or
plot is requested. Importing the numerical package requires neither VTK nor a
windowing system.

The converters return ordinary PyVista `UnstructuredGrid`, `ImageData` and
`PolyData` objects. Their standard slicing, glyph, camera and export operations
remain available. The package does not merge coincident points, average cell
fields into nodal fields, or rescale physical coordinates.

## Broken finite-element fields

A reconstructed MHM field has independent values on opposite sides of a
macroface. Keep these values separate during conversion. For a scalar P1 field
registered with the bound problem API, gather its local mesh-aware views:

```python
pressure = solution.field("pressure")
local_meshes = tuple(field.mesh for field in pressure)
local_values = tuple(field.portable_coefficients for field in pressure)
```

`portable_coefficients` uses each view's executed nodal ordering. The selected
macro mesh is `macro`, as declared when binding the problem. The following
converter accepts triangular local meshes; use the corresponding data/grid
adapter for another geometry:


```python
from pymhm.postprocessing.visualization import broken_triangle_grid, plot_field

grid = broken_triangle_grid(
    local_meshes,
    local_values,
    degree=1,
    macro_mesh=macro,
    name="pressure",
    subdivision=4,
)
grid.save("pressure.vtu")
plotter = plot_field(grid, "pressure", off_screen=True)
try:
    plotter.show(screenshot="pressure.png", auto_close=False)
finally:
    plotter.close()
```

Every fine triangle receives its own display vertices. P1–P4 arrays use the
package's `nodal_space` ordering; P0 arrays contain one value per fine triangle.
Scalar arrays and two- or three-component vectors are accepted. Planar vectors
are padded with a zero z component so VTK glyph and vector operations have
three spatial components.

Polynomial values are evaluated at the display points. Straight display
triangles approximate higher-order fields between these points; this display
sampling is not a quadrature error norm. `subdivision` controls that display
resolution and does not change the numerical solution.

Supplying `macro_mesh` embeds its geometry in the returned grid. `plot_field`
then overlays the actual macrofaces automatically. Its line layer is independent
of the field's fine triangulation. `macro_edges(mesh)` exposes the same geometry
for custom subplot layouts. Overlays on planar slices use the slice's physical
z coordinate.

For an existing triangular mesh with explicitly associated data, use
`triangle_grid(mesh, point_data={...}, cell_data={...})`. A cell field stays
cell-centered, so material jumps remain distinct. Both converters copy geometry
and values, and validate field shapes and finite real entries.

## Structured reservoir arrays

Geometry is mandatory and explicit; a final array axis of length three is
never used to guess spatial dimension:

```python
from pymhm.postprocessing.visualization import structured_cell_grid

grid = structured_cell_grid(
    (nx, ny, nz),
    cell_data={
        "Kx": permeability[..., 0],
        "Ky": permeability[..., 1],
        "Kz": permeability[..., 2],
        "porosity": porosity,
    },
    spacing=(dx, dy, dz),
    origin=(x0, y0, z0),
)
grid.save("reservoir.vti")
section = grid.slice(normal="z", origin=(x0, y0, selected_depth))
```

A scalar has exactly `cell_shape`; a vector appends one component axis of length
two or three. Thus `(nx, ny, 3)` is a three-dimensional scalar when
`cell_shape=(nx, ny, 3)`, and a planar vector when `cell_shape=(nx, ny)`.
Separate names for diagonal permeability components avoid confusing a material
tensor's diagonal with a physical velocity vector.

The array's x/I index varies fastest in VTK cell storage. This is Fortran
flattening of arrays indexed `(x, y, z)`, following PyVista's
[uniform-grid data convention](https://docs.pyvista.org/examples/00-load/create_uniform_grid.html).
The geometry uses `nx+1`, `ny+1`, `nz+1` points for that many volume cells. A 2D
`cell_shape=(nx, ny)` creates an XY plane with one z point, not an artificial
layer of 3D cells. Positive spacing describes physical cell widths, and origin
is the lower cell corner. No unit conversion or axis reversal occurs.

## Rendering and resource ownership

`plot_field` returns an open native PyVista Plotter for further composition.
The caller closes it after showing or exporting the scene. A failed setup closes
the partial plotter before propagating the original exception. Planar fields use
an XY camera; volume fields use an isometric camera. Color mapping defaults to
cell data if the same field name occurs in both associations; pass
`preference="point"` to select nodal values explicitly.

```python
plotter = plot_field(grid, "Kx", off_screen=True, log_scale=True)
try:
    image = plotter.screenshot("permeability.png")
finally:
    plotter.close()
```

A logarithmic color scale requires physically appropriate positive values.
Signed errors and vector components generally require a linear or explicitly
chosen signed scale. Vector colors use PyVista's magnitude convention unless
a component is requested through its plotting options.

For a desktop window, use `off_screen=False` and `plotter.show()`. Notebook
static output can use `plotter.show(jupyter_backend="static")`; interactive web
backends require PyVista's additional notebook dependencies. See its
[installation and headless rendering guidance](https://docs.pyvista.org/getting-started/installation).

Headless rendering requires a VTK build and system OpenGL libraries with an
appropriate EGL, OSMesa or display backend. `off_screen=True` requests offscreen
rendering; it does not select a particular graphics device. Configure the
rendering environment before launch. PyMHM does not start display servers or
install system libraries at import time.

## Preserve scientific conventions

Overlay the actual macro mesh on analytical, numerical and error panels.
Keep independent one-sided interface values in profiles and mark macroface
intersections. Use matching field color limits when comparing solutions, and
reserve space for each panel's axes, title and color scale. Display sampling
and interpolation are for visualization; calculate field errors with the
stated physical quadrature instead of comparing image pixels.

Label Darcy quantities as flux or flux magnitude, and vector flow quantities
as velocity or velocity magnitude. Distinguish raw pressure gradients from
recovered H(div) fluxes. The [field recovery tutorials](tutorials/index.md)
explain their mathematical meaning.
