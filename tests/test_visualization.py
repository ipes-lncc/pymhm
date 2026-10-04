"""Portable data contracts and real PyVista/VTK conversion and rendering tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import TriangleMesh
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.postprocessing import visualization as visual


class ContractGrid:
    """Record constructor inputs without claiming to emulate VTK rendering."""

    def __init__(self, cells=None, celltypes=None, points=None, **kwargs):
        """Retain only the data association contract used by these unit checks."""
        self.cells = cells
        self.celltypes = celltypes
        self.points = points
        self.point_data, self.cell_data, self.field_data = {}, {}, {}
        self.__dict__.update(kwargs)

    @property
    def bounds(self):
        """Return axis bounds for the renderer's camera contract."""
        return tuple(np.array([self.points.min(axis=0), self.points.max(axis=0)]).T.ravel())


class ContractPlotter:
    """Track composition and cleanup calls, without producing an image."""

    instances = []

    def __init__(self, **kwargs):
        """Record constructor options and an initially open resource."""
        self.options, self.meshes, self.closed = kwargs, [], False
        self.instances.append(self)

    def add_mesh(self, mesh, **kwargs):
        """Record one explicitly associated mesh layer."""
        self.meshes.append((mesh, kwargs))
        if kwargs.get("raise_test_error"):
            raise RuntimeError("rendering failed")

    def view_xy(self):
        """Record an XY camera request."""
        self.camera = "xy"

    def view_isometric(self):
        """Record an isometric camera request."""
        self.camera = "iso"

    def close(self):
        """Record explicit resource cleanup."""
        self.closed = True


@pytest.fixture(params=["contract", pytest.param("native", marks=pytest.mark.visualization)])
def backend(request, monkeypatch):
    """Run the same array contracts against a recorder and against native VTK."""
    if request.param == "native":
        return pytest.importorskip("pyvista")
    backend = SimpleNamespace(
        UnstructuredGrid=ContractGrid,
        ImageData=ContractGrid,
        PolyData=lambda points, **kwargs: ContractGrid(points=points, **kwargs),
        CellType=SimpleNamespace(TRIANGLE=5),
        Plotter=ContractPlotter,
    )
    monkeypatch.setattr(visual, "import_module", lambda name: backend)
    return backend


def test_triangle_scalar_vector_cell_fields_are_copied(backend):
    """Retain association and pad planar vectors without averaging cell jumps."""
    mesh = TriangleMesh.unit_square()
    pressure = np.arange(len(mesh.points), dtype=float)
    vector = np.column_stack((pressure, -pressure))
    grid = visual.triangle_grid(
        mesh, point_data={"p": pressure, "u": vector}, cell_data={"K": [1, 100]}
    )
    pressure[:] = -7
    vector[:] = 9
    assert_array_equal(grid.point_data["p"], np.arange(4))
    assert_array_equal(grid.point_data["u"][:, 2], 0)
    assert_array_equal(grid.cell_data["K"], [1, 100])
    assert_array_equal(grid.points[:, :2], mesh.points)
    assert_array_equal(grid.celltypes, 5)
    empty = visual.triangle_grid(mesh)
    assert not empty.point_data and not empty.cell_data


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("vector", [False, True])
def test_broken_polynomials_and_macro_jumps_are_preserved(backend, degree, vector):
    """Recover polynomial samples and both sides of a discontinuous macro interface."""
    macro = TriangleMesh.unit_square()
    meshes = [macro.submesh(cell, 2) for cell in range(2)]
    fields = []
    for cell, mesh in enumerate(meshes):
        nodes = (
            mesh.points[mesh.cells].mean(axis=1) if degree == 0 else nodal_space(mesh, degree)[1]
        )
        values = (
            np.full(len(nodes), cell + 1.0) if degree == 0 else cell + 1 + nodes[:, 0] ** degree
        )
        fields.append(np.column_stack((values, -values)) if vector else values)
    grid = visual.broken_triangle_grid(meshes, fields, degree, macro_mesh=macro, subdivision=3)
    x = grid.points[:, 0]
    data = grid.point_data["field"]
    data = data[:, 0] if vector else data
    count = len(x) // 2
    expected = np.repeat([1.0, 2.0], count) + (x**degree if degree else 0)
    assert_allclose(data, expected, atol=3e-13)
    assert len(x) > len(np.unique(grid.points, axis=0))
    assert_array_equal(grid.field_data["pymhm:macro_cells"], macro.cells)
    assert_array_equal(np.unique(grid.cell_data["pymhm:macro_cell"]), [0, 1])
    assert_array_equal(np.unique(grid.cell_data["pymhm:fine_cell"]), np.arange(4))
    if vector:
        assert_allclose(grid.point_data["field"][:, 1], -expected, atol=3e-13)
        assert_array_equal(grid.point_data["field"][:, 2], 0)


def test_default_subdivision_and_no_macro_metadata(backend):
    """Support an explicitly broken local mesh without requiring a macro parent."""
    mesh = TriangleMesh.unit_square()
    grid = visual.broken_triangle_grid([mesh], [np.ones((2, 3))], degree=0)
    assert not grid.field_data
    assert grid.point_data["field"].shape == (6, 3)
    grid = visual.broken_triangle_grid([mesh], [np.ones(4)])
    assert len(grid.point_data["field"]) == 12


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
@pytest.mark.parametrize("components", [0, 2, 3])
def test_structured_shape_is_explicit_and_x_index_is_fastest(backend, shape, components):
    """Disambiguate a third spatial axis from a vector component axis."""
    base = np.arange(np.prod(shape)).reshape(shape, order="F")
    values = (
        base if components == 0 else np.stack([base + 100 * i for i in range(components)], axis=-1)
    )
    grid = visual.structured_cell_grid(
        shape, cell_data={"rock": values}, spacing=(20, 10, 2), origin=(3, 4, 5)
    )
    assert tuple(grid.dimensions) == tuple(n + 1 for n in shape) + ((1,) if len(shape) == 2 else ())
    assert tuple(grid.spacing) == (20, 10, 2)
    assert tuple(grid.origin) == (3, 4, 5)
    assert not grid.point_data
    stored = grid.cell_data["rock"]
    if components:
        assert_array_equal(stored[:, 0], np.arange(base.size))
        assert_array_equal(stored[:, 1], np.arange(base.size) + 100)
        assert_array_equal(stored[:, 2], 0 if components == 2 else np.arange(base.size) + 200)
    else:
        assert_array_equal(stored, np.arange(base.size))
    values[:] = -1
    assert np.min(stored) >= 0


def test_planar_origin_spacing_and_multiple_cell_fields(backend):
    """Accept planar geometry pairs while keeping distinct cell fields."""
    grid = visual.structured_cell_grid(
        (2, 3),
        cell_data={"K": np.ones((2, 3)), "phi": np.full((2, 3), 0.2)},
        spacing=(2, 4),
        origin=(-1, 3),
    )
    assert tuple(grid.spacing) == (2, 4, 1)
    assert tuple(grid.origin) == (-1, 3, 0)
    assert set(grid.cell_data) == {"K", "phi"}
    edges = visual.macro_edges(TriangleMesh.unit_square())
    assert_array_equal(np.asarray(edges.lines).reshape(-1, 3)[:, 0], 2)
    assert len(np.asarray(edges.lines).reshape(-1, 3)) == 5


@pytest.mark.parametrize(
    "field",
    [
        np.array([1, 2]),
        np.ones((4, 1)),
        np.ones((4, 4)),
        1.0,
        ["x"] * 4,
        [1j] * 4,
        [np.nan] * 4,
        [np.inf] * 4,
    ],
)
def test_invalid_fields_are_rejected_before_optional_import(field):
    """Do not cast away complex values, accept bad shapes or plot nonfinite data."""
    with pytest.raises(ValueError):
        visual.triangle_grid(TriangleMesh.unit_square(), point_data={"p": field})


@pytest.mark.parametrize("name", ["", "pymhm:macro_points", 7])
def test_reserved_or_invalid_names_are_rejected(name):
    """Protect embedded geometry metadata from colliding field names."""
    with pytest.raises(ValueError, match="field names"):
        visual.triangle_grid(TriangleMesh.unit_square(), point_data={name: np.ones(4)})


@pytest.mark.parametrize(
    "kwargs", [{"degree": 5}, {"degree": -1}, {"degree": 1.2}, {"subdivision": 0}]
)
def test_invalid_broken_discretization(kwargs):
    """Reject unsupported polynomial orders and invalid display partitions."""
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError):
        visual.broken_triangle_grid([mesh], [np.ones(4)], **kwargs)


def test_broken_sequence_and_component_mismatches():
    """Require one compatible scalar/vector field per requested local mesh."""
    mesh = TriangleMesh.unit_square()
    for meshes, fields in [([], []), ([mesh], [])]:
        with pytest.raises(ValueError, match="nonempty"):
            visual.broken_triangle_grid(meshes, fields)
    with pytest.raises(ValueError, match="macrocell"):
        visual.broken_triangle_grid([mesh], [np.ones(4)], macro_mesh=mesh)
    with pytest.raises(ValueError, match="components"):
        visual.broken_triangle_grid([mesh, mesh], [np.ones(4), np.ones((4, 2))])


@pytest.mark.parametrize("shape", [(2,), (1, 2, 3, 4), (0, 2), (2.5, 3), (True, 2)])
def test_invalid_structured_geometry_shape(shape):
    """Never infer spatial dimension or truncate noninteger cell counts."""
    with pytest.raises(ValueError):
        visual.structured_cell_grid(shape, cell_data={})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spacing": (0, 1, 1)},
        {"spacing": (-1, 1, 1)},
        {"spacing": (1, 2)},
        {"origin": (1, 2)},
        {"origin": (np.nan, 0, 0)},
        {"spacing": (1 + 1j, 1, 1)},
    ],
)
def test_invalid_structured_coordinates(kwargs):
    """Reject nonphysical spacing and incomplete three-dimensional geometry."""
    with pytest.raises(ValueError):
        visual.structured_cell_grid((2, 3, 4), cell_data={}, **kwargs)


@pytest.mark.parametrize("error", [ImportError("missing"), OSError("shared library")])
def test_optional_dependency_failure_has_installation_hint(monkeypatch, error):
    """Preserve native import causes and name the optional installation extra."""

    def unavailable(name):
        """Simulate only the dependency-loader failure path."""
        raise error

    monkeypatch.setattr(visual, "import_module", unavailable)
    with pytest.raises(ImportError, match=r"pymhm\[visualization\]") as exc:
        visual.macro_edges(TriangleMesh.unit_square())
    assert exc.value.__cause__ is error


def test_plotter_contract_overlays_cameras_and_cleanup(monkeypatch):
    """Check composition and exception cleanup without treating a recorder as rendering."""
    backend = SimpleNamespace(
        Plotter=ContractPlotter, PolyData=lambda points, **kw: ContractGrid(points=points, **kw)
    )
    monkeypatch.setattr(visual, "import_module", lambda name: backend)
    mesh = TriangleMesh.unit_square()
    grid = ContractGrid(points=np.column_stack((mesh.points, np.zeros(4))))
    grid.point_data["p"] = np.ones(4)
    grid.points[:, 2] = 4
    grid.field_data.update({"pymhm:macro_points": mesh.points, "pymhm:macro_cells": mesh.cells})
    plotter = visual.plot_field(grid, "p", off_screen=True, cmap="viridis")
    assert plotter.camera == "xy" and len(plotter.meshes) == 2
    assert plotter.meshes[0][1]["smooth_shading"] is False
    assert plotter.meshes[0][1]["cmap"] == "viridis"
    assert_array_equal(plotter.meshes[1][0].points[:, 2], 4)
    plotter.close()
    grid.field_data.clear()
    grid.points[-1, 2] = 1
    grid.cell_data["cell"] = np.ones(2)
    plotter = visual.plot_field(grid, "cell", macro_mesh=mesh)
    assert plotter.camera == "iso" and len(plotter.meshes) == 2
    assert_array_equal(plotter.meshes[1][0].points[:, 2], 0)
    plotter.close()
    plotter = visual.plot_field(grid, "cell")
    assert len(plotter.meshes) == 1
    plotter.close()
    with pytest.raises(ValueError, match="not present"):
        visual.plot_field(grid, "absent")
    with pytest.raises(RuntimeError, match="rendering failed"):
        visual.plot_field(grid, "p", macro_mesh=mesh, raise_test_error=True)
    assert ContractPlotter.instances[-1].closed


@pytest.mark.visualization
@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
def test_native_structured_cell_centers_and_vtk_export(tmp_path, shape):
    """Verify physical I/J/K ordering and lossless native VTK round trips."""
    pv = pytest.importorskip("pyvista")
    values = np.einsum("i,i...->...", 10 ** np.arange(len(shape)), np.indices(shape))
    grid = visual.structured_cell_grid(
        shape, cell_data={"K": values}, spacing=(20, 10, 2), origin=(3, 4, 5)
    )
    ijk = np.indices(shape).reshape(len(shape), -1, order="F").T
    centers = np.array([3, 4, 5])[: len(shape)] + (ijk + 0.5) * np.array([20, 10, 2])[: len(shape)]
    if len(shape) == 2:
        centers = np.column_stack((centers, np.full(len(centers), 5)))
    assert_allclose(grid.cell_centers().points, centers)
    filename = tmp_path / "reservoir.vti"
    grid.save(filename)
    loaded = pv.read(filename)
    assert_array_equal(loaded.cell_data["K"], values.ravel(order="F"))
    assert loaded.n_cells == np.prod(shape)
    mesh = TriangleMesh.unit_square()
    broken = visual.broken_triangle_grid([mesh], [np.arange(4.0)], name="pressure")
    filename = tmp_path / "broken.vtu"
    broken.save(filename)
    loaded = pv.read(filename)
    assert_array_equal(loaded.point_data["pressure"], broken.point_data["pressure"])
    assert_array_equal(loaded.cell_data["pymhm:macro_cell"], broken.cell_data["pymhm:macro_cell"])


@pytest.mark.visualization
def test_native_offscreen_rendering_preserves_input_and_closes(tmp_path):
    """Produce an actual nonblank VTK image, with macro geometry kept separate."""
    pytest.importorskip("pyvista")
    mesh = TriangleMesh.unit_square()
    local = [mesh.submesh(i, 2) for i in range(2)]
    fields = [np.full(len(part.points), i, dtype=float) for i, part in enumerate(local)]
    grid = visual.broken_triangle_grid(local, fields, macro_mesh=mesh)
    before = grid.point_data["field"].copy()
    plotter = visual.plot_field(grid, "field", off_screen=True, window_size=(320, 240))
    try:
        output = tmp_path / "broken.png"
        plotter.show(screenshot=output, auto_close=False)
        assert plotter.image.shape == (240, 320, 3)
        assert np.ptp(plotter.image) > 100
        assert len(np.unique(plotter.image.reshape(-1, 3), axis=0)) > 20
        assert output.stat().st_size > 1000
        assert_array_equal(grid.point_data["field"], before)
    finally:
        plotter.close()


@pytest.mark.visualization
def test_native_reservoir_slice_and_three_dimensional_view(tmp_path):
    """Render a physical 3D reservoir block and preserve cell data on a slice."""
    pytest.importorskip("pyvista")
    grid = visual.structured_cell_grid(
        (2, 3, 4),
        cell_data={"Kx": np.arange(24.0).reshape(2, 3, 4, order="F")},
        spacing=(20, 10, 2),
    )
    section = grid.slice(normal="z", origin=(20, 15, 3))
    assert set(section.cell_data) == {"Kx"}
    assert_array_equal(np.sort(section.cell_data["Kx"]), np.arange(6, 12))
    plotter = visual.plot_field(grid, "Kx", off_screen=True, window_size=(320, 240))
    try:
        image = plotter.screenshot(tmp_path / "reservoir.png")
        assert image.shape == (240, 320, 3)
        assert np.ptp(image) > 100
    finally:
        plotter.close()
