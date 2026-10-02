"""Volume connectivity, physical-ID contracts and native meshio file round trips."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import mesh_exchange as exchange
from pymhm.hdiv3d_mesh import AffineMixedMesh
from pymhm.mapped_rt import HexMesh
from pymhm.polyhedral import PolyhedralMesh
from pymhm.tetrahedral import TetraMesh


def families():
    """Return each supported native volume family, including multiple shared faces."""
    return [
        TetraMesh.unit_cube(),
        AffineMixedMesh.unit_cube(kind="prism"),
        HexMesh.unit_cube(2),
        PolyhedralMesh.cubes(2),
    ]


def fake_mesh(points, cells, point_data, cell_data):
    """Store the declared meshio API arguments without performing geometry conversions."""
    return SimpleNamespace(
        points=points,
        cells=[SimpleNamespace(type=k, data=v) for k, v in cells],
        point_data=point_data,
        cell_data=cell_data,
    )


@pytest.fixture
def codec(monkeypatch):
    """Expose in-memory API contracts while native file tests exercise meshio itself."""
    state = {}

    def mesh(*args, **kwargs):
        """Record original blocks for structural import checks."""
        result = fake_mesh(*args, **kwargs)
        result.write = lambda path, **kw: state.update(source=result, path=path)
        return result

    module = SimpleNamespace(Mesh=mesh, read=lambda *a, **k: state["source"])
    monkeypatch.setattr(exchange, "_optional", lambda name: module)
    return state


@pytest.mark.parametrize("index", range(4))
def test_in_memory_roundtrip_preserves_fields_and_physical_tags(codec, index):
    """Each physical face has its own identity, including marked interior interfaces."""
    mesh = families()[index]
    data = exchange.VolumeMeshData(
        mesh,
        cell_tags=np.arange(len(mesh.cells)),
        face_tags=np.arange(len(mesh.faces)) + 1,
        point_data={"p": mesh.points[:, 0]},
        cell_data={"q": np.ones((len(mesh.cells), 3))},
    )
    source = exchange.to_meshio(data)
    result = exchange.from_meshio(source)
    assert type(result.mesh) is type(mesh)
    assert_allclose(result.mesh.points, mesh.points)
    assert_array_equal(result.cell_tags, data.cell_tags)
    expected = {
        tuple(sorted(face)): tag for face, tag in zip(mesh.faces, data.face_tags, strict=True)
    }
    for face, tag in zip(result.mesh.faces, result.face_tags, strict=True):
        assert tag == expected[tuple(sorted(face))]
    assert_allclose(result.cell_data["q"], data.cell_data["q"])
    exchange.write_volume_mesh("grid.vtu", data)
    actual = exchange.read_volume_mesh("grid.vtu")
    assert_array_equal(actual.cell_tags, data.cell_tags)
    exchange.to_meshio(mesh)


def test_explicit_hexahedron_and_wedge_connectivity(codec):
    """Standard meshio corner order maps to exact native physical reference coordinates."""
    mesh = HexMesh.unit_cube()
    source = exchange.to_meshio(mesh)
    vtk = mesh.points[source.cells[0].data[0]]
    assert_array_equal(
        vtk,
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    )
    prism = AffineMixedMesh.unit_cube(kind="prism")
    assert_array_equal(exchange.to_meshio(prism).cells[0].data, prism.cells)
    tetra = AffineMixedMesh.unit_cube(kind="tetrahedron")
    assert exchange.to_meshio(tetra).cells[0].type == "tetra"


def test_surface_blocks_and_padding(codec):
    """External physical surface groups combine with explicit per-cell face metadata."""
    mesh = TetraMesh.unit_cube()
    source = exchange.to_meshio(mesh)
    face = mesh.faces[0]
    source.cells.extend(
        [
            SimpleNamespace(type="triangle", data=[face]),
            SimpleNamespace(type="line", data=[[0, 1]]),
            SimpleNamespace(type="vertex", data=[[0]]),
        ]
    )
    source.cell_data["physical_tag"].extend(
        [np.array([7]), np.zeros(1, dtype=int), np.zeros(1, dtype=int)]
    )
    source.cell_data["pymhm_face_tags"].extend([np.zeros((1, 4), dtype=int)] * 3)
    result = exchange.from_meshio(source)
    assert result.face_tags[0] == 7
    source.cell_data.pop("pymhm_face_tags")
    assert exchange.from_meshio(source).face_tags[0] == 7
    source.cells[-1] = SimpleNamespace(type="polygon3", data=[face])
    source.cell_data["physical_tag"][-1][0] = 7
    assert exchange.from_meshio(source).face_tags[0] == 7
    source.cell_data["physical_tag"][-1][0] = 8
    with pytest.raises(ValueError, match="conflicting"):
        exchange.from_meshio(source)
    source.cells[-1] = SimpleNamespace(type="triangle", data=[[0, 1, 6]])
    with pytest.raises(ValueError, match="not a face"):
        exchange.from_meshio(source)


@pytest.mark.parametrize("kind", ["tetra10", "wedge15", "hexahedron20", "pyramid"])
def test_curved_or_unsupported_cells_are_not_silently_linearized(kind):
    """Unsupported cell types fail before any finite-element geometry is inferred."""
    with pytest.raises(ValueError, match="unsupported"):
        exchange.from_meshio(
            SimpleNamespace(cells=[SimpleNamespace(type=kind, data=[[0]])], cell_data={})
        )


def test_metadata_and_family_failures(codec):
    """Reject malformed tags, mixed volume families and unportable output formats."""
    mesh = TetraMesh.unit_cube()
    with pytest.raises(ValueError, match="unsupported native"):
        exchange.VolumeMeshData(object())
    for key in ("point_data", "cell_data"):
        with pytest.raises(ValueError, match="reserved"):
            exchange.VolumeMeshData(
                mesh,
                **{
                    key: {
                        "physical_tag": np.zeros(
                            len(mesh.points) if key == "point_data" else len(mesh.cells)
                        )
                    }
                },
            )
    with pytest.raises(ValueError, match=".vtu"):
        exchange.write_volume_mesh("grid.msh", mesh)
    for blocks in ([], [("tetra", mesh.cells), ("wedge", [[0, 1, 2, 3, 4, 5]])]):
        with pytest.raises(ValueError, match="one supported"):
            exchange.from_meshio(
                SimpleNamespace(
                    cells=[SimpleNamespace(type=k, data=v) for k, v in blocks], cell_data={}
                )
            )
    source = exchange.to_meshio(mesh)
    source.cell_data["pymhm_face_tags"][0] = np.zeros((len(mesh.cells), 3), dtype=int)
    with pytest.raises(ValueError, match="per-cell"):
        exchange.from_meshio(source)
    source.cell_data["pymhm_face_tags"][0] = np.ones((len(mesh.cells), 5), dtype=int)
    with pytest.raises(ValueError, match="padding"):
        exchange.from_meshio(source)


def test_vtu_noncanonical_polyhedron_blocks_are_rejected(codec):
    """Reject geometry/data layouts that meshio 5.3 can pair inconsistently."""
    codec["source"] = SimpleNamespace(
        cells=[SimpleNamespace(type="polyhedron8"), SimpleNamespace(type="polyhedron6")]
    )
    for path, fmt in (("mesh.vtu", None), ("mesh.data", "vtu")):
        with pytest.raises(ValueError, match="increasing vertex"):
            exchange.read_volume_mesh(path, file_format=fmt)
    mesh = families()[0]
    codec["source"] = exchange.to_meshio(mesh)
    assert isinstance(exchange.read_volume_mesh("mesh.data", file_format="vtk").mesh, TetraMesh)


@pytest.mark.meshing
@pytest.mark.parametrize("index", range(4))
def test_native_meshio_vtu_geometry_and_data(tmp_path, index):
    """Write/read through the real meshio codec, including polyhedron face arrays."""
    pytest.importorskip("meshio")
    mesh = families()[index]
    data = exchange.VolumeMeshData(
        mesh,
        cell_tags=np.arange(len(mesh.cells)),
        face_tags=np.arange(len(mesh.faces)) + 1,
        point_data={"x": mesh.points[:, 0]},
        cell_data={"vector": np.tile([1.0, 2.0, 3.0], (len(mesh.cells), 1))},
    )
    path = tmp_path / "native.vtu"
    exchange.write_volume_mesh(path, data)
    actual = exchange.read_volume_mesh(path)
    assert_array_equal(actual.cell_tags, data.cell_tags)
    assert_allclose(actual.point_data["x"], data.point_data["x"])
    expected = {
        tuple(sorted(face)): tag for face, tag in zip(mesh.faces, data.face_tags, strict=True)
    }
    assert all(
        tag == expected[tuple(sorted(face))]
        for face, tag in zip(actual.mesh.faces, actual.face_tags, strict=True)
    )
    assert_allclose(actual.cell_data["vector"], data.cell_data["vector"])


@pytest.mark.meshing
def test_native_polyhedra_with_different_vertex_and_face_counts(tmp_path):
    """Ragged polyhedron groups retain paired fields and tags when meshio groups cell types."""
    pytest.importorskip("meshio")
    from pymhm.polygon import PolygonMesh

    polygon = PolygonMesh(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]], [[0, 1, 2, 3], [1, 4, 2]]
    )
    mesh = PolyhedralMesh.extrude(polygon)
    data = exchange.VolumeMeshData(
        mesh,
        cell_tags=[7, 9],
        face_tags=np.arange(len(mesh.faces)) + 1,
        cell_data={"volume": mesh.volumes},
    )
    path = tmp_path / "ragged.vtu"
    exchange.write_volume_mesh(path, data)
    actual = exchange.read_volume_mesh(path)
    for tag, volume, field in zip(
        actual.cell_tags, actual.mesh.volumes, actual.cell_data["volume"], strict=True
    ):
        assert_allclose(volume, 1.0 if tag == 7 else 0.5)
        assert_allclose(field, volume)
    expected = {
        tuple(sorted(face)): tag for face, tag in zip(mesh.faces, data.face_tags, strict=True)
    }
    for face, tag in zip(actual.mesh.faces, actual.face_tags, strict=True):
        assert tag == expected[tuple(sorted(face))]


@pytest.mark.meshing
def test_native_warped_hexahedron_retains_trilinear_map(tmp_path):
    """VTU connectivity preserves physical maps and Jacobians of a nonaffine hexahedron."""
    pytest.importorskip("meshio")
    base = HexMesh.unit_cube()
    points = base.points.copy()
    points[np.argmax(points.sum(axis=1))] += [0.1, 0.05, 0.2]
    mesh = HexMesh(points, base.cells)
    path = tmp_path / "warped.vtu"
    exchange.write_volume_mesh(path, mesh)
    restored = exchange.read_volume_mesh(path).mesh
    samples = np.random.default_rng(624).uniform(0.0, 1.0, (19, 3))
    for expected, actual in zip(mesh.geometry(samples), restored.geometry(samples), strict=True):
        assert_allclose(actual, expected, atol=1e-15)
    assert np.ptp(mesh.geometry(samples)[2]) > 0.01
