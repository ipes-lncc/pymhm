"""Mesh exchange contracts and optional native generator round trips."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.io import planar as meshing
from pymhm.io.planar import (
    MeshData,
    from_gmsh,
    from_netgen,
    read_mesh,
    unit_square_gmsh,
    unit_square_netgen,
    write_mesh,
)
from pymhm.meshes.triangle import TriangleMesh


@pytest.fixture
def tagged() -> MeshData:
    mesh = TriangleMesh.unit_square()
    return MeshData(
        mesh,
        {7: tuple(mesh.boundary_faces), 9: (int(np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0]),)},
        np.array([2, 3]),
        {(1, 7): "outside wall", (1, 9): "interface", (2, 2): "material α", (2, 3): "rock"},
        {"temperature": np.arange(4.0)},
        {"permeability": np.array([1.0, 10.0])},
    )


@pytest.fixture
def meshio_contract(monkeypatch: pytest.MonkeyPatch) -> Any:
    saved: dict[str, Any] = {}

    def mesh(points: Any, cells: Any, **kwargs: Any) -> Any:
        blocks = [SimpleNamespace(type=kind, data=np.asarray(data)) for kind, data in cells]
        return SimpleNamespace(points=points, cells=blocks, **kwargs)

    def write(path: Any, value: Any, file_format: Any = None) -> None:
        saved[str(path)] = value
        saved["format"] = file_format

    fake = SimpleNamespace(
        Mesh=mesh,
        write=write,
        read=lambda path, file_format=None: saved[str(path)],
        saved=saved,
    )
    monkeypatch.setattr(meshing, "import_module", lambda name: fake)
    return fake


def assert_same_mesh_data(actual: MeshData, expected: MeshData) -> None:
    np.testing.assert_allclose(actual.mesh.points, expected.mesh.points)
    np.testing.assert_array_equal(actual.mesh.cells, expected.mesh.cells)
    assert actual.face_tags == expected.face_tags
    np.testing.assert_array_equal(actual.cell_tags, expected.cell_tags)
    assert actual.physical_names == expected.physical_names
    for name in expected.point_data:
        np.testing.assert_array_equal(actual.point_data[name], expected.point_data[name])
    for name in expected.cell_data:
        np.testing.assert_array_equal(actual.cell_data[name], expected.cell_data[name])


@pytest.mark.parametrize("suffix", [".msh", ".vtu"])
def test_meshio_contract_roundtrip(tagged: MeshData, meshio_contract: Any, suffix: str) -> None:
    if suffix == ".msh":
        tagged.cell_data = {}
    write_mesh("mesh" + suffix, tagged)
    if suffix == ".vtu":
        meshio_contract.saved["mesh" + suffix].field_data = {}
    result = read_mesh("mesh" + suffix)
    assert_same_mesh_data(result, tagged)
    assert len(result.boundary_tags[7]) == 4
    assert result.boundary_tags[9] == ()


def test_geometry_only_and_explicit_fields(meshio_contract: Any) -> None:
    mesh = TriangleMesh.unit_square()
    write_mesh("geometry.obj", mesh)
    result = read_mesh("geometry.obj", file_format="obj")
    assert result.cell_tags is None
    assert result.face_tags == {}
    write_mesh(
        "geometry.vtu",
        mesh,
        point_data={"u": np.ones((4, 2))},
        cell_data={"tensor": np.ones((2, 3))},
    )
    assert read_mesh("geometry.vtu").point_data["u"].shape == (4, 2)
    write_mesh("geometry.msh", mesh)
    np.testing.assert_array_equal(read_mesh("geometry.msh").cell_tags, [0, 0])


def test_boundary_only_export_and_field_override(meshio_contract: Any) -> None:
    mesh = TriangleMesh.unit_square()
    data = MeshData(mesh, {1: (0,)}, cell_data={"u": np.ones(2)})
    write_mesh("mesh.vtu", data, cell_data={"u": np.full(2, 3.0)})
    result = read_mesh("mesh.vtu")
    np.testing.assert_array_equal(result.cell_tags, [0, 0])
    np.testing.assert_array_equal(result.cell_data["u"], [3, 3])
    np.testing.assert_array_equal(data.cell_data["u"], [1, 1])


@pytest.mark.parametrize("error", [ImportError("missing"), OSError("shared library")])
def test_missing_dependency_actionable(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def fail(name: str) -> Any:
        raise error

    monkeypatch.setattr(meshing, "import_module", fail)
    with pytest.raises(ImportError, match=r"pymhm\[meshing\]") as exc:
        read_mesh("absent.vtu")
    assert exc.value.__cause__ is error


@pytest.mark.parametrize("points", [[[0, 1, 1]], [[0, 1, 0.1]], [[0, 1, np.nan]], [0, 1], [[0]]])
def test_nonplanar_or_invalid_points(points: Any) -> None:
    with pytest.raises(ValueError):
        meshing._xy(points)


def test_xy_accepts_native_2d() -> None:
    np.testing.assert_array_equal(meshing._xy([[1, 2]]), [[1, 2]])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"face_tags": {0: (0,)}}, "physical tag"),
        ({"face_tags": {1: (-1,)}}, "face index"),
        ({"face_tags": {1: (100,)}}, "outside"),
        ({"cell_tags": [1.0, 2.0]}, "integer vector"),
        ({"cell_tags": [1]}, "integer vector"),
        ({"cell_tags": [-1, 2]}, "nonnegative"),
        ({"physical_names": {(3, 1): "volume"}}, "dimension"),
        ({"physical_names": {(1, 1): ""}}, "dimension"),
        ({"physical_names": {(1, 0): "wall"}}, "physical tag"),
        ({"point_data": {"": np.ones(4)}}, "field names"),
        ({"cell_data": {"gmsh:physical": np.ones(2)}}, "reserved"),
        ({"cell_data": {"pymhm:physical:1:1:wall": np.ones(2)}}, "prefix"),
        ({"point_data": {"u": np.ones(3)}}, "one scalar"),
        ({"point_data": {"u": np.ones((4, 1, 1))}}, "one scalar"),
        ({"point_data": {"u": ["a"] * 4}}, "finite real numbers"),
        ({"point_data": {"u": [np.inf] * 4}}, "finite real numbers"),
        ({"point_data": {"u": [1j] * 4}}, "finite real numbers"),
        ({"point_data": {3: np.ones(4)}}, "field names"),
    ],
)
def test_invalid_mesh_metadata(kwargs: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MeshData(TriangleMesh.unit_square(), **kwargs)


def test_metadata_is_copied() -> None:
    tags = np.array([1, 2])
    value = np.ones(4)
    data = MeshData(TriangleMesh.unit_square(), {1: (0, 0)}, tags, point_data={"u": value})
    tags[:] = 9
    value[:] = 9
    np.testing.assert_array_equal(data.cell_tags, [1, 2])
    np.testing.assert_array_equal(data.point_data["u"], np.ones(4))
    assert data.face_tags[1] == (0,)


@pytest.mark.parametrize("cell_type", ["quad", "triangle6", "tetra", "line3"])
def test_read_rejects_untransformed_geometry(meshio_contract: Any, cell_type: str) -> None:
    write_mesh("mesh.vtu", TriangleMesh.unit_square())
    meshio_contract.saved["mesh.vtu"].cells[0].type = cell_type
    with pytest.raises(ValueError, match="unsupported cell"):
        read_mesh("mesh.vtu")


def test_read_no_triangles_and_unattached_lines(meshio_contract: Any) -> None:
    write_mesh("mesh.vtu", TriangleMesh.unit_square())
    source = meshio_contract.saved["mesh.vtu"]
    source.cells = [SimpleNamespace(type="vertex", data=np.array([[0]]))]
    with pytest.raises(ValueError, match="no linear triangles"):
        read_mesh("mesh.vtu")
    source.cells = [
        SimpleNamespace(type="triangle", data=np.array([[0, 1, 2]])),
        SimpleNamespace(type="line", data=np.array([[1, 3], [0, 1]])),
    ]
    source.cell_data = {"gmsh:physical": [np.array([0]), np.array([1, 0])]}
    with pytest.raises(ValueError, match="does not coincide"):
        read_mesh("mesh.vtu")


def test_read_combines_blocks_and_ignores_unmarked_lines_and_points(meshio_contract: Any) -> None:
    write_mesh("mesh.vtu", TriangleMesh.unit_square())
    source = meshio_contract.saved["mesh.vtu"]
    source.cells.extend(
        [
            SimpleNamespace(type="line", data=np.array([[0, 1]])),
            SimpleNamespace(type="vertex", data=np.array([[0]])),
        ]
    )
    assert len(read_mesh("mesh.vtu").mesh.cells) == 2


@pytest.mark.parametrize("tags", [[1.5, 2.0], [1], [-1, 2]])
def test_read_rejects_noninteger_negative_or_missized_physical_tags(
    meshio_contract: Any, tags: Any
) -> None:
    write_mesh("mesh.vtu", TriangleMesh.unit_square())
    meshio_contract.saved["mesh.vtu"].cell_data = {"gmsh:physical": [np.asarray(tags)]}
    with pytest.raises(ValueError, match="nonnegative integers"):
        read_mesh("mesh.vtu")


@pytest.mark.parametrize("kind", ["format", "overlap", "names"])
def test_export_refuses_metadata_loss(meshio_contract: Any, kind: str) -> None:
    mesh = TriangleMesh.unit_square()
    data = MeshData(mesh, {1: (0,)})
    path = "mesh.vtu"
    if kind == "format":
        path = "mesh.obj"
    elif kind == "overlap":
        data = MeshData(mesh, {1: (0,), 2: (0,)})
    else:
        data = MeshData(mesh, physical_names={(1, 1): "same", (2, 1): "same"})
    with pytest.raises(ValueError):
        write_mesh(path, data)
    assert path not in meshio_contract.saved


def test_gmsh22_refuses_unreadable_field_layouts(meshio_contract: Any) -> None:
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="tagged lines require VTU"):
        write_mesh("mesh.msh", MeshData(mesh, {1: (0,)}), cell_data={"u": np.ones(2)})
    with pytest.raises(ValueError, match="1, 3 or 9"):
        write_mesh("mesh.msh", mesh, point_data={"u": np.ones((4, 2))})
    write_mesh("mesh.msh", mesh, cell_data={"u": np.ones((2, 3))})
    assert read_mesh("mesh.msh").cell_data["u"].shape == (2, 3)


@pytest.fixture
def gmsh_contract(monkeypatch: pytest.MonkeyPatch) -> Any:
    class Model:
        current = "previous"
        dimension = 2
        nonlinear = False
        overlap = False
        failure = False
        created: list[str] = []

        def __init__(self) -> None:
            self.mesh = SimpleNamespace(
                getNodes=self.nodes, getElements=self.elements, generate=self.generate
            )
            self.geo = SimpleNamespace(
                addPoint=lambda *a: 1,
                addLine=lambda *a: 1,
                addCurveLoop=lambda *a: 1,
                addPlaneSurface=lambda *a: 1,
                synchronize=lambda: None,
            )

        def getDimension(self) -> int:
            return self.dimension

        def getCurrent(self) -> str:
            return self.current

        def setCurrent(self, name: str) -> None:
            self.current = name

        def add(self, name: str) -> None:
            self.created.append(name)
            self.current = name

        def remove(self) -> None:
            self.created.remove(self.current)
            self.current = ""

        def addPhysicalGroup(self, *args: Any) -> None:
            pass

        def nodes(self) -> Any:
            return (
                [10, 30, 50, 70],
                np.c_[TriangleMesh.unit_square().points, np.zeros(4)].ravel(),
                [],
            )

        def elements(self, dim: int, entity: int) -> Any:
            if entity == 99:
                return [], [], []
            if dim == 2:
                return (
                    [9 if self.nonlinear else 2],
                    [np.array([101, 201])],
                    [np.array([10, 30, 70, 10, 70, 50])],
                )
            return [1], [np.array([301])], [np.array([10, 30])]

        def getPhysicalGroups(self) -> Any:
            return [(0, 1), (1, 7), (1, 9), (2, 2)] + ([(2, 3)] if self.overlap else [])

        def getPhysicalName(self, dim: int, tag: int) -> str:
            return "" if tag == 9 else f"name-{dim}-{tag}"

        def getEntitiesForPhysicalGroup(self, dim: int, tag: int) -> Any:
            return [99 if tag == 9 else 1]

        def generate(self, dim: int) -> None:
            if self.failure:
                raise RuntimeError("meshing failed")

    fake = SimpleNamespace(model=Model(), initialized=True, finalized=0)

    def initialize() -> None:
        fake.initialized = True

    def finalize() -> None:
        fake.initialized = False
        fake.finalized += 1

    fake.initialize = initialize
    fake.finalize = finalize
    fake.isInitialized = lambda: fake.initialized
    monkeypatch.setattr(meshing, "import_module", lambda name: fake)
    return fake


def test_gmsh_sparse_tags_import_and_session_ownership(gmsh_contract: Any) -> None:
    data = from_gmsh()
    np.testing.assert_allclose(data.mesh.areas.sum(), 1)
    assert data.face_tags[7] == (0,)
    assert data.face_tags[9] == ()
    np.testing.assert_array_equal(data.cell_tags, [2, 2])
    unit_square_gmsh(0.3)
    assert gmsh_contract.initialized
    assert gmsh_contract.finalized == 0
    assert gmsh_contract.model.current == "previous"
    gmsh_contract.initialized = False
    gmsh_contract.model.current = ""
    unit_square_gmsh(0.3)
    assert not gmsh_contract.initialized
    assert gmsh_contract.finalized == 1
    assert gmsh_contract.model.created == []


def test_gmsh_failure_restores_caller_state(gmsh_contract: Any) -> None:
    gmsh_contract.model.failure = True
    with pytest.raises(RuntimeError, match="meshing failed"):
        unit_square_gmsh()
    assert gmsh_contract.initialized
    assert gmsh_contract.model.current == "previous"
    assert gmsh_contract.model.created == []


@pytest.mark.parametrize(
    "kind,match",
    [
        ("uninitialized", "initialized"),
        ("dimension", "dimension"),
        ("nonlinear", "linear triangles"),
        ("overlap", "overlapping"),
    ],
)
def test_gmsh_invalid_contract(gmsh_contract: Any, kind: str, match: str) -> None:
    if kind == "uninitialized":
        gmsh_contract.initialized = False
    elif kind == "dimension":
        gmsh_contract.model.dimension = 3
    else:
        setattr(gmsh_contract.model, kind, True)
    with pytest.raises((ValueError, RuntimeError), match=match):
        from_gmsh()


@pytest.fixture
def netgen_contract(monkeypatch: pytest.MonkeyPatch) -> Any:
    def element(vertices: Any, tag: int) -> Any:
        nodes = [SimpleNamespace(nr=vertex) for vertex in vertices]
        return SimpleNamespace(vertices=nodes, points=nodes, index=tag)

    source = SimpleNamespace(dim=2, triangles=[element([1, 2, 3], 1)], lines=[element([1, 2], 1)])
    source.Points = lambda: [
        SimpleNamespace(p=point) for point in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    ]
    source.Elements2D = lambda: source.triangles
    source.Elements1D = lambda: source.lines
    source.GetBCName = lambda tag: "bottom"
    source.GetMaterial = lambda tag: "domain"
    generator = SimpleNamespace(unit_square=SimpleNamespace(GenerateMesh=lambda maxh: source))
    monkeypatch.setattr(meshing, "import_module", lambda name: generator)
    return source


def test_netgen_contract(netgen_contract: Any) -> None:
    data = unit_square_netgen(0.2)
    np.testing.assert_allclose(data.mesh.areas, [0.5])
    assert data.physical_names == {(1, 1): "bottom", (2, 1): "domain"}
    assert data.face_tags == {1: (0,)}


@pytest.mark.parametrize("kind", ["dimension", "triangle", "line"])
def test_netgen_invalid_contract(netgen_contract: Any, kind: str) -> None:
    if kind == "dimension":
        netgen_contract.dim = 3
    elif kind == "triangle":
        netgen_contract.triangles[0].points.append(SimpleNamespace(nr=4))
    else:
        netgen_contract.lines[0].points.append(SimpleNamespace(nr=3))
    with pytest.raises(ValueError):
        from_netgen(netgen_contract)


@pytest.mark.parametrize("size", [0, -1, True, np.inf, np.nan])
@pytest.mark.parametrize("generate", [unit_square_gmsh, unit_square_netgen])
def test_invalid_size_precedes_dependency_import(generate: Any, size: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        generate(size)


@pytest.mark.meshing
@pytest.mark.parametrize("generator", ["gmsh", "netgen"])
@pytest.mark.parametrize("suffix", [".vtu", ".msh"])
def test_native_generators_and_meshio_roundtrip(
    tmp_path: Path, generator: str, suffix: str
) -> None:
    pytest.importorskip("meshio")
    pytest.importorskip(generator)
    data = unit_square_gmsh(0.35) if generator == "gmsh" else unit_square_netgen(0.35)
    np.testing.assert_allclose(data.mesh.areas.sum(), 1, atol=1e-14)
    assert len(data.boundary_tags) == 4
    assert sum(len(faces) for faces in data.boundary_tags.values()) == len(data.mesh.boundary_faces)
    path = tmp_path / ("mesh" + suffix)
    fields = {"area": data.mesh.areas} if suffix == ".vtu" else {}
    write_mesh(path, data, point_data={"x": data.mesh.points[:, 0]}, cell_data=fields)
    actual = read_mesh(path)
    assert_same_mesh_data(actual, data)
    np.testing.assert_allclose(actual.point_data["x"], data.mesh.points[:, 0])
    if suffix == ".vtu":
        np.testing.assert_allclose(actual.cell_data["area"], data.mesh.areas)


@pytest.mark.meshing
def test_native_gmsh_caller_model_survives() -> None:
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.model.add("caller")
        gmsh.model.geo.addPoint(3, 4, 0)
        gmsh.model.geo.synchronize()
        unit_square_gmsh(0.5)
        assert gmsh.isInitialized()
        assert gmsh.model.getCurrent() == "caller"
        assert len(gmsh.model.getEntities(0)) == 1
        assert "caller" in gmsh.model.list()
    finally:
        gmsh.finalize()


@pytest.mark.meshing
@pytest.mark.parametrize("generator", ["gmsh", "netgen"])
def test_native_meshes_solve_affine_darcy(generator: str) -> None:
    pytest.importorskip(generator)
    data = unit_square_gmsh(0.5) if generator == "gmsh" else unit_square_netgen(0.5)

    def exact(points: Any) -> Any:
        return 1 + points[:, 0] - 2 * points[:, 1]

    result = solve_darcy(data.mesh, dirichlet=exact, local_refinement=2)
    assert result.l2_error(exact) < 1e-11


@pytest.mark.meshing
@pytest.mark.parametrize("suffix", [".msh", ".vtu"])
def test_native_named_interface_roundtrip(tmp_path: Path, tagged: MeshData, suffix: str) -> None:
    pytest.importorskip("meshio")
    if suffix == ".msh":
        tagged.cell_data = {}
    path = tmp_path / ("interface" + suffix)
    write_mesh(path, tagged)
    assert_same_mesh_data(read_mesh(path), tagged)
