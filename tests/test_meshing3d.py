"""Mesh exchange contracts and native meshio tetrahedral roundtrips."""

from types import SimpleNamespace

import numpy as np
import pytest

from pymhm.io import tetrahedral as meshing3d
from pymhm.io.tetrahedral import TetraMeshData, read_tetra_mesh, write_tetra_mesh
from pymhm.meshes.tetrahedron import TetraMesh


def mesh():
    return TetraMesh.unit_cube(1)


def test_metadata_validation():
    m = mesh()
    for kwargs in (
        {"cell_tags": [-1] * len(m.cells)},
        {"face_tags": [1.0] * len(m.faces)},
        {"cell_data": {"physical_tag": np.ones(len(m.cells))}},
    ):
        with pytest.raises(ValueError):
            TetraMeshData(m, **kwargs)


def source(blocks, physical=None, cell_data=None):
    data = {} if cell_data is None else cell_data
    if physical is not None:
        data["gmsh:physical"] = physical
    return SimpleNamespace(
        points=mesh().points,
        cells=[SimpleNamespace(type=t, data=c) for t, c in blocks],
        cell_data=data,
        point_data={},
    )


def test_simulated_meshio_roundtrip_and_tags(monkeypatch):
    m = mesh()
    state = {}

    def write(path, points, cells, **kwargs):
        state.update(points=points, cells=cells, **kwargs)

    def read(path, **kwargs):
        return SimpleNamespace(
            points=state["points"],
            cells=[SimpleNamespace(type=t, data=c) for t, c in state["cells"]],
            cell_data=state["cell_data"],
            point_data=state["point_data"],
        )

    monkeypatch.setattr(
        meshing3d, "_optional", lambda name: SimpleNamespace(read=read, write_points_cells=write)
    )
    data = TetraMeshData(
        m,
        cell_tags=np.arange(len(m.cells)),
        face_tags=np.ones(len(m.faces), dtype=int),
        point_data={"p": m.points[:, 0]},
        cell_data={"q": np.ones((len(m.cells), 3))},
    )
    write_tetra_mesh("contract.vtu", data)
    actual = read_tetra_mesh("contract.vtu")
    np.testing.assert_allclose(actual.cell_data["q"], data.cell_data["q"])
    np.testing.assert_array_equal(actual.face_tags, data.face_tags)
    write_tetra_mesh("plain.vtu", m)
    actual = read_tetra_mesh("plain.vtu")
    assert actual.face_tags.sum() == 0


@pytest.mark.parametrize("kind", ["empty", "unsupported", "bad-face", "conflict", "marked"])
def test_meshio_read_contracts(monkeypatch, kind):
    m = mesh()
    blocks = [("tetra", m.cells), ("line", [[0, 1]]), ("vertex", [[0]])]
    physical = None
    if kind == "empty":
        blocks = blocks[1:]
    if kind == "unsupported":
        blocks.append(("hexahedron", np.arange(8)[None]))
    if kind == "bad-face":
        blocks.append(("triangle", [[0, 0, 0]]))
    if kind == "conflict":
        blocks.extend([("triangle", [m.faces[0]]), ("triangle", [m.faces[0]])])
        physical = [
            np.ones(len(m.cells), dtype=int),
            np.zeros(1, dtype=int),
            np.zeros(1, dtype=int),
            np.array([1]),
            np.array([2]),
        ]
    if kind == "marked":
        blocks.append(("triangle", [m.faces[0]]))
        physical = [
            np.ones(len(m.cells), dtype=int),
            np.zeros(1, dtype=int),
            np.zeros(1, dtype=int),
            np.array([5]),
        ]
    fixture = source(blocks, physical)
    monkeypatch.setattr(
        meshing3d, "_optional", lambda name: SimpleNamespace(read=lambda *args, **kwargs: fixture)
    )
    if kind == "marked":
        actual = read_tetra_mesh("marked.msh")
        assert actual.face_tags[0] == 5
    else:
        with pytest.raises(ValueError):
            read_tetra_mesh("invalid.msh")


@pytest.mark.meshing
def test_native_meshio_tetrahedral_vtu_roundtrip(tmp_path):
    pytest.importorskip("meshio")
    m = mesh()
    face_tags = np.zeros(len(m.faces), dtype=int)
    face_tags[m.boundary_faces] = 7
    data = TetraMeshData(
        m,
        cell_tags=np.arange(len(m.cells)),
        face_tags=face_tags,
        point_data={"x": m.points[:, 0]},
        cell_data={"vector": np.ones((len(m.cells), 3))},
    )
    path = tmp_path / "tetra.vtu"
    write_tetra_mesh(path, data)
    actual = read_tetra_mesh(path)
    np.testing.assert_array_equal(actual.mesh.cells, data.mesh.cells)
    np.testing.assert_array_equal(actual.cell_tags, data.cell_tags)
    np.testing.assert_array_equal(actual.face_tags, data.face_tags)
    np.testing.assert_allclose(actual.cell_data["vector"], 1)
