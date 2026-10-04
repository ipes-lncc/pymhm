"""Optional linear volume-mesh exchange with explicit native connectivity conventions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pymhm.io.planar import _data, _optional
from pymhm.io.tetrahedral import _tags
from pymhm.meshes.hexahedron import HexMesh
from pymhm.meshes.mixed import AffineMixedMesh
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import TetraMesh

VolumeMesh = TetraMesh | AffineMixedMesh | HexMesh | PolyhedralMesh
_HEX_TO_MESHIO = np.array([0, 4, 6, 2, 1, 5, 7, 3])
_HEX_FROM_MESHIO = np.argsort(_HEX_TO_MESHIO)
_CELL_TAG = "physical_tag"
_FACE_TAG = "pymhm_face_tags"
_RESERVED = {_CELL_TAG, _FACE_TAG, "gmsh:physical", "gmsh:geometrical"}


@dataclass
class VolumeMeshData:
    """A native volume mesh with integer cell/face IDs and finite nodal/cell fields.

    Geometry is linear tetrahedral, affine wedge, trilinear hexahedral or star-shaped
    polyhedral with planar faces. Integer face tags can describe internal material
    interfaces as well as the exterior. Physical-name strings and curved
    higher-order geometry are outside this exchange contract.
    """

    mesh: VolumeMesh
    cell_tags: Any = None
    face_tags: Any = None
    point_data: dict[str, Any] = field(default_factory=dict)
    cell_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the actual native geometry and every metadata-array length."""
        if not isinstance(self.mesh, (TetraMesh, AffineMixedMesh, HexMesh, PolyhedralMesh)):
            raise ValueError("unsupported native volume mesh")
        self.cell_tags = _tags(self.cell_tags, len(self.mesh.cells))
        self.face_tags = _tags(self.face_tags, len(self.mesh.faces))
        self.point_data = _data(self.point_data, len(self.mesh.points))
        self.cell_data = _data(self.cell_data, len(self.mesh.cells))
        if _RESERVED.intersection(self.point_data) or _RESERVED.intersection(self.cell_data):
            raise ValueError("physical and per-face tag field names are reserved")


def _polyhedral_mesh(points: Any, cells: list[Any]) -> PolyhedralMesh:
    """Deduplicate complete original polygonal faces without subdividing their traces."""
    faces: list[Any] = []
    connectivity: list[list[int]] = []
    lookup: dict[tuple[int, ...], int] = {}
    for cell in cells:
        indices = []
        for face in cell:
            key = tuple(sorted(int(node) for node in face))
            if key not in lookup:
                lookup[key] = len(faces)
                faces.append(np.asarray(face))
            indices.append(lookup[key])
        connectivity.append(indices)
    return PolyhedralMesh(points, faces, connectivity)


def from_meshio(source: Any) -> VolumeMeshData:
    """Import one linear volume family from a meshio-compatible in-memory object.

    Wedge6 follows meshio's triangle/translate order, already shared by
    ``AffineMixedMesh``. Hexahedron8 is explicitly permuted from VTK corner order
    into native lexicographic (x,y,z). PolyhedronN keeps each polygonal face;
    it is not replaced by independent triangular macrofaces. Mixed volume
    families and higher-order geometry are rejected rather than linearized.
    Triangular/quad/polygon boundary blocks and integer physical IDs are accepted.
    """
    volumes: list[Any] = []
    boundaries: list[tuple[Any, Any]] = []
    kinds: set[str] = set()
    physical = source.cell_data.get("gmsh:physical", source.cell_data.get(_CELL_TAG))
    tags: list[Any] = []
    fields: dict[str, list[Any]] = {}
    face_blocks: list[Any] = []
    for index, block in enumerate(source.cells):
        kind = block.type
        values = block.data
        ids = _tags(None if physical is None else physical[index], len(values))
        is_poly = kind.startswith("polyhedron")
        if kind in ("tetra", "wedge", "hexahedron") or is_poly:
            kinds.add("polyhedron" if is_poly else kind)
            volumes.extend(values)
            tags.append(ids)
            if _FACE_TAG in source.cell_data:
                face_blocks.extend(source.cell_data[_FACE_TAG][index])
            for name, arrays in source.cell_data.items():
                if name not in _RESERVED:
                    fields.setdefault(name, []).append(arrays[index])
        elif kind in ("triangle", "quad") or kind.startswith("polygon"):
            boundaries.extend(zip(values, ids, strict=True))
        elif kind not in ("line", "vertex"):
            raise ValueError(f"unsupported volume geometry {kind!r}; linear cells are required")
    if len(kinds) != 1:
        raise ValueError("require exactly one supported volume family")
    kind = kinds.pop()
    mesh: VolumeMesh
    if kind == "tetra":
        mesh = TetraMesh(source.points, np.asarray(volumes))
    elif kind == "wedge":
        mesh = AffineMixedMesh(source.points, np.asarray(volumes), "prism")
    elif kind == "hexahedron":
        mesh = HexMesh(source.points, np.asarray(volumes)[:, _HEX_FROM_MESHIO])
    else:
        mesh = _polyhedral_mesh(source.points, volumes)
    lookup = {tuple(sorted(face)): i for i, face in enumerate(mesh.faces)}
    face_tags = np.zeros(len(mesh.faces), dtype=np.int64)

    def assign(face: Any, tag: Any) -> None:
        """Resolve a physical face by vertices and reject conflicting nonzero IDs."""
        key = tuple(sorted(face))
        if key not in lookup:
            raise ValueError("tagged surface is not a face of the volume mesh")
        index = lookup[key]
        if face_tags[index] and tag and face_tags[index] != tag:
            raise ValueError("conflicting physical IDs on a volume face")
        if tag:
            face_tags[index] = tag

    for face, tag in boundaries:
        assign(face, tag)
    if face_blocks:
        for cell, saved in enumerate(face_blocks):
            local_faces = mesh.cell_faces[cell]
            values = _tags(saved, len(saved))
            if len(values) < len(local_faces) or np.any(values[len(local_faces) :]):
                raise ValueError(
                    "per-cell face tags need one entry per local face and zero padding"
                )
            # Polyhedron records retain their input local face ordering; fixed
            # families use the native local-face order after corner conversion.
            for face, tag in zip(local_faces, values[: len(local_faces)], strict=True):
                assign(mesh.faces[face], tag)
    return VolumeMeshData(
        mesh,
        np.concatenate(tags),
        face_tags,
        source.point_data,
        {name: np.concatenate(values) for name, values in fields.items()},
    )


def read_volume_mesh(path: str | Path, *, file_format: str | None = None) -> VolumeMeshData:
    """Read a linear volume mesh without silently pairing fields with other cells.

    VTU polyhedron blocks must be ordered by increasing vertex count. This
    canonical layout keeps geometry and cell data aligned in meshio 5.3, whose
    reader sorts polyhedron field blocks independently of geometry blocks.
    ``write_volume_mesh`` emits that layout. In-memory ``from_meshio`` accepts
    any correctly paired block order.
    """
    source = _optional("meshio").read(path, file_format=file_format)
    if file_format == "vtu" or (file_format is None and Path(path).suffix.lower() == ".vtu"):
        counts = [
            int(block.type[10:]) for block in source.cells if block.type.startswith("polyhedron")
        ]
        if counts != sorted(counts):
            raise ValueError(
                "VTU polyhedron blocks require increasing vertex counts for paired fields"
            )
    return from_meshio(source)


def to_meshio(data: VolumeMeshData | VolumeMesh) -> Any:
    """Create a meshio Mesh carrying native linear geometry and lossless integer tags.

    Per-cell ``pymhm_face_tags`` stores the local native face ordering, with zero
    padding for polyhedra with different face counts. This avoids adding surface
    cells to VTU polyhedron grids, which meshio cannot mix with surface blocks.
    It is an explicit PyMHM metadata convention, not a CAD physical-name model.
    """
    if not isinstance(data, VolumeMeshData):
        data = VolumeMeshData(data)
    mesh = data.mesh
    cells: list[tuple[str, Any]]
    if isinstance(mesh, PolyhedralMesh):
        grouped: dict[int, list[int]] = {}
        for cell, faces in enumerate(mesh.cells):
            count = len(np.unique(np.concatenate([mesh.faces[face] for face in faces])))
            grouped.setdefault(count, []).append(cell)
        grouped = dict(sorted(grouped.items()))
        groups = list(grouped.values())
        cells = [
            (
                f"polyhedron{count}",
                [[mesh.faces[face] for face in mesh.cells[cell]] for cell in indices],
            )
            for count, indices in grouped.items()
        ]
    else:
        groups = [list(range(len(mesh.cells)))]
        if isinstance(mesh, HexMesh):
            cells = [("hexahedron", mesh.cells[:, _HEX_TO_MESHIO])]
        else:
            cells = [
                (
                    "wedge"
                    if isinstance(mesh, AffineMixedMesh) and mesh.kind == "prism"
                    else "tetra",
                    mesh.cells,
                )
            ]
    width = max(len(faces) for faces in mesh.cell_faces)
    face_tags = np.zeros((len(mesh.cells), width), dtype=np.int64)
    for cell, faces in enumerate(mesh.cell_faces):
        face_tags[cell, : len(faces)] = data.face_tags[faces]
    all_fields = {**data.cell_data, _CELL_TAG: data.cell_tags, _FACE_TAG: face_tags}
    return _optional("meshio").Mesh(
        mesh.points,
        cells,
        point_data=data.point_data,
        cell_data={
            name: [values[indices] for indices in groups] for name, values in all_fields.items()
        },
    )


def write_volume_mesh(path: str | Path, data: VolumeMeshData | VolumeMesh) -> None:
    """Write VTU geometry and finite fields without discarding internal face tags.

    VTU is required because other meshio formats do not uniformly preserve
    arbitrary-component per-face tag arrays and polyhedral connectivity.
    """
    if Path(path).suffix.lower() != ".vtu":
        raise ValueError("lossless volume exchange requires the .vtu format")
    to_meshio(data).write(path, file_format="vtu")
