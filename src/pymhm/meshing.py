"""Optional mesh exchange and generation for straight-sided planar triangles.

Dependencies are loaded only when the corresponding operation is requested.
Physical curve groups may include material interfaces as well as boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import quote, unquote
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from pymhm.mesh import TriangleMesh, positive_int

Array = NDArray[Any]
_NAME_PREFIX = "pymhm:physical:"
_RESERVED = ("gmsh:physical", "gmsh:geometrical")


def _optional(name: str) -> ModuleType:
    """Load a meshing dependency with an actionable installation error."""
    try:
        return import_module(name)
    except (ImportError, OSError) as exc:
        raise ImportError(f"Cannot load {name}; install pymhm[meshing]. {exc}") from exc


def _xy(points: Any) -> Array:
    """Require finite XY coordinates, allowing the conventional zero Z column."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] not in (2, 3) or not np.isfinite(values).all():
        raise ValueError("mesh points must be finite with two or three coordinates")
    if values.shape[1] == 3 and np.any(values[:, 2] != 0):
        raise ValueError("only meshes in the XY plane (z=0) are supported")
    return values[:, :2]


def _data(values: Mapping[str, Any], size: int) -> dict[str, Array]:
    """Copy finite numeric fields with one value or vector per mesh entity."""
    result = {}
    for name, value in values.items():
        array = np.array(value, copy=True)
        if not isinstance(name, str) or not name or name in _RESERVED:
            raise ValueError("field names must be nonempty and not reserved physical-tag names")
        if name.startswith(_NAME_PREFIX):
            raise ValueError("field name uses the reserved physical-name prefix")
        if array.ndim not in (1, 2) or array.shape[0] != size:
            raise ValueError(f"field {name!r} must have one scalar or vector per entity")
        if (
            not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"field {name!r} must contain finite real numbers")
        result[name] = array
    return result


@dataclass
class MeshData:
    """A triangular mesh with physical groups and optional point/cell fields.

    ``face_tags`` maps positive physical IDs to mesh face indices; it includes
    interior interfaces. ``cell_tags`` is one nonnegative material ID per cell,
    with zero meaning unmarked. ``physical_names`` uses ``(dimension, ID)`` keys.
    Fields in ``cell_data`` refer only to triangles. Input arrays are copied.
    """

    mesh: TriangleMesh
    face_tags: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    cell_tags: Array | None = None
    physical_names: Mapping[tuple[int, int], str] = field(default_factory=dict)
    point_data: Mapping[str, Array] = field(default_factory=dict)
    cell_data: Mapping[str, Array] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate physical IDs, field dimensions and references to faces."""
        groups = {}
        for tag, faces in self.face_tags.items():
            tag = positive_int(tag, "physical tag")
            indices = tuple(positive_int(face, "face index", 0) for face in faces)
            if any(face >= len(self.mesh.faces) for face in indices):
                raise ValueError("physical face index outside mesh")
            groups[tag] = tuple(sorted(set(indices)))
        self.face_tags = groups
        if self.cell_tags is not None:
            tags = np.asarray(self.cell_tags)
            if tags.shape != (len(self.mesh.cells),) or not np.issubdtype(tags.dtype, np.integer):
                raise ValueError("cell_tags must be an integer vector with one tag per triangle")
            if np.any(tags < 0):
                raise ValueError("cell_tags must be nonnegative")
            self.cell_tags = tags.astype(np.int64, copy=True)
        names = {}
        for (dim, tag), name in self.physical_names.items():
            if dim not in (1, 2) or not isinstance(name, str) or not name:
                raise ValueError("physical names require dimension 1 or 2 and a nonempty name")
            names[(dim, positive_int(tag, "physical tag"))] = name
        self.physical_names = names
        self.point_data = _data(self.point_data, len(self.mesh.points))
        self.cell_data = _data(self.cell_data, len(self.mesh.cells))

    @property
    def boundary_tags(self) -> dict[int, tuple[int, ...]]:
        """Return physical groups restricted to external boundary faces."""
        boundary = set(self.mesh.boundary_faces)
        return {
            tag: tuple(face for face in faces if face in boundary)
            for tag, faces in self.face_tags.items()
        }


def _tagged_faces(mesh: TriangleMesh, groups: Mapping[int, Any]) -> dict[int, tuple[int, ...]]:
    """Map unoriented vertex pairs to canonical mesh face indices."""
    lookup = {tuple(sorted(face)): i for i, face in enumerate(mesh.faces)}
    result = {}
    for tag, lines in groups.items():
        faces = []
        for line in lines:
            key = tuple(sorted(line))
            if key not in lookup:
                raise ValueError("tagged line does not coincide with a triangle face")
            faces.append(lookup[key])
        result[tag] = tuple(faces)
    return result


def read_mesh(path: str | Path, *, file_format: str | None = None) -> MeshData:
    """Read meshio-supported linear triangles and physical groups.

    Point elements are ignored. Quadrilaterals, volume cells, higher-order cells,
    and nonzero Z coordinates are rejected rather than silently transformed.
    Arbitrary line fields are not part of the returned triangular cell fields.
    """
    source = _optional("meshio").read(path, file_format=file_format)
    triangles, tags = [], []
    groups: dict[int, list[Array]] = {}
    fields: dict[str, list[Array]] = {}
    names = {(int(v[1]), int(v[0])): k for k, v in source.field_data.items() if v[1] in (1, 2)}
    physical = source.cell_data.get("gmsh:physical")
    for block_index, block in enumerate(source.cells):
        if block.type not in ("triangle", "line", "vertex"):
            raise ValueError(f"unsupported cell type {block.type!r}; use linear 2D triangles")
        marked = np.zeros(len(block.data), dtype=np.int64)
        if physical is not None:
            marked = np.asarray(physical[block_index])
            if (
                marked.shape != (len(block.data),)
                or not np.issubdtype(marked.dtype, np.integer)
                or np.any(marked < 0)
            ):
                raise ValueError("physical tags must be nonnegative integers, one per element")
        if block.type == "triangle":
            triangles.append(block.data)
            tags.append(marked)
        elif block.type == "line":
            for tag in np.unique(marked):
                if tag > 0:
                    groups.setdefault(int(tag), []).extend(block.data[marked == tag])
        for name, blocks in source.cell_data.items():
            if name.startswith(_NAME_PREFIX):
                dim, tag, label = name[len(_NAME_PREFIX) :].split(":", 2)
                names[(int(dim), int(tag))] = unquote(label)
            elif name not in _RESERVED and block.type == "triangle":
                fields.setdefault(name, []).append(blocks[block_index])
    if not triangles:
        raise ValueError("mesh contains no linear triangles")
    mesh = TriangleMesh(_xy(source.points), np.concatenate(triangles))
    return MeshData(
        mesh,
        _tagged_faces(mesh, groups),
        np.concatenate(tags) if physical is not None else None,
        names,
        source.point_data,
        {name: np.concatenate(blocks) for name, blocks in fields.items()},
    )


def write_mesh(
    path: str | Path,
    data: TriangleMesh | MeshData,
    *,
    point_data: Mapping[str, Any] | None = None,
    cell_data: Mapping[str, Any] | None = None,
    file_format: str | None = None,
) -> None:
    """Export geometry and fields through meshio, preserving physical metadata.

    VTU preserves physical IDs, names and numeric fields. Gmsh 2.2 also supports
    point fields; cell fields combined with tagged lines require VTU because of
    a meshio 5.3 reader limitation. Other formats are accepted for geometry alone.
    Multiple physical IDs on the same face
    cannot be represented by the scalar physical-tag field and are rejected.
    Existing fields may be replaced explicitly by name in the keyword mappings.
    """
    meshio = _optional("meshio")
    source = data if isinstance(data, MeshData) else MeshData(data)
    mesh = source.mesh
    points = dict(source.point_data) | _data(point_data or {}, len(mesh.points))
    fields = dict(source.cell_data) | _data(cell_data or {}, len(mesh.cells))
    format_name = file_format or {".msh": "gmsh22", ".vtu": "vtu"}.get(Path(path).suffix.lower())
    metadata = bool(source.face_tags or source.physical_names or points or fields)
    metadata = metadata or source.cell_tags is not None
    if metadata and format_name not in ("vtu", "gmsh22"):
        raise ValueError("metadata-preserving export requires VTU or Gmsh 2.2")
    cells: list[tuple[str, Array]] = [("triangle", mesh.cells)]
    field_blocks = {name: [array] for name, array in fields.items()}
    marked: dict[int, int] = {}
    for tag, faces in source.face_tags.items():
        for face in faces:
            if face in marked:
                raise ValueError("export cannot represent overlapping physical face groups")
            marked[face] = tag
    if marked:
        cells.append(("line", mesh.faces[list(marked)]))
        for name, array in fields.items():
            field_blocks[name].append(np.zeros((len(marked), *array.shape[1:]), dtype=array.dtype))
    if format_name == "gmsh22":
        if marked and fields:
            raise ValueError(
                "Gmsh 2.2 cell fields with tagged lines require VTU for reliable import"
            )
        if any(
            array.ndim == 2 and array.shape[1] not in (1, 3, 9)
            for array in (*points.values(), *fields.values())
        ):
            raise ValueError("Gmsh fields require 1, 3 or 9 components; use VTU for other vectors")
    if source.cell_tags is not None or marked or format_name == "gmsh22":
        material = source.cell_tags
        physical = [np.zeros(len(mesh.cells), dtype=np.int64) if material is None else material]
        if marked:
            physical.append(np.asarray(list(marked.values()), dtype=np.int64))
        field_blocks["gmsh:physical"] = physical
        if format_name == "gmsh22":
            field_blocks["gmsh:geometrical"] = [array.copy() for array in physical]
    if format_name == "vtu":
        for (dim, tag), name in source.physical_names.items():
            key = f"{_NAME_PREFIX}{dim}:{tag}:{quote(name, safe='')}"
            field_blocks[key] = [np.zeros(len(block), dtype=np.int8) for _, block in cells]
    names: dict[str, Array] = {}
    for (dim, tag), name in source.physical_names.items():
        if name in names:
            raise ValueError("physical names must be distinct across dimensions for export")
        names[name] = np.array([tag, dim], dtype=np.int64)
    output = meshio.Mesh(
        np.column_stack((mesh.points, np.zeros(len(mesh.points)))),
        cells,
        point_data=points,
        cell_data=field_blocks,
        field_data=names,
    )
    meshio.write(path, output, file_format=format_name)


def from_gmsh(model: Any | None = None) -> MeshData:
    """Copy the current Gmsh model without modifying its session or mesh.

    An explicit model-like object may be passed. Node tags need not be contiguous.
    Only first-order triangles, lines and points are supported.
    """
    if model is None:
        gmsh = _optional("gmsh")
        if not gmsh.isInitialized():
            raise RuntimeError("Gmsh must be initialized before importing its current model")
        model = gmsh.model
    if model.getDimension() != 2:
        raise ValueError("Gmsh model must have dimension 2")
    node_tags, coordinates, _ = model.mesh.getNodes()
    nodes = {int(tag): i for i, tag in enumerate(node_tags)}

    def connectivity(dimension: int, entity: int = -1) -> tuple[Array, Array]:
        """Extract linear cells of one dimension and remap their node tags."""
        types, element_tags, node_blocks = model.mesh.getElements(dimension, entity)
        expected, width = (2, 3) if dimension == 2 else (1, 2)
        arrays, ids = [], []
        for kind, tags, block in zip(types, element_tags, node_blocks, strict=True):
            if kind != expected:
                raise ValueError("Gmsh mesh must use linear triangles and linear lines")
            arrays.append(np.array([nodes[int(tag)] for tag in block]).reshape(-1, width))
            ids.append(np.asarray(tags))
        return (
            np.concatenate(arrays) if arrays else np.empty((0, width), dtype=np.int64),
            np.concatenate(ids) if ids else np.empty(0, dtype=np.int64),
        )

    triangles, element_ids = connectivity(2)
    mesh = TriangleMesh(_xy(np.asarray(coordinates).reshape(-1, 3)), triangles)
    lookup = {int(tag): i for i, tag in enumerate(element_ids)}
    materials = np.zeros(len(triangles), dtype=np.int64)
    groups: dict[int, list[Array]] = {}
    names = {}
    for dimension, tag in model.getPhysicalGroups():
        if dimension not in (1, 2):
            continue
        name = model.getPhysicalName(dimension, tag)
        if name:
            names[(dimension, tag)] = name
        for entity in model.getEntitiesForPhysicalGroup(dimension, tag):
            elements, ids = connectivity(dimension, entity)
            if dimension == 1:
                groups.setdefault(tag, []).extend(elements)
            else:
                indices = [lookup[int(identifier)] for identifier in ids]
                if np.any((materials[indices] != 0) & (materials[indices] != tag)):
                    raise ValueError("overlapping physical surface groups are not supported")
                materials[indices] = tag
    return MeshData(mesh, _tagged_faces(mesh, groups), materials, names)


def _size(value: float) -> float:
    """Validate a positive finite requested mesh size."""
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError("mesh size must be finite and positive")
    return float(value)


def unit_square_gmsh(size: float = 0.25) -> MeshData:
    """Generate a triangular unit square with four named boundary groups.

    A temporary model is removed afterwards. A caller-owned Gmsh session stays
    initialized and its previously current model is restored, including on error.
    Gmsh's process-global API is not safe for concurrent calls from threads.
    """
    size = _size(size)
    gmsh = _optional("gmsh")
    owns_session = not gmsh.isInitialized()
    if owns_session:
        gmsh.initialize()
    previous = gmsh.model.getCurrent()
    name = f"pymhm-{uuid4().hex}"
    gmsh.model.add(name)
    try:
        vertices = [
            gmsh.model.geo.addPoint(x, y, 0, size) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))
        ]
        edges = [gmsh.model.geo.addLine(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
        loop = gmsh.model.geo.addCurveLoop(edges)
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        for tag, (edge, label) in enumerate(
            zip(edges, ("bottom", "right", "top", "left"), strict=True), 1
        ):
            gmsh.model.addPhysicalGroup(1, [edge], tag, label)
        gmsh.model.addPhysicalGroup(2, [surface], 1, "domain")
        gmsh.model.mesh.generate(2)
        return from_gmsh(gmsh.model)
    finally:
        gmsh.model.setCurrent(name)
        gmsh.model.remove()
        if previous:
            gmsh.model.setCurrent(previous)
        if owns_session:
            gmsh.finalize()


def from_netgen(source: Any) -> MeshData:
    """Copy a two-dimensional first-order Netgen mesh and boundary/material IDs."""
    if source.dim != 2:
        raise ValueError("Netgen mesh must have dimension 2")
    points = _xy([point.p for point in source.Points()])
    triangles, materials = [], []
    groups: dict[int, list[list[int]]] = {}
    for element in source.Elements2D():
        if len(element.points) != 3:
            raise ValueError("Netgen mesh must contain first-order triangles")
        triangles.append([vertex.nr - 1 for vertex in element.vertices])
        materials.append(element.index)
    mesh = TriangleMesh(points, np.asarray(triangles, dtype=np.int64))
    for element in source.Elements1D():
        if len(element.points) != 2:
            raise ValueError("Netgen mesh must contain first-order boundary lines")
        groups.setdefault(element.index, []).append([vertex.nr - 1 for vertex in element.vertices])
    names = {(1, tag): source.GetBCName(tag - 1) for tag in groups}
    names.update({(2, tag): source.GetMaterial(tag) for tag in set(materials)})
    return MeshData(mesh, _tagged_faces(mesh, groups), np.asarray(materials), names)


def unit_square_netgen(maxh: float = 0.25) -> MeshData:
    """Generate a Netgen triangular unit square with its native physical names."""
    maxh = _size(maxh)
    geometry = _optional("netgen.geom2d").unit_square
    return from_netgen(geometry.GenerateMesh(maxh=maxh))
