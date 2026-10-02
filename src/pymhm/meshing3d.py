"""Optional meshio exchange of linear tetrahedra and integer material/boundary tags."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pymhm.mesh import IntArray
from pymhm.meshing import _data, _optional
from pymhm.tetrahedral import TetraMesh


def _tags(value: Any, size: int) -> IntArray:
    """Copy one nonnegative integer physical ID per entity, defaulting to zero."""
    if value is None:
        return np.zeros(size, dtype=np.int64)
    raw = np.asarray(value)
    if raw.shape != (size,) or not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0):
        raise ValueError("physical tags must be nonnegative integers, one per entity")
    return np.array(raw, dtype=np.int64, copy=True)


@dataclass
class TetraMeshData:
    """Linear tetrahedral mesh with integer physical IDs and point/volume fields.

    ``face_tags`` includes interior interfaces as well as external boundaries.
    Physical name strings and arbitrary surface fields are outside this format.
    """

    mesh: TetraMesh
    cell_tags: Any = None
    face_tags: Any = None
    point_data: dict[str, Any] = field(default_factory=dict)
    cell_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate lengths, physical IDs and real finite user fields."""
        self.cell_tags = _tags(self.cell_tags, len(self.mesh.cells))
        self.face_tags = _tags(self.face_tags, len(self.mesh.faces))
        self.point_data = _data(self.point_data, len(self.mesh.points))
        self.cell_data = _data(self.cell_data, len(self.mesh.cells))
        if "physical_tag" in self.point_data or "physical_tag" in self.cell_data:
            raise ValueError("physical_tag is reserved for integer physical IDs")


def read_tetra_mesh(path: str | Path, *, file_format: str | None = None) -> TetraMeshData:
    """Read linear tetrahedra and tagged triangular faces without changing geometry.

    Gmsh ``gmsh:physical`` IDs and the portable ``physical_tag`` field are
    accepted. Hexahedra, prisms, polyhedra and curved/high-order geometry are
    rejected. Lines and vertices may accompany a tetrahedral mesh but are not
    converted into volume elements.
    """
    source = _optional("meshio").read(path, file_format=file_format)
    tetrahedra, tags = [], []
    triangles: list[tuple[Any, Any]] = []
    fields: dict[str, list[Any]] = {}
    physical = source.cell_data.get("gmsh:physical", source.cell_data.get("physical_tag"))
    for index, block in enumerate(source.cells):
        if block.type not in ("tetra", "triangle", "line", "vertex"):
            raise ValueError(f"unsupported 3D cell type {block.type!r}; use linear tetrahedra")
        marked = _tags(None if physical is None else physical[index], len(block.data))
        if block.type == "tetra":
            tetrahedra.append(block.data)
            tags.append(marked)
            for name, values in source.cell_data.items():
                if name not in ("gmsh:physical", "gmsh:geometrical", "physical_tag"):
                    fields.setdefault(name, []).append(values[index])
        elif block.type == "triangle":
            triangles.extend(zip(block.data, marked, strict=True))
    if not tetrahedra:
        raise ValueError("file contains no tetrahedra")
    mesh = TetraMesh(source.points, np.concatenate(tetrahedra))
    lookup = {tuple(face): index for index, face in enumerate(mesh.faces)}
    face_tags = np.zeros(len(mesh.faces), dtype=np.int64)
    for vertices, tag in triangles:
        key = tuple(sorted(vertices))
        if key not in lookup:
            raise ValueError("tagged triangle is not a tetrahedral face")
        face = lookup[key]
        if face_tags[face] and face_tags[face] != tag:
            raise ValueError("conflicting physical IDs on the same triangular face")
        face_tags[face] = tag
    return TetraMeshData(
        mesh,
        np.concatenate(tags),
        face_tags,
        source.point_data,
        {name: np.concatenate(values) for name, values in fields.items()},
    )


def write_tetra_mesh(
    path: str | Path, data: TetraMeshData | TetraMesh, *, file_format: str | None = None
) -> None:
    """Write tetrahedral geometry, tagged triangles and finite nodal/volume fields.

    The VTK/VTU ``physical_tag`` arrays preserve integer physical IDs. Formats
    requiring Gmsh-specific model entities can instead be prepared directly
    through meshio; this writer does not manufacture CAD entity semantics.
    """
    data = TetraMeshData(data) if isinstance(data, TetraMesh) else data
    selected = np.flatnonzero(data.face_tags)
    cells = [("tetra", data.mesh.cells)]
    cell_data = {name: [values] for name, values in data.cell_data.items()}
    cell_data["physical_tag"] = [data.cell_tags]
    if len(selected):
        cells.append(("triangle", data.mesh.faces[selected]))
        cell_data["physical_tag"].append(data.face_tags[selected])
        for name, values in data.cell_data.items():
            cell_data[name].append(np.zeros((len(selected), *values.shape[1:])))
    _optional("meshio").write_points_cells(
        path,
        data.mesh.points,
        cells,
        point_data=data.point_data,
        cell_data=cell_data,
        file_format=file_format,
    )
