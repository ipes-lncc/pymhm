"""Lazy native Gmsh and Netgen importers for supported three-dimensional geometry."""

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np

from pymhm.io.planar import _optional, _size
from pymhm.io.volume import VolumeMeshData, from_meshio


def from_gmsh_3d(model: Any | None = None) -> VolumeMeshData:
    """Copy first-order tetrahedra, wedges or hexahedra from a live Gmsh model.

    Noncontiguous node/element tags and integer volume/surface physical IDs are
    resolved explicitly. Overlapping physical groups with conflicting IDs are
    rejected. Wedges must be affine; hexahedra use the native trilinear validity
    checks. A single volume family is required. The model/session is not modified.
    """
    if model is None:
        gmsh = _optional("gmsh")
        if not gmsh.isInitialized():
            raise RuntimeError("initialize Gmsh before importing its current model")
        model = gmsh.model
    if model.getDimension() != 3:
        raise ValueError("Gmsh model must have dimension three")
    node_tags, coordinates, _ = model.mesh.getNodes()
    nodes = {int(tag): index for index, tag in enumerate(node_tags)}
    definitions = {
        4: ("tetra", 4),
        5: ("hexahedron", 8),
        6: ("wedge", 6),
        2: ("triangle", 3),
        3: ("quad", 4),
    }
    blocks: list[Any] = []
    physical: list[Any] = []
    lookup: dict[int, tuple[int, int]] = {}
    for dimension in (3, 2):
        types, elements, connectivity = model.mesh.getElements(dimension)
        for kind, identifiers, flat in zip(types, elements, connectivity, strict=True):
            if kind not in definitions or (dimension == 3) != (kind in (4, 5, 6)):
                raise ValueError("Gmsh volume import requires supported first-order cells/faces")
            name, width = definitions[kind]
            try:
                cells = np.array([nodes[int(tag)] for tag in flat], dtype=np.int64).reshape(
                    -1, width
                )
            except (KeyError, ValueError) as exc:
                raise ValueError("Gmsh element connectivity contains invalid node tags") from exc
            lookup.update({int(tag): (len(blocks), index) for index, tag in enumerate(identifiers)})
            blocks.append(SimpleNamespace(type=name, data=cells))
            physical.append(np.zeros(len(cells), dtype=np.int64))
    for dimension, tag in model.getPhysicalGroups():
        if dimension not in (2, 3):
            continue
        for entity in model.getEntitiesForPhysicalGroup(dimension, tag):
            _, groups, _ = model.mesh.getElements(dimension, entity)
            for identifiers in groups:
                for identifier in identifiers:
                    block, index = lookup[int(identifier)]
                    if physical[block][index] and physical[block][index] != tag:
                        raise ValueError("conflicting Gmsh physical groups on one element")
                    physical[block][index] = tag
    return from_meshio(
        SimpleNamespace(
            points=np.asarray(coordinates).reshape(-1, 3),
            cells=blocks,
            point_data={},
            cell_data={"gmsh:physical": physical},
        )
    )


def unit_cube_gmsh(size: float = 0.35) -> VolumeMeshData:
    """Generate a tetrahedral unit cube, material one and side IDs x0/x1/y0/y1/z0/z1=1..6.

    The function owns only its temporary model. An existing Gmsh session and
    previously current model are preserved, including when import raises.
    """
    size = _size(size)
    gmsh = _optional("gmsh")
    owns_session = not gmsh.isInitialized()
    if owns_session:
        gmsh.initialize()
    previous = gmsh.model.getCurrent()
    name = f"pymhm-volume-{uuid4().hex}"
    gmsh.model.add(name)
    try:
        volume = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), size)
        gmsh.model.addPhysicalGroup(3, [volume], 1, "domain")
        for _, surface in gmsh.model.getEntities(2):
            center = np.asarray(gmsh.model.occ.getCenterOfMass(2, surface))
            axis = int(np.argmax(abs(center - 0.5)))
            side = int(center[axis] > 0.5)
            gmsh.model.addPhysicalGroup(2, [surface], 1 + 2 * axis + side, f"{'xyz'[axis]}{side}")
        gmsh.model.mesh.generate(3)
        return from_gmsh_3d(gmsh.model)
    finally:
        gmsh.model.setCurrent(name)
        gmsh.model.remove()
        if previous:
            gmsh.model.setCurrent(previous)
        if owns_session:
            gmsh.finalize()


def from_netgen_3d(source: Any) -> VolumeMeshData:
    """Copy first-order Netgen tetrahedra and face-descriptor boundary/material IDs.

    ``Element2D.index`` identifies a Netgen face descriptor; its ``bc`` value,
    rather than the descriptor index, is the exported physical boundary ID.
    Higher-order volume/surface geometry is rejected instead of dropping nodes.
    """
    if source.dim != 3:
        raise ValueError("Netgen mesh must have dimension three")
    points = np.array([point.p for point in source.Points()])
    volumes, materials, surfaces, tags = [], [], [], []
    for element in source.Elements3D():
        if len(element.points) != 4:
            raise ValueError("Netgen volume import requires first-order tetrahedra")
        volumes.append([vertex.nr - 1 for vertex in element.vertices])
        materials.append(element.index)
    for element in source.Elements2D():
        if len(element.points) != 3:
            raise ValueError("Netgen boundary import requires first-order triangles")
        surfaces.append([vertex.nr - 1 for vertex in element.vertices])
        tags.append(source.FaceDescriptor(element.index).bc)
    return from_meshio(
        SimpleNamespace(
            points=points,
            point_data={},
            cells=[
                SimpleNamespace(type="tetra", data=np.asarray(volumes, dtype=np.int64)),
                SimpleNamespace(type="triangle", data=np.asarray(surfaces, dtype=np.int64)),
            ],
            cell_data={
                "physical_tag": [
                    np.asarray(materials, dtype=np.int64),
                    np.asarray(tags, dtype=np.int64),
                ]
            },
        )
    )


def unit_cube_netgen(maxh: float = 0.35) -> VolumeMeshData:
    """Generate a Netgen tetrahedral unit cube and preserve its native integer IDs."""
    return from_netgen_3d(_optional("netgen.csg").unit_cube.GenerateMesh(maxh=_size(maxh)))
