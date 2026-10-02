"""Conforming simplex partitions fitted to arbitrary affine material interfaces."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

from pymhm.mesh import FaceSpace, FloatArray, IntArray, SkeletonSpace, TriangleMesh
from pymhm.planar_material import PlanarMaterial
from pymhm.tetrahedral import TetraMesh


def _unique(points: FloatArray) -> FloatArray:
    """Merge only coordinate-roundoff copies of the same geometric vertex."""
    scale = max(float(np.max(abs(points))), float(np.ptp(points, axis=0).max()))
    pairs = cKDTree(points).query_pairs(16 * np.finfo(float).eps * scale, output_type="ndarray")
    graph = sparse.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points))
    )
    _, groups = connected_components(graph, directed=False)
    _, ids = np.unique(groups, return_index=True)
    return points[ids]


def _hull(points: FloatArray) -> ConvexHull:
    """Compute topology in translated, uniformly scaled coordinates."""
    shift = points[0]
    scale = np.ptp(points, axis=0).max()
    return ConvexHull((points - shift) / scale)


def _split(vertices: FloatArray, normal: FloatArray, offset: float) -> tuple[FloatArray, ...]:
    """Split a convex polygon/polyhedron at one physical plane with positive-volume children."""
    distance = vertices @ normal - offset
    tolerance = 16 * np.finfo(float).eps * (abs(vertices) @ abs(normal) + abs(offset))
    distance[abs(distance) <= tolerance] = 0
    if not (np.any(distance < 0) and np.any(distance > 0)):
        return (vertices,)
    hull = _hull(vertices)
    edges = {
        tuple(sorted((int(a), int(b)))) for face in hull.simplices for a, b in combinations(face, 2)
    }
    intersections = [vertices[i] for i in range(len(vertices)) if distance[i] == 0]
    for a, b in sorted(edges):
        if distance[a] * distance[b] < 0:
            t = distance[a] / (distance[a] - distance[b])
            intersections.append(vertices[a] + t * (vertices[b] - vertices[a]))
    cut = np.asarray(intersections)
    children = []
    for sign in (-1, 1):
        points = _unique(np.vstack((vertices[sign * distance > 0], cut)))
        children.append(points[_hull(points).vertices])
    return tuple(children)


def _triangles(vertices: FloatArray) -> tuple[FloatArray, ...]:
    """Triangulate a planar convex polygon using a geometry-canonical boundary fan."""
    if vertices.shape[1] == 3:
        _, _, frame = np.linalg.svd(vertices - vertices[0], full_matrices=False)
        coordinates = (vertices - vertices[0]) @ frame[:2].T
    else:
        coordinates = vertices - vertices[0]
    indices = ConvexHull(coordinates / np.ptp(coordinates, axis=0).max()).vertices
    ordered = vertices[indices]
    candidates = np.arange(len(ordered))
    envelope = 16 * np.finfo(float).eps * max(np.max(abs(ordered)), np.ptp(ordered, axis=0).max())
    for axis in range(ordered.shape[1]):
        lower = ordered[candidates, axis].min()
        candidates = candidates[ordered[candidates, axis] <= lower + envelope]
    first = int(candidates[0])
    ordered = np.roll(ordered, -first, axis=0)
    return tuple(ordered[[0, i, i + 1]] for i in range(1, len(ordered) - 1))


def _simplices(vertices: FloatArray) -> tuple[FloatArray, ...]:
    """Cone canonical polygonal boundary triangles to a convex polyhedron's interior."""
    if vertices.shape[1] == 2:
        return _triangles(vertices)
    hull = _hull(vertices)
    used: set[int] = set()
    center = vertices.mean(axis=0)
    result: list[FloatArray] = []
    for i, equation in enumerate(hull.equations):
        if i in used:
            continue
        same = np.max(abs(hull.equations - equation), axis=1) < 256 * np.finfo(float).eps
        group = np.flatnonzero(same)
        used.update(int(j) for j in group)
        face = vertices[np.unique(hull.simplices[group])]
        result.extend(np.vstack((center, triangle)) for triangle in _triangles(face))
    return tuple(result)


@dataclass(frozen=True)
class PlanarFittedMesh:
    """Material-fitted simplex mesh with exact original-cell and material-region ownership."""

    mesh: TriangleMesh | TetraMesh
    parents: IntArray
    regions: IntArray
    material: PlanarMaterial

    @property
    def tensors(self) -> FloatArray:
        """One physical permeability tensor per fitted simplex, including background."""
        return self.material.tensors[self.regions]


def fit_planar_material(
    mesh: TriangleMesh | TetraMesh, material: PlanarMaterial
) -> PlanarFittedMesh:
    """Fit a conforming simplex mesh to all planes defining a material.

    Planes are extended across each cell before triangulation, so intersections
    match on incident cells. Canonical face fans preserve conformity in 3D.
    This operation enriches geometry; it does not automatically enrich a MHM
    trace. Its original-cell ancestry lets callers retain their macro mesh.
    Geometric predicates use a coordinate-roundoff envelope, and independent
    parent-volume and interior-face checks reject unresolved output geometry.
    """
    if (
        not isinstance(mesh, (TriangleMesh, TetraMesh))
        or mesh.points.shape[1] != material.dimension
    ):
        raise ValueError("material fitting requires a simplex mesh of matching dimension")
    vertices: list[FloatArray] = []
    parents: list[int] = []
    regions: list[int] = []
    normals, offsets = material.planes
    for cell, ids in enumerate(mesh.cells):
        pieces = [mesh.points[ids]]
        for normal, offset in zip(normals, offsets, strict=True):
            pieces = [part for piece in pieces for part in _split(piece, normal, float(offset))]
        for piece in pieces:
            region = int(material.region_ids(piece.mean(axis=0)[None])[0])
            simplices = _simplices(piece)
            vertices.extend(simplices)
            parents.extend([cell] * len(simplices))
            regions.extend([region] * len(simplices))
    raw = np.concatenate(vertices)
    points = _unique(raw)
    distances, indices = cKDTree(points).query(raw)
    envelope = 16 * np.finfo(float).eps * max(np.max(abs(raw)), np.ptp(raw, axis=0).max())
    if np.any(distances > envelope):
        raise ValueError("material intersections are not resolvable at the coordinate scale")
    cells = indices.reshape(-1, material.dimension + 1)
    result = type(mesh)(points, cells)
    original = mesh.areas if isinstance(mesh, TriangleMesh) else mesh.volumes
    measure = result.areas if isinstance(result, TriangleMesh) else result.volumes
    accumulated = np.bincount(parents, weights=measure, minlength=len(mesh.cells))
    # The tolerance accounts for the input coordinate scale, independently of
    # material contrast; permeability never changes a geometric acceptance test.
    condition = np.array(
        [np.linalg.cond((mesh.points[c[1:]] - mesh.points[c[0]]).T) for c in mesh.cells]
    )
    scale = max(1.0, np.max(abs(mesh.points)) / np.ptp(mesh.points, axis=0).max())
    if np.any(
        abs(accumulated - original) > 2048 * np.finfo(float).eps * condition * scale * original
    ):
        raise ValueError("material fitting did not preserve the original cell volumes")
    # An exposed child face must lie on an original exterior plane/segment.
    # This detects inconsistent triangulations rather than relying on total volume.
    for face in result.boundary_faces:
        nodes = result.points[result.faces[face]]
        found = False
        for parent in mesh.boundary_faces:
            base = mesh.points[mesh.faces[parent, 0]]
            normal = mesh.normals[parent]
            bound = (
                128
                * np.finfo(float).eps
                * max(np.max(abs(nodes)), np.max(abs(base)), np.ptp(mesh.points, axis=0).max())
            )
            if np.max(abs((nodes - base) @ normal)) <= bound:
                found = True
                break
        if not found:
            raise ValueError("material fitting produced an unmatched interior face")
    return PlanarFittedMesh(
        result, np.asarray(parents, dtype=np.int64), np.asarray(regions, dtype=np.int64), material
    )


def fit_planar_skeleton(
    mesh: TriangleMesh, material: PlanarMaterial, *, degree: int = 0
) -> SkeletonSpace:
    """Insert exact interface intersections into every two-dimensional macroface."""
    if material.dimension != 2:
        raise ValueError("edge partitions require a two-dimensional material")
    faces = []
    normals, offsets = material.planes
    for ids in mesh.faces:
        a, b = mesh.points[ids]
        denominators = normals @ (b - a)
        intersections = np.divide(
            offsets - normals @ a,
            denominators,
            out=np.full(len(offsets), np.inf),
            where=denominators != 0,
        )
        values = np.sort(np.r_[0.0, intersections[(intersections > 0) & (intersections < 1)], 1.0])
        distinct = np.r_[True, np.diff(values) > 32 * np.finfo(float).eps]
        breaks = tuple(float(v) for v in values[distinct])
        faces.append(FaceSpace(breaks, (degree,) * (len(breaks) - 1)))
    return SkeletonSpace(mesh, tuple(faces))


def planar_face_partitions(mesh: TetraMesh, material: PlanarMaterial) -> tuple[FloatArray, ...]:
    """Partition each original triangular macroface at all material-plane intersections.

    Each returned array contains subtriangle vertices in the original face's
    barycentric coordinates. The same physical boundary fan as volume fitting
    fixes its diagonals. These partitions describe actual unequal subface areas;
    consumers must integrate using their area fractions, not a uniform count.
    """
    if material.dimension != 3:
        raise ValueError("triangular face partitions require a three-dimensional material")
    normals, offsets = material.planes
    reference = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = []
    for face in mesh.faces:
        vertices = mesh.points[face]
        tangent = (vertices[1:] - vertices[0]).T
        restricted, values = normals @ tangent, offsets - normals @ vertices[0]
        pieces = [reference]
        for normal, value in zip(restricted, values, strict=True):
            if np.linalg.norm(normal) > 32 * np.finfo(float).eps * np.linalg.norm(tangent):
                pieces = [part for piece in pieces for part in _split(piece, normal, float(value))]
        parts = []
        for piece in pieces:
            physical = vertices[0] + piece @ tangent.T
            for triangle in _triangles(physical):
                coordinates = (triangle - vertices[0]) @ np.linalg.pinv(tangent).T
                parts.append(np.column_stack((1 - coordinates.sum(axis=1), coordinates)))
        result.append(np.asarray(parts))
    return tuple(result)
