"""Globally continuous scalar pressure traces on planar macro edges.

Vertices shared by different macrofaces share coordinates. This differs from
facewise C0 multiplier spaces, whose separate macrofaces remain independent.
"""

from dataclasses import dataclass, field

import numpy as np

from pymhm.core.validation import FloatArray, IntArray
from pymhm.fem.traces.interval import FaceSpace
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class PressureTraceSpace:
    """Globally continuous nodal polynomials on triangular or polygonal edges.

    ``faces`` may prescribe different positive degrees and segment partitions.
    Endpoints of different macrofaces share the mesh vertex unknown. Interior
    edge nodes belong to that face only. This is Gamma, not the flux skeleton.
    """

    mesh: TriangleMesh | PolygonMesh
    faces: tuple[FaceSpace, ...] = ()
    nodes: FloatArray = field(init=False, repr=False)
    face_dofs: tuple[IntArray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Assign shared vertices and private face-interior interpolation nodes."""
        if not isinstance(self.mesh, (TriangleMesh, PolygonMesh)):
            raise TypeError("pressure traces require a TriangleMesh or PolygonMesh")
        faces = self.faces
        faces = (
            tuple(FaceSpace.uniform(1, continuous=True) for _ in self.mesh.faces)
            if not faces
            else tuple(faces)
        )
        if len(faces) != len(self.mesh.faces) or any(
            not isinstance(f, FaceSpace) or not f.continuous for f in faces
        ):
            raise ValueError(
                "Gamma requires one continuous positive-degree FaceSpace per macroface"
            )
        points, dofs = list(self.mesh.points), []
        for edge, space in zip(self.mesh.faces, faces, strict=True):
            t = np.r_[
                space.breaks,
                np.concatenate(
                    [
                        a + (b - a) * np.arange(1, k) / k
                        for a, b, k in zip(
                            space.breaks[:-1], space.breaks[1:], space.degrees, strict=True
                        )
                    ]
                ),
            ]
            ids = np.empty(space.size, dtype=np.int64)
            ids[0], ids[len(space.breaks) - 1] = edge
            for j in range(space.size):
                if j not in (0, len(space.breaks) - 1):
                    ids[j] = len(points)
                    points.append(
                        (1 - t[j]) * self.mesh.points[edge[0]] + t[j] * self.mesh.points[edge[1]]
                    )
            dofs.append(ids)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "nodes", np.asarray(points))
        object.__setattr__(self, "face_dofs", tuple(dofs))

    @classmethod
    def uniform(
        cls, mesh: TriangleMesh | PolygonMesh, degree: int = 1, segments: int = 1
    ) -> "PressureTraceSpace":
        """Use the same continuous nodal degree and subdivision on every face."""
        face = FaceSpace.uniform(degree, segments, continuous=True)
        return cls(mesh, tuple(face for _ in mesh.faces))

    @property
    def size(self) -> int:
        """Return the number of shared pressure-trace unknowns before boundary elimination."""
        return len(self.nodes)

    def cell_dofs(self, cell: int) -> IntArray:
        """Return unique Gamma indices touching one macrocell, in increasing order."""
        return np.unique(np.concatenate([self.face_dofs[f] for f in self.mesh.cell_faces[cell]]))
