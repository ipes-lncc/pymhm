"""Residual-indicator mesh sizes and optional FreeFEM/BAMG remeshing.

The nodal metric follows the scalar residual-indicator construction documented
by FreeFEM. It is independent of the PDE solver and does not prescribe a bulk
marking rule or preserve nested meshes.
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from pymhm.mesh import FloatArray, TriangleMesh, positive_int


@dataclass(frozen=True)
class ResidualMeshSize:
    """Current and requested nodal sizes and the lumped indicator projection."""

    current: FloatArray
    requested: FloatArray
    projected_indicator: FloatArray
    threshold: float
    factors: FloatArray


def residual_mesh_size(
    mesh: TriangleMesh, local_squared: Any, *, coefficient: float = 1.0
) -> ResidualMeshSize:
    """Compute the documented isotropic residual metric from squared indicators.

    The current size at a vertex is the average length of its incident edges,
    counted once per incident triangle. A continuous P1 indicator is obtained
    by area-weighted lumped projection of the cellwise square roots. Divide the
    current size by ``clip(projected / (coefficient * mean(cellwise)), 1/3, 3)``.
    Thus both refinement and coarsening are allowed. A zero indicator leaves
    sizes unchanged. Unused vertices and invalid data are rejected explicitly.
    """
    values = np.asarray(local_squared)
    if (
        values.shape != (len(mesh.cells),)
        or np.iscomplexobj(values)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("provide one finite nonnegative squared indicator per cell")
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("coefficient must be finite and positive")
    current, counts = np.zeros(len(mesh.points)), np.zeros(len(mesh.points))
    for side in range(3):
        nodes = mesh.cells[:, [side, (side + 1) % 3]]
        lengths = mesh.lengths[mesh.cell_faces[:, side]]
        np.add.at(current, nodes.ravel(), np.repeat(lengths, 2))
        np.add.at(counts, nodes.ravel(), 1)
    if np.any(counts == 0):
        raise ValueError("mesh must not contain unused vertices")
    current /= counts
    indicators = np.sqrt(values.astype(float))
    weights, projected = np.zeros(len(mesh.points)), np.zeros(len(mesh.points))
    np.add.at(weights, mesh.cells.ravel(), np.repeat(mesh.areas, 3))
    np.add.at(projected, mesh.cells.ravel(), np.repeat(mesh.areas * indicators, 3))
    projected /= weights
    # Normalize before averaging to avoid an overflowing reduction of valid data.
    maximum = float(np.max(indicators))
    if maximum == 0:
        threshold, factors = 0.0, np.ones(len(mesh.points))
    else:
        scaled_threshold = coefficient * float(np.mean(indicators / maximum))
        threshold = maximum * scaled_threshold
        factors = np.clip((projected / maximum) / scaled_threshold, 1 / 3, 3)
    return ResidualMeshSize(current, current / factors, projected, threshold, factors)


def _write_mesh(path: Path, mesh: TriangleMesh) -> None:
    """Write the simple FreeFEM mesh format with an unchanged polygonal boundary."""
    boundary = mesh.faces[mesh.boundary_faces]
    lines = [f"{len(mesh.points)} {len(mesh.cells)} {len(boundary)}"]
    lines.extend(f"{x:.17g} {y:.17g} 1" for x, y in mesh.points)
    lines.extend(f"{a + 1} {b + 1} {c + 1} 1" for a, b, c in mesh.cells)
    lines.extend(f"{a + 1} {b + 1} 1" for a, b in boundary)
    path.write_text("\n".join(lines) + "\n")


def _read_mesh(path: Path) -> TriangleMesh:
    """Read FreeFEM's simple planar triangle format, validating its entity counts."""
    lines = path.read_text().splitlines()
    nv, nt, ne = (int(value) for value in lines[0].split())
    if min(nv, nt, ne) <= 0 or len(lines) != 1 + nv + nt + ne:
        raise ValueError("invalid FreeFEM output mesh entity counts")
    points = np.array([[float(v) for v in line.split()[:2]] for line in lines[1 : nv + 1]])
    cells = np.array(
        [[int(v) - 1 for v in line.split()[:3]] for line in lines[nv + 1 : nv + nt + 1]],
        dtype=np.int64,
    )
    return TriangleMesh(points, cells)


def remesh_freefem(
    mesh: TriangleMesh,
    sizes: Any,
    *,
    executable: str = "FreeFem++",
    max_vertices: int = 10000,
    timeout: float = 120.0,
) -> TriangleMesh:
    """Remesh with BAMG using a supplied continuous P1 physical size field.

    A separately installed FreeFEM executable is required. Only mesh generation
    is delegated: the call uses ``adaptmesh(IsMetric=1, splitpbedge=1)`` and does
    not solve a PDE. The requested size is interpolated in the input P1 space;
    the output need not be nested. Material interfaces and boundary data must be
    transferred by the caller's physical functions, not by cell-index ancestry.
    The unchanged domain area is checked after reading the returned mesh.
    """
    values = np.asarray(sizes)
    if (
        values.shape != (len(mesh.points),)
        or np.iscomplexobj(values)
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError("sizes must contain one finite positive value per vertex")
    count = positive_int(max_vertices, "max_vertices")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    program = shutil.which(executable)
    if program is None:
        raise ImportError("FreeFEM executable not found; install FreeFEM or pass its full path")
    with TemporaryDirectory(prefix="pymhm-metric-") as directory:
        folder = Path(directory)
        _write_mesh(folder / "input.msh", mesh)
        np.savetxt(folder / "sizes.txt", values, fmt="%.17g")
        script = (
            'verbosity=0;\nmesh Th=readmesh("input.msh");\n'
            "fespace Vh(Th,P1); Vh h; real[int] sizes(Th.nv);\n"
            'ifstream input("sizes.txt"); for(int i=0;i<Th.nv;i++) input >> sizes[i];\n'
            "for(int k=0;k<Th.nt;k++) for(int j=0;j<3;j++) "
            "h[][Vh(k,j)]=sizes[int(Th[k][j])];\n"
            f"Th=adaptmesh(Th,h,IsMetric=1,splitpbedge=1,nbvx={count});\n"
            'savemesh(Th,"output.msh");\n'
        )
        (folder / "remesh.edp").write_text(script)
        result = subprocess.run(
            [program, "-nw", "remesh.edp"],
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"FreeFEM remeshing failed: {result.stdout}\n{result.stderr}")
        refined = _read_mesh(folder / "output.msh")
    if not np.isclose(refined.areas.sum(), mesh.areas.sum(), rtol=1e-10, atol=0):
        raise ValueError("FreeFEM remeshing changed the domain area")
    return refined
