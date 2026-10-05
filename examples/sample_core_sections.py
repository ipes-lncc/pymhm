"""Sample full one-sided polynomials on physical sections of the 3D capability cases.

Every display triangle owns its vertices. Interpolation for display therefore
stays within one fine cell and never averages across a material or macro face.
Physical error checks integrate volume polynomials independently of the plot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull
from threadpoolctl import threadpool_limits

from examples import core_extension_data as exact
from examples.hdiv3d_field_archive import read_field as read_hdiv_field
from examples.hdiv3d_sections import replay_section as replay_hdiv_section
from examples.mshho3d_field_archive import read_field
from examples.mshho3d_sections import replay_section
from examples.verify_hdiv3d import physical_errors as hdiv_physical_errors
from examples.verify_mshho3d import physical_errors
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/core-extensions"
HEIGHT = 0.37


def digest(path: Path) -> str:
    """Identify a numerical input, executed source or generated field archive."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cut_polygon(vertices: np.ndarray, height: float = HEIGHT) -> np.ndarray:
    """Intersect a convex affine fine cell with a horizontal plane, in cyclic order."""
    cut = [point for point in vertices if point[2] == height]
    for a, b in combinations(vertices, 2):
        if (a[2] - height) * (b[2] - height) < 0:
            cut.append(a + (height - a[2]) / (b[2] - a[2]) * (b - a))
    if len(cut) < 3:
        return np.empty((0, 3))
    points = np.unique(np.asarray(cut), axis=0)
    if len(points) < 3 or np.linalg.matrix_rank(points[:, :2] - points[0, :2]) < 2:
        return np.empty((0, 3))
    return points[ConvexHull(points[:, :2]).vertices]


def section_samples(
    meshes: tuple[Any, ...],
    evaluate: Callable[[int, int, np.ndarray], np.ndarray],
    refinement: int,
) -> dict[str, np.ndarray]:
    """Evaluate full fields on independent refined fans of each fine-cell section."""
    reference = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, refinement)
    bary = np.column_stack((1 - reference.points.sum(axis=1), reference.points))
    points, cells, values, owners = [], [], [], []
    count = 0
    for macro, fine in enumerate(meshes):
        for cell, vertices in enumerate(fine.points[fine.cells]):
            polygon = cut_polygon(vertices)
            if not len(polygon):
                continue
            center = polygon.mean(axis=0)
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
                physical = bary @ np.array([center, a, b])
                points.append(physical)
                cells.append(reference.cells + count)
                values.append(evaluate(macro, cell, physical))
                owners.extend([(macro, cell)] * len(physical))
                count += len(physical)
    xyz = np.concatenate(points)
    return dict(
        points=xyz,
        cells=np.concatenate(cells),
        owners=np.asarray(owners, dtype=np.int64),
        actual=np.concatenate(values),
        exact=np.column_stack((exact.pressure3d(xyz), exact.flux3d(xyz))),
    )


def mixed_fields(
    archive: dict[str, np.ndarray], kind: str, refinement: int, order: int
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Use literal producer C/T/DOFs and the acquisition's terminal physical norm rule."""
    declared = ("tetrahedron", "prism")[int(archive["cell_kind_code"])]
    if kind != declared or order != int(archive["norm_orders"][-1]):
        raise ValueError("Declared geometry and terminal executed norm rule are required")
    return replay_hdiv_section(archive, refinement), hdiv_physical_errors(archive, order)


def mshho_fields(
    name: str, resolution: int, refinement: int, *, archive: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    """Replay sections and the terminal executed norm rule used by the acquisition."""
    if (
        name not in {"tetra-p0", "cube-p0"}
        or type(resolution) is not int
        or resolution < 1
        or archive.get("display_refinement") != refinement
    ):
        raise ValueError("Current acquired case and its executed section refinement required")
    order = int(archive["norm_orders"][-1])
    return replay_section(archive), physical_errors(archive, order), {}


def run(suite: str, refinement: int = 6) -> None:
    """Publish display samples only after coefficient replay reproduces the acquired norms."""
    record_path = DATA / f"{suite}.json"
    record = json.loads(record_path.read_text())
    names = [
        "fem/hdiv/family_3d",
        "fem/hdiv/moments_3d",
        "meshes/mixed",
        "fem/scalar/tetrahedron",
        "fem/scalar/tetrahedron_topology",
    ]
    if suite == "mshho3d":
        names += [
            "methods/hho",
            "methods/hho_3d",
            "_legacy/models/darcy/primal_3d",
            "meshes/polyhedral",
            "_legacy/models/transport/polyhedral",
            "core/contracts",
            "linalg/linear",
        ]
    paths = [ROOT / f"src/pymhm/{name}.py" for name in names] + [
        Path(__file__),
        ROOT / "examples/core_extension_data.py",
    ]
    if suite == "hdiv3d":
        paths += [
            ROOT / "examples/hdiv3d_field_archive.py",
            ROOT / "examples/hdiv3d_sections.py",
            ROOT / "examples/verify_hdiv3d.py",
            ROOT / "examples/archive_precision.py",
        ]
    hashes = current_source_manifest({p.relative_to(ROOT).as_posix(): digest(p) for p in paths})
    report = dict(
        suite=suite,
        section_height=HEIGHT,
        sampling_refinement=refinement,
        sampling=(
            "Full polynomial evaluation on private fine-cell section vertices; "
            "no averaging across interfaces"
        ),
        norm_check_rtol=2e-10,
        norm_check_atol=2e-13,
        acquisition_record_sha256=digest(record_path),
        source_sha256=hashes,
        cases={},
    )
    for name, original in record["fields"].items():
        path = DATA / original["archive"]
        if digest(path) != original["sha256"]:
            raise ValueError("a numerical input differs from its acquisition digest")
        if suite == "mshho3d":
            archive, _ = read_field(path)
        else:
            archive, _ = read_hdiv_field(path)
        row = [r for r in record["rows"] if r["case"] == name][-1]
        if suite == "hdiv3d":
            fields, errors = mixed_fields(
                archive,
                "prism" if name.startswith("prism") else "tetrahedron",
                refinement,
                int(archive["norm_orders"][-1]),
            )
            coefficients = {}
        else:
            fields, errors, coefficients = mshho_fields(
                name, row["resolution"], refinement, archive=archive
            )
        for key, value in errors.items():
            if not np.isclose(value, row[key], rtol=2e-10, atol=2e-13):
                raise ArithmeticError(
                    f"{name}: replayed {key} does not reproduce the acquired norm"
                )
        output = DATA / f"{name}-polynomial-fields.npz"
        np.savez_compressed(output, **fields, **coefficients)
        info = dict(
            archive=output.name,
            sha256=digest(output),
            original_archive=path.relative_to(DATA).as_posix(),
            original_sha256=digest(path),
            resolution=row["resolution"],
            **errors,
        )
        info["norm_order"] = int(archive["norm_orders"][-1])
        report["cases"][name] = info
        print(json.dumps({name: info}), flush=True)
    report["source_changed"] = hashes != current_source_manifest(
        {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    )
    if report["source_changed"]:
        raise RuntimeError("a polynomial replay source changed during acquisition")
    (DATA / f"{suite}-field-sampling.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("hdiv3d", "mshho3d"))
    parser.add_argument("--refinement", type=int, default=6)
    arguments = parser.parse_args()
    with threadpool_limits(1):
        run(arguments.suite, arguments.refinement)
