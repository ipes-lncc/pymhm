"""Compare unscaled L11 absolute errors with independently extracted graphical intervals."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.transport_face_resolution import local_bound
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/transport"
FIGURES = ROOT / "docs/figures/transport"


def digest(path: Path) -> str:
    """Identify the bytes actually used by this numerical comparison."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verify field bytes against one already loaded, internally consistent record."""
    for row in rows:
        expected = row.get("archive_sha256", row.get("fields_sha256", row.get("sha256")))
        if expected is None or digest(DATA / row["archive"]) != expected:
            raise ValueError(f"field provenance differs for {row['archive']}")
    return rows


def checked_rows(path: Path, key: str = "records") -> list[dict[str, Any]]:
    """Read a current record only after verifying every persisted field digest."""
    return _checked_records(json.loads(path.read_text())[key])


def checked_endpoint(path: Path) -> dict[str, Any]:
    """Require a single r512 field on the unchanged s16, epsilon-one trace space."""
    report = json.loads(path.read_text())
    expected = dict(
        epsilon=1.0,
        macro_resolution=8,
        local_refinement=512,
        local_degree=1,
        trace_degree=0,
        prepared_segments=16,
    )
    rows = _checked_records(report["records"])
    if (
        any(report.get(key) != value for key, value in expected.items())
        or len(rows) != 1
        or rows[0].get("local_refinement") != 512
        or rows[0].get("epsilon") != 1.0
        or rows[0].get("segments") != 16
        or rows[0].get("free_trace_dofs") != 5888
        or rows[0].get("retained_constant_coordinates") != 240
        or report.get("source_changed_during_run") is not False
        or rows[0].get("source_changed_during_run") is not False
        or set(rows[0].get("quadrature", {})) != {"8", "12"}
        or set(report.get("gradient_dg0_projection_error", {})) != {"8", "12"}
    ):
        raise ValueError("the r512 endpoint must preserve the s16 trace and both norm rules")
    return report


def compare_rows(rows: list[dict[str, Any]], published: dict, method: str) -> list[dict]:
    """Associate only DOF counts inside original horizontal graphical intervals."""
    comparisons = []
    for row in rows:
        for panel, key in zip(published["panels"], ("l2_error", "broken_h1_error"), strict=True):
            series = next(s for s in panel["series"] if s["method"] == method)
            matches = [
                point
                for point in series["points"]
                if point["N1_graphical_interval"][0]
                <= row["free_trace_dofs"]
                <= point["N1_graphical_interval"][1]
            ]
            if len(matches) > 1:
                raise ValueError("ambiguous published abscissa association")
            if not matches:
                continue
            point = matches[0]
            value = row["quadrature"]["12"][key]
            comparisons.append(
                dict(
                    archive=row["archive"],
                    archive_sha256=row.get(
                        "archive_sha256", row.get("fields_sha256", row.get("sha256"))
                    ),
                    free_trace_dofs=row["free_trace_dofs"],
                    norm=key,
                    computed=value,
                    published=point["value"],
                    published_interval=point["graphical_interval"],
                    ratio_to_marker=value / point["value"],
                    inside_graphical_interval=point["graphical_interval"][0]
                    <= value
                    <= point["graphical_interval"][1],
                    visibility=point["visibility"],
                )
            )
    return comparisons


def coarse_gradient_bound() -> Path:
    """Recompute the finite-r16 P1 lower bound on the actual archived fine cells."""
    path = DATA / "mixed-face-uniform-e1-r16.json"
    row = checked_rows(path)[-1]
    with np.load(DATA / row["archive"]) as data:
        meshes = [
            TriangleMesh(points, cells)
            for points, cells in zip(data["local_points"], data["local_cells"], strict=True)
        ]
    with threadpool_limits(1):
        norms = {
            str(order): float(np.sqrt(sum(local_bound(mesh, order=order) for mesh in meshes)))
            for order in (8, 12)
        }
    report = dict(
        local_refinement=16,
        gradient_dg0_projection_error=norms,
        archive=row["archive"],
        archive_sha256=row["archive_sha256"],
        source_hashes=current_source_manifest(
            {
                p: digest(ROOT / p)
                for p in (
                    "examples/verify_transport_published.py",
                    "examples/transport_trace_family.py",
                    "examples/transport_face_resolution.py",
                    "src/pymhm/fem/scalar/operators.py",
                    "src/pymhm/meshes/triangle.py",
                )
            }
        ),
    )
    output = DATA / "mixed-gradient-bound-r16.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    return output


def main() -> None:
    """Require completed controls and preserve every input and primary-source digest."""
    primary = DATA / "published-figure-12.json"
    published = json.loads(primary.read_text())
    nominal_path = DATA / "mixed-campaign.json"
    nominal = checked_rows(nominal_path, "spatial")
    coefficient_path = DATA / "mixed-coefficient-e1.json"
    coefficient = checked_rows(coefficient_path)
    if [row["macro_resolution"] for row in coefficient] != [8, 16, 32, 64, 128]:
        raise ValueError("the five declared coefficient controls must be complete")
    sources = [primary, nominal_path, coefficient_path]
    face_rows, bounds = {}, {}
    refinements = (16, 32, 64, 128)
    if (DATA / "mixed-face-uniform-e1-r256.json").exists():
        refinements += (256,)
    for refinement in refinements:
        path = DATA / f"mixed-face-uniform-e1-r{refinement}.json"
        rows = checked_rows(path)
        if [row["free_trace_dofs"] for row in rows] != [368 * s for s in (1, 2, 4, 8, 16)]:
            raise ValueError("a uniform-face local-resolution sequence is incomplete")
        sources.append(path)
        face_rows[str(refinement)] = compare_rows(rows, published, "space")
        if refinement != 16:
            bounds[str(refinement)] = json.loads(path.read_text())["gradient_dg0_projection_error"]
    endpoint_path = DATA / "mixed-face-endpoint-e1-r512.json"
    endpoint_rows = []
    if endpoint_path.exists():
        endpoint = checked_endpoint(endpoint_path)
        endpoint_rows = compare_rows(endpoint["records"], published, "space")
        bounds["512"] = endpoint["gradient_dg0_projection_error"]
        sources.append(endpoint_path)
    adaptive_path = DATA / "mixed-face-adaptive-e1-r16.json"
    adaptive = checked_rows(adaptive_path)
    sources.append(adaptive_path)
    bound_path = coarse_gradient_bound()
    sources.append(bound_path)
    bounds["16"] = json.loads(bound_path.read_text())["gradient_dg0_projection_error"]
    fixed_controls = [adaptive[-1]]
    with np.load(DATA / adaptive[-1]["archive"]) as data:
        geometry = {
            k: data[k].copy()
            for k in ("macro_points", "macro_cells", "trace_breaks", "trace_break_offsets")
        }
    for refinement in (32, 64, 128):
        path = DATA / f"mixed-adaptive-fixed-e1-r{refinement}.json"
        row = json.loads(path.read_text())
        if digest(DATA / row["archive"]) != row["archive_sha256"]:
            raise ValueError("fixed-adaptive local control archive differs")
        with np.load(DATA / row["archive"]) as data:
            if any(not np.array_equal(data[k], value) for k, value in geometry.items()):
                raise ValueError("a local control changed the macrogeometry or skeletal space")
        fixed_controls.append(row)
        sources.append(path)
    report = dict(
        doi="10.1137/130938499",
        figure=12,
        printed_epsilon=0.1,
        comparison_epsilon=1.0,
        coefficient_scope=(
            "The epsilon=1 regime is explicitly published in Figure7. Its compatibility "
            "with Figure12 markers does not establish the historical inputs or an editorial cause."
        ),
        norm_convention="Absolute L2 and unweighted broken H1 seminorm; no rescaling",
        dof_convention=(
            "Computed abscissae count free interior multipliers only. Constant retained "
            "coordinates are algebraic and are recorded separately in each acquisition."
        ),
        association="Computed count must lie inside the original graphical abscissa interval",
        uncertainty="Graphical reading intervals only; not certified discretization-error bounds",
        nominal_mesh=compare_rows(nominal, published, "mesh"),
        coefficient_mesh=compare_rows(coefficient, published, "mesh"),
        uniform_face_controls=face_rows,
        finest_trace_endpoint=endpoint_rows,
        gradient_projection_lower_bounds=bounds,
        fixed_adaptive_local_controls=[
            {
                key: row[key]
                for key in (
                    "archive",
                    "archive_sha256",
                    "local_refinement",
                    "free_trace_dofs",
                    "quadrature",
                )
            }
            for row in fixed_controls
        ],
        literal_marking_control=dict(
            theta=0.75,
            epsilon=1,
            local_refinement=16,
            free_trace_dofs=[row["free_trace_dofs"] for row in adaptive],
            maximum_segments=[row["maximum_segments"] for row in adaptive],
            marked_segment_counts=[row["marked_segment_count"] for row in adaptive],
        ),
        input_records={path.name: digest(path) for path in sources},
        verifier_sha256=digest(Path(__file__)),
    )
    output = DATA / "mixed-published-comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (*sources, output):
        (FIGURES / path.name).write_bytes(path.read_bytes())
    print(output.name, digest(output))


if __name__ == "__main__":
    main()
