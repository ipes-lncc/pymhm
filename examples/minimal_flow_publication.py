"""Collect accepted scalar flow/elasticity levels into initial convergence figures.

Executed sources and acquisition UUIDs retain their original identity. This
consumer performs no PDE solve and does not replay coefficient fields.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pymhm.io.workspace import case_workspace, source_file, source_label

ROOT = case_workspace()
GROUPS = {
    "flow3d": ("flow3d", ("stokes-th2", "brinkman-usfem1", "brinkman-usfem2", "oseen-p2")),
    "gals3d": ("gals3d", ("gals-p1", "gals-p2", "th-p2")),
    "primal-elasticity3d": ("elasticity3d", ("anisotropic-p2", "anisotropic-p3")),
    "mixed-elasticity3d": ("mixed-elasticity3d", ("bdm2", "bdm3")),
    "stokes2d": ("flow2d", ("stokes-usfem-l0", "stokes-usfem-l1", "stokes-usfem-l2")),
    "oseen2d-trace": ("flow2d", ("oseen-smooth-l0", "oseen-smooth-l1", "oseen-smooth-l2")),
    "oseen2d-viscosity": (
        "flow2d",
        ("oseen-smooth-l1", "oseen-smooth-nu001", "oseen-smooth-nu00001"),
    ),
    "oseen2d-data": ("flow2d", ("oseen-boundary", "oseen-internal", "oseen-variable")),
}
LABELS = {
    "velocity_l2": r"Velocity $L^2$ error",
    "pressure_l2": r"Pressure $L^2$ error",
    "velocity_h1_seminorm": r"Velocity $H^1$ seminorm error",
    "divergence_l2": r"Velocity divergence $L^2$",
    "displacement_l2": r"Displacement $L^2$ error",
    "displacement_h1_seminorm": r"Displacement $H^1$ seminorm error",
    "stress_l2": r"Stress $L^2$ error",
    "rotation_l2": r"Rotation $L^2$ error",
    "compressibility_l2": r"Compressibility $L^2$",
    "u_l2": r"Displacement $L^2$ error",
    "sigma_l2": r"Stress $L^2$ error",
}


def digest(path: Path) -> str:
    """Hash a complete scalar record, figure or executed source snapshot."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(
    group: str,
    directories: tuple[Path, ...],
    output: Path,
    *,
    cases: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Join accepted immutable levels and preserve refusals as explicit limitations."""
    directories, output = tuple(path.resolve() for path in directories), output.resolve()
    if group not in GROUPS or output.exists():
        raise ValueError("A declared group and fresh publication staging directory are required")
    family, declared_cases = GROUPS[group]
    cases = declared_cases if cases is None else cases
    if not cases or len(set(cases)) != len(cases) or not set(cases).issubset(declared_cases):
        raise ValueError("Select distinct declared formulation variants")
    rows, rejected = [], []
    for directory in directories:
        for path in sorted(directory.glob("*/record.json")):
            record = json.loads(path.read_text())
            if record["family"] != family or record["case"] not in cases:
                continue
            if not record["accepted"]:
                rejected.append({"record": source_label(path, ROOT), "reason": "criteria"})
                continue
            for source, expected in record["source_sha256"].items():
                if digest(path.parent / "executed-sources/files" / source) != expected:
                    raise ValueError("The original executed source snapshot is inconsistent")
                if (
                    source.startswith("src/pymhm/")
                    and digest(source_file(source, root=ROOT)) != expected
                ):
                    raise ValueError("The current numerical core differs from the accepted study")
            resources = tuple(
                (ROOT / "build/reports/completion").glob(
                    f"minimal-{family}-{record['case']}-n{record['resolution']}-*-resource.json"
                )
            )
            accepted_resources = []
            for receipt in resources:
                data = json.loads(receipt.read_text())
                if data.get("execution_accepted") and any(
                    path.parent.resolve() == Path(argument).resolve()
                    for argument in data["command"]
                ):
                    accepted_resources.append((receipt, data))
            if not accepted_resources:
                continue
            if len(accepted_resources) != 1:
                raise ValueError(
                    "Every accepted acquisition requires one exact owned resource closure"
                )
            receipt, resource = accepted_resources[0]
            record.update(
                numerical_record=source_label(path, ROOT),
                numerical_record_sha256=digest(path),
                resource_record=source_label(receipt, ROOT),
                resource_record_sha256=digest(receipt),
                owned_elapsed_seconds=resource["elapsed_seconds"],
                owned_peak_rss_bytes=resource["peak_owned_rss_bytes"],
                closed_owned_children=resource["owned_processes_remaining"],
            )
            rows.append(record)
    rows.sort(key=lambda row: (cases.index(row["case"]), row["resolution"]))
    if not rows:
        raise ValueError("No accepted levels exist for the selected initial study")
    identities = {(row["case"], row["resolution"]) for row in rows}
    if len(identities) != len(rows) or len({row["acquisition_uuid"] for row in rows}) != len(rows):
        raise ValueError("Distinct producer levels must retain distinct acquisition UUIDs")
    levels = [2, 4, 8] if family == "flow2d" else [1, 2, 3]
    complete = all((case, n) in identities for case in cases for n in levels)
    rates = {}
    for case in cases:
        selected = [row for row in rows if row["case"] == case]
        if len(selected) < 2:
            continue
        rates[case] = [
            {
                "levels": [a["resolution"], b["resolution"]],
                "observed_orders": {
                    key: float(
                        np.log(a["terminal_norms"][key] / b["terminal_norms"][key])
                        / np.log(b["resolution"] / a["resolution"])
                    )
                    for key in b["terminal_norms"]
                    if a["terminal_norms"][key] > 0 and b["terminal_norms"][key] > 0
                },
            }
            for a, b in zip(selected[:-1], selected[1:], strict=True)
        ]
    output.mkdir(parents=True)
    for row in rows:
        shutil.copyfile(
            ROOT / row["numerical_record"],
            output / (Path(row["numerical_record"]).parent.name + ".json"),
        )
    result: dict[str, Any] = {
        "schema": "pymhm-initial-flow-elasticity-convergence-v1",
        "group": group,
        "rows": rows,
        "observed_orders": rates,
        "expected_levels": levels,
        "selected_cases": list(cases),
        "omitted_cases": [case for case in declared_cases if case not in cases],
        "three_levels_complete": complete,
        "rejected_levels": rejected,
        "consumer_sha256": digest(Path(__file__)),
        "pde_solves_during_collection": 0,
        "historical_literature_reproduction": False,
        "field_replay_claimed": False,
        "scope": "Initial analytical convergence at fixed formulation and physical data",
    }
    plot(result, output / "convergence.png")
    result["figure_sha256"] = {path.name: digest(path) for path in output.glob("convergence.*")}
    (output / "study.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def plot(record: dict[str, Any], path: Path) -> None:
    """Render separate physical metrics with independent axes and readable legends."""
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    rows, group = record["rows"], record["group"]
    metrics = list(rows[0]["terminal_norms"])
    if group == "gals3d":
        metrics.remove("compressibility_l2")
    count = len(metrics)
    columns = 2 if count > 1 else 1
    figure, axes = plt.subplots(
        (count + columns - 1) // columns,
        columns,
        figsize=(8.8, 3.3 * ((count + columns - 1) // columns)),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for index, metric in enumerate(metrics):
        axis = axes[index]
        for case in record["selected_cases"]:
            selected = [row for row in rows if row["case"] == case]
            if selected:
                axis.loglog(
                    [row["resolution"] for row in selected],
                    [row["terminal_norms"][metric] for row in selected],
                    "o-",
                    label=case,
                )
        title = LABELS.get(metric, metric.replace("_", " "))
        if group == "mixed-elasticity3d" and metric == "divergence_l2":
            title = r"Stress divergence $L^2$ error"
        axis.set(title=title, xlabel="Macro subdivisions per direction, n", ylabel="Norm")
        axis.grid(True, which="both", alpha=0.25)
    for axis in axes[count:]:
        axis.set_visible(False)
    figure.suptitle(group.replace("-", " ") + ": initial three-level study", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=9)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    """Parse the declared CLI controls and run the original case with its thread limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=tuple(GROUPS), required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(
        args.group,
        tuple(args.input),
        args.output,
        cases=None if args.cases is None else tuple(args.cases),
    )


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.minimal_flow_publication").main()
