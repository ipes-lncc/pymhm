"""Observe matched smooth anisotropic elasticity fields and plot their errors.

Both acquisition owners execute their original assembly, solve and independent
quadrature controls unchanged. Observers sample the actual physical fields
before native resources close. No coefficient replay or interface averaging is
used, and every panel shows the actual MHM macro mesh.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

from examples import tutorial_primal_elastic_reference as classical
from examples import tutorial_vector_acquisition as acquisition
from examples.plot_mesh import draw_macro_mesh
from examples.solve_primal_elasticity import TensorData
from pymhm import TriangleMesh
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.io.workspace import source_file, source_identity
from pymhm.postprocessing.fields import DiscreteField
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.primal_elasticity import PrimalElasticitySolution

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
LICENSE = "CC BY 4.0; IPES Research Group; https://creativecommons.org/licenses/by/4.0/"


@contextmanager
def observe_mhm() -> Iterator[list[PrimalElasticitySolution]]:
    """Retain the actual MHM reconstruction without changing its owner or controls."""
    solutions: list[PrimalElasticitySolution] = []
    original = acquisition.recover_primal_elasticity

    def recover(*args: Any, **kwargs: Any) -> PrimalElasticitySolution:
        """Delegate the unchanged reconstruction, then retain its physical solution."""
        solution = original(*args, **kwargs)
        solutions.append(solution)
        return solution

    with patch.object(acquisition, "recover_primal_elasticity", recover):
        yield solutions


def sample_columns(points: np.ndarray, displacement: Any, stress: Any) -> np.ndarray:
    """Report displacement magnitude, physical stress Frobenius norm and true errors."""
    data = TensorData(False)
    exact_u, exact_sigma = data.displacement(points), data.stress(points)
    return np.column_stack(
        (
            np.linalg.norm(exact_u, axis=1),
            np.linalg.norm(displacement, axis=1),
            np.linalg.norm(exact_sigma, axis=(1, 2)),
            np.linalg.norm(stress, axis=(1, 2)),
            np.linalg.norm(displacement - exact_u, axis=1),
            np.linalg.norm(stress - exact_sigma, axis=(1, 2)),
        )
    )


def samples(mesh: TriangleMesh, values: np.ndarray) -> dict[str, Any]:
    """Describe flat physical centroid samples on a literal display triangulation."""
    return {
        "sampling": "independent display-triangle centroid values; no macroface averaging",
        "points": mesh.points.tolist(),
        "cells": mesh.cells.tolist(),
        "columns": [
            "analytical_displacement_magnitude",
            "displacement_magnitude",
            "analytical_cauchy_stress_frobenius",
            "cauchy_stress_frobenius",
            "displacement_error_magnitude",
            "cauchy_stress_error_frobenius",
        ],
        "values": values.tolist(),
    }


def mhm_samples(solution: PrimalElasticitySolution) -> dict[str, Any]:
    """Sample the polynomial field independently on each side of every macroface."""
    reference = TriangleMesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]]).submesh(0, 8)
    centers = reference.points[reference.cells].mean(axis=1)
    bary = np.column_stack((1 - centers.sum(axis=1), centers))
    points, cells, values = [], [], []
    offset = 0
    for cell, fine in enumerate(solution.local_meshes):
        if len(fine.cells) != 1:
            raise ValueError("The declared r1 primal field has one fine cell per macrocell")
        display = solution.skeleton.mesh.submesh(cell, 8)
        physical = display.points[display.cells].mean(axis=1)
        if not np.allclose(physical, bary @ fine.points[fine.cells[0]], atol=1e-14, rtol=0):
            raise ValueError("The display and evaluation cells use inconsistent coordinates")
        field = DiscreteField(
            nodal_field("displacement", fine, 3, components=2), solution.values[cell].ravel()
        )
        displacement = field.evaluate(physical, cells=np.zeros(len(physical), dtype=np.int64))
        sigma = solution.stress(cell, bary).reshape(-1, 2, 2)
        values.append(sample_columns(physical, displacement, sigma))
        points.append(display.points)
        cells.append(display.cells + offset)
        offset += len(display.points)
    display_mesh = TriangleMesh(np.vstack(points), np.vstack(cells))
    return samples(display_mesh, np.vstack(values))


@contextmanager
def observe_classical() -> Iterator[list[dict[str, Any]]]:
    """Sample the native solution while its space is live; delegate stress unchanged."""
    from dolfinx import fem

    bindings, fields = [], []
    original_binding, original_stress = classical.bind_space, classical.physical_stress

    def bind(*args: Any, **kwargs: Any) -> Any:
        """Delegate construction and retain the live binding only for observation."""
        binding = original_binding(*args, **kwargs)
        bindings.append(binding)
        return binding

    def stress(displacement: Any) -> Any:
        """Delegate the physical constitutive expression and sample a populated Function."""
        expression = original_stress(displacement)
        if isinstance(displacement, fem.Function) and not fields:
            if len(bindings) != 1:
                raise ValueError("One native binding is required for the field observation")
            binding = bindings[0]
            mesh = binding.portable_mesh
            points = mesh.points[mesh.cells].mean(axis=1)
            values = binding.evaluate(
                displacement.x.array, points, cells=np.arange(len(mesh.cells))
            )
            sigma = fem.Expression(expression, np.array([[1 / 3, 1 / 3]])).eval(
                binding.mesh, binding.native_cells.astype(np.int32)
            )
            fields.append(samples(mesh, sample_columns(points, values, sigma.reshape(-1, 2, 2))))
        return expression

    with (
        patch.object(classical, "bind_space", bind),
        patch.object(classical, "physical_stress", stress),
    ):
        yield fields


def export_figures(record: dict[str, Any], output: Path) -> list[dict[str, str]]:
    """Export matched fields and errors with physical scales and literal macro edges."""
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    macro = TriangleMesh(record["macro_points"], record["macro_cells"])
    observations = [record["mhm_samples"], record["classical_samples"]]
    arrays = [np.asarray(observation["values"]) for observation in observations]
    triangles = [
        Triangulation(*np.asarray(observation["points"]).T, np.asarray(observation["cells"]))
        for observation in observations
    ]
    fields, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained")
    for row, (label, exact_column, actual_column) in enumerate(
        (("Displacement magnitude", 0, 1), ("Cauchy stress Frobenius norm", 2, 3))
    ):
        physical_values = [
            arrays[0][:, exact_column],
            arrays[0][:, actual_column],
            arrays[1][:, actual_column],
        ]
        lower = min(value.min() for value in physical_values)
        upper = max(value.max() for value in physical_values)
        for axis, title, values, triangulation in zip(
            axes[row],
            ("Analytical", "Primal MHM: n=8", "Classical P3: n=64"),
            physical_values,
            (triangles[0], triangles[0], triangles[1]),
            strict=True,
        ):
            image = axis.tripcolor(
                triangulation,
                facecolors=values,
                shading="flat",
                cmap="viridis",
                vmin=lower,
                vmax=upper,
                rasterized=True,
            )
            draw_macro_mesh(axis, macro)
            axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            fields.colorbar(image, ax=axis, label=label, fraction=0.046, pad=0.035)
    fields.suptitle("Smooth anisotropic elasticity; actual n=8 macro mesh on every panel")
    errors, axes = plt.subplots(2, 2, figsize=(9.4, 8.6), layout="constrained")
    for row, (label, column) in enumerate(
        (("Displacement error magnitude", 4), ("Cauchy stress error Frobenius norm", 5))
    ):
        for title, axis, triangulation, array in zip(
            ("Primal MHM: n=8", "Classical P3: n=64"), axes[row], triangles, arrays, strict=True
        ):
            image = axis.tripcolor(
                triangulation,
                facecolors=array[:, column],
                shading="flat",
                cmap="magma",
                vmin=0,
                rasterized=True,
            )
            draw_macro_mesh(axis, macro)
            axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            errors.colorbar(image, ax=axis, label=label, fraction=0.046, pad=0.035)
    errors.suptitle("Physical errors against the analytical solution; separate error scales")
    description = (
        "Constant anisotropic Kelvin tensor and sine displacement, identical material/force/"
        "zero exterior displacement for primal MHM n8 P3/r1/P1 and native conforming P3 n64. "
        "Cauchy stress is the physical constitutive gradient, not an H(div) recovery. "
        "All panels show actual MHM macro boundaries; one-sided values are not averaged. " + LICENSE
    )
    outputs = []
    for figure, stem in (
        (fields, "primal-elasticity-smooth-fields"),
        (errors, "primal-elasticity-smooth-errors"),
    ):
        for suffix in ("png", "svg", "pdf"):
            target = output / f"{stem}.{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            metadata = (
                {"Author": "IPES Research Group", "Description": description, "Copyright": LICENSE}
                if suffix == "png"
                else {
                    "Creator": "IPES Research Group",
                    "Description": description,
                    "Rights": LICENSE,
                    "Date": None,
                }
                if suffix == "svg"
                else {
                    "Author": "IPES Research Group",
                    "Subject": description,
                    "CreationDate": None,
                    "ModDate": None,
                }
            )
            figure.savefig(target, dpi=240, bbox_inches="tight", pad_inches=0.06, metadata=metadata)
            if suffix == "svg":
                target.write_text(
                    "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
                )
            outputs.append(
                {"path": target.relative_to(ROOT).as_posix(), "sha256": file_digest(target)}
            )
        preview = ROOT / "build/docs-restructure/vector-canonical-previews" / f"{stem}-850.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            preview, dpi=850 / figure.get_figwidth(), bbox_inches="tight", pad_inches=0.06
        )
        plt.close(figure)
    return outputs


def main() -> None:
    """Acquire unchanged owners, verify matched norms and sources, and export fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("A field observation record already exists")
    sources = {**acquisition.source_hashes(), **classical.source_hashes()}
    sources.update(current_source_manifest(source_identity(ROOT, [Path(__file__)])))
    with threadpool_limits(1):
        with observe_mhm() as solutions:
            mhm = acquisition.primal_row(8, workers=1)
        with observe_classical() as observations:
            reference = classical.reference_row(64)
        sampled_mhm = mhm_samples(solutions[0])
    if not mhm["accepted"] or not reference["accepted"]:
        raise ValueError("Original equations or independent error quadrature rejected")
    classical_record = (
        ROOT / "examples/results/tutorial-methods/primal-elasticity-classical-current.json"
    )
    archived_reference = next(
        row for row in json.loads(classical_record.read_text())["rows"] if row["resolution"] == 64
    )
    mhm_record = ROOT / "examples/results/tutorial-methods/primal-elasticity-asymptotic.json"
    archived_mhm = next(row for row in json.loads(mhm_record.read_text())["rows"] if row["n"] == 8)
    for name in ("displacement_l2", "stress_l2"):
        if not np.isclose(mhm[name], archived_mhm[name], rtol=1e-8, atol=0):
            raise ValueError("MHM field norm differs from the declared refinement study")
        if not np.isclose(reference[name], archived_reference[name], rtol=1e-8, atol=0):
            raise ValueError("Classical field norm differs from its same-case native study")
    if not all(
        file_digest(source_file(name, root=ROOT)) == digest for name, digest in sources.items()
    ):
        raise ValueError("An executed field source changed during observation")
    macro = solutions[0].skeleton.mesh
    report = {
        "schema": "pymhm-sampled-physical-fields-v1",
        "problem": "smooth constant anisotropic primal elasticity",
        "dimension": 2,
        "tensor_kelvin": classical.KELVIN_STIFFNESS.tolist(),
        "exact_displacement": "(sin(pi*x)*sin(pi*y), sin(2*pi*x)*sin(pi*y))",
        "boundary": "zero exterior displacement",
        "stress": "physical broken symmetric Cauchy stress; not an H(div) recovery",
        "source_sha256": sources,
        "source_unchanged": True,
        "inputs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)}
            for path in (mhm_record, classical_record)
        ],
        "mhm": mhm,
        "classical": reference,
        "macro_points": macro.points.tolist(),
        "macro_cells": macro.cells.tolist(),
        "mhm_samples": sampled_mhm,
        "classical_samples": observations[0],
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "license": LICENSE,
        "accepted": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    outputs = export_figures(report, ROOT / "docs/assets/tutorials/methods")
    receipt = {
        "record": args.output.as_posix(),
        "sha256": file_digest(args.output),
        "figures": outputs,
    }
    (ROOT / "build/docs-restructure/primal-elasticity-smooth-field-figures.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
