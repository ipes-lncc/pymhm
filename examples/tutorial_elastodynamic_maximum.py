"""Measure uniform-in-time physical errors in the unchanged smooth dynamic march.

The existing acquisition owns equations and endpoint verification. Independent
error quadrature caches its affine basis/geometry once, then observes every
returned state. All endpoint errors are retained alongside the maximum norms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from examples import tutorial_vector_asymptotic as acquisition
from pymhm.fem.vector.curl import physical_basis, physical_points, quadrature
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from pymhm.postprocessing.dynamics import stress_from_gradient
from scripts.build_notebook_companions import module_sources

ROOT = case_workspace()


def norm_tables(locals_: tuple[Any, ...], order: int) -> dict[str, Any]:
    """Cache actual local affine maps and independently evaluated physical truth."""
    points, basis, gradient, weights = [], [], [], []
    for local in locals_:
        if len(local.mesh.cells) != 1:
            raise ValueError("This declared P3/r1 study has one triangle per macrocell")
        bary, measure, _ = quadrature(local.mesh, 1, order)
        phi, derivative = physical_basis(local.mesh, local.degree, bary)
        points.append(physical_points(local.mesh, bary)[0])
        basis.append(phi[0])
        gradient.append(derivative[0])
        weights.append(measure[0])
    x = np.stack(points)
    w, stress, _ = acquisition.isotropic_wave_data(x.reshape(-1, 2))
    return {
        "points": x,
        "basis": np.stack(basis),
        "gradient": np.stack(gradient),
        "weights": np.stack(weights),
        "truth_shape": w.reshape(x.shape),
        "stress_shape": stress.reshape(*x.shape[:2], 2, 2),
    }


def physical_norms(state: Any, table: dict[str, Any]) -> dict[str, float]:
    """Integrate displacement, velocity and raw Cauchy-stress physical errors."""
    coefficients = [
        np.stack(
            [
                values.reshape(-1, 2)[local.dofs[0]]
                for local, values in zip(state.locals, fields, strict=True)
            ]
        )
        for fields in (state.displacement, state.velocity)
    ]
    displacement, velocity = [
        np.einsum("tqi,tia->tqa", table["basis"], values) for values in coefficients
    ]
    derivative = np.einsum("tqib,tia->tqab", table["gradient"], coefficients[0])
    stress = stress_from_gradient(state.locals[0], table["points"], derivative)
    errors = (
        displacement - 0.5 * state.time**2 * table["truth_shape"],
        velocity - state.time * table["truth_shape"],
        stress - 0.5 * state.time**2 * table["stress_shape"],
    )
    return {
        name: float(
            np.sqrt(
                np.sum(
                    table["weights"] * np.sum(delta**2, axis=tuple(range(2, delta.ndim))),
                    dtype=np.longdouble,
                )
            )
        )
        for name, delta in zip(("displacement_l2", "velocity_l2", "stress_l2"), errors, strict=True)
    }


def maximum_row(n: int, dt: float, final: float, workers: int) -> dict[str, Any]:
    """Observe each physical state without changing its source or endpoint solve."""
    march = acquisition.advance
    tables: dict[int, dict[str, Any]] = {}
    maxima = {str(order): {} for order in (10, 14)}
    last = {}

    def observe(data: Any, state: Any, source: Any) -> Any:
        """Delegate the unchanged step and observe independent physical norms."""
        result = march(data, state, source)
        for order in (10, 14):
            if order not in tables:
                tables[order] = norm_tables(result.locals, order)
            values = physical_norms(result, tables[order])
            for name, error in values.items():
                maxima[str(order)][name] = max(maxima[str(order)].get(name, 0.0), error)
            last[order] = values
        return result

    with patch.object(acquisition, "advance", observe):
        endpoint = acquisition.elastodynamic_row(n, dt=dt, final=final, workers=workers)
    same_owner = {
        name: abs(last[14][name] - endpoint[name]) / max(endpoint[name], np.finfo(float).tiny)
        for name in last[14]
    }
    low, high = maxima.values()
    changes = {
        name: abs(low[name] - high[name]) / max(high[name], np.finfo(float).tiny) for name in high
    }
    row = {**endpoint, **high, "endpoint": {name: endpoint[name] for name in high}}
    row["norm_scope"] = "Maximum over every returned common physical time, including final time"
    row["maximum_norm_quadrature"] = {
        "orders": [10, 14],
        "maxima": maxima,
        "relative_changes": changes,
        "endpoint_agreement_with_standard_owner": same_owner,
    }
    row["accepted"] = bool(
        endpoint["accepted"] and max(changes.values()) <= 1e-8 and max(same_owner.values()) <= 1e-10
    )
    return row


def main() -> None:
    """Acquire uniform-in-time norm rows with explicit immutable source identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--final", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("An existing maximum-norm acquisition cannot be overwritten")
    paths = [
        *sorted((ROOT / "src/pymhm").rglob("*.py")),
        *(
            ROOT / name
            for name in module_sources(ROOT, ["examples.tutorial_elastodynamic_maximum"])
        ),
        ROOT / "pixi.lock",
        ROOT / "pixi.toml",
        ROOT / "pyproject.toml",
    ]
    sources = current_source_manifest(source_identity(ROOT, paths), packages=("pymhm",))
    record = {
        "schema": "pymhm-elastodynamic-maximum-physical-norms-v1",
        "source_sha256": sources,
        "norm_scope": "Maximum over every computed physical time",
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        for n in args.levels:
            row = maximum_row(n, args.dt, args.final, args.workers)
            record["rows"].append(row)
            args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
            if not row["accepted"]:
                raise RuntimeError("Original endpoint or independent physical norm checks failed")
    if any(file_digest(source_file(name, root=ROOT)) != digest for name, digest in sources.items()):
        raise RuntimeError("An executed maximum-norm source changed during acquisition")


if __name__ == "__main__":
    main()
