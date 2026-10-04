"""Original-equation diagnostics for bounded initial flow and elasticity studies.

Only scalar diagnostics are persisted. Coefficient vectors and numerical bases
remain together in the executing solution; these records do not claim replay.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy import sparse

from examples.campaign_provenance import file_digest
from examples.core_elasticity_field_archive import ProductionObservation
from examples.transport_checkpoints import write_progress
from pymhm.hybrid import HybridSystem
from pymhm.solvers import _accurate_residual

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def observe_originals() -> Iterator[ProductionObservation]:
    """Observe unchanged HybridSystem rows, boundary moments and physical gauge calls.

    This scalar-only observer imposes no basis or constant-material archive
    requirements. One serial producer may execute in this process-scoped context.
    """
    observed = ProductionObservation()
    assemble_owner, solve_owner, mean_owner = (
        HybridSystem._assemble_global,
        HybridSystem.solve,
        HybridSystem.mean_constraint,
    )

    def assemble(system: HybridSystem, responses: Any, metadata: Any, boundary_load: Any) -> None:
        """Delegate assembly and retain its actual signed weak-boundary vector."""
        assemble_owner(system, responses, metadata, boundary_load)
        if observed.applied_boundary is not None or boundary_load is None:
            raise ValueError("One explicit production boundary vector is required")
        observed.applied_boundary = np.asarray(boundary_load).copy()

    def mean(system: HybridSystem, local_weights: Any, value: float = 0.0) -> Any:
        """Delegate a physical mean constraint and retain its unchanged target."""
        result = mean_owner(system, local_weights, value)
        observed.mean_weights.append(tuple(np.asarray(w).copy() for w in local_weights))
        observed.mean_values.append(value)
        return result

    def solve(system: HybridSystem, *args: Any, **kwargs: Any) -> Any:
        """Delegate solve and retain actual fixed moments and physical gauge rows."""
        result = solve_owner(system, *args, **kwargs)
        if observed.system is not None:
            raise ValueError("One production system is required")
        observed.system = system
        observed.fixed = dict(kwargs.get("fixed") or {})
        observed.constraints = [
            (np.asarray(row).copy(), value) for row, value in kwargs.get("constraints", ())
        ]
        return result

    with (
        patch.object(HybridSystem, "_assemble_global", assemble),
        patch.object(HybridSystem, "mean_constraint", mean),
        patch.object(HybridSystem, "solve", solve),
    ):
        yield observed


def capture_sources(output: Path, extra: Sequence[Path]) -> dict[str, str]:
    """Copy exact core, observer, analytical data and lock bytes before solving."""
    paths = [
        *sorted((ROOT / "src/pymhm").glob("*.py")),
        ROOT / "examples/minimal_flow_originals.py",
        ROOT / "examples/core_elasticity_field_archive.py",
        ROOT / "examples/archive_precision.py",
        ROOT / "examples/campaign_provenance.py",
        ROOT / "examples/transport_checkpoints.py",
        ROOT / "examples/local_response_cache.py",
        ROOT / "pixi.lock",
        ROOT / "pixi.toml",
        ROOT / "pyproject.toml",
        *extra,
    ]
    hashes = {str(path.relative_to(ROOT)): file_digest(path) for path in paths}
    for name, expected in hashes.items():
        target = output / "executed-sources/files" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if file_digest(target) != expected:
            raise ValueError("An executed source snapshot differs")
    write_progress(output / "executed-sources/manifest.json", {"source_sha256": hashes})
    return hashes


def sources_unchanged(hashes: Mapping[str, str]) -> bool:
    """Check every declared numerical owner/input after execution."""
    return all(file_digest(ROOT / name) == expected for name, expected in hashes.items())


def original_diagnostics(
    solution: Any,
    observed: ProductionObservation,
    local_blocks: Sequence[Mapping[str, int]],
) -> dict[str, Any]:
    """Check actual A*x+B*lambda=f, global coupling and physical gauges separately.

    Per-block values are coefficient-row backward errors, not physical L2
    residuals. The complete uncondensed residual is normalized by prescribed
    force/weak boundary data. The unchanged acceptance criterion is 1e-10.
    """
    system = observed.system
    if system is None or observed.applied_boundary is None:
        raise ValueError("The actual executed original system was not observed")
    trace = np.asarray(solution.hybrid.trace)
    boundary = np.asarray(observed.applied_boundary)
    fixed = np.array(sorted(observed.fixed), dtype=np.int64)
    free = np.ones(len(trace), dtype=bool)
    free[fixed] = False
    weak = -boundary.astype(np.longdouble)
    squared = rhs_squared = np.longdouble(0)
    maximum = 0.0
    blocks = []
    means = np.zeros(len(observed.mean_weights), dtype=np.longdouble)
    mean_scales = np.zeros_like(means)
    for cell, (response, field, declared) in enumerate(
        zip(system.responses, solution.hybrid.fields, local_blocks, strict=True)
    ):
        problem = response.problem
        matrix = sparse.hstack((problem.matrix, sparse.csr_matrix(problem.coupling)), format="csr")
        field = np.asarray(field)
        values = np.r_[field, trace[problem.trace_dofs]]
        defect = _accurate_residual(matrix, problem.load, values)
        scale = abs(matrix) @ abs(values) + abs(problem.load)
        start, row = 0, {}
        for name, size in declared.items():
            stop = start + size
            if size <= 0 or stop > len(field):
                raise ValueError("An original physical block has invalid size")
            row[name] = {
                "row_count": size,
                "absolute_row_residual_norm": float(np.linalg.norm(defect[start:stop])),
                "row_action_scale_norm": float(np.linalg.norm(scale[start:stop])),
                "backward_error": float(
                    np.linalg.norm(defect[start:stop])
                    / max(np.linalg.norm(scale[start:stop]), np.finfo(float).tiny)
                ),
            }
            maximum = max(maximum, row[name]["backward_error"])
            start = stop
        if start != len(field):
            raise ValueError("Every original local equation requires a named block")
        squared += np.sum(defect**2, dtype=np.longdouble)
        local_fixed = np.isin(problem.trace_dofs, fixed)
        forcing = (
            problem.load - problem.coupling[:, local_fixed] @ trace[problem.trace_dofs[local_fixed]]
        )
        rhs_squared += np.sum(forcing**2, dtype=np.longdouble)
        np.add.at(weak, problem.trace_dofs, problem.test_coupling.T @ field)
        for gauge, weights in enumerate(observed.mean_weights):
            means[gauge] += weights[cell] @ field
            mean_scales[gauge] += abs(weights[cell]) @ abs(field)
        blocks.append({"macrocell": cell, "blocks": row})
    rhs_squared += np.sum(boundary[free] ** 2, dtype=np.longdouble)
    full_squared = squared + np.sum(weak[free] ** 2, dtype=np.longdouble)
    full = (
        float(np.sqrt(full_squared / rhs_squared)) if rhs_squared else float(np.sqrt(full_squared))
    )
    coarse = np.concatenate(solution.hybrid.coarse)
    global_values = np.r_[trace, coarse]
    gauge_rows = np.array([row for row, _ in observed.constraints]).reshape(-1, len(global_values))
    gauge_targets = np.asarray([value for _, value in observed.constraints])
    gauge_defect = gauge_rows @ global_values - gauge_targets
    gauge_relative = float(
        np.max(
            abs(gauge_defect)
            / np.maximum(
                abs(gauge_rows) @ abs(global_values) + abs(gauge_targets), np.finfo(float).tiny
            ),
            initial=0,
        )
    )
    physical_targets = np.asarray(observed.mean_values)
    mean_relative = float(
        np.max(
            abs(means - physical_targets)
            / np.maximum(mean_scales + abs(physical_targets), np.finfo(float).tiny),
            initial=0,
        )
    )
    accepted = max(maximum, full, gauge_relative, mean_relative) <= 1e-10
    return {
        "accepted": accepted,
        "criterion": 1e-10,
        "full_uncondensed_relative_to_physical_rhs": full,
        "physical_rhs_row_norm": float(np.sqrt(rhs_squared)),
        "maximum_original_block_backward_error": maximum,
        "global_weak_row_absolute_residual_norm": float(np.linalg.norm(weak[free])),
        "physical_mean_relative_defect": mean_relative,
        "reduced_gauge_relative_defect": gauge_relative,
        "physical_mean_values": means.astype(float).tolist(),
        "physical_mean_targets": physical_targets.astype(float).tolist(),
        "gauge_multipliers": np.asarray(solution.hybrid.gauge_multipliers).astype(float).tolist(),
        "blocks": blocks,
        "norm_convention": (
            "Original coefficient-row block norms; physical field L2 errors are separate"
        ),
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
    }


def write_record(path: Path, record: Mapping[str, Any]) -> None:
    """Persist only a fresh scalar diagnostic record with explicit method conventions."""
    if path.exists():
        raise ValueError("A fresh minimal-study record is required")
    path.write_text(
        json.dumps(dict(record), indent=2, allow_nan=False, default=_json_scalar) + "\n"
    )


def _json_scalar(value: Any) -> bool | int | float:
    """Convert real NumPy scalars while rejecting coefficient arrays and complex values."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError("Scalar diagnostics require real scalar JSON values")
