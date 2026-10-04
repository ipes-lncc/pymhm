"""Validate primary graph data and independently acquired Helmholtz comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from examples.campaign_checkpoint import require_sources, verify_checkpoint
from examples.helmholtz_article import article_hashes
from pymhm.io.provenance import file_digest


def support_rows(
    directory: Path, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Require current 44 primary markers and the twelve complete native controls.

    Primary PDF coordinates remain literature data. Their derived comparison
    identifies the exact analytical acquisition bytes. Native records identify
    current candidate operators separately from the independent reference code,
    and retain runtime attribution plus physical coefficient and residual gates.
    """
    published = json.loads((directory / "published-convergence.json").read_text())
    native = json.loads((directory / "native-convergence-verification.json").read_text())
    expected = {
        (r["ell"], r["n"], r["trace_basis"], field): r
        for r in rows
        if r["study"] == "convergence"
        for field in ("pressure", "gradient")
    }
    graph_keys = [(r["ell"], r["n"], r["basis"], r["field"]) for r in published["rows"]]
    if len(graph_keys) != 44 or len(set(graph_keys)) != 44 or set(graph_keys) != expected.keys():
        raise ValueError("the published Helmholtz graph comparison is incomplete")
    identity = published["pymhm_source"]
    require_sources(identity["source_sha256"], article_hashes())
    if identity["sha256"] != file_digest(directory / "article.json"):
        raise ValueError("the primary graph comparison identifies different analytical data")
    for key, row in zip(graph_keys, published["rows"], strict=True):
        case = expected[key]
        verify_checkpoint(
            row,
            {
                "pymhm_case_key": case["key"],
                "pymhm_relative_error": case[f"{row['field']}_relative_error"],
            },
            directory=directory,
            metrics=("published_graph_value",),
        )
        interval = np.asarray(row["published_graph_interval"])
        value = row["published_graph_value"]
        if (
            interval.shape != (2,)
            or not np.isfinite(interval).all()
            or not 0 < interval[0] <= value <= interval[1]
        ):
            raise ValueError("published graph uncertainty must bracket its positive ordinate")
    current_core = {
        name: value for name, value in article_hashes().items() if name.startswith("src/pymhm/")
    }
    observed = native.get("executed_source_sha256", native.get("pymhm_source_sha256", {}))
    require_sources(
        {name: value for name, value in observed.items() if name.startswith("src/pymhm/")},
        current_core,
    )
    if not native.get("runtime_provenance"):
        raise ValueError("native Helmholtz comparisons require executed runtime provenance")
    native_keys = [(r["ell"], r["n"], r["oscillatory"]) for r in native["rows"]]
    target = {
        (ell, n, oscillatory)
        for ell, sizes in ((2, (12, 16, 24, 32)), (3, (24, 32)))
        for n in sizes
        for oscillatory in (False, True)
    }
    if len(native_keys) != 12 or set(native_keys) != target:
        raise ValueError("the native Helmholtz convergence controls are incomplete")
    for row in native["rows"]:
        verify_checkpoint(
            row,
            {
                "local_degree": row["ell"] + 2,
                "refinement": 2,
                "omega": 10 * np.pi,
                "theta": np.pi / 13,
            },
            directory=directory,
            metrics=(
                "native_original_equation_relative_residual",
                "pressure_coefficient_relative_l2_difference",
                "native_relative_pressure_l2",
                "native_relative_gradient_l2",
                "pymhm_relative_pressure_l2",
                "pymhm_relative_gradient_l2",
            ),
        )
        if (
            max(
                row["native_original_equation_relative_residual"],
                row["pressure_coefficient_relative_l2_difference"],
            )
            > 1e-10
        ):
            raise ValueError("native original-equation or physical coefficient agreement failed")
    return published["rows"], native["rows"]
