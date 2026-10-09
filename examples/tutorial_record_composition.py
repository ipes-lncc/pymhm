"""Compose attributed refinement records without recomputing their observations.

Each retained row identifies the exact input file and digest. Executed source
manifests remain attached to those input files; this reader never substitutes
today's source tree for an acquisition's historical numerical provenance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pymhm.io.provenance import file_digest


def compose_records(
    root: Path,
    inputs: tuple[tuple[str, str], ...],
    output: str,
    *,
    resolution: str,
    description: dict[str, Any],
    controls: tuple[str, ...] = (),
    selected_levels: Mapping[str, tuple[float, ...]] | None = None,
    field_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Copy all refinement rows and preserve per-file source and data identity.

    ``inputs`` contains repository-relative paths and their row-array keys.
    Duplicate resolutions are rejected: separate time/quadrature controls belong
    in ``controls`` and do not become extra spatial refinement observations.
    Original-equation block diagnostics stay in their explicitly cited files.
    ``selected_levels`` identifies the retained rows of an input containing
    repeated control levels; the full input and its digest are still attributed.
    ``field_names`` records a terminology correction without changing any value.
    """
    attributed = []
    rows = []
    levels: set[float] = set()
    for name, key in inputs:
        path = root / name
        record = json.loads(path.read_text())
        digest = file_digest(path)
        selected = None if selected_levels is None else selected_levels.get(name)
        available = {float(row[resolution]) for row in record[key]}
        if selected is not None and not set(selected).issubset(available):
            raise ValueError(f"Selected levels are absent from the input: {name}")
        attributed.append(
            {
                "path": name,
                "sha256": digest,
                "source_sha256": record.get("source_sha256", {}),
                "selected_levels": selected,
            }
        )
        for original in record[key]:
            n = float(original[resolution])
            if selected is not None and n not in selected:
                continue
            if n in levels:
                raise ValueError(f"Resolution {n} occurs twice in the refinement sequence")
            if original.get("accepted") is False:
                raise ValueError(f"An explicitly rejected row cannot be published: {name}, n={n}")
            levels.add(n)
            row = {
                (field_names or {}).get(k, k): value
                for k, value in original.items()
                if k != "original_equations"
            }
            row["input_record"] = {"path": name, "sha256": digest, "row_key": key}
            rows.append(row)
    control_records = [{"path": name, "sha256": file_digest(root / name)} for name in controls]
    composite = {
        "schema": "pymhm-attributed-tutorial-refinement-v1",
        **description,
        "inputs": attributed,
        "controls": control_records,
        "rows": sorted(rows, key=lambda row: float(row[resolution])),
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "composition": "Declared input levels retained; no numerical observations recomputed",
        "field_names": dict(field_names or {}),
    }
    target = root / output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(composite, indent=2, allow_nan=False) + "\n")
    return composite
