"""Verify current acoustic field provenance before publishing Marmousi comparisons.

These checks retain the declared primary crop as a separate experiment from
the unidentified historical arrays. They validate recorded original-equation
gates and actual archive bytes; they do not assert reference resolution or
replace an independent physical-field comparison.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from examples.campaign_provenance import require_equal, verify_archive
from examples.marmousi_data import FILES
from pymhm.io.provenance import file_digest
from pymhm.io.workspace import (
    case_workspace,
    local_resource,
    read_resource_text,
    source_file,
    source_label,
)

ROOT = case_workspace()
EXCLUSION = [5000.0, 50.0, 50.0]


def check_sources(record: Mapping[str, Any], required: tuple[str, ...]) -> None:
    """Require executed public owners and all guarded core files to match current bytes."""
    hashes = record.get("source_sha256")
    core = {
        source_label(path, ROOT)
        for path in source_file("src/pymhm/__init__.py", root=ROOT).parent.rglob("*.py")
    }
    if (
        record.get("source_changed_during_run") is not False
        or not isinstance(hashes, dict)
        or not core.union(required).issubset(hashes)
    ):
        raise ValueError("Marmousi publication requires complete current executed sources")
    for name, digest in hashes.items():
        path = (source_file(name, root=ROOT)).resolve()
        if not local_resource(path).is_file() or file_digest(local_resource(path)) != digest:
            raise ValueError("Marmousi executed source bytes differ from the current owner")


def check_material(record: Mapping[str, Any]) -> None:
    """Check pinned primary inputs, declared crop, SI sampling and acoustic forcing."""
    material = record.get("material", {})
    expected = {
        "crop_origin_m": [3395.0, 515.0],
        "cell_shape": [2048, 512],
        "cell_spacing_m": 5.0,
        "primary_shape": [13601, 2801],
        "primary_spacing_m": 1.25,
        "coordinate_order": ["x", "depth"],
        "density_unit": "kg/m^3",
        "velocity_unit": "m/s",
        "bulk_modulus_unit": "Pa",
        "historical_article_arrays_identified": False,
    }
    if not isinstance(material, dict) or any(name not in material for name in expected):
        raise ValueError("Marmousi publication requires the declared primary material crop")
    require_equal({name: material[name] for name in expected}, expected, label="material crop")
    for key, (filename, url, digest) in FILES.items():
        require_equal(
            material.get("sources", {}).get(key),
            {"filename": filename, "url": url, "sha256": digest},
            label="primary material source",
        )
    if record.get("omega") != 40 * np.pi or record.get("point_source") != [5000, 50, 1.0]:
        raise ValueError("Marmousi publication requires identical declared acoustic forcing")


def check_residual(record: Mapping[str, Any], name: str) -> None:
    """Enforce the recorded original-operator relative gate without accepting NaN or a skip."""
    value = record.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not np.isfinite(value)
        or not 0 <= value <= 1e-10
    ):
        raise ValueError(f"Marmousi publication requires the accepted {name} gate")


def checked_mhm(path: Path) -> dict[str, Any]:
    """Verify a full declared Q3 field and its original local and skeleton equations."""
    record = json.loads(read_resource_text(path))
    check_sources(
        record,
        (
            "examples/marmousi_campaign.py",
            "examples/marmousi_data.py",
            "examples/helmholtz_trace_family.py",
            "examples/campaign_provenance.py",
        ),
    )
    check_material(record)
    for name in ("algebraic_residual", "original_field_trace_residual"):
        check_residual(record, name)
    local_gate = (
        "original_local_equation_residual_max"
        if "original_local_equation_residual_max" in record
        else "local_equation_residual_max"
    )
    check_residual(record, local_gate)
    if "local_equation_residual_max" in record:
        check_residual(record, "local_equation_residual_max")
    width, degree = record.get("H_m"), record.get("trace_degree")
    if (
        type(width) is not int
        or width not in (20, 40, 80)
        or type(degree) is not int
        or degree not in range(5)
    ):
        raise ValueError("Marmousi publication requires a declared H20/H40/H80 polynomial trace")
    requested = record.get("requested_assembly_order")
    prepared = record.get("prepared_trace_degree", degree)
    if (
        type(requested) is not int
        or requested < 1
        or type(prepared) is not int
        or prepared not in range(degree, 5)
    ):
        raise ValueError("Marmousi fields require requested and prepared quadrature identity")
    # The solver's volume/face lower bounds depend on the executed prepared
    # degree. A lower-degree restriction retains that actual assembly rule.
    effective_order = max(requested, 3 + 2, 3 + prepared + 2)
    expected = {
        "macro_shape": [10240 // width, 2560 // width],
        "macro_cells": (10240 // width) * (2560 // width),
        "local_degree": 3,
        "local_refinement": width // 5 * 2,
        "assembly_order": effective_order,
        "boundary": "Top weak Dirichlet zero; other sides outgoing first-order absorption zero",
    }
    if any(name not in record for name in expected):
        raise ValueError("Marmousi field configuration is incomplete")
    require_equal({name: record[name] for name in expected}, expected, label="MHM configuration")
    verify_archive(path.parent / record["archive"], record["archive_sha256"])
    return record


def checked_reference(path: Path) -> dict[str, Any]:
    """Verify a native full-crop CG acquisition, original residual and runtime attribution."""
    record = json.loads(read_resource_text(path))
    check_sources(
        record,
        (
            "examples/marmousi_reference.py",
            "examples/marmousi_data.py",
            "examples/hpc4e_parallel.py",
            "examples/campaign_provenance.py",
        ),
    )
    check_material(record)
    check_residual(record, "stored_original_residual")
    if (
        record.get("geometry") != [2048, 512]
        or type(record.get("degree")) is not int
        or record.get("degree") not in (1, 2, 3, 4)
        or record.get("bounds") != [0, 10240, 0, 2560]
        or record.get("boundary")
        != "Top conforming Dirichlet zero; other sides outgoing first-order absorption zero"
    ):
        raise ValueError(
            "Marmousi publication requires a full pixel-conforming native P1--P4 field"
        )
    runtime = record.get("runtime_provenance", {})
    if (
        not isinstance(runtime, dict)
        or not runtime.get("observed_before_execution_at_utc")
        or runtime.get("upstream", {}).get("project") != "FEniCS/dolfinx"
        or not runtime.get("upstream", {}).get("revision")
        or not runtime.get("upstream", {}).get("revision_url")
        or not runtime.get("upstream", {}).get("verification")
        or not runtime.get("loaded_cpp_extension_sha256")
        or not runtime.get("loaded_libdolfinx_sha256")
        or not runtime.get("verification_limits")
    ):
        raise ValueError(
            "native reference publication requires executed runtime/source attribution"
        )
    archives = record.get("archives")
    if not isinstance(archives, list) or not archives:
        raise ValueError("native reference publication requires acquired nodal archives")
    for archive in archives:
        verify_archive(path.parent / archive["archive"], archive["sha256"])
    return record


def check_physical_norms(values: Mapping[str, Any], order: int) -> None:
    """Require complete physical measurements with one common declared disk and denominator.

    Full-domain pressure and derivative-domain norms retain different areas.
    This checks the stated integration convention; quadrature increments
    remain measured evidence of accuracy, rather than a mathematical bound.
    """
    if (
        not isinstance(values, Mapping)
        or values.get("quadrature_order") != order
        or values.get("gradient_exclusion") != EXCLUSION
        or values.get("gradient_cutout") is not None
    ):
        raise ValueError("physical norms require the declared common 50 m source disk")
    area = values.get("gradient_domain_area")
    if (
        values.get("domain_area") != 10240 * 2560
        or isinstance(area, bool)
        or not isinstance(area, (int, float))
        or not np.isfinite(area)
        or not 0 < area < 10240 * 2560
    ):
        raise ValueError("pressure and derivative measures require their stated distinct areas")
    for name in ("pressure", "gradient", "weighted_gradient", "flux", "graph"):
        for key in (f"{name}_difference", f"reference_{name}_norm", f"{name}_relative_difference"):
            value = values.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
                or value < 0
                or (key == f"reference_{name}_norm" and value == 0)
            ):
                raise ValueError("Marmousi physical norms require complete finite measurements")


def checked_comparison(path: Path, candidate: Path, reference: Path) -> dict[str, Any]:
    """Validate the current comparison chain and its common circular derivative domains."""
    record = json.loads(read_resource_text(path))
    check_sources(
        record,
        (
            "examples/marmousi_comparison.py",
            "examples/marmousi_fields.py",
            "examples/marmousi_data.py",
            "examples/campaign_provenance.py",
            "examples/marmousi_records.py",
        ),
    )
    for prefix, field in (("candidate", candidate), ("reference", reference)):
        if record.get(f"{prefix}_record_sha256") != file_digest(local_resource(field)):
            raise ValueError("Marmousi comparison identifies different acquired field records")
    norms = record.get("norms")
    if not isinstance(norms, dict) or set(norms) != {"8", "10"}:
        raise ValueError("Marmousi comparison requires both physical quadrature controls")
    for order, values in norms.items():
        check_physical_norms(values, int(order))
    samples = record.get("sampled_pressure", [])
    sides = [tuple(row["incident_side"]) for row in samples]
    if set(sides) != {(-1, -1), (-1, 1), (1, -1), (1, 1)} or len(sides) != 4:
        raise ValueError("four distinct incident sampling conventions are required")
    for row in samples:
        for name in ("difference", "reference_norm", "relative_difference"):
            value = row.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError("complete finite full-grid sampled pressure measures are required")
    return record


def checked_convergence(path: Path, reference: Path) -> dict[str, Any]:
    """Require complete P1--P4 refinement evidence under the same declared disk mask.

    Measured reference increments remain explicit; this validation does not
    treat an increment as an error bound or an unrefined field as exact.
    """
    record = json.loads(read_resource_text(path))
    check_sources(
        record,
        (
            "examples/marmousi_fields.py",
            "examples/marmousi_data.py",
            "examples/marmousi_records.py",
            "examples/campaign_provenance.py",
        ),
    )
    rows = record.get("references", [])
    if [row.get("degree") for row in rows] != [1, 2, 3, 4]:
        raise ValueError("Marmousi family publication requires all P1--P4 refinement levels")
    for row in rows:
        field_path = path.parent / row["reference_record"]
        checked_reference(field_path)
        if row.get("reference_record_sha256") != file_digest(local_resource(field_path)):
            raise ValueError("reference refinement identifies different native fields")
        verify_archive(path.parent / row["sample_archive"], row["sample_archive_sha256"])
        orders = (("physical_norms", 8),)
        if row["degree"] > 1:
            orders += (("increment", 6), ("increment_quadrature_check", 8))
        for key, order in orders:
            values = row.get(key, {})
            check_physical_norms(values, order)
    if rows[-1]["reference_record_sha256"] != file_digest(local_resource(reference)):
        raise ValueError("the plotted P4 denominator differs from the measured finest reference")
    return record
