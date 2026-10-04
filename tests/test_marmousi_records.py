"""Reject stale or physically unaccepted fields before acoustic publication.

Opaque field bytes exercise the record chain without pretending to solve a
large Helmholtz problem. Actual field integration has separate native tests.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from examples import marmousi_records as driver
from examples.marmousi_data import FILES
from pymhm.io.provenance import current_source_manifest, file_digest


@pytest.fixture
def sources() -> dict[str, str]:
    """Identify current real owners; fixture coefficients carry no numerical claim."""
    names = (
        "marmousi_campaign",
        "marmousi_data",
        "marmousi_fields",
        "marmousi_reference",
        "marmousi_records",
        "marmousi_comparison",
        "helmholtz_trace_family",
        "campaign_provenance",
        "hpc4e_parallel",
    )
    paths = [
        *sorted((driver.ROOT / "src/pymhm").rglob("*.py")),
        *(driver.ROOT / f"examples/{name}.py" for name in names),
    ]
    return current_source_manifest(
        {str(path.relative_to(driver.ROOT)): file_digest(path) for path in paths}
    )


@pytest.fixture
def material() -> dict:
    """Use the declared primary-grid identity, independent of opaque pressure vectors."""
    return {
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
        "sources": {
            key: {"filename": name, "url": url, "sha256": digest}
            for key, (name, url, digest) in FILES.items()
        },
    }


@pytest.fixture
def mhm(tmp_path, sources, material):
    """Record a complete whole-domain configuration with explicit original gates."""
    archive = tmp_path / "candidate.npz"
    archive.write_bytes(b"opaque executed pressure vector")
    path = tmp_path / "candidate.json"
    record = dict(
        source_sha256=sources,
        source_changed_during_run=False,
        material=material,
        omega=40 * np.pi,
        point_source=[5000, 50, 1.0],
        H_m=80,
        trace_degree=2,
        macro_shape=[128, 32],
        macro_cells=4096,
        local_degree=3,
        local_refinement=32,
        prepared_trace_degree=4,
        requested_assembly_order=9,
        assembly_order=9,
        boundary="Top weak Dirichlet zero; other sides outgoing first-order absorption zero",
        algebraic_residual=1e-15,
        original_field_trace_residual=2e-15,
        local_equation_residual_max=3e-15,
        archive=archive.name,
        archive_sha256=file_digest(archive),
    )
    path.write_text(json.dumps(record))
    return path, record


@pytest.fixture
def reference(tmp_path, sources, material):
    """Record source-attributed native nodal data independently of the MHM vector."""
    archive = tmp_path / "native.npz"
    archive.write_bytes(b"opaque independently assembled native pressure vector")
    path = tmp_path / "native.json"
    record = dict(
        source_sha256=sources,
        source_changed_during_run=False,
        material=material,
        omega=40 * np.pi,
        point_source=[5000, 50, 1.0],
        degree=4,
        geometry=[2048, 512],
        bounds=[0, 10240, 0, 2560],
        boundary="Top conforming Dirichlet zero; other sides outgoing first-order absorption zero",
        stored_original_residual=2e-15,
        archives=[{"archive": archive.name, "sha256": file_digest(archive)}],
        runtime_provenance={
            "observed_before_execution_at_utc": "2026-10-03T00:00:00+00:00",
            "upstream": {
                "project": "FEniCS/dolfinx",
                "revision": "6443e3b27d29aec04ce83d8424b57c339e35f865",
                "revision_url": (
                    "https://github.com/FEniCS/dolfinx/commit/"
                    "6443e3b27d29aec04ce83d8424b57c339e35f865"
                ),
                "verification": "Fixture attribution from the separate primary-source audit",
            },
            "loaded_cpp_extension_sha256": "f" * 64,
            "loaded_libdolfinx_sha256": "e" * 64,
            "verification_limits": ["Fixture attribution; no native execution claimed"],
        },
    )
    path.write_text(json.dumps(record))
    return path, record


def test_current_original_gates_and_exact_archive_are_required(mhm, reference):
    """Valid records keep both source and numerical vectors instead of changing their metadata."""
    path, record = mhm
    assert driver.checked_mhm(path) == record
    path, record = reference
    assert driver.checked_reference(path) == record


@pytest.mark.parametrize("degree", range(5))
def test_standalone_executed_order_depends_on_actual_trace_degree(mhm, degree):
    """Q3 standalone faces execute order eight except the degree-four rule nine."""
    path, record = mhm
    record.pop("prepared_trace_degree")
    record.update(
        trace_degree=degree, requested_assembly_order=8, assembly_order=9 if degree == 4 else 8
    )
    path.write_text(json.dumps(record))
    assert driver.checked_mhm(path) == record


@pytest.mark.parametrize("gate", ["original_field_trace_residual", "local_equation_residual_max"])
@pytest.mark.parametrize("value", [None, float("nan"), -1e-15, 2e-10, True])
def test_small_schur_residual_does_not_accept_bad_original_field_gates(mhm, gate, value):
    """The unchanged tiny reduced residual cannot override either physical field defect."""
    path, record = mhm
    assert record["algebraic_residual"] < 1e-10
    record[gate] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="accepted.*gate"):
        driver.checked_mhm(path)


@pytest.mark.parametrize(
    "defect", ["missing_core", "changed_core", "changed_archive", "quadrature", "crop", "boundary"]
)
def test_stale_or_different_acquisition_is_not_published(mhm, defect):
    """Identity validation happens independently of stored physical residual magnitudes."""
    path, record = mhm
    if defect == "missing_core":
        record["source_sha256"].pop("src/pymhm/_legacy/models/waves/helmholtz.py")
    elif defect == "changed_core":
        record["source_sha256"]["src/pymhm/_legacy/models/waves/helmholtz.py"] = "0" * 64
    elif defect == "changed_archive":
        (path.parent / record["archive"]).write_bytes(b"different coefficients")
    elif defect == "quadrature":
        record["assembly_order"] = 8
    elif defect == "crop":
        record["material"]["crop_origin_m"] = [3390.0, 515.0]
    else:
        record["boundary"] = "all sides Dirichlet"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        driver.checked_mhm(path)


@pytest.mark.parametrize("defect", ["residual", "runtime", "geometry", "boundary", "archive"])
def test_native_reference_requires_own_operator_and_runtime_evidence(reference, defect):
    """Attribution or physical native-field failures are not replaced by MHM residuals."""
    path, record = reference
    if defect == "residual":
        record["stored_original_residual"] = 2e-10
    elif defect == "runtime":
        record.pop("runtime_provenance")
    elif defect == "geometry":
        record["geometry"] = [32, 16]
    elif defect == "boundary":
        record["boundary"] = "all exterior impedance"
    else:
        record["archives"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        driver.checked_reference(path)


def norm_values(order):
    """Retain global pressure and explicit derivative-domain areas in opaque fixture norms."""
    values = dict(
        quadrature_order=order,
        gradient_exclusion=[5000.0, 50.0, 50.0],
        gradient_cutout=None,
        domain_area=10240 * 2560,
        gradient_domain_area=10240 * 2560 - np.pi * 50**2,
    )
    for name in ("pressure", "gradient", "weighted_gradient", "flux", "graph"):
        values.update(
            {
                f"{name}_difference": 1.0,
                f"reference_{name}_norm": 2.0,
                f"{name}_relative_difference": 0.5,
            }
        )
    return values


def comparison(tmp_path, sources, mhm, reference):
    """Retain pressure on the complete grid and the same disk for each derivative norm."""
    candidate, _ = mhm
    native, _ = reference
    norms = {str(order): norm_values(order) for order in (8, 10)}
    record = dict(
        source_sha256=sources,
        source_changed_during_run=False,
        candidate_record_sha256=file_digest(candidate),
        reference_record_sha256=file_digest(native),
        norms=norms,
        sampled_pressure=[
            dict(incident_side=[x, y], difference=1.0, reference_norm=2.0, relative_difference=0.5)
            for x in (-1, 1)
            for y in (-1, 1)
        ],
    )
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(record))
    return path, record


@pytest.mark.parametrize(
    "defect", [None, "field_chain", "mask", "quadrature", "nan", "incident_side"]
)
def test_physical_comparison_chain_and_norm_conventions(tmp_path, sources, mhm, reference, defect):
    """Do not merge incident values or change a derivative denominator's physical domain."""
    path, record = comparison(tmp_path, sources, mhm, reference)
    candidate, _ = mhm
    native, _ = reference
    if defect == "field_chain":
        record["candidate_record_sha256"] = "0" * 64
    elif defect == "mask":
        record["norms"]["10"]["gradient_exclusion"] = [5000.0, 50.0, 25.0]
    elif defect == "quadrature":
        record["norms"].pop("8")
    elif defect == "nan":
        record["norms"]["10"]["flux_difference"] = float("nan")
    elif defect == "incident_side":
        record["sampled_pressure"][0]["incident_side"] = [1, 1]
    path.write_text(json.dumps(record))
    if defect is None:
        assert driver.checked_comparison(path, candidate, native) == record
    else:
        with pytest.raises(ValueError):
            driver.checked_comparison(path, candidate, native)


@pytest.mark.parametrize(
    "defect", [None, "missing_level", "different_mask", "missing_norm", "different_finest"]
)
def test_reference_refinement_uses_complete_levels_and_identical_masks(
    tmp_path, sources, reference, defect
):
    """Refinement evidence must describe the same P4 denominator under each mask control."""
    finest_path, native = reference
    rows = []
    for degree in (1, 2, 3, 4):
        path = finest_path if degree == 4 else tmp_path / f"native-p{degree}.json"
        field_record = dict(native, degree=degree)
        path.write_text(json.dumps(field_record))
        sample = tmp_path / f"samples-p{degree}.npz"
        sample.write_bytes(b"opaque sampled reference pressure")
        row = dict(
            degree=degree,
            reference_record=path.name,
            reference_record_sha256=file_digest(path),
            sample_archive=sample.name,
            sample_archive_sha256=file_digest(sample),
            physical_norms=norm_values(8),
        )
        if degree > 1:
            row.update(increment=norm_values(6), increment_quadrature_check=norm_values(8))
        rows.append(row)
    record = dict(source_sha256=sources, source_changed_during_run=False, references=rows)
    path = tmp_path / "convergence.json"
    if defect == "missing_level":
        rows.pop(1)
    elif defect == "different_mask":
        rows[-1]["increment_quadrature_check"]["gradient_exclusion"] = [5000.0, 50.0, 25.0]
    elif defect == "missing_norm":
        rows[-1]["increment"].pop("graph_relative_difference")
    elif defect == "different_finest":
        finest_path = tmp_path / rows[0]["reference_record"]
    path.write_text(json.dumps(record))
    if defect is None:
        assert driver.checked_convergence(path, finest_path) == record
    else:
        with pytest.raises(ValueError):
            driver.checked_convergence(path, finest_path)
