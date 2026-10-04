"""Physical norm and acquisition contracts of the published-curve comparison."""

import hashlib
import json
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

import examples.unfitted_convergence as campaign
from examples.unfitted_convergence import (
    error_norms,
    smooth_configurations,
    smooth_field,
    smooth_source,
)
from examples.unfitted_trace_family import ScalarTraceFamily
from pymhm.meshes.triangle import TriangleMesh


def test_distinct_pressure_gradient_flux_and_energy_norms_have_analytic_values():
    """Use a zero computed field and a nonzero exact quadratic with anisotropic K."""
    material = np.diag([2.0, 3.0])
    family = ScalarTraceFamily.prepare(
        TriangleMesh.unit_square(1),
        trace_degree=0,
        segments=1,
        local_degree=2,
        local_refinement=2,
        permeability=material,
    )
    solution, _ = family.solve(0, 1)

    def exact(points):
        """Provide pressure x²+2y² and its gradient independently."""
        x, y = points.T
        return x * x + 2 * y * y, np.column_stack((2 * x, 4 * y))

    result = error_norms(solution, exact, 5)
    for name, squared in zip(
        ("pressure", "gradient", "flux", "energy"), (13 / 9, 20 / 3, 160 / 3, 56 / 3), strict=True
    ):
        assert_allclose(result[f"{name}_absolute"], np.sqrt(squared), rtol=2e-14)
        assert result[f"{name}_relative"] == 1


def test_manufactured_derivatives_and_distinct_printed_trace_sweeps():
    """Check analytic gradients by complex steps and preserve all 27 distinct settings."""
    points = np.array([[0.13, 0.27], [0.39, 0.83]])
    values, gradient = smooth_field(points)
    assert_allclose(smooth_source(points), 8 * np.pi**2 * values)
    for axis in range(2):
        perturbed = points.astype(complex)
        perturbed[:, axis] += 1e-30j
        assert_allclose(smooth_field(perturbed)[0].imag / 1e-30, gradient[:, axis], rtol=3e-15)
    cases = smooth_configurations(32)
    assert len(cases) == len(set(cases)) == 27
    assert (4, 4) in cases and (3, 32) in cases and (4, 8) not in cases
    assert len(smooth_configurations(16)) == 23


def test_cli_quadrature_controls_isolate_fields_and_resume_provenance(monkeypatch, tmp_path):
    """Execute small acquisitions without mixing assembly orders or norm controls."""
    monkeypatch.setattr(campaign, "macro_mesh", lambda delta: TriangleMesh.unit_square(1))
    arguments = [
        "unfitted_convergence",
        "--study",
        "smooth",
        "--degree",
        "4",
        "--refinement",
        "2",
        "--maximum-segments",
        "1",
        "--names",
        "ell0-s1",
        "--output",
        str(tmp_path),
    ]

    def acquire(options):
        """Use the actual shared local owner and CLI acquisition path."""
        monkeypatch.setattr(sys, "argv", arguments + options)
        campaign.main()

    acquire([])
    default = json.loads((tmp_path / "smooth-p4-r2.json").read_text())
    assert default["configuration"]["assembly_order"] == 7
    assert default["configuration"]["requested_assembly_order"] == 7
    assert default["configuration"]["norm_orders"] == [7, 9]
    archive = tmp_path / default["cases"][0]["archive"]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == default["cases"][0]["archive_sha256"]

    acquire(["--assembly-order", "1"])
    floored = json.loads((tmp_path / "smooth-p4-r2-q6.json").read_text())
    assert floored["configuration"]["requested_assembly_order"] == 1
    assert floored["configuration"]["assembly_order"] == 6

    acquire(["--assembly-order", "13"])
    higher = json.loads((tmp_path / "smooth-p4-r2-q13.json").read_text())
    assert higher["configuration"]["assembly_order"] == 13
    assert higher["configuration"]["requested_assembly_order"] == 13
    assert higher["configuration"]["norm_orders"] == [7, 9]
    assert higher["cases"][0]["archive"] == "smooth-p4-r2-q13-ell0-s1.npz"
    assert higher["complete"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest

    acquire(["--assembly-order", "13", "--norm-orders", "15", "13"])
    norms = json.loads((tmp_path / "smooth-p4-r2-q13-nq13-15.json").read_text())
    assert norms["configuration"]["norm_orders"] == [13, 15]
    assert set(norms["cases"][0]["norms"]) == {"quadrature_13", "quadrature_15"}
    assert norms["cases"][0]["archive"] != higher["cases"][0]["archive"]
    with (
        np.load(tmp_path / higher["cases"][0]["archive"]) as original,
        np.load(tmp_path / norms["cases"][0]["archive"]) as independent_norms,
    ):
        assert original.files == independent_norms.files
        for name in original.files:
            assert_allclose(original[name], independent_norms[name], rtol=0, atol=0)

    def forbid_solve(self, degree, segments):
        """A complete matching acquisition must not recompute a finished row."""
        pytest.fail("matching acquisition did not resume its complete case")

    monkeypatch.setattr(ScalarTraceFamily, "solve", forbid_solve)
    acquire(["--assembly-order", "13"])
    assert len(json.loads((tmp_path / "smooth-p4-r2-q13.json").read_text())["cases"]) == 1
    higher["configuration"]["assembly_order"] = 11
    (tmp_path / "smooth-p4-r2-q13.json").write_text(json.dumps(higher))
    with pytest.raises(ValueError, match="configuration or sources differ"):
        acquire(["--assembly-order", "13"])


@pytest.mark.parametrize(
    "options, message",
    [
        (["--assembly-order", "0"], "quadrature_order"),
        (["--norm-orders", "0"], "norm_orders"),
        (["--norm-orders", "13", "13"], "distinct"),
    ],
)
def test_cli_rejects_invalid_quadrature_before_acquisition(monkeypatch, tmp_path, options, message):
    """Reject invalid rules without creating files or assembling a local operator."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unfitted_convergence",
            "--study",
            "smooth",
            "--refinement",
            "2",
            "--output",
            str(tmp_path / "not-created"),
            *options,
        ],
    )
    with pytest.raises(ValueError, match=message):
        campaign.main()
    assert not (tmp_path / "not-created").exists()
