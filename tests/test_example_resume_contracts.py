"""Original example adapters reject incompatible numerical checkpoint reuse."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def example(monkeypatch):
    """Resolve examples without a dependency on the caller's current directory."""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "examples"))
    return lambda name: importlib.import_module("examples." + name)


def test_frequency_filename_collision_rejected_before_solve(example, tmp_path, monkeypatch):
    """Frequency formatting cannot alias two distinct mathematical wave numbers."""
    m = example("helmholtz_stability")
    # Build a legitimate empty checkpoint without performing any solve.
    m.run(tmp_path, 0, 10, [], 1)
    # run writes only completed rows; construct its declared source and identity instead.
    sources = m.source_hashes()
    for path in (Path(m.__file__), m.ROOT / "examples/helmholtz_threshold.py"):
        sources[path.relative_to(m.ROOT).as_posix()] = m.hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    record = dict(
        source_sha256=sources,
        ell=0,
        omega=20 * np.pi,
        local_space="Q3 on 2x2 fine cells per macrocell",
        requested_assembly_order=12,
        norm_order=16,
        projection_order=24,
        rows=[],
    )
    path = tmp_path / "ell0-frequency10.json"
    raw = json.dumps(record)
    path.write_text(raw)

    def unexpected_solve(*args, **kwargs):
        raise AssertionError("filename collisions must be rejected before solving")

    monkeypatch.setattr(m, "solve_acoustic", unexpected_solve)
    with pytest.raises(ValueError, match="identity"):
        m.run(tmp_path, 0, 10.000001, [8], 1)
    assert path.read_text() == raw


def test_primal_precision_and_source_changes_rejected(example, tmp_path):
    """A double-precision control cannot fulfill an explicitly extended request."""
    m = example("solve_gals3d")
    sources = {"operator.py": "operator", "examples/solve_gals3d.py": "driver"}
    row = dict(
        lame_lambda=1.0,
        degree=2,
        local_refinement=2,
        macro_subdivisions=2,
        local_refinement_precision="double",
        assembly_quadrature_order=7,
        error_quadrature_order=8,
        backward_residual=1e-15,
        displacement_l2=0.1,
        stress_l2=0.1,
    )
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(dict(source_sha256=sources, rows=[], primal_control=[row])))
    assert (
        m.resume_report(path, sources)["primal_control"][0]["local_refinement_precision"]
        == "double"
    )
    raw = path.read_bytes()
    with pytest.raises(ValueError, match="identity"):
        m.resume_report(path, sources, primal_precision="extended")
    with pytest.raises(ValueError, match="sources differ"):
        m.resume_report(path, {**sources, "examples/solve_gals3d.py": "changed"})
    assert path.read_bytes() == raw


def test_article_complete_configuration_and_archive(example, tmp_path):
    """A stored key cannot conceal a changed omega, incomplete norms or changed field."""
    m = example("helmholtz_article")
    c = m.Configuration("convergence", 8, 2, False, 10.0, 0.1, archive_fields=True)
    p = tmp_path / "field.npz"
    p.write_bytes(b"original")
    row = {
        **c.__dict__,
        "key": c.key,
        "local_degree": 4,
        "requested_assembly_order": 10,
        "assembly_order": 10,
        "error_order": 12,
        "trace_basis": "polynomial",
        "macro_edge_length": 1 / c.n,
        "macro_diameter": np.sqrt(2) / c.n,
        **m.archive_identity(p),
        **dict.fromkeys(
            (
                "pressure_l2",
                "gradient_l2",
                "pressure_exact_l2",
                "gradient_exact_l2",
                "pressure_relative_error",
                "gradient_relative_error",
                "energy_relative_error",
                "residual",
                "macro_balance_max",
            ),
            0.1,
        ),
        "original_field_trace_residual": 1e-15,
        "original_local_equation_residual_max": 1e-15,
    }
    m.validate_row(row, tmp_path)
    for changed in (
        {"omega": 11.0},
        {"local_degree": 5},
        {"pressure_l2": None},
        {"requested_assembly_order": 12},
        {"original_field_trace_residual": 2e-10},
        {"assembly_order": 9},
        {"error_order": 8},
        {"trace_basis": "oscillatory"},
        {"macro_edge_length": 1 / (c.n + 1)},
    ):
        with pytest.raises(ValueError):
            m.validate_row({**row, **changed}, tmp_path)
    for key in ("requested_assembly_order", "assembly_order", "error_order"):
        incomplete = dict(row)
        incomplete.pop(key)
        with pytest.raises(ValueError, match="mathematical identity"):
            m.validate_row(incomplete, tmp_path)
    p.write_bytes(b"different")
    with pytest.raises(ValueError, match="digest"):
        m.validate_row(row, tmp_path)


@pytest.mark.parametrize("published", [False, True])
def test_spe_material_policy_and_bytes(example, tmp_path, published):
    """A prior estimator/marking field is used only with unchanged PDE and policy."""
    m = example("solve_spe10_published" if published else "solve_spe10_balanced")
    p = tmp_path / "field.npz"
    p.write_bytes(b"accepted")
    helper = example("campaign_checkpoint")
    sources = {"material.npz": "material", "examples/solve_spe10.py": "boundary"}
    row = dict(
        local_degree=2,
        local_refinement=2,
        trace_degree=0,
        reconstruction_degree=2,
        material_fitted=not published,
        assembly_order=6,
        estimator_order=6,
        estimator_convention="published" if published else "energy",
        source_changed_during_solve=False,
        local_refinement_precision="extended",
        source_hashes=sources,
        **helper.archive_identity(p),
        **dict.fromkeys(
            (
                "estimator",
                "local_indicator",
                "flux_defect",
                "nonconformity",
                "divergence_defect",
                "oscillation",
                "macro_balance_linf",
            ),
            0.1,
        ),
    )
    if published:
        row.update(
            metric_coefficient=1.0,
            remesher_max_vertices=10000,
            macro_refinement="FreeFEM/BAMG isotropic residual metric",
            remesher_executable_sha256="binary",
        )
        arguments = (tmp_path, sources, "binary")
    else:
        row.update(theta=0.5, local_error_ratio=0.25)
        arguments = (tmp_path, sources)
    m.validate_state(row, *arguments)
    for changed in (
        {"trace_degree": 1},
        {"estimator_convention": "different"},
        {"source_changed_during_solve": True},
        {"source_hashes": {**sources, "material.npz": "other"}},
        {"source_hashes": {**sources, "examples/solve_spe10.py": "other"}},
    ):
        with pytest.raises(ValueError):
            m.validate_state({**row, **changed}, *arguments)
    p.write_bytes(b"other")
    with pytest.raises(ValueError, match="digest"):
        m.validate_state(row, *arguments)


def test_helmholtz_campaign_rejects_changed_wave_number(example, tmp_path):
    """A matching study/resolution label cannot reuse another physical frequency."""
    m = example("helmholtz_campaign")
    row = dict(
        study="convergence",
        wave="plane",
        resolution=8,
        trace_degree=2,
        trace_basis="polynomial",
        angle=float(np.pi / 13),
        omega=11.0,
        local_degree=4,
        local_refinement=2,
        pressure_l2=0.1,
        gradient_l2=0.2,
        energy_relative_error=0.1,
        residual=1e-15,
        macro_balance_max=1e-15,
        original_field_trace_residual=1e-15,
        original_local_equation_residual_max=1e-15,
    )
    path = tmp_path / "comparison.json"
    path.write_text(
        json.dumps(
            dict(
                source_sha256=m.source_hashes(),
                rows=[row],
                mathematical_configuration=dict(resolutions=[8], angle_count=1),
                assembly_quadrature=10,
                error_quadrature=12,
            )
        )
    )
    raw = path.read_bytes()
    with pytest.raises(ValueError, match="identity"):
        m.run(tmp_path, [8], 1, 1, resume=True)
    assert path.read_bytes() == raw


def test_mh_changed_field_rejected_before_vanishing_robin(example, tmp_path):
    """The smooth-phase archive is verified before starting a new continuation phase."""
    m = example("mh_campaign")
    p = tmp_path / "field.npz"
    p.write_bytes(b"original")
    rows = [
        dict(
            mesh=kind,
            trace_degree=ell,
            resolution=8 if kind == "triangles" else 4,
            local_degree=ell + 2,
            local_refinement=2,
            robin_parameter=0.25,
            pressure_error_l2=0.1,
            flux_error_l2=0.1,
            residual=1e-15,
            **m.archive_identity(p),
        )
        for kind in ("triangles", "L-polygons")
        for ell in (1, 2)
    ]
    path = tmp_path / "comparison.json"
    path.write_text(
        json.dumps(
            dict(
                source_sha256=m.source_hashes(),
                smooth=rows,
                assembly_quadrature=8,
                error_quadrature=10,
            )
        )
    )
    p.write_bytes(b"changed")
    raw = path.read_bytes()
    with pytest.raises(ValueError, match="digest"):
        m.run(tmp_path, [8], resume_smooth=True)
    assert path.read_bytes() == raw


def test_helmholtz_campaign_changed_grid_requires_new_output(example, tmp_path):
    """A changed study grid cannot overwrite a former midpoint field filename."""
    m = example("helmholtz_campaign")
    path = tmp_path / "comparison.json"
    path.write_text(
        json.dumps(
            dict(
                source_sha256=m.source_hashes(),
                rows=[],
                mathematical_configuration=dict(resolutions=[8], angle_count=1),
                assembly_quadrature=10,
                error_quadrature=12,
            )
        )
    )
    raw = path.read_bytes()
    for sizes, angles in (([8, 16], 1), ([8], 2)):
        with pytest.raises(ValueError, match="identity"):
            m.run(tmp_path, sizes, 1, angles, resume=True)
    assert path.read_bytes() == raw
