"""Execute each tutorial formulation with physical affine and zero data."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples import tutorial_vector_variants as tutorial
from examples.tutorial_vector_variants import VARIANTS, run_variant


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("homogeneous", [False, True])
def test_each_vector_tutorial_has_separate_physical_errors(variant: str, homogeneous: bool) -> None:
    row = run_variant(variant, homogeneous=homogeneous)
    assert row["macrocells"] == 2
    assert row["native_threads"] == 1
    assert row["dimension"] == (3 if variant.endswith("3d") else 2)
    assert row["scope"]
    assert max(row["errors"].values()) < 1e-9
    if variant == "maxwell-vector-3d":
        assert set(row["errors"]) == {"electric_l2", "magnetic_l2"}
        assert row["original_equation_residual"] is None
        assert row["electric_time"] == pytest.approx(0.0045)
        assert row["magnetic_time"] == pytest.approx(0.004)
        assert abs(row["energy_change"]) < 1e-12
        assert row["energy_balance_residual_max"] < 1e-12
    else:
        assert row["original_equation_residual"] is not None
        assert row["original_equation_residual"] < 1e-10
        gradient = np.asarray(row["gradient"])
        if variant.startswith("flow-"):
            assert set(row["errors"]) == {"velocity_l2", "pressure_l2", "divergence_l2"}
            assert_allclose(np.trace(gradient), 0, atol=1e-15)
            assert row["provenance"]["pressure_mean"] == (0 if homogeneous else 0.7)
        else:
            assert row["provenance"]["expected_pressure"] == -np.trace(gradient)
            assert_allclose(
                row["provenance"]["expected_stress"],
                gradient + gradient.T + np.trace(gradient) * np.eye(len(gradient)),
            )
            assert "displacement_l2" in row["errors"] and "stress_l2" in row["errors"]


@pytest.mark.parametrize(
    "variant",
    [name for name in VARIANTS if name.startswith("elasticity-") and "primal" not in name],
)
@pytest.mark.parametrize("homogeneous", [False, True])
def test_mixed_elasticity_incompressible_gauge_is_physical(variant: str, homogeneous: bool) -> None:
    row = run_variant(variant, incompressible=True, homogeneous=homogeneous)
    assert max(row["errors"].values()) < 1e-9
    gradient = np.asarray(row["gradient"])
    assert_allclose(np.trace(gradient), 0, atol=1e-15)
    expected_pressure = 0 if homogeneous else 2.3
    assert row["provenance"]["expected_pressure"] == expected_pressure
    assert row["provenance"]["pressure_role"] == "physical mean gauge"
    assert_allclose(
        row["provenance"]["expected_stress"],
        gradient + gradient.T - expected_pressure * np.eye(len(gradient)),
    )


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_vector_tutorial_callable_data_support_actual_workers(backend: Any) -> None:
    row = run_variant("elasticity-bdm-plus-2d", backend=backend, workers=2)
    assert row["backend"] == backend
    assert row["original_equation_residual"] < 1e-10
    assert max(row["errors"].values()) < 1e-9


def test_explicit_variant_and_material_restrictions() -> None:
    with pytest.raises(ValueError, match="declared"):
        run_variant("unsupported")
    for dimension in (2, 3):
        with pytest.raises(ValueError, match="finite lambda"):
            run_variant(f"elasticity-primal-{dimension}d", incompressible=True)
    for options in ({"backend": "process"}, {"incompressible": True}):
        with pytest.raises(ValueError, match="Maxwell stepper"):
            run_variant("maxwell-vector-3d", **options)


def test_tutorial_rejects_inaccurate_physical_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tutorial, "_elasticity", lambda *args, **kwargs: (None, {"displacement_l2": 0.01}, {})
    )
    with pytest.raises(RuntimeError, match="physical patch was not recovered"):
        run_variant("elasticity-primal-2d")


@pytest.mark.parametrize("field_error,energy_defect", [(0.01, 0.0), (0.0, 0.01)])
def test_maxwell_rejects_field_or_energy_defect(
    monkeypatch: pytest.MonkeyPatch, field_error: float, energy_defect: float
) -> None:
    class Stepper:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.solution = SimpleNamespace(
                energy_balance_residual=energy_defect,
                l2_errors=lambda *args, **kwargs: (field_error, 0.0),
            )

        def __enter__(self) -> Stepper:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def initialize(self, *args: Any) -> None:
            return None

        def advance(self) -> Any:
            return self.solution

    monkeypatch.setattr(tutorial, "MaxwellStepper", Stepper)
    with pytest.raises(RuntimeError, match="Maxwell patch was not recovered"):
        run_variant("maxwell-vector-3d")


def test_all_variants_cli_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["tutorial_vector_variants", "--variant", "all"])
    tutorial.main()
    record = json.loads(capsys.readouterr().out)
    assert record["variant_count"] == 17
    assert tuple(row["variant"] for row in record["rows"]) == VARIANTS


def test_default_cli_has_guarded_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["tutorial_vector_variants"])
    runpy.run_path(str(Path(tutorial.__file__)), run_name="__main__")
    record = json.loads(capsys.readouterr().out)
    assert record["variant_count"] == 1
    assert record["rows"][0]["variant"] == "elasticity-primal-2d"
