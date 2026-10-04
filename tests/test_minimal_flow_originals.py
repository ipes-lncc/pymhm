"""Check that field blocks and physical gauges cannot be hidden by a global row scale."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from examples.core_elasticity_field_archive import ProductionObservation
from examples.minimal_flow_originals import original_diagnostics, write_record


def _case(*, pressure_error=0.0, fixed=False, homogeneous=False):
    """Create declared exact rows without running another simultaneous PDE solve."""
    matrix = sparse.diags([1e10, 1.0], format="csr")
    coupling = np.array([[1.0], [0.0]]) if fixed else np.zeros((2, 1))
    values = np.array([1.0, 1.0 + pressure_error])
    trace = np.array([2.0 if fixed else 0.0])
    load = np.array([1e10 + (2 if fixed else 0), 1.0])
    if homogeneous:
        values[:] = 0
        load[:] = 0
    problem = SimpleNamespace(
        matrix=matrix,
        coupling=coupling,
        test_coupling=coupling,
        load=load,
        trace_dofs=np.array([0]),
    )
    system = SimpleNamespace(responses=[SimpleNamespace(problem=problem)])
    observation = ProductionObservation(
        system=system,
        applied_boundary=np.zeros(1),
        fixed={0: 2.0} if fixed else {},
    )
    solution = SimpleNamespace(
        hybrid=SimpleNamespace(
            trace=trace,
            coarse=[np.zeros(1)],
            fields=[values],
            gauge_multipliers=np.zeros(0),
        )
    )
    return solution, observation


def test_separate_pressure_block_rejects_error_hidden_by_momentum_scale():
    solution, observed = _case(pressure_error=1e-4)
    result = original_diagnostics(solution, observed, [{"momentum": 1, "continuity": 1}])
    assert result["full_uncondensed_relative_to_physical_rhs"] < 1e-10
    assert result["maximum_original_block_backward_error"] > 1e-5
    assert not result["accepted"]


def test_prescribed_trace_rows_do_not_enter_free_original_equations():
    solution, observed = _case(fixed=True)
    observed.applied_boundary = np.array([3.0])
    result = original_diagnostics(solution, observed, [{"momentum": 1, "continuity": 1}])
    assert result["accepted"]
    assert result["global_weak_row_absolute_residual_norm"] == 0
    observed.fixed.clear()
    assert not original_diagnostics(solution, observed, [{"momentum": 1, "continuity": 1}])[
        "accepted"
    ]


def test_physical_mean_is_checked_independently_of_original_rows():
    solution, observed = _case()
    observed.mean_weights = [(np.array([0.0, 1.0]),)]
    observed.mean_values = [2.0]
    result = original_diagnostics(solution, observed, [{"momentum": 1, "continuity": 1}])
    assert result["full_uncondensed_relative_to_physical_rhs"] == 0
    assert result["physical_mean_relative_defect"] == pytest.approx(1 / 3)
    assert not result["accepted"]


def test_homogeneous_original_rows_use_zero_absolute_fallback():
    solution, observed = _case(homogeneous=True)
    result = original_diagnostics(solution, observed, [{"momentum": 1, "continuity": 1}])
    assert result["accepted"]
    assert result["full_uncondensed_relative_to_physical_rhs"] == 0
    assert result["physical_rhs_row_norm"] == 0


@pytest.mark.parametrize("blocks", [{"one": 0}, {"one": 3}, {"one": 1}])
def test_every_original_row_requires_a_valid_declared_physical_block(blocks):
    solution, observed = _case()
    with pytest.raises(ValueError):
        original_diagnostics(solution, observed, [blocks])


def test_real_numpy_scalar_records_preserve_acceptance_and_norms(tmp_path):
    path = tmp_path / "record.json"
    write_record(path, {"accepted": np.bool_(True), "n": np.int64(3), "norm": np.float64(0.25)})
    assert json.loads(path.read_text()) == {"accepted": True, "n": 3, "norm": 0.25}


@pytest.mark.parametrize("value", [np.ones(2), np.complex128(1j), np.float64(np.nan)])
def test_scalar_records_reject_arrays_complex_and_nonfinite_values(tmp_path, value):
    with pytest.raises((TypeError, ValueError)):
        write_record(tmp_path / "record.json", {"field": value})
