"""Isotropic diffusion validation and unchanged tensor-valued material contracts."""

from typing import Any

import numpy as np
import pytest

from pymhm.materials.evaluation import tensor_values, tensor_values_3d


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("as_callable", [False, True])
@pytest.mark.parametrize("pointwise", [False, True])
def test_positive_isotropic_diffusion_skips_spectral_work(
    monkeypatch: pytest.MonkeyPatch, dimension: int, as_callable: bool, pointwise: bool
) -> None:
    """Positive subnormal and ordinary scalars retain exact diagonal expansion."""
    points = np.zeros((4, dimension))
    smallest = np.nextafter(0.0, 1.0)
    values = np.array([smallest, np.finfo(float).tiny, 1.5, np.finfo(float).max])
    coefficient: Any = values if pointwise else smallest
    expected = np.broadcast_to(coefficient, (len(points),))[:, None, None] * np.eye(dimension)

    def reject_spectral_call(matrix: Any) -> Any:
        """Fail if scalar diffusion takes the matrix-only eigenvalue path."""
        raise AssertionError("Scalar isotropic diffusion does not need eigvalsh")

    monkeypatch.setattr(np.linalg, "eigvalsh", reject_spectral_call)
    evaluator = tensor_values if dimension == 2 else tensor_values_3d
    field = (lambda samples: coefficient) if as_callable else coefficient
    result = evaluator(field, points)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.dtype(np.float64)
    assert result.flags.writeable == (dimension == 2)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("pointwise", [False, True])
def test_empty_isotropic_queries_preserve_tensor_layout(dimension: int, pointwise: bool) -> None:
    """A valid empty sample batch retains dtype, tensor axes and storage flags."""
    points = np.empty((0, dimension))
    coefficient: Any = np.empty(0) if pointwise else 1.5
    evaluator = tensor_values if dimension == 2 else tensor_values_3d
    result = evaluator(coefficient, points)
    assert result.shape == (0, dimension, dimension)
    assert result.dtype == np.dtype(np.float64)
    assert result.flags.writeable == (dimension == 2)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("coefficient", [0.0, -0.0, -np.nextafter(0.0, 1.0), np.nan, np.inf])
def test_empty_queries_do_not_admit_invalid_scalar_diffusion(
    dimension: int, coefficient: float
) -> None:
    """Declared scalar diffusion stays finite and strictly positive without samples."""
    evaluator = tensor_values if dimension == 2 else tensor_values_3d
    with pytest.raises(ValueError):
        evaluator(coefficient, np.empty((0, dimension)))


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("as_callable", [False, True])
@pytest.mark.parametrize("pointwise", [False, True])
@pytest.mark.parametrize(
    "bad", [0.0, -0.0, -np.nextafter(0.0, 1.0), -1.0, np.nan, np.inf, -np.inf, 1.0 + 0.0j]
)
def test_invalid_isotropic_diffusion_remains_rejected(
    dimension: int, as_callable: bool, pointwise: bool, bad: Any
) -> None:
    """Scalar and per-point paths reject nonpositive, nonfinite and complex data."""
    points = np.zeros((4, dimension))
    coefficient = np.array([1.0, bad, 2.0, 3.0]) if pointwise else bad
    field = (lambda samples: coefficient) if as_callable else coefficient
    evaluator = tensor_values if dimension == 2 else tensor_values_3d
    with pytest.raises(ValueError):
        evaluator(field, points)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("as_callable", [False, True])
@pytest.mark.parametrize("shape", [(3,), (4, 1), (4, 2, 3)])
def test_diffusion_shape_contract_is_not_broadened(
    dimension: int, as_callable: bool, shape: tuple[int, ...]
) -> None:
    """Wrong scalar counts and incompatible tensor axes still fail explicitly."""
    points = np.zeros((4, dimension))
    coefficient = np.ones(shape)
    field = (lambda samples: coefficient) if as_callable else coefficient
    evaluator = tensor_values if dimension == 2 else tensor_values_3d
    with pytest.raises(ValueError):
        evaluator(field, points)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("pointwise", [False, True])
def test_matrix_diffusion_retains_eigenvalue_check_and_scalar_equivalence(
    monkeypatch: pytest.MonkeyPatch, dimension: int, pointwise: bool
) -> None:
    """Even diagonal matrix inputs take eigvalsh and match the scalar field exactly."""
    points = np.zeros((4, dimension))
    scalars: Any = np.array([0.5, 1.5, 2.0, 3.0]) if pointwise else 1.5
    tensors = np.broadcast_to(scalars, (len(points),))[:, None, None] * np.eye(dimension)
    coefficient = tensors if pointwise else 1.5 * np.eye(dimension)
    original = np.linalg.eigvalsh
    observed = []

    def record_spectral_call(matrix: Any) -> Any:
        """Record actual matrix positivity validation while preserving its operation."""
        observed.append(np.array(matrix, copy=True))
        return original(matrix)

    monkeypatch.setattr(np.linalg, "eigvalsh", record_spectral_call)
    evaluator = tensor_values if dimension == 2 else tensor_values_3d
    matrix_values = evaluator(coefficient, points)
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], tensors)
    scalar_values = evaluator(scalars, points)
    assert len(observed) == 1
    np.testing.assert_array_equal(matrix_values, scalar_values)
    assert matrix_values.dtype == scalar_values.dtype == np.dtype(np.float64)
    assert matrix_values.flags.writeable == scalar_values.flags.writeable == (dimension == 2)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("failure", ["asymmetric", "indefinite", "zero", "nonfinite", "complex"])
def test_invalid_matrix_diffusion_still_fails(dimension: int, failure: str) -> None:
    """The faster scalar path cannot admit invalid full diffusion tensors."""
    points = np.zeros((4, dimension))
    tensor = np.eye(dimension)
    if failure == "asymmetric":
        tensor[0, 1] = 0.25
    elif failure == "indefinite":
        tensor[0, 1] = tensor[1, 0] = 2.0
    elif failure == "zero":
        tensor[-1, -1] = 0.0
    elif failure == "nonfinite":
        tensor[-1, -1] = np.inf
    else:
        tensor = tensor.astype(complex)
    evaluator = tensor_values if dimension == 2 else tensor_values_3d
    with pytest.raises(ValueError):
        evaluator(tensor, points)
