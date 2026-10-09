"""Execute genuine UFL primary equations from the physical introductions."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytestmark = pytest.mark.fem
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "family,retained",
    [("flow", 2), ("elasticity", 3), ("transport", 1), ("waves/helmholtz", 0)],
)
def test_user_native_affine_controls_recover_manufactured_fields(
    family: str,
    retained: int,
    monkeypatch: pytest.MonkeyPatch,
    prepare_notebook_companion: Callable[[str], Path],
) -> None:
    """Execute the preserved native affine controls independently of lesson ordering."""
    pytest.importorskip("dolfinx")
    workspace = prepare_notebook_companion(f"{family}/introductory_methods.ipynb")
    monkeypatch.chdir(workspace)
    path = ROOT / "notebooks" / family / "introductory_methods.ipynb"
    notebook = json.loads(path.read_text())
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    controls = [
        source
        for source in code[1:]
        if any(
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "native_available"
                for target in statement.targets
            )
            for statement in ast.parse(source).body
        )
    ]
    assert len(controls) == 1
    scope: dict[str, Any] = {"__name__": "__main__", "np": np}
    for source in (code[0], controls[0]):
        exec(compile(source, str(path), "exec"), scope)
    assert scope["ROOT"] == workspace
    assert scope["native_available"] is True
    system, solution = scope["system"], scope["solution"]
    assert_array_equal(system.kernel_offsets - system.trace_size, (0, retained, 2 * retained))
    assert solution.raw_residual < 1e-12
    for response, field in zip(system.responses, solution.fields, strict=True):
        problem = response.problem
        rows = problem.matrix @ field + problem.coupling @ solution.trace[problem.trace_dofs]
        assert_allclose(rows, problem.load, atol=2e-12, rtol=0)
    if family == "flow":
        assert max(scope["errors"].values()) < 1e-10
        pressure_integral = sum(
            float(weights @ field)
            for weights, field in zip(scope["weights"], solution.fields, strict=True)
        )
        assert_allclose(pressure_integral, 0.7, atol=2e-13, rtol=0)
        # P1 trace coefficients represent the declared pseudotraction directly.
        expected = []
        mesh = scope["mesh"]
        gradient = scope["gradient"]
        for endpoints, normal in zip(mesh.points[mesh.faces], mesh.normals, strict=True):
            pressure = scope["exact_pressure"](endpoints)
            values = -gradient @ normal + pressure[:, None] * normal
            expected.extend(np.r_[values.mean(axis=0), (values[1] - values[0]) / 2])
        assert_allclose(solution.trace, expected, atol=2e-12, rtol=0)
    else:
        assert max(scope["errors"]) < 1e-10
