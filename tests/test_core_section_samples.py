"""Display sections preserve cell ownership and evaluate polynomial variation."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from pymhm.tetrahedral import TetraMesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "core_section_samples", ROOT / "examples/sample_core_sections.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_section_uses_private_vertices_and_full_polynomial() -> None:
    """Two coincident independent cells retain different traces and quadratic interior values."""
    mesh = TetraMesh(np.vstack((np.zeros(3), np.eye(3))), np.array([[0, 1, 2, 3]]))

    def evaluate(macro: int, cell: int, points: np.ndarray) -> np.ndarray:
        """Specify independent quadratic pressure and affine flux on each cell."""
        assert cell == 0
        x, y, z = points.T
        return np.column_stack((macro + x**2 + 2 * y**2 + z, 1 + x, 2 * y, 3 * z))

    values = module.section_samples((mesh, mesh), evaluate, 4)
    n = len(values["points"]) // 2
    assert_allclose(values["points"][:n], values["points"][n:])
    assert_allclose(values["actual"][n:, 0] - values["actual"][:n, 0], 1)
    for macro in (0, 1):
        subset = values["owners"][:, 0] == macro
        assert_allclose(values["actual"][subset], evaluate(macro, 0, values["points"][subset]))
    assert np.ptp(values["actual"][:n, 0]) > 0.1
    # Display elements never combine the two independently evaluated traces.
    ownership = values["owners"][values["cells"], 0]
    assert np.all(ownership == ownership[:, :1])
    polygon = module.cut_polygon(mesh.points)
    area = 0.5 * abs(np.linalg.det((polygon[1:] - polygon[0])[:, :2]))
    triangles = values["points"][values["cells"], :2]
    display_area = abs(np.linalg.det(triangles[:, 1:] - triangles[:, :1])).sum() / 2
    assert_allclose(area, (1 - module.HEIGHT) ** 2 / 2)
    assert_allclose(display_area, 2 * area)
