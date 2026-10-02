"""Indicator-ranked macro refinement with an explicit mesh-complexity budget."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.adaptive_darcy import mark_dorfler
from pymhm.longest_edge import refine_longest_edge
from pymhm.mesh import TriangleMesh, positive_int
from pymhm.refinement import TriangleRefinement


@dataclass(frozen=True)
class DarcyBudgetRefinement:
    """A conforming refinement, its indicator prefix and the requested cell count.

    The actual count can exceed the target because of conformity propagation or
    the required bulk set. The construction does not claim minimum complexity
    among all conforming meshes or use an exact/reference solution for marking.
    """

    marked: np.ndarray
    refinement: TriangleRefinement
    target_cells: int
    captured_fraction: float


def refine_darcy_budget(
    mesh: TriangleMesh,
    local_squared: Any,
    *,
    target_cells: int,
    theta: float = 0.5,
    refiner: Callable[[TriangleMesh, Any], TriangleRefinement] = refine_longest_edge,
) -> DarcyBudgetRefinement:
    """Enlarge a Dörfler set along decreasing indicators to meet a cell budget.

    Start with the deterministic minimal bulk prefix. If its conforming closure
    already reaches the target, keep it. Otherwise bracket larger prefixes by
    the full marked set and search between tested prefixes. The returned set
    contains the bulk prefix and its actual conforming mesh has at least the
    target count; conformity can cause overshoot. No monotonicity of arbitrary
    user-supplied refiners or minimal-prefix optimality is assumed. A target
    beyond the full set's one-call refinement is rejected; use multiple levels.
    Exact cell/face ancestry is supplied unchanged by the selected refiner.
    """
    count = positive_int(target_cells, "target_cells")
    if count <= len(mesh.cells):
        raise ValueError("target_cells must exceed the current number of macro cells")
    marked = mark_dorfler(local_squared, theta)
    if marked.shape != (len(mesh.cells),):
        raise ValueError("provide one squared indicator per macro cell")
    if not np.any(marked):
        raise ValueError("a refinement budget requires positive indicators")
    normalized = np.asarray(local_squared, dtype=float)
    normalized = normalized / np.max(normalized)
    order = np.argsort(-normalized, kind="stable")
    refined = refiner(mesh, marked)
    if len(refined.mesh.cells) < count:
        left, right = int(marked.sum()), len(marked)
        chosen = np.ones(len(marked), dtype=bool)
        refined = refiner(mesh, chosen)
        if len(refined.mesh.cells) < count:
            raise ValueError("target_cells exceeds the full marked set's one-level refinement")
        while left + 1 < right:
            middle = (left + right) // 2
            candidate = np.zeros(len(marked), dtype=bool)
            candidate[order[:middle]] = True
            trial = refiner(mesh, candidate)
            if len(trial.mesh.cells) >= count:
                right, chosen, refined = middle, candidate, trial
            else:
                left = middle
        marked = chosen
    return DarcyBudgetRefinement(
        marked,
        refined,
        count,
        float(np.sum(normalized[marked]) / np.sum(normalized)),
    )
