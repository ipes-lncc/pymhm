"""Public mathematical callbacks compose with the shared adaptive numerical policy."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.formulations.application import brinkman, darcy, flow
from pymhm import FaceSpace, SkeletonSpace
from pymhm.adaptivity.darcy import solve_adaptive_darcy
from pymhm.adaptivity.darcy_balanced import solve_balanced_adaptive_darcy
from pymhm.adaptivity.flow import adapt_flow
from pymhm.adaptivity.flow_macro import adapt_flow_macros
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("formulation,degree", [("taylor-hood", 2), ("usfem", 2), ("oseen", 3)])
def test_brinkman_selects_explicit_mathematical_formulation(formulation: str, degree: int) -> None:
    """Tensor drag, affine divergence-free velocity and pressure verify selectable forms."""
    tensor = np.array([[3.0, 0.5], [0.5, 2.0]])

    def velocity(points: np.ndarray) -> np.ndarray:
        """Independent affine velocity has zero divergence and Laplacian."""
        return np.column_stack((points[:, 1], -points[:, 0]))

    def force(points: np.ndarray) -> np.ndarray:
        """The operator applied analytically is D*u+grad(p)."""
        return velocity(points) @ tensor.T + [1.0, 2.0]

    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    solution = brinkman(
        mesh,
        skeleton=skeleton,
        formulation=formulation,
        degree=degree,
        local_refinement=2,
        drag=tensor,
        source=force,
        dirichlet=velocity,
    )
    assert solution.l2_error(velocity) < 1e-11
    assert solution.pressure_l2_error(lambda x: x[:, 0] + 2 * x[:, 1] - 1.5) < 1e-10
    assert solution.hybrid.residual < 1e-10
    default = brinkman(TriangleMesh.unit_square(), degree=2)
    assert default.pressure_degree == 1
    assert default.l2_error((0.0, 0.0)) < 1e-12


@pytest.mark.parametrize("balanced", [False, True])
def test_darcy_callback_retains_mixed_boundary_ancestry_and_estimates(balanced: bool) -> None:
    """Each transferred state solves declared forms; prepared defaults give the same fields."""
    mesh = TriangleMesh.unit_square()
    natural = {int(face): 0.0 for face in mesh.boundary_faces if mesh.normals[face, 0] > 0.5}
    states: list[tuple[TriangleMesh, dict[str, Any]]] = []

    def equations(current: TriangleMesh, **data: Any) -> Any:
        """Record the public physical data then execute the mathematical provider."""
        states.append((current, data))
        return darcy(current, **data)

    callback = solve_balanced_adaptive_darcy if balanced else solve_adaptive_darcy
    options: dict[str, Any] = dict(
        iterations=2,
        degree=2,
        local_refinement=2,
        estimator_order=5,
        reconstruction_degree=2,
        source=1.0,
        dirichlet=2.0,
        neumann=natural,
    )
    if balanced:
        options["local_error_ratio"] = 1e3
    own = callback(mesh, solve_step=equations, **options)
    previous = callback(mesh, **options)
    own_result = own.result if balanced else own
    previous_result = previous.result if balanced else previous
    assert len(states) == len(own_result.solutions) == 2
    assert len(states[-1][0].cells) > len(mesh.cells)
    assert len(states[-1][1]["neumann"]) > len(natural)
    for solution, reference, indicator, ref_indicator in zip(
        own_result.solutions,
        previous_result.solutions,
        own_result.estimators,
        previous_result.estimators,
        strict=True,
    ):
        for first, second in zip(solution.pressure, reference.pressure, strict=True):
            assert_allclose(first, second, atol=1e-12, rtol=1e-10)
        assert_allclose(
            indicator.local_squared, ref_indicator.local_squared, atol=1e-12, rtol=1e-10
        )
        assert np.max(np.abs(solution.conservation_residuals())) < 1e-12


@pytest.mark.parametrize("macro", [False, True])
def test_flow_callback_retains_current_spaces_and_physical_pressure(macro: bool) -> None:
    """User-defined GaLS blocks reproduce adaptive fields and their unmodified indicators."""
    mesh = TriangleMesh.unit_square()
    states: list[tuple[TriangleMesh, dict[str, Any]]] = []

    def equations(current: TriangleMesh, **data: Any) -> Any:
        """Build the current velocity/pressure forms through the explicit provider."""
        states.append((current, data))
        return flow(current, **data)

    callback = adapt_flow_macros if macro else adapt_flow
    options = dict(iterations=1, source=lambda x: 2 * x, estimator_order=5)
    own = callback(mesh, solve_step=equations, **options)
    previous = callback(mesh, **options)
    assert len(states) == len(own.solutions) == 2
    for (current, data), solution, reference, indicator, ref_indicator in zip(
        states,
        own.solutions,
        previous.solutions,
        own.estimators,
        previous.estimators,
        strict=True,
    ):
        assert current is solution.skeleton.mesh
        assert data["skeleton"] is solution.skeleton
        for first, second in zip(solution.pressure, reference.pressure, strict=True):
            assert_allclose(first, second, atol=1e-12, rtol=1e-10)
        assert_allclose(
            indicator.local_squared, ref_indicator.local_squared, atol=1e-12, rtol=1e-10
        )
        assert solution.hybrid.residual < 1e-10


@pytest.mark.parametrize(
    "callback", [solve_adaptive_darcy, solve_balanced_adaptive_darcy, adapt_flow, adapt_flow_macros]
)
def test_invalid_solve_step_is_rejected_before_any_pde(callback: Any) -> None:
    """Invalid callbacks fail explicitly rather than entering the ready method default."""
    with pytest.raises(TypeError, match="solve_step"):
        callback(TriangleMesh.unit_square(), solve_step=1)


def test_declared_face_space_is_preserved_on_new_macrofaces() -> None:
    """A custom declared P1 face space survives callback-driven macro refinement."""
    space = FaceSpace((0.0, 0.5, 1.0), (1, 1))
    result = solve_adaptive_darcy(
        TriangleMesh.unit_square(),
        solve_step=darcy,
        trace_space=space,
        iterations=2,
        degree=3,
        local_refinement=2,
        source=1.0,
        reconstruction_degree=2,
        estimator_order=6,
    )
    assert len(result.solutions) == 2
    assert all(face == space for face in result.solutions[0].skeleton.faces)
    refined = result.refinements[0]
    for face, parent in enumerate(refined.parent_faces):
        if parent < 0:
            assert result.solutions[-1].skeleton.faces[face] == space


def test_equation_factory_adapter_uses_shared_adaptive_owner() -> None:
    """A formulation factory composes without reimplementing marking or mesh ancestry."""
    from examples.formulations.adaptive import refine_declared_pressure
    from examples.formulations.darcy import define_darcy

    def equations(mesh: TriangleMesh, skeleton: SkeletonSpace) -> Any:
        """The declaration owns polynomial energy and nonzero source on this mesh."""
        return define_darcy(mesh, skeleton=skeleton, degree=2, local_refinement=2, source=1.0)

    result = refine_declared_pressure(TriangleMesh.unit_square(), equations, iterations=2)
    assert len(result.solutions) == 2
    assert len(result.solutions[-1].skeleton.mesh.cells) > 2
    with pytest.raises(TypeError, match="estimator options"):
        refine_declared_pressure(
            TriangleMesh.unit_square(), equations, estimator_options={"unsupported": True}
        )
