"""Original-equation and quadrature contracts for the bounded initial wave studies."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from examples.elastodynamics_campaign import ElasticWave, norms
from examples.minimal_wave_convergence import quadrature_change, require_original, run
from examples.minimal_wave_elastic_wave import field_norms, original_update
from examples.minimal_wave_marmousi import physical_norm
from examples.minimal_wave_three_layer import physical_norms
from pymhm.fem.scalar.quadrilateral import _cardinals
from pymhm.fem.scalar.tetrahedron import tetra_tabulate, tetrahedron_quadrature
from pymhm.meshes.tetrahedron import TetraMesh


def test_newmark_original_force_and_kinematic_invariants() -> None:
    """A constant-force free motion satisfies every original block and physical work."""
    dt, force = 0.1, np.asarray([2.0, -3.0])
    local = SimpleNamespace(
        mass=sparse.eye(2),
        stiffness=sparse.csr_matrix((2, 2)),
        coupling=np.zeros((2, 1)),
        trace_dofs=np.asarray([0]),
    )
    u, v = dt**2 * force / 2, dt * force
    energy = float(v @ v) / 2
    stepper = SimpleNamespace(
        time_step=dt,
        locals=(local,),
        displacement=(u,),
        velocity=(v,),
        trace=np.zeros(1),
        boundary=np.zeros(1),
        free=np.asarray([0]),
        solution=lambda: SimpleNamespace(energy=energy),
        _moments=lambda fields: np.zeros(1),
    )
    checks = original_update(
        stepper, (np.zeros(2),), (np.zeros(2),), {id(local): [force, force]}, 0
    )
    require_original(checks)
    stepper.velocity = (v + np.asarray([0.01, 0]),)
    with pytest.raises(ArithmeticError):
        require_original(
            original_update(stepper, (np.zeros(2),), (np.zeros(2),), {id(local): [force, force]}, 0)
        )


def test_separate_quadrature_fields_and_excluded_nonfinite_norm() -> None:
    """A stable pressure norm cannot conceal an unresolved independent velocity norm."""
    assert quadrature_change({"pressure": 2, "velocity": 1}, {"pressure": 2, "velocity": 2}) == 0.5
    with pytest.raises(ValueError):
        quadrature_change({"pressure": np.nan}, {"pressure": 1})
    with pytest.raises(ValueError):
        quadrature_change({"pressure": 1}, {"velocity": 1})
    with pytest.raises(ArithmeticError):
        require_original({"momentum": 0, "weak_trace": np.nan})


def test_case_requires_fresh_output_and_known_scope(tmp_path: Path) -> None:
    """No previous result may silently seed a changed initial numerical study."""
    with pytest.raises(ValueError):
        run("unknown", tmp_path / "new")
    with pytest.raises(ValueError):
        run("nanoguide", tmp_path / "private-reference")
    with pytest.raises(ValueError):
        run("helmholtz", tmp_path)


def test_reduced_field_norms_preserve_original_three_physical_functionals() -> None:
    """Removing unused H(div)/Hessian diagnostics preserves the three retained field errors."""
    mesh = TetraMesh.unit_cube()
    bary, _ = tetrahedron_quadrature(4)
    _, nodes, _, _ = tetra_tabulate(mesh, 3, bary)
    local = SimpleNamespace(
        mesh=mesh, degree=3, nodes=nodes, constitutive=None, lame_lambda=0.4, lame_mu=0.4
    )
    coefficients = np.linspace(-0.2, 0.3, 3 * len(nodes))
    solution = SimpleNamespace(
        locals=(local,), time=0.025, displacement=(coefficients,), velocity=(coefficients * 2,)
    )
    original = norms(solution, ElasticWave(), 4)
    reduced = field_norms(solution, ElasticWave(), 4)
    assert reduced.keys() == {"displacement_l2", "velocity_l2", "stress_l2"}
    for name, value in reduced.items():
        assert value == pytest.approx(original[name], rel=2e-14, abs=1e-14)


def test_crop_literal_pressure_norm_and_complex_increment() -> None:
    """The retained Q3 coordinate matrix integrates a known constant complex field."""
    matrix = np.asarray(_cardinals(3))
    values = np.full((2, 3, 16), 3 + 4j)
    assert physical_norm(values, matrix) == pytest.approx(5 * np.sqrt(6 * 2.5**2))
    assert physical_norm(values, matrix, values / 2, matrix, 5) == pytest.approx(
        2.5 * np.sqrt(6 * 2.5**2)
    )
    with pytest.raises(ValueError):
        physical_norm(values, matrix, values)


def test_literal_native_component_injection_and_physical_units() -> None:
    """Saved component maps and absolute native affine determinants set physical norms."""
    arrays = {
        "native_geometry_points": np.asarray([[0, 0], [0, 1], [1, 0]]),
        "native_geometry_dofs": np.asarray([[0, 1, 2]]),
        "native_cell_dofs": np.asarray([[0, 1, 2]]),
        "component_dofs": np.asarray([[1, 0], [3, 2], [5, 4]]),
        "basis_q8_values_and_gradients": np.asarray([[[1 / 3, 1 / 3, 1 / 3]]]),
        "basis_q8_weights": np.asarray([0.5]),
    }
    u = np.asarray([4, 3] * 3)
    v = np.asarray([8, 6] * 3)
    result = physical_norms(u, v, arrays, 8, [2, 1, 3, 4])
    assert result["displacement_l2"] == pytest.approx(30 / np.sqrt(2))
    assert result["velocity_l2"] == pytest.approx(15 / np.sqrt(2))
    arrays["native_geometry_points"][2] = [0, 0]
    with pytest.raises(ValueError):
        physical_norms(u, v, arrays, 8, [2, 1, 3, 4])
