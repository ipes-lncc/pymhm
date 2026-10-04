"""Data-only temporal increments on the held executed conforming P3 h8 basis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from examples.minimal_wave_convergence import ROOT, digest, quadrature_change, require_original

BASE = ROOT / "build/results/completion/three-layer-classical-h8-basis-v2-afd43c6b731c"
REFERENCE = ROOT / "build/results/completion/three-layer-classical-h8-dt001-basis-v2-afd43c6b731c"


def state(
    directory: Path, dt: float, operators_sha: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Verify all original physical checkpoints and retain their independent dt/source identity."""
    identity = json.loads((directory / "identity.json").read_text())
    if identity["physical_dt_s"] != dt or identity["operators_sha256"] != operators_sha:
        raise ValueError("Three-layer time-step or held executed operator identity differs")
    count = round(0.3 / dt)
    maxima = {"momentum_relative": 0.0, "energy_work_relative": 0.0}
    for index in range(count + 1):
        path = directory / f"state-{index:03}.json"
        record = json.loads(path.read_text())
        if record["step"] != index or abs(record["physical_time_s"] - index * dt) > 1e-14:
            raise ValueError("Three-layer archived physical observation time differs")
        if digest(directory / record["archive"]) != record["sha256"]:
            raise ValueError("Three-layer original state payload changed")
        checks = {
            "momentum_relative": record["momentum_relative_residual"],
            "energy_work_relative": record.get("energy_balance_relative_residual", 0.0),
        }
        require_original(checks)
        maxima = {key: max(maxima[key], value) for key, value in checks.items()}
    completion = json.loads((directory / "completion.json").read_text())
    if (
        completion["status"] != "trajectory_complete"
        or completion["persisted_time_states"] != count + 1
    ):
        raise ValueError("The entire selected three-layer time interval is required")
    with np.load(directory / f"state-{count:03}.npz", allow_pickle=False) as arrays:
        u, v = arrays["displacement"].copy(), arrays["velocity"].copy()
    return (
        u,
        v,
        {
            "identity": str(directory / "identity.json"),
            "identity_sha256": digest(directory / "identity.json"),
            "state_count": count + 1,
            "original_equations": maxima,
            "final_state_sha256": record["sha256"],
        },
    )


def physical_norms(
    displacement: np.ndarray,
    velocity: np.ndarray,
    arrays: dict[str, np.ndarray],
    order: int,
    scales: list[float],
) -> dict[str, float]:
    """Integrate held P3 values using producer tables and physical units."""
    points, cells, dofs = (
        arrays["native_geometry_points"],
        arrays["native_geometry_dofs"],
        arrays["native_cell_dofs"],
    )
    determinants = abs(
        np.linalg.det((points[cells[:, 1:]] - points[cells[:, :1]]).transpose(0, 2, 1))
    )
    if np.any(determinants == 0):
        raise ValueError("Nondegenerate actual native geometry required")
    basis = arrays[f"basis_q{order}_values_and_gradients"][0]
    weights = arrays[f"basis_q{order}_weights"]
    total: np.ndarray = np.zeros(2, dtype=np.longdouble)
    for first in range(0, len(cells), 256):
        ids = dofs[first : first + 256]
        for component, coefficients in enumerate((displacement, velocity)):
            value = np.einsum("qi,tia->tqa", basis, coefficients[arrays["component_dofs"]][ids])
            total[component] += np.sum(
                determinants[first : first + 256, None] * weights * np.sum(value**2, axis=-1),
                dtype=np.longdouble,
            )
    length, _, amplitude, time = scales
    return {
        "displacement_l2": float(np.sqrt(total[0]) * length * amplitude),
        "velocity_l2": float(np.sqrt(total[1]) * length * amplitude / time),
    }


def acquire(output: Path) -> dict[str, Any]:
    """Reuse the finest301 states and independently acquired dt.004/.002 without a new solve."""
    operator_path = BASE / "operators.json"
    operators = json.loads(operator_path.read_text())
    for name, expected in operators["archives"].items():
        if digest(BASE / name) != expected:
            raise ValueError("Actual held classical operator/basis artifact changed")
    with np.load(BASE / "space-and-force.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    l2_mass = sparse.load_npz(BASE / "l2_mass.npz")
    scales = operators["scales_L0_rho0_u0_t0"]
    rows, held = [], []
    for dt, directory in (
        (0.004, ROOT / "build/results/minimal-three-layer-h8-dt004-afd-v1"),
        (0.002, ROOT / "build/results/minimal-three-layer-h8-dt002-afd-v1"),
        (0.001, REFERENCE),
    ):
        u, v, proof = state(directory, dt, digest(operator_path))
        low, high = (
            physical_norms(u, v, arrays, 8, scales),
            physical_norms(u, v, arrays, 10, scales),
        )
        change = quadrature_change(low, high)
        if change > 1e-10:
            raise ArithmeticError("Literal classical physical field norm quadrature is unresolved")
        length, _, amplitude, time = scales
        native = {
            "displacement_l2": float(np.sqrt(u @ (l2_mass @ u)) * length * amplitude),
            "velocity_l2": float(np.sqrt(v @ (l2_mass @ v)) * length * amplitude / time),
        }
        native_change = quadrature_change(native, high)
        if native_change > 1e-10:
            raise ArithmeticError(
                "Literal basis and independently assembled native L2 forms differ"
            )
        rows.append(
            {
                "level": dt,
                "physical_time_s": 0.3,
                "norms": high,
                **proof,
                "quadrature_orders": [8, 10],
                "quadrature_relative_change": change,
                "native_l2_form_relative_change": native_change,
            }
        )
        held.append((u, v))
    for row, (u, v) in zip(rows[:-1], held[:-1], strict=True):
        low = physical_norms(u - held[-1][0], v - held[-1][1], arrays, 8, scales)
        high = physical_norms(u - held[-1][0], v - held[-1][1], arrays, 10, scales)
        if quadrature_change(low, high) > 1e-10:
            raise ArithmeticError("Literal classical temporal increment quadrature is unresolved")
        row["norms"].update(
            {key.replace("_l2", "_increment_l2"): value for key, value in high.items()}
        )
    return {
        "reference": "Gomes et al. (2017), doi:10.20906/CPS/CILAMCE2017-0399, Section5.2",
        "scope": (
            "Initial temporal refinement of the selected three-layer problem "
            "on one fixed conforming P3 h8 mesh"
        ),
        "configuration": {
            "physical_extent_m": [1000, 450],
            "h_m": 8,
            "finite_elements": "continuous vector P3",
            "time_steps_s": [0.004, 0.002, 0.001],
            "final_time_s": 0.3,
            "beta": 0.25,
            "gamma": 0.5,
            "initial_fields": "zero displacement and velocity",
            "boundary": "Zero physical traction everywhere; no rigid pin/gauge",
        },
        "physical_inputs": operators["physical_input_hashes"],
        "source": operators["physical_source"],
        "operators": str(operator_path),
        "operators_sha256": digest(operator_path),
        "executed_basis": operators["executed_scalar_basis"],
        "basis_contract": (
            "Literal executed Basix coefficient matrix, native cell/component DOFs "
            "and q8/q10 tables reused; no fitted basis or new element constructor"
        ),
        "rows": rows,
        "plot_fields": [
            "displacement_increment_l2",
            "velocity_increment_l2",
            "displacement_l2",
            "velocity_l2",
        ],
        "level_label": "Time step dt (s)",
        "norm_label": "Physical L2 norm / temporal increment",
        "figure_title": "Three layers: initial temporal refinement, h=8m",
        "exact_solution_available": False,
        "finest_numerical_reference_time_step_s": 0.001,
        "reference_refinement_verified": False,
        "asymptotic_convergence_verified": False,
        "pde_solves_during_acquire": 0,
        "limitations": [
            "Spatial refinement of h8 is unresolved; "
            "temporal increments do not certify spatial accuracy",
            "dt.001 is a numerical comparison level, not an exact temporal solution",
            "This conforming temporal study does not replace a complete "
            "MHM/refined-reference physical comparison",
            "Selected mesh, horizons and Ormsby data "
            "do not identify the historical author's inputs",
        ],
    }
