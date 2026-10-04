"""CILAMCE 2017 three-dimensional analytical displacement, velocity and stress checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.waves.elastodynamics import (
    ElastodynamicSolution,
    ElastodynamicStepper,
    _stress_from_gradient,
)
from pymhm.fem.scalar.tetrahedron import tetra_element_tabulate, tetrahedron_quadrature
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.tetrahedron import TetraMesh

ROOT = Path(__file__).resolve().parents[1]


class ElasticWave:
    """Equation (53), lambda=mu=.4, rho=1; analytical spatial derivatives independent of FE."""

    omega: float = np.pi * np.sqrt(0.8)

    def __init__(self) -> None:
        """Own force-shape caches for the lifetime of one numerical acquisition."""
        self._source_cache: dict[tuple[tuple[int, ...], bytes], tuple[np.ndarray, np.ndarray]] = {}

    def spatial(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the three sine products and their exact full gradients/Hessians."""
        values = np.empty((len(points), 3))
        gradient = np.empty((len(points), 3, 3))
        hessian = np.empty((len(points), 3, 3, 3))
        for component, (frequency, sign) in enumerate(
            ((2 * np.pi, 1.0), (2 * np.pi, -1.0), (np.pi, 1.0))
        ):
            s, c = np.sin(frequency * points), np.cos(frequency * points)
            values[:, component] = sign * np.prod(s, axis=1)
            for a in range(3):
                factors = s.copy()
                factors[:, a] = frequency * c[:, a]
                gradient[:, component, a] = sign * np.prod(factors, axis=1)
                for b in range(3):
                    if a == b:
                        hessian[:, component, a, b] = -(frequency**2) * values[:, component]
                    else:
                        factors = s.copy()
                        factors[:, a] = frequency * c[:, a]
                        factors[:, b] = frequency * c[:, b]
                        hessian[:, component, a, b] = sign * np.prod(factors, axis=1)
        return values, gradient, hessian

    def amplitudes(self, t: float) -> tuple[float, float, float]:
        """Displacement amplitude and its first two exact time derivatives."""
        return (
            (1 - np.cos(self.omega * t)) / 2,
            self.omega * np.sin(self.omega * t) / 2,
            self.omega**2 * np.cos(self.omega * t) / 2,
        )

    @staticmethod
    def divergence(hessian: np.ndarray) -> np.ndarray:
        """Compute div(sigma) with constant isotropic lambda=mu=.4."""
        return 0.4 * np.trace(hessian, axis1=-2, axis2=-1) + 0.8 * np.einsum(
            "...jij->...i", hessian
        )

    def source(self, t: float, points: np.ndarray) -> np.ndarray:
        """Body force rho*u_tt-div(sigma), derived directly from analytical derivatives."""
        points = np.asarray(points, dtype=float)
        u, divergence = self._source_shapes(points.shape, points.tobytes())
        a, _, acc = self.amplitudes(t)
        return acc * u - a * divergence

    def _source_shapes(
        self, shape: tuple[int, ...], coordinates: bytes
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cache the two time-independent force factors at immutable quadrature coordinates."""
        key = (shape, coordinates)
        if key not in self._source_cache:
            points = np.frombuffer(coordinates, dtype=float).reshape(shape)
            u, _, h = self.spatial(points)
            divergence = self.divergence(h)
            u.setflags(write=False)
            divergence.setflags(write=False)
            self._source_cache[key] = u, divergence
        return self._source_cache[key]


def norms(solution: ElastodynamicSolution, model: ElasticWave, order: int) -> dict[str, float]:
    """Integrate the original six norms using one Pk tabulation per macrocell.

    The quadrature, analytical data, constitutive stress implementation, tensor
    contractions, and long-double reductions are the acquisition's operations.
    Only repeated construction of the same nodal basis and gradients is removed.
    """
    bary, weights = tetrahedron_quadrature(order)
    totals = np.zeros(6, dtype=np.longdouble)
    amplitude, velocity, _ = model.amplitudes(solution.time)
    for index, local in enumerate(solution.locals):
        points = np.einsum("qi,tij->tqj", bary, local.mesh.points[local.mesh.cells])
        exact, exact_gradient, exact_hessian = model.spatial(points.reshape(-1, 3))
        exact = exact.reshape(points.shape)
        exact_gradient = exact_gradient.reshape(*points.shape, 3)
        exact_hessian = exact_hessian.reshape(*points.shape, 3, 3)
        dofs, _, basis, derivative, hessian = tetra_element_tabulate(local.mesh, local.degree, bary)
        displacement_coefficients = solution.displacement[index].reshape(-1, 3)[dofs]
        velocity_coefficients = solution.velocity[index].reshape(-1, 3)[dofs]
        displacement = np.einsum("qi,tia->tqa", basis, displacement_coefficients)
        numerical_velocity = np.einsum("qi,tia->tqa", basis, velocity_coefficients)
        displacement_gradient = np.einsum("tqib,tia->tqab", derivative, displacement_coefficients)
        velocity_gradient = np.einsum("tqib,tia->tqab", derivative, velocity_coefficients)
        stress = _stress_from_gradient(local, points, displacement_gradient)
        exact_stress = (
            0.4
            * amplitude
            * (
                exact_gradient
                + exact_gradient.swapaxes(-1, -2)
                + np.trace(exact_gradient, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)
            )
        )
        displacement_hessian = np.einsum("tqnij,tna->tqaij", hessian, displacement_coefficients)
        divergence = model.divergence(displacement_hessian) - amplitude * model.divergence(
            exact_hessian
        )
        defects = (
            np.sum((displacement - amplitude * exact) ** 2, axis=-1),
            np.sum((numerical_velocity - velocity * exact) ** 2, axis=-1),
            np.sum((displacement_gradient - amplitude * exact_gradient) ** 2, axis=(-1, -2)),
            np.sum((velocity_gradient - velocity * exact_gradient) ** 2, axis=(-1, -2)),
            np.sum((stress - exact_stress) ** 2, axis=(-1, -2)),
            np.sum(divergence**2, axis=-1),
        )
        for component, defect in enumerate(defects):
            totals[component] += np.sum(
                local.mesh.volumes[:, None] * weights * defect, dtype=np.longdouble
            )
    u, v, du, dv, stress, divergence = totals
    return dict(
        zip(
            (
                "displacement_l2",
                "velocity_l2",
                "displacement_h1",
                "velocity_h1",
                "stress_l2",
                "stress_broken_hdiv",
            ),
            np.sqrt([u, v, u + du, v + dv, stress, stress + divergence]).astype(float).tolist(),
            strict=True,
        )
    )


def run(
    n: int,
    dt: float,
    final: float,
    output: Path,
    order: int = 12,
    substeps: int = 1,
    workers: int = 1,
) -> None:
    """Acquire source-frozen P3/P1 tetrahedral MHM at one spatial/temporal resolution."""
    files = [Path(__file__)] + [
        ROOT / f"src/pymhm/{name}.py"
        for name in (
            "_legacy/models/waves/elastodynamics",
            "_legacy/models/elasticity/primal_3d",
            "fem/scalar/tetrahedron",
            "fem/scalar/tetrahedron_topology",
            "_legacy/models/darcy/primal_3d",
            "core/contracts",
            "linalg/linear",
            "execution/cpu",
        )
    ]
    original = current_source_manifest(
        {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    )
    model = ElasticWave()
    start = time.perf_counter()
    steps = round(final / dt)
    if abs(steps * dt - final) > 1e-13:
        raise ValueError("final time must equal an integer number of macro time steps")
    with (
        threadpool_limits(1),
        ElastodynamicStepper(
            TetraMesh.unit_cube(n),
            time_step=dt,
            degree=3,
            local_refinement=2,
            local_substeps=substeps,
            lame_lambda=0.4,
            lame_mu=0.4,
            quadrature_order=order,
            backend="process" if workers > 1 else "serial",
            workers=workers,
        ) as stepper,
    ):
        result = stepper.initialize()
        setup = time.perf_counter() - start
        history = []
        for step in range(steps):
            result = stepper.advance(model.source)
            history.append((result.time, result.energy, result.constraint_residual))
            if step % 20 == 0:
                print(
                    json.dumps({"step": step + 1, "n": n, "dt": dt, "time": result.time}),
                    flush=True,
                )
        error_order = 18 if n == 1 else 12
        measured = norms(result, model, error_order)
        high = norms(result, model, error_order + 4)
        sensitivity = max(
            abs(measured[k] - high[k]) / max(high[k], np.finfo(float).tiny) for k in measured
        )
        if sensitivity > 1e-6:
            raise RuntimeError(f"error integration not converged: {sensitivity}")
        output.mkdir(parents=True, exist_ok=True)
        name = f"n{n}-dt{dt:g}-q{order}-s{substeps}"
        np.savez_compressed(
            output / (name + ".npz"),
            macro_points=stepper.mesh.points,
            macro_cells=stepper.mesh.cells,
            displacement=np.array(result.displacement),
            velocity=np.array(result.velocity),
            trace=result.trace,
            time=result.time,
            local_degree=3,
            local_refinement=2,
        )
        record = {
            "paper": "Gomes et al. CILAMCE2017-0399, Equation (53)",
            "n": n,
            "dt": dt,
            "time": result.time,
            "trace_degree": 1,
            "local_degree": 3,
            "local_refinement": 2,
            "local_substeps": substeps,
            "assembly_workers": workers,
            "lambda": 0.4,
            "mu": 0.4,
            "rho": 1.0,
            "macro_cells": len(stepper.mesh.cells),
            "trace_dofs": len(result.trace),
            "norms": measured,
            "quadrature_relative_change": sensitivity,
            "history_columns": ["time", "physical_energy", "displacement_constraint_residual"],
            "history": history,
            "setup_seconds": setup,
            "elapsed_seconds": time.perf_counter() - start,
            "source_sha256": original,
        }
        if original != current_source_manifest(
            {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        ):
            raise RuntimeError("elastodynamic acquisition sources changed")
        (output / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "history"}), flush=True)


def main() -> None:
    """Run one reproducible spatial/time resolution of the published analytical PDE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--final", type=float, default=0.5)
    parser.add_argument("--order", type=int, default=12)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/elastodynamics")
    args = parser.parse_args()
    run(args.n, args.dt, args.final, args.output, args.order, args.substeps, args.workers)


if __name__ == "__main__":
    main()
