"""Q2 MHM propagation through the fifteen-cylinder photonic device of Section 6.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.waves.maxwell import MaxwellStepper
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.vector.curl import TangentialTraceSpace as MaxwellSkeleton
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.cartesian import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NanoWaveguide:
    """Exact geometric dimensions/material values of Figure 6.7, in article units.

    The incident wave travels in +x with unit amplitude, zero phase and frequency
    0.5. A causal sine and zero initial fields specify the otherwise unreported
    turn-on convention. Those data are shared unchanged with classical references.
    """

    radius: float = 0.3125
    frequency: float = 0.5

    def permittivity(self, points: np.ndarray) -> np.ndarray:
        """Return 3.14 in the fifteen disks, 1.5 in the silica device, 1 in air."""
        x, y = points.T
        silica = (x >= 3.75) & (x <= 7.5) & (y >= 1.875) & (y <= 8.125)
        silica |= (x >= 1.875) & (x <= 3.75) & (y >= 3.125) & (y <= 6.875)
        silica |= (x >= 7.5) & (x <= 9.375) & (y >= 4.375) & (y <= 5.625)
        result = np.where(silica, 1.5, 1.0)
        for cx in (4.375, 5.625, 6.875):
            for cy in (2.5, 3.75, 5.0, 6.25, 7.5):
                result[(x - cx) ** 2 + (y - cy) ** 2 <= self.radius**2] = 3.14
        return result

    def incident(self, time: float, points: np.ndarray) -> np.ndarray:
        """Evaluate the causal unit-amplitude incident electric field."""
        phase = time - points[:, 0]
        return np.where(phase >= 0, np.sin(2 * np.pi * self.frequency * phase), 0)

    def boundary(self, time: float, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Impose E_i-alpha(H_i cross n), with H_i=(0,-E_i) in exterior air."""
        return (1 - normals[:, 0]) * self.incident(time, points)


def hashes() -> dict[str, str]:
    """Record immutable acquisition, operator, geometry and algebra sources."""
    paths = [Path(__file__)] + [
        ROOT / f"src/pymhm/{name}.py"
        for name in (
            "_legacy/models/waves/maxwell",
            "fem/vector/curl",
            "_legacy/models/darcy/cartesian",
            "fem/scalar/helmholtz",
            "meshes/triangle",
            "core/contracts",
            "linalg/linear",
            "fem/quadrature/material",
            "fem/traces/scalar",
            "fem/quadrature/planar",
        )
    ]
    return current_source_manifest(
        {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    )


def save_fields(solution: Any, path: Path, fine_resolution: int) -> None:
    """Store Q2 cell coefficients by physical grid position, retaining every DG trace."""
    electric = np.empty((fine_resolution, fine_resolution, 9))
    magnetic = np.empty((fine_resolution, fine_resolution, 9, 2))
    present = np.zeros((fine_resolution, fine_resolution), dtype=bool)
    for local, e, h in zip(solution.locals, solution.electric, solution.magnetic, strict=True):
        starts = local.mesh.points[local.mesh.cells[:, 0]]
        indices = np.rint(starts * fine_resolution / 10).astype(int)
        if not np.allclose(indices, starts * fine_resolution / 10, atol=1e-12, rtol=0):
            raise RuntimeError("fine cells do not share the declared physical Cartesian grid")
        x, y = indices.T
        if np.any(present[y, x]):
            raise RuntimeError("duplicate physical cells in the persisted DG field")
        present[y, x] = True
        electric[y, x] = e.reshape(-1, 9)
        magnetic[y, x] = h.reshape(-1, 9, 2)
    if not np.all(present):
        raise RuntimeError("incomplete physical grid in the persisted DG field")
    np.savez_compressed(
        path,
        electric=electric,
        magnetic=magnetic,
        bounds=np.array([0, 10, 0, 10]),
        degree=2,
        macro_points=solution.skeleton.mesh.points,
        macro_faces=solution.skeleton.mesh.faces,
        trace=solution.trace,
        electric_time=solution.electric_time,
        magnetic_time=solution.magnetic_time,
        time_step=solution.time_step,
    )


def run(output: Path, macro: int, fine: int, order: int, steps: int, dt: float) -> None:
    """Solve one source-frozen paper-space case with energy checks and field replay."""
    if fine % macro:
        raise ValueError("fine resolution must be divisible by macro resolution")
    output.mkdir(parents=True, exist_ok=True)
    initial_hashes = hashes()
    model = NanoWaveguide()
    mesh = CartesianMacroMesh(macro, macro, (0, 10, 0, 10))
    r = fine // macro
    skeleton = MaxwellSkeleton(
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, r) for _ in mesh.faces))
    )
    start = time.perf_counter()
    history = []
    with (
        threadpool_limits(1),
        MaxwellStepper(
            mesh,
            time_step=dt,
            degree=2,
            local_refinement=r,
            skeleton=skeleton,
            permittivity=model.permittivity,
            absorbing=1,
            boundary_data=model.boundary,
            quadrature_order=order,
        ) as stepper,
    ):
        setup = time.perf_counter() - start
        result = stepper.initialize()
        for index in range(steps):
            result = stepper.advance()
            history.append(
                (
                    result.electric_time,
                    result.magnetic_time,
                    result.energy,
                    result.energy_balance_residual,
                )
            )
            if index % 100 == 0:
                print(
                    json.dumps(
                        {
                            "step": index + 1,
                            "energy": result.energy,
                            "balance": result.energy_balance_residual,
                        }
                    ),
                    flush=True,
                )
        field_path = output / f"mhm-n{macro}-f{fine}-q{order}-dt{dt:g}-fields.npz"
        save_fields(result, field_path, fine)
        record = {
            "reference": "Lanteri et al. (2018), Section 6.3, DOI 10.1137/16M110037X",
            "data": "Figure 6.7 exact dimensions; epsilon=1/1.5/3.14; mu=1; f=0.5",
            "incident_convention": "unit causal sin(pi*(t-x)), +x propagation; initially E=H=0",
            "macro_resolution": macro,
            "fine_resolution": fine,
            "local_refinement": r,
            "local_degree": 2,
            "trace_degree": 1,
            "trace_subdivisions": r,
            "trace_dofs": skeleton.size,
            "local_electric_dofs": fine**2 * 9,
            "quadrature_order": order,
            "time_step": dt,
            "steps": steps,
            "electric_time": result.electric_time,
            "magnetic_time": result.magnetic_time,
            "cfl_bound": result.frequency_bound,
            "cfl_product": dt * result.frequency_bound,
            "energy_balance_max": float(np.max(abs(np.asarray(history)[:, 3]), initial=0)),
            "history_columns": [
                "electric_time",
                "magnetic_time",
                "modified_energy",
                "energy_balance_residual",
            ],
            "history": history,
            "setup_seconds": setup,
            "elapsed_seconds": time.perf_counter() - start,
            "fields": field_path.name,
            "fields_sha256": hashlib.sha256(field_path.read_bytes()).hexdigest(),
            "source_sha256": initial_hashes,
        }
        if hashes() != initial_hashes:
            raise RuntimeError("sources changed during the nanoguide calculation")
        field_path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "history"}), flush=True)


def main() -> None:
    """Acquire quadrilateral MHM fields at the published local/trace resolutions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/maxwell-nanoguide")
    parser.add_argument("--macro", type=int, default=16)
    parser.add_argument("--fine", type=int, default=128)
    parser.add_argument("--order", type=int, default=12)
    parser.add_argument("--steps", type=int, default=1131)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()
    run(args.output, args.macro, args.fine, args.order, args.steps, args.dt)


if __name__ == "__main__":
    main()
