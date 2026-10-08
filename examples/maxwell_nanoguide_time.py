"""Nanoguide temporal control evaluated at identical staggered physical times."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from examples.maxwell_nanoguide import ROOT, NanoWaveguide, hashes, save_fields
from examples.tutorial_maxwell_equations import EquationLeapfrog
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.vector.curl import TangentialTraceSpace as MaxwellSkeleton
from pymhm.meshes.cartesian import CartesianMacroMesh


def run(output: Path, dt: float, order: int) -> None:
    """Bracket E(11.315), preserve H(11.31), and record the time interpolation."""
    model = NanoWaveguide()
    mesh = CartesianMacroMesh(16, 16, (0, 10, 0, 10))
    skeleton = MaxwellSkeleton(
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 8) for _ in mesh.faces))
    )
    original = hashes()
    original[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    start = time.perf_counter()
    magnetic_step = round(11.31 / dt)
    if not np.isclose(magnetic_step * dt, 11.31, atol=1e-12, rtol=0):
        raise ValueError("the control must sample the target magnetic time exactly")
    lower_index = int(np.floor(11.315 / dt - 0.5 + 1e-12))
    upper_index = lower_index + 1
    snapshots = {}
    balance = 0.0
    with (
        threadpool_limits(1),
        EquationLeapfrog(
            mesh,
            time_step=dt,
            degree=2,
            local_refinement=8,
            skeleton=skeleton,
            permittivity=model.permittivity,
            absorbing=1,
            boundary_data=model.boundary,
            quadrature_order=order,
        ) as stepper,
    ):
        setup = time.perf_counter() - start
        stepper.initialize()
        for index in range(1, max(upper_index, magnetic_step) + 1):
            result = stepper.advance()
            balance = max(balance, abs(result.energy_balance_residual))
            if index in (lower_index, upper_index, magnetic_step):
                snapshots[index] = result
            if index % 100 == 1:
                print(json.dumps({"step": index, "energy_balance_max": balance}), flush=True)
        lower, upper = snapshots[lower_index], snapshots[upper_index]
        weight = (11.315 - lower.electric_time) / dt
        aligned = replace(
            snapshots[magnetic_step],
            electric=tuple(
                (1 - weight) * a + weight * b
                for a, b in zip(lower.electric, upper.electric, strict=True)
            ),
            electric_time=11.315,
        )
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"mhm-n16-f128-q{order}-dt{dt:g}-aligned.npz"
        save_fields(aligned, path, 128)
        current = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in original
        }
        if current != original:
            raise RuntimeError("temporal-control acquisition sources changed")
        record = {
            "electric_time": 11.315,
            "magnetic_time": 11.31,
            "time_step": dt,
            "electric_interpolation": {
                "rule": "linear",
                "lower_time": lower.electric_time,
                "upper_time": upper.electric_time,
                "upper_weight": weight,
            },
            "magnetic_interpolation": "none",
            "energy_balance_max": balance,
            "setup_seconds": setup,
            "elapsed_seconds": time.perf_counter() - start,
            "source_sha256": original,
            "fields": path.name,
            "fields_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


def main() -> None:
    """Acquire a temporal control with explicit matched-time interpolation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/maxwell-nanoguide")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--order", type=int, default=12)
    args = parser.parse_args()
    run(args.output, args.dt, args.order)


if __name__ == "__main__":
    main()
