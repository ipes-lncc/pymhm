"""Plane-wave and outgoing Hankel experiments from the 2020 MHM Helmholtz paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import hankel1
from threadpoolctl import threadpool_limits

from examples.campaign_checkpoint import archive_identity, require_sources, verify_checkpoint
from pymhm.helmholtz import HelmholtzSolution, solve_helmholtz
from pymhm.helmholtz_forms import acoustic_quadrature
from pymhm.helmholtz_spaces import helmholtz_skeleton
from pymhm.quadrilateral import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/results/helmholtz"


@dataclass(frozen=True)
class AcousticWave:
    """Analytical homogeneous Helmholtz solution and its physical absorbing datum."""

    omega: float
    angle: float = np.pi / 13
    kind: str = "plane"

    def pressure(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the plane wave or Hankel source centered outside the domain."""
        if self.kind == "pml":
            return np.exp(
                1j * self.omega * points[:, 0] - 50 * np.maximum(points[:, 0] - 0.6, 0) ** 3
            )
        if self.kind == "plane":
            direction = np.array([np.cos(self.angle), np.sin(self.angle)])
            return np.exp(1j * self.omega * (points @ direction))
        distance = np.linalg.norm(points - [1.5, 0.5], axis=1)
        return hankel1(0, self.omega * distance)

    def gradient(self, points: np.ndarray) -> np.ndarray:
        """Differentiate analytically, using dH0/dz=-H1 for the radial field."""
        if self.kind == "pml":
            return np.column_stack(
                (
                    1j * self.omega * self.stretch(points)[:, 0] * self.pressure(points),
                    np.zeros(len(points)),
                )
            )
        if self.kind == "plane":
            direction = np.array([np.cos(self.angle), np.sin(self.angle)])
            return 1j * self.omega * self.pressure(points)[:, None] * direction
        displacement = points - [1.5, 0.5]
        distance = np.linalg.norm(displacement, axis=1)
        return (
            -self.omega
            * hankel1(1, self.omega * distance)[:, None]
            * displacement
            / distance[:, None]
        )

    def absorbing(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Return grad(u).n-i*omega*u, with the declared outward normal."""
        return np.sum(self.gradient(points) * normals, axis=1) - 1j * self.omega * self.pressure(
            points
        )

    def stretch(self, points: np.ndarray) -> np.ndarray:
        """Return a cubic integrated attenuation profile, starting at x=0.6."""
        return np.column_stack(
            (1 + 150j / self.omega * np.maximum(points[:, 0] - 0.6, 0) ** 2, np.ones(len(points)))
        )


def source_hashes() -> dict[str, str]:
    """Record every original operator, geometry and acquisition owner used here."""
    names = [
        "helmholtz",
        "helmholtz_forms",
        "helmholtz_spaces",
        "loads",
        "hybrid",
        "parallel",
        "solvers",
        "mesh",
        "lagrange",
        "quadrilateral",
        "_geometry_roundoff",
        "cut_cells",
        "scalar_boundary",
        "elements",
        "planar_quadrature",
        "planar_fitting",
        "planar_material",
    ]
    paths = [ROOT / f"src/pymhm/{name}.py" for name in names] + [Path(__file__)]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def norms(solution: HelmholtzSolution, wave: AcousticWave, order: int = 12) -> dict[str, float]:
    """Compute physical complex L², broken gradient and frequency-weighted energy norms."""
    pressure_squared, gradient_squared = 0.0, 0.0
    for fine in solution.local_meshes:
        _, points, weights, _, _, _ = acoustic_quadrature(fine, solution.degree, 1.0, order)
        flat = points.reshape(-1, 2)
        pressure_squared += float(
            np.sum(weights * abs(wave.pressure(flat).reshape(weights.shape)) ** 2)
        )
        gradient_squared += float(
            np.sum(weights * np.sum(abs(wave.gradient(flat).reshape(points.shape)) ** 2, axis=-1))
        )
    p_error = solution.l2_error(wave.pressure, order)
    g_error = solution.gradient_l2_error(wave.gradient, order)
    return {
        "pressure_l2": p_error,
        "gradient_l2": g_error,
        "pressure_exact_l2": float(np.sqrt(pressure_squared)),
        "gradient_exact_l2": float(np.sqrt(gradient_squared)),
        "pressure_relative_error": p_error / np.sqrt(pressure_squared),
        "gradient_relative_error": g_error / np.sqrt(gradient_squared),
        "energy_relative_error": float(
            np.sqrt(
                (g_error**2 + wave.omega**2 * p_error**2)
                / (gradient_squared + wave.omega**2 * pressure_squared)
            )
        ),
    }


def archive(solution: HelmholtzSolution, path: Path, wave: AcousticWave) -> None:
    """Persist one-sided complex fields and all executed oscillatory coordinate maps."""
    t = np.linspace(0, 1, 5)
    reference = np.array([(x, y) for y in t for x in t])
    points, values, gradients, cells = [], [], [], []
    offset = 0
    for cell in range(len(solution.local_meshes)):
        p, u, g = solution.sample(cell, reference)
        for fine_points, fine_values, fine_gradient in zip(p, u, g, strict=True):
            points.append(fine_points)
            values.append(fine_values)
            gradients.append(fine_gradient)
            for j in range(4):
                for i in range(4):
                    a = offset + 5 * j + i
                    cells.extend(([a, a + 1, a + 6], [a, a + 6, a + 5]))
            offset += 25
    transforms = {}
    for face, space in enumerate(solution.skeleton.faces):
        for segment, transform in enumerate(getattr(space, "transforms", ())):
            transforms[f"face_{face}_segment_{segment}_transform"] = transform
    mesh = solution.skeleton.mesh
    np.savez_compressed(
        path,
        points=np.concatenate(points),
        values=np.concatenate(values),
        gradient=np.concatenate(gradients),
        cells=np.asarray(cells),
        macro_points=mesh.points,
        macro_faces=mesh.faces,
        trace=solution.trace,
        omega=wave.omega,
        angle=wave.angle,
        kind=wave.kind,
        **transforms,
    )


def run(
    output: Path, resolutions: list[int], workers: int, angle_count: int, *, resume: bool = False
) -> None:
    """Acquire convergence and direction studies, checking immutable runtime provenance."""
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    record: dict[str, Any] = {
        "reference": (
            "Chaumont-Frelet and Valentin, SIAM J. Numer. Anal. 58(2), 2020; DOI 10.1137/19M1255616"
        ),
        "scope": (
            "Section 6 analytical data with explicitly selected local Qk spaces "
            "and finite resolution sequences"
        ),
        "boundary": {
            "convergence_and_direction": "grad(u).n-i*omega*u=g on the entire boundary",
            "pml": "analytical Dirichlet on the entire boundary",
        },
        "mathematical_configuration": {"resolutions": resolutions, "angle_count": angle_count},
        "assembly_quadrature": 10,
        "error_quadrature": 12,
        "source_sha256": hashes,
        "rows": [],
    }
    phases = [hashes]
    if resume:
        previous = json.loads((output / "comparison.json").read_text())
        require_sources(previous["source_sha256"], hashes)
        verify_checkpoint(
            previous,
            {"mathematical_configuration": record["mathematical_configuration"]},
            directory=output,
        )
        verify_checkpoint(
            previous, {"assembly_quadrature": 10, "error_quadrature": 12}, directory=output
        )
        for row in previous["rows"]:
            verify_checkpoint(
                row,
                {"local_refinement": 2},
                directory=output,
                metrics=(
                    "pressure_l2",
                    "gradient_l2",
                    "energy_relative_error",
                    "residual",
                    "macro_balance_max",
                ),
            )
        phases = previous.get("acquisition_sources", [previous["source_sha256"]]) + [hashes]
        record["rows"] = previous["rows"]
        for row in record["rows"]:
            row.setdefault("source_phase", 0)
    record["acquisition_sources"] = phases

    def solve_row(study: str, wave: AcousticWave, n: int, ell: int, oscillatory: bool) -> None:
        """Solve one immutable configuration and write its completed numerical checkpoint."""
        basis = "oscillatory" if oscillatory else "polynomial"
        degree = 4 if study == "pml" else ell + 2
        identity = dict(
            study=study,
            wave=wave.kind,
            resolution=n,
            trace_degree=ell,
            trace_basis=basis,
            angle=wave.angle,
        )
        for row in record["rows"]:
            if all(row[key] == value for key, value in identity.items()):
                verify_checkpoint(
                    row,
                    {
                        **identity,
                        "omega": wave.omega,
                        "local_degree": degree,
                        "local_refinement": 2,
                    },
                    directory=output,
                )
                return
        mesh = CartesianMacroMesh(n)
        skeleton = helmholtz_skeleton(mesh, wave.omega, degree=ell, oscillatory=oscillatory)
        start = time.perf_counter()
        solution = solve_helmholtz(
            mesh,
            omega=wave.omega,
            skeleton=skeleton,
            degree=degree,
            local_refinement=2,
            absorbing=None if study == "pml" else wave.absorbing,
            dirichlet=wave.pressure if study == "pml" else 0j,
            pml_stretch=wave.stretch if study == "pml" else None,
            quadrature_order=10,
            backend="process" if workers > 1 else "serial",
            workers=workers,
        )
        row = {
            "study": study,
            "source_phase": len(phases) - 1,
            "wave": wave.kind,
            "omega": wave.omega,
            "angle": wave.angle,
            "resolution": n,
            "macro_edge_length": 1 / n,
            "macro_diameter": np.sqrt(2) / n,
            "trace_degree": ell,
            "trace_basis": "oscillatory" if oscillatory else "polynomial",
            "local_degree": degree,
            "local_refinement": 2,
            "free_complex_trace_dofs": int(
                sum(
                    space.size
                    for face, space in enumerate(skeleton.faces)
                    if study == "pml" or mesh.face_cells[face, 1] >= 0
                )
            ),
            "residual": solution.hybrid.residual,
            "macro_balance_max": float(np.max(abs(solution.conservation_residuals()))),
            **norms(solution, wave),
            "elapsed_seconds": time.perf_counter() - start,
        }
        if study == "convergence" and n == resolutions[len(resolutions) // 2] and ell == 2:
            filename = f"{wave.kind}-{row['trace_basis']}-fields.npz"
            archive(solution, output / filename, wave)
            row.update(archive_identity(output / filename))
        if study == "pml" and n == 24:
            filename = "pml-fields.npz"
            archive(solution, output / filename, wave)
            row.update(archive_identity(output / filename))
        record["rows"].append(row)
        (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    with threadpool_limits(1):
        for kind, omega in (("plane", 10 * np.pi), ("hankel", 20 * np.pi)):
            for ell in (2, 3):
                for oscillatory in (False, True):
                    for n in resolutions:
                        solve_row(
                            "convergence", AcousticWave(omega, kind=kind), n, ell, oscillatory
                        )
        for angle in np.linspace(0, np.pi / 2, angle_count):
            for oscillatory in (False, True):
                solve_row("direction", AcousticWave(20 * np.pi, float(angle)), 11, 2, oscillatory)
        for n in (16, 20, 24, 28, 32):
            solve_row("pml", AcousticWave(10 * np.pi, kind="pml"), n, 0, False)
    record["source_changed_during_run"] = hashes != source_hashes()
    (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    if record["source_changed_during_run"]:
        raise RuntimeError("operator sources changed during the Helmholtz acquisition")


def main() -> None:
    """Run the published analytical wave data with the declared discretizations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--angles", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.output, args.resolutions, args.workers, args.angles, resume=args.resume)


if __name__ == "__main__":
    main()
