"""Acquire original 3D general-tensor convergence and a bounded-force primal locking study."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.elasticity.primal_3d import solve_elasticity_3d
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.tetrahedron import TetraMesh

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/elasticity3d"
STIFFNESS = np.diag([5.0, 6, 7, 2, 3, 4]) + 0.1 * np.ones((6, 6))


def potential(points: np.ndarray, derivative: tuple[int, int, int]) -> np.ndarray:
    """Differentiate the product sin²(pi*x) sin²(pi*y) sin²(pi*z) analytically."""
    result = np.ones(len(points))
    for axis, order in enumerate(derivative):
        angle = 2 * np.pi * points[:, axis]
        values = (
            (1 - np.cos(angle)) / 2,
            np.pi * np.sin(angle),
            2 * np.pi**2 * np.cos(angle),
            -4 * np.pi**3 * np.sin(angle),
        )
        result *= values[order]
    return result


@dataclass(frozen=True)
class ElasticityData3D:
    """Solenoidal boundary-vanishing displacement with exact general-tensor stress and force."""

    anisotropic: bool = True

    def displacement(self, points: np.ndarray) -> np.ndarray:
        """Return curl(0,0,psi), which has identically zero divergence."""
        return np.column_stack(
            (potential(points, (0, 1, 0)), -potential(points, (1, 0, 0)), np.zeros(len(points)))
        )

    def constitutive(self, points: np.ndarray) -> np.ndarray:
        """Evaluate a spatially varying uniformly SPD anisotropic Kelvin matrix."""
        return (1 + points @ np.array([1.0, 2, 3]))[:, None, None] * STIFFNESS

    def fields(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Differentiate the symmetric constitutive stress with every coefficient derivative."""
        gradient = np.zeros((len(points), 3, 3))
        hessian = np.zeros((len(points), 3, 3, 3))
        for component, (axis, sign) in enumerate(((1, 1), (0, -1))):
            for first in range(3):
                derivative = np.zeros(3, dtype=int)
                derivative[axis] += 1
                derivative[first] += 1
                gradient[:, component, first] = sign * potential(points, tuple(derivative))
                for second in range(3):
                    higher = derivative.copy()
                    higher[second] += 1
                    hessian[:, component, first, second] = sign * potential(points, tuple(higher))
        if not self.anisotropic:
            # Exact divergence vanishes, so lambda does not enter stress or force.
            sigma = gradient + gradient.swapaxes(1, 2)
            divergence = np.trace(hessian, axis1=2, axis2=3)
            return sigma, -divergence
        pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        strain = np.stack(
            [
                (gradient[:, i, j] + gradient[:, j, i]) / (2 if i == j else np.sqrt(2))
                for i, j in pairs
            ],
            axis=1,
        )
        strain_derivative = np.stack(
            [
                (hessian[:, i, j] + hessian[:, j, i]) / (2 if i == j else np.sqrt(2))
                for i, j in pairs
            ],
            axis=1,
        )
        kelvin = strain @ STIFFNESS.T
        derivative = np.einsum("ab,nbj->naj", STIFFNESS, strain_derivative)
        sigma = np.zeros((len(points), 3, 3))
        sigma_derivative = np.zeros((len(points), 3, 3, 3))
        for a, (i, j) in enumerate(pairs):
            factor = 1 if i == j else np.sqrt(2)
            sigma[:, i, j] = sigma[:, j, i] = kelvin[:, a] / factor
            sigma_derivative[:, i, j] = sigma_derivative[:, j, i] = derivative[:, a] / factor
        multiplier = 1 + points @ np.array([1.0, 2, 3])
        divergence = np.einsum("nijj->ni", sigma_derivative) * multiplier[
            :, None
        ] + sigma @ np.array([1.0, 2, 3])
        return multiplier[:, None, None] * sigma, -divergence

    def stress(self, points: np.ndarray) -> np.ndarray:
        """Return the analytical symmetric stress without a spurious hydrostatic term."""
        return self.fields(points)[0]

    def source(self, points: np.ndarray) -> np.ndarray:
        """Return minus the analytical stress divergence."""
        return self.fields(points)[1]


def acquire(
    n: int, degree: int, refinement: int, data: ElasticityData3D, lam: float = 1.0, workers: int = 4
) -> tuple:
    """Solve one declared discretization with complete local assembly in worker processes."""
    mesh = TetraMesh.unit_cube(n)
    result = solve_elasticity_3d(
        mesh,
        degree=degree,
        local_refinement=refinement,
        constitutive=data.constitutive if data.anisotropic else None,
        lame_lambda=lam,
        source=data.source,
        dirichlet=data.displacement,
        quadrature_order=7,
        backend="process" if workers > 1 else "serial",
        workers=workers,
    )
    errors = result.errors(data.displacement, data.stress, order=9)
    row = dict(
        resolution=n,
        macro_tetrahedra=len(mesh.cells),
        degree=degree,
        local_refinement=refinement,
        local_tetrahedra=sum(len(m.cells) for m in result.local_meshes),
        vector_trace_dofs=3 * result.skeleton.size,
        coarse_rigid_modes=6 * len(mesh.cells),
        lame_lambda=lam,
        **errors,
        algebraic_residual=result.hybrid.residual,
        force_moment_linf=float(np.abs(result.equilibrium_residuals()).max()),
    )
    print(json.dumps(row), flush=True)
    return result, row


def main() -> None:
    """Run five spatial levels or five Lamé values, keeping the PDE data consistent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--degree", type=int, choices=[2, 3], default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    refinement = 2 if args.degree == 2 else 1
    data = ElasticityData3D(not args.sweep)
    sources = [
        Path(__file__),
        ROOT / "src/pymhm/_legacy/models/elasticity/primal_3d.py",
        ROOT / "src/pymhm/fem/scalar/tetrahedron.py",
        ROOT / "src/pymhm/fem/scalar/tetrahedron_topology.py",
    ]
    start_hashes = current_source_manifest(
        {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        }
    )
    rows = []
    with threadpool_limits(1):
        for n, lam in (
            [(2, v) for v in (1.0, 10.0, 100.0, 1000.0, 10000.0)]
            if args.sweep
            else [(n, 1.0) for n in args.resolutions]
        ):
            solution, row = acquire(n, args.degree, refinement, data, lam, args.workers)
            rows.append(row)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = f"{'locking' if args.sweep else 'anisotropic'}-p{args.degree}"
    path = OUTPUT / (name + ".npz")
    np.savez_compressed(
        path,
        macro_points=solution.skeleton.mesh.points,
        macro_cells=solution.skeleton.mesh.cells,
        local_points=np.stack([m.points for m in solution.local_meshes]),
        local_cells=np.stack([m.cells for m in solution.local_meshes]),
        values=np.stack(solution.values),
    )
    if start_hashes != current_source_manifest(
        {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        }
    ):
        raise RuntimeError("Acquisition sources changed during the numerical campaign")
    record = dict(
        case="Original solenoidal trigonometric 3D elasticity",
        anisotropic=data.anisotropic,
        constitutive="(1+x+2y+3z)*Kelvin SPD"
        if data.anisotropic
        else "isotropic, finite lambda; bounded force",
        trace="P1 vector densities on each triangular macroface",
        rows=rows,
        assembly_quadrature=7,
        error_quadrature=9,
        workers=args.workers,
        archive=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_hashes=current_source_manifest(
            {
                p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            }
        ),
    )
    (OUTPUT / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
