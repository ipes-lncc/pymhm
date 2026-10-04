"""Generalized RAD conditioning with the five material cases of L12 section 5.3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from polygon_meshes import polygon_partition
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, HybridSystem, LocalProblem, SkeletonSpace
from pymhm._legacy.models.transport.rad import _rad_local
from pymhm.linalg.linear import LinearSolveError

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (75.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 1.0),
    (1.0, 0.5, 0.0, 0.0),
    (1.0, 0.0, 1.0, 10.0),
    (25.0, 1.0, 1.0, 0.0),
)


@dataclass(frozen=True)
class Coefficients:
    """Published continuous checkerboard circulation, horizontal drift and reaction."""

    case: int
    omega: float

    def velocity(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the divergence-free vector field without changing its amplitude."""
        chi, omega0, _, _ = CASES[self.case - 1]
        indices = np.minimum(np.floor(4 * points).astype(int), 3)
        s, t = (4 * points - indices).T
        sign = (-1.0) ** np.sum(indices, axis=1)
        beta = sign[:, None] * np.column_stack(
            ((s * s - s) * (1 - 2 * t), (1 - 2 * s) * (t - t * t))
        )
        y = points[:, 1]
        delta = np.where(
            y <= 0.5,
            self.omega,
            np.where(y < 0.75, (3 - 4 * y) * (self.omega - omega0) + omega0, omega0),
        )
        beta *= chi
        beta[:, 0] += delta
        return beta

    def reaction(self, points: np.ndarray) -> np.ndarray:
        """Return nu1 on the left half and nu0 on the right half, as in Table 1."""
        _, _, nu0, nu1 = CASES[self.case - 1]
        return np.where(points[:, 0] <= 0.5, nu1, nu0)


def measure(
    case: int,
    epsilon: float,
    omega: float,
    refinement: int = 1,
    segments: int = 1,
    order: int = 6,
    condition_local: bool = True,
) -> dict:
    """Measure unscaled Euclidean condition numbers and verified solved residuals.

    The red square in Figure 7 is [1/4,1/2]^2, cell 5 in row-major numbering.
    Nullspace locals use the reduced nodal matrix of the article's section 5.1;
    invertible locals use the full nodal matrix. The global matrix uses physical
    constant face and cell bases, with no solver equilibration in the SVD.
    """
    mesh = polygon_partition(4, "square")
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, segments) for _ in mesh.faces))
    data = Coefficients(case, omega)
    assemblies = [
        _rad_local(
            cell,
            mesh=mesh,
            skeleton=skeleton,
            degree=2,
            refinement=refinement,
            diffusion=epsilon,
            diffusion_divergence=(0.0, 0.0),
            velocity=data.velocity,
            velocity_divergence=0.0,
            reaction=data.reaction,
            source=1.0,
            stabilization="galerkin",
            order=order,
            coarse_space="kernel",
        )
        for cell in range(len(mesh.cells))
    ]
    problem = assemblies[5].problem
    local_size = problem.matrix.shape[0] - problem.kernel.shape[1]
    local_condition = None
    if condition_local:
        operator = problem.matrix.toarray()
        if problem.kernel.shape[1]:
            operator = operator[:-1, :-1]
        singular = np.linalg.svd(operator, compute_uv=False)
        local_condition = float(singular[0] / singular[-1])
    row = dict(
        case=case,
        epsilon=epsilon,
        omega=omega,
        refinement=refinement,
        segments=segments,
        order=order,
        local_dofs=local_size,
        local_condition=local_condition,
        kernel_cells=sum(a.problem.kernel.shape[1] for a in assemblies),
        global_refinement_precision="extended",
    )
    try:
        # Keep constants while solving the physical equations. Eliminate only
        # invertible coarse amplitudes afterwards to measure the article's
        # coordinates without factoring nearly singular unrestricted locals.
        problems = [
            LocalProblem(
                a.problem.matrix,
                a.problem.coupling,
                a.problem.load,
                a.problem.trace_dofs,
                coarse_basis=np.ones((len(a.problem.load), 1)),
                constraints=a.metadata[1][:, None],
            )
            for a in assemblies
        ]
        system = HybridSystem(problems, local_refinement_precision="extended")
        full = np.asarray(system.matrix.toarray(), dtype=np.longdouble)
        eliminate = np.array(
            [
                system.trace_size + cell
                for cell, a in enumerate(assemblies)
                if not a.problem.kernel.shape[1]
            ],
            dtype=int,
        )
        keep = np.setdiff1d(np.arange(len(full)), eliminate)
        diagonal = np.diag(full)[eliminate]
        assert np.array_equal(full[np.ix_(eliminate, eliminate)], np.diag(diagonal))
        reduced = (
            full[np.ix_(keep, keep)]
            - (full[np.ix_(keep, eliminate)] / diagonal) @ full[np.ix_(eliminate, keep)]
        )
        singular = np.linalg.svd(np.asarray(reduced, dtype=float), compute_uv=False)
        result = system.solve(refinement_precision="extended")
    except LinearSolveError as error:
        row.update(status="not_certified", reason=str(error))
    else:
        row.update(
            status="verified",
            global_condition=float(singular[0] / singular[-1]),
            global_dofs=len(keep),
            residual=result.residual,
            reduction="exact elimination of non-null coarse amplitudes",
        )
    return row


def main() -> None:
    """Run independent coefficient, local-resolution and skeletal-resolution sweeps."""
    target = ROOT / "examples/results/rad-conditioning.json"
    records = json.loads(target.read_text())["records"] if target.exists() else []
    settings = []
    for case in range(1, 6):
        for value in np.logspace(-8, 4, 25):
            settings.append(("epsilon", case, float(value), 1.0, 1, 1, 6))
        for value in np.logspace(-4, 8, 25):
            settings.append(("omega", case, 1.0, float(value), 1, 1, 6))
        for segments in (1, 2, 4, 8, 16):
            settings.append(("skeleton", case, 1.0, 1.0, segments, segments, 6))
    for refinement in (1, 2, 4, 8, 16, 32):
        settings.append(("local", 2, 1.0, 1.0, refinement, 1, 6))
    for case in range(1, 6):
        settings.append(("quadrature", case, 0.01, 1.0, 1, 1, 8))
        settings.append(("quadrature", case, 0.01, 1.0, 1, 1, 6))
    with threadpool_limits(1):
        for study, case, epsilon, omega, refinement, segments, order in settings:
            key = [study, case, epsilon, omega, refinement, segments, order]
            previous = next((row for row in records if row["key"] == key), None)
            if previous is not None and previous.get("global_refinement_precision") == "extended":
                continue
            row = measure(
                case, epsilon, omega, refinement, segments, order, condition_local=study != "local"
            )
            row.update(study=study, key=key)
            if previous is not None:
                records.remove(previous)
            records.append(row)
            target.write_text(
                json.dumps(
                    dict(
                        reference="10.1016/j.cma.2024.117089, Table 1 and Figures 7–10",
                        macro_mesh="16 squares; four center-fan triangles per square initially",
                        method="selective MHM; local P2; discontinuous P0 face segments",
                        condition_norm="Euclidean spectral condition; unscaled physical bases",
                        records=records,
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(row, flush=True)


if __name__ == "__main__":
    main()
