"""Five-level energy estimator studies for anisotropy and mixed material boundaries."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.reservoir import CartesianCellField
from pymhm.weighted_estimator import estimate_weighted_darcy_error

ROOT = Path(__file__).resolve().parents[1]


def problem(name: str) -> tuple:
    """Return exact pressure, gradient, source and literal certified material."""
    if name == "anisotropic":
        angle = np.pi / 6
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        material = rotation @ np.diag([100.0, 1.0]) @ rotation.T

        def pressure(x):
            """Zero-boundary oscillatory pressure of the L09 smooth example."""
            return np.sin(2 * np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1])

        def gradient(x):
            """Gradient evaluated independently of the finite element functions."""
            a, b = 2 * np.pi * x.T
            return 2 * np.pi * np.column_stack((np.cos(a) * np.sin(b), np.sin(a) * np.cos(b)))

        def source(x):
            """Full rotated-tensor diffusion, including both off-diagonal terms."""
            a, b = 2 * np.pi * x.T
            return (
                4
                * np.pi**2
                * (np.trace(material) * pressure(x) - 2 * material[0, 1] * np.cos(a) * np.cos(b))
            )
    else:
        material = CartesianCellField(np.array([[1.0], [100.0]]), (0.5, 1.0))

        def pressure(x):
            """Continuous pressure with the exact reciprocal-permeability derivative jump."""
            return 1 - np.sin(2 * np.pi * x[:, 0]) / (2 * np.pi * material(x))

        def gradient(x):
            """One-sided exact gradients; the physical normal flux is continuous."""
            return np.column_stack((-np.cos(2 * np.pi * x[:, 0]) / material(x), np.zeros(len(x))))

        def source(x):
            """Divergence of q=(cos(2 pi x),0), including no interface distribution."""
            return -2 * np.pi * np.sin(2 * np.pi * x[:, 0])

    return pressure, gradient, source, material


def main() -> None:
    """Acquire all norms, separate indicator terms and quadrature sensitivity."""
    output = ROOT / "examples/results/weighted-estimator.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    if parser.parse_args().plot_only:
        plots(json.loads(output.read_text())["records"])
        return
    rows = []
    with threadpool_limits(1):
        for name in ("anisotropic", "layered-mixed"):
            pressure, gradient, source, material = problem(name)
            for n in (2, 4, 8, 16, 32):
                mesh = TriangleMesh.unit_square(n)
                skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
                neumann = None
                if name == "layered-mixed":
                    neumann = {
                        int(face): 0.0
                        for face in mesh.boundary_faces
                        if abs(mesh.normals[face, 1]) > 0.5
                    }
                result = solve_darcy(
                    mesh,
                    skeleton=skeleton,
                    permeability=material,
                    source=source,
                    dirichlet=pressure if neumann is not None else 0.0,
                    neumann=neumann,
                    degree=3,
                    local_refinement=2,
                    quadrature_order=10,
                )
                estimate = estimate_weighted_darcy_error(
                    result,
                    degree=2,
                    dirichlet=pressure if neumann is not None else 0.0,
                    neumann=neumann,
                    quadrature_order=10,
                )
                error = estimate.energy_error(gradient, 12)
                row = dict(
                    case=name,
                    n=n,
                    macros=len(mesh.cells),
                    true_energy_error=error,
                    estimator=estimate.total,
                    effectivity=estimate.total / error,
                    flux_defect=float(np.linalg.norm(estimate.flux_defect)),
                    nonconformity=float(np.linalg.norm(estimate.nonconformity)),
                    divergence_defect=float(np.linalg.norm(estimate.divergence_defect)),
                    oscillation=float(np.linalg.norm(estimate.oscillation)),
                    equilibrium_defect=float(np.max(estimate.equilibrium_defect)),
                    residual=result.hybrid.residual,
                )
                assert estimate.total >= error * (1 - 1e-10)
                rows.append(row)
                output.write_text(
                    json.dumps(
                        dict(local_degree=3, face_degree=1, reconstruction_degree=2, records=rows),
                        indent=2,
                    )
                    + "\n"
                )
                print(row, flush=True)
    plots(rows)


def plots(rows: list[dict]) -> None:
    """Render fixed refinement ticks with room for scientific legends and labels."""
    figures = ROOT / "docs/figures/weighted-estimator"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for name, marker in (("anisotropic", "o"), ("layered-mixed", "s")):
        series = [r for r in rows if r["case"] == name]
        n = [r["n"] for r in series]
        line = axes[0].loglog(
            n, [r["true_energy_error"] for r in series], marker=marker, label=f"{name}: error"
        )[0]
        axes[0].loglog(
            n,
            [r["estimator"] for r in series],
            linestyle="--",
            marker=marker,
            color=line.get_color(),
            label=f"{name}: estimator",
        )
        axes[1].semilogx(n, [r["effectivity"] for r in series], marker=marker, label=name)
    axes[0].set(xlabel="Macro divisions N", ylabel="Energy norm")
    axes[1].set(xlabel="Macro divisions N", ylabel="Estimator / true energy error")
    axes[1].axhline(1, color="black", linewidth=0.8, linestyle=":")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32], labels=["2", "4", "8", "16", "32"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for extension in ("png", "svg"):
        fig.savefig(figures / f"convergence.{extension}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
