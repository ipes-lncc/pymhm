"""Acquire analytical tests of scalar MHM-UNUSUAL reaction--diffusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from field_sampling import sample_field, sample_profile
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.elements import tensor_values, triangle_quadrature
from pymhm.lagrange import nodal_space, tabulate
from pymhm.rad import solve_rad
from pymhm.unusual import UnusualParameters

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "examples/results/unusual"
TENSOR = np.array([[2.0, 0.3], [0.3, 1.0]])


def layer_exact(points: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Evaluate Eq. (22) stably with reaction one and vertical Dirichlet sides."""
    a = 1 / np.sqrt(epsilon)
    x = points[:, 0]
    return 1 - (np.exp(-a * x) + np.exp(-a * (1 - x))) / (1 + np.exp(-a))


def layer_gradient(points: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Differentiate the two exact reaction boundary layers."""
    a = 1 / np.sqrt(epsilon)
    x = points[:, 0]
    derivative = a * (np.exp(-a * x) - np.exp(-a * (1 - x))) / (1 + np.exp(-a))
    return np.column_stack((derivative, np.zeros(len(points))))


def tensor_material(points: np.ndarray) -> np.ndarray:
    """Return a smooth anisotropic SPD tensor with an analytical ellipticity bound."""
    return (1 + points.sum(axis=1))[:, None, None] * TENSOR


def tensor_exact(points: np.ndarray) -> np.ndarray:
    """Return the sine manufactured scalar with homogeneous boundary values."""
    return np.prod(np.sin(np.pi * points), axis=1)


def tensor_gradient(points: np.ndarray) -> np.ndarray:
    """Differentiate the manufactured sine field independently of the solver."""
    x, y = (np.pi * points).T
    return np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))


def tensor_reaction(points: np.ndarray) -> np.ndarray:
    """Return the nonnegative variable reaction sigma=3+x."""
    return 3 + points[:, 0]


def tensor_source(points: np.ndarray) -> np.ndarray:
    """Apply sigma*u-div(A grad u), including derivatives of both tensor columns."""
    x, y = (np.pi * points).T
    value = tensor_exact(points)
    return (
        tensor_reaction(points) * value
        + (1 + points.sum(axis=1)) * np.pi**2 * (3 * value - 0.6 * np.cos(x) * np.cos(y))
        - tensor_gradient(points) @ (TENSOR @ np.ones(2))
    )


def configuration(case: str, epsilon: float) -> tuple[dict[str, Any], Any, Any]:
    """Return physical data and independent exact fields for one campaign family."""
    if case == "tensor":
        return (
            dict(
                diffusion=tensor_material,
                diffusion_divergence=TENSOR @ np.ones(2),
                reaction=tensor_reaction,
                source=tensor_source,
                unusual_parameters=UnusualParameters(np.linalg.eigvalsh(TENSOR)[0], 4.0),
            ),
            tensor_exact,
            tensor_gradient,
        )
    return (
        dict(diffusion=epsilon, reaction=1.0, source=1.0),
        partial(layer_exact, epsilon=epsilon),
        partial(layer_gradient, epsilon=epsilon),
    )


def errors(solution: Any, material: Any, exact: Any, exact_gradient: Any, order: int) -> dict:
    """Integrate absolute scalar, gradient, physical flux and diffusion-energy errors."""
    bary, weights = triangle_quadrature(order)
    totals = np.zeros(6)
    for fine, values in zip(solution.local_meshes, solution.values, strict=True):
        dofs, _, basis, gradient, _ = tabulate(fine, solution.degree, bary)
        physical = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
        flat = physical.reshape(-1, 2)
        truth = exact(flat).reshape(physical.shape[:2])
        grad_truth = exact_gradient(flat).reshape(physical.shape)
        material_values = tensor_values(material, flat).reshape(*physical.shape[:2], 2, 2)
        delta = values[dofs] @ basis.T - truth
        derivative = np.einsum("ti,tqia->tqa", values[dofs], gradient)
        delta_grad = derivative - grad_truth
        flux_delta = np.einsum("tqab,tqb->tqa", material_values, delta_grad)
        flux_truth = np.einsum("tqab,tqb->tqa", material_values, grad_truth)
        quantities = (
            delta**2,
            np.sum(delta_grad**2, axis=-1),
            np.sum(flux_delta**2, axis=-1),
            np.sum(flux_delta * delta_grad, axis=-1),
            truth**2,
            np.sum(flux_truth**2, axis=-1),
        )
        totals += [float(fine.areas @ (quantity @ weights)) for quantity in quantities]
    l2, gradient, flux, energy, scalar_norm, flux_norm = np.sqrt(totals)
    return dict(
        scalar_l2=float(l2),
        gradient_l2=float(gradient),
        flux_l2=float(flux),
        diffusion_energy=float(energy),
        scalar_relative_l2=float(l2 / scalar_norm),
        flux_relative_l2=float(flux / flux_norm),
    )


def acquire(case: str, n: int, epsilon: float, method: str, *, archive: bool) -> dict:
    """Solve matching macro/local spaces, verify norms and optionally preserve display data."""
    degree = 2 if case == "tensor" else 1
    mesh = TriangleMesh.unit_square(n)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(degree - 1) for _ in mesh.faces))
    options, exact, gradient = configuration(case, epsilon)
    if method == "galerkin":
        options.pop("unusual_parameters", None)
    natural = (
        {}
        if case == "tensor"
        else {int(f): 0.0 for f in mesh.boundary_faces if abs(mesh.normals[f, 1]) > 0.99}
    )
    solution = solve_rad(
        mesh,
        skeleton=skeleton,
        degree=degree,
        local_refinement=2,
        stabilization=method,
        dirichlet_enforcement="strong",
        neumann=natural,
        quadrature_order=10,
        **options,
    )
    order = 32 if epsilon < 1e-3 and case != "tensor" else 12
    metrics = errors(solution, options["diffusion"], exact, gradient, order)
    nodal_error, minimum, maximum = 0.0, np.inf, -np.inf
    for fine, values in zip(solution.local_meshes, solution.values, strict=True):
        nodes = nodal_space(fine, degree)[1]
        nodal_error = max(nodal_error, float(abs(values - exact(nodes)).max()))
        minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
    row = dict(
        case=case,
        method=method,
        n=n,
        epsilon=epsilon,
        local_degree=degree,
        trace_degree=degree - 1,
        local_refinement=2,
        macro_cells=len(mesh.cells),
        fine_cells=sum(len(fine.cells) for fine in solution.local_meshes),
        trace_dofs=skeleton.size,
        coarse_dofs=sum(len(coarse) for coarse in solution.hybrid.coarse),
        residual=solution.hybrid.residual,
        nodal_error=nodal_error,
        scalar_min=minimum,
        scalar_max=maximum,
        error_quadrature=order,
        **metrics,
    )
    if archive or n == 32:
        higher = errors(solution, options["diffusion"], exact, gradient, order + 8)
        row["quadrature_check"] = dict(order=order + 8, **higher)
    if archive:
        data = sample_field(solution.local_meshes, solution.values, degree, refinement=3)
        data["exact"] = exact(data["points"])
        data["exact_gradient"] = gradient(data["points"])
        tensors = tensor_values(options["diffusion"], data["points"])
        data["flux"] = -np.einsum("qab,qb->qa", tensors, data["gradient"])
        data["exact_flux"] = -np.einsum("qab,qb->qa", tensors, data["exact_gradient"])
        data.update(sample_profile(solution, solution.values, degree))
        data["profile_exact"] = exact(data["profile_points"].reshape(-1, 2)).reshape(
            data["profile_parameter"].shape
        )
        data.update(macro_points=mesh.points, macro_cells=mesh.cells)
        name = f"{case}-n{n}-eps{epsilon:g}-{method}.npz"
        np.savez_compressed(TARGET / name, **data)
        row["archive"] = name
        row["archive_sha256"] = hashlib.sha256((TARGET / name).read_bytes()).hexdigest()
    return row


def main() -> None:
    """Run five spatial levels and five diffusion regimes with exact solutions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Run only the first two spatial levels"
    )
    arguments = parser.parse_args()
    TARGET.mkdir(parents=True, exist_ok=True)
    sources = ("src/pymhm/unusual.py", "src/pymhm/rad.py", "examples/solve_unusual.py")
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    levels = (2, 4) if arguments.smoke else (2, 4, 8, 16, 32)
    settings = [
        (case, n, epsilon, method)
        for case, epsilon in (("smooth", 1.0), ("layer", 1e-3), ("tensor", 1.0))
        for n in levels
        for method in ("galerkin", "unusual")
    ]
    settings += [
        ("reaction-sweep", 16, epsilon, method)
        for epsilon in (1.0, 1e-2, 1e-3, 1e-4, 1e-5)
        for method in ("galerkin", "unusual")
    ]
    rows = []
    with threadpool_limits(1):
        for case, n, epsilon, method in settings:
            archive = n == 16 and case != "smooth" and (case != "reaction-sweep" or epsilon == 1e-5)
            row = acquire(case, n, epsilon, method, archive=archive)
            rows.append(row)
            print(json.dumps(row), flush=True)
            record = dict(
                formulation="Santiago, Valentin and Martins, CILAMCE2025, Eqs14-15; beta=0",
                doi="10.55592/cilamce2025.v5i.14270",
                spatial_configuration=(
                    "SW-NE triangular macros; one red local refinement; DG trace P(k-1)"
                ),
                dirichlet_enforcement="strong on local exterior nodes",
                inverse_parameter="P1 m=1/3; P2 automatic physical operator-inverse bound",
                norms=(
                    "Absolute broken physical norms integrated elementwise; "
                    "relative denominators are exact fields."
                ),
                source_hashes=hashes,
                source_changed_during_run=any(
                    hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value
                    for name, value in hashes.items()
                ),
                rows=rows,
            )
            (TARGET / "analytical.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
