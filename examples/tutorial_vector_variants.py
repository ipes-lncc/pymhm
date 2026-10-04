"""Small analytical tutorials for vector primal and mixed MHM formulations.

All variants solve two macrocells with explicit physical data. Run
``python -m examples.tutorial_vector_variants --variant all`` in the locked
Pixi test environment. These affine patches are tutorials, not convergence
studies or reproductions of published heterogeneous applications.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.elasticity.mixed_pressure import solve_displacement_pressure
from pymhm._legacy.models.elasticity.mixed_pressure_3d import solve_elasticity_gals_3d
from pymhm._legacy.models.elasticity.primal import solve_primal_elasticity
from pymhm._legacy.models.elasticity.primal_3d import solve_elasticity_3d
from pymhm._legacy.models.elasticity.stress import solve_elasticity_mixed
from pymhm._legacy.models.elasticity.stress_3d import solve_elasticity_mixed_3d
from pymhm._legacy.models.elasticity.stress_tensor import solve_elasticity_tensor_rt
from pymhm._legacy.models.flow.solver import solve_flow
from pymhm._legacy.models.flow.solver_3d import solve_flow_3d
from pymhm._legacy.models.waves.maxwell import MaxwellStepper
from pymhm.core.validation import FloatArray
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.mixed import AffineMixedMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh

VARIANTS = (
    "elasticity-primal-2d",
    "elasticity-primal-3d",
    "elasticity-gals-2d",
    "elasticity-gals-3d",
    "elasticity-bdm-2d",
    "elasticity-bdm-plus-2d",
    "elasticity-bdm-plusplus-2d",
    "elasticity-bdm-3d",
    "elasticity-rt-2d",
    "elasticity-rt-plus-2d",
    "flow-taylor-hood-2d",
    "flow-taylor-hood-3d",
    "flow-usfem-2d",
    "flow-usfem-3d",
    "flow-oseen-2d",
    "flow-oseen-3d",
    "maxwell-vector-3d",
)


@dataclass(frozen=True)
class AffineVector:
    """An explicit physical displacement or velocity ``u(x)=G x+b``.

    Gradient axes are (component, derivative). This top-level callable can be
    sent to spawn workers without transferring a native finite-element object.
    """

    gradient: FloatArray
    offset: FloatArray

    def __call__(self, points: FloatArray) -> FloatArray:
        """Evaluate the vector at physical points of matching dimension."""
        return points @ self.gradient.T + self.offset


@dataclass(frozen=True)
class AffinePressure:
    """An affine pressure with a stated volume-centroid mean."""

    gradient: FloatArray
    center: FloatArray
    mean: float

    def __call__(self, points: FloatArray) -> FloatArray:
        """Evaluate ``p(x)=gradient.(x-center)+mean``."""
        return (points - self.center) @ self.gradient + self.mean


@dataclass(frozen=True)
class AffineMomentum:
    """Independent affine source for ``-Delta u+2u+beta.grad(u)+grad(p)``."""

    velocity: AffineVector
    pressure_gradient: FloatArray
    advection: FloatArray

    def __call__(self, points: FloatArray) -> FloatArray:
        """Evaluate the strong source; affine velocity has zero Laplacian."""
        return (
            2 * self.velocity(points)
            + self.advection @ self.velocity.gradient.T
            + self.pressure_gradient
        )


@dataclass(frozen=True)
class MaxwellBoundary:
    """Physical impedance data ``E_tan-(H cross n)`` for constant vector fields."""

    electric: FloatArray
    magnetic: FloatArray

    def __call__(self, time: float, points: FloatArray, normals: FloatArray) -> FloatArray:
        """Return the prescribed field; the stepper projects its tangential part."""
        return self.electric - np.cross(self.magnetic, normals)


def _tetra_patch() -> TetraMesh:
    """Select two adjacent cube tetrahedra without replacing their geometry."""
    cube = TetraMesh.unit_cube()
    return TetraMesh(cube.points, cube.cells[:2])


def _gradient(dimension: int, *, solenoidal: bool, homogeneous: bool) -> FloatArray:
    """Choose a reproducible matrix with strain, shear and rotation components."""
    gradient = (
        np.array([[1.0, 2], [-1, 3]])
        if dimension == 2
        else np.array([[1.0, 0.3, -0.2], [0.2, 3, 0.4], [-0.1, 0.5, 1]])
    )
    if solenoidal:
        gradient -= np.trace(gradient) / dimension * np.eye(dimension)
    return np.zeros_like(gradient) if homogeneous else gradient


def _maximum(values: Any) -> float:
    """Return an absolute maximum over explicitly separate fine-cell records."""
    return max((float(np.max(np.abs(value), initial=0)) for value in values), default=0.0)


def _stress_balance(solution: Any) -> dict[str, float]:
    """Report fine equilibrium, weak symmetry and normal-traction defects separately."""
    return {
        "fine_force_linf": _maximum(solution.fine_force_residuals()),
        "weak_symmetry_linf": _maximum(solution.weak_symmetry_residuals()),
        "normal_traction_linf": _maximum(solution.normal_traction_residuals()),
    }


def _maxwell(*, homogeneous: bool) -> dict[str, Any]:
    """Advance a stationary nonzero vector patch with declared impedance data."""
    electric = np.zeros(3) if homogeneous else np.array([1.2, -0.3, 0.7])
    magnetic = np.zeros(3) if homogeneous else np.array([0.4, 0.8, -0.2])
    with (
        threadpool_limits(1),
        MaxwellStepper(
            _tetra_patch(),
            time_step=0.001,
            degree=2,
            local_refinement=2,
            absorbing=1,
            boundary_data=MaxwellBoundary(electric, magnetic),
            quadrature_order=5,
        ) as stepper,
    ):
        initial = stepper.initialize(electric, magnetic)
        balance = 0.0
        for _ in range(4):
            solution = stepper.advance()
            balance = max(balance, abs(solution.energy_balance_residual))
        e_error, h_error = solution.l2_errors(electric, magnetic, order=5)
    errors = {"electric_l2": e_error, "magnetic_l2": h_error}
    if max(errors.values()) > 1e-9 or balance > 1e-10:
        raise RuntimeError("the stationary vector Maxwell patch was not recovered")
    return {
        "variant": "maxwell-vector-3d",
        "dimension": 3,
        "macrocells": 2,
        "boundary_data": "homogeneous" if homogeneous else "nonhomogeneous impedance",
        "method": "PyMHM Maxwell leapfrog stepper with reused mass/skeleton factors",
        "scope": (
            "stationary vector patch over 4 steps; no temporal convergence or wave-accuracy claim"
        ),
        "backend": "serial",
        "native_threads": 1,
        "errors": errors,
        "original_equation_residual": None,
        "solver_check": (
            "stepper verifies tangential original equations at 1e-10; "
            "scalar residual is not exposed"
        ),
        "energy_balance_residual_max": balance,
        "energy_change": solution.energy - initial.energy,
        "electric_time": solution.electric_time,
        "magnetic_time": solution.magnetic_time,
        "provenance": {
            "spaces": (
                "broken vector P2 DG fields on refined tetrahedra; tangential P1 macroface traces"
            ),
            "boundary": "E_tan-(H cross n)=given data, impedance 1",
            "electric": electric.tolist(),
            "magnetic": magnetic.tolist(),
            "time_step": 0.001,
            "steps": 4,
            "field_kind": "broken DG curl; no H(curl)-conforming reconstruction claim",
        },
    }


def _elasticity(
    variant: str,
    field: AffineVector,
    *,
    incompressible: bool,
    homogeneous: bool,
    backend: Literal["serial", "thread", "process"],
    workers: int | None,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    """Select existing elasticity owners without changing their local formulas."""
    dimension = field.gradient.shape[0]
    lame_lambda = np.inf if incompressible else 1.0
    pressure = (0.0 if homogeneous else 2.3) if incompressible else -np.trace(field.gradient)
    stress = field.gradient + field.gradient.T - pressure * np.eye(dimension)
    options: dict[str, Any] = dict(
        dirichlet=field, lame_lambda=lame_lambda, backend=backend, workers=workers
    )
    solution: Any
    mesh: Any
    method: Any
    provenance: dict[str, Any] = {
        "operator": "-div(2 sym(grad(u))+lambda div(u) I)",
        "lame_lambda": "infinity" if incompressible else 1.0,
        "pressure_role": "physical mean gauge" if incompressible else "finite compressibility",
        "expected_pressure": float(pressure),
        "expected_stress": stress.tolist(),
        "global_rigid_gauge": "none; full displacement boundary removes rigid ambiguity",
    }
    if variant == "elasticity-primal-2d":
        mesh = TriangleMesh.unit_square()
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
        solution = solve_primal_elasticity(
            mesh, skeleton=skeleton, degree=3, local_refinement=1, quadrature_order=5, **options
        )
        errors = {
            "displacement_l2": solution.l2_error(field, 5),
            "stress_l2": solution.stress_l2_error(stress, 5),
        }
        provenance["spaces"] = "local H1 P3 displacement; P1 vector macroface traces"
        provenance["stress_kind"] = "raw symmetric stress; no general fine H(div) claim"
    elif variant == "elasticity-primal-3d":
        solution = solve_elasticity_3d(
            _tetra_patch(), degree=3, local_refinement=1, quadrature_order=5, **options
        )
        errors = solution.errors(field, stress, 5)
        provenance["spaces"] = "local H1 P3 displacement; P1 vector triangular traces"
        provenance["stress_kind"] = "raw symmetric stress; no general fine H(div) claim"
    elif variant.startswith("elasticity-gals"):
        method = solve_displacement_pressure if dimension == 2 else solve_elasticity_gals_3d
        mesh = TriangleMesh.unit_square() if dimension == 2 else _tetra_patch()
        solution = method(
            mesh,
            degree=3,
            formulation="gals",
            local_refinement=1,
            quadrature_order=5,
            mean_pressure=float(pressure) if incompressible else 0.0,
            **options,
        )
        errors = {
            "displacement_l2": solution.l2_error(field, 5),
            "pressure_l2": solution.pressure_l2_error(pressure, 5),
            "stress_l2": solution.stress_l2_error(stress, 5),
            "compressibility_l2": solution.compressibility_l2(5),
        }
        provenance["spaces"] = "local H1 P3 displacement/P3 Herrmann pressure with GaLS; P1 traces"
        provenance["pressure_convention"] = "p=-lambda div(u); sigma=2 sym(grad(u))-p I"
    elif variant == "elasticity-bdm-3d":
        mesh = _tetra_patch()
        mixed_mesh = AffineMixedMesh(mesh.points, mesh.cells)
        solution = solve_elasticity_mixed_3d(
            mixed_mesh,
            stress_degree=2,
            local_refinement=1,
            quadrature_order=5,
            mean_pressure=float(pressure) if incompressible else 0.0,
            **options,
        )
        gradient = field.gradient
        rotation = (
            np.array(
                [
                    gradient[1, 2] - gradient[2, 1],
                    gradient[2, 0] - gradient[0, 2],
                    gradient[0, 1] - gradient[1, 0],
                ]
            )
            / 2
        )
        errors = solution.errors(field, stress, rotation, order=5) | _stress_balance(solution)
        provenance["spaces"] = "row-wise H(div) tetrahedral BDM2; DG P1 displacement/axial rotation"
        provenance["stress_kind"] = "full H(div) stress; symmetry imposed weakly"
    elif variant.startswith("elasticity-bdm"):
        enrichment = {
            "elasticity-bdm-2d": 0,
            "elasticity-bdm-plus-2d": 1,
            "elasticity-bdm-plusplus-2d": 2,
        }[variant]
        solution = solve_elasticity_mixed(
            TriangleMesh.unit_square(),
            stress_degree=2,
            enrichment=enrichment,
            local_refinement=1,
            quadrature_order=5,
            mean_pressure=float(pressure) if incompressible else 0.0,
            **options,
        )
        rotation = (field.gradient[0, 1] - field.gradient[1, 0]) / 2
        errors = {
            "displacement_l2": solution.l2_error(field, 5),
            "stress_l2": solution.stress_l2_error(stress, 5),
            "rotation_l2": solution.rotation_l2_error(rotation, 5),
        } | _stress_balance(solution)
        provenance["spaces"] = (
            f"BDM normal degree 2; interior BDM{2 + enrichment}; "
            f"DG P{1 + enrichment} displacement/rotation"
        )
        provenance["stress_kind"] = "full H(div) stress; symmetry imposed weakly"
    else:
        enrichment = int(variant == "elasticity-rt-plus-2d")
        solution = solve_elasticity_tensor_rt(
            CartesianMacroMesh(2, 1),
            degree=1,
            enrichment=enrichment,
            local_refinement=1,
            quadrature_order=5,
            mean_pressure=float(pressure) if incompressible else 0.0,
            **options,
        )
        rotation = (field.gradient[0, 1] - field.gradient[1, 0]) / 2
        errors = solution.errors(field, stress, (0, 0), rotation, order=5) | _stress_balance(
            solution
        )
        provenance["spaces"] = (
            f"tensor RT normal degree 1/interior order {1 + enrichment}; "
            f"Q{1 + enrichment} displacement/P{1 + enrichment} rotation"
        )
        provenance["stress_kind"] = "full H(div) stress; symmetry imposed weakly"
    return solution, errors, provenance


def run_variant(
    variant: str,
    *,
    homogeneous: bool = False,
    incompressible: bool = False,
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> dict[str, Any]:
    """Run one bounded affine patch and report each physical field independently.

    ``homogeneous=True`` sets the exact field and loads to zero. Otherwise full
    nonhomogeneous displacement/velocity data are prescribed. Elasticity uses
    lambda=mu=1, except an explicitly requested incompressible mixed limit.
    Primal elasticity requires finite lambda. Flow always uses div(u)=0,
    viscosity 1, resistance 2, an explicit pressure mean, and grad-grad diffusion.
    No exact patch is substituted for a published application in this tutorial.
    """
    if variant not in VARIANTS:
        raise ValueError("variant must be one of the declared vector tutorials")
    if incompressible and variant.startswith("elasticity-primal"):
        raise ValueError("primal elasticity requires finite lambda; select a mixed variant")
    if variant == "maxwell-vector-3d":
        if incompressible or backend != "serial":
            raise ValueError(
                "the Maxwell stepper is serial and has no elasticity compressibility option"
            )
        return _maxwell(homogeneous=homogeneous)
    dimension = 3 if variant.endswith("3d") else 2
    flow = variant.startswith("flow-")
    gradient = _gradient(dimension, solenoidal=flow or incompressible, homogeneous=homogeneous)
    offset = np.zeros(dimension) if homogeneous else np.array([1.0, -2, 3])[:dimension]
    field = AffineVector(gradient, offset)
    solution: Any
    mesh: Any
    method: Any
    with threadpool_limits(1):
        if flow:
            mesh = TriangleMesh.unit_square() if dimension == 2 else _tetra_patch()
            measures = mesh.areas if dimension == 2 else mesh.volumes
            center = measures @ mesh.points[mesh.cells].mean(axis=1) / sum(measures)
            pressure_gradient = (
                np.zeros(dimension) if homogeneous else np.array([1.0, 2, 1])[:dimension]
            )
            mean = 0.0 if homogeneous else 0.7
            pressure = AffinePressure(pressure_gradient, center, mean)
            formulation = variant.removeprefix("flow-").removesuffix(f"-{dimension}d")
            beta = (
                np.array([1.0, 0.5, 0.2])[:dimension]
                if formulation == "oseen"
                else np.zeros(dimension)
            )
            method = solve_flow if dimension == 2 else solve_flow_3d
            trace_options: dict[str, Any] = (
                {
                    "skeleton": SkeletonSpace(
                        mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2
                    )
                }
                if dimension == 2
                else {}
            )
            solution = method(
                mesh,
                formulation=formulation,
                degree=3,
                local_refinement=1,
                quadrature_order=5,
                viscosity=1,
                drag=2,
                advection=beta,
                source=AffineMomentum(field, pressure_gradient, beta),
                dirichlet=field,
                mean_pressure=mean,
                backend=backend,
                workers=workers,
                **trace_options,
                **({"stabilization": "minimum-2017"} if formulation == "usfem" else {}),
            )
            errors = {
                "velocity_l2": solution.l2_error(field, 5),
                "pressure_l2": solution.pressure_l2_error(pressure, 5),
                "divergence_l2": (
                    solution.divergence_l2() if dimension == 2 else solution.divergence_l2(5)
                ),
            }
            provenance = {
                "operator": "-Delta u+2u+beta.grad(u)+grad(p); div(u)=0",
                "spaces": "local H1 velocity P3/pressure P2"
                if formulation == "taylor-hood"
                else "local H1 velocity P3/pressure P3 with residual stabilization",
                "trace_space": "P1 vector macroface traces; represents affine pseudotraction",
                "boundary_kind": "grad-grad pseudotraction; not symmetric Cauchy traction",
                "pressure_mean": mean,
                "pressure_gradient": pressure_gradient.tolist(),
                "pressure_center": center.tolist(),
                "advection": beta.tolist(),
                "stabilization": "none"
                if formulation == "taylor-hood"
                else ("minimum-2017" if formulation == "usfem" else "Oseen-2021"),
            }
        else:
            solution, errors, provenance = _elasticity(
                variant,
                field,
                incompressible=incompressible,
                homogeneous=homogeneous,
                backend=backend,
                workers=workers,
            )
    if max(errors.values(), default=0.0) > 1e-9:
        raise RuntimeError(
            f"{variant}: the exact affine physical patch was not recovered: {errors}"
        )
    hybrid = solution.hybrid
    return {
        "variant": variant,
        "dimension": dimension,
        "macrocells": len(solution.local_meshes),
        "boundary_data": "homogeneous" if homogeneous else "nonhomogeneous affine",
        "method": "PyMHM implementation of the stated local spaces and shared hybrid algebra",
        "scope": "analytical patch; no convergence-rate or matched-literature reproduction claim",
        "backend": backend,
        "native_threads": 1,
        "errors": {name: float(value) for name, value in errors.items()},
        "original_equation_residual": hybrid.raw_residual,
        "gradient": gradient.tolist(),
        "offset": offset.tolist(),
        "provenance": provenance,
    }


def main() -> None:
    """Select one tutorial or all 17 variants, preserving a guarded spawn entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default=VARIANTS[0])
    parser.add_argument("--homogeneous", action="store_true")
    parser.add_argument("--incompressible", action="store_true")
    parser.add_argument("--backend", choices=("serial", "thread", "process"), default="serial")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    selected = VARIANTS if args.variant == "all" else (args.variant,)
    rows = [
        run_variant(
            variant,
            homogeneous=args.homogeneous,
            incompressible=args.incompressible,
            backend=args.backend,
            workers=args.workers,
        )
        for variant in selected
    ]
    print(json.dumps({"variant_count": len(rows), "rows": rows}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
