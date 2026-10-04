"""Small analytical patches for public scalar solvers and local element choices.

Run ``python -m examples.tutorial_scalar_variants --variant all``. These are
introductory physical checks, not refinement studies or paper reproductions.
Darcy uses ``q=-grad(p)`` and ``div(q)=f`` on the unit square/cube. Norms are
reported separately from conservation moments and algebraic residuals.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from pymhm import (
    AffineMixedMesh,
    CartesianMacroMesh,
    FaceSpace,
    SkeletonSpace,
    TetraMesh,
    TriangleMesh,
    TriangularSkeleton,
)
from pymhm._legacy.models.darcy.cartesian import solve_darcy_quadrilateral
from pymhm._legacy.models.darcy.hdiv_3d import solve_darcy_hdiv3d
from pymhm._legacy.models.darcy.mixed_bdm import solve_darcy_bdm
from pymhm._legacy.models.darcy.mixed_rt import solve_darcy_rt, solve_darcy_rt_conforming
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm._legacy.models.darcy.primal_3d import solve_darcy_3d
from pymhm._legacy.models.darcy.tensor import solve_darcy_tensor_rt
from pymhm._legacy.models.transport.solver import solve_transport
from pymhm._legacy.models.waves.helmholtz import solve_helmholtz
from pymhm.core.validation import FloatArray, positive_int
from pymhm.methods.hho import solve_mshho
from pymhm.methods.hho_3d import solve_mshho_3d
from pymhm.methods.petrov_galerkin import solve_pgmhm
from pymhm.methods.robin import solve_mh
from pymhm.methods.robin_3d import solve_mh_3d
from pymhm.methods.three_field import solve_mh2m
from pymhm.methods.three_field_3d import solve_mh2m_3d


class _PatchOptions(TypedDict):
    """Keep the shared physical patch inputs explicit at every solver call."""

    dirichlet: Callable[[FloatArray], FloatArray]
    source: float
    local_refinement: int
    quadrature_order: int


@dataclass(frozen=True)
class Variant:
    """Declare the local spaces and the global method of one small patch.

    ``refinement`` subdivides each macrocell edge, except for ``classical-rt1``
    where it subdivides the globally conforming mesh. There is one unsplit
    skeletal segment/subtriangle per macroface. ``flux_kind`` refers to the
    volume field, not to a skeletal multiplier or an H(div) reconstruction.
    """

    dimension: int
    method: str
    pressure_space: str
    flux_space: str
    trace_space: str
    flux_kind: str = "raw negative pressure gradient"
    refinement: int = 2
    boundary: str = "nonhomogeneous pressure Dirichlet"
    gauge: float | None = None
    scalar_name: str = "Darcy pressure"
    quadratic: bool = False


VARIANTS: dict[str, Variant] = {
    "primal": Variant(2, "primal MHM", "local continuous P2", "[P1]^2", "P0 normal flux"),
    "primal-quadratic": Variant(
        2, "primal MHM", "local continuous P2", "[P1]^2", "P1 normal flux", quadratic=True
    ),
    "primal-neumann": Variant(
        2,
        "primal MHM",
        "local continuous P2",
        "[P1]^2",
        "P0 normal flux",
        boundary="outward physical flux Neumann on every boundary face",
        gauge=2.5,
    ),
    "primal-rectangle": Variant(
        2, "primal MHM", "local continuous Q2", "negative Q2 gradient", "P0 normal flux"
    ),
    "primal3d": Variant(3, "primal MHM", "local continuous P2", "[P1]^3", "P0 normal flux"),
    "rt0": Variant(
        2, "mixed-local MHM", "discontinuous P0", "RT0", "P0 normal flux", "local H(div)", 1
    ),
    "rt1": Variant(
        2, "mixed-local MHM", "discontinuous P1", "RT1", "P1 normal flux", "local H(div)", 1
    ),
    "classical-rt1": Variant(
        2,
        "classical conforming mixed FEM",
        "discontinuous P1",
        "RT1",
        "no MHM skeleton",
        "global H(div)",
    ),
    "bdm": Variant(
        2, "mixed-local MHM", "discontinuous P1", "BDM2", "P1 normal flux", "local H(div)", 1
    ),
    "bdm-plus": Variant(
        2,
        "mixed-local MHM",
        "discontinuous P1",
        "BDM1 face modes + all BDM2 zero-normal bubbles",
        "P1 normal flux",
        "local H(div)",
        1,
    ),
    "bdm-double-plus": Variant(
        2,
        "mixed-local MHM",
        "discontinuous P2",
        "BDM1 face modes + all BDM3 zero-normal bubbles",
        "P1 normal flux",
        "local H(div)",
        1,
    ),
    "tensor-rt": Variant(
        2,
        "mixed-local MHM",
        "discontinuous Q1",
        "RT1 on rectangles",
        "P1 normal flux",
        "local H(div)",
        1,
    ),
    "tensor-rt-plus": Variant(
        2,
        "mixed-local MHM",
        "discontinuous Q1",
        "RT0 face modes + all RT1 zero-normal bubbles",
        "P0 normal flux",
        "local H(div)",
        1,
    ),
    "hdiv-tetra": Variant(
        3,
        "mixed-local MHM",
        "discontinuous P1",
        "restricted [P2]^3, dimension 18",
        "P1 normal flux",
        "local H(div)",
        1,
    ),
    "hdiv-tetra-plus": Variant(
        3,
        "mixed-local MHM",
        "discontinuous P2",
        "P1 face modes + all [P3]^3 zero-normal bubbles, dimension 32",
        "P1 normal flux",
        "local H(div)",
        1,
    ),
    "hdiv-prism": Variant(
        3,
        "mixed-local MHM",
        "discontinuous W(1,1)=P1(triangle) tensor P1(interval)",
        "restricted prism tensor construction, dimension 27",
        "P1 on triangular faces; Q1 on rectangular faces",
        "local H(div)",
        1,
    ),
    "mh-robin": Variant(2, "Robin MH", "local continuous P2", "[P1]^2", "P1 Robin multiplier"),
    "mh-robin3d": Variant(
        3,
        "Robin MH dimensional extension",
        "local continuous P2",
        "[P1]^3",
        "P1 Robin multiplier",
    ),
    "mh2m": Variant(
        2,
        "three-field MH2M",
        "local continuous P2",
        "[P1]^2",
        "continuous pressure Gamma=P1; outward conormal Lambda=P0",
    ),
    "mh2m3d": Variant(
        3,
        "three-field MH2M",
        "local continuous P2",
        "[P1]^3",
        "continuous pressure Gamma=P1; outward conormal Lambda=P0",
    ),
    "mshho": Variant(
        2,
        "MsHHO projected source, cell degree m=0",
        "local continuous P2",
        "[P1]^2",
        "P0 pressure moments",
    ),
    "mshho-face": Variant(
        2,
        "MsHHO reconstructed source, cell degree m=-1",
        "local continuous P2",
        "[P1]^2",
        "P0 pressure moments",
    ),
    "mshho3d": Variant(
        3,
        "MsHHO projected source, cell degree m=0",
        "local continuous P2",
        "[P1]^3",
        "P0 pressure moments",
    ),
    "pgmhm": Variant(
        2,
        "Petrov-Galerkin MHM, alpha=0.1, enriched field",
        "local continuous P3",
        "[P2]^2",
        "P1 base multiplier with conservative residual enrichment",
    ),
    "rad": Variant(
        2,
        "RAD with Galerkin locals",
        "local continuous P2",
        "negative P2 gradient",
        "P1 half-advection Robin multiplier",
        scalar_name="RAD scalar",
    ),
    "rad-supg": Variant(
        2,
        "RAD with consistent SUPG locals",
        "local continuous P2",
        "negative P2 gradient",
        "P1 half-advection Robin multiplier",
        scalar_name="RAD scalar",
    ),
    "rad-unusual": Variant(
        2,
        "reaction-diffusion with UNUSUAL locals",
        "local continuous P2",
        "negative P2 gradient",
        "P1 physical normal diffusive flux",
        scalar_name="reaction-diffusion scalar",
    ),
    "helmholtz": Variant(
        2,
        "complex Helmholtz MHM, omega=0.5",
        "local continuous complex P2",
        "negative complex P2 gradient",
        "P1 physical normal flux",
        scalar_name="complex acoustic pressure",
    ),
}


def affine_pressure(points: FloatArray) -> FloatArray:
    """Return ``1+x+2y`` or ``1+x+2y+3z`` in physical coordinates."""
    return 1 + points @ np.arange(1, points.shape[1] + 1, dtype=float)


def affine_flux(points: FloatArray) -> FloatArray:
    """Return the independently differentiated physical Darcy flux ``-grad(p)``."""
    return np.broadcast_to(-np.arange(1, points.shape[1] + 1, dtype=float), points.shape)


def quadratic_pressure(points: FloatArray) -> FloatArray:
    """Return the nonharmonic pressure ``1+x²+2y²``, with source ``-6``."""
    return 1 + points[:, 0] ** 2 + 2 * points[:, 1] ** 2


def quadratic_flux(points: FloatArray) -> FloatArray:
    """Return ``(-2x,-4y)`` from differentiation of the analytical pressure."""
    return -2 * points * np.array([1.0, 2.0])


def rad_source(points: FloatArray) -> FloatArray:
    """Return ``beta.grad(u)+c*u=-0.75+0.5*u`` for the affine RAD patch."""
    return -0.75 + 0.5 * affine_pressure(points)


def reaction_source(points: FloatArray) -> FloatArray:
    """Return ``0.5*u`` for the zero-advection reaction-diffusion patch."""
    return 0.5 * affine_pressure(points)


def acoustic_pressure(points: FloatArray) -> NDArray[np.complex128]:
    """Return ``(1+i)*(1+x+2y)`` for a complex nonresonant acoustic patch."""
    return np.asarray((1 + 1j) * affine_pressure(points), dtype=np.complex128)


def acoustic_source(points: FloatArray) -> NDArray[np.complex128]:
    """Return ``-omega²*p`` at ``omega=0.5``, since the affine Laplacian vanishes."""
    return np.asarray(-0.25 * acoustic_pressure(points), dtype=np.complex128)


def solve_variant(variant: str, *, refinement: int | None = None) -> Any:
    """Execute one named patch through its public high-level solver.

    All patches use identity diffusion/permeability. ``refinement`` overrides
    the declared edge subdivisions; 3D tetrahedral refinements must be dyadic.
    Classical RT uses global mesh subdivisions instead. High-level solvers
    own assembly, actual bases, trace signs, boundary elimination and gauges.
    An invalid family name raises ``ValueError`` before constructing a mesh.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown scalar variant: {variant}")
    spec = VARIANTS[variant]
    r = spec.refinement if refinement is None else positive_int(refinement, "refinement")
    boundary = quadratic_pressure if spec.quadratic else affine_pressure
    source = -6.0 if spec.quadratic else 0.0
    options: _PatchOptions = dict(
        dirichlet=boundary, source=source, local_refinement=r, quadrature_order=6
    )
    if variant.startswith("hdiv-"):
        return solve_darcy_hdiv3d(
            AffineMixedMesh.unit_cube(kind="prism" if variant == "hdiv-prism" else "tetrahedron"),
            pressure_degree=2 if variant == "hdiv-tetra-plus" else 1,
            normal_degree=1,
            trace_degree=1,
            **options,
        )
    if variant in {"primal3d", "mh-robin3d", "mh2m3d", "mshho3d"}:
        tetra = TetraMesh.unit_cube()
        if variant == "primal3d":
            return solve_darcy_3d(tetra, degree=2, **options)
        if variant == "mh-robin3d":
            return solve_mh_3d(
                tetra,
                degree=2,
                skeleton=TriangularSkeleton(tetra, degree=1),
                robin_parameter=0.1,
                **options,
            )
        if variant == "mh2m3d":
            return solve_mh2m_3d(tetra, degree=2, **options)
        return solve_mshho_3d(tetra, degree=2, cell_degree=0, **options)
    if variant in {"primal-rectangle", "tensor-rt", "tensor-rt-plus"}:
        rectangle = CartesianMacroMesh(2, 1)
        if variant == "primal-rectangle":
            return solve_darcy_quadrilateral(rectangle, degree=2, **options)
        return solve_darcy_tensor_rt(
            rectangle,
            degree=0 if variant == "tensor-rt-plus" else 1,
            enrichment=1 if variant == "tensor-rt-plus" else 0,
            **options,
        )
    if variant == "classical-rt1":
        return solve_darcy_rt_conforming(
            TriangleMesh.unit_square(r),
            degree=1,
            source=0,
            dirichlet=affine_pressure,
            quadrature_order=6,
        )
    triangle = TriangleMesh.unit_square()
    if variant in {"primal", "primal-quadratic", "primal-neumann"}:
        skeleton = SkeletonSpace(
            triangle, tuple(FaceSpace.uniform(int(spec.quadratic)) for _ in triangle.faces)
        )
        neumann = None
        if spec.gauge is not None:
            neumann = {
                int(face): float(np.array([-1.0, -2.0]) @ triangle.normals[face])
                for face in triangle.boundary_faces
            }
        return solve_darcy(
            triangle,
            degree=2,
            skeleton=skeleton,
            neumann=neumann,
            mean_pressure=spec.gauge or 0.0,
            **options,
        )
    if variant in {"rt0", "rt1"}:
        return solve_darcy_rt(triangle, degree=int(variant == "rt1"), **options)
    if variant in {"bdm", "bdm-plus", "bdm-double-plus"}:
        return solve_darcy_bdm(
            triangle,
            degree=2 if variant == "bdm" else 1,
            enrichment={"bdm": 0, "bdm-plus": 1, "bdm-double-plus": 2}[variant],
            **options,
        )
    if variant == "mh2m":
        return solve_mh2m(triangle, degree=2, **options)
    if variant in {"mshho", "mshho-face"}:
        return solve_mshho(
            triangle,
            degree=2,
            cell_degree=-1 if variant == "mshho-face" else 0,
            source_variant="reconstructed" if variant == "mshho-face" else "projected",
            **options,
        )
    linear_trace = SkeletonSpace(triangle, tuple(FaceSpace.uniform(1) for _ in triangle.faces))
    if variant == "mh-robin":
        return solve_mh(triangle, degree=2, skeleton=linear_trace, robin_parameter=0.1, **options)
    if variant == "pgmhm":
        return solve_pgmhm(
            triangle, degree=3, skeleton=linear_trace, stabilization_parameter=0.1, **options
        )
    if variant == "helmholtz":
        return solve_helmholtz(
            triangle,
            omega=0.5,
            degree=2,
            skeleton=SkeletonSpace(triangle, linear_trace.faces, components=2),
            absorbing=None,
            dirichlet=acoustic_pressure,
            source=acoustic_source,
            local_refinement=r,
            quadrature_order=6,
        )
    stabilization = {"rad": "galerkin", "rad-supg": "supg", "rad-unusual": "unusual"}[variant]
    return solve_transport(
        triangle,
        degree=2,
        skeleton=linear_trace,
        diffusion=1,
        velocity=(0.0, 0.0) if variant == "rad-unusual" else (0.25, -0.5),
        reaction=0.5,
        source=reaction_source if variant == "rad-unusual" else rad_source,
        dirichlet=affine_pressure,
        local_refinement=r,
        quadrature_order=6,
        stabilization=stabilization,
    )


def _largest_moment(values: Any) -> float:
    """Take the largest absolute moment without mixing it with a field norm."""
    if isinstance(values, tuple):
        return max((_largest_moment(value) for value in values), default=0.0)
    return float(np.max(np.abs(values), initial=0))


def measure_variant(variant: str, solution: Any, *, order: int = 8) -> dict[str, float]:
    """Integrate physical errors and report separately named conservation diagnostics.

    RAD exposes only its scalar norm in this introductory consumer. For unit
    acoustic density the negative-gradient error equals the flux error. PGMHM
    uses the enriched pressure and its conservative face flux. Conservation
    moments are unnormalized absolute defects, never combined L2 residuals.
    """
    spec = VARIANTS[variant]
    exact = quadratic_pressure if spec.quadratic else affine_pressure
    flux = quadratic_flux if spec.quadratic else affine_flux
    if variant.startswith("tensor-rt"):
        errors = solution.errors(exact, flux, 0, order=order)
        metrics = {
            "scalar_l2": errors["pressure_l2"],
            "flux_l2": errors["flux_l2"],
            "divergence_l2": errors["divergence_l2"],
        }
    elif variant.startswith("hdiv-"):
        errors = solution.errors(exact, flux, order=order)
        metrics = {"scalar_l2": errors["pressure_l2"], "flux_l2": errors["flux_l2"]}
    elif variant == "helmholtz":
        metrics = {
            "scalar_l2": solution.l2_error(acoustic_pressure, order=order),
            "flux_l2": solution.gradient_l2_error(np.array([1.0, 2.0]) * (1 + 1j), order=order),
        }
    elif variant.startswith("rad"):
        metrics = {"scalar_l2": solution.l2_error(exact, order=order)}
    elif variant == "pgmhm":
        metrics = {
            "scalar_l2": solution.l2_error(exact, order=order, enriched=True),
            "flux_l2": solution.flux_l2_error(flux, order=order, enriched=True),
        }
    else:
        metrics = {
            "scalar_l2": solution.l2_error(exact, order=order),
            "flux_l2": solution.flux_l2_error(flux, order=order),
        }
    if hasattr(solution, "conservation_residuals"):
        key = "global_balance_max" if variant == "classical-rt1" else "macro_balance_max"
        metrics[key] = _largest_moment(solution.conservation_residuals())
    if hasattr(solution, "fine_equilibrium_residuals"):
        metrics["fine_pressure_moment_max"] = _largest_moment(solution.fine_equilibrium_residuals())
    elif hasattr(solution, "equilibrium_residuals"):
        metrics["fine_pressure_moment_max"] = _largest_moment(solution.equilibrium_residuals())
    if hasattr(solution, "normal_flux_residuals") and solution.skeleton is not None:
        metrics["normal_flux_moment_max"] = _largest_moment(solution.normal_flux_residuals())
    if hasattr(solution, "physical_residuals"):
        for column, name in enumerate(("constitutive", "divergence", "normal_flux")):
            metrics[f"{name}_relative_residual"] = _largest_moment(
                solution.physical_residuals[:, column]
            )
    if hasattr(solution, "local_equation_residuals"):
        metrics["local_equation_max"] = _largest_moment(solution.local_equation_residuals())
        metrics["pressure_trace_moment_max"] = _largest_moment(solution.trace_moment_residuals())
    hybrid = getattr(solution, "hybrid", None)
    metrics["algebraic_relative_residual"] = float(
        hybrid.residual if hybrid is not None else solution.residual
    )
    return {key: float(value) for key, value in metrics.items()}


def run_variant(variant: str, *, refinement: int | None = None) -> dict[str, Any]:
    """Return one self-contained tutorial record, using one native thread for the patch."""
    with threadpool_limits(1):
        solution = solve_variant(variant, refinement=refinement)
        metrics = measure_variant(variant, solution)
    spec = VARIANTS[variant]
    subdivisions = spec.refinement if refinement is None else refinement
    declared = asdict(spec)
    declared["refinement"] = subdivisions
    return {
        "variant": variant,
        "spaces_and_conventions": declared,
        "edge_subdivisions": subdivisions,
        "norm_quadrature_order": 8,
        "metrics": metrics,
        "scope": "analytical patch; no convergence-rate or paper-reproduction claim",
    }


def main(argv: list[str] | None = None) -> None:
    """Select a local/method variant and print its independently measured physical errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("all", *VARIANTS), default="primal")
    parser.add_argument("--refinement", type=int, help="override edge subdivisions (dyadic in 3D)")
    parser.add_argument("--output", type=Path, help="optional JSON destination")
    args = parser.parse_args(argv)
    names = tuple(VARIANTS) if args.variant == "all" else (args.variant,)
    records = [run_variant(name, refinement=args.refinement) for name in names]
    text = json.dumps({"records": records}, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
