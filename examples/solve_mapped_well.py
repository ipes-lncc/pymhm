"""Acquire a trilinear-hexahedral RT1 MHM Dupuit--Thiem reservoir study in SI units."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm.mapped_rt import HexMesh, cube_quadrature, mapped_rt_dofs, solve_darcy_mapped_rt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/mapped-well"


@dataclass(frozen=True)
class WellData:
    """Table 1 physical data of Duran et al. (2019), with the exact radial Darcy field."""

    inner_radius: float = 0.2
    outer_radius: float = 50.0
    height: float = 10.0
    permeability: float = 1e-13
    viscosity: float = 0.001
    outer_pressure: float = 25e6
    rate: float = 0.01

    def pressure(self, points: np.ndarray) -> np.ndarray:
        """Return the Dupuit--Thiem pressure at physical Cartesian points."""
        radius = np.linalg.norm(points[:, :2], axis=1)
        return self.outer_pressure + self.rate * self.viscosity / (
            2 * np.pi * self.permeability * self.height
        ) * np.log(radius / self.outer_radius)

    def flux(self, points: np.ndarray) -> np.ndarray:
        """Return the complete physical Darcy flux, including its zero vertical component."""
        value = np.zeros_like(points)
        value[:, :2] = (
            -self.rate
            / (2 * np.pi * self.height)
            * points[:, :2]
            / np.sum(points[:, :2] ** 2, axis=1)[:, None]
        )
        return value


def main() -> None:
    """Keep a fixed fine grid while refining the macro partition to its classical limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-factor", type=int, default=8)
    parser.add_argument("--macro-factor", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quadrature", type=int, default=6)
    parser.add_argument("--global-rtol", type=float, default=1e-10)
    parser.add_argument(
        "--global-refinement-precision", choices=["double", "extended"], default="double"
    )
    args = parser.parse_args()
    if args.fine_factor % args.macro_factor:
        parser.error("macro factor must divide the fixed fine factor")
    data = WellData()
    # The printed radial-spacing expression does not determine consistent radii.
    # This original graded geometry is declared independently of historical meshes.
    radii = np.geomspace(data.inner_radius, data.outer_radius, 5)
    base = HexMesh.annular_prism(radii, data.height, 8)
    mesh = base.refined(args.macro_factor)
    r = args.fine_factor // args.macro_factor
    zero_flux = {
        int(face): 0.0
        for face in mesh.boundary_faces
        if np.ptp(mesh.points[mesh.faces[face], 2]) < 1e-12
    }
    sources = [
        Path(__file__),
        ROOT / "src/pymhm/mapped_rt.py",
        ROOT / "src/pymhm/solvers.py",
    ]
    hybrid_source = ROOT / "src/pymhm/hybrid.py"
    hybrid_start = hashlib.sha256(hybrid_source.read_bytes()).hexdigest()
    snapshot = ROOT / "build/source-snapshots/mapped-well"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / f"{hybrid_start}-hybrid.py").write_bytes(hybrid_source.read_bytes())
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with threadpool_limits(1):
        result = solve_darcy_mapped_rt(
            mesh,
            degree=1,
            trace_degree=1,
            local_refinement=r,
            permeability=data.permeability / data.viscosity,
            dirichlet=data.pressure,
            neumann=zero_flux,
            quadrature_order=args.quadrature,
            backend="process" if args.workers > 1 else "serial",
            workers=args.workers,
            global_rtol=args.global_rtol,
            global_refinement_precision=args.global_refinement_precision,
        )
        norms = result.errors(data.pressure, data.flux, order=9)
        checked = result.errors(data.pressure, data.flux, order=11)
        points, weights = cube_quadrature(11)
        reference_norms = np.zeros(3)
        for local in result.local_meshes:
            physical, _, det = local.geometry(points)
            p = data.pressure(physical.reshape(-1, 3)).reshape(det.shape)
            q = data.flux(physical.reshape(-1, 3)).reshape(*det.shape, 3)
            reference_norms += [
                np.sum(det * weights * p * p),
                np.sum(det * weights * (p - data.outer_pressure) ** 2),
                np.sum(det * weights * np.sum(q * q, axis=2)),
            ]
        boundaries = dict(inner=0.0, outer=0.0, top_bottom=0.0)
        for face in mesh.boundary_faces:
            nodes = mesh.points[mesh.faces[face]]
            radii_face = np.linalg.norm(nodes[:, :2], axis=1)
            label = (
                "top_bottom"
                if int(face) in zero_flux
                else "inner"
                if np.max(radii_face) < 1.0
                else "outer"
            )
            boundaries[label] += float(result.hybrid.trace[4 * face])
        equilibrium = max(float(np.abs(m).max()) for m in result.equilibrium_residuals())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = f"fine{args.fine_factor}-macro{args.macro_factor}-q{args.quadrature}"
    if args.global_rtol != 1e-10 or args.global_refinement_precision != "double":
        name += f"-rtol{args.global_rtol:g}-{args.global_refinement_precision}"
    archive = OUTPUT / (name + ".npz")
    np.savez_compressed(
        archive,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        local_points=np.stack([m.points for m in result.local_meshes]),
        local_cells=np.stack([m.cells for m in result.local_meshes]),
        pressure=np.stack(result.pressure),
        flux=np.stack(result.flux),
        trace=result.hybrid.trace,
        algebraic_residual=np.array(result.hybrid.residual),
        physical_residuals=result.physical_residuals,
    )
    report = dict(
        global_rtol=args.global_rtol,
        global_refinement_precision=args.global_refinement_precision,
        case="L05 Problem 4 physical data on an original explicitly graded polygonal reservoir",
        geometry=(
            "Trilinear eight-sector polygonal annulus, four logarithmically graded "
            "radial bands, extruded in z"
        ),
        radii=radii.tolist(),
        height=data.height,
        physical_parameters=data.__dict__,
        units="m, Pa, s; physical Darcy flux m/s",
        rate_convention="Positive Q denotes extraction: inner outward rate +Q and exterior -Q",
        historical_geometry=(
            "Printed graded-distance expression does not determine radii consistent "
            "with the stated domain; no identical historical-mesh claim"
        ),
        macro_factor=args.macro_factor,
        fine_factor=args.fine_factor,
        local_refinement=r,
        macro_cells=len(mesh.cells),
        fine_cells=sum(len(m.cells) for m in result.local_meshes),
        local_space="Contravariant Piola RT1 / scalar pullback Q1",
        trace_space="Q1 reference-face flux density",
        trace_dofs=result.skeleton.size,
        pressure_coarse_dofs=len(mesh.cells),
        fine_flux_unknowns_sum=sum(
            int(mapped_rt_dofs(m, 1).max()) + 1 for m in result.local_meshes
        ),
        assembly_quadrature=args.quadrature,
        error_quadrature=11,
        quadrature9=norms,
        errors=checked,
        pressure_relative=checked["pressure_l2"] / np.sqrt(reference_norms[0]),
        pressure_increment_relative=checked["pressure_l2"] / np.sqrt(reference_norms[1]),
        flux_relative=checked["flux_l2"] / np.sqrt(reference_norms[2]),
        boundary_rates=boundaries,
        exact_boundary_rates=dict(inner=data.rate, outer=-data.rate, top_bottom=0.0),
        divergence_moment_linf=equilibrium,
        algebraic_residual=result.hybrid.residual,
        physical_block_backward_residual_max=result.physical_residuals.max(axis=0).tolist(),
        local_scaling=(
            "Exact symmetric flux/pressure change of units; physical fields restored "
            "before all errors and balances"
        ),
        archive=archive.name,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_hashes=hashes,
        hybrid_source_hash_start=hybrid_start,
        hybrid_source_hash_end=hashlib.sha256(hybrid_source.read_bytes()).hexdigest(),
    )
    (OUTPUT / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    if hashes != {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }:
        raise RuntimeError("Acquisition source changed during the numerical solve")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
