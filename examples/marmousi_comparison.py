"""Compare broken Qk MHM acoustics and classical Pk fields on a common mesh.

Pixel-aligned tensor local cells are split along SW--NE diagonals, respecting
both pressure spaces and every material interface. No broken field is averaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import fsum
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.marmousi_campaign import evaluate_fields
from examples.marmousi_data import load_marmousi_crop
from examples.marmousi_fields import (
    PixelCGField,
    acoustic_norm_contributions,
    acoustic_norm_result,
    load_reference,
    radial_exclusion_mask,
)
from examples.marmousi_records import checked_mhm, checked_reference
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.quadrilateral import qk_basis
from pymhm.fem.scalar.triangle import reference_basis
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.meshes.cartesian import CartesianMacroMesh


@dataclass(frozen=True)
class BrokenQField:
    """Retain row-major complex local Qk coefficients and their actual macro mesh."""

    pressure: np.ndarray
    mesh: CartesianMacroMesh
    refinement: int
    degree: int

    def __post_init__(self) -> None:
        """Check the complete local grid without imposing continuity across macros."""
        if (
            not isinstance(self.refinement, int)
            or self.refinement < 1
            or not isinstance(self.degree, int)
            or self.degree not in (1, 2, 3, 4)
            or self.pressure.shape
            != (len(self.mesh.cells), (self.refinement * self.degree + 1) ** 2)
            or not np.isfinite(self.pressure).all()
        ):
            raise ValueError("complete finite Q1--Q4 local fields are required")

    def coefficients(self, cells: np.ndarray) -> np.ndarray:
        """Gather a physical fine rectangle from its unique incident macro interior."""
        owners, local = cells // self.refinement, cells % self.refinement
        macro = owners[:, 0] + self.mesh.nx * owners[:, 1]
        width = self.refinement * self.degree + 1
        first = self.degree * (local[:, 0] + width * local[:, 1])
        offsets = np.array(
            [j * width + i for j in range(self.degree + 1) for i in range(self.degree + 1)]
        )
        return self.pressure[macro[:, None], first[:, None] + offsets]

    @classmethod
    def load(cls, record: Path) -> BrokenQField:
        """Verify the executed archive digest and its complete Cartesian macro geometry."""
        metadata = json.loads(record.read_text())
        path = record.parent / metadata["archive"]
        if file_digest(path) != metadata["archive_sha256"]:
            raise ValueError("MHM archive digest does not match its acquisition record")
        bounds = tuple(metadata.get("bounds", (0, 10240, 0, 2560)))
        mesh = CartesianMacroMesh(*metadata["macro_shape"], bounds)
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["macro_points"], mesh.points) or not np.array_equal(
                archive["macro_cells"], mesh.cells
            ):
                raise ValueError("archived macro geometry differs from the declared partition")
            pressure = archive["pressure"]
        return cls(pressure, mesh, metadata["local_refinement"], metadata["local_degree"])


def mhm_difference(
    candidate: BrokenQField,
    reference: PixelCGField,
    density: np.ndarray,
    bulk_modulus: np.ndarray,
    *,
    omega: float,
    order: int = 8,
    batch_size: int = 1024,
    workers: int = 1,
    gradient_cutout: tuple[float, float, float, float] | None = None,
    gradient_exclusion: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Integrate physical complex differences on an exact nested common triangulation.

    The isotropic local-cell subdivision of each reference pixel must be an
    integer. An optional cutout follows the original pixel boundaries. Ordered
    batch reductions are identical for serial and threaded execution; fields
    remain shared and memory is bounded by the number of concurrent batches.
    A circular ``gradient_exclusion`` instead uses the shared physical point
    mask on both restricted numerator and denominator; quadrature controls
    include its curved boundary. Full-domain pressure remains a separate norm.
    """
    spacing = candidate.mesh.spacing / candidate.refinement
    if gradient_exclusion is not None:
        radial_exclusion_mask(np.zeros((1, 2)), gradient_exclusion)
        if gradient_cutout is not None:
            raise ValueError("declare one derivative exclusion geometry")
    ratio = reference.spacing / spacing
    if (
        candidate.mesh.bounds != reference.bounds
        or not np.allclose(ratio, round(float(ratio[0])), rtol=0, atol=1e-12)
        or ratio[0] < 1
    ):
        raise ValueError("MHM fine cells must isotropically subdivide each reference pixel")
    division = round(float(ratio[0]))
    density, bulk_modulus = np.asarray(density), np.asarray(bulk_modulus)
    if (
        density.shape != reference.counts
        or bulk_modulus.shape != reference.counts
        or not np.isfinite(density).all()
        or not np.isfinite(bulk_modulus).all()
        or np.min(density) <= 0
        or np.min(bulk_modulus) <= 0
        or not np.isfinite(omega)
        or omega <= 0
        or not isinstance(workers, int)
        or workers < 1
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("positive finite acoustic coefficients and batching are required")
    cutout = None
    if gradient_cutout is not None:
        cutout = np.asarray(gradient_cutout, dtype=float).reshape(2, 2)
        scaled = (cutout - np.asarray(reference.bounds)[[0, 2], None]) / reference.spacing[:, None]
        if (
            not np.isfinite(scaled).all()
            or np.any(np.diff(scaled, axis=1) <= 0)
            or np.any(scaled < 0)
            or np.any(scaled > np.asarray(reference.counts)[:, None])
            or np.max(abs(scaled - np.rint(scaled))) > 1e-10
        ):
            raise ValueError("gradient cutout must be a pixel-aligned interior rectangle")
        cutout = np.rint(scaled).astype(int)
    bary, weights = triangle_quadrature(order)
    hx, hy = reference.spacing
    gradients = np.array(
        [
            [[-1 / hx, 0], [1 / hx, -1 / hy], [0, 1 / hy]],
            [[0, -1 / hy], [1 / hx, 0], [-1 / hx, 1 / hy]],
        ]
    )
    tables = []
    for ix in range(division):
        for iy in range(division):
            for half, corners in enumerate(
                (np.array([[0, 0], [1, 0], [1, 1]]), np.array([[0, 0], [1, 1], [0, 1]]))
            ):
                points = bary @ corners
                q_basis, q_gradient = qk_basis(candidate.degree, points)
                px, py = ((points + [ix, iy]) / division).T
                reference_half = int(iy > ix) if ix != iy else half
                p_bary = (
                    np.column_stack((1 - px, px - py, py))
                    if reference_half == 0
                    else np.column_stack((1 - py, px, py - px))
                )
                p_basis, p_derivative, _ = reference_basis(reference.degree, p_bary)
                tables.append(
                    (
                        np.array([ix, iy]),
                        reference_half,
                        q_basis,
                        q_gradient / spacing,
                        p_basis,
                        np.einsum("qib,bd->qid", p_derivative, gradients[reference_half]),
                        points,
                    )
                )
    count = int(np.prod(reference.counts))

    def integrate(start: int) -> list[list[float]]:
        """Evaluate one bounded physical batch and retain the original reduction order."""
        index = np.arange(start, min(start + batch_size, count))
        cells = np.column_stack((index // reference.counts[1], index % reference.counts[1]))
        rho, kappa = density[cells[:, 0], cells[:, 1]], bulk_modulus[cells[:, 0], cells[:, 1]]
        keep = np.ones(len(cells), dtype=bool)
        if cutout is not None:
            keep = ~np.all((cells >= cutout[:, 0]) & (cells < cutout[:, 1]), axis=1)
        p_coefficients = [reference.coefficients(cells, side) for side in (0, 1)]
        result = []
        for offset, side, q_basis, q_gradient, p_basis, p_gradient, points in tables:
            mask = keep
            if gradient_exclusion is not None:
                physical = (division * cells[:, None, :] + offset + points[None, :, :]) * spacing
                physical += np.asarray(reference.bounds)[[0, 2]]
                mask = radial_exclusion_mask(physical, gradient_exclusion)
            q_coefficients = candidate.coefficients(division * cells + offset)
            values = np.einsum("qi,ti->tq", q_basis, q_coefficients)
            derivatives = np.einsum("qid,ti->tqd", q_gradient, q_coefficients)
            p_values = np.einsum("qi,ti->tq", p_basis, p_coefficients[side])
            p_derivatives = np.einsum("qid,ti->tqd", p_gradient, p_coefficients[side])
            integrands = acoustic_norm_contributions(
                values, p_values, derivatives, p_derivatives, rho, kappa, omega, mask
            )
            result.append(
                [float(np.sum(value @ weights) * np.prod(spacing) / 2) for value in integrands]
            )
            measure = (
                mask
                if mask.ndim == 2
                else np.broadcast_to(mask[:, None], (len(cells), len(weights)))
            )
            result[-1].append(float(np.sum(measure @ weights) * np.prod(spacing) / 2))
        return result

    starts = range(0, count, batch_size)
    with threadpool_limits(1):
        if workers == 1:
            batches = list(map(integrate, starts))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                batches = list(executor.map(integrate, starts))
    totals = np.array([fsum(row[i] for batch in batches for row in batch) for i in range(10)])
    return acoustic_norm_result(
        totals,
        {
            "quadrature_order": order,
            "common_triangles": 2 * count * division**2,
            "domain_area": float(np.prod(reference.spacing) * count),
            "gradient_cutout": gradient_cutout,
            "gradient_exclusion": gradient_exclusion,
            "gradient_domain_area": fsum(row[10] for batch in batches for row in batch)
            if gradient_exclusion is not None
            else float(
                np.prod(reference.spacing)
                * (count - (np.prod(cutout[:, 1] - cutout[:, 0]) if cutout is not None else 0))
            ),
            "derivative_domain_integration": "Physical quadrature-point disk mask"
            if gradient_exclusion
            else "Pixel-aligned rectangle or full domain",
        },
    )


def main() -> None:
    """Archive MHM/reference physical norms and all incident sampling conventions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    material = load_marmousi_crop(args.data)
    metadata = [checked_mhm(args.candidate), checked_reference(args.reference)]
    for record in metadata:
        if (
            record["material"] != material.provenance
            or record["omega"] != 40 * np.pi
            or record["point_source"] != [5000, 50, 1.0]
        ):
            raise ValueError("comparison requires identical material, frequency and point source")
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        *(
            Path(__file__).with_name(f"marmousi_{name}.py")
            for name in ("fields", "campaign", "data")
        ),
        root / "examples/campaign_provenance.py",
        root / "examples/helmholtz_trace_family.py",
        root / "examples/marmousi_records.py",
        *sorted((root / "src/pymhm").rglob("*.py")),
    ]
    hashes = current_source_manifest(
        {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    )
    candidate, reference = BrokenQField.load(args.candidate), load_reference(args.reference)
    exclusion = (5000.0, 50.0, 50.0)
    measured = {
        str(order): mhm_difference(
            candidate,
            reference,
            material.density.values,
            material.bulk_modulus.values,
            omega=40 * np.pi,
            order=order,
            workers=args.workers,
            gradient_exclusion=exclusion,
        )
        for order in (8, 10)
    }
    with np.load(args.candidate.parent / metadata[0]["archive"], allow_pickle=False) as archive:
        points = archive["sample_points"]
        stored = archive["sample_pressure"]
        sides = archive["incident_sides"]
    exact_samples = reference.sample(points)
    samples = []
    for values, side in zip(stored, sides, strict=True):
        replay = evaluate_fields(
            candidate.pressure,
            candidate.mesh,
            candidate.refinement,
            candidate.degree,
            points,
            side=tuple(side),
        )
        if not np.array_equal(replay, values):
            raise ValueError("MHM sampling replay differs from its acquisition archive")
        denominator = float(np.linalg.norm(exact_samples))
        difference = float(np.linalg.norm(values - exact_samples))
        samples.append(
            {
                "incident_side": side.tolist(),
                "difference": difference,
                "reference_norm": denominator,
                "relative_difference": difference / denominator if denominator else None,
            }
        )
    if hashes != current_source_manifest(
        {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    ):
        raise RuntimeError("comparison sources changed during integration")
    result = {
        "candidate": args.candidate.name,
        "candidate_record_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "reference": args.reference.name,
        "reference_record_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "norms": measured,
        "sampled_pressure": samples,
        "denominator": "Physical or sampled norm of the named classical reference field",
        "source_sha256": hashes,
        "source_changed_during_run": False,
        "material": material.provenance,
    }
    output = args.output or args.candidate.with_name(
        f"{args.candidate.stem}-vs-{args.reference.stem}.json"
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
