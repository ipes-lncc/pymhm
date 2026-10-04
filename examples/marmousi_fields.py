"""Replay native acoustic CG fields and compare polynomial reference levels.

The geometry is the common pixel-conforming SW--NE triangulation. Physical
integrals are separate from the unweighted vertex sampling used in Table 6.1.
No continuity is imposed on fields from the MHM acquisition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from math import fsum
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.marmousi_data import load_marmousi_crop
from examples.marmousi_records import checked_reference
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import multiindices, reference_basis
from pymhm.io.provenance import current_source_manifest, file_digest


@dataclass(frozen=True)
class PixelCGField:
    """Store continuous complex Pk coefficients on an equispaced rectangular grid.

    The first array axis is x and the second y. Each original pixel is divided
    into SW--SE--NE and SW--NE--NW triangles; the intermediate nodal grid does
    not subdivide the underlying finite elements.
    """

    nodes: np.ndarray
    degree: int
    bounds: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        """Reject incomplete, nonfinite or geometrically incompatible nodal data."""
        if (
            self.degree not in (1, 2, 3, 4)
            or self.nodes.ndim != 2
            or min(self.nodes.shape) < 2
            or any((size - 1) % self.degree for size in self.nodes.shape)
            or not np.isfinite(self.nodes).all()
            or len(self.bounds) != 4
            or not np.isfinite(self.bounds).all()
            or self.bounds[1] <= self.bounds[0]
            or self.bounds[3] <= self.bounds[2]
        ):
            raise ValueError("a complete finite P1--P4 nodal rectangle is required")

    @property
    def counts(self) -> tuple[int, int]:
        """Return the underlying material-pixel counts, independent of degree."""
        return tuple((size - 1) // self.degree for size in self.nodes.shape)

    @property
    def spacing(self) -> np.ndarray:
        """Return physical pixel widths in x and y."""
        return np.diff(np.asarray(self.bounds).reshape(2, 2), axis=1)[:, 0] / self.counts

    def coefficients(self, cells: np.ndarray, half: int) -> np.ndarray:
        """Gather Pk nodal coefficients in declared barycentric lattice order."""
        corners = np.array([[0, 0], [1, 0], [1, 1]])
        if half == 1:
            corners = np.array([[0, 0], [1, 1], [0, 1]])
        elif half != 0:
            raise ValueError("triangle half must be zero or one")
        offsets = multiindices(self.degree) @ corners
        indices = self.degree * cells[:, None, :] + offsets
        return self.nodes[indices[..., 0], indices[..., 1]]

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the continuous pressure at arbitrary physical domain points."""
        points = np.asarray(points, dtype=float)
        lower = np.asarray(self.bounds)[[0, 2]]
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("finite two-dimensional points are required")
        scaled = (points - lower) / self.spacing
        if np.any(scaled < 0) or np.any(scaled > self.counts):
            raise ValueError("sampling point lies outside the reference rectangle")
        cells = np.minimum(np.floor(scaled).astype(int), np.asarray(self.counts) - 1)
        xy = scaled - cells
        half = (xy[:, 1] > xy[:, 0]).astype(int)
        result = np.empty(len(points), dtype=self.nodes.dtype)
        for side in (0, 1):
            mask = half == side
            x, y = xy[mask].T
            bary = np.column_stack((1 - x, x - y, y))
            if side:
                bary = np.column_stack((1 - y, x, y - x))
            basis, _, _ = reference_basis(self.degree, bary)
            result[mask] = np.einsum("qi,qi->q", basis, self.coefficients(cells[mask], side))
        return result


def load_reference(record: Path) -> PixelCGField:
    """Verify MPI archive digests and preserve their executed nodal coefficient dtype.

    A first bounded archive pass determines the common dtype before allocation;
    a second copies each unique owned node. Digests stream without reading an
    entire compressed field archive into an extra byte buffer. Replay never
    converts a wider persisted coefficient vector to double precision.
    """
    metadata = json.loads(record.read_text())
    degree = int(metadata["degree"])
    counts = np.asarray(metadata["geometry"], dtype=int)
    bounds = tuple(metadata.get("bounds", (0.0, 10240.0, 0.0, 2560.0)))
    lower = np.asarray(bounds)[[0, 2]]
    spacing = np.diff(np.asarray(bounds).reshape(2, 2), axis=1)[:, 0] / counts
    dtype = np.dtype(complex)
    for entry in metadata["archives"]:
        path = record.parent / entry["archive"]
        if file_digest(path) != entry["sha256"]:
            raise ValueError("reference archive digest does not match its acquisition record")
        with np.load(path, allow_pickle=False) as archive:
            dtype = np.result_type(dtype, archive["pressure"].dtype)
    nodes = np.empty(tuple(counts * degree + 1), dtype=dtype)
    seen = np.zeros(nodes.shape, dtype=bool)
    for entry in metadata["archives"]:
        path = record.parent / entry["archive"]
        if file_digest(path) != entry["sha256"]:
            raise ValueError("reference archive digest does not match its acquisition record")
        with np.load(path, allow_pickle=False) as archive:
            coordinates = archive["coordinates"][:, :2]
            pressure = archive["pressure"]
        scaled = (coordinates - lower) / spacing * degree
        indices = np.rint(scaled).astype(int)
        tolerance = 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(scaled))
        if (
            np.any(np.abs(scaled - indices) > tolerance)
            or np.any(indices < 0)
            or np.any(indices > counts * degree)
            or pressure.shape != (len(indices),)
        ):
            raise ValueError("native nodal coordinates do not match the declared grid")
        key = np.ravel_multi_index(indices.T, nodes.shape)
        if len(np.unique(key)) != len(key) or seen.ravel()[key].any():
            raise ValueError("owned nodal coordinates must appear exactly once")
        nodes.ravel()[key] = pressure
        seen.ravel()[key] = True
    if not seen.all():
        raise ValueError("native owned nodal archives do not cover the global field")
    return PixelCGField(nodes, degree, bounds)


def acoustic_norm_contributions(
    pressure: np.ndarray,
    reference_pressure: np.ndarray,
    gradient: np.ndarray,
    reference_gradient: np.ndarray,
    density: np.ndarray,
    bulk_modulus: np.ndarray,
    omega: float,
    keep: np.ndarray,
) -> list[np.ndarray]:
    """Return Hermitian-square physical integrands with one shared cutout convention.

    Pressure retains the entire domain. Gradient, acoustic flux and both
    positive graph-norm contributions use the same derivative-domain mask.
    Inputs have cell/point axes; gradients add the physical component axis.
    """
    keep = np.asarray(keep)
    if keep.dtype.kind != "b" or keep.shape not in (pressure.shape, pressure.shape[:1]):
        raise ValueError(
            "the common derivative mask must select cells or physical quadrature points"
        )
    mask = keep if keep.ndim == 2 else keep[:, None]
    p_difference = np.abs(pressure - reference_pressure) ** 2
    p_reference = np.abs(reference_pressure) ** 2
    g_difference = np.sum(np.abs(gradient - reference_gradient) ** 2, axis=2)
    g_reference = np.sum(np.abs(reference_gradient) ** 2, axis=2)
    g_difference *= mask
    g_reference *= mask
    return [
        p_difference,
        p_reference,
        g_difference,
        g_reference,
        g_difference / density[:, None],
        g_reference / density[:, None],
        omega**2 * p_difference * mask / bulk_modulus[:, None],
        omega**2 * p_reference * mask / bulk_modulus[:, None],
        g_difference / density[:, None] ** 2,
        g_reference / density[:, None] ** 2,
    ]


def radial_exclusion_mask(points: np.ndarray, exclusion: tuple[float, float, float]) -> np.ndarray:
    """Keep physical points outside the declared open disk (center x, center y, radius).

    Boundary points are retained. The same mask must multiply numerator and
    denominator integrands of each restricted measure. Full-domain pressure
    L2 and the published complete sampling grid have separate conventions.
    A pointwise quadrature mask requires a stated boundary integration control.
    """
    points, disk = np.asarray(points), np.asarray(exclusion)
    if (
        points.shape[-1:] != (2,)
        or np.iscomplexobj(points)
        or np.iscomplexobj(disk)
        or not np.isfinite(points).all()
        or disk.shape != (3,)
        or not np.isfinite(disk).all()
        or disk[2] <= 0
    ):
        raise ValueError("radial exclusion requires finite physical points and positive radius")
    return np.sum((points - disk[:2]) ** 2, axis=-1) >= disk[2] ** 2


def acoustic_norm_result(totals: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert integrated squares to norms using the stated reference as denominator."""
    result = dict(metadata)
    for name, numerator, denominator in (
        ("pressure", totals[0], totals[1]),
        ("gradient", totals[2], totals[3]),
        ("weighted_gradient", totals[4], totals[5]),
        ("graph", totals[4] + totals[6], totals[5] + totals[7]),
        ("flux", totals[8], totals[9]),
    ):
        result[f"{name}_difference"] = float(np.sqrt(numerator))
        result[f"reference_{name}_norm"] = float(np.sqrt(denominator))
        result[f"{name}_relative_difference"] = (
            float(np.sqrt(numerator / denominator)) if denominator > 0 else None
        )
    return result


def reference_difference(
    coarse: PixelCGField,
    fine: PixelCGField,
    density: np.ndarray,
    bulk_modulus: np.ndarray,
    *,
    omega: float,
    order: int = 8,
    batch_size: int = 2048,
    gradient_cutout: tuple[float, float, float, float] | None = None,
    gradient_exclusion: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Integrate reference increments on their shared original pixel triangulation.

    Relative quantities use the finer polynomial level's physical norm. The
    positive graph norm is the sum of rho-inverse gradient and omega-squared
    kappa-inverse pressure contributions; it is not the indefinite Helmholtz
    bilinear form. The discrete gradient norms do not assert finite H1
    regularity of the continuum solution with a point source. If supplied,
    ``gradient_cutout`` excludes a fixed pixel-aligned rectangle from gradient
    and graph norms only. Pressure L2 differences retain the complete domain.
    Alternatively ``gradient_exclusion`` declares a circular center/radius in
    physical units. Its boundary is integrated with a pointwise quadrature
    mask; separate quadrature orders control that geometric integration error.
    """
    if coarse.counts != fine.counts or coarse.bounds != fine.bounds:
        raise ValueError("reference levels must share the same material-pixel triangulation")
    if gradient_exclusion is not None:
        radial_exclusion_mask(np.zeros((1, 2)), gradient_exclusion)
        if gradient_cutout is not None:
            raise ValueError("declare one derivative exclusion geometry")
    density, bulk_modulus = np.asarray(density), np.asarray(bulk_modulus)
    if (
        density.shape != fine.counts
        or bulk_modulus.shape != fine.counts
        or not np.isfinite(density).all()
        or not np.isfinite(bulk_modulus).all()
        or np.min(density) <= 0
        or np.min(bulk_modulus) <= 0
        or not np.isfinite(omega)
        or omega <= 0
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("positive finite cellwise acoustic coefficients and batching required")
    bary, weights = triangle_quadrature(order)
    tables = [reference_basis(field.degree, bary)[:2] for field in (coarse, fine)]
    hx, hy = fine.spacing
    cutout = None
    if gradient_cutout is not None:
        cutout = np.asarray(gradient_cutout, dtype=float).reshape(2, 2)
        lower = np.asarray(fine.bounds)[[0, 2]]
        scaled = (cutout - lower[:, None]) / fine.spacing[:, None]
        if (
            not np.isfinite(cutout).all()
            or np.any(np.diff(cutout, axis=1) <= 0)
            or np.any(scaled < 0)
            or np.any(scaled > np.asarray(fine.counts)[:, None])
            or np.max(abs(scaled - np.rint(scaled))) > 1e-10
        ):
            raise ValueError("the fixed gradient cutout must be a pixel-aligned interior rectangle")
        cutout = np.rint(scaled).astype(int)
    gradients = np.array(
        [
            [[-1 / hx, 0], [1 / hx, -1 / hy], [0, 1 / hy]],
            [[0, -1 / hy], [1 / hx, 0], [-1 / hx, 1 / hy]],
        ]
    )
    sums: list[list[float]] = [[] for _ in range(10)]
    areas = []
    for start in range(0, np.prod(fine.counts), batch_size):
        index = np.arange(start, min(start + batch_size, np.prod(fine.counts)))
        cells = np.column_stack((index // fine.counts[1], index % fine.counts[1]))
        rho = density[cells[:, 0], cells[:, 1]]
        kappa = bulk_modulus[cells[:, 0], cells[:, 1]]
        keep = np.ones(len(cells), dtype=bool)
        if cutout is not None:
            keep = ~np.all((cells >= cutout[:, 0]) & (cells < cutout[:, 1]), axis=1)
        for half in (0, 1):
            mask = keep
            if gradient_exclusion is not None:
                corners = np.array(
                    [[0, 0], [1, 0], [1, 1]] if half == 0 else [[0, 0], [1, 1], [0, 1]]
                )
                points = (cells[:, None, :] + (bary @ corners)[None, :, :]) * fine.spacing
                points += np.asarray(fine.bounds)[[0, 2]]
                mask = radial_exclusion_mask(points, gradient_exclusion)
            measure = (
                mask
                if mask.ndim == 2
                else np.broadcast_to(mask[:, None], (len(cells), len(weights)))
            )
            areas.append(float(np.sum(measure @ weights) * hx * hy / 2))
            values, derivatives = [], []
            for field, (basis, derivative) in zip((coarse, fine), tables, strict=True):
                coefficients = field.coefficients(cells, half)
                values.append(np.einsum("qi,ti->tq", basis, coefficients))
                physical = np.einsum("qib,bd->qid", derivative, gradients[half])
                derivatives.append(np.einsum("qid,ti->tqd", physical, coefficients))
            integrands = acoustic_norm_contributions(
                values[0], values[1], derivatives[0], derivatives[1], rho, kappa, omega, mask
            )
            for total, integrand in zip(sums, integrands, strict=True):
                total.append(float(np.sum(integrand @ weights) * hx * hy / 2))
    totals = np.array([fsum(parts) for parts in sums])
    result: dict[str, Any] = {
        "quadrature_order": order,
        "domain_area": hx * hy * np.prod(fine.counts),
        "gradient_cutout": gradient_cutout,
        "gradient_exclusion": gradient_exclusion,
        "gradient_domain_area": fsum(areas)
        if gradient_exclusion is not None
        else hx
        * hy
        * (
            np.prod(fine.counts)
            - (np.prod(cutout[:, 1] - cutout[:, 0]) if cutout is not None else 0)
        ),
        "derivative_domain_integration": "Physical quadrature-point disk mask"
        if gradient_exclusion
        else "Pixel-aligned rectangle or full domain",
    }
    return acoustic_norm_result(totals, result)


def main() -> None:
    """Archive native-field replay, reference increments and the paper's sample grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("examples/results/marmousi"))
    args = parser.parse_args()
    material = load_marmousi_crop(args.data)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        Path(__file__).with_name("marmousi_data.py"),
        Path(__file__).with_name("campaign_provenance.py"),
        Path(__file__).with_name("marmousi_records.py"),
        *sorted((root / "src/pymhm").rglob("*.py")),
    ]
    hashes = current_source_manifest(
        {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    )
    x, y = np.meshgrid(np.linspace(0, 10240, 513), np.linspace(0, 2560, 129), indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel()))
    rows = []
    exclusion = (5000.0, 50.0, 50.0)
    previous, previous_samples = None, None
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        for path in args.records:
            metadata = checked_reference(path)
            if metadata["material"] != material.provenance:
                raise ValueError(
                    "all reference records must use the declared primary material crop"
                )
            field = load_reference(path)
            omega = float(metadata["omega"])
            if (
                omega != 40 * np.pi
                or metadata["point_source"] != [5000, 50, 1.0]
                or (previous is not None and previous.degree >= field.degree)
            ):
                raise ValueError(
                    "reference levels require increasing degree and identical wave data"
                )
            norms = reference_difference(
                field,
                field,
                material.density.values,
                material.bulk_modulus.values,
                omega=omega,
                order=8,
                gradient_exclusion=exclusion,
            )
            samples = field.sample(points)
            archive = args.output / f"classical-p{field.degree}-samples.npz"
            np.savez_compressed(archive, points=points, pressure=samples)
            row = {
                "reference_record": path.name,
                "reference_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "degree": field.degree,
                "complex_dofs": int(field.nodes.size),
                "physical_norms": norms,
                "native_pressure_norm_relative_difference": abs(
                    norms["reference_pressure_norm"] / metadata["pressure_l2"] - 1
                ),
                "sample_archive": archive.name,
                "sample_archive_sha256": file_digest(archive),
            }
            if row["native_pressure_norm_relative_difference"] > 1e-11:
                raise ValueError(
                    "field replay disagrees with the independently assembled native norm"
                )
            if previous is not None:
                row["increment"] = reference_difference(
                    previous,
                    field,
                    material.density.values,
                    material.bulk_modulus.values,
                    omega=omega,
                    order=6,
                    gradient_exclusion=exclusion,
                )
                row["increment_quadrature_check"] = reference_difference(
                    previous,
                    field,
                    material.density.values,
                    material.bulk_modulus.values,
                    omega=omega,
                    order=8,
                    gradient_exclusion=exclusion,
                )
                row["sample_relative_increment"] = float(
                    np.linalg.norm(samples - previous_samples) / np.linalg.norm(samples)
                )
            rows.append(row)
            previous, previous_samples = field, samples
            result = {
                "references": rows,
                "material": material.provenance,
                "sampling": "513 by 129 uniform points; unweighted complex Euclidean norm",
                "gradient_scope": (
                    "A declared open disk of radius 50 m excludes the source from gradient "
                    "and graph norms; "
                    "pressure L2 and sampled norms retain the complete domain"
                ),
                "gradient_exclusion_m": exclusion,
                "graph_norm": "Integral rho^-1 |grad p|^2 + omega^2 kappa^-1 |p|^2",
                "source_sha256": hashes,
                "source_changed_during_run": hashes
                != current_source_manifest(
                    {
                        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sources
                    }
                ),
            }
            if result["source_changed_during_run"]:
                raise RuntimeError("reference postprocessing sources changed during evaluation")
            (args.output / "classical-convergence.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
