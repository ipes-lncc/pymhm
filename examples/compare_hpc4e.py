"""Compare archived PyMHM stresses with the digitized published profile bands."""

from __future__ import annotations

import argparse
import json

import numpy as np

from examples.hpc4e_data import LENGTH_SCALE
from examples.hpc4e_fields import RectangularElasticityField
from pymhm.io.workspace import case_workspace, read_resource_text

ROOT = case_workspace()


def stress_interval(
    field: RectangularElasticityField, centres_m: np.ndarray, half_width_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Find exact piecewise-quadratic RT1 profile extrema on each horizontal interval.

    Every intersected fine cell contributes both one-sided endpoint limits and
    any interior stationary point. Interior interpolation points avoid ambiguous
    evaluation precisely on a shared fine edge.
    """
    if field.degree != 1 or field.enrichment != 0:
        raise ValueError("published-profile intervals require the declared RT1 space")
    edges = np.linspace(field.bounds[0], field.bounds[1], field.nx + 1) * LENGTH_SCALE
    positions = edges[:-1, None] + np.array([0.125, 0.5, 0.875]) * np.diff(edges)[:, None]
    points = np.column_stack((positions.ravel(), np.full(positions.size, 2250.25))) / LENGTH_SCALE
    samples = field.evaluate(points)[1][:, 0, 0].reshape(field.nx, 3) * 100
    a = (samples[:, 0] + samples[:, 2]) / 2 - samples[:, 1]
    b = (samples[:, 2] - samples[:, 0]) / 2
    c = samples[:, 1]
    lower, upper = [], []
    for centre in centres_m:
        lo, hi = max(edges[0], centre - half_width_m), min(edges[-1], centre + half_width_m)
        cells = np.flatnonzero((edges[:-1] < hi) & (edges[1:] > lo))
        extrema = []
        for cell in cells:
            bounds = (np.clip([lo, hi], edges[cell], edges[cell + 1]) - edges[cell]) / (
                edges[cell + 1] - edges[cell]
            )
            left, right = (bounds - 0.5) / 0.375
            coordinates = [left, right]
            if a[cell] != 0:
                stationary = -b[cell] / (2 * a[cell])
                if left < stationary < right:
                    coordinates.append(stationary)
            extrema.extend((a[cell] * r + b[cell]) * r + c[cell] for r in coordinates)
        lower.append(min(extrema))
        upper.append(max(extrema))
    return np.asarray(lower), np.asarray(upper)


def profile_comparison(segments: int) -> dict:
    """Measure nominal profile differences and overlap with calibrated raster intervals.

    Horizontal uncertainty is propagated through the numerical profile, including
    its separate fine-cell traces. The overlap count is a raster consistency
    diagnostic, not a field error or a replacement for a refined classical solve.
    """
    metadata = json.loads(
        read_resource_text(ROOT / "examples/results/hpc4e/published-profile.json")
    )
    curve = next(row for row in metadata["curves"] if row["segments"] == segments)
    field = RectangularElasticityField.load(ROOT / f"build/results/hpc4e/mhm-s{segments}.npz")
    x = np.array([row["x_m"] for row in curve["samples"]])
    observed = np.array([row["stress_mpa"] for row in curve["samples"]])
    lower = np.array([row["stress_lower_mpa"] for row in curve["samples"]])
    upper = np.array([row["stress_upper_mpa"] for row in curve["samples"]])
    half_width = metadata["calibration"]["horizontal_uncertainty_pixels"] * 10000 / 1170
    points = np.column_stack((x, np.full(len(x), 2250.25))) / LENGTH_SCALE
    nominal = field.evaluate(points)[1][:, 0, 0] * 100
    numerical_lower, numerical_upper = stress_interval(field, x, half_width)
    separation = np.maximum.reduce(
        (lower - numerical_upper, numerical_lower - upper, np.zeros(len(x)))
    )
    difference = nominal - observed
    return {
        "segments": segments,
        "samples": len(x),
        "nominal_rms_difference_mpa": float(np.sqrt(np.mean(difference**2))),
        "nominal_max_difference_mpa": float(np.max(np.abs(difference))),
        "interval_overlap_count": int(np.count_nonzero(separation == 0)),
        "interval_max_separation_mpa": float(np.max(separation)),
        "horizontal_half_width_m": half_width,
        "x_m": x.tolist(),
        "numerical_stress_mpa": nominal.tolist(),
        "numerical_lower_mpa": numerical_lower.tolist(),
        "numerical_upper_mpa": numerical_upper.tolist(),
    }


def main() -> None:
    """Write compact numerical provenance for every requested available profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    rows = [profile_comparison(value) for value in args.segments]
    target = ROOT / "examples/results/hpc4e/published-profile-comparison.json"
    target.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print({key: value for key, value in row.items() if not isinstance(value, list)})


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.compare_hpc4e").main()
