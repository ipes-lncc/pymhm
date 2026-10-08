"""Physical field and complementary-energy comparisons for HPC4E elasticity.

The reference and comparison must use the same stress/displacement/rotation
spaces for the Galerkin orthogonality interpretation. The polarization identity
itself is valid for arbitrary archived physical fields.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pymhm.fem.scalar.quadrilateral import quadrilateral_quadrature

if __package__:
    from .hpc4e_data import DATA_DIRECTORY, HPC4EData, load_data
    from .solve_hpc4e_reference import (
        ARCHIVES,
        OUTPUT,
        ReferenceField,
        difference,
        load_field,
    )
else:
    from examples.hpc4e_data import DATA_DIRECTORY, HPC4EData, load_data
    from examples.solve_hpc4e_reference import (
        ARCHIVES,
        OUTPUT,
        ReferenceField,
        difference,
        load_field,
    )


def complementary_energy(
    reference: ReferenceField, comparison: ReferenceField, data: HPC4EData, *, order: int = 5
) -> dict[str, float]:
    """Integrate pixelwise plane-strain compliance and test discrete orthogonality.

    A^{-1}sigma = [sigma - lambda/(2mu+2lambda) tr(sigma)I]/(2mu).
    For equal local spaces, exact equilibrium, equal weak-symmetry tests and
    nested admissible stress spaces, the classical minimizer satisfies
    ||sigma_mhm-sigma_ref||²_Ainv = E_mhm-E_ref. This is an identity relative to
    the specified discrete reference, not a continuum error certificate.
    """
    if reference.bounds != comparison.bounds or reference.degree != comparison.degree:
        raise ValueError("energy orthogonality requires the same bounds and local spaces")
    if reference.nx % comparison.nx or reference.ny % comparison.ny:
        raise ValueError("reference grid must contain the comparison partition")
    if reference.nx % data.young.shape[0] or reference.ny % data.young.shape[1]:
        raise ValueError("quadrature cells must align with material pixels")
    if order < reference.degree + 2:
        raise ValueError("quadrature must integrate squared RT polynomials exactly")
    points, weights = quadrilateral_quadrature(order)
    spacing = np.array(
        [
            (reference.bounds[1] - reference.bounds[0]) / reference.nx,
            (reference.bounds[3] - reference.bounds[2]) / reference.ny,
        ]
    )
    origin = np.array(reference.bounds)[[0, 2]]
    lam, mu, _ = data.fields()
    sums = np.zeros(4, dtype=np.longdouble)
    count = reference.nx * reference.ny
    for begin in range(0, count, 512):
        ids = np.arange(begin, min(begin + 512, count))
        xy = np.column_stack((ids % reference.nx, ids // reference.nx))
        physical = (origin + (xy[:, None] + points[None]) * spacing).reshape(-1, 2)
        ref = reference.evaluate(physical)[0]
        other = comparison.evaluate(physical)[0]
        shear, lame = mu(physical), lam(physical)
        ratio = lame / (2 * (shear + lame))
        delta = other - ref
        for index, (a, b) in enumerate(((ref, ref), (other, other), (ref, other), (delta, delta))):
            product = (
                np.einsum("qij,qij->q", a, b)
                - ratio * np.trace(a, axis1=1, axis2=2) * np.trace(b, axis1=1, axis2=2)
            ) / (2 * shear)
            sums[index] += np.sum(
                product.reshape(len(ids), -1) * weights, dtype=np.longdouble
            ) * np.prod(spacing)
    ref_energy, other_energy, cross, squared_error = map(float, sums)
    energy_difference = other_energy - ref_energy
    defect = squared_error - energy_difference
    scale = max(abs(ref_energy), abs(other_energy), np.finfo(float).tiny)
    error_scale = max(abs(squared_error), abs(energy_difference), np.finfo(float).tiny)
    return dict(
        reference_energy=ref_energy,
        comparison_energy=other_energy,
        cross_energy=cross,
        stress_compliance_difference_squared=squared_error,
        energy_difference=energy_difference,
        identity_defect=defect,
        identity_relative_energy=abs(defect) / scale,
        identity_relative_difference=abs(defect) / error_scale,
        orthogonality_defect=cross - ref_energy,
        polarization_defect=squared_error - (other_energy + ref_energy - 2 * cross),
        quadrature_order=order,
    )


def main() -> None:
    """Compare the accepted classical RT1 archive against all supplied MHM enrichments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ARCHIVES / "classical-rt1-512x256.npz")
    parser.add_argument("--segments", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--output", type=Path, default=OUTPUT / "complementary-energy.json")
    args = parser.parse_args()
    ref = load_field(args.reference)
    data = load_data(args.data_dir)
    rows = []
    for segments in args.segments:
        path = ARCHIVES / f"mhm-s{segments}.npz"
        other = load_field(path)
        rows.append(
            dict(
                segments=segments,
                fields=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                l2=difference(ref, other),
                energy=complementary_energy(ref, other, data),
            )
        )
    result = dict(
        reference=args.reference.name,
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        convention="Discrete RT1 complementary-energy identity; no continuum error bound",
        material="Pinned HPC4E pixels, plane strain, dimensionless domain and moduli",
        rows=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
