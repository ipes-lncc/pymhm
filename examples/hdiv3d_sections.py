"""One-sided affine H(div) sections using literal producer bases and moment maps."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from examples.hdiv3d_field_archive import reference_tables, restore
from pymhm.meshes.mixed import AffineMixedMesh


def evaluate(
    arrays: Mapping[str, np.ndarray], macro: int, cell: int, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate p, physical Piola flux and divergence at physical owning-cell points.

    The reference matrix, moment transform, geometry and coordinates all come
    from the executed field archive. No new numerical basis is constructed.
    Points may lie on an interface: the explicitly selected owner determines
    their independent one-sided value. Evaluation extends that owner's
    polynomial to any finite physical point; containment is the caller's
    responsibility. The section sampler uses points inside each selected cell.
    """
    points = np.asarray(points)
    if (
        type(macro) is not int
        or macro not in range(int(arrays["local_count"]))
        or type(cell) is not int
        or cell not in range(len(arrays[f"cells_{macro}"]))
        or points.ndim != 2
        or points.shape[1] != 3
        or np.iscomplexobj(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("An archived owner and finite real physical triples are required")
    origin = arrays[f"points_{macro}"][arrays[f"cells_{macro}"][cell, 0]]
    reference = (points - origin) @ arrays[f"inverse_{macro}"][cell].T
    values, divergence, pressure = reference_tables(arrays, reference)
    dofs = arrays[f"flux_dofs_{macro}"][cell]
    coefficients = (
        arrays[f"moment_transform_{macro}"][cell] @ restore(arrays, f"flux_{macro}")[dofs]
    )
    jacobian = arrays[f"jacobian_{macro}"][cell]
    determinant = arrays[f"determinants_{macro}"][cell]
    flux = np.einsum("i,qia->qa", coefficients, values) @ jacobian.T / determinant
    return (
        pressure @ restore(arrays, f"pressure_{macro}")[cell],
        flux,
        divergence @ coefficients / determinant,
    )


def replay_section(arrays: Mapping[str, np.ndarray], refinement: int = 6) -> dict[str, np.ndarray]:
    """Sample the selected unit-cube sine fields at z=.37 without interface averaging.

    Each fine-cell fan owns its vertices. Original macrocell intersections are
    separate overlays; drawing interpolation stays within a fine-cell fan.
    Sampling refinement controls display density, not the solved FE space.
    """
    from examples.sample_core_sections import cut_polygon, section_samples

    if type(refinement) is not int or refinement < 1:
        raise ValueError("Positive integer section sampling refinement required")
    kind = ("tetrahedron", "prism")[int(arrays["cell_kind_code"])]
    meshes = tuple(
        AffineMixedMesh(arrays[f"points_{macro}"], arrays[f"cells_{macro}"], kind)
        for macro in range(int(arrays["local_count"]))
    )

    def sample(macro: int, cell: int, points: np.ndarray) -> np.ndarray:
        """Evaluate pressure and physical flux on the specified incident fine cell."""
        scalar, flux, _ = evaluate(arrays, macro, cell, points)
        return np.column_stack((scalar, flux))

    fields = section_samples(meshes, sample, refinement)
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for ids in arrays["macro_cells"]:
        cut = cut_polygon(arrays["macro_points"][ids])
        if len(cut):
            segments.extend(zip(cut[:, :2], np.roll(cut[:, :2], -1, axis=0), strict=True))
    fields["macro_edges"] = np.asarray(segments)
    if np.any(fields["owners"][fields["cells"]] != fields["owners"][fields["cells"][:, :1]]):
        raise ValueError("Display triangles merge independent fine-cell owners")
    return fields
