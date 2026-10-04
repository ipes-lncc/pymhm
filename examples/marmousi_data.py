"""Read a declared Marmousi II acoustic crop from the primary SEG-Y data.

The selected origin is inferred from rendered material maps; it does not
identify the historical arrays. Native samples have nominal spacing 1.25 m,
with x varying between traces and depth within a trace. The 5-metre material
cells sample their centres without interpolation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pymhm.materials.cartesian import CartesianCellField

FILES = {
    "vp": (
        "vp_marmousi-ii.segy",
        "https://ahay.org/data/marm2/vp_marmousi-ii.segy",
        "2722a5de0645d8fcf5750f380893d4875383af9b24988ece0bd07b943f5b0c91",
    ),
    "density": (
        "density_marmousi-ii.segy",
        "https://ahay.org/data/marm2/density_marmousi-ii.segy",
        "10ec5c4274a63ffde2ac94dda4f062d5b83827efcefeb1a2ab22438a85f32967",
    ),
}


def _ibm_values(words: Any) -> Any:
    """Decode IBM base-16 floats exactly into binary64 samples."""
    words = np.asarray(words, dtype=np.uint32)
    mantissa = (words & 0xFFFFFF).astype(float)
    exponent = ((words >> 24) & 0x7F).astype(int) - 64
    sign = np.where(words >> 31, -1.0, 1.0)
    return sign * np.ldexp(mantissa, 4 * exponent - 24)


def _read_samples(path: Path, ix: Any, iz: Any, *, shape: tuple[int, int]) -> Any:
    """Read indexed samples, validating IBM format and fixed trace lengths."""
    nx, nz = shape
    stride = 240 + 4 * nz
    if path.stat().st_size != 3600 + nx * stride:
        raise ValueError("SEG-Y file size does not match the declared sample grid")
    mapped = np.memmap(path, mode="r", dtype=np.uint8)
    try:
        if int.from_bytes(mapped[3224:3226].tobytes(), "big") != 1:
            raise ValueError("Marmousi reader requires SEG-Y IBM float format 1")
        if int.from_bytes(mapped[3220:3222].tobytes(), "big") != nz:
            raise ValueError("SEG-Y binary-header sample count differs from the grid")
        counts = np.ndarray((nx,), dtype=">u2", buffer=mapped, offset=3714, strides=(stride,))
        if np.any(counts != nz):
            raise ValueError("SEG-Y trace sample counts must agree with the grid")
        words = np.ndarray((nx, nz), dtype=">u4", buffer=mapped, offset=3840, strides=(stride, 4))
        return _ibm_values(words[np.ix_(ix, iz)])
    finally:
        mapped._mmap.close()


@dataclass(frozen=True)
class MarmousiMaterial:
    """SI acoustic coefficients on a crop with computational origin zero."""

    density: CartesianCellField
    velocity: CartesianCellField
    bulk_modulus: CartesianCellField
    provenance: dict[str, Any]


def load_marmousi_crop(
    directory: str | Path,
    *,
    origin: tuple[float, float] = (3395.0, 515.0),
    shape: tuple[int, int] = (2048, 512),
    spacing: float = 5.0,
) -> MarmousiMaterial:
    """Load pinned primary files and sample a rectangular cell-centred crop.

    Download the two files named in FILES explicitly before calling this
    function. SHA-256 verification is mandatory. The source covers 17 km by
    3.5 km; its nominal 1.25-metre convention differs from the 1.249-metre
    value in Madagascar's example conversion recipe. Native velocity (km/s)
    and density (g/cm3) are multiplied by 1000. No smoothing, clipping,
    rescaling to a figure, or silent extrapolation occurs.
    """
    raw_shape, raw_spacing = (13601, 2801), 1.25
    counts = np.asarray(shape)
    start = np.asarray(origin)
    if (
        counts.shape != (2,)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 1)
        or start.shape != (2,)
        or np.iscomplexobj(start)
        or not np.isfinite(start).all()
        or np.iscomplexobj(spacing)
        or not np.isfinite(spacing)
        or spacing <= 0
    ):
        raise ValueError("origin, shape and spacing must define a finite positive crop")
    indices = []
    for axis in range(2):
        coordinate = (start[axis] + spacing * (np.arange(counts[axis]) + 0.5)) / raw_spacing
        nearest = np.rint(coordinate).astype(np.int64)
        if not np.allclose(coordinate, nearest, rtol=0, atol=1e-10):
            raise ValueError("material cell centres must coincide with primary sample nodes")
        if np.min(nearest) < 0 or np.max(nearest) >= raw_shape[axis]:
            raise ValueError("crop cell centres lie outside the primary dataset")
        indices.append(nearest)
    values, sources = {}, {}
    for key, (filename, url, expected) in FILES.items():
        path = Path(directory) / filename
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {filename}")
        samples = 1000 * _read_samples(path, *indices, shape=raw_shape)
        if not np.isfinite(samples).all() or np.any(samples <= 0):
            raise ValueError("acoustic material samples must be finite and strictly positive")
        values[key] = samples
        sources[key] = {"filename": filename, "url": url, "sha256": digest}
    mesh_spacing = (float(spacing), float(spacing))
    return MarmousiMaterial(
        CartesianCellField(values["density"], mesh_spacing),
        CartesianCellField(values["vp"], mesh_spacing),
        CartesianCellField(values["density"] * values["vp"] ** 2, mesh_spacing),
        {
            "dataset": "Marmousi II, Martin, Wiley and Marfurt (2006)",
            "doi": "10.1190/1.2172306",
            "sources": sources,
            "primary_shape": list(raw_shape),
            "primary_spacing_m": raw_spacing,
            "crop_origin_m": start.tolist(),
            "cell_shape": counts.tolist(),
            "cell_spacing_m": spacing,
            "sampling": "Primary nodes at material-cell centres; no interpolation",
            "coordinate_order": ["x", "depth"],
            "density_unit": "kg/m^3",
            "velocity_unit": "m/s",
            "bulk_modulus_unit": "Pa",
            "historical_article_arrays_identified": False,
        },
    )
