"""Cartesian reservoir fields and explicitly downloaded SPE10 Model 2 data.

ECLIPSE values are ordered with I/x varying fastest. Permeability in the
source files is in millidarcies and geometry is in feet. Neither coefficients
nor coordinates are silently nondimensionalized. The OPM porosity file replaces
zero porosity by 1e-7; this reader preserves those published values.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
from numpy.typing import NDArray

from pymhm._geometry_roundoff import cartesian_coordinates
from pymhm.mesh import positive_int

Array = NDArray[np.float64]
SPE10_REVISION = "eaa2261683a97027e057c2bc49612ad1c86390b3"
SPE10_FILES = {
    "SPE10MODEL2_PERM.INC": "1829e331b64d2efb416fc40f4d68388087a11e4f7ff5ac6866eb8f1a88f8993d",
    "SPE10MODEL2_PHI.INC": "1426c8b40c8a1ffd3cce6653aad264685f100f82018d212ca74c8ae6d0bd1295",
}
MILLIDARCY_TO_M2 = 9.869233e-16
FOOT_TO_METRE = 0.3048


@dataclass(frozen=True)
class CartesianCellField:
    """Piecewise constant scalar/tensor values on a Cartesian cell grid.

    The first ``len(spacing)`` array axes are spatial (x, y, optionally z).
    Remaining axes are value components. Internal interfaces use the cell on
    their positive side; the external upper boundary uses the last cell. This
    pointwise convention resolves grid-line representations within their propagated
    physical-coordinate roundoff; it does not average material jumps. Quadrature must
    resolve those jumps when using this field as a PDE coefficient.
    """

    values: Array
    spacing: tuple[float, ...]
    origin: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate and freeze independent real field and geometry arrays."""
        spacing = np.asarray(self.spacing, dtype=float)
        if spacing.shape not in ((2,), (3,)) or not np.all(np.isfinite(spacing) & (spacing > 0)):
            raise ValueError("spacing must contain two or three finite positive lengths")
        dimension = len(spacing)
        origin = (
            np.zeros(dimension) if self.origin is None else np.asarray(self.origin, dtype=float)
        )
        if origin.shape != spacing.shape or not np.isfinite(origin).all():
            raise ValueError("origin must contain one finite coordinate per spatial axis")
        values = np.asarray(self.values)
        if (
            values.ndim < dimension
            or 0 in values.shape
            or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError("values must be a nonempty finite real array with spatial axes")
        copied = np.array(values, dtype=float, copy=True)
        copied.flags.writeable = False
        object.__setattr__(self, "values", copied)
        object.__setattr__(self, "spacing", tuple(spacing))
        object.__setattr__(self, "origin", tuple(origin))

    def __call__(self, points: Any) -> Array:
        """Evaluate points of shape ``(n, dimension)`` without extrapolation."""
        points = np.asarray(points, dtype=float)
        dimension = len(self.spacing)
        if points.ndim != 2 or points.shape[1] != dimension or not np.isfinite(points).all():
            raise ValueError("points must be a finite array of shape (n, dimension)")
        coordinates, error = cartesian_coordinates(
            points, np.asarray(self.origin), np.asarray(self.spacing)
        )
        shape = np.asarray(self.values.shape[:dimension])
        tolerance = np.maximum(32 * np.finfo(float).eps * np.maximum(shape, 1), 2 * error)
        if np.any(coordinates < -tolerance) or np.any(coordinates > shape + tolerance):
            raise ValueError("points lie outside the Cartesian field")
        levels = np.rint(coordinates)
        coordinates = np.where(abs(coordinates - levels) <= 2 * error, levels, coordinates)
        indices = np.minimum(np.maximum(coordinates, 0).astype(np.int64), shape - 1)
        return self.values[tuple(indices.T)]


def read_eclipse_properties(path: str | Path) -> dict[str, Array]:
    """Read slash-terminated numeric property blocks, including ``count*value``.

    This deliberately reads property include files, not a complete simulator
    deck. Keywords must be unique, repetitions must specify a finite value,
    and malformed or unterminated blocks raise instead of truncating data.
    Returned flat arrays retain ECLIPSE I-fastest ordering.
    """
    clean = re.sub(r"--[^\n]*", "", Path(path).read_text(encoding="ascii"))
    blocks = clean.split("/")
    if blocks[-1].strip():
        raise ValueError("property blocks must end with a slash")
    result = {}
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
    pattern = re.compile(rf"(?:(\d+)\*)?({number})\Z")
    for block in blocks[:-1]:
        tokens = block.split()
        if not tokens or not re.fullmatch(r"[A-Z][A-Z0-9_]*", tokens[0]):
            raise ValueError("each property requires an uppercase keyword and values")
        key = tokens[0]
        if key in result or len(tokens) == 1:
            raise ValueError("properties must be nonempty and have unique keywords")
        # Validate every token before fromstring: NumPy can otherwise silently
        # accept only the numeric prefix of a malformed input.
        counts, values = [], []
        for token in tokens[1:]:
            match = pattern.fullmatch(token)
            if match is None:
                raise ValueError(f"invalid numeric property token: {token}")
            count = 1 if match[1] is None else int(match[1])
            if count < 1:
                raise ValueError("property repetition counts must be positive")
            counts.append(count)
            values.append(float(match[2].replace("D", "e").replace("d", "e")))
        array = np.repeat(np.asarray(values), counts)
        if not np.isfinite(array).all():
            raise ValueError("property values must be finite")
        result[key] = array
    if not result:
        raise ValueError("no numeric properties found")
    return result


@dataclass(frozen=True)
class ReservoirData:
    """Cellwise principal permeability and porosity with explicit physical units.

    ``permeability`` has shape ``(nx, ny, nz, 3)`` and ``porosity`` has shape
    ``(nx, ny, nz)``. Components are Kx, Ky, Kz. Horizontal slices preserve the
    x/y ordering and original geometry, with no interpolation or upscaling.
    """

    permeability: Array
    porosity: Array
    spacing: tuple[float, ...] = (20.0, 10.0, 2.0)
    permeability_unit: str = "mD"
    length_unit: str = "ft"

    def __post_init__(self) -> None:
        """Validate material ranges and store immutable independent arrays."""
        k = CartesianCellField(self.permeability, self.spacing)
        phi = CartesianCellField(self.porosity, self.spacing)
        if k.values.shape != phi.values.shape + (3,) or phi.values.ndim != 3 or len(k.spacing) != 3:
            raise ValueError("permeability/porosity shapes must be (nx,ny,nz,3)/(nx,ny,nz)")
        if np.any(k.values <= 0) or np.any((phi.values < 0) | (phi.values > 1)):
            raise ValueError("permeability must be positive and porosity lie in [0,1]")
        if not self.permeability_unit or not self.length_unit:
            raise ValueError("physical units must be explicit nonempty strings")
        object.__setattr__(self, "permeability", k.values)
        object.__setattr__(self, "porosity", phi.values)
        object.__setattr__(self, "spacing", k.spacing)

    def layer(self, number: int) -> tuple[CartesianCellField, CartesianCellField]:
        """Return planar tensor K and scalar porosity for a **one-based** layer."""
        number = positive_int(number, "layer")
        if number > self.porosity.shape[2]:
            raise ValueError("layer exceeds the number of reservoir layers")
        horizontal = self.permeability[:, :, number - 1, :2]
        tensor = np.zeros(horizontal.shape[:2] + (2, 2))
        tensor[..., 0, 0] = horizontal[..., 0]
        tensor[..., 1, 1] = horizontal[..., 1]
        return (
            CartesianCellField(tensor, self.spacing[:2]),
            CartesianCellField(self.porosity[:, :, number - 1], self.spacing[:2]),
        )

    def to_si(self) -> ReservoirData:
        """Convert an mD/ft reservoir to m²/m; return SI input unchanged."""
        if (self.permeability_unit, self.length_unit) == ("m2", "m"):
            return self
        if (self.permeability_unit, self.length_unit) != ("mD", "ft"):
            raise ValueError("SI conversion requires permeability in mD and lengths in ft")
        return ReservoirData(
            self.permeability * MILLIDARCY_TO_M2,
            self.porosity,
            tuple(x * FOOT_TO_METRE for x in self.spacing),
            "m2",
            "m",
        )


def load_spe10_model2(directory: str | Path) -> ReservoirData:
    """Load the OPM Model 2 property includes as a 60×220×85 reservoir.

    This function performs no network requests. Use :func:`download_spe10_model2`
    to obtain the pinned files explicitly. Source porosity values, including the
    OPM zero replacement of 1e-7, are preserved.
    """
    directory = Path(directory)
    properties = read_eclipse_properties(directory / "SPE10MODEL2_PERM.INC")
    properties.update(read_eclipse_properties(directory / "SPE10MODEL2_PHI.INC"))
    shape = (60, 220, 85)
    if set(properties) != {"PERMX", "PERMY", "PERMZ", "PORO"} or any(
        len(value) != np.prod(shape) for value in properties.values()
    ):
        raise ValueError("SPE10 Model 2 requires PERMX/PERMY/PERMZ/PORO with 60*220*85 values each")
    permeability = np.stack(
        [properties[key].reshape(shape, order="F") for key in ("PERMX", "PERMY", "PERMZ")],
        axis=-1,
    )
    return ReservoirData(permeability, properties["PORO"].reshape(shape, order="F"))


def download_spe10_model2(directory: str | Path) -> Path:
    """Explicitly download ~77 MB of pinned OPM properties with SHA-256 checks.

    Valid cached files are reused. A mismatched existing file raises, avoiding
    an unnoticed change to user-provided data. Incomplete or corrupt downloads
    never replace the target file. OPM attributes these property files to
    Christie and Blunt; dataset/deck licensing is documented by OPM/opm-data.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/OPM/opm-data/{SPE10_REVISION}/spe10model2"
    for name, checksum in SPE10_FILES.items():
        target = directory / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"SHA-256 mismatch for cached file {name}")
            continue
        with tempfile.TemporaryDirectory(dir=directory) as temporary_directory:
            temporary = Path(temporary_directory) / name
            with (
                urlopen(f"{base}/{name}", timeout=120) as response,
                temporary.open("wb") as stream,
            ):
                digest = hashlib.sha256()
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    stream.write(chunk)
            if digest.hexdigest() != checksum:
                raise ValueError(f"SHA-256 mismatch for downloaded file {name}")
            temporary.replace(target)
    return directory
