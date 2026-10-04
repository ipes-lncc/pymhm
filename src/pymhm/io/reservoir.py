"""ECLIPSE property I/O and reservoir data with explicit physical units.

ECLIPSE values are ordered with I/x varying fastest. Permeability in the
source files is in millidarcies and geometry is in feet. Neither coefficients
nor coordinates are silently nondimensionalized. The OPM porosity file replaces
zero porosity by 1e-7; this reader preserves those published values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pymhm.core.validation import positive_int
from pymhm.io.datasets.spe10 import (
    SPE10_FILES as SPE10_FILES,
)
from pymhm.io.datasets.spe10 import (
    SPE10_REVISION as SPE10_REVISION,
)
from pymhm.io.datasets.spe10 import (
    download_spe10_model2 as download_spe10_model2,
)
from pymhm.io.datasets.spe10 import (
    load_spe10_model2 as load_spe10_model2,
)
from pymhm.materials.cartesian import (
    CartesianCellField as CartesianCellField,
)

Array = NDArray[np.float64]
MILLIDARCY_TO_M2 = 9.869233e-16
FOOT_TO_METRE = 0.3048


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
