"""Explicit acquisition and loading of the pinned SPE10 Model 2 properties.

The dataset retains ECLIPSE I-fastest ordering, mD/ft units and published
porosity values. No network request occurs except an explicit download call.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlopen

import numpy as np

if TYPE_CHECKING:
    from pymhm.io.reservoir import ReservoirData


SPE10_REVISION = "eaa2261683a97027e057c2bc49612ad1c86390b3"


SPE10_FILES = {
    "SPE10MODEL2_PERM.INC": "1829e331b64d2efb416fc40f4d68388087a11e4f7ff5ac6866eb8f1a88f8993d",
    "SPE10MODEL2_PHI.INC": "1426c8b40c8a1ffd3cce6653aad264685f100f82018d212ca74c8ae6d0bd1295",
}


def load_spe10_model2(directory: str | Path) -> ReservoirData:
    """Load the OPM Model 2 property includes as a 60×220×85 reservoir.

    This function performs no network requests. Use :func:`download_spe10_model2`
    to obtain the pinned files explicitly. Source porosity values, including the
    OPM zero replacement of 1e-7, are preserved.
    """
    from pymhm.io.reservoir import ReservoirData, read_eclipse_properties

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
