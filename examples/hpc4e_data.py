"""Pinned material samples for the 2021 heterogeneous mixed-elasticity case.

The original files list horizontal index first and depth downward. Returned
arrays use horizontal index first and vertical height upward. Material values
are retained verbatim; neither clay interfaces nor intermediate samples are
thresholded or smoothed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from pymhm.io.workspace import case_workspace, read_resource_text, resource_file
from pymhm.materials.cartesian import CartesianCellField

REVISION = "f978f29d657d28fe58bcea20fabee68953093482"
BASE_URL = f"https://raw.githubusercontent.com/labmec/MHM/{REVISION}/Data_13_Set"
LENGTH_SCALE = 10000.0
STRESS_SCALE = 1e8
GRAVITY = 9.81
BOUNDS = (0.0, 1.0, 0.0, 0.45)
DATA_DIRECTORY = case_workspace() / "build/datasets/hpc4e"


def read_samples(path: Path) -> np.ndarray:
    """Read a finite rectangular array and convert depth to upward height."""
    tokens = read_resource_text(path, encoding="ascii").split()
    if len(tokens) < 3:
        raise ValueError("material file needs a two-integer shape and sample values")
    try:
        nx, nz = int(tokens[0]), int(tokens[1])
        values = np.asarray(tokens[2:], dtype=float)
    except ValueError as error:
        raise ValueError("invalid material shape or numeric samples") from error
    if min(nx, nz) < 1 or len(values) != nx * nz or not np.isfinite(values).all():
        raise ValueError("material shape must match all finite samples")
    result = values.reshape(nx, nz)[:, ::-1].copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class HPC4EData:
    """Physical Young modulus [Pa], Poisson ratio and density [kg/m³]."""

    young: np.ndarray
    poisson: np.ndarray
    density: np.ndarray

    def __post_init__(self) -> None:
        """Reject incompatible, nonphysical or mutable input material arrays."""
        shape = np.shape(self.young)
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("material arrays must be nonempty and two-dimensional")
        for name in ("young", "poisson", "density"):
            values = np.asarray(getattr(self, name))
            if values.shape != shape or np.iscomplexobj(values) or not np.isfinite(values).all():
                raise ValueError("material arrays must share a finite real shape")
            copied = np.array(values, dtype=float, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if np.any(self.young <= 0) or np.any(self.density <= 0):
            raise ValueError("Young modulus and density must be positive")
        if np.any((self.poisson <= -1) | (self.poisson >= 0.5)):
            raise ValueError("Poisson ratio must lie strictly between -1 and 1/2")

    def fields(self) -> tuple[CartesianCellField, CartesianCellField, CartesianCellField]:
        """Return dimensionless plane-strain Lamé fields and gravitational body force.

        Coordinates and displacement are divided by 10000 m, stresses/moduli
        by 1e8 Pa. Thus div(sigma') + (10000/1e8) f = 0 on [0,1] × [0,.45].
        Applying one common modulus scale also to the body force preserves
        physical displacements; the original physical data remain available.
        """
        nu = self.poisson
        mu = self.young / (2 * (1 + nu)) / STRESS_SCALE
        lam = 2 * mu * nu / (1 - 2 * nu)
        force = np.stack((np.zeros_like(nu), -GRAVITY * self.density), axis=-1)
        force *= LENGTH_SCALE / STRESS_SCALE
        spacing = (1 / self.young.shape[0], 0.45 / self.young.shape[1])
        return tuple(CartesianCellField(values, spacing) for values in (lam, mu, force))


def load_data(directory: Path = DATA_DIRECTORY, *, download: bool = False) -> HPC4EData:
    """Load the pinned 512×256 samples, downloading only when explicitly requested."""
    import json

    manifest = resource_file("examples/results/hpc4e/dataset.json")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    fields = []
    for name in ("VE", "VPoisso", "Vden"):
        path = directory / f"{name}.txt"
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"missing {path}; use --download for the pinned data")
            directory.mkdir(parents=True, exist_ok=True)
            with urlopen(f"{BASE_URL}/{name}.txt", timeout=60) as response:
                content = response.read(8_000_001)
            if len(content) > 8_000_000:
                raise ValueError("material download exceeds the expected file-size bound")
            if hashlib.sha256(content).hexdigest() != metadata["sha256"][name]:
                raise ValueError(f"unexpected checksum for {name}")
            path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"][name]:
            raise ValueError(f"unexpected checksum for {name}")
        values = read_samples(path)
        if values.shape != (512, 256):
            raise ValueError("the published material grid must have shape 512×256")
        fields.append(values)
    return HPC4EData(*fields)
