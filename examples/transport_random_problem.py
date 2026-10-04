"""Explicit inputs for a selected L11 Section 5.4 Darcy--transport realization.

The paper specifies a 64-by-16 exponential permeability field in [1e-2,10],
but does not supply its array, probability law, seed, or Darcy boundary drive.
This example declares those inputs independently: iid uniform logarithms,
PCG64, pressure 3-x on the vertical sides and zero normal flux horizontally.
It is a selected physical case, not the historical random realization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from pymhm import TriangleMesh
from pymhm.mesh import positive_int
from pymhm.reservoir import CartesianCellField

DEFAULT_SEED = 20261003
SHAPE = (64, 16)
SPACING = (3 / 64, 1 / 16)
INPUT = Path(__file__).resolve().parent / "data/transport-random-2015/permeability.json"


def permeability_digest(values: np.ndarray) -> str:
    """Identify actual ordered binary64 values with x/y axes and little-endian bytes."""
    array = np.asarray(values, dtype="<f8", order="C")
    contract = json.dumps({"shape": list(array.shape), "dtype": "<f8"}, sort_keys=True)
    return hashlib.sha256(contract.encode("ascii") + array.tobytes()).hexdigest()


def generate_permeability(seed: int = DEFAULT_SEED) -> CartesianCellField:
    """Draw K=10**xi with independent xi in [-2,1]; preserve every sampled value.

    The explicit PCG64 seed selects a reproducible case. No endpoint clipping,
    extrema normalization, spatial smoothing or matching to a published image
    is performed. Axes are x first, y second on [0,3] by [0,1].
    """
    positive_int(seed, "seed", 0)
    generator = np.random.Generator(np.random.PCG64(seed))
    values = np.power(10.0, generator.uniform(-2.0, 1.0, size=SHAPE))
    return CartesianCellField(values, SPACING)


def save_realization(path: Path = INPUT, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Atomically persist the actual selected material array and its declared drive."""
    field = generate_permeability(seed)
    record = {
        "schema": "pymhm-selected-random-transport-v1",
        "primary_source": {"doi": "10.1137/130938499", "section": "5.4"},
        "scope": "Selected realization; historical array and Darcy drive unavailable",
        "generator": {"bit_generator": "PCG64", "seed": seed, "law": "iid xi~U[-2,1]; K=10**xi"},
        "shape": list(SHAPE),
        "axes": ["x", "y"],
        "origin": [0.0, 0.0],
        "spacing": list(SPACING),
        "units": "Dimensionless domain, permeability, unit viscosity and capacity",
        "permeability_sha256": permeability_digest(field.values),
        "values": field.values.tolist(),
        "darcy": {
            "source": 0.0,
            "pressure_left": 3.0,
            "pressure_right": 0.0,
            "horizontal_normal_flux": 0.0,
        },
        "transport": {
            "initial": 0.0,
            "inflow": 1.0,
            "source": 0.0,
            "natural_diffusive_flux": 0.0,
            "final_time": 7.0,
            "molecular": 1e-6,
            "longitudinal": 1e-2,
            "transverse": 1e-3,
        },
    }
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record


def load_realization(path: Path = INPUT) -> tuple[CartesianCellField, dict[str, Any]]:
    """Replay archived values, requiring the full numerical input contract.

    The stored array is authoritative; loading does not redraw it from the
    seed. Changed axes, geometry, physical data or material bytes are rejected.
    """
    record = json.loads(path.read_text())
    positive_int(record["generator"]["seed"], "seed", 0)
    if (
        record["schema"] != "pymhm-selected-random-transport-v1"
        or record["primary_source"] != {"doi": "10.1137/130938499", "section": "5.4"}
        or record["scope"] != "Selected realization; historical array and Darcy drive unavailable"
        or record["shape"] != list(SHAPE)
        or record["axes"] != ["x", "y"]
        or record["origin"] != [0.0, 0.0]
        or record["spacing"] != list(SPACING)
        or record["generator"]["bit_generator"] != "PCG64"
        or record["generator"]["law"] != "iid xi~U[-2,1]; K=10**xi"
        or record["units"] != "Dimensionless domain, permeability, unit viscosity and capacity"
        or record["darcy"]
        != {
            "source": 0.0,
            "pressure_left": 3.0,
            "pressure_right": 0.0,
            "horizontal_normal_flux": 0.0,
        }
        or record["transport"]
        != {
            "initial": 0.0,
            "inflow": 1.0,
            "source": 0.0,
            "natural_diffusive_flux": 0.0,
            "final_time": 7.0,
            "molecular": 1e-6,
            "longitudinal": 1e-2,
            "transverse": 1e-3,
        }
    ):
        raise ValueError("selected random transport input contract differs")
    values = np.asarray(record["values"], dtype=float)
    if (
        values.shape != SHAPE
        or not np.isfinite(values).all()
        or np.any((values < 1e-2) | (values > 10))
        or permeability_digest(values) != record["permeability_sha256"]
    ):
        raise ValueError("selected permeability array or digest differs")
    return CartesianCellField(values, SPACING), record


def macro_mesh(nx: int = 32, ny: int = 8) -> TriangleMesh:
    """Return the selected 512 SW-to-NE macrotriangles at the default resolution."""
    unit = TriangleMesh.unit_square(nx, ny)
    return TriangleMesh(unit.points * (3.0, 1.0), unit.cells)


def require_fitted_partition(nx: int, ny: int, refinement: int) -> None:
    """Require the fine Cartesian grid to resolve all 64-by-16 material pixels."""
    for name, value in (("nx", nx), ("ny", ny), ("refinement", refinement)):
        positive_int(value, name)
    if nx * refinement % SHAPE[0] or ny * refinement % SHAPE[1]:
        raise ValueError("fine partition must resolve the 64-by-16 material interfaces")


def darcy_pressure(points: np.ndarray) -> np.ndarray:
    """Evaluate the declared exterior pressure drive 3-x, not an exact interior field."""
    return 3.0 - points[:, 0]


def natural_faces(mesh: TriangleMesh, *, transport: bool) -> dict[int, float]:
    """Select zero diffusive transport data except inflow, or horizontal Darcy no-flow."""
    normal_x = mesh.normals[:, 0]
    selected = normal_x != -1.0 if transport else normal_x == 0.0
    return {int(face): 0.0 for face in mesh.boundary_faces if selected[face]}


def main() -> None:
    """Write a selected input explicitly; no PDE campaign is run by this command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=INPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    record = save_realization(args.output, seed=args.seed)
    print(args.output, record["permeability_sha256"])


if __name__ == "__main__":
    main()
