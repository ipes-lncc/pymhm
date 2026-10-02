"""Export small, verified Marmousi pressure samples for reproducible notebooks.

The four one-sided values are copied from each acquisition without averaging
or re-evaluating the broken field. Integrated physical norms remain separate
records: these unweighted point samples do not replace a volume integral.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def digest(path: Path) -> str:
    """Hash an archive in bounded memory, including materialized LFS payloads."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_samples(record_path: Path, destination: Path) -> dict[str, Any]:
    """Verify an acquisition and copy its finite, distinct incident pressure samples.

    Only the sample coordinates, complex pressures and side identifiers are
    loaded. The potentially large local-coefficient array stays in its archive.
    Both the original acquisition digest and the small output digest are retained.
    """
    record_digest = digest(record_path)
    record = json.loads(record_path.read_text())
    if record["source_changed_during_run"]:
        raise ValueError("the acquisition sources changed during execution")
    archive_name = record["archive"]
    if (
        not isinstance(archive_name, str)
        or Path(archive_name).name != archive_name
        or "/" in archive_name
        or "\\" in archive_name
        or Path(archive_name).suffix != ".npz"
    ):
        raise ValueError("the acquisition archive must be a local NPZ filename")
    source = record_path.parent / archive_name
    if digest(source) != record["archive_sha256"]:
        raise ValueError("the acquired archive does not match its recorded digest")
    with np.load(source, allow_pickle=False) as archive:
        points = archive["sample_points"]
        pressure = archive["sample_pressure"]
        sides = archive["incident_sides"]
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not len(points)
        or pressure.shape != (4, len(points))
        or sides.shape != (4, 2)
        or set(map(tuple, sides)) != {(-1, -1), (-1, 1), (1, -1), (1, 1)}
        or not np.isfinite(points).all()
        or not np.isfinite(pressure).all()
    ):
        raise ValueError("finite two-dimensional samples and four distinct incident sides required")
    if digest(record_path) != record_digest or digest(source) != record["archive_sha256"]:
        raise ValueError("the acquisition record or archive changed during extraction")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{record_path.stem}-samples.npz"
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, sample_points=points, sample_pressure=pressure, incident_sides=sides
    )
    temporary.replace(output)
    return {
        "H_m": record["H_m"],
        "trace_degree": record["trace_degree"],
        "archive": output.name,
        "archive_sha256": digest(output),
        "archive_bytes": output.stat().st_size,
        "sample_count": len(points),
        "incident_sides": sides.tolist(),
        "acquisition_record": record_path.name,
        "acquisition_record_sha256": record_digest,
        "acquisition_archive": source.name,
        "acquisition_archive_sha256": record["archive_sha256"],
    }


def main() -> None:
    """Export the fifteen acquired H/trace-degree pairs and their sample manifest."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "examples/results/marmousi/family")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.source / "samples"
    rows = []
    for width in (20, 40, 80):
        for degree in range(5):
            row = export_samples(args.source / f"mhm-H{width}-ell{degree}-q9.json", output)
            if row["H_m"] != width or row["trace_degree"] != degree:
                raise ValueError("the acquired discretization differs from its filename")
            rows.append(row)
    manifest = {
        "rows": rows,
        "sampling": "Four separate incident values copied from each acquisition; no averaging",
        "scope": "Unweighted pressure sample norms; integrated norms require full field archives",
        "source_sha256": {Path(__file__).name: digest(Path(__file__))},
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")


if __name__ == "__main__":
    main()
