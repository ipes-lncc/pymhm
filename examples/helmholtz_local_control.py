"""Fine local-resolution control at fixed published Helmholtz macro and trace spaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.helmholtz_article import Configuration, solve_configuration
from examples.helmholtz_campaign import source_hashes

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Checkpoint four r=8 calculations without changing macro geometry or trace degree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/helmholtz-article")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    path = output / "local-refinement-eight.json"
    hashes = source_hashes()
    for name in ("helmholtz_article", "helmholtz_local_control"):
        source = ROOT / "examples" / f"{name}.py"
        hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    record = {"source_sha256": hashes, "rows": []}
    if path.exists():
        record = json.loads(path.read_text())
        if record["source_sha256"] != hashes:
            raise ValueError("local-resolution acquisition sources changed")
    done = {row["key"] for row in record["rows"]}
    for ell, n, omega in ((2, 11, 20 * np.pi), (4, 21, 40 * np.pi)):
        for oscillatory in (False, True):
            config = Configuration("local-control-fine", n, ell, oscillatory, omega, np.pi / 13, 8)
            if config.key in done:
                continue
            row = solve_configuration(config, output)
            if hashes != {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in hashes
            }:
                raise RuntimeError("local-resolution sources changed during acquisition")
            record["rows"].append(row)
            temporary = path.with_suffix(".json.new")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(path)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
