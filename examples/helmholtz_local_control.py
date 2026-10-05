"""Fine local-resolution control at fixed published Helmholtz macro and trace spaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from examples.campaign_checkpoint import require_sources
from examples.helmholtz_article import (
    Configuration,
    article_hashes,
    complete_rows,
    solve_configuration,
    validate_row,
)

ROOT = Path(__file__).resolve().parents[1]


def configurations() -> list[Configuration]:
    """Return the four fixed macro/trace cases with local refinement eight."""
    return [
        Configuration("local-control-fine", n, ell, oscillatory, omega, np.pi / 13, 8)
        for ell, n, omega in ((2, 11, 20 * np.pi), (4, 21, 40 * np.pi))
        for oscillatory in (False, True)
    ]


def hashes() -> dict[str, str]:
    """Bind the analytical, incident and local-control numerical owners."""
    result = article_hashes()
    result[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return result


def publication_rows(output: Path) -> list[dict[str, Any]]:
    """Require all four current local-resolution controls before rendering them."""
    record = json.loads((output / "local-refinement-eight.json").read_text())
    require_sources(record["source_sha256"], hashes())
    return complete_rows(record, configurations(), output)


def main() -> None:
    """Checkpoint four r=8 calculations without changing macro geometry or trace degree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/helmholtz-article")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    path = output / "local-refinement-eight.json"
    sources = hashes()
    record = {"source_sha256": sources, "rows": []}
    if path.exists():
        record = json.loads(path.read_text())
        require_sources(record["source_sha256"], sources)
        for row in record["rows"]:
            validate_row(row, output)
    done = {row["key"] for row in record["rows"]}
    for config in configurations():
        if config.key in done:
            continue
        row = solve_configuration(config, output)
        if sources != {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        }:
            raise RuntimeError("local-resolution sources changed during acquisition")
        record["rows"].append(row)
        temporary = path.with_suffix(".json.new")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
