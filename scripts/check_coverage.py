"""Enforce independent line and branch coverage thresholds from coverage.py JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Reject incomplete measurement or either percentage below the quality gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON report produced by coverage.py")
    parser.add_argument(
        "--minimum", type=float, default=99.0, help="Required percentage (default: 99)"
    )
    args = parser.parse_args()
    if not 0 <= args.minimum <= 100:
        parser.error("minimum must be a percentage between zero and 100")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not report["meta"]["branch_coverage"]:
        raise SystemExit("Branch measurement must be enabled")
    totals = report["totals"]
    if totals["num_statements"] == 0:
        raise SystemExit("No executable statements were measured")
    lines = 100 * totals["covered_lines"] / totals["num_statements"]
    branches = (
        100 * totals["covered_branches"] / totals["num_branches"]
        if totals["num_branches"]
        else 100.0
    )
    print(f"Line coverage: {lines:.4f}%; branch coverage: {branches:.4f}%")
    if min(lines, branches) < args.minimum:
        raise SystemExit(f"Both line and branch coverage must reach {args.minimum:g}%")


if __name__ == "__main__":
    main()
