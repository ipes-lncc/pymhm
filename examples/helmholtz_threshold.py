"""Finite-sequence Helmholtz diagnostics including rejected local inverses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def sampled_threshold(
    rows: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]] = (),
    *,
    factor: float = 3.0,
) -> dict[str, Any]:
    """Find the largest verified suffix, treating every rejected resolution as a barrier.

    A sampled suffix establishes no bound at uncomputed mesh sizes. In particular,
    a local resonance between two successful solves cannot be omitted from the
    set of attempted resolutions when reporting a threshold.
    """
    attempted = {int(row["n"]): float(row["ratio"]) <= factor for row in rows}
    for row in rejected:
        attempted[int(row["n"])] = False
    suffix: list[int] = []
    for n, accepted in sorted(attempted.items(), reverse=True):
        if not accepted:
            break
        suffix.append(n)
    threshold = 1.0 / min(suffix) if suffix else None
    preceding = [n for n in attempted if suffix and n < min(suffix)]
    return {
        "sampled_suffix_threshold": threshold,
        "transition_bracket": [threshold, 1.0 / max(preceding)] if preceding else None,
        "accepted_suffix_resolutions": sorted(suffix),
        "rejected_resolutions": sorted(int(row["n"]) for row in rejected),
    }
