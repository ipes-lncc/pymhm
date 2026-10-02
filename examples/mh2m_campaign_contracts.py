"""Validate the physical norm schema returned by the shared MH2M overlay integrator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

from examples.campaign_provenance import require_equal, verify_result

NORM_NAMES = ("pressure", "flux", "gradient", "energy")


def validate_difference_norm(values: Mapping[str, Any], order: int) -> None:
    """Require complete finite norms and preserve undefined zero-denominator ratios.

    Absolute norms and differences are finite and nonnegative. A relative
    difference is ``None`` exactly when its reference norm is zero, matching
    ``mh2m_heterogeneous_norms.difference``. Additional JSON metadata are retained.
    This validates completeness and representation, not numerical accuracy.
    """
    required = {"quadrature_order", "overlay_area"}
    for name in NORM_NAMES:
        required.update(
            (f"{name}_difference", f"reference_{name}_norm", f"{name}_relative_difference")
        )
    if not required <= values.keys():
        raise ValueError("completed physical norm is missing required fields")
    require_equal(values["quadrature_order"], order, label="physical norm quadrature")

    def absolute(value: Any) -> bool:
        """Recognize a finite nonnegative real number without treating bool as a norm."""
        if not isinstance(value, Real) or isinstance(value, bool):
            return False
        try:
            number = float(value)
        except OverflowError:
            return False
        return isfinite(number) and number >= 0

    if not absolute(values["overlay_area"]) or values["overlay_area"] == 0:
        raise ValueError("completed physical norm requires a finite positive overlay area")
    for name in NORM_NAMES:
        numerator = values[f"{name}_difference"]
        denominator = values[f"reference_{name}_norm"]
        relative = values[f"{name}_relative_difference"]
        if not absolute(numerator) or not absolute(denominator):
            raise ValueError("absolute physical norms must be finite and nonnegative")
        if denominator == 0:
            if relative is not None:
                raise ValueError("a zero reference norm requires an undefined relative difference")
        elif not absolute(relative):
            raise ValueError("a positive reference norm requires a finite nonnegative ratio")


def verify_difference_result(
    recorded: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    archives: Mapping[Path, str],
    norm_orders: Sequence[int],
) -> None:
    """Check the archive contract together with the integrator's complete physical schema."""
    verify_result(
        recorded,
        identity,
        archives=archives,
        norm_orders=norm_orders,
        norm_validator=validate_difference_norm,
    )
