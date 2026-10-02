"""Reusable color-scale layouts for scientific PyVista example figures."""

from __future__ import annotations

import numpy as np
import pyvista as pv


def signed_asinh_scale(values: np.ndarray, extent: float) -> tuple[np.ndarray, dict[float, str]]:
    """Map signed data to [-1, 1] and label five colors in physical units.

    The odd, monotone map is asinh(100*q/extent)/asinh(100). Its linear
    transition scale is one percent of the maximum absolute extent. No values
    are clipped or averaged. Use one common extent for compared fields and
    a separate extent for their signed difference.
    """
    scale = np.arcsinh(100.0)
    ticks = np.linspace(-1, 1, 5)
    if extent == 0:
        return np.zeros_like(values), {0.0: "0"}
    transformed = np.arcsinh(100.0 * values / extent) / scale
    physical_ticks = extent * np.sinh(scale * ticks) / 100.0
    labels = {float(t): f"{q:.3g}" for t, q in zip(ticks, physical_ticks, strict=True)}
    return transformed, labels


def horizontal_color_scale(
    plotter: pv.Plotter,
    actor: pv.Actor,
    title: str,
    scientific: bool = True,
) -> None:
    """Draw a horizontal scale with a separately positioned, centered title.

    The caller reserves the lower viewport band for this scale. The title is
    placed at normalized height 0.135; the bar starts at 0.035 and occupies
    height 0.065. The mapper, its limits and the numerical field are unchanged.
    Text sizes support a figure width of 180 mm for a 2100-pixel image. Three
    labels leave room for scientific notation at that size. Separate keys
    prevent PyVista from merging scales across subplot renderers.
    """
    scale_title = plotter.add_text(
        title, position=(0.5, 0.135), viewport=True, font_size=18, color="black"
    )
    scale_title.GetTextProperty().SetJustificationToCentered()
    scale_title.GetTextProperty().SetVerticalJustificationToBottom()
    bar = plotter.add_scalar_bar(
        title=f"{title}:{plotter.renderer.GetViewport()}",
        mapper=actor.mapper,
        vertical=False,
        position_x=0.14,
        position_y=0.035,
        width=0.72,
        height=0.065,
        n_labels=3,
        label_font_size=32,
        unconstrained_font_size=True,
        fmt="%.2e" if scientific else "%.3g",
        color="black",
    )
    bar.SetTitle("")
    bar.GetLabelTextProperty().SetJustificationToCentered()
