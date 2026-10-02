"""Explicit shape-controlled local refinement for a diffusion--reaction boundary layer."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymhm.cut_cells import fit_material_mesh
from pymhm.longest_edge import refine_longest_edge
from pymhm.mesh import TriangleMesh


def reaction_layer_mesh(
    mesh: TriangleMesh,
    material: Any,
    *,
    relative_diameter: float = 0.5,
    layer_lengths: float = 6.0,
    max_cells: int = 400_000,
    max_steps: int = 48,
) -> tuple[TriangleMesh, list[dict[str, float | int]]]:
    """Resolve h/sqrt(K) in cells meeting the bottom reaction layer, without changing K.

    This geometry control assumes scalar diffusion, reaction one and a bottom
    nonzero Dirichlet datum. It preserves each existing material-fitted cell
    through conforming longest-edge refinement. The parameter is a declared
    spatial resolution, not a change to the stabilization coefficient.
    """
    if relative_diameter <= 0 or layer_lengths <= 0 or max_cells < 1 or max_steps < 0:
        raise ValueError("require positive geometric scales and valid refinement limits")
    history = []
    for step in range(max_steps + 1):
        vertices = mesh.points[mesh.cells]
        values = np.asarray(material(vertices.mean(axis=1)))
        scalar = values if values.ndim == 1 else values[:, 0, 0]
        if values.ndim != 1 and (
            values.shape != (len(vertices), 2, 2)
            or not np.array_equal(values, scalar[:, None, None] * np.eye(2))
        ):
            raise ValueError("reaction-layer metric requires the stated scalar isotropic diffusion")
        if np.any(scalar <= 0) or not np.all(np.isfinite(scalar)):
            raise ValueError("reaction-layer geometry requires positive finite diffusion")
        length = np.sqrt(scalar)
        diameter = mesh.lengths[mesh.cell_faces].max(axis=1)
        active = vertices[..., 1].min(axis=1) < layer_lengths * length
        marked = active & (diameter > relative_diameter * length)
        history.append(
            {
                "step": step,
                "cells": len(mesh.cells),
                "marked": int(marked.sum()),
                "largest_active_h_over_length": float(
                    np.max(diameter[active] / length[active], initial=0.0)
                ),
            }
        )
        if not np.any(marked):
            return mesh, history
        if step == max_steps or len(mesh.cells) > max_cells:
            raise RuntimeError("reaction-layer resolution exceeds its explicit geometry budget")
        mesh = refine_longest_edge(mesh, marked).mesh
    raise AssertionError("finite refinement loop must return or raise")


def prepare_local(
    cell: int,
    *,
    macro: TriangleMesh,
    material: Any,
    refinement: int,
    layer_resolution: float,
    max_cells: int = 400_000,
) -> tuple[TriangleMesh, list[dict[str, float | int]]]:
    """Fit physical material first, then apply the independently declared layer metric."""
    fine = fit_material_mesh(macro.submesh(cell, refinement), material)
    return reaction_layer_mesh(
        fine, material, relative_diameter=layer_resolution, max_cells=max_cells
    )
