"""Audit the ideal constant-moment local space for identity-diffusion macros.

For a tetrahedron U^{0,0} is span(1,x,y,z,|x|²). For an axis-aligned
box it is span(1,x,y,z,x²,y²,z²). These polynomials have constant
Laplacians and constant normal derivatives on each original face.
Compatible Neumann data plus the constant kernel give dimensions five and
seven, respectively. Volume and face moments are unisolvent: integration
by parts makes the energy of a polynomial with zero moments vanish.

Conforming fine P2 contains these spaces. Integration by parts also makes
their energy lifts orthogonal to every fine function with zero moments;
therefore the ideal and finite lifts coincide in these selected cases.
This argument does not cover other materials, general polyhedra or degrees.
The numerical audit compares the actual archived lift; it never replaces it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from examples.mshho3d_field_archive import restore


def ideal_p0_audit(arrays: Mapping[str, np.ndarray], kind: str) -> dict[str, Any]:
    """Check every executed lift against its independently declared polynomial space.

    Coordinates are centered and scaled by the macro diameter for this audit.
    Physical moments remain the archived unnormalized volume/face integrals.
    The small moment inversion constructs an analytical comparison, never a
    new replay basis. A relative discrepancy above1e-10 is rejected.
    """
    if kind not in {"tetra", "cube"}:
        raise ValueError("Only tetrahedra and axis-aligned boxes are covered")
    if int(arrays["degree"]) != 2 or int(arrays["cell_degree"]) != 0:
        raise ValueError("This exact-space argument requires conforming P2 and P0 moments")
    expected = 5 if kind == "tetra" else 7
    offsets = arrays["macro_cell_face_offsets"]
    faces = arrays["macro_cell_faces"]
    maxima = {
        "lift_relative_linf": 0.0,
        "moment_inverse_defect": 0.0,
        "moment_matrix_condition": 0.0,
    }
    rows = []
    for cell in range(int(arrays["local_count"])):
        face_ids = faces[offsets[cell] : offsets[cell + 1]]
        if len(face_ids) != expected - 1:
            raise ValueError("Original macroface count differs from the ideal polynomial space")
        normals = arrays["macro_normals"][face_ids]
        face_vertices: list[np.ndarray] = []
        for face, normal in zip(face_ids, normals, strict=True):
            begin, end = arrays["macro_face_offsets"][face : face + 2]
            points = arrays["macro_points"][arrays["macro_face_vertices"][begin:end]]
            if not np.allclose((points - points[0]) @ normal, 0, rtol=0, atol=1e-12):
                raise ValueError("The original macroface is not planar")
            face_vertices.extend(points)
        vertices = np.unique(np.asarray(face_vertices), axis=0)
        if kind == "tetra" and len(vertices) != 4:
            raise ValueError("Four tetrahedral vertices required")
        if kind == "cube":
            if len(vertices) != 8 or not np.allclose(
                np.sort(np.abs(normals), axis=1), [0, 0, 1], rtol=0, atol=1e-12
            ):
                raise ValueError("Six axis-aligned box faces required")
            if any(len(np.unique(vertices[:, axis])) != 2 for axis in range(3)):
                raise ValueError("The macro vertices do not form an axis-aligned box")
        for q in arrays["norm_orders"]:
            if not np.array_equal(
                arrays[f"q{q}_material_{cell}"],
                np.broadcast_to(np.eye(3), arrays[f"q{q}_material_{cell}"].shape),
            ):
                raise ValueError("This exact-space argument requires identity permeability")
        center = vertices.mean(axis=0)
        scale = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        xi = (arrays[f"nodes_{cell}"] - center) / scale
        quadratic = np.sum(xi**2, axis=1, keepdims=True) if kind == "tetra" else xi**2
        polynomial = np.column_stack((np.ones(len(xi)), xi, quadratic))
        moments = restore(arrays, f"executed_moments_{cell}")
        actual = restore(arrays, f"executed_reconstruction_{cell}")
        matrix = moments.T @ polynomial
        condition = float(np.linalg.cond(np.asarray(matrix, dtype=float)))
        inverse = np.linalg.solve(np.asarray(matrix, dtype=float), np.eye(expected))
        ideal = polynomial @ inverse
        discrepancy = float(np.max(np.abs(actual - ideal)) / np.max(np.abs(actual)))
        moment_defect = float(np.max(np.abs(matrix @ inverse - np.eye(expected))))
        if not np.isfinite(condition) or discrepancy > 1e-10 or moment_defect > 1e-10:
            raise ArithmeticError(
                "Actual local lift does not satisfy the selected ideal-space audit"
            )
        row = {
            "cell": cell,
            "dimension": expected,
            "lift_relative_linf": discrepancy,
            "moment_inverse_defect": moment_defect,
            "moment_matrix_condition": condition,
        }
        rows.append(row)
        for key in maxima:
            maxima[key] = max(maxima[key], row[key])
    return {
        "definition": "U^{0,0}: constant divergence and original-face normal derivatives",
        "primary_source": "10.1051/m2an/2021082, Equation(2.19)",
        "geometry": kind,
        "permeability": "identity",
        "dimension": expected,
        "relative_tolerance": 1e-10,
        "executed_lift_replaced": False,
        "uniform_inf_sup_verified": False,
        "general_polyhedral_exactness_claimed": False,
        "macro_cells": len(rows),
        "maxima": maxima,
        "rows": rows,
        "accepted": True,
    }
