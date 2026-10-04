"""Capture and replay one-sided analytical MsHHO3D section sampling tables."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from examples.archive_precision import precision_fields, restore_precision
from pymhm.fem.scalar.tetrahedron import tetra_basis, tetra_nodal_space
from pymhm.materials.evaluation import tensor_values_3d
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.methods.hho_3d import MsHHO3DSolution


def capture_section(
    solution: MsHHO3DSolution, refinement: int = 6, height: float = 0.37
) -> dict[str, np.ndarray]:
    """Capture actual sampling tables for the sine unit-cube case after its field solve.

    Every fine-cell fan owns its sampling vertices. Original macro section
    intersections remain separate overlays on analytical/numerical/error panels.
    Physical coefficient digits remain portable before final plotting casts.
    """
    from examples.sample_core_sections import cut_polygon, section_samples

    if type(refinement) is not int or refinement < 1 or height != 0.37:
        raise ValueError("Positive sampling refinement at the declared z=.37 section required")
    meshes = tuple(local.mesh for local in solution.local)
    dofs = tuple(tetra_nodal_space(mesh, 2)[0] for mesh in meshes)
    homogeneous = tuple(
        np.concatenate((np.ones((len(mesh.cells), 4, 1)), mesh.points[mesh.cells]), axis=2)
        for mesh in meshes
    )
    inverses = tuple(np.linalg.inv(value) for value in homogeneous)
    tables: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "barycentric",
            "cardinal_values",
            "cardinal_derivatives",
            "barycentric_gradients",
            "material",
            "nodal_dofs",
        )
    }

    def evaluate(macro: int, cell: int, points: np.ndarray) -> np.ndarray:
        inverse = inverses[macro][cell]
        bary = np.column_stack((np.ones(len(points)), points)) @ inverse
        values, derivatives = tetra_basis(2, bary)
        gradient_map = inverse[1:].T
        material = tensor_values_3d(solution.permeability, points)
        coefficients = solution.pressure[macro][dofs[macro][cell]]
        pressure = np.einsum("qi,i->q", values, coefficients, optimize=False)
        gradient = np.einsum(
            "i,qia,ab->qb", coefficients, derivatives, gradient_map, optimize=False
        )
        flux = -np.einsum("qab,qb->qa", material, gradient, optimize=False)
        entries = {
            "barycentric": bary,
            "cardinal_values": values,
            "cardinal_derivatives": derivatives,
            "barycentric_gradients": np.broadcast_to(gradient_map, (len(points), 4, 3)),
            "material": material,
            "nodal_dofs": np.broadcast_to(dofs[macro][cell], (len(points), 10)),
        }
        for key, value in entries.items():
            tables[key].append(value)
        return np.column_stack((pressure, flux))

    sampled = section_samples(meshes, evaluate, refinement)
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    macro = solution.skeleton.mesh
    for ids in macro.cells:
        if isinstance(macro, PolyhedralMesh):
            ids = np.unique(np.concatenate([macro.faces[face] for face in ids]))
        cut = cut_polygon(macro.points[ids], height)
        if len(cut):
            edges.extend(zip(cut[:, :2], np.roll(cut[:, :2], -1, axis=0), strict=True))
    arrays = {f"display_{name}": np.concatenate(values) for name, values in tables.items()}
    arrays.update({f"display_{name}": value for name, value in sampled.items() if name != "actual"})
    arrays.update(precision_fields("display_actual", sampled["actual"]))
    arrays["display_macro_edges"] = np.asarray(edges)
    arrays["display_refinement"] = np.asarray(refinement, dtype=np.int64)
    arrays["display_height"] = np.asarray(height)
    return arrays


def validate_section(arrays: Mapping[str, np.ndarray]) -> None:
    """Check saved sampling geometry and all P2 polynomial identities without a new basis."""
    if "display_points" not in arrays:
        return
    points, owners, cells = (
        arrays["display_points"],
        arrays["display_owners"],
        arrays["display_cells"],
    )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or owners.shape != (len(points), 2)
        or owners.dtype.kind not in "iu"
        or np.any(owners < 0)
        or np.any(owners[:, 0] >= int(arrays["local_count"]))
        or cells.ndim != 2
        or cells.shape[1] != 3
        or cells.dtype.kind not in "iu"
        or np.any(cells < 0)
        or np.any(cells >= len(points))
        or not np.allclose(points[:, 2], 0.37, rtol=0, atol=1e-13)
    ):
        raise ValueError("Independent one-sided section geometry differs")
    bary = arrays["display_barycentric"]
    basis = arrays["display_cardinal_values"]
    derivative = arrays["display_cardinal_derivatives"]
    reference = arrays["reference_nodal_barycentric"][:, 1:]
    if (
        bary.shape != (len(points), 4)
        or basis.shape != (len(points), 10)
        or derivative.shape != (len(points), 10, 4)
    ):
        raise ValueError("Executed display cardinal dimensions differ")
    for a in range(3):
        for b in range(3 - a):
            for c in range(3 - a - b):
                powers = np.asarray([a, b, c])
                coefficients = np.prod(reference**powers, axis=1)
                target = np.prod(bary[:, 1:] ** powers, axis=1)
                if not np.allclose(basis @ coefficients, target, rtol=0, atol=1e-10):
                    raise ValueError("Display basis fails its declared P2 polynomial identity")
                for axis in range(3):
                    reduced = powers.copy()
                    reduced[axis] = max(0, reduced[axis] - 1)
                    target_gradient = powers[axis] * np.prod(bary[:, 1:] ** reduced, axis=1)
                    if not np.allclose(
                        (derivative[..., axis + 1] - derivative[..., 0]) @ coefficients,
                        target_gradient,
                        rtol=0,
                        atol=1e-10,
                    ):
                        raise ValueError("Display derivative fails its declared P2 identity")
    for macro in np.unique(owners[:, 0]):
        selected = owners[:, 0] == macro
        ids = owners[selected, 1]
        if np.any(ids >= len(arrays[f"cells_{macro}"])):
            raise ValueError("Display fine-cell owner differs")
        dofs = arrays[f"nodal_dofs_{macro}"][ids]
        if not np.array_equal(arrays["display_nodal_dofs"][selected], dofs):
            raise ValueError("Display nodal injection differs from the executed field")
        vertices = arrays[f"points_{macro}"][arrays[f"cells_{macro}"][ids]]
        physical = np.einsum("qi,qia->qa", bary[selected], vertices)
        if not np.allclose(physical, points[selected], rtol=0, atol=1e-12):
            raise ValueError("Display physical barycentric map differs")
        if not np.allclose(
            arrays["display_barycentric_gradients"][selected],
            arrays[f"barycentric_gradients_{macro}"][ids],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("Display physical gradient map differs")
    if np.any(owners[cells] != owners[cells[:, :1]]):
        raise ValueError("Display triangles silently merge independent fine-cell interfaces")


def replay_section(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluate the saved actual basis and coefficients without new solves or tabulation."""
    validate_section(arrays)
    if "display_points" not in arrays:
        raise ValueError("Producer section tables are missing; acquire the current finest case")
    owners = arrays["display_owners"]
    coefficients = np.zeros((len(owners), 10), dtype=np.longdouble)
    for macro in np.unique(owners[:, 0]):
        selected = owners[:, 0] == macro
        pressure = restore_precision(
            arrays[f"pressure_{macro}"],
            arrays[f"pressure_{macro}_correction"],
            arrays[f"pressure_{macro}_tail"],
        )
        coefficients[selected] = pressure[arrays["display_nodal_dofs"][selected]]
    pressure = np.einsum(
        "qi,qi->q", arrays["display_cardinal_values"], coefficients, optimize=False
    )
    gradient = np.einsum(
        "qi,qia,qab->qb",
        coefficients,
        arrays["display_cardinal_derivatives"],
        arrays["display_barycentric_gradients"],
        optimize=False,
    )
    flux = -np.einsum("qab,qb->qa", arrays["display_material"], gradient, optimize=False)
    actual = np.column_stack((pressure, flux))
    stored = restore_precision(
        arrays["display_actual"], arrays["display_actual_correction"], arrays["display_actual_tail"]
    )
    if not np.allclose(actual, stored, rtol=1e-14, atol=1e-14):
        raise ValueError("Display replay differs from the actual producer field")
    return {
        "points": arrays["display_points"],
        "cells": arrays["display_cells"],
        "owners": owners,
        "actual": actual,
        "exact": arrays["display_exact"],
        "macro_edges": arrays["display_macro_edges"],
    }
