"""Physical scalar trajectory archives with executed bases and macro-side identity.

These archives replay observed fields. They do not restore an integrator or
reconstruct fields from coarse coordinates. Continuous nodal coefficients remain
independent between macroelements; skeletal coefficients retain their own
physical normal or Robin convention and executed polynomial evaluation matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from examples.archive_precision import precision_fields, restore_precision
from examples.local_response_cache import array_identity
from pymhm.elements import p1_geometry, triangle_quadrature
from pymhm.lagrange import nodal_space, reference_basis
from pymhm.mesh import SkeletonSpace, TriangleMesh, positive_int
from pymhm.refinement import validate_submesh


def scalar_geometry(
    skeleton: SkeletonSpace,
    meshes: Sequence[TriangleMesh],
    degree: int,
    *,
    order: int = 6,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Persist nodal maps and executed reference derivatives on actual fine meshes.

    The positive Duffy rule uses ``order`` points per coordinate. Its order
    must integrate the squared Pk field; nonpolynomial physical integrands
    still require independent quadrature controls. The stored affine gradient
    maps and reference matrices, rather than recomputed bases, define replay.
    """
    degree, order = positive_int(degree, "degree"), positive_int(order, "order")
    if order < degree + 1:
        raise ValueError("scalar archive quadrature must integrate the squared local field")
    if len(meshes) != len(skeleton.mesh.cells):
        raise ValueError("one fine mesh per macroelement is required")
    for cell, mesh in enumerate(meshes):
        validate_submesh(skeleton.mesh, cell, mesh)
    bary, weights = triangle_quadrature(order)
    values, derivative, second = reference_basis(degree, bary)
    spaces = [nodal_space(mesh, degree) for mesh in meshes]
    point_offsets = np.r_[0, np.cumsum([len(mesh.points) for mesh in meshes])]
    nodal_offsets = np.r_[0, np.cumsum([len(nodes) for _, nodes in spaces])]
    arrays = {
        "macro_points": skeleton.mesh.points,
        "macro_cells": skeleton.mesh.cells,
        "macro_faces": skeleton.mesh.faces,
        "macro_cell_faces": skeleton.mesh.cell_faces,
        "macro_normals": skeleton.mesh.normals,
        "macro_signs": skeleton.mesh.signs,
        "fine_points": np.concatenate([mesh.points for mesh in meshes]),
        "fine_cells": np.concatenate(
            [mesh.cells + point_offsets[cell] for cell, mesh in enumerate(meshes)]
        ),
        "fine_cell_macro": np.repeat(np.arange(len(meshes)), [len(m.cells) for m in meshes]),
        "fine_cell_offsets": np.r_[0, np.cumsum([len(m.cells) for m in meshes])],
        "fine_areas": np.concatenate([mesh.areas for mesh in meshes]),
        "nodal_coordinates": np.concatenate([nodes for _, nodes in spaces]),
        "nodal_offsets": nodal_offsets,
        "cell_nodal_dofs": np.concatenate(
            [dofs + nodal_offsets[cell] for cell, (dofs, _) in enumerate(spaces)]
        ),
        "quadrature_barycentric": bary,
        "quadrature_weights": weights,
        "reference_values": values,
        "reference_derivatives": derivative,
        "reference_second_derivatives": second,
        "affine_barycentric_gradients": np.concatenate([p1_geometry(m)[0] for m in meshes]),
    }
    record = {
        "degree": degree,
        "quadrature_order": order,
        "macro_cells": len(meshes),
        "fine_cells": len(arrays["fine_cells"]),
        "nodal_count": int(nodal_offsets[-1]),
        "coefficient_convention": "Physical continuous nodal Pk values within each macroelement",
        "interface_convention": "Independent macro sides; no averaging or interface smoothing",
        "basis_sha256": {key: array_identity(value) for key, value in arrays.items()},
    }
    return arrays, record


def uniform_trace_geometry(
    skeleton: SkeletonSpace, *, convention: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Archive an executed uniform scalar trace basis and oriented macro injection.

    ``convention`` must declare what the coefficients represent, for example
    Darcy normal flux or the transport half-advection Robin multiplier.
    Quadrature samples are separate within each skeletal subface; repeated
    physical coordinates at distinct macro sides are not merged.
    """
    if skeleton.components != 1 or not convention.strip():
        raise ValueError("a scalar skeleton and explicit multiplier convention are required")
    face = skeleton.faces[0]
    if any(space != face for space in skeleton.faces):
        raise ValueError("trajectory trace archive requires uniform face spaces")
    parameter, weights = face.quadrature(max(face.degrees) + 2)
    basis = face.evaluate(parameter)
    arrays = {
        "face_breaks": np.asarray(face.breaks),
        "face_degrees": np.asarray(face.degrees),
        "face_parameters": parameter,
        "face_weights": weights,
        "face_basis_values": basis,
        "face_dofs": np.stack([skeleton.dofs(i) for i in range(len(skeleton.faces))]),
        "macro_faces": skeleton.mesh.faces,
        "macro_cell_faces": skeleton.mesh.cell_faces,
        "macro_normals": skeleton.mesh.normals,
        "macro_signs": skeleton.mesh.signs,
        "macro_lengths": skeleton.mesh.lengths,
    }
    record = {
        "convention": convention,
        "trace_size": skeleton.size,
        "face_width": face.size,
        "orientation": "Global normal points out of first adjacent macro; signs select macro side",
        "basis_sha256": {key: array_identity(value) for key, value in arrays.items()},
    }
    return arrays, record


def transport_trace_geometry(
    skeleton: SkeletonSpace,
    *,
    diffusive_faces: Sequence[int],
    strong_faces: Sequence[int],
    numerical_normal_velocity: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Declare interior Robin, exterior diffusive and removed strong-inflow traces.

    These exterior conditions partition the boundary. Zero coefficients on a
    removed strong trace do not prescribe zero physical transport flux there.
    On natural faces the multiplier is the diffusive flux, rather than the
    interior half-advection Robin multiplier. The caller declares the numerical
    normal velocity; it need not equal the raw volume field's normal trace.
    """
    diffusive, strong = set(diffusive_faces), set(strong_faces)
    if (
        diffusive & strong
        or diffusive | strong != set(skeleton.mesh.boundary_faces)
        or not numerical_normal_velocity.strip()
    ):
        raise ValueError("diffusive and strong faces must partition the exterior boundary")
    arrays, record = uniform_trace_geometry(
        skeleton,
        convention=(
            "Interior half-advection Robin multiplier; exterior natural diffusive flux; "
            "removed strong-inflow trace"
        ),
    )
    kinds = np.zeros(len(skeleton.faces), dtype=np.int64)
    kinds[list(diffusive)] = 1
    kinds[list(strong)] = 2
    arrays["face_boundary_kind"] = kinds
    record["boundary_kind"] = {
        "0": "Interior -D grad(u).n+numerical_normal_velocity*u/2",
        "1": "Exterior natural (-D grad(u)).n",
        "2": "Removed strong boundary trace; no prescribed physical flux",
    }
    record["numerical_normal_velocity"] = numerical_normal_velocity
    record["basis_sha256"]["face_boundary_kind"] = array_identity(kinds)
    return arrays, record


def coefficient_arrays(
    name: str, fields: Sequence[np.ndarray], geometry: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Store executed physical nodal fields in portable high/remainder components.

    The archive represents actual fields, including any executed arithmetic
    corrections. It does not persist arbitrary coarse amplitudes without their
    basis, or substitute skeletal multipliers for the physical vector field.
    """
    lengths = np.diff(geometry["nodal_offsets"])
    if len(fields) != len(lengths) or any(
        value.shape != (length,) for value, length in zip(fields, lengths, strict=True)
    ):
        raise ValueError("physical coefficient shapes differ from the archived nodal maps")
    if len({value.dtype for value in fields}) != 1:
        raise ValueError("physical fields require one common executed dtype")
    coefficients = np.concatenate(fields)
    return vector_coefficients(
        name,
        coefficients,
        convention="Physical continuous nodal Pk values within each macroelement",
    )


def vector_coefficients(
    name: str, coefficients: np.ndarray, *, convention: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Archive a real vector with its explicit meaning and portable residual digits.

    The caller must persist its executed evaluation matrix and orientation or
    nodal assignment map separately. A coefficient name does not declare the
    vector to be a physical flux; numerical multipliers keep their own meaning.
    """
    if not name.isidentifier():
        raise ValueError("a coefficient name must be an identifier")
    if coefficients.ndim != 1 or not convention.strip():
        raise ValueError("a coefficient vector and explicit convention are required")
    if coefficients.dtype.kind != "f":
        raise ValueError("physical scalar coefficients must have a real floating dtype")
    arrays = precision_fields(name, coefficients)
    record = {
        "name": name,
        "executed_dtype": coefficients.dtype.str,
        "executed_mantissa_bits": np.finfo(coefficients.dtype).nmant,
        "coefficient_convention": convention,
        "coefficient_sha256": {key: array_identity(value) for key, value in arrays.items()},
    }
    return arrays, record


def validate_arrays(arrays: dict[str, np.ndarray], digests: dict[str, str]) -> None:
    """Require the actual basis, geometry or coefficient components before replay."""
    for key, digest in digests.items():
        if array_identity(arrays[key]) != digest:
            raise ValueError(f"trajectory array digest differs: {key}")


def replay_scalar(
    geometry: dict[str, np.ndarray],
    geometry_record: dict[str, Any],
    arrays: dict[str, np.ndarray],
    coefficient_record: dict[str, Any],
    *,
    first_cell: int = 0,
    last_cell: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a physical scalar and gradient using only archived matrices/maps.

    Cell slicing bounds temporary quadrature arrays for large trajectories.
    Portable coefficient components are accumulated in native widest real
    precision; their separate components remain authoritative on other hosts.
    """
    validate_arrays(geometry, geometry_record["basis_sha256"])
    validate_arrays(arrays, coefficient_record["coefficient_sha256"])
    count = len(geometry["fine_cells"])
    first_cell = positive_int(first_cell, "first_cell", 0)
    last_cell = count if last_cell is None else positive_int(last_cell, "last_cell")
    if not 0 <= first_cell < last_cell <= count:
        raise ValueError("trajectory replay cell interval is invalid")
    name = coefficient_record["name"]
    coefficients = restore_precision(
        arrays[name], arrays[f"{name}_correction"], arrays[f"{name}_tail"]
    )
    dofs = geometry["cell_nodal_dofs"][first_cell:last_cell]
    nodal = coefficients[dofs]
    gradient_basis = np.einsum(
        "qib,tba->tqia",
        geometry["reference_derivatives"],
        geometry["affine_barycentric_gradients"][first_cell:last_cell],
    )
    return nodal @ geometry["reference_values"].T, np.einsum("ti,tqia->tqa", nodal, gradient_basis)


def replay_trace(
    geometry: dict[str, np.ndarray],
    record: dict[str, Any],
    coefficients: np.ndarray,
) -> np.ndarray:
    """Replay oriented multiplier samples, preserving its declared physical meaning."""
    validate_arrays(geometry, record["basis_sha256"])
    if coefficients.shape != (record["trace_size"],) or not np.isfinite(coefficients).all():
        raise ValueError("trace coefficient vector differs from the archived skeleton")
    return coefficients[geometry["face_dofs"]] @ geometry["face_basis_values"].T
