"""UFL pairings in existing planar face bases without user-managed orientations.

Existing FaceSpace tabulation defines every polynomial. This adapter represents
those same polynomials as UFL expressions on aligned native exterior facets.
It never interpolates them into the local volume unknown's function space.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.backends.forms import _require
from pymhm.backends.spaces import NativeSpace
from pymhm.core.equations import Equation, LinearForms, compile_form
from pymhm.fem.traces.interval import SkeletonSpace


def _polynomial(parameter: Any, nodes: Any, values: Any) -> Any:
    """Represent an existing degree-p tabulation by exact polynomial interpolation.

    Interior Gauss nodes avoid assigning an internal break to its adjacent
    segment. These cardinal products only convert FaceSpace's existing basis
    into UFL; they do not define another finite-element basis or node ordering.
    """
    expression: Any = 0
    for index, value in enumerate(values):
        if np.isscalar(value) and value == 0:
            continue
        cardinal: Any = float(cast(float, value)) if np.isscalar(value) else value
        for other, node in enumerate(nodes):
            if other != index:
                cardinal *= (parameter - float(node)) / float(nodes[index] - node)
        expression += cardinal
    return expression


def _face_segments(context: Any, native: NativeSpace, skeleton: SkeletonSpace) -> Any:
    """Tag native boundary facets by canonical macroface and polynomial segment."""
    mesh = native.mesh
    if mesh.topology.dim != 2:
        raise ValueError("native built-in trace pairings currently require planar local meshes")
    backend = _require("dolfinx.mesh")
    mesh.topology.create_connectivity(1, 2)
    exterior = backend.exterior_facet_indices(mesh.topology)
    vertices = backend.entities_to_geometry(mesh, 1, exterior, False)
    points = mesh.geometry.x[vertices, :2]
    indices: list[int] = []
    tags: list[int] = []
    segments: list[tuple[int, int, int]] = []
    visited: set[int] = set()
    tag = 0
    for face in context.macro.cell_faces[context.cell]:
        start, end = context.macro.points[context.macro.faces[face]]
        tangent = end - start
        squared = float(tangent @ tangent)
        parameter = (points - start) @ tangent / squared
        projected = start + parameter[..., None] * tangent
        tolerance = (
            256
            * np.finfo(float).eps
            * max(np.sqrt(squared), float(np.max(np.abs(points))), float(np.max(np.abs(start))))
        )
        parameter_tolerance = tolerance / np.sqrt(squared)
        on_face = (
            np.all(np.linalg.norm(points - projected, axis=-1) <= tolerance, axis=1)
            & np.all(parameter >= -parameter_tolerance, axis=1)
            & np.all(parameter <= 1 + parameter_tolerance, axis=1)
        )
        selected = np.flatnonzero(on_face)
        if len(selected) == 0:
            raise ValueError("the local mesh must cover each declared macroface")
        space = skeleton.faces[int(face)]
        midpoint = parameter[selected].mean(axis=1)
        owners = np.clip(
            np.searchsorted(space.breaks, midpoint, side="right") - 1, 0, len(space.degrees) - 1
        )
        for segment, (lo, hi) in enumerate(zip(space.breaks[:-1], space.breaks[1:], strict=True)):
            tag += 1
            members = selected[owners == segment]
            if len(members) == 0:
                raise ValueError("local boundary facets must resolve every trace segment")
            if np.any(parameter[members] < lo - parameter_tolerance) or np.any(
                parameter[members] > hi + parameter_tolerance
            ):
                raise ValueError("local boundary facets must align with trace breakpoints")
            facets = exterior[members]
            if any(int(facet) in visited for facet in facets):
                raise ValueError("a boundary facet cannot belong to multiple macrofaces")
            visited.update(int(facet) for facet in facets)
            indices.extend(int(facet) for facet in facets)
            tags.extend([tag] * len(facets))
            segments.append((int(face), segment, tag))
    if len(visited) != len(exterior):
        raise ValueError("the macrocell faces must cover the local mesh boundary")
    order = np.argsort(indices)
    markers = backend.meshtags(
        mesh, 1, np.asarray(indices, dtype=np.int32)[order], np.asarray(tags, dtype=np.int32)[order]
    )
    return markers, tuple(segments)


def trace_pairings(
    context: Any,
    native: NativeSpace,
    expression: Callable[[Any, Any], Any],
    *,
    axis: Literal["columns", "rows"] = "columns",
    interface: Any = None,
) -> LinearForms:
    """Apply a user UFL boundary form to each existing unsigned local trace mode.

    ``expression(phi, ds)`` returns a local linear form. Scalar traces provide
    scalar ``phi``; vector traces provide a component-interleaved vector.
    Trial/test pairings are compiled independently. Context equation binding
    subsequently applies the mesh incidence sign for normal traces exactly once.
    Planar piecewise polynomials require native facets aligned to their breaks;
    arbitrary custom and 3D traces declare their own capability explicitly.
    UFL determines quadrature from the supplied form; callers may set measure
    metadata when coefficients require more accurate integration.

    ``interface`` optionally selects another bound trace basis on the same
    macro mesh, such as a private local conormal space alongside a global
    pressure trace. Its forms remain unsigned; the caller chooses their
    mathematical signs and applies any declared coefficient transport.
    """
    if axis not in {"columns", "rows"}:
        raise ValueError("linear form axis must be columns or rows")
    if not callable(expression):
        raise TypeError("trace expression must be callable")
    selected = context.global_context.interface if interface is None else interface
    skeleton = getattr(selected, "space", None)
    if not isinstance(skeleton, SkeletonSpace):
        raise TypeError("custom interface must declare a native trace-pairings capability")
    if skeleton.mesh is not context.macro:
        raise ValueError("interface and hierarchy must declare the same macro mesh")
    markers, segments = _face_segments(context, native, skeleton)
    ufl = _require("ufl")
    fem = _require("dolfinx.fem")
    coordinate = ufl.SpatialCoordinate(native.mesh)
    ds = ufl.Measure("ds", domain=native.mesh, subdomain_data=markers)
    offsets: dict[int, int] = {}
    width = 0
    for face in context.macro.cell_faces[context.cell]:
        offsets[int(face)] = width
        width += skeleton.faces[int(face)].size * skeleton.components
    forms: list[Any] = [0] * width
    for face, segment, tag in segments:
        space = skeleton.faces[face]
        lo, hi = space.breaks[segment : segment + 2]
        start, end = context.macro.points[context.macro.faces[face]]
        tangent = end - start
        # Constants preserve one kernel signature across translated/scaled
        # macrofaces; their values are packed for each actual local form.
        geometry = fem.Constant(
            native.mesh, np.r_[start, tangent, 1 / float(tangent @ tangent), lo, 1 / (hi - lo)]
        )
        parameter = (
            sum((coordinate[axis] - geometry[axis]) * geometry[axis + 2] for axis in range(2))
            * geometry[4]
        )
        normalized = (parameter - geometry[5]) * geometry[6]
        nodes = (leggauss(space.degrees[segment] + 1)[0] + 1) / 2
        values = space.evaluate(lo + (hi - lo) * nodes)
        for mode in np.flatnonzero(np.any(values != 0, axis=0)):
            coefficients = fem.Constant(native.mesh, np.array(values[:, mode], copy=True))
            phi = _polynomial(
                normalized, nodes, tuple(coefficients[index] for index in range(len(nodes)))
            )
            for component in range(skeleton.components):
                trace = (
                    phi
                    if skeleton.components == 1
                    else ufl.as_vector(
                        [phi if index == component else 0 for index in range(skeleton.components)]
                    )
                )
                index = offsets[face] + int(mode) * skeleton.components + component
                forms[index] = forms[index] + expression(trace, ds(tag))
    return LinearForms(tuple(forms), axis)


def _interface_mesh(skeleton: SkeletonSpace) -> tuple[Any, tuple[tuple[int, int], ...]]:
    """Create embedded interval cells with declared face-segment topology."""
    mesh = skeleton.mesh
    if mesh.points.shape[1] != 2 or mesh.faces.shape[1] != 2:
        raise ValueError("native global interface forms currently require planar edge spaces")
    points = list(mesh.points)
    cells: list[tuple[int, int]] = []
    segments: list[tuple[int, int]] = []
    for face, space in enumerate(skeleton.faces):
        first, last = (int(node) for node in mesh.faces[face])
        start, end = mesh.points[[first, last]]
        ids = [first]
        for cut in space.breaks[1:-1]:
            ids.append(len(points))
            points.append(start + cut * (end - start))
        ids.append(last)
        for segment, (left, right) in enumerate(zip(ids[:-1], ids[1:], strict=True)):
            cells.append((left, right))
            segments.append((face, segment))
    basix = _require("basix.ufl")
    ufl = _require("ufl")
    domain = ufl.Mesh(basix.element("Lagrange", "interval", 1, shape=(2,)))
    native = _require("dolfinx.mesh").create_mesh(
        _require("mpi4py.MPI").COMM_SELF,
        np.asarray(cells, dtype=np.int64),
        np.asarray(points),
        domain,
    )
    return native, tuple(segments)


def _interface_transport(
    skeleton: SkeletonSpace, native: Any, space: Any, segments: tuple[tuple[int, int], ...]
) -> Any:
    """Embed the executed canonical face basis into native discontinuous coefficients."""
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    reference = np.asarray(space.element.basix_element.points)[:, 0]
    for cell, original in enumerate(native.topology.original_cell_index):
        face, segment = segments[int(original)]
        declaration = skeleton.faces[face]
        lo, hi = declaration.breaks[segment : segment + 2]
        start, end = skeleton.mesh.points[skeleton.mesh.faces[face]]
        tangent = end - start
        vertices = native.geometry.x[native.geometry.dofmap[cell], :2]
        physical = (1 - reference[:, None]) * vertices[0] + reference[:, None] * vertices[1]
        parameter = (physical - start) @ tangent / float(tangent @ tangent)
        normalized = (parameter - lo) / (hi - lo)
        nodes = (leggauss(declaration.degrees[segment] + 1)[0] + 1) / 2
        samples = declaration.evaluate(lo + (hi - lo) * nodes)
        local = np.column_stack(
            [
                np.broadcast_to(_polynomial(normalized, nodes, samples[:, mode]), normalized.shape)
                for mode in range(declaration.size)
            ]
        )
        block = np.kron(local, np.eye(skeleton.components))
        dofs = np.asarray(space.dofmap.cell_dofs(cell), dtype=np.int64)
        dofs = (dofs[:, None] * skeleton.components + np.arange(skeleton.components)).ravel()
        selected = skeleton.dofs(face)
        ii, jj = np.nonzero(block)
        rows.extend(dofs[ii].tolist())
        columns.extend(selected[jj].tolist())
        values.extend(block[ii, jj].tolist())
    width = int(space.dofmap.index_map.size_local * space.dofmap.index_map_bs)
    return sparse.coo_matrix((values, (rows, columns)), shape=(width, skeleton.size)).tocsc()


def interface_equation(context: Any, builder: Callable[[Any, Any, Any], Any]) -> Equation:
    """Compile a user-defined global UFL form in canonical planar trace coordinates.

    ``builder(lambda_, mu, dx)`` returns ``Equation(a, L)`` or ``(a, L)`` on
    an embedded interval mesh. Its discontinuous space contains the existing
    FaceSpace basis, including independent degrees and continuous within-face
    partitions. A sparse coefficient embedding transports both forms back to
    the declared global layout. Retained coordinates receive zero added terms.
    Physical normal/traction signs remain those written by the user. This
    adapter supplies no cross-mesh volume/interface UFL argument coupling.
    """
    if not callable(builder):
        raise TypeError("global interface form builder must be callable")
    skeleton = getattr(context.interface, "space", None)
    if not isinstance(skeleton, SkeletonSpace):
        raise TypeError("custom interface must declare a global UFL equation capability")
    native, segments = _interface_mesh(skeleton)
    basix, ufl = _require("basix"), _require("ufl")
    degree = max(degree for face in skeleton.faces for degree in face.degrees)
    element = _require("basix.ufl").element(
        "Lagrange",
        "interval",
        degree,
        discontinuous=True,
        lagrange_variant=basix.LagrangeVariant.equispaced,
        shape=() if skeleton.components == 1 else (skeleton.components,),
    )
    space = _require("dolfinx.fem").functionspace(native, element)
    embedding = _interface_transport(skeleton, native, space, segments)
    supplied = builder(ufl.TrialFunction(space), ufl.TestFunction(space), ufl.dx(domain=native))
    if isinstance(supplied, tuple) and len(supplied) == 2:
        supplied = Equation(*supplied)
    if not isinstance(supplied, Equation):
        raise TypeError("global interface builder must return Equation or (a, L)")
    width = embedding.shape[0]
    operator = compile_form(supplied.a, (width, width))
    load = compile_form(supplied.L, (width,))
    operator = embedding.T @ operator @ embedding
    retained = context.size - context.trace_size
    matrix = sparse.block_diag((operator, sparse.csc_matrix((retained, retained))), format="csc")
    return Equation(matrix, np.r_[embedding.T @ load, np.zeros(retained)])
