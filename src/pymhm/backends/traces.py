"""UFL pairings in executed edge/triangle bases with explicit trace transport.

Existing FaceSpace tabulation defines every polynomial. This adapter represents
those same polynomials as UFL expressions on aligned native exterior facets.
It never interpolates them into the local volume unknown's function space.
"""

from __future__ import annotations

from collections.abc import Callable
from math import factorial
from typing import Any, Literal, cast

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.backends.forms import _require
from pymhm.backends.spaces import NativeSpace
from pymhm.core.equations import Equation, LinearForms, compile_form
from pymhm.core.spaces import ComponentTraceSpace
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.pressure_2d import PressureTraceSpace
from pymhm.fem.traces.pressure_3d import PressureTraceSpace3D
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.fem.vector.curl import TangentialTraceSpace


def _scalar_trace(space: Any) -> Any:
    """Select the scalar basis owner while preserving the declared coefficient layout."""
    base = space.base if isinstance(space, (ComponentTraceSpace, TangentialTraceSpace)) else space
    if not isinstance(
        base, (SkeletonSpace, PressureTraceSpace, TriangularSkeleton, PressureTraceSpace3D)
    ):
        raise TypeError("custom interface must declare a native trace-pairings capability")
    return base


def _face_dofs(space: Any, face: int) -> Any:
    """Obtain canonical face coordinates, including globally shared pressure nodes."""
    owner = getattr(space, "dofs", None)
    return owner(face) if callable(owner) else space.face_dofs[face]


def _local_positions(space: Any, cell: int, face: int) -> Any:
    """Map a face into the cell's actual coordinate order without assuming disjoint faces."""
    indices = {int(dof): index for index, dof in enumerate(space.cell_dofs(cell))}
    return np.asarray([indices[int(dof)] for dof in _face_dofs(space, face)])


def _triangle_parts(space: Any, face: int) -> Any:
    """Return a triangular basis owner's canonical barycentric subfaces."""
    return (
        space.face_partition(face)
        if isinstance(space, TriangularSkeleton)
        else space.partitions[face]
    )


def _triangle_degree(space: Any, face: int) -> int:
    """Read the declared degree, without imposing a volume-space compatibility rule."""
    return int(space.degrees[face]) if isinstance(space, TriangularSkeleton) else int(space.degree)


def _trace_value(space: Any, face: int, phi: Any, component: int, ufl: Any) -> Any:
    """Lift a scalar basis mode into its declared Cartesian or tangential frame."""
    components = getattr(space, "components", 1)
    if isinstance(space, TangentialTraceSpace) and components == 2:
        return ufl.as_vector([float(value) * phi for value in space.frames[face][:, component]])
    return (
        phi
        if components == 1
        else ufl.as_vector([phi if index == component else 0 for index in range(components)])
    )


def _triangle_segments(context: Any, native: NativeSpace, space: Any) -> Any:
    """Associate native exterior triangles with exactly one declared physical subface."""
    mesh, macro = native.mesh, context.macro
    if mesh.topology.dim != 3:
        raise ValueError("triangular trace pairings require three-dimensional local meshes")
    backend = _require("dolfinx.mesh")
    mesh.topology.create_connectivity(2, 3)
    exterior = backend.exterior_facet_indices(mesh.topology)
    vertices = backend.entities_to_geometry(mesh, 2, exterior, False)
    points = mesh.geometry.x[vertices, :3]
    indices: list[int] = []
    tags: list[int] = []
    segments: list[tuple[int, int, int]] = []
    visited: set[int] = set()
    for raw in macro.cell_faces[context.cell]:
        face = int(raw)
        coordinates = macro.points[macro.faces[face]]
        edges = coordinates[1:] - coordinates[0]
        inverse = np.linalg.pinv(edges)
        parameter = (points - coordinates[0]) @ inverse
        projected = coordinates[0] + parameter @ edges
        scale = float(np.max(np.linalg.norm(edges, axis=1)))
        tolerance = 256 * np.finfo(float).eps * max(scale, float(np.max(abs(points))))
        bary = np.concatenate((1 - parameter.sum(axis=-1, keepdims=True), parameter), axis=-1)
        on_face = np.all(np.linalg.norm(points - projected, axis=-1) <= tolerance, axis=1) & np.all(
            bary >= -tolerance / scale, axis=(1, 2)
        )
        selected = np.flatnonzero(on_face)
        if not len(selected):
            raise ValueError("the local mesh must cover each declared macroface")
        parts = _triangle_parts(space, face)
        transformed = np.einsum("qvi,sij->sqvj", bary[selected], np.linalg.inv(parts))
        candidates = np.min(transformed.mean(axis=2), axis=-1) >= -tolerance / scale
        if np.any(candidates.sum(axis=0) != 1):
            raise ValueError("local boundary facets must lie in one trace subtriangle")
        owners = np.argmax(candidates, axis=0)
        for segment in range(len(parts)):
            members = np.flatnonzero(owners == segment)
            if not len(members):
                raise ValueError("local boundary facets must resolve every trace subtriangle")
            if np.min(transformed[segment, members]) < -tolerance / scale:
                raise ValueError("local boundary facets must align with trace subtriangles")
            facets = exterior[selected[members]]
            if any(int(facet) in visited for facet in facets):
                raise ValueError("a boundary facet cannot belong to multiple macrofaces")
            visited.update(int(facet) for facet in facets)
            tag = len(segments) + 1
            indices.extend(int(facet) for facet in facets)
            tags.extend([tag] * len(facets))
            segments.append((face, segment, tag))
    if len(visited) != len(exterior):
        raise ValueError("the macrocell faces must cover the local mesh boundary")
    order = np.argsort(indices)
    markers = backend.meshtags(
        mesh, 2, np.asarray(indices, dtype=np.int32)[order], np.asarray(tags, dtype=np.int32)[order]
    )
    return markers, tuple(segments)


def _triangle_modes(native: Any, space: Any, face: int, segment: int) -> Any:
    """Represent the owner's executed polynomial basis on one physical subtriangle.

    Interior unisolvent Bernstein lattice samples identify the existing basis.
    Only its polynomial representation changes; global coefficient vectors and
    shared C0 degrees of freedom retain their original basis and numbering.
    """
    from pymhm.fem.reference import bernstein_tabulation

    degree = _triangle_degree(space, face)
    powers = tuple(
        (i, j, degree - i - j) for i in range(degree, -1, -1) for j in range(degree - i, -1, -1)
    )
    nodes = (np.asarray(powers, dtype=float) + 1) / (degree + 3)
    part = _triangle_parts(space, face)[segment]
    samples = space.evaluate(face, nodes @ part)
    values = bernstein_tabulation(nodes[:, 1:], powers)[0]
    coefficients = np.linalg.solve(values, samples)
    coordinates = part @ space.mesh.points[space.mesh.faces[face]]
    edges = coordinates[1:] - coordinates[0]
    inverse = np.linalg.pinv(edges)
    fem, ufl = _require("dolfinx.fem"), _require("ufl")
    geometry = fem.Constant(native, np.r_[coordinates[0], inverse.ravel()])
    x = ufl.SpatialCoordinate(native)
    b1 = sum((x[axis] - geometry[axis]) * geometry[3 + 2 * axis] for axis in range(3))
    b2 = sum((x[axis] - geometry[axis]) * geometry[4 + 2 * axis] for axis in range(3))
    bary = (1 - b1 - b2, b1, b2)
    basis = [
        factorial(degree)
        / np.prod([factorial(p) for p in power])
        * np.prod([b**p for b, p in zip(bary, power, strict=True)])
        for power in powers
    ]
    for mode in np.flatnonzero(np.any(samples != 0, axis=0)):
        data = fem.Constant(native, np.array(coefficients[:, mode], copy=True))
        yield int(mode), sum(data[index] * phi for index, phi in enumerate(basis))


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
    Piecewise edge/triangle polynomials require native facets aligned to their
    partitions. Shared pressure nodes, facewise C0 modes and tangential frames
    retain the basis declared by the interface owner. Arbitrary custom traces
    may supply their own capability explicitly.
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
    skeleton: Any = getattr(selected, "space", None)
    base = _scalar_trace(skeleton)
    if skeleton.mesh is not context.macro:
        raise ValueError("interface and hierarchy must declare the same macro mesh")
    triangular = isinstance(base, (TriangularSkeleton, PressureTraceSpace3D))
    markers, segments = (
        _triangle_segments(context, native, base)
        if triangular
        else _face_segments(context, native, base)
    )
    ufl = _require("ufl")
    fem = _require("dolfinx.fem")
    coordinate = ufl.SpatialCoordinate(native.mesh)
    ds = ufl.Measure("ds", domain=native.mesh, subdomain_data=markers)
    components = getattr(skeleton, "components", 1)
    width = len(skeleton.cell_dofs(context.cell))
    forms: list[Any] = [0] * width
    for face, segment, tag in segments:
        positions = _local_positions(skeleton, context.cell, face)
        if triangular:
            for mode, phi in _triangle_modes(native.mesh, base, face, segment):
                for component in range(components):
                    index = positions[mode * components + component]
                    trace = _trace_value(skeleton, face, phi, component, ufl)
                    forms[index] = forms[index] + expression(trace, ds(tag))
            continue
        space = base.faces[face]
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
            for component in range(components):
                trace = _trace_value(skeleton, face, phi, component, ufl)
                index = positions[int(mode) * components + component]
                forms[index] = forms[index] + expression(trace, ds(tag))
    return LinearForms(tuple(forms), axis)


def _interface_mesh(skeleton: Any) -> tuple[Any, tuple[tuple[int, int], ...]]:
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
        x=np.asarray(points),
        e=domain,
    )
    return native, tuple(segments)


def _interface_transport(
    skeleton: Any, native: Any, space: Any, segments: tuple[tuple[int, int], ...]
) -> Any:
    """Embed the executed canonical face basis into native discontinuous coefficients."""
    base = _scalar_trace(skeleton)
    components = getattr(skeleton, "components", 1)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    reference = np.asarray(space.element.basix_element.points)[:, 0]
    for cell, original in enumerate(native.topology.original_cell_index):
        face, segment = segments[int(original)]
        declaration = base.faces[face]
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
        block = np.kron(local, np.eye(components))
        dofs = np.asarray(space.dofmap.cell_dofs(cell), dtype=np.int64)
        dofs = (dofs[:, None] * components + np.arange(components)).ravel()
        selected = _face_dofs(skeleton, face)
        ii, jj = np.nonzero(block)
        rows.extend(dofs[ii].tolist())
        columns.extend(selected[jj].tolist())
        values.extend(block[ii, jj].tolist())
    width = int(space.dofmap.index_map.size_local * space.dofmap.index_map_bs)
    return sparse.coo_matrix((values, (rows, columns)), shape=(width, skeleton.size)).tocsc()


def _triangular_interface_mesh(base: Any) -> tuple[Any, tuple[tuple[int, int], ...]]:
    """Create independent embedded subtriangles for volume integrals on the skeleton.

    Independence avoids treating junctions of more than two macrofaces as a
    manifold. Shared coefficient constraints enter through the embedding;
    cross-face facet terms remain explicit global algebraic forms.
    """
    points: list[Any] = []
    cells: list[tuple[int, int, int]] = []
    segments: list[tuple[int, int]] = []
    for face in range(len(base.mesh.faces)):
        for segment, part in enumerate(_triangle_parts(base, face)):
            start = len(points)
            points.extend(part @ base.mesh.points[base.mesh.faces[face]])
            cells.append((start, start + 1, start + 2))
            segments.append((face, segment))
    domain = _require("ufl").Mesh(
        _require("basix.ufl").element("Lagrange", "triangle", 1, shape=(3,))
    )
    native = _require("dolfinx.mesh").create_mesh(
        _require("mpi4py.MPI").COMM_SELF,
        np.asarray(cells, dtype=np.int64),
        x=np.asarray(points),
        e=domain,
    )
    return native, tuple(segments)


def _triangular_interface_transport(skeleton: Any, native: Any, space: Any, segments: Any) -> Any:
    """Evaluate the declared face basis at native DG nodes in physical coordinates."""
    base = _scalar_trace(skeleton)
    components = getattr(skeleton, "components", 1)
    tangent = isinstance(skeleton, TangentialTraceSpace) and components == 2
    value_size = 3 if tangent else components
    reference = np.asarray(space.element.basix_element.points)
    bary = np.column_stack((1 - reference.sum(axis=1), reference))
    rows, columns, values = [], [], []
    for cell, original in enumerate(native.topology.original_cell_index):
        face, _ = segments[int(original)]
        vertices = native.geometry.x[native.geometry.dofmap[cell], :3]
        points = bary @ vertices
        coordinates = base.mesh.points[base.mesh.faces[face]]
        parameter = (points - coordinates[0]) @ np.linalg.pinv(coordinates[1:] - coordinates[0])
        canonical = np.column_stack((1 - parameter.sum(axis=1), parameter))
        local = base.evaluate(face, canonical)
        frame = skeleton.frames[face] if tangent else np.eye(components)
        block = np.kron(local, frame)
        dofs = np.asarray(space.dofmap.cell_dofs(cell), dtype=np.int64)
        dofs = (dofs[:, None] * value_size + np.arange(value_size)).ravel()
        selected = _face_dofs(skeleton, face)
        ii, jj = np.nonzero(block)
        rows.extend(dofs[ii].tolist())
        columns.extend(selected[jj].tolist())
        values.extend(block[ii, jj].tolist())
    width = int(space.dofmap.index_map.size_local * space.dofmap.index_map_bs)
    return sparse.coo_matrix((values, (rows, columns)), shape=(width, skeleton.size)).tocsc()


def interface_equation(context: Any, builder: Callable[[Any, Any, Any], Any]) -> Equation:
    """Compile a user-defined global UFL form in declared edge/triangle coordinates.

    ``builder(lambda_, mu, dx)`` returns ``Equation(a, L)`` or ``(a, L)`` on
    an embedded interval or triangular mesh. Its discontinuous space contains
    the executed face basis, including independent degrees, facewise C0 modes,
    shared pressure nodes and Cartesian/tangential components. An embedding
    transports both forms back to
    the declared global layout. Retained coordinates receive zero added terms.
    Physical normal/traction signs remain those written by the user. This
    adapter supplies no cross-mesh volume/interface UFL argument coupling.
    """
    if not callable(builder):
        raise TypeError("global interface form builder must be callable")
    skeleton = getattr(context.interface, "space", None)
    try:
        base = _scalar_trace(skeleton)
    except TypeError as error:
        raise TypeError("custom interface must declare a global UFL equation capability") from error
    triangular = isinstance(base, (TriangularSkeleton, PressureTraceSpace3D))
    native, segments = _triangular_interface_mesh(base) if triangular else _interface_mesh(base)
    basix, ufl = _require("basix"), _require("ufl")
    degree = (
        max(_triangle_degree(base, face) for face in range(len(base.mesh.faces)))
        if triangular
        else max(degree for face in base.faces for degree in face.degrees)
    )
    components = getattr(skeleton, "components", 1)
    if isinstance(skeleton, TangentialTraceSpace) and components == 2:
        components = 3
    element = _require("basix.ufl").element(
        "Lagrange",
        "triangle" if triangular else "interval",
        degree,
        discontinuous=True,
        lagrange_variant=basix.LagrangeVariant.equispaced,
        shape=() if components == 1 else (components,),
    )
    space = _require("dolfinx.fem").functionspace(native, element)
    embedding = (
        _triangular_interface_transport(skeleton, native, space, segments)
        if triangular
        else _interface_transport(skeleton, native, space, segments)
    )
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
