"""Owned, lazy native bindings for user-declared finite-element spaces.

Native coefficients remain in DOLFINx order. Conversion to PyMHM's existing
nodal order uses cell incidence and Basix's reference interpolation lattice,
never a global nearest-neighbour search. Moment-based elements, arbitrary
custom bases and mixed fields retain their native representation unless an
explicitly supported nodal component is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import prod
from typing import Any

import numpy as np

from pymhm.backends.forms import _require
from pymhm.core.validation import FloatArray, IntArray, positive_int, real_array


def _cell_geometry(mesh: Any) -> tuple[str, IntArray]:
    """Translate existing portable vertex conventions to Basix cell ordering."""
    from pymhm.meshes.cartesian import CartesianMacroMesh
    from pymhm.meshes.hexahedron import HexMesh
    from pymhm.meshes.tetrahedron import TetraMesh
    from pymhm.meshes.triangle import TriangleMesh

    if isinstance(mesh, TriangleMesh):
        return "triangle", mesh.cells
    if isinstance(mesh, TetraMesh):
        return "tetrahedron", mesh.cells
    if isinstance(mesh, CartesianMacroMesh):
        return "quadrilateral", mesh.cells[:, [0, 1, 3, 2]]
    if isinstance(mesh, HexMesh):
        return "hexahedron", mesh.cells[:, [0, 4, 2, 6, 1, 5, 3, 7]]
    raise ValueError("native binding requires a triangle, tetrahedron, Cartesian or HexMesh")


def create_native_mesh(mesh: Any) -> Any:
    """Create a COMM_SELF DOLFINx mesh from a supported portable mesh.

    The existing portable owner supplies geometric validation and incidence.
    Only the documented vertex permutation to Basix's reference cell changes.
    The shared COMM_SELF communicator is borrowed and is never freed here.
    """
    cell, connectivity = _cell_geometry(mesh)
    basix = _require("basix.ufl")
    ufl = _require("ufl")
    native = _require("dolfinx.mesh")
    mpi = _require("mpi4py.MPI")
    coordinates = basix.element("Lagrange", cell, 1, shape=(mesh.points.shape[1],))
    return native.create_mesh(mpi.COMM_SELF, connectivity, x=mesh.points, e=ufl.Mesh(coordinates))


def _native_cells(mesh: Any, native: Any) -> IntArray:
    """Verify shared input topology and invert DOLFINx's cell reordering."""
    _, cells = _cell_geometry(mesh)
    if native.comm.size != 1:
        raise ValueError("native spaces must use a single-rank communicator (COMM_SELF)")
    original = np.asarray(native.topology.original_cell_index, dtype=np.int64)
    if len(original) != len(cells) or not np.array_equal(np.sort(original), np.arange(len(cells))):
        raise ValueError("native and portable meshes must have the same input cells")
    input_vertices = np.asarray(native.geometry.input_global_indices, dtype=np.int64)
    native_vertices = input_vertices[native.geometry.dofmap]
    if not np.array_equal(native_vertices, cells[original]):
        raise ValueError("native vertex numbering must match the declared portable input topology")
    dimension = mesh.points.shape[1]
    physical = native.geometry.x[:, :dimension]
    if not np.allclose(physical, mesh.points[input_vertices], rtol=0, atol=0):
        raise ValueError("native coordinates must match the declared portable mesh")
    inverse = np.argsort(original).astype(np.int64)
    inverse.setflags(write=False)
    return inverse


@dataclass
class NativeSpace:
    """Associate a portable mesh with one owned native function-space binding.

    ``mapping`` converts the existing equispaced nodal ordering to flattened
    native coefficients. Mixed spaces require an explicit component, and
    H(div)/H(curl) spaces use native coefficients without pretending their
    moment degrees of freedom are point values. ``close`` releases this
    binding's references; separately held forms and spaces remain valid.
    """

    portable_mesh: Any
    native_mesh: Any
    native_space: Any
    cell_type: str
    native_cells: IntArray

    @property
    def mesh(self) -> Any:
        """Return the native mesh while this binding is open."""
        self._open()
        return self.native_mesh

    @property
    def space(self) -> Any:
        """Return the native space while this binding is open."""
        self._open()
        return self.native_space

    @property
    def size(self) -> int:
        """Count flattened, owned native coefficients on the serial mesh."""
        space = self.space
        return int(space.dofmap.index_map.size_local * space.dofmap.index_map_bs)

    @property
    def mapping(self) -> IntArray:
        """Return the supported nodal coefficient map to the native space."""
        return coefficient_map(self)

    def _open(self) -> None:
        """Reject use after the owner releases its native references."""
        if self.native_space is None:
            raise RuntimeError("native space binding is closed")

    def close(self) -> None:
        """Release native references without destroying a borrowed communicator."""
        self.native_space = None
        self.native_mesh = None

    def to_portable(self, coefficients: Any, *, component: int | None = None) -> FloatArray:
        """Delegate conversion to the shared native-to-portable coefficient owner."""
        return to_portable(self, coefficients, component=component)

    def to_native(self, coefficients: Any, *, component: int | None = None) -> FloatArray:
        """Delegate scattering to the shared portable-to-native coefficient owner."""
        return to_native(self, coefficients, component=component)

    def transport_coupling(self, coupling: Any, *, component: int | None = None) -> FloatArray:
        """Delegate test-row conversion without altering mathematical trace signs."""
        return transport_coupling(self, coupling, component=component)

    def interpolate(self, expression: Any) -> FloatArray:
        """Delegate interpolation in the executed native coefficient basis."""
        return interpolate(self, expression)

    def evaluate(
        self,
        coefficients: Any,
        points: Any,
        *,
        cells: Any = None,
        component: int | None = None,
    ) -> FloatArray:
        """Delegate field evaluation, retaining explicitly selected one-sided cells."""
        return evaluate(self, coefficients, points, cells=cells, component=component)

    def descriptor(self, *, component: int | None = None) -> NativeFieldDescriptor:
        """Describe an executed nodal field without retaining native resources."""
        return describe_field(self, component=component)


def bind_space(mesh: Any, element: Any) -> NativeSpace:
    """Bind a user-supplied Basix/UFL element or existing native FunctionSpace.

    Existing native spaces must have been created from the same input topology
    and coordinates. Unrelated numbering is rejected instead of guessed.
    Any native element may be assembled; portable coefficient conversion is
    separately restricted to supported equispaced Lagrange components.
    """
    cell, _ = _cell_geometry(mesh)
    if hasattr(element, "dofmap") and hasattr(element, "mesh"):
        space, native = element, element.mesh
    else:
        native = create_native_mesh(mesh)
        space = _require("dolfinx.fem").functionspace(native, element)
    return NativeSpace(mesh, native, space, cell, _native_cells(mesh, native))


def _component(binding: NativeSpace, component: int | None) -> tuple[Any, IntArray]:
    """Collapse a declared mixed component and retain its parent coefficient map."""
    space = binding.space
    if component is None:
        return space, np.arange(binding.size, dtype=np.int64)
    component = positive_int(component, "component", 0)
    if component >= space.num_sub_spaces:
        raise ValueError("component outside native function space")
    subspace, indices = space.sub(component).collapse()
    return subspace, np.asarray(indices, dtype=np.int64)


def _portable_nodes(mesh: Any, cell: str, degree: int) -> tuple[IntArray, FloatArray, int]:
    """Reuse existing nodal topology and its literal reference interpolation nodes."""
    if cell == "triangle":
        from pymhm.fem.scalar.triangle import multiindices, nodal_space

        dofs, nodes = nodal_space(mesh, degree)
        return dofs, multiindices(degree)[:, 1:] / degree, len(nodes)
    if cell == "tetrahedron":
        from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
        from pymhm.fem.scalar.tetrahedron_topology import tetra_indices

        dofs, nodes = tetra_nodal_space(mesh, degree)
        return dofs, tetra_indices(degree)[:, 1:] / degree, len(nodes)
    if cell == "quadrilateral":
        from pymhm.fem.scalar.quadrilateral import qk_space

        points = np.array(
            [(i / degree, j / degree) for j in range(degree + 1) for i in range(degree + 1)]
        )
        dofs, nodes = qk_space(mesh, degree)
        return dofs, points, len(nodes)
    raise ValueError("portable nodal conversion is not defined for hexahedral spaces")


def _nodal_layout(
    mesh: Any, cell: str, degree: int, discontinuous: bool, points: FloatArray
) -> tuple[IntArray, IntArray, int]:
    """Reuse nodal topology and exact lattice identities for conversion and replay."""
    if degree == 0:
        return (
            np.arange(len(mesh.cells), dtype=np.int64)[:, None],
            np.array([0], dtype=np.int64),
            len(mesh.cells),
        )
    dofs, reference, size = _portable_nodes(mesh, cell, degree)
    lattice = points * degree
    if not np.allclose(lattice, np.rint(lattice), rtol=0, atol=64 * np.finfo(float).eps * degree):
        raise ValueError("portable conversion requires equispaced interpolation nodes")
    lookup = {tuple(node): index for index, node in enumerate(np.rint(lattice).astype(int))}
    identities = np.rint(reference * degree).astype(int)
    if len(lookup) != len(identities) or any(tuple(node) not in lookup for node in identities):
        raise ValueError("native and portable reference interpolation nodes differ")
    permutation = np.array([lookup[tuple(node)] for node in identities], dtype=np.int64)
    if discontinuous:
        dofs = np.arange(dofs.size, dtype=np.int64).reshape(dofs.shape)
        size = dofs.size
    return dofs, permutation, size


def coefficient_map(binding: NativeSpace, *, component: int | None = None) -> IntArray:
    """Map portable nodal coefficients into a selected native component.

    Reference nodes are matched by exact integer lattice identities from
    Basix. Native cell incidence supplies global numbering, including edge
    permutations; no physical coordinate search or proximity merging occurs.
    Vectors use node-major interleaved components. DG fields use cell-major
    local nodes. Mixed fields are selected explicitly through ``component``.
    """
    space, parent = _component(binding, component)
    try:
        element = space.element.basix_element
    except RuntimeError as error:
        raise ValueError("mixed spaces require an explicit nodal component") from error
    basix = _require("basix")
    if element.family != basix.ElementFamily.P or not element.interpolation_is_identity:
        raise ValueError("portable conversion requires a nodal Lagrange element")
    degree = int(element.degree)
    dofs, permutation, count = _nodal_layout(
        binding.portable_mesh,
        binding.cell_type,
        degree,
        bool(element.discontinuous),
        element.points,
    )
    blocks = int(space.dofmap.bs)
    native_dofs = np.array(
        [space.dofmap.cell_dofs(int(cell))[permutation] for cell in binding.native_cells]
    )
    size = count * blocks
    result = np.full(size, -1, dtype=np.int64)
    for local_component in range(blocks):
        portable = (dofs * blocks + local_component).ravel()
        native = parent[(native_dofs * blocks + local_component).ravel()]
        order = np.argsort(portable, kind="stable")
        ordered, targets = portable[order], native[order]
        if np.any((ordered[1:] == ordered[:-1]) & (targets[1:] != targets[:-1])):
            raise ValueError("native incidence disagrees with portable nodal continuity")
        result[portable] = native
    if np.any(result < 0) or len(np.unique(result)) != len(result):
        raise ValueError("native coefficient map must cover distinct portable coordinates")
    result.setflags(write=False)
    return result


def _coefficients(binding: NativeSpace, coefficients: Any) -> FloatArray:
    """Validate native coefficient vectors or basis columns without complex coercion."""
    result = real_array(coefficients, "coefficients")
    if result.ndim not in {1, 2} or result.shape[0] != binding.size:
        raise ValueError("coefficients must match the flattened native space dimension")
    return result


def to_portable(
    binding: NativeSpace, coefficients: Any, *, component: int | None = None
) -> FloatArray:
    """Copy a native coefficient vector or basis columns into portable nodal order."""
    return _coefficients(binding, coefficients)[coefficient_map(binding, component=component)]


def to_native(
    binding: NativeSpace, coefficients: Any, *, component: int | None = None
) -> FloatArray:
    """Scatter portable nodal coefficients or basis columns into a zero native array."""
    indices = coefficient_map(binding, component=component)
    values = real_array(coefficients, "coefficients")
    if values.ndim not in {1, 2} or values.shape[0] != len(indices):
        raise ValueError("coefficients must match the portable nodal component dimension")
    result = np.zeros((binding.size, *values.shape[1:]))
    result[indices] = values
    return result


def transport_coupling(
    binding: NativeSpace, coupling: Any, *, component: int | None = None
) -> FloatArray:
    """Scatter portable local test rows into native rows without changing trace signs."""
    indices = coefficient_map(binding, component=component)
    values = real_array(coupling, "coupling")
    if values.ndim != 2 or values.shape[0] != len(indices):
        raise ValueError("coupling rows must match the portable nodal component dimension")
    result = np.zeros((binding.size, values.shape[1]))
    result[indices] = values
    return result


def interpolate(binding: NativeSpace, expression: Any) -> FloatArray:
    """Interpolate a user callback, expression or function in native coefficients."""
    function = _require("dolfinx.fem").Function(binding.space)
    function.interpolate(expression)
    function.x.scatter_forward()
    return _coefficients(binding, function.x.array)


def _evaluation_points(
    native_mesh: Any,
    portable_mesh: Any,
    native_cells: IntArray,
    points: Any,
    cells: Any,
) -> tuple[FloatArray, IntArray]:
    """Locate physical points without merging independently selected side values."""
    coordinates = real_array(points, "points")
    dimension = portable_mesh.points.shape[1]
    if coordinates.ndim != 2 or coordinates.shape[1] != dimension:
        raise ValueError("points must have shape (points, geometric_dimension)")
    padded = np.zeros((len(coordinates), 3))
    padded[:, :dimension] = coordinates
    if cells is None:
        geometry = _require("dolfinx.geometry")
        tree = geometry.bb_tree(native_mesh, dimension)
        candidates = geometry.compute_collisions_points(tree, padded)
        collisions = geometry.compute_colliding_cells(native_mesh, candidates, padded)
        owners = np.array(
            [
                collisions.links(point)[0] if len(collisions.links(point)) else -1
                for point in range(len(padded))
            ],
            dtype=np.int64,
        )
        if np.any(owners < 0):
            raise ValueError("evaluation point lies outside the native mesh")
    else:
        raw = np.asarray(cells)
        if raw.shape != (len(coordinates),) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("cells must contain one integer portable cell per point")
        if np.any(raw < 0) or np.any(raw >= len(native_cells)):
            raise ValueError("evaluation cell outside portable mesh")
        owners = native_cells[raw]
    return padded, owners


def evaluate(
    binding: NativeSpace,
    coefficients: Any,
    points: Any,
    *,
    cells: Any = None,
    component: int | None = None,
) -> FloatArray:
    """Evaluate native coefficients at physical points on declared owning cells.

    ``cells`` contains portable fine-cell indices, one per point. Supplying it
    preserves independent one-sided traces on discontinuous interfaces. When
    omitted, DOLFINx finds a containing cell; points outside the mesh fail.
    Scalar values return ``(points,)`` and vectors return ``(points, value_size)``.
    """
    padded, owners = _evaluation_points(
        binding.mesh, binding.portable_mesh, binding.native_cells, points, cells
    )
    space, parent = _component(binding, component)
    if space.element.num_sub_elements > 0 and component is None and space.dofmap.bs == 1:
        raise ValueError("mixed evaluation requires an explicit component")
    function = _require("dolfinx.fem").Function(space)
    values = _coefficients(binding, coefficients)
    if values.ndim != 1:
        raise ValueError("field evaluation requires a coefficient vector")
    function.x.array[:] = values[parent]
    values = np.asarray(function.eval(padded, owners.astype(np.int32)), dtype=float)
    return values[:, 0] if values.ndim == 2 and values.shape[1] == 1 else values


@dataclass(frozen=True)
class NativeFieldDescriptor:
    """Persist a native nodal field's executed basis and coefficient layout.

    Arrays are copied and immutable. The stored coefficient map transports an
    executed native vector into portable nodal order before rebinding, so
    replay does not depend on a later DOLFINx numbering. Only equispaced nodal
    Lagrange components have this automatic descriptor; other elements need
    an explicit evaluator and declared basis contract.
    """

    mesh: Any
    cell_type: str
    degree: int
    value_shape: tuple[int, ...]
    discontinuous: bool
    size: int
    mapping: IntArray
    basis_matrix: FloatArray
    basis_points: FloatArray
    basis_digest: str = ""

    def __post_init__(self) -> None:
        """Freeze and digest the executed basis and its parent coefficient map."""
        cell, _ = _cell_geometry(self.mesh)
        if self.cell_type != cell:
            raise ValueError("field reference cell must match its declared mesh topology")
        degree = positive_int(self.degree, "field degree", 0)
        size = positive_int(self.size, "field native size", 0)
        if not isinstance(self.discontinuous, bool):
            raise TypeError("field discontinuous must be boolean")
        if self.degree == 0 and not self.discontinuous:
            raise ValueError("degree-zero field descriptors must be discontinuous")
        if not isinstance(self.value_shape, tuple):
            raise TypeError("field value_shape must be a tuple of positive dimensions")
        shape = tuple(
            positive_int(dimension, "field value dimension") for dimension in self.value_shape
        )
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "value_shape", shape)
        if not isinstance(self.basis_digest, str):
            raise TypeError("field basis_digest must be a string")
        raw = np.asarray(self.mapping)
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("field mapping must contain integer executed coefficient indices")
        mapping = np.array(raw, dtype=np.int64, copy=True)
        matrix = real_array(self.basis_matrix, "basis_matrix")
        points = real_array(self.basis_points, "basis_points")
        if mapping.ndim != 1 or np.any(mapping < 0) or np.any(mapping >= self.size):
            raise ValueError("field mapping must contain valid executed coefficient indices")
        if len(np.unique(mapping)) != len(mapping) or matrix.ndim != 2 or points.ndim != 2:
            raise ValueError("field basis and mapping must have valid independent coordinates")
        basix = _require("basix")
        width = basix.polynomials.dim(
            basix.PolynomialType.legendre, basix.CellType[cell], self.degree
        )
        if matrix.shape != (width, width) or points.shape != (width, self.mesh.points.shape[1]):
            raise ValueError("field basis rows, columns and points must match its polynomial space")
        _, _, count = _nodal_layout(self.mesh, cell, self.degree, self.discontinuous, points)
        if len(mapping) != count * prod(self.value_shape):
            raise ValueError("field mapping must cover its declared portable component dimension")
        digest = sha256()
        digest.update(
            repr(
                (self.cell_type, self.degree, self.value_shape, self.discontinuous, self.size)
            ).encode()
        )
        for array in (mapping, matrix, points):
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
            array.setflags(write=False)
        identity = digest.hexdigest()
        if self.basis_digest and self.basis_digest != identity:
            raise ValueError("field basis digest does not match the executed basis")
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "basis_matrix", matrix)
        object.__setattr__(self, "basis_points", points)
        object.__setattr__(self, "basis_digest", identity)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Restore immutable arrays and verify basis identity when unpickling."""
        return type(self), (
            self.mesh,
            self.cell_type,
            self.degree,
            self.value_shape,
            self.discontinuous,
            self.size,
            self.mapping,
            self.basis_matrix,
            self.basis_points,
            self.basis_digest,
        )


def describe_field(binding: NativeSpace, *, component: int | None = None) -> NativeFieldDescriptor:
    """Describe a supported nodal component using its executed Basix basis."""
    mapping = coefficient_map(binding, component=component)
    space, _ = _component(binding, component)
    element = space.element.basix_element
    return NativeFieldDescriptor(
        binding.portable_mesh,
        binding.cell_type,
        int(element.degree),
        tuple(space.ufl_element().reference_value_shape),
        bool(element.discontinuous),
        binding.size,
        mapping,
        element.coefficient_matrix,
        element.points,
    )


def _evaluate_descriptor(
    descriptor: NativeFieldDescriptor,
    coefficients: Any,
    points: Any,
    *,
    cells: Any = None,
    gradient: bool = False,
) -> tuple[FloatArray, FloatArray | None]:
    """Replay field values and optional physical gradients with one geometry owner."""
    from pymhm.fem.reference import tabulate_archived_nodal_basis

    values = real_array(coefficients, "coefficients")
    if values.shape != (descriptor.size,):
        raise ValueError("coefficients must match the descriptor's executed dimension")
    native = create_native_mesh(descriptor.mesh)
    native_cells = _native_cells(descriptor.mesh, native)
    padded, owners = _evaluation_points(native, descriptor.mesh, native_cells, points, cells)
    dofs, permutation, _ = _nodal_layout(
        descriptor.mesh,
        descriptor.cell_type,
        descriptor.degree,
        descriptor.discontinuous,
        descriptor.basis_points,
    )
    blocks = prod(descriptor.value_shape)
    coefficients_by_node = values[descriptor.mapping].reshape(-1, blocks)
    result = np.empty((len(padded), blocks))
    dimension = descriptor.mesh.points.shape[1]
    gradients = np.empty((len(padded), blocks, dimension)) if gradient else None
    original = np.asarray(native.topology.original_cell_index)
    for owner in np.unique(owners):
        indices = np.flatnonzero(owners == owner)
        geometry = native.geometry.x[native.geometry.dofmap[owner]]
        reference_points = native.geometry.cmap.pull_back(padded[indices], geometry)
        tables = tabulate_archived_nodal_basis(
            descriptor.cell_type,
            descriptor.degree,
            descriptor.basis_matrix,
            reference_points,
            nderiv=int(gradient),
        )
        basis = tables[0] if gradient else tables
        local = coefficients_by_node[dofs[original[owner]]]
        result[indices] = basis[:, permutation] @ local
        if gradients is not None:
            reference_gradient = np.moveaxis(tables[1 : dimension + 1, :, permutation], 0, -1)
            coordinate_element = native.ufl_domain().ufl_coordinate_element().basix_element
            coordinate_table = coordinate_element.tabulate(1, reference_points)
            jacobian = np.einsum(
                "dpn,na->pad", coordinate_table[1 : dimension + 1, :, :, 0], geometry[:, :dimension]
            )
            field_gradient = np.einsum("pnd,nc->pcd", reference_gradient, local)
            gradients[indices] = np.einsum("pcd,pda->pca", field_gradient, np.linalg.inv(jacobian))
    field_values = result[:, 0] if blocks == 1 else result
    field_gradient = None if gradients is None else (gradients[:, 0] if blocks == 1 else gradients)
    return field_values, field_gradient


def evaluate_descriptor(
    descriptor: NativeFieldDescriptor, coefficients: Any, points: Any, *, cells: Any = None
) -> FloatArray:
    """Replay coefficients using the actual archived basis polynomial matrix.

    Native geometry supplies cell location and physical pullback only. Basis
    values use the archived matrix directly, so regenerated DOF numbering or
    equivalent rotations of a newly computed basis cannot alter replay.
    """
    return _evaluate_descriptor(descriptor, coefficients, points, cells=cells)[0]


def evaluate_descriptor_data(
    descriptor: NativeFieldDescriptor, coefficients: Any, points: Any, *, cells: Any = None
) -> tuple[FloatArray, FloatArray]:
    """Evaluate archived values and physical gradients in one geometry traversal.

    Scalar gradients have shape ``(points, dimension)``; vector gradients have
    shape ``(points, flattened_components, dimension)``. Each row differentiates
    that component in physical Cartesian coordinates. The native mesh's actual
    Basix coordinate element supplies its Jacobian. Explicit portable ``cells``
    preserve one-sided values and derivatives on discontinuous interfaces.
    Geometry and basis evaluation are shared; no process-global cache retains
    native resources. Gradient construction never imposes H(div) conformity.
    """
    values, gradients = _evaluate_descriptor(
        descriptor, coefficients, points, cells=cells, gradient=True
    )
    assert gradients is not None
    return values, gradients


def evaluate_descriptor_gradient(
    descriptor: NativeFieldDescriptor, coefficients: Any, points: Any, *, cells: Any = None
) -> FloatArray:
    """Evaluate physical component gradients using the archived executed basis."""
    return evaluate_descriptor_data(descriptor, coefficients, points, cells=cells)[1]
