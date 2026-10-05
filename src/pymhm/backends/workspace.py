"""Reusable, data-independent UFL kernels and bounded native assembly workspaces.

Only kernels, native form bindings and zeroed assembly buffers are reused.
Every assembly returns an independent binary64 array or CSR matrix. Materials,
loads, condensed responses and solver factors remain specific to each problem.
Optional DOLFINx/UFL imports occur only when the native API is invoked.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from threading import get_ident
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
from scipy import sparse

from pymhm.backends.forms import (
    _assemble_compiled,
    _form_domains,
    _real_coefficients,
    _require,
    _serial_domains,
    _shape,
    _space_size,
    _zero,
)
from pymhm.core.validation import FloatArray


class Closable(Protocol):
    """Worker-owned resource with an idempotent explicit lifetime boundary."""

    def close(self) -> None:
        """Release retained native resources."""


Resource = TypeVar("Resource", bound=Closable)


@dataclass(frozen=True)
class CompiledFormBundle:
    """Named data-independent DOLFINx kernels, with explicit rank/shape contracts.

    Keep this object alive while any workspace uses its compiled code. Original
    UFL symbols identify the coefficient/constant/space maps when binding to a
    different mesh. Coefficient values and mesh coordinates are not compiled.
    """

    forms: Mapping[str, Any]
    compiled: Mapping[str, Any]
    shapes: Mapping[str, tuple[int, ...] | None]


@dataclass
class FormWorkspace:
    """Mutable native form bindings and buffers owned by one worker thread.

    Instantiate lazily inside a spawn worker, or separately in each thread.
    Geometry updates preserve topology, finite-element spaces and integration
    entities. The caller must preserve a valid mesh, including its orientation.
    Native Functions/Constants may also be updated directly: each assembly
    freshly packs their current values. No assembled result aliases buffers.
    """

    bundle: CompiledFormBundle | None
    mesh: Any
    bound: dict[str, Any]
    shapes: dict[str, tuple[int, ...]]
    coefficients: dict[Any, Any]
    constants: dict[Any, Any]
    buffers: dict[str, Any] = field(default_factory=dict)
    _thread: int = field(default_factory=get_ident)
    _closed: bool = False

    def assemble(self, name: str) -> FloatArray | sparse.csr_matrix:
        """Assemble a named form, preserving independent owned coefficients."""
        return assemble_workspace_form(self, name)

    def close(self) -> None:
        """Release buffers, native form bindings and mesh references idempotently."""
        self.buffers.clear()
        self.bound.clear()
        self.coefficients.clear()
        self.constants.clear()
        self.bundle = None
        self.mesh = None
        self._closed = True

    def __enter__(self) -> FormWorkspace:
        """Enter an explicitly scoped native workspace lifetime."""
        _check_workspace(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        """Release native resources when the scope ends, including exceptions."""
        self.close()

    def __getstate__(self) -> Any:
        """Prevent native handles from crossing a process serialization boundary."""
        raise TypeError("native form workspaces cannot be pickled; create them inside the worker")


def _check_workspace(workspace: FormWorkspace) -> None:
    """Reject released bindings or concurrent cross-thread mutable use."""
    if workspace._closed:
        raise RuntimeError("the form workspace is closed")
    if workspace._thread != get_ident():
        raise RuntimeError("a form workspace must be used by its owning thread")


def _binary64_data(value: Any) -> None:
    """Require native coordinate/coefficient/constant storage compatible with real kernels."""
    if np.asarray(value).dtype != np.dtype(np.float64):
        raise ValueError("native geometry, coefficient and constant data must use real binary64")


def _compilation_domain(form: Any) -> Any:
    """Validate one compiled integration domain and its native coordinate precision."""
    domains = _form_domains(form)
    if len(domains) != 1:
        raise ValueError("each nonzero local form must have exactly one integration domain")
    domain = domains[0]
    prototype = domain.ufl_cargo()
    if prototype is not None:
        _binary64_data(prototype.geometry.x)
    return domain


def compile_form_bundle(
    forms: Mapping[str, Any],
    comm: Any,
    *,
    form_compiler_options: Mapping[str, Any] | None = None,
    jit_options: Mapping[str, Any] | None = None,
    shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> CompiledFormBundle:
    """Compile named linear/bilinear UFL forms once, independently of native data.

    DOLFINx 0.9 or later is required. Forms retain their quadrature metadata;
    compiler/JIT options are copied. The scalar type is real binary64. Nonzero
    forms each use one integration domain with binary64 native geometry. Literal
    zero forms skip native compilation and require ``shapes`` if UFL erased
    their arguments. A single-rank communicator is required for local assembly.
    """
    if comm.size != 1:
        raise ValueError("local form bundles require a single-rank communicator (COMM_SELF)")
    sources = dict(forms)
    if not sources or any(not isinstance(name, str) or not name for name in sources):
        raise ValueError("forms must contain nonempty string names")
    declared = dict(shapes or {})
    if declared.keys() - sources.keys():
        raise ValueError("shape names must identify declared forms")
    expected = {name: _shape(declared.get(name)) for name in sources}
    compiled = {}
    fem = _require("dolfinx.fem")
    options = dict(form_compiler_options or {})
    if "scalar_type" in options and np.dtype(options["scalar_type"]) != np.dtype(np.float64):
        raise ValueError("form workspaces require real binary64 scalar_type")
    options["scalar_type"] = np.float64
    for name, form in sources.items():
        _serial_domains(form)
        rank = len(form.arguments())
        zero = _zero(form)
        shape = expected[name]
        if rank not in {1, 2} and not (zero and shape is not None):
            raise ValueError("forms must be linear/bilinear; argument-free zeros require shape")
        if shape is not None and rank and rank != len(shape):
            raise ValueError("shape rank does not match the declared form arguments")
        if not zero:
            _compilation_domain(form)
        compiled[name] = (
            None
            if zero
            else fem.compile_form(
                comm, form, form_compiler_options=dict(options), jit_options=dict(jit_options or {})
            )
        )
    return CompiledFormBundle(sources, compiled, expected)


def _symbols(form: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Extract ordered UFL symbols without importing UFL in the portable core."""
    ufl = _require("ufl")
    return (
        tuple(ufl.algorithms.extract_coefficients(form)),
        tuple(ufl.algorithms.analysis.extract_constants(form)),
    )


def _binding(symbol: Any, mapping: Mapping[Any, Any]) -> Any:
    """Resolve a native prototype symbol or an explicit data-independent binding."""
    value = next(
        (value for key, value in mapping.items() if key is symbol or key == symbol), symbol
    )
    if not hasattr(value, "_cpp_object"):
        raise ValueError("plain UFL symbols require explicit native bindings")
    return value


def _same_mesh(actual: Any, mesh: Any) -> bool:
    """Compare native mesh identity without relying on mutable physical coordinates."""
    return actual is mesh or actual._cpp_object is mesh._cpp_object


def _validate_space(original: Any, native: Any, mesh: Any) -> None:
    """Reject stale cross-mesh spaces or different finite-element layouts."""
    if not _same_mesh(native.mesh, mesh):
        raise ValueError("native function spaces and coefficients must belong to the bound mesh")
    if original.ufl_element() != native.ufl_element():
        raise ValueError("native bindings must preserve the compiled finite element")


def _integration_domains(form: Any, mesh: Any, fem: Any) -> dict[Any, Any]:
    """Pack tagged integration entities through DOLFINx's public domain API."""
    domains: dict[Any, Any] = {}
    for integral in form.integrals():
        data = integral.subdomain_data()
        marker = integral.subdomain_id()
        if data is None or marker == "everywhere":
            continue
        kind = getattr(fem.IntegralType, integral.integral_type())
        ids = marker if isinstance(marker, tuple) else (marker,)
        packed = domains.setdefault(kind, {})
        for identifier in ids:
            if identifier in packed:
                continue
            if hasattr(data, "find"):
                dim = mesh.topology.dim
                mesh.topology.create_connectivity(dim - 1, dim)
                mesh.topology.create_connectivity(dim, dim - 1)
                packed[identifier] = np.asarray(
                    fem.compute_integration_domains(
                        kind, mesh.topology, data.find(identifier), data.dim
                    ),
                    dtype=np.int32,
                )
            else:
                packed.update((key, np.array(value, copy=True)) for key, value in data)
    return {kind: list(values.items()) for kind, values in domains.items()}


def create_workspace(
    bundle: CompiledFormBundle,
    mesh: Any,
    *,
    space_map: Mapping[Any, Any] | None = None,
    coefficient_map: Mapping[Any, Any] | None = None,
    constant_map: Mapping[Any, Any] | None = None,
    subdomains: Mapping[Any, Any] | None = None,
    entity_maps: Mapping[Any, Any] | None = None,
) -> FormWorkspace:
    """Bind compiled kernels to a mesh, native spaces, coefficients and constants.

    Coefficient/constant maps use original UFL symbols as keys. Space maps use
    argument numbers (test zero, trial one), or hashable original UFL spaces.
    Omitting a map binds an original native prototype to itself. For a new mesh
    supply spaces and native data on that mesh explicitly. ``subdomains`` contains
    DOLFINx integration-domain
    pairs; if omitted, original tagged UFL integrals are packed on ``mesh``.
    Tags and integration entities must describe the same topology.
    The bound mesh preserves the compiled coordinate element. Native mesh,
    Function and Constant storage uses binary64 precision. All native argument
    and coefficient spaces belong to this bound integration mesh.
    Rows follow test arguments, columns trial arguments; no gauge, sign or boundary rule
    is inferred. A workspace holds no solver or material-dependent response.
    """
    if mesh.comm.size != 1:
        raise ValueError("local form workspaces require a single-rank communicator (COMM_SELF)")
    _binary64_data(mesh.geometry.x)
    fem = _require("dolfinx.fem")
    spaces = dict(space_map or {})
    coefficients, constants = dict(coefficient_map or {}), dict(constant_map or {})
    bound: dict[str, Any] = {}
    expected: dict[str, tuple[int, ...]] = {}
    used_coefficients, used_constants = {}, {}
    for name, form in bundle.forms.items():
        if bundle.compiled[name] is not None and (
            _compilation_domain(form).ufl_coordinate_element()
            != mesh.ufl_domain().ufl_coordinate_element()
        ):
            raise ValueError("native mesh must preserve the compiled coordinate element")
        native_spaces = [
            _binding(spaces.get(arg.number(), arg.ufl_function_space()), spaces)
            for arg in form.arguments()
        ]
        for argument, native_space in zip(form.arguments(), native_spaces, strict=True):
            _validate_space(argument.ufl_function_space(), native_space, mesh)
        shape = bundle.shapes[name] or tuple(_space_size(space) for space in native_spaces)
        expected[name] = shape
        if bundle.compiled[name] is None:
            bound[name] = None
            continue
        symbols, constant_symbols = _symbols(form)
        local_coefficients = {symbol: _binding(symbol, coefficients) for symbol in symbols}
        local_constants = {symbol: _binding(symbol, constants) for symbol in constant_symbols}
        for symbol, native in local_coefficients.items():
            _validate_space(symbol.ufl_function_space(), native.function_space, mesh)
            _binary64_data(native.x.array)
        for symbol, native in local_constants.items():
            if symbol.ufl_shape != native.ufl_shape:
                raise ValueError("native constants must preserve the compiled tensor shape")
            _binary64_data(native.value)
        used_coefficients.update(local_coefficients)
        used_constants.update(local_constants)
        domains = (
            _integration_domains(form, mesh, fem)
            if subdomains is None
            else {
                kind: [
                    (identifier, np.array(entities, copy=True)) for identifier, entities in values
                ]
                for kind, values in subdomains.items()
            }
        )
        bound[name] = fem.create_form(
            bundle.compiled[name],
            native_spaces,
            mesh,
            domains,
            local_coefficients,
            local_constants,
            entity_maps=None if entity_maps is None else dict(entity_maps),
        )
    return FormWorkspace(bundle, mesh, bound, expected, used_coefficients, used_constants)


def assemble_workspace_form(workspace: FormWorkspace, name: str) -> FloatArray | sparse.csr_matrix:
    """Assemble current data into zeroed reusable buffers and return an owned copy."""
    _check_workspace(workspace)
    compiled = workspace.bound[name]
    shape = workspace.shapes[name]
    if compiled is None:
        return np.zeros(shape) if len(shape) == 1 else sparse.csr_matrix(shape)
    result, buffer = _assemble_compiled(compiled, len(shape), shape, workspace.buffers.get(name))
    workspace.buffers[name] = buffer
    return result


def update_workspace(
    workspace: FormWorkspace,
    *,
    geometry: Any = None,
    coefficients: Mapping[Any, Any] | None = None,
    constants: Mapping[Any, Any] | None = None,
) -> None:
    """Update physical coordinates and declared native data without recompilation.

    Array updates preserve the native coefficient/constant shape and require
    real finite values. A coefficient may instead be an interpolation callable;
    it is evaluated after the coordinate update, on the current physical mesh.
    Analytical UFL expressions involving SpatialCoordinate keep their original
    evaluation/quadrature and read updated coordinates directly. Function ghost
    entries are synchronized after updates. This operation never modifies FE
    spaces, topology, quadrature, trace orientation or integration entities.
    """
    _check_workspace(workspace)
    updates = []
    if geometry is not None:
        updates.append((workspace.mesh.geometry.x, _real_coefficients(geometry, rank=2)))
    callbacks = []
    for symbol, value in (coefficients or {}).items():
        native = workspace.coefficients[symbol]
        if callable(value):
            callbacks.append((native, value))
        else:
            updates.append((native.x.array, _real_coefficients(value, rank=1)))
    for symbol, value in (constants or {}).items():
        target = workspace.constants[symbol].value
        updates.append((target, _real_coefficients(value, rank=np.ndim(target))))
    if any(target.shape != value.shape for target, value in updates):
        raise ValueError("updated native data must preserve its original shape")
    for target, value in updates:
        target[...] = value
    for native, callback in callbacks:
        native.interpolate(callback)
    for symbol in coefficients or {}:
        workspace.coefficients[symbol].x.scatter_forward()


def _digest_array(value: Any) -> tuple[Any, ...]:
    """Hash an array's layout and values for a structural compatibility key."""
    array = np.asarray(value)
    return array.shape, array.dtype.str, sha256(array.tobytes()).hexdigest()


def workspace_key(
    mesh: Any,
    forms: Mapping[str, Any],
    *,
    space_map: Mapping[Any, Any] | None = None,
    subdomains: Mapping[Any, Any] | None = None,
    entity_maps: Mapping[Any, Any] | None = None,
    form_compiler_options: Mapping[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Describe kernel, FE, geometry-layout and topology compatibility, excluding data.

    Form signatures include approximation elements and quadrature metadata.
    Geometry coordinates are intentionally absent; their layout and cell
    connectivity remain fixed. Values of coefficients/constants are absent.
    Custom cache keys must enforce these same structural distinctions and any
    additional application-specific reusable data contracts.
    """
    geometry = mesh.geometry
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    connectivity = mesh.topology.connectivity(mesh.topology.dim, 0)
    spaces = dict(space_map or {})
    space_layouts, integral_layouts = [], []
    for form in forms.values():
        for argument in form.arguments():
            native = _binding(spaces.get(argument.number(), argument.ufl_function_space()), spaces)
            space_layouts.append(_digest_array(native.dofmap.list))
        for integral in form.integrals():
            data = integral.subdomain_data()
            if data is not None:
                integral_layouts.append(
                    (_digest_array(data.indices), _digest_array(data.values))
                    if hasattr(data, "indices")
                    else tuple((key, _digest_array(entities)) for key, entities in data)
                )
    return (
        tuple((name, form.signature()) for name, form in forms.items()),
        str(mesh.topology.cell_type),
        mesh.topology.dim,
        geometry.dim,
        geometry.cmap.degree,
        geometry.cmap.variant,
        geometry.x.shape,
        geometry.x.dtype.str,
        _digest_array(geometry.dofmap),
        _digest_array(connectivity.array),
        _digest_array(connectivity.offsets),
        tuple(space_layouts),
        tuple(integral_layouts),
        tuple(
            (str(kind), tuple((key, _digest_array(entities)) for key, entities in values))
            for kind, values in (subdomains or {}).items()
        ),
        tuple(
            (id(domain), _digest_array(entities))
            for domain, entities in (entity_maps or {}).items()
        ),
        tuple(sorted((key, repr(value)) for key, value in (form_compiler_options or {}).items())),
    )


class WorkspaceCache(Generic[Resource]):
    """Bounded LRU of worker-owned closable workspaces or application records.

    A record may contain a FormWorkspace plus proven geometry-only data. Keys
    identify structural compatibility; factories run only on misses. Eviction
    and explicit close release the whole record. Create separate caches inside
    each process/thread, rather than serializing or sharing native resources.
    """

    def __init__(self, capacity: int = 4) -> None:
        """Create a positive-capacity cache without importing a native backend."""
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("workspace cache capacity must be a positive integer")
        self.capacity = capacity
        self._items: OrderedDict[Hashable, Resource] = OrderedDict()
        self._thread = get_ident()
        self._closed = False

    def get(self, key: Hashable, factory: Callable[[], Resource]) -> Resource:
        """Return the most recently used resource, constructing only on a miss."""
        if self._closed or self._thread != get_ident():
            raise RuntimeError("workspace cache is closed or belongs to another thread")
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        resource = factory()
        self._items[key] = resource
        if len(self._items) > self.capacity:
            _, evicted = self._items.popitem(last=False)
            evicted.close()
        return resource

    def close(self) -> None:
        """Close every retained resource once, then reject further cache use."""
        resources = tuple(self._items.values())
        self._items.clear()
        self._closed = True
        error = None
        for resource in resources:
            try:
                resource.close()
            except Exception as failure:
                if error is None:
                    error = failure
        if error is not None:
            raise error

    def __enter__(self) -> WorkspaceCache[Resource]:
        """Scope the lifetime of all worker-owned cached native records."""
        if self._closed:
            raise RuntimeError("workspace cache is closed")
        return self

    def __exit__(self, *exc: Any) -> None:
        """Release all records on normal completion or an exceptional exit."""
        self.close()

    def __getstate__(self) -> Any:
        """Require cache creation inside the process that owns native resources."""
        raise TypeError("workspace caches cannot be pickled; create them inside the worker")
