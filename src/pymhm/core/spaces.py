"""Portable interface layouts and transport of user-defined variational blocks.

An interface binding maps selected global coefficients into the declared local
trial/test trace bases. Test functionals travel through the transpose map. The
maps may encode signs, permutations or dense basis changes; they never select a
physical coupling sign. Numerical condensation remains owned by ``equations``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal, Protocol

import numpy as np
from scipy import sparse

from pymhm.core.equations import FormCompiler, LocalEquations, _compile_block, _dofs, compile_form
from pymhm.core.validation import FloatArray, positive_int, real_array


def _restriction(value: Any, width: int, name: str, require_injective: bool) -> FloatArray:
    """Own a finite coefficient map and check injectivity when declared necessary."""
    result = np.eye(width) if value is None else real_array(value, name)
    if result.ndim != 2 or result.shape[1] != width:
        raise ValueError(f"{name} must have one column per selected global coordinate")
    if require_injective and (
        result.shape[0] < width or (width and np.linalg.matrix_rank(result) != width)
    ):
        raise ValueError(f"{name} must be injective on its selected global coordinates")
    result.setflags(write=False)
    return result


def _binding_digest(binding: TraceBinding) -> str:
    """Identify the declared numerical basis and both coefficient transports."""
    digest = sha256(binding.basis_id.encode("utf8"))
    for value in (binding.dofs, binding.test_dofs, binding.trial_map, binding.test_map):
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TraceBinding:
    """Independent trial/test embeddings into a square global interface layout.

    ``trial_map @ global[dofs]`` gives local trial trace coefficients, and
    ``test_map @ global[test_dofs]`` gives local test coefficients. Each map
    has one column per selected global coordinate. Rectangular and rank-deficient
    maps permit projections/restrictions; ``require_injective=True`` requests
    an injectivity check when the formulation needs it. Omitted maps
    are identities; omitted test coordinates/maps reuse the trial binding.
    ``basis_id`` names the executed local basis supplied by an adapter. Its
    identity, maps and selected coordinates enter ``basis_digest`` together.
    The digest identifies declared representation data, not PDE consistency.
    """

    dofs: Any
    trial_map: Any = None
    test_dofs: Any = None
    test_map: Any = None
    basis_id: str = ""
    require_injective: bool = False
    basis_digest: str = field(init=False)

    def __post_init__(self) -> None:
        """Freeze owned coordinate maps and validate independent basis transports."""
        trial = _dofs(self.dofs, "dofs")
        test = trial if self.test_dofs is None else _dofs(self.test_dofs, "test_dofs")
        if not isinstance(self.require_injective, bool):
            raise TypeError("require_injective must be boolean")
        trial_map = _restriction(self.trial_map, len(trial), "trial_map", self.require_injective)
        test_map = (
            trial_map
            if self.test_map is None and self.test_dofs is None
            else _restriction(self.test_map, len(test), "test_map", self.require_injective)
        )
        if not isinstance(self.basis_id, str):
            raise TypeError("basis_id must be a string identifying the declared basis")
        for name, value in (
            ("dofs", trial),
            ("test_dofs", test),
            ("trial_map", trial_map),
            ("test_map", test_map),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "basis_digest", _binding_digest(self))

    @property
    def trial_size(self) -> int:
        """Return the local trial basis width before transport."""
        return int(self.trial_map.shape[0])

    @property
    def test_size(self) -> int:
        """Return the local test basis width before transport."""
        return int(self.test_map.shape[0])

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Preserve immutable coefficient maps and basis identity across spawn/replay."""
        if self.basis_digest != _binding_digest(self):
            raise ValueError("trace binding basis identity is stale")
        return type(self), (
            self.dofs,
            self.trial_map,
            self.test_dofs,
            self.test_map,
            self.basis_id,
            self.require_injective,
        )


class InterfaceSpace(Protocol):
    """Describe global interface coordinates without requiring inheritance.

    A custom implementation owns its basis convention and local support. It
    can use arbitrary numbering and dense changes of trace basis. Geometry,
    basis evaluation and native form binding are separate optional capabilities.
    """

    @property
    def size(self) -> int:
        """Return the complete global interface coefficient dimension."""
        ...

    def binding(self, cell: int) -> TraceBinding:
        """Return independent trace maps for the selected macrocell."""
        ...


@dataclass(frozen=True)
class ComponentTraceSpace:
    """Interleave scalar trace coordinates for a declared vector-valued interface.

    ``base`` owns the scalar basis, geometry and any continuity constraints.
    Coordinate ``components*i+j`` is component j of scalar coordinate i. This
    wrapper shares exactly the same vertices/edges as the base and introduces
    no normal or tangential rotation. Use ``bind_interface`` to declare value
    or canonical-normal transport, or ``TraceBinding`` for a different frame.
    """

    base: Any
    components: int

    def __post_init__(self) -> None:
        """Validate a scalar mesh-associated layout without loading a FEM backend."""
        if not hasattr(self.base, "mesh") or not callable(getattr(self.base, "cell_dofs", None)):
            raise TypeError("component traces require a mesh-associated scalar trace space")
        if getattr(self.base, "components", 1) != 1:
            raise ValueError("component trace base must be scalar")
        positive_int(self.base.size, "scalar interface size", 0)
        object.__setattr__(self, "components", positive_int(self.components, "components"))

    @property
    def mesh(self) -> Any:
        """Return the unchanged scalar interface's macro mesh."""
        return self.base.mesh

    @property
    def size(self) -> int:
        """Return the component-interleaved global coefficient dimension."""
        return int(self.base.size * self.components)

    def _interleave(self, dofs: Any) -> np.ndarray:
        """Expand scalar indices without changing their declared order."""
        return (np.asarray(dofs)[:, None] * self.components + np.arange(self.components)).ravel()

    def dofs(self, face: int) -> np.ndarray:
        """Return one face's vector coefficients, including any shared scalar nodes."""
        owner = getattr(self.base, "dofs", None)
        return self._interleave(owner(face) if callable(owner) else self.base.face_dofs[face])

    def cell_dofs(self, cell: int) -> np.ndarray:
        """Preserve the scalar cell-coordinate order with interleaved components."""
        return self._interleave(self.base.cell_dofs(cell))


def validate_trace_binding(binding: TraceBinding, size: int) -> TraceBinding:
    """Reject stale or out-of-range layouts before assembling local equations."""
    if not isinstance(binding, TraceBinding):
        raise TypeError("an interface space must return TraceBinding")
    size = positive_int(size, "interface size", 0)
    if any(np.any(dofs >= size) for dofs in (binding.dofs, binding.test_dofs)):
        raise ValueError("trace binding selects coordinates outside the interface space")
    if binding.basis_digest != _binding_digest(binding):
        raise ValueError("trace binding basis identity is stale")
    return binding


def _source_basis_id(space: Any, cell: int) -> str:
    """Snapshot existing face declarations and their physical geometry locally."""
    digest = sha256(f"{type(space).__module__}.{type(space).__qualname__}".encode())
    faces = space.mesh.cell_faces[cell]
    digest.update(str(space.size).encode("ascii"))
    digest.update(np.asarray(space.cell_dofs(cell), dtype=np.int64).tobytes(order="C"))
    digest.update(str(getattr(space, "components", 1)).encode("ascii"))
    digest.update(str(getattr(space, "basis_id", "")).encode("utf8"))
    base = getattr(space, "base", None)
    if base is not None and callable(getattr(base, "cell_dofs", None)):
        digest.update(_source_basis_id(base, cell).encode("ascii"))
    face_dofs = getattr(space, "dofs", None)
    if callable(face_dofs):
        for face in faces:
            coordinates = np.asarray(face_dofs(int(face)), dtype=np.int64)
            digest.update(str(coordinates.shape).encode("ascii"))
            digest.update(coordinates.tobytes(order="C"))
    for name in (
        "faces",
        "degree",
        "degrees",
        "subdivisions",
        "continuous",
        "face_dofs",
        "element_dofs",
        "partitions",
    ):
        records = getattr(space, name, None)
        if np.isscalar(records):
            digest.update(repr((name, records)).encode("utf8"))
        elif records is not None and not isinstance(records, dict):
            digest.update(name.encode("ascii"))
            for face in faces:
                record = records[int(face)]
                if isinstance(record, np.ndarray):
                    digest.update(str(record.shape).encode("ascii"))
                    digest.update(record.dtype.str.encode("ascii"))
                    digest.update(record.tobytes(order="C"))
                else:
                    digest.update(repr(record).encode("utf8"))
    partition = getattr(space, "face_partition", None)
    if callable(partition):
        for face in faces:
            digest.update(np.asarray(partition(int(face))).tobytes(order="C"))
    subtriangle_dofs = getattr(space, "subtriangle_dofs", None)
    if callable(partition) and callable(subtriangle_dofs):
        for face in faces:
            for segment in range(len(partition(int(face)))):
                digest.update(
                    np.asarray(subtriangle_dofs(int(face), segment), dtype=np.int64).tobytes(
                        order="C"
                    )
                )
    if hasattr(space.mesh, "points") and hasattr(space.mesh, "faces"):
        for face in faces:
            digest.update(space.mesh.points[space.mesh.faces[int(face)]].tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class BoundInterface:
    """Adapt an existing face space to the structural interface-space contract.

    ``convention='value'`` preserves scalar/vector face values. ``'normal'``
    transports coefficients expressed in canonical face normals to outward
    macrocell normals, reusing the mesh's existing incidence signs. The adapter
    never infers a physical sign in B or C. Existing operators that already
    include orientation must use ``coordinates='global'`` when bound.
    """

    space: Any
    convention: Literal["value", "normal"] = "value"
    _basis_ids: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Require a mesh-associated trace layout and a supported convention."""
        if self.convention not in {"value", "normal"}:
            raise ValueError("trace convention must be value or normal")
        if not (
            hasattr(self.space, "mesh")
            and hasattr(self.space.mesh, "cell_faces")
            and callable(getattr(self.space, "cell_dofs", None))
        ):
            raise TypeError("a built-in interface requires a mesh-associated face space")
        positive_int(self.space.size, "interface size", 0)
        if self.convention == "normal" and not (
            hasattr(self.space.mesh, "signs") and callable(getattr(self.space, "dofs", None))
        ):
            raise TypeError("normal trace binding requires face coordinates and incidence signs")
        object.__setattr__(
            self,
            "_basis_ids",
            tuple(_source_basis_id(self.space, cell) for cell in range(len(self.mesh.cell_faces))),
        )

    @property
    def mesh(self) -> Any:
        """Return the macro mesh associated with the underlying trace basis."""
        return self.space.mesh

    @property
    def size(self) -> int:
        """Return the complete scalar/component-interleaved interface dimension."""
        return int(self.space.size)

    def binding(self, cell: int) -> TraceBinding:
        """Derive local coefficient transport from the existing topology owner."""
        cell = positive_int(cell, "cell", 0)
        if cell >= len(self.mesh.cell_faces):
            raise ValueError("cell index outside interface mesh")
        if self._basis_ids[cell] != _source_basis_id(self.space, cell):
            raise ValueError("the bound source face basis identity is stale")
        dofs = _dofs(self.space.cell_dofs(cell), "cell dofs")
        transform = np.eye(len(dofs))
        if self.convention == "normal":
            indices = {int(dof): index for index, dof in enumerate(dofs)}
            visited: set[int] = set()
            for side, face in enumerate(self.mesh.cell_faces[cell]):
                face_dofs = _dofs(self.space.dofs(int(face)), "face dofs")
                if any(int(dof) in visited or int(dof) not in indices for dof in face_dofs):
                    raise ValueError("normal trace faces need independent, cell-owned coordinates")
                positions = np.array([indices[int(dof)] for dof in face_dofs], dtype=np.int64)
                sign = self.mesh.signs[cell, side]
                if sign not in {-1, 1}:
                    raise ValueError("face incidence signs must be minus or plus one")
                transform[positions, positions] = sign
                visited.update(int(dof) for dof in face_dofs)
            if len(visited) != len(dofs):
                raise ValueError("face coordinates must cover the selected cell trace")
        basis_id = f"{self._basis_ids[cell]}:{self.convention}"
        return validate_trace_binding(TraceBinding(dofs, transform, basis_id=basis_id), self.size)


def bind_interface(
    space: Any, *, convention: Literal["value", "normal"] = "value"
) -> BoundInterface:
    """Bind a built-in face space using its declared value or normal convention.

    Custom spaces implement ``InterfaceSpace`` directly and return their own
    ``TraceBinding``; no inheritance, registry or native FEM dependency is needed.
    Tangential/component rotations require an explicitly declared custom map.
    """
    return BoundInterface(space, convention)


def _dense_form(compiler: FormCompiler, form: Any, shape: tuple[int, ...], name: str) -> FloatArray:
    """Compile a local block once and materialize only the trace-sized sparse result."""
    result = _compile_block(compiler, form, name, shape)
    return result.toarray() if sparse.issparse(result) else result


def bind_local_equations(
    binding: TraceBinding,
    *,
    a: Any,
    L: Any,
    b: Any,
    c: Any,
    d: Any = 0,
    g: Any = 0,
    kernel: Any = None,
    moments: Any = None,
    coarse_basis: Any = None,
    left_kernel: Any = None,
    test_basis: Any = None,
    test_moments: Any = None,
    metadata: Any = None,
    coordinates: Literal["local", "global"] = "local",
    compiler: FormCompiler = compile_form,
) -> LocalEquations:
    """Transport trace pairings independently before the existing condensation.

    For maps R and S, local blocks become ``B R``, ``S.T C``, ``S.T D R``
    and ``S.T g``. Local volume coefficients, retained modes and their moment
    functionals are unchanged. ``coordinates='global'`` declares that supplied
    pairings already use selected global coefficients and skips this transport,
    preventing double orientation for existing signed coupling owners. Native
    forms compile in their own local integration spaces through ``compiler``.
    """
    if not isinstance(binding, TraceBinding):
        raise TypeError("binding must be TraceBinding")
    if binding.basis_digest != _binding_digest(binding):
        raise ValueError("trace binding basis identity is stale")
    if coordinates not in {"local", "global"}:
        raise ValueError("equation coordinates must be local or global")
    operator = _compile_block(compiler, a, "a")
    if len(operator.shape) != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("the local trial and test operator must be square")
    width = operator.shape[0]
    trial = binding.trial_size if coordinates == "local" else len(binding.dofs)
    test = binding.test_size if coordinates == "local" else len(binding.test_dofs)
    paired_b = _dense_form(compiler, b, (width, trial), "b")
    paired_c = _dense_form(compiler, c, (test, width), "c")
    paired_d = _dense_form(compiler, d, (test, trial), "d")
    paired_g = _dense_form(compiler, g, (test,), "g")
    if coordinates == "local":
        paired_b = paired_b @ binding.trial_map
        paired_c = binding.test_map.T @ paired_c
        paired_d = binding.test_map.T @ paired_d @ binding.trial_map
        paired_g = binding.test_map.T @ paired_g
    return LocalEquations(
        a=operator,
        L=L,
        b=paired_b,
        c=paired_c,
        dofs=binding.dofs,
        test_dofs=binding.test_dofs,
        d=paired_d,
        g=paired_g,
        kernel=kernel,
        moments=moments,
        coarse_basis=coarse_basis,
        left_kernel=left_kernel,
        test_basis=test_basis,
        test_moments=test_moments,
        metadata=metadata,
        trace_binding=binding,
    )


@dataclass(frozen=True, init=False)
class MeshHierarchy:
    """Associate macro cells with worker-created local meshes.

    ``local_meshes`` is a callable ``cell -> mesh`` or one portable mesh per
    macrocell. The callable runs only when a local context requests its mesh,
    so native objects can remain worker-owned. ``items`` selects macrocell
    indices and their deterministic assembly order; its default selects all.
    Native objects, factories and meshes retain their documented execution
    requirements: process execution needs picklable portable descriptions.
    """

    macro: Any
    local_meshes: Callable[[int], Any] | Sequence[Any]
    items: tuple[int, ...]

    def __init__(
        self,
        macro: Any,
        local_meshes: Callable[[int], Any] | Sequence[Any],
        items: Sequence[int] | None = None,
    ) -> None:
        """Check topology association without eagerly constructing local meshes."""
        if not hasattr(macro, "cells"):
            raise TypeError("a mesh hierarchy requires macrocell topology")
        count = len(macro.cells)
        selected = tuple(range(count)) if items is None else tuple(items)
        checked = _dofs(selected, "hierarchy items")
        if np.any(checked >= count):
            raise ValueError("hierarchy items select cells outside the macro mesh")
        if not callable(local_meshes):
            meshes = tuple(local_meshes)
            if len(meshes) != count:
                raise ValueError("provide one local mesh per macrocell")
            local_meshes = meshes
        object.__setattr__(self, "macro", macro)
        object.__setattr__(self, "local_meshes", local_meshes)
        object.__setattr__(self, "items", tuple(int(cell) for cell in checked))

    def local_mesh(self, cell: int) -> Any:
        """Construct or retrieve a local mesh belonging to this hierarchy."""
        cell = positive_int(cell, "cell", 0)
        if cell not in self.items:
            raise ValueError("cell is not selected in this mesh hierarchy")
        return self.local_meshes(cell) if callable(self.local_meshes) else self.local_meshes[cell]
