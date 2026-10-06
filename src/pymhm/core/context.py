"""Mesh-associated variational contexts lowering to the existing hybrid algebra.

Mathematical signs, spaces, modes and gauges are declarations. Contexts own their
coordinate binding; condensation, reduction and solving keep their current owners.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from pymhm.core.equations import Equation, FormCompiler, LocalEquations, compile_form
from pymhm.core.multiscale import MultiscaleProblem, NestedEquations
from pymhm.core.spaces import (
    InterfaceSpace,
    MeshHierarchy,
    TraceBinding,
    bind_local_equations,
    validate_trace_binding,
)
from pymhm.core.validation import FloatArray, positive_int, real_array

_EMPTY_EQUATION = Equation(0, 0)


@dataclass(frozen=True)
class GlobalContext:
    """Bound interface and declared retained coordinates for global forms.

    Coordinates are interface first, then retained modes in hierarchy item order.
    The counts describe user-declared physical modes; no local PDE is solved or
    nullspace guessed to determine this layout. Physical load signs remain explicit.
    """

    hierarchy: MeshHierarchy
    interface: InterfaceSpace
    coarse_sizes: tuple[int, ...]
    _trace_size: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Freeze declared dimensions independently of a mutable custom provider."""
        if not isinstance(self.hierarchy, MeshHierarchy):
            raise TypeError("global context requires MeshHierarchy")
        if not callable(getattr(self.interface, "binding", None)):
            raise TypeError("interface binding must be callable")
        sizes = tuple(positive_int(size, "retained modes", 0) for size in self.coarse_sizes)
        if len(sizes) != len(self.hierarchy.items):
            raise ValueError("retained counts must follow hierarchy item order")
        object.__setattr__(self, "coarse_sizes", sizes)
        object.__setattr__(
            self, "_trace_size", positive_int(self.interface.size, "interface size", 0)
        )

    @property
    def trace_size(self) -> int:
        """Return the number of declared global interface coordinates."""
        return self._trace_size

    @property
    def size(self) -> int:
        """Return the square global dimension, including declared retained modes."""
        return self.trace_size + sum(self.coarse_sizes)

    def trace_load(self, value: Any) -> FloatArray:
        """Embed an interface functional, leaving retained compatibility loads zero."""
        trace = compile_form(value, (self.trace_size,))
        return np.concatenate((trace, np.zeros(sum(self.coarse_sizes))))

    def interface_equation(self, forms: Callable[[Any, Any, Any], Any]) -> Equation:
        """Assemble user-written global interface UFL forms in the bound coordinates.

        forms(trial, test, measure) returns Equation or its (a, L) pair. The
        built-in adapter integrates supported planar polynomial face spaces;
        custom spaces may declare the same capability. Retained-mode rows and
        columns are padded by zero, without inventing a coupling or boundary sign.
        """
        custom = getattr(self.interface, "interface_equation", None)
        if callable(custom):
            return custom(self, forms)
        from pymhm.backends.traces import interface_equation

        return interface_equation(self, forms)

    def retained_dofs(self, cell: int) -> np.ndarray:
        """Expose a cell's retained coordinates for explicitly assembled global forms."""
        try:
            position = self.hierarchy.items.index(cell)
        except ValueError as error:
            raise ValueError("cell is not selected in this mesh hierarchy") from error
        first = self.trace_size + sum(self.coarse_sizes[:position])
        return np.arange(first, first + self.coarse_sizes[position], dtype=np.int64)

    def boundary_data(
        self, value: Any, neumann: Mapping[int, Any] | None = None, *, order: int = 6
    ) -> tuple[FloatArray, dict[int, float]]:
        """Project declared planar value/normal data using the existing boundary owner.

        This returns an unsigned interface functional and fixed coefficients;
        callers choose its sign in their global equation. Custom spaces may
        implement their own boundary_data(value, neumann, order=...) capability.
        No Dirichlet/Neumann role is inferred from a local differential operator.
        """
        custom = getattr(self.interface, "boundary_data", None)
        if callable(custom):
            return custom(value, neumann, order=order)
        space = getattr(self.interface, "space", None)
        if space is None:
            raise TypeError("custom interface does not declare a boundary-data capability")
        from pymhm.fem.traces.interval import SkeletonSpace

        if not isinstance(space, SkeletonSpace) or space.mesh.points.shape[1] != 2:
            raise TypeError("this interface requires an explicit boundary-data projection")
        from pymhm.fem.scalar.operators import boundary_data

        return boundary_data(space, value, None if neumann is None else dict(neumann), order=order)

    def fix_faces(self, faces: Any, value: float = 0.0) -> dict[int, float]:
        """Prescribe a constant face value in a declared built-in scalar/vector basis.

        This prescribes the interface variable itself, without interpreting a PDE
        boundary condition. Nonconstant or custom data use a declared projection.
        """
        space = getattr(self.interface, "space", None)
        if space is None or not hasattr(space, "faces"):
            raise TypeError("custom interface must declare its own face-value projection")
        constant = real_array([value], "face value")[0]
        fixed: dict[int, float] = {}
        for face in faces:
            face = positive_int(face, "face", 0)
            if face >= len(space.faces):
                raise ValueError("face index outside interface mesh")
            face_dofs = getattr(space, "dofs", None)
            dofs = face_dofs(face) if callable(face_dofs) else np.asarray(space.face_dofs[face])
            coefficients = np.repeat(
                space.faces[face].constant_coefficients(), getattr(space, "components", 1)
            )
            fixed.update(zip(dofs.tolist(), (constant * coefficients).tolist(), strict=True))
        return fixed


@dataclass
class LocalContext:
    """One worker-owned macrocell's mesh, spaces and trace coordinate binding.

    Local native spaces are created lazily. The context itself never travels back
    to the coordinator; equation metadata must contain portable evaluation data.
    Custom local solvers may use only mesh/binding and assembled numerical blocks.
    """

    global_context: GlobalContext
    cell: int
    compiler: FormCompiler = compile_form
    _binding: TraceBinding = field(init=False, repr=False)
    _mesh: Any = field(default=None, init=False, repr=False)
    _native: list[Any] = field(default_factory=list, init=False, repr=False)
    _native_mesh: Any = field(default=None, init=False, repr=False)
    _fields: list[Any] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Check selected entity and trace association without creating native objects."""
        if self.cell not in self.global_context.hierarchy.items:
            raise ValueError("cell is not selected in this mesh hierarchy")
        self._binding = validate_trace_binding(
            self.global_context.interface.binding(self.cell), self.global_context.trace_size
        )

    @property
    def mesh(self) -> Any:
        """Return the lazily constructed portable local mesh."""
        if self._mesh is None:
            self._mesh = self.global_context.hierarchy.local_mesh(self.cell)
        return self._mesh

    @property
    def macro(self) -> Any:
        """Return the hierarchy's macro mesh for declared material/geometry data."""
        return self.global_context.hierarchy.macro

    @property
    def binding(self) -> TraceBinding:
        """Return this context's single executed coefficient-map snapshot."""
        return self._binding

    def native_space(
        self, element: Any = None, *, degree: int = 1, shape: tuple[int, ...] = ()
    ) -> Any:
        """Bind a user element, or equispaced Lagrange default, to this local mesh.

        Native optional libraries load only on this call. Returned NativeSpace
        exposes its FunctionSpace and coordinate conversions for advanced use.
        Its resources belong to this context and are released by close().
        """
        from dolfinx import fem

        from pymhm.backends.spaces import bind_space, create_native_mesh

        if hasattr(element, "dofmap") and hasattr(element, "mesh"):
            if self._native_mesh is not None and element.mesh is not self._native_mesh:
                raise ValueError("local native spaces must share their context's integration mesh")
            space = bind_space(self.mesh, element)
            self._native_mesh = space.mesh
        else:
            if self._native_mesh is None:
                self._native_mesh = create_native_mesh(self.mesh)
            if element is None:
                import basix
                import basix.ufl

                element = basix.ufl.element(
                    "Lagrange",
                    self._native_mesh.topology.cell_name(),
                    degree,
                    shape=shape,
                    lagrange_variant=basix.LagrangeVariant.equispaced,
                )
            space = bind_space(self.mesh, fem.functionspace(self._native_mesh, element))
        self._native.append(space)
        return space

    def trace_pairings(
        self,
        expression: Callable[[Any, Any], Any],
        *,
        axis: Literal["columns", "rows"] = "columns",
        interface: InterfaceSpace | None = None,
    ) -> Any:
        """Bind a user-written boundary form to each supported local trace basis.

        expression(trace_value, face_measure) returns its UFL linear form. The
        trace value has the declared scalar/vector shape; geometry orientation
        is applied once by equations(), not hidden in the expression. Built-in
        planar polynomial traces have a native adapter; custom spaces may supply
        a trace_pairings(context, expression, axis=...) capability explicitly.
        interface selects an independent local boundary basis when a formulation
        has local trace unknowns as well as its global interface. Returned forms
        use unsigned local basis coordinates; equations() transports only the
        primary global pairings, while local operator blocks retain their basis.
        """
        selected = self.global_context.interface if interface is None else interface
        custom = getattr(selected, "trace_pairings", None)
        if callable(custom):
            return custom(self, expression, axis=axis)
        if not self._native:
            raise ValueError("bind a native local space before defining trace pairings")
        from pymhm.backends.traces import trace_pairings

        return trace_pairings(self, self._native[-1], expression, axis=axis, interface=selected)

    def interface_pairing(self, test_interface: InterfaceSpace, *, order: int = 6) -> FloatArray:
        """Pair an independent local boundary basis with the primary interface basis.

        The built-in operation is the unsigned L2 boundary mass matrix, with
        local test coordinates in rows and local primary trial coordinates in
        columns. Face numbering, union partitions and shared-vertex accumulation
        belong to the trace owner. Users choose its signs and block placement.
        A custom test interface may define interface_pairing(context, order=...).
        Primary global maps are applied later by equations(), exactly once.
        """
        custom = getattr(test_interface, "interface_pairing", None)
        if callable(custom):
            return custom(self, order=order)
        test_space = getattr(test_interface, "space", None)
        trial_space = getattr(self.global_context.interface, "space", None)
        if test_space is None or trial_space is None:
            raise TypeError("custom interface must declare a boundary-pairing capability")
        from pymhm.fem.traces.interval import interface_pairing

        return interface_pairing(test_space, trial_space, self.cell, order=order)

    def equations(
        self, *, coordinates: Literal["local", "global"] = "local", **forms: Any
    ) -> LocalEquations:
        """Bind declared forms without exposing global DOF numbering to the provider.

        Local trace coordinates are transported through the interface maps.
        coordinates='global' explicitly retains already oriented existing blocks.
        No coupling transpose, retained mode, moment or physical sign is inferred.
        """
        equations = bind_local_equations(
            validate_trace_binding(self.binding, self.global_context.trace_size),
            coordinates=coordinates,
            compiler=self.compiler,
            **forms,
        )
        return replace(equations, field_data=tuple(self._fields), trace_binding=self.binding)

    def field(
        self,
        name: str,
        space: Any = None,
        *,
        component: int | None = None,
        evaluator: Callable[..., Any] | None = None,
        gradient_evaluator: Callable[..., Any] | None = None,
        reconstruction: Any = None,
        mesh: Any = None,
        basis_id: str = "",
    ) -> None:
        """Declare a named native or custom field without transferring live resources.

        space is an optional NativeSpace; component selects a mixed subfield.
        Custom fields provide an evaluator and basis_id, or expose coefficients
        only. reconstruction explicitly maps the local solution into field data.
        """
        from pymhm.postprocessing.fields import FieldDefinition

        if any(definition.name == name for definition in self._fields):
            raise ValueError("a field name must be unique within its local context")
        descriptor = None if space is None else space.descriptor(component=component)
        self._fields.append(
            FieldDefinition(
                name,
                self.mesh if mesh is None else mesh,
                descriptor,
                evaluator,
                reconstruction,
                basis_id,
                gradient_evaluator,
            )
        )

    def nested(
        self,
        problem: MultiscaleProblem[Any],
        *,
        boundary_dofs: Any = None,
        trace_map: Any = None,
        **options: Any,
    ) -> NestedEquations:
        """Bind a child problem to the parent interface through an exact restriction.

        Automatic restriction reuses the existing planar normal-trace owner for
        bound SkeletonSpaces. Supplied boundary_dofs/trace_map declare a complete
        map from selected parent GLOBAL coefficients into child boundary trace
        coefficients; it is not oriented again. Child boundary/gauge, injectivity
        and parallel restrictions remain those of NestedEquations. No projection
        of an unrepresentable parent trace is silently introduced.
        """
        if (boundary_dofs is None) != (trace_map is None):
            raise ValueError("declare both child boundary coordinates and their trace map")
        if trace_map is None:
            from pymhm.core.nested import nested_trace_map
            from pymhm.core.spaces import BoundInterface
            from pymhm.fem.traces.interval import SkeletonSpace

            outer = self.global_context.interface
            inner = getattr(getattr(problem, "context", None), "interface", None)
            if not (
                isinstance(outer, BoundInterface)
                and isinstance(inner, BoundInterface)
                and outer.convention == inner.convention == "normal"
                and isinstance(outer.space, SkeletonSpace)
                and isinstance(inner.space, SkeletonSpace)
                and outer.mesh.points.shape[1] == inner.mesh.points.shape[1] == 2
            ):
                raise TypeError("this hierarchy requires an explicit child trace restriction")
            boundary_dofs, trace_map = nested_trace_map(outer.space, self.cell, inner.space)
        return NestedEquations(
            problem,
            boundary_dofs,
            trace_map,
            self.binding.dofs,
            trace_binding=self.binding,
            **options,
        )

    def close(self) -> None:
        """Release native spaces while retaining independently owned numerical data."""
        for space in reversed(self._native):
            space.close()
        self._native.clear()
        self._native_mesh = None


@dataclass
class _ContextProvider:
    """Create local contexts in workers and delegate the mathematical provider."""

    context: GlobalContext
    provider: Callable[[LocalContext], Any]
    compiler: FormCompiler

    def __call__(self, cell: int) -> Any:
        """Supply one existing equation contract and always release native resources."""
        local = LocalContext(self.context, cell, self.compiler)
        try:
            result = self.provider(local)
            if not isinstance(result, (LocalEquations, NestedEquations)):
                from pymhm.core.equations import CompiledLocalEquations

                if not isinstance(result, CompiledLocalEquations):
                    raise TypeError("local provider must return an existing equation contract")
            return result
        finally:
            local.close()

    def prepare_runtime(self) -> None:
        """Forward worker-owned native initialization before numerical thread limits."""
        prepare = getattr(self.provider, "prepare_runtime", None)
        if callable(prepare):
            prepare()

    def close(self) -> None:
        """Delegate shutdown of explicitly reusable provider resources."""
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True, init=False, kw_only=True)
class BoundProblem(MultiscaleProblem[int]):
    """An existing multiscale problem with mesh-associated variational contexts."""

    context: GlobalContext

    def local_context(self, cell: int) -> LocalContext:
        """Create an explicit context for diagnostics or a manual provider invocation.

        The caller owns this context and must close any native resources it creates.
        Normal assemble/solve calls manage that lifetime automatically in workers.
        """
        return LocalContext(self.context, cell, self.compiler)


def bind_problem(
    hierarchy: MeshHierarchy,
    interface: InterfaceSpace,
    local_provider: Callable[[LocalContext], Any],
    *,
    global_equation: Equation | Callable[[GlobalContext], Equation] = _EMPTY_EQUATION,
    retained: int | tuple[int, ...] = 0,
    fixed: Mapping[int, float] | Callable[[GlobalContext], Mapping[int, float]] | None = None,
    constraints: tuple[tuple[FloatArray, float], ...] = (),
    compiler: FormCompiler = compile_form,
) -> BoundProblem:
    """Bind meshes/spaces and user forms to the existing assemble/solve pipeline.

    retained declares the number of retained modes per selected cell; a scalar
    count repeats it, while a tuple follows hierarchy item order. The provider
    receives LocalContext rather than an integer. A global callback can use the
    declared layout to express additional interface/retained terms. Fully manual
    MultiscaleProblem and custom InterfaceSpace implementations remain available.
    """
    if not isinstance(hierarchy, MeshHierarchy):
        raise TypeError("hierarchy must be MeshHierarchy")
    if not callable(local_provider) or not callable(getattr(interface, "binding", None)):
        raise TypeError("local provider and interface binding must be callable")
    positive_int(interface.size, "interface size", 0)
    mesh = getattr(interface, "mesh", hierarchy.macro)
    if mesh is not hierarchy.macro:
        raise ValueError("interface and hierarchy must declare the same macro mesh")
    count = len(hierarchy.items)
    sizes = (
        (positive_int(retained, "retained modes", 0),) * count
        if isinstance(retained, (int, np.integer))
        else tuple(positive_int(size, "retained modes", 0) for size in retained)
    )
    if len(sizes) != count:
        raise ValueError("declare retained dimensions in hierarchy item order")
    context = GlobalContext(hierarchy, interface, sizes)
    for cell in hierarchy.items:
        validate_trace_binding(interface.binding(cell), context.trace_size)
    equation = global_equation(context) if callable(global_equation) else global_equation
    coefficients = fixed(context) if callable(fixed) else fixed
    result = BoundProblem.__new__(BoundProblem)
    MultiscaleProblem.__init__(
        result,
        equation,
        _ContextProvider(context, local_provider, compiler),
        hierarchy.items,
        context.trace_size,
        sizes,
        fixed=coefficients,
        constraints=constraints,
        compiler=compiler,
    )
    object.__setattr__(result, "context", context)
    return result
