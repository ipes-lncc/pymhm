"""Repeated variational sources with unchanged operators and executed bases."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from copy import copy
from dataclasses import replace
from types import TracebackType
from typing import Any

import numpy as np

from pymhm.core.equations import compile_form
from pymhm.core.multiscale import MultiscaleSolution, MultiscaleSystem, _contribution
from pymhm.core.offline import OfflineLocalProblem


class OfflineMultiscaleSystem:
    """Reuse local factors for arbitrary new volume and interface functionals.

    The input system supplies A/B/C/D, retained trial/test modes, coefficient
    transports and field definitions. Only L, g and the additional global
    functional may change. Loads use the compiled test-coordinate order;
    ``balance_loads`` uses the compiled union of local trace coordinates.
    No physical method, sign, source projection or time scheme is selected.

    This object owns local factor resources and must be closed. Updated systems
    preserve their operator and harmonic lifts literally and are independently
    reconstructible after close. A global factor can be reused through the
    ordinary solve ``factorization`` option. Source-dependent gauge targets
    must be recomputed from the updated system's ``mean_constraint`` or
    ``leaf_moment``. A recursive hierarchy updates its child sources before
    assembling its parent; this leaf-source cache does not flatten that tree.
    """

    def __init__(self, system: MultiscaleSystem, *, solver: str = "scipy") -> None:
        """Factor the declared local constrained operators with explicit ownership."""
        if not isinstance(system, MultiscaleSystem):
            raise TypeError("offline variational system requires MultiscaleSystem")
        if any(record.child is not None for record in system.cells):
            raise ValueError("recursive source updates must be declared in the child hierarchy")
        self.template = system
        self._resources = ExitStack()
        self._closed = False
        try:
            self.locals = tuple(
                self._resources.enter_context(OfflineLocalProblem(response.problem, solver=solver))
                for response in system.responses
            )
            for local, response in zip(self.locals, system.responses, strict=True):
                local.template = response
        except BaseException:
            self.close()
            raise

    def with_loads(
        self,
        local_loads: Sequence[Any],
        *,
        global_load: Any = None,
        balance_loads: Sequence[Any] | None = None,
    ) -> MultiscaleSystem:
        """Return a reconstructed-source system without rebuilding any operator.

        Omitted global/balance loads retain the original additional functional
        and g. Explicit ``global_load`` replaces that additional functional in
        the full trace-plus-retained layout, rather than the condensed RHS.
        Local compatibility contributions are recomputed from the new L.
        """
        if self._closed:
            raise RuntimeError("offline variational system is closed")
        count = len(self.locals)
        if len(local_loads) != count or (balance_loads is not None and len(balance_loads) != count):
            raise ValueError("one volume and optional balance load per assembled cell is required")
        original = self.template
        responses = tuple(
            local.response(load) for local, load in zip(self.locals, local_loads, strict=True)
        )
        cells = tuple(
            replace(
                record,
                equations=replace(
                    record.equations,
                    problem=response.problem,
                    load=record.equations.load
                    if balance_loads is None
                    else compile_form(balance_loads[index], record.equations.load.shape),
                ),
            )
            for index, (record, response) in enumerate(zip(original.cells, responses, strict=True))
        )
        additional = (
            original.global_load
            if global_load is None
            else compile_form(global_load, original.rhs.shape)
        )
        rhs, scale = np.array(additional, copy=True), np.abs(additional)
        for index, (response, record) in enumerate(zip(responses, cells, strict=True)):
            coarse = np.arange(original.kernel_offsets[index], original.kernel_offsets[index + 1])
            indices, _, load = _contribution(response, record, coarse)
            rhs[indices] += load
            scale[indices] += np.abs(load)
        updated = copy(original)
        updated.responses, updated.cells = responses, cells
        updated.rhs, updated.load_scale, updated.global_load = (
            rhs,
            scale,
            np.array(additional, copy=True),
        )
        return updated

    def solve(
        self,
        local_loads: Sequence[Any],
        *,
        global_load: Any = None,
        balance_loads: Sequence[Any] | None = None,
        **options: Any,
    ) -> MultiscaleSolution:
        """Solve an updated source through the existing generic solver contract."""
        return self.with_loads(
            local_loads, global_load=global_load, balance_loads=balance_loads
        ).solve(**options)

    def close(self) -> None:
        """Release owned local factors once; previously returned systems remain usable."""
        self._closed = True
        self._resources.close()

    def __enter__(self) -> OfflineMultiscaleSystem:
        """Enter the explicit lifetime of an open factor cache."""
        if self._closed:
            raise RuntimeError("offline variational system is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release resources on normal completion or an unsuccessful source query."""
        self.close()
