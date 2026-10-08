"""Explicit one-sided coefficients associated with a macro partition."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MacroCoefficient:
    """Explicit one-sided coefficient callbacks, one for each macroelement.

    Entries follow mesh.cells. This wrapper distinguishes a tuple of local
    fields from an ordinary constant vector or tensor. A field is evaluated
    exclusively on the corresponding macroelement, including its boundary.
    """

    fields: tuple[Any, ...]

    def for_cell(self, cell: int, count: int) -> Any:
        """Select a field after checking the coefficient-to-macro association."""
        if len(self.fields) != count:
            raise ValueError("MacroCoefficient requires one field per macroelement")
        return self.fields[cell]
