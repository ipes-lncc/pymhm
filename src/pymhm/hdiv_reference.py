"""Basix reference bases in declared H(div) candidate coordinates.

Only polynomial tabulation is delegated here. Physical moment definitions,
normal restrictions, bubble coordinates and Piola maps belong to their callers.
Native handles are cached in each process and never passed to spawn workers.
"""

from pymhm.element_backends import (
    ReferenceElementSpec,
    create_reference_element,
    tabulate_reference,
)
from pymhm.element_backends import (
    bernstein_tabulation as bernstein_tabulation,
)
from pymhm.mesh import FloatArray


def vector_tabulation(
    family: str, cell: str, degree: int, points: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Tabulate a native RT/BDM basis and its Cartesian reference divergence.

    ``degree`` follows Basix: mathematical RT order m uses degree m+1; BDM
    uses its vector polynomial degree. Values have axes (point, DOF, component)
    and divergence (point, DOF). Moment numbering is supplied by the caller.
    """
    element = create_reference_element(
        ReferenceElementSpec(family, cell, degree, lagrange_variant="legendre")
    )
    table = tabulate_reference(element, points, 1)
    divergence = table[1, :, :, 0].copy()
    for axis in range(1, element.cell_dimension):
        divergence += table[axis + 1, :, :, axis]
    return table[0], divergence
