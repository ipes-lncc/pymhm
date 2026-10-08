"""BDM-family forms in the declared normal and interior coefficient convention."""

from pymhm.core.validation import FloatArray
from pymhm.fem.hdiv.bdm import bdm2_basis, bdm2_evaluate, bdm2_trace_map
from pymhm.fem.hdiv.bdm_family import BDMFamily
from pymhm.fem.hdiv.mixed import triangle_pressure_basis
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


def bdm_pressure_basis(family: BDMFamily, bary: FloatArray) -> FloatArray:
    """Return a cardinal complete pressure space with partition of unity."""
    return triangle_pressure_basis(family.polynomial_degree - 1, bary)


def bdm_basis(
    family: BDMFamily, mesh: TriangleMesh, bary: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Tabulate BDM fluxes in their declared normal/interior moment coordinates."""
    return bdm2_basis(mesh, bary) if family == BDMFamily() else family.basis(mesh, bary)


def bdm_evaluate(
    family: BDMFamily, mesh: TriangleMesh, flux: FloatArray, bary: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Evaluate vectors in the same executed coordinate convention as assembly."""
    return (
        bdm2_evaluate(mesh, flux, bary)
        if family == BDMFamily()
        else family.evaluate(mesh, flux, bary)
    )


def bdm_trace_map(
    family: BDMFamily, mesh: TriangleMesh, cell: int, fine: TriangleMesh, skeleton: SkeletonSpace
) -> FloatArray:
    """Map the physical multiplier into the family's oriented boundary moments."""
    return (
        bdm2_trace_map(mesh, cell, fine, skeleton)
        if family == BDMFamily()
        else family.trace_map(mesh, cell, fine, skeleton)
    )
