"""Compatibility imports for the public numerical operator owner."""

from pymhm.fem.vector.inequalities_3d import strain_inverse_bound_3d as _compat_strain_inverse_bound
from pymhm.fem.vector.pressure_3d import (
    ElasticityPressure3DOperators as ElasticityPressure3DOperators,
)
from pymhm.fem.vector.pressure_3d import elasticity_contract_3d as elasticity_contract_3d
from pymhm.fem.vector.pressure_3d import (
    tetra_elasticity_pressure_operators as tetra_elasticity_pressure_operators,
)

_strain_inverse_bound = _compat_strain_inverse_bound
