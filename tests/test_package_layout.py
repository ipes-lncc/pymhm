"""Canonical public imports reject removed paths and leave native adapters unopened."""

from __future__ import annotations

import importlib
import inspect
import pickle
import subprocess
import sys
from dataclasses import is_dataclass
from pathlib import Path

import pytest

import pymhm
from pymhm._registry import PUBLIC_EXPORTS

REMOVED_MODULE_PATHS = (
    "pymhm._compat",
    "pymhm._geometry_roundoff",
    "pymhm.adaptive_darcy",
    "pymhm.adaptive_darcy3d",
    "pymhm.adaptive_darcy_balanced",
    "pymhm.adaptive_darcy_budget",
    "pymhm.analytic",
    "pymhm.assembly",
    "pymhm.bdm",
    "pymhm.bdm_family",
    "pymhm.block",
    "pymhm.conforming",
    "pymhm.core.hybrid",
    "pymhm.crisscross",
    "pymhm.cut_cells",
    "pymhm.darcy",
    "pymhm.darcy3d",
    "pymhm.darcy_hdiv3d",
    "pymhm.darcy_jump_estimator",
    "pymhm.darcy_local_error",
    "pymhm.darcy_mixed",
    "pymhm.darcy_rt",
    "pymhm.darcy_transport",
    "pymhm.darcy_velocity",
    "pymhm.distributed",
    "pymhm.elasticity",
    "pymhm.elasticity3d",
    "pymhm.elasticity_compatibility",
    "pymhm.elasticity_compliance",
    "pymhm.elasticity_estimator",
    "pymhm.elasticity_mixed",
    "pymhm.elasticity_mixed3d",
    "pymhm.elasticity_mixed3d_forms",
    "pymhm.elasticity_primal",
    "pymhm.elasticity_tensor_rt",
    "pymhm.elastodynamics",
    "pymhm.element_backends",
    "pymhm.elements",
    "pymhm.estimator",
    "pymhm.estimator3d",
    "pymhm.estimator_spaces",
    "pymhm.fem.tensor.elasticity",
    "pymhm.fem.tensor.elasticity_3d",
    "pymhm.fem.vector.elasticity",
    "pymhm.fem.vector.maxwell",
    "pymhm.fem.vector.stress",
    "pymhm.fem.vector.tensor_rt",
    "pymhm.fenics",
    "pymhm.flow",
    "pymhm.flow3d",
    "pymhm.flow3d_forms",
    "pymhm.flow_adaptive",
    "pymhm.flow_estimator",
    "pymhm.flow_local_refinement",
    "pymhm.flow_macro_adaptive",
    "pymhm.gals3d",
    "pymhm.gals3d_forms",
    "pymhm.gpu",
    "pymhm.hdiv3d_family",
    "pymhm.hdiv3d_general",
    "pymhm.hdiv3d_mesh",
    "pymhm.hdiv_reference",
    "pymhm.helmholtz",
    "pymhm.helmholtz_forms",
    "pymhm.helmholtz_spaces",
    "pymhm.hybrid",
    "pymhm.hybrid_refinement",
    "pymhm.lagrange",
    "pymhm.loads",
    "pymhm.longest_edge",
    "pymhm.mapped_rt",
    "pymhm.maxwell",
    "pymhm.maxwell_dg",
    "pymhm.mesh",
    "pymhm.mesh_exchange",
    "pymhm.meshing",
    "pymhm.meshing3d",
    "pymhm.meshing_native3d",
    "pymhm.metric_adapt",
    "pymhm.mh",
    "pymhm.mh2m",
    "pymhm.mh2m3d",
    "pymhm.mh3d",
    "pymhm.mh_boundary",
    "pymhm.mh_trace3d",
    "pymhm.models",
    "pymhm.models.darcy",
    "pymhm.models.darcy._mixed",
    "pymhm.models.darcy.analytic",
    "pymhm.models.darcy.cartesian",
    "pymhm.models.darcy.conforming",
    "pymhm.models.darcy.hdiv_3d",
    "pymhm.models.darcy.mapped",
    "pymhm.models.darcy.mixed_bdm",
    "pymhm.models.darcy.mixed_rt",
    "pymhm.models.darcy.primal",
    "pymhm.models.darcy.primal_3d",
    "pymhm.models.darcy.separable",
    "pymhm.models.darcy.velocity",
    "pymhm.models.elasticity",
    "pymhm.models.elasticity.boundary",
    "pymhm.models.elasticity.mixed_pressure",
    "pymhm.models.elasticity.mixed_pressure_3d",
    "pymhm.models.elasticity.pressure_forms_3d",
    "pymhm.models.elasticity.primal",
    "pymhm.models.elasticity.primal_3d",
    "pymhm.models.elasticity.stress",
    "pymhm.models.elasticity.stress_3d",
    "pymhm.models.elasticity.stress_forms_3d",
    "pymhm.models.elasticity.stress_tensor",
    "pymhm.models.flow",
    "pymhm.models.flow.forms_3d",
    "pymhm.models.flow.solver",
    "pymhm.models.flow.solver_3d",
    "pymhm.models.geometry",
    "pymhm.models.transport",
    "pymhm.models.transport.dispersion",
    "pymhm.models.transport.polyhedral",
    "pymhm.models.transport.rad",
    "pymhm.models.transport.rad_3d",
    "pymhm.models.transport.solver",
    "pymhm.models.transport.stabilization",
    "pymhm.models.transport.transient",
    "pymhm.models.waves",
    "pymhm.models.waves.elastodynamics",
    "pymhm.models.waves.helmholtz",
    "pymhm.models.waves.maxwell",
    "pymhm.mshho",
    "pymhm.mshho3d",
    "pymhm.nested",
    "pymhm.offline",
    "pymhm.parallel",
    "pymhm.pgmhm",
    "pymhm.planar_fitting",
    "pymhm.planar_material",
    "pymhm.planar_quadrature",
    "pymhm.polygon",
    "pymhm.polyhedral",
    "pymhm.polyhedral_geometry",
    "pymhm.polyhedral_rad",
    "pymhm.quadrilateral",
    "pymhm.rad",
    "pymhm.rad3d",
    "pymhm.reconstruction",
    "pymhm.reconstruction3d",
    "pymhm.reconstruction_moments",
    "pymhm.refinement",
    "pymhm.refinement3d",
    "pymhm.reservoir",
    "pymhm.rt",
    "pymhm.rt3d",
    "pymhm.scalar_adaptive",
    "pymhm.scalar_boundary",
    "pymhm.scalar_trace_integration",
    "pymhm.scalar_transient",
    "pymhm.separable",
    "pymhm.separable_krylov",
    "pymhm.solvers",
    "pymhm.subspaces",
    "pymhm.tensor_rt",
    "pymhm.tetra_lagrange",
    "pymhm.tetra_validation",
    "pymhm.tetrahedral",
    "pymhm.transport",
    "pymhm.triangle_fields",
    "pymhm.unusual",
    "pymhm.variational",
    "pymhm.vector",
    "pymhm.visualization",
    "pymhm.weighted_estimator",
)

REMOVED_ROOT_EXPORTS = (
    "AdaptiveDarcy3DResult",
    "AdaptiveDarcyResult",
    "AdaptiveFlowMacroResult",
    "AdaptiveFlowResult",
    "AdaptiveTransportResult",
    "AnalyticDarcySolution",
    "AnalyticDarcySpace",
    "BDMDarcySolution",
    "BalancedDarcyResult",
    "ConformingPotential",
    "ConformingPotential3D",
    "ConformingQuadrilateralSolution",
    "Darcy3DEstimator",
    "Darcy3DSolution",
    "DarcyBudgetRefinement",
    "DarcyEstimator",
    "DarcyJumpEstimator",
    "DarcySolution",
    "Elasticity3DSolution",
    "ElasticityPressure3DOperators",
    "ElasticitySolution",
    "ElastodynamicSolution",
    "ElastodynamicStepper",
    "FaceIndicators",
    "Flow3DOperators",
    "Flow3DSolution",
    "FlowEstimator",
    "GaLS3DSolution",
    "HelmholtzSolution",
    "HexSkeleton",
    "HydrodynamicDispersion",
    "LocalDarcyRefinementEstimate",
    "MacroCoefficient",
    "MappedRTDarcySolution",
    "MaxwellSkeleton",
    "MaxwellSolution",
    "MaxwellStepper",
    "Mixed3DDarcySolution",
    "Mixed3DSkeleton",
    "MixedElasticity3DSolution",
    "MixedElasticitySolution",
    "PolygonalSkeleton3D",
    "PolyhedralRADSolution",
    "PolynomialDarcyVelocity",
    "PrimalDarcyVelocity",
    "PrimalElasticityIndicator",
    "PrimalElasticitySolution",
    "PublishedDarcyIndicator",
    "QuadrilateralDarcySolution",
    "RAD3DSolution",
    "RT0DarcyVelocity",
    "RTDarcySolution",
    "ScalarSolution",
    "SeparableField",
    "TensorRTDarcySolution",
    "TensorRTElasticitySolution",
    "TractionSkeleton3D",
    "TransientTransportResult",
    "TransportBounds",
    "UnusualParameters",
    "VectorSolution",
    "WeightedDarcyEstimator",
    "adapt_flow",
    "adapt_flow_macros",
    "analytic_darcy_local",
    "constitutive_values",
    "darcy_local_provider",
    "estimate_darcy_error",
    "estimate_darcy_error_3d",
    "estimate_darcy_indicator",
    "estimate_darcy_jumps",
    "estimate_darcy_local_refinement",
    "estimate_flow_error",
    "estimate_primal_elasticity_error",
    "estimate_transport_faces",
    "estimate_weighted_darcy_error",
    "mark_flow_cells",
    "mark_flow_faces",
    "polynomial_darcy_velocity",
    "recover_dirichlet_potential",
    "recover_potential",
    "recover_potential_3d",
    "refine_darcy_budget",
    "refine_skeleton_faces",
    "separable_diffusion_operators",
    "solve_adaptive_darcy",
    "solve_adaptive_darcy_3d",
    "solve_adaptive_transport",
    "solve_balanced_adaptive_darcy",
    "solve_brinkman",
    "solve_brinkman_polygons",
    "solve_conforming_quadrilateral",
    "solve_darcy",
    "solve_darcy_3d",
    "solve_darcy_analytic",
    "solve_darcy_bdm",
    "solve_darcy_hdiv3d",
    "solve_darcy_mapped_rt",
    "solve_darcy_polygons",
    "solve_darcy_quadrilateral",
    "solve_darcy_rt",
    "solve_darcy_rt_conforming",
    "solve_darcy_tensor_rt",
    "solve_darcy_transport",
    "solve_displacement_pressure",
    "solve_elasticity",
    "solve_elasticity_3d",
    "solve_elasticity_gals_3d",
    "solve_elasticity_mixed",
    "solve_elasticity_mixed_3d",
    "solve_elasticity_mixed_polygons",
    "solve_elasticity_tensor_rt",
    "solve_elastodynamics",
    "solve_flow_3d",
    "solve_heat",
    "solve_helmholtz",
    "solve_maxwell",
    "solve_mh",
    "solve_mh2m",
    "solve_mh2m_3d",
    "solve_mh_3d",
    "solve_mshho",
    "solve_mshho_3d",
    "solve_mshho_polygons",
    "solve_pgmhm",
    "solve_polyhedral_rad",
    "solve_primal_elasticity",
    "solve_rad_3d",
    "solve_rad_3d_conforming",
    "solve_separable_diffusion",
    "solve_transient_transport",
    "solve_transport",
    "solve_transport_polygons",
    "tetra_elasticity_pressure_operators",
    "tetra_flow_operators",
)


@pytest.mark.parametrize("module_name", REMOVED_MODULE_PATHS)
def test_removed_module_paths_are_rejected(module_name: str) -> None:
    """Every former import route is absent instead of resolving through an alias."""
    with pytest.raises(ModuleNotFoundError) as failure:
        importlib.import_module(module_name)
    assert failure.value.name is not None
    assert module_name == failure.value.name or module_name.startswith(failure.value.name + ".")


@pytest.mark.parametrize("name,owner", sorted(PUBLIC_EXPORTS.items()))
def test_root_exports_are_canonical_objects(name: str, owner: tuple[str, str]) -> None:
    """The root API returns the implementation held by its declared owner."""
    module, symbol = owner
    assert getattr(pymhm, name) is getattr(importlib.import_module(module), symbol)


@pytest.mark.parametrize("name", REMOVED_ROOT_EXPORTS)
def test_removed_physical_root_exports_are_rejected(name: str) -> None:
    """Physical drivers require explicit owner imports outside the generic root API."""
    assert name not in pymhm.__all__
    assert name not in PUBLIC_EXPORTS
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(pymhm, name)


def test_generic_api_exposes_free_operations_and_equation_records() -> None:
    """User forms compose through functions around explicit problem/solution records."""
    from pymhm import Equation, LocalEquations, MultiscaleProblem

    assert all(is_dataclass(record) for record in (Equation, LocalEquations, MultiscaleProblem))
    for name in ("assemble", "solve", "compile_local_equations", "leaf_moment", "newmark_step"):
        assert inspect.isfunction(getattr(pymhm, name))
    assert set(pymhm.__all__) == set(PUBLIC_EXPORTS)
    assert set(pymhm.__all__) <= set(dir(pymhm))
    with pytest.raises(AttributeError, match="no attribute 'unknown_export'"):
        _ = pymhm.unknown_export


def test_current_pickle_globals_roundtrip_their_canonical_owners() -> None:
    """New records serialize only the modules where their implementations live."""
    from pymhm.core.contracts import LocalProblem
    from pymhm.meshes.triangle import TriangleMesh

    for record in (LocalProblem, TriangleMesh):
        serialized = pickle.dumps(record)
        assert record.__module__.encode() in serialized
        assert pickle.loads(serialized) is record
    with pytest.raises(ModuleNotFoundError, match="pymhm.hybrid"):
        pickle.loads(b"cpymhm.hybrid\nLocalProblem\n.")
    with pytest.raises(ModuleNotFoundError, match="pymhm.mesh"):
        pickle.loads(b"cpymhm.mesh\nTriangleMesh\n.")


def test_cold_namespace_has_no_alias_finder_or_physical_dispatch() -> None:
    """A fresh interpreter has no import hook or hidden root fallback for old APIs."""
    script = """
import sys
import pymhm
assert 'LocalEquations' in dir(pymhm)
for name in ('solve_darcy', 'solve_mh2m', 'DarcySolution', 'MaxwellStepper', 'mesh', 'models'):
    assert name not in dir(pymhm), name
    assert not hasattr(pymhm, name), name
assert 'pymhm._compat' not in sys.modules
assert not any(type(finder).__module__ == 'pymhm._compat' for finder in sys.meta_path)
try:
    from pymhm import solve_darcy
except ImportError:
    pass
else:
    raise AssertionError('removed root solver unexpectedly imported')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_algebraic_import_does_not_load_native_backends() -> None:
    """Root and local algebra imports leave optional/native FEM resources unopened."""
    script = """
import sys
import pymhm
assert 'pymhm.core.contracts' not in sys.modules
from pymhm.core.contracts import LocalProblem
from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.multiscale import assemble
for name in ('basix', 'dolfinx', 'ufl', 'mpi4py', 'cupy', 'gmsh', 'netgen'):
    assert name not in sys.modules, name
assert pymhm.LocalProblem is LocalProblem
assert pymhm.Equation is Equation and pymhm.LocalEquations is LocalEquations
assert pymhm.assemble is assemble
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
