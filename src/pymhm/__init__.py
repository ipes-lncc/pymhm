"""Composable Multiscale Hybrid Mixed finite element methods.

The portable reference backend requires only NumPy, SciPy and threadpoolctl.
FEniCS, meshing and accelerator adapters import their dependencies on demand.
"""

from pymhm.adaptive_darcy import AdaptiveDarcyResult, mark_dorfler, solve_adaptive_darcy
from pymhm.adaptive_darcy3d import AdaptiveDarcy3DResult, solve_adaptive_darcy_3d
from pymhm.adaptive_darcy_balanced import BalancedDarcyResult, solve_balanced_adaptive_darcy
from pymhm.adaptive_darcy_budget import DarcyBudgetRefinement, refine_darcy_budget
from pymhm.analytic import (
    AnalyticDarcySolution,
    AnalyticDarcySpace,
    analytic_darcy_local,
    solve_darcy_analytic,
)
from pymhm.bdm_family import BDMFamily
from pymhm.block import SaddleBlockSolver
from pymhm.conforming import ConformingQuadrilateralSolution, solve_conforming_quadrilateral
from pymhm.cut_cells import (
    cartesian_edge_quadrature,
    cartesian_trace_values,
    fit_material_faces,
    fit_material_mesh,
    material_triangle_quadrature,
)
from pymhm.darcy import DarcySolution, solve_darcy
from pymhm.darcy3d import Darcy3DSolution, TriangularSkeleton, solve_darcy_3d
from pymhm.darcy_hdiv3d import Mixed3DDarcySolution, Mixed3DSkeleton, solve_darcy_hdiv3d
from pymhm.darcy_jump_estimator import DarcyJumpEstimator, estimate_darcy_jumps
from pymhm.darcy_local_error import LocalDarcyRefinementEstimate, estimate_darcy_local_refinement
from pymhm.darcy_mixed import BDMDarcySolution, solve_darcy_bdm
from pymhm.darcy_rt import RTDarcySolution, solve_darcy_rt, solve_darcy_rt_conforming
from pymhm.darcy_transport import HydrodynamicDispersion, RT0DarcyVelocity, solve_darcy_transport
from pymhm.darcy_velocity import (
    PolynomialDarcyVelocity,
    PrimalDarcyVelocity,
    polynomial_darcy_velocity,
)
from pymhm.distributed import DistributedHybridSolution, solve_distributed
from pymhm.elasticity import ElasticitySolution, solve_displacement_pressure
from pymhm.elasticity3d import Elasticity3DSolution, solve_elasticity_3d
from pymhm.elasticity_estimator import PrimalElasticityIndicator, estimate_primal_elasticity_error
from pymhm.elasticity_mixed import MixedElasticitySolution, solve_elasticity_mixed
from pymhm.elasticity_mixed3d import (
    MixedElasticity3DSolution,
    TractionSkeleton3D,
    solve_elasticity_mixed_3d,
)
from pymhm.elasticity_primal import (
    PrimalElasticitySolution,
    constitutive_values,
    solve_primal_elasticity,
)
from pymhm.elasticity_tensor_rt import TensorRTElasticitySolution, solve_elasticity_tensor_rt
from pymhm.elastodynamics import (
    ElastodynamicSolution,
    ElastodynamicStepper,
    solve_elastodynamics,
)
from pymhm.estimator import (
    ConformingPotential,
    DarcyEstimator,
    estimate_darcy_error,
    recover_potential,
)
from pymhm.estimator3d import (
    ConformingPotential3D,
    Darcy3DEstimator,
    estimate_darcy_error_3d,
    recover_potential_3d,
)
from pymhm.flow3d import Flow3DSolution, solve_flow_3d
from pymhm.flow3d_forms import Flow3DOperators, tetra_flow_operators
from pymhm.flow_adaptive import AdaptiveFlowResult, adapt_flow, mark_flow_faces
from pymhm.flow_estimator import FlowEstimator, estimate_flow_error
from pymhm.flow_macro_adaptive import AdaptiveFlowMacroResult, adapt_flow_macros, mark_flow_cells
from pymhm.gals3d import GaLS3DSolution, solve_elasticity_gals_3d
from pymhm.gals3d_forms import ElasticityPressure3DOperators, tetra_elasticity_pressure_operators
from pymhm.gpu import BatchedFactorization, assemble_p1_batch, condense_batched
from pymhm.hdiv3d_family import HDiv3DFamily
from pymhm.hdiv3d_mesh import AffineMixedMesh
from pymhm.helmholtz import HelmholtzSolution, solve_helmholtz
from pymhm.helmholtz_spaces import OscillatoryFaceSpace, PolynomialNeumannTrace, helmholtz_skeleton
from pymhm.hybrid import HybridSolution, HybridSystem, LocalAssembly, LocalProblem, LocalResponse
from pymhm.hybrid_refinement import (
    HybridRefinement,
    HybridRefinementCase,
    HybridRefinementLocal,
    HybridRefinementStore,
    HybridStreamRefinement,
    refine_hybrid,
    refine_hybrid_stream,
)
from pymhm.longest_edge import refine_longest_edge
from pymhm.mapped_rt import HexMesh, HexSkeleton, MappedRTDarcySolution, solve_darcy_mapped_rt
from pymhm.maxwell import MaxwellSolution, MaxwellStepper, solve_maxwell
from pymhm.maxwell_dg import MaxwellSkeleton
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.mesh_exchange import (
    VolumeMeshData,
    from_meshio,
    read_volume_mesh,
    to_meshio,
    write_volume_mesh,
)
from pymhm.meshing3d import TetraMeshData, read_tetra_mesh, write_tetra_mesh
from pymhm.meshing_native3d import from_gmsh_3d, from_netgen_3d
from pymhm.metric_adapt import ResidualMeshSize, remesh_freefem, residual_mesh_size
from pymhm.mh import MHSolution, solve_mh
from pymhm.mh2m import MH2MLocal, MH2MSolution, PressureTraceSpace, solve_mh2m
from pymhm.mh2m3d import MH2M3DSolution, solve_mh2m_3d
from pymhm.mh3d import MH3DSolution, solve_mh_3d
from pymhm.mh_trace3d import PressureTraceSpace3D
from pymhm.mshho import MsHHOLocal, MsHHOSolution, solve_mshho
from pymhm.mshho3d import MsHHO3DSolution, solve_mshho_3d
from pymhm.nested import NestedLocalProblem, NestedSolution, nest_hybrid_system, nested_trace_map
from pymhm.offline import (
    LocalFactorCache,
    OfflineHybridSystem,
    OfflineLocalProblem,
    condense_cached,
)
from pymhm.pgmhm import PGMHMSolution, solve_pgmhm
from pymhm.planar_fitting import (
    PlanarFittedMesh,
    fit_planar_material,
    fit_planar_skeleton,
    planar_face_partitions,
)
from pymhm.planar_material import PlanarMaterial, PlanarRegion
from pymhm.planar_quadrature import planar_edge_quadrature, planar_simplex_quadrature
from pymhm.polygon import (
    PolygonMesh,
    solve_brinkman_polygons,
    solve_darcy_polygons,
    solve_elasticity_mixed_polygons,
    solve_mshho_polygons,
    solve_transport_polygons,
)
from pymhm.polyhedral import PolyhedralMesh
from pymhm.polyhedral_rad import PolygonalSkeleton3D, PolyhedralRADSolution, solve_polyhedral_rad
from pymhm.quadrilateral import (
    CartesianMacroMesh,
    QuadrilateralDarcySolution,
    solve_darcy_quadrilateral,
)
from pymhm.rad3d import RAD3DSolution, solve_rad_3d, solve_rad_3d_conforming
from pymhm.reconstruction3d import MomentFlux3DSolution, reconstruct_darcy_moments_3d
from pymhm.reconstruction_moments import (
    MomentFluxSolution,
    reconstruct_darcy_moments,
    reconstruct_flux_moments,
)
from pymhm.refinement import (
    TriangleRefinement,
    refine_triangles,
    transfer_skeleton,
    validate_submesh,
)
from pymhm.refinement3d import TetraRefinement, refine_tetrahedra
from pymhm.rt import RTField
from pymhm.rt3d import RTTetraFamily
from pymhm.scalar_adaptive import (
    AdaptiveTransportResult,
    FaceIndicators,
    TransportBounds,
    estimate_transport_faces,
    refine_skeleton_faces,
    solve_adaptive_transport,
)
from pymhm.scalar_transient import (
    MacroCoefficient,
    TransientTransportResult,
    solve_transient_transport,
)
from pymhm.separable import (
    SeparableField,
    separable_diffusion_operators,
    solve_separable_diffusion,
)
from pymhm.separable_krylov import (
    SeparableKrylovSolution,
    TensorDiffusionOperator,
    solve_separable_krylov,
    tensor_diffusion_operator,
)
from pymhm.subspaces import restrict_response
from pymhm.tensor_rt import TensorRTDarcySolution, solve_darcy_tensor_rt
from pymhm.tetrahedral import TetraMesh
from pymhm.transport import ScalarSolution, solve_heat, solve_transport
from pymhm.triangle_fields import PolylineLayerField, RadialDiskLoad
from pymhm.unusual import UnusualParameters
from pymhm.vector import VectorSolution, solve_brinkman, solve_elasticity
from pymhm.weighted_estimator import (
    PublishedDarcyIndicator,
    WeightedDarcyEstimator,
    estimate_darcy_indicator,
    estimate_weighted_darcy_error,
    recover_dirichlet_potential,
)

__version__ = "0.1.0"

__all__ = [
    "ElastodynamicSolution",
    "ElastodynamicStepper",
    "solve_elastodynamics",
    "MixedElasticity3DSolution",
    "TractionSkeleton3D",
    "solve_elasticity_mixed_3d",
    "HelmholtzSolution",
    "solve_helmholtz",
    "OscillatoryFaceSpace",
    "PolynomialNeumannTrace",
    "helmholtz_skeleton",
    "GaLS3DSolution",
    "solve_elasticity_gals_3d",
    "ElasticityPressure3DOperators",
    "tetra_elasticity_pressure_operators",
    "PGMHMSolution",
    "solve_pgmhm",
    "AdaptiveDarcy3DResult",
    "solve_adaptive_darcy_3d",
    "ConformingPotential3D",
    "Darcy3DEstimator",
    "estimate_darcy_error_3d",
    "recover_potential_3d",
    "Flow3DOperators",
    "tetra_flow_operators",
    "VolumeMeshData",
    "from_meshio",
    "to_meshio",
    "read_volume_mesh",
    "write_volume_mesh",
    "from_gmsh_3d",
    "from_netgen_3d",
    "MHSolution",
    "MaxwellSkeleton",
    "MaxwellSolution",
    "MaxwellStepper",
    "solve_maxwell",
    "solve_mh",
    "PlanarMaterial",
    "PolylineLayerField",
    "RadialDiskLoad",
    "planar_edge_quadrature",
    "planar_simplex_quadrature",
    "PlanarRegion",
    "PlanarFittedMesh",
    "fit_planar_material",
    "fit_planar_skeleton",
    "planar_face_partitions",
    "MomentFlux3DSolution",
    "reconstruct_darcy_moments_3d",
    "TetraRefinement",
    "refine_tetrahedra",
    "RTField",
    "RTTetraFamily",
    "Flow3DSolution",
    "solve_flow_3d",
    "MH2MLocal",
    "MH2MSolution",
    "MH2M3DSolution",
    "MH3DSolution",
    "PressureTraceSpace",
    "PressureTraceSpace3D",
    "solve_mh2m",
    "solve_mh2m_3d",
    "solve_mh_3d",
    "MsHHO3DSolution",
    "solve_mshho_3d",
    "solve_elasticity_mixed_polygons",
    "UnusualParameters",
    "AffineMixedMesh",
    "HDiv3DFamily",
    "Mixed3DDarcySolution",
    "Mixed3DSkeleton",
    "solve_darcy_hdiv3d",
    "DarcyBudgetRefinement",
    "refine_darcy_budget",
    "ResidualMeshSize",
    "remesh_freefem",
    "residual_mesh_size",
    "AdaptiveFlowMacroResult",
    "adapt_flow_macros",
    "mark_flow_cells",
    "PolyhedralMesh",
    "PolygonalSkeleton3D",
    "PolyhedralRADSolution",
    "solve_polyhedral_rad",
    "SeparableKrylovSolution",
    "TensorDiffusionOperator",
    "solve_separable_krylov",
    "tensor_diffusion_operator",
    "LocalDarcyRefinementEstimate",
    "estimate_darcy_local_refinement",
    "SeparableField",
    "separable_diffusion_operators",
    "solve_separable_diffusion",
    "BalancedDarcyResult",
    "solve_balanced_adaptive_darcy",
    "DarcyJumpEstimator",
    "estimate_darcy_jumps",
    "AnalyticDarcySolution",
    "AnalyticDarcySpace",
    "analytic_darcy_local",
    "solve_darcy_analytic",
    "Elasticity3DSolution",
    "solve_elasticity_3d",
    "HexMesh",
    "HexSkeleton",
    "MappedRTDarcySolution",
    "solve_darcy_mapped_rt",
    "AdaptiveDarcyResult",
    "mark_dorfler",
    "solve_adaptive_darcy",
    "SaddleBlockSolver",
    "ConformingQuadrilateralSolution",
    "solve_conforming_quadrilateral",
    "fit_material_mesh",
    "Darcy3DSolution",
    "TriangularSkeleton",
    "solve_darcy_3d",
    "RTDarcySolution",
    "solve_darcy_rt",
    "solve_darcy_rt_conforming",
    "PrimalElasticityIndicator",
    "estimate_primal_elasticity_error",
    "PrimalElasticitySolution",
    "constitutive_values",
    "solve_primal_elasticity",
    "TensorRTElasticitySolution",
    "solve_elasticity_tensor_rt",
    "TetraMeshData",
    "read_tetra_mesh",
    "write_tetra_mesh",
    "NestedLocalProblem",
    "NestedSolution",
    "nest_hybrid_system",
    "nested_trace_map",
    "RAD3DSolution",
    "solve_rad_3d",
    "solve_rad_3d_conforming",
    "TriangleRefinement",
    "refine_triangles",
    "refine_longest_edge",
    "transfer_skeleton",
    "validate_submesh",
    "restrict_response",
    "TetraMesh",
    "WeightedDarcyEstimator",
    "PublishedDarcyIndicator",
    "estimate_darcy_indicator",
    "estimate_weighted_darcy_error",
    "recover_dirichlet_potential",
    "cartesian_edge_quadrature",
    "cartesian_trace_values",
    "fit_material_faces",
    "material_triangle_quadrature",
    "BDMFamily",
    "AdaptiveFlowResult",
    "adapt_flow",
    "mark_flow_faces",
    "FlowEstimator",
    "estimate_flow_error",
    "AdaptiveTransportResult",
    "FaceIndicators",
    "TransportBounds",
    "estimate_transport_faces",
    "refine_skeleton_faces",
    "solve_adaptive_transport",
    "MacroCoefficient",
    "TransientTransportResult",
    "solve_transient_transport",
    "HydrodynamicDispersion",
    "RT0DarcyVelocity",
    "PolynomialDarcyVelocity",
    "PrimalDarcyVelocity",
    "polynomial_darcy_velocity",
    "solve_darcy_transport",
    "DistributedHybridSolution",
    "solve_distributed",
    "BatchedFactorization",
    "assemble_p1_batch",
    "condense_batched",
    "PolygonMesh",
    "solve_brinkman_polygons",
    "solve_darcy_polygons",
    "solve_mshho_polygons",
    "solve_transport_polygons",
    "CartesianMacroMesh",
    "ConformingPotential",
    "DarcyEstimator",
    "estimate_darcy_error",
    "recover_potential",
    "DarcySolution",
    "BDMDarcySolution",
    "solve_darcy_bdm",
    "ElasticitySolution",
    "solve_displacement_pressure",
    "FaceSpace",
    "HybridSolution",
    "HybridSystem",
    "HybridRefinement",
    "refine_hybrid",
    "HybridRefinementCase",
    "HybridRefinementLocal",
    "HybridRefinementStore",
    "HybridStreamRefinement",
    "refine_hybrid_stream",
    "LocalAssembly",
    "LocalProblem",
    "LocalResponse",
    "LocalFactorCache",
    "OfflineHybridSystem",
    "OfflineLocalProblem",
    "condense_cached",
    "MsHHOLocal",
    "MsHHOSolution",
    "solve_mshho",
    "TensorRTDarcySolution",
    "solve_darcy_tensor_rt",
    "MixedElasticitySolution",
    "MomentFluxSolution",
    "QuadrilateralDarcySolution",
    "reconstruct_darcy_moments",
    "reconstruct_flux_moments",
    "SkeletonSpace",
    "TriangleMesh",
    "VectorSolution",
    "solve_brinkman",
    "solve_darcy",
    "solve_darcy_quadrilateral",
    "solve_elasticity",
    "solve_elasticity_mixed",
    "ScalarSolution",
    "solve_heat",
    "solve_transport",
]
