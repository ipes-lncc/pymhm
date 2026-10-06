"""Typed primary API; import specialized functions from their implementation owners."""

from pymhm.adaptivity.darcy import (
    mark_dorfler as mark_dorfler,
)
from pymhm.adaptivity.metric import (
    ResidualMeshSize as ResidualMeshSize,
)
from pymhm.adaptivity.metric import (
    remesh_freefem as remesh_freefem,
)
from pymhm.adaptivity.metric import (
    residual_mesh_size as residual_mesh_size,
)
from pymhm.backends.forms import (
    assemble_form as assemble_form,
)
from pymhm.backends.forms import (
    assemble_pairing as assemble_pairing,
)
from pymhm.backends.spaces import (
    NativeSpace as NativeSpace,
)
from pymhm.backends.spaces import (
    bind_space as bind_space,
)
from pymhm.backends.spaces import (
    create_native_mesh as create_native_mesh,
)
from pymhm.core.assembly import (
    HybridProblem as HybridProblem,
)
from pymhm.core.assembly import (
    LocalLinearSolver as LocalLinearSolver,
)
from pymhm.core.assembly import (
    SolverConfig as SolverConfig,
)
from pymhm.core.assembly import (
    assemble_hybrid as assemble_hybrid,
)
from pymhm.core.assembly import (
    solve_hybrid as solve_hybrid,
)
from pymhm.core.condensation import (
    condense_local as condense_local,
)
from pymhm.core.condensation import (
    local_condensation_matrix as local_condensation_matrix,
)
from pymhm.core.condensation import (
    local_condensation_system as local_condensation_system,
)
from pymhm.core.condensation import (
    local_response_from_solution as local_response_from_solution,
)
from pymhm.core.context import (
    BoundProblem as BoundProblem,
)
from pymhm.core.context import (
    GlobalContext as GlobalContext,
)
from pymhm.core.context import (
    LocalContext as LocalContext,
)
from pymhm.core.context import (
    bind_problem as bind_problem,
)
from pymhm.core.contracts import (
    HybridSolution as HybridSolution,
)
from pymhm.core.contracts import (
    LocalAssembly as LocalAssembly,
)
from pymhm.core.contracts import (
    LocalProblem as LocalProblem,
)
from pymhm.core.contracts import (
    LocalResponse as LocalResponse,
)
from pymhm.core.contributions import (
    assemble_hybrid_contributions as assemble_hybrid_contributions,
)
from pymhm.core.contributions import (
    local_global_contribution as local_global_contribution,
)
from pymhm.core.equations import (
    CompiledLocalEquations as CompiledLocalEquations,
)
from pymhm.core.equations import (
    Equation as Equation,
)
from pymhm.core.equations import (
    FormCompiler as FormCompiler,
)
from pymhm.core.equations import (
    LinearForms as LinearForms,
)
from pymhm.core.equations import (
    LocalEquations as LocalEquations,
)
from pymhm.core.equations import (
    columns as columns,
)
from pymhm.core.equations import (
    compile_form as compile_form,
)
from pymhm.core.equations import (
    compile_local_equations as compile_local_equations,
)
from pymhm.core.equations import (
    rows as rows,
)
from pymhm.core.moments import (
    energy_reconstruction as energy_reconstruction,
)
from pymhm.core.multiscale import (
    MultiscaleProblem as MultiscaleProblem,
)
from pymhm.core.multiscale import (
    MultiscaleSolution as MultiscaleSolution,
)
from pymhm.core.multiscale import (
    MultiscaleSystem as MultiscaleSystem,
)
from pymhm.core.multiscale import (
    NestedEquations as NestedEquations,
)
from pymhm.core.multiscale import (
    assemble as assemble,
)
from pymhm.core.multiscale import (
    leaf_moment as leaf_moment,
)
from pymhm.core.multiscale import (
    reconstruct_multiscale as reconstruct_multiscale,
)
from pymhm.core.multiscale import (
    solve as solve,
)
from pymhm.core.multiscale import (
    solve_multiscale_system as solve_multiscale_system,
)
from pymhm.core.multiscale import (
    with_global_equation as with_global_equation,
)
from pymhm.core.multiscale import (
    with_global_load as with_global_load,
)
from pymhm.core.nested import (
    NestedLocalProblem as NestedLocalProblem,
)
from pymhm.core.nested import (
    NestedSolution as NestedSolution,
)
from pymhm.core.nested import (
    nest_hybrid_system as nest_hybrid_system,
)
from pymhm.core.nested import (
    nested_trace_map as nested_trace_map,
)
from pymhm.core.offline import (
    LocalFactorCache as LocalFactorCache,
)
from pymhm.core.offline import (
    OfflineHybridSystem as OfflineHybridSystem,
)
from pymhm.core.offline import (
    OfflineLocalProblem as OfflineLocalProblem,
)
from pymhm.core.offline import (
    condense_cached as condense_cached,
)
from pymhm.core.reconstruction import (
    local_condensed_load as local_condensed_load,
)
from pymhm.core.reconstruction import (
    reconstruct_local as reconstruct_local,
)
from pymhm.core.reconstruction import (
    reconstruct_response as reconstruct_response,
)
from pymhm.core.refinement import (
    HybridRefinement as HybridRefinement,
)
from pymhm.core.refinement import (
    HybridRefinementCase as HybridRefinementCase,
)
from pymhm.core.refinement import (
    HybridRefinementLocal as HybridRefinementLocal,
)
from pymhm.core.refinement import (
    HybridRefinementStore as HybridRefinementStore,
)
from pymhm.core.refinement import (
    HybridStreamRefinement as HybridStreamRefinement,
)
from pymhm.core.refinement import (
    refine_hybrid as refine_hybrid,
)
from pymhm.core.refinement import (
    refine_hybrid_stream as refine_hybrid_stream,
)
from pymhm.core.spaces import (
    BoundInterface as BoundInterface,
)
from pymhm.core.spaces import (
    InterfaceSpace as InterfaceSpace,
)
from pymhm.core.spaces import (
    MeshHierarchy as MeshHierarchy,
)
from pymhm.core.spaces import (
    TraceBinding as TraceBinding,
)
from pymhm.core.spaces import (
    bind_interface as bind_interface,
)
from pymhm.core.spaces import (
    bind_local_equations as bind_local_equations,
)
from pymhm.core.spaces import (
    validate_trace_binding as validate_trace_binding,
)
from pymhm.core.subspaces import (
    restrict_response as restrict_response,
)
from pymhm.core.system import (
    HybridSystem as HybridSystem,
)
from pymhm.core.system import (
    hybrid_mean_constraint as hybrid_mean_constraint,
)
from pymhm.core.system import (
    solve_hybrid_system as solve_hybrid_system,
)
from pymhm.core.variational import (
    GlobalForm as GlobalForm,
)
from pymhm.core.variational import (
    LocalForm as LocalForm,
)
from pymhm.core.variational import (
    LocalProvider as LocalProvider,
)
from pymhm.core.variational import (
    compile_local_forms as compile_local_forms,
)
from pymhm.execution.cpu import (
    ExecutionConfig as ExecutionConfig,
)
from pymhm.execution.cpu import (
    iter_local as iter_local,
)
from pymhm.execution.cpu import (
    map_local as map_local,
)
from pymhm.execution.cuda import (
    BatchedFactorization as BatchedFactorization,
)
from pymhm.execution.cuda import (
    assemble_p1_batch as assemble_p1_batch,
)
from pymhm.execution.cuda import (
    condense_batched as condense_batched,
)
from pymhm.execution.cuda import (
    condense_multi_gpu as condense_multi_gpu,
)
from pymhm.execution.mpi import (
    DistributedHybridSolution as DistributedHybridSolution,
)
from pymhm.execution.mpi import (
    solve_distributed as solve_distributed,
)
from pymhm.fem.hdiv.bdm_family import (
    BDMFamily as BDMFamily,
)
from pymhm.fem.hdiv.family_3d import (
    HDiv3DFamily as HDiv3DFamily,
)
from pymhm.fem.hdiv.rt import (
    RTField as RTField,
)
from pymhm.fem.hdiv.rt_3d import (
    RTTetraFamily as RTTetraFamily,
)
from pymhm.fem.quadrature.material import (
    cartesian_edge_quadrature as cartesian_edge_quadrature,
)
from pymhm.fem.quadrature.material import (
    cartesian_trace_values as cartesian_trace_values,
)
from pymhm.fem.quadrature.material import (
    fit_material_faces as fit_material_faces,
)
from pymhm.fem.quadrature.material import (
    fit_material_mesh as fit_material_mesh,
)
from pymhm.fem.quadrature.material import (
    material_triangle_quadrature as material_triangle_quadrature,
)
from pymhm.fem.quadrature.planar import (
    planar_edge_quadrature as planar_edge_quadrature,
)
from pymhm.fem.quadrature.planar import (
    planar_simplex_quadrature as planar_simplex_quadrature,
)
from pymhm.fem.reference import (
    NodalReferenceBasis as NodalReferenceBasis,
)
from pymhm.fem.reference import (
    ReferenceElement as ReferenceElement,
)
from pymhm.fem.reference import (
    ReferenceElementSpec as ReferenceElementSpec,
)
from pymhm.fem.reference import (
    barycentric_simplex_tabulation as barycentric_simplex_tabulation,
)
from pymhm.fem.reference import (
    create_reference_element as create_reference_element,
)
from pymhm.fem.reference import (
    interpolate_reference as interpolate_reference,
)
from pymhm.fem.reference import (
    legendre_tabulation as legendre_tabulation,
)
from pymhm.fem.reference import (
    legendre_values as legendre_values,
)
from pymhm.fem.reference import (
    monomial_tabulation as monomial_tabulation,
)
from pymhm.fem.reference import (
    nodal_base_transformations as nodal_base_transformations,
)
from pymhm.fem.reference import (
    orthogonal_polynomial_tabulation as orthogonal_polynomial_tabulation,
)
from pymhm.fem.reference import (
    reference_base_transformations as reference_base_transformations,
)
from pymhm.fem.reference import (
    reference_entity_dofs as reference_entity_dofs,
)
from pymhm.fem.reference import (
    reference_entity_transformations as reference_entity_transformations,
)
from pymhm.fem.reference import (
    reference_interpolation_points as reference_interpolation_points,
)
from pymhm.fem.reference import (
    simplex_lagrange_basis as simplex_lagrange_basis,
)
from pymhm.fem.reference import (
    simplex_lagrange_tabulation as simplex_lagrange_tabulation,
)
from pymhm.fem.reference import (
    tabulate_reference as tabulate_reference,
)
from pymhm.fem.reference import (
    tensor_lagrange_basis as tensor_lagrange_basis,
)
from pymhm.fem.reference import (
    tensor_lagrange_tabulation as tensor_lagrange_tabulation,
)
from pymhm.fem.traces.helmholtz import (
    OscillatoryFaceSpace as OscillatoryFaceSpace,
)
from pymhm.fem.traces.helmholtz import (
    PolynomialNeumannTrace as PolynomialNeumannTrace,
)
from pymhm.fem.traces.helmholtz import (
    helmholtz_skeleton as helmholtz_skeleton,
)
from pymhm.fem.traces.interval import (
    FaceSpace as FaceSpace,
)
from pymhm.fem.traces.interval import (
    SkeletonSpace as SkeletonSpace,
)
from pymhm.fem.traces.interval import (
    interface_pairing as interface_pairing,
)
from pymhm.fem.traces.pressure_3d import (
    PressureTraceSpace3D as PressureTraceSpace3D,
)
from pymhm.fem.traces.triangle_3d import (
    TriangularSkeleton as TriangularSkeleton,
)
from pymhm.fem.vector.curl import (
    CurlOperators as CurlOperators,
)
from pymhm.fem.vector.curl import (
    TangentialTraceSpace as TangentialTraceSpace,
)
from pymhm.io.native import (
    from_gmsh_3d as from_gmsh_3d,
)
from pymhm.io.native import (
    from_netgen_3d as from_netgen_3d,
)
from pymhm.io.tetrahedral import (
    TetraMeshData as TetraMeshData,
)
from pymhm.io.tetrahedral import (
    read_tetra_mesh as read_tetra_mesh,
)
from pymhm.io.tetrahedral import (
    write_tetra_mesh as write_tetra_mesh,
)
from pymhm.io.volume import (
    VolumeMeshData as VolumeMeshData,
)
from pymhm.io.volume import (
    from_meshio as from_meshio,
)
from pymhm.io.volume import (
    read_volume_mesh as read_volume_mesh,
)
from pymhm.io.volume import (
    to_meshio as to_meshio,
)
from pymhm.io.volume import (
    write_volume_mesh as write_volume_mesh,
)
from pymhm.linalg.block import (
    SaddleBlockSolver as SaddleBlockSolver,
)
from pymhm.linalg.dynamics import (
    newmark_step as newmark_step,
)
from pymhm.linalg.moments import (
    solve_moment_system as solve_moment_system,
)
from pymhm.linalg.separable import (
    SeparableKrylovSolution as SeparableKrylovSolution,
)
from pymhm.linalg.separable import (
    TensorDiffusionOperator as TensorDiffusionOperator,
)
from pymhm.linalg.separable import (
    solve_separable_krylov as solve_separable_krylov,
)
from pymhm.linalg.separable import (
    tensor_diffusion_operator as tensor_diffusion_operator,
)
from pymhm.materials.planar import (
    PlanarMaterial as PlanarMaterial,
)
from pymhm.materials.planar import (
    PlanarRegion as PlanarRegion,
)
from pymhm.materials.sources import (
    PolylineLayerField as PolylineLayerField,
)
from pymhm.materials.sources import (
    RadialDiskLoad as RadialDiskLoad,
)
from pymhm.meshes.cartesian import (
    CartesianMacroMesh as CartesianMacroMesh,
)
from pymhm.meshes.fitting import (
    PlanarFittedMesh as PlanarFittedMesh,
)
from pymhm.meshes.fitting import (
    fit_planar_material as fit_planar_material,
)
from pymhm.meshes.fitting import (
    fit_planar_skeleton as fit_planar_skeleton,
)
from pymhm.meshes.fitting import (
    planar_face_partitions as planar_face_partitions,
)
from pymhm.meshes.hexahedron import (
    HexMesh as HexMesh,
)
from pymhm.meshes.longest_edge import (
    refine_longest_edge as refine_longest_edge,
)
from pymhm.meshes.mixed import (
    AffineMixedMesh as AffineMixedMesh,
)
from pymhm.meshes.polygonal import (
    PolygonMesh as PolygonMesh,
)
from pymhm.meshes.polyhedral import (
    PolyhedralMesh as PolyhedralMesh,
)
from pymhm.meshes.refinement import (
    TriangleRefinement as TriangleRefinement,
)
from pymhm.meshes.refinement import (
    refine_triangles as refine_triangles,
)
from pymhm.meshes.refinement import (
    transfer_skeleton as transfer_skeleton,
)
from pymhm.meshes.refinement import (
    validate_submesh as validate_submesh,
)
from pymhm.meshes.refinement_3d import (
    TetraRefinement as TetraRefinement,
)
from pymhm.meshes.refinement_3d import (
    refine_tetrahedra as refine_tetrahedra,
)
from pymhm.meshes.tetrahedron import (
    TetraMesh as TetraMesh,
)
from pymhm.meshes.triangle import (
    TriangleMesh as TriangleMesh,
)
from pymhm.methods.hho import (
    MsHHOLocal as MsHHOLocal,
)
from pymhm.methods.hho import (
    MsHHOSolution as MsHHOSolution,
)
from pymhm.methods.hho_3d import (
    MsHHO3DSolution as MsHHO3DSolution,
)
from pymhm.methods.petrov_galerkin import (
    PGMHMSolution as PGMHMSolution,
)
from pymhm.methods.robin import (
    MHSolution as MHSolution,
)
from pymhm.methods.robin_3d import (
    MH3DSolution as MH3DSolution,
)
from pymhm.methods.three_field import (
    MH2MLocal as MH2MLocal,
)
from pymhm.methods.three_field import (
    MH2MSolution as MH2MSolution,
)
from pymhm.methods.three_field import (
    PressureTraceSpace as PressureTraceSpace,
)
from pymhm.methods.three_field_3d import (
    MH2M3DSolution as MH2M3DSolution,
)
from pymhm.postprocessing.fields import (
    DiscreteField as DiscreteField,
)
from pymhm.postprocessing.fields import (
    FieldDefinition as FieldDefinition,
)
from pymhm.postprocessing.fields import (
    evaluate_field as evaluate_field,
)
from pymhm.postprocessing.fields import (
    evaluate_field_and_gradient as evaluate_field_and_gradient,
)
from pymhm.postprocessing.fields import (
    evaluate_field_gradient as evaluate_field_gradient,
)
from pymhm.postprocessing.fields import (
    local_trace as local_trace,
)
from pymhm.postprocessing.fields import (
    portable_field_coefficients as portable_field_coefficients,
)
from pymhm.postprocessing.fields import (
    solution_field as solution_field,
)
from pymhm.recovery.moments import (
    MomentFluxSolution as MomentFluxSolution,
)
from pymhm.recovery.moments import (
    reconstruct_darcy_moments as reconstruct_darcy_moments,
)
from pymhm.recovery.moments import (
    reconstruct_flux_moments as reconstruct_flux_moments,
)
from pymhm.recovery.moments_3d import (
    MomentFlux3DSolution as MomentFlux3DSolution,
)
from pymhm.recovery.moments_3d import (
    reconstruct_darcy_moments_3d as reconstruct_darcy_moments_3d,
)

__version__: str
__all__: list[str]
