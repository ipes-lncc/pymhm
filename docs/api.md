# API reference

The reference is organized by mathematical formulation and numerical infrastructure.
Each family page contains the complete signatures and docstrings of its documented
objects. Optional dependencies are identified by the relevant adapters; the native
multiscale formulations remain part of the portable package.

| Family | Scope |
| --- | --- |
| [Meshes and geometric refinement](api/geometry.md) | Geometric partitions, skeletal topology and conforming refinement. |
| [Materials and reservoir data](api/materials.md) | Material fields, physical loads and integration across interfaces. |
| [Mesh exchange and visualization](api/meshing.md) | Optional mesh generators, mesh-file exchange and visualization adapters. |
| [Hybrid operators and multiscale constructions](api/hybrid.md) | Local condensation, operator reuse, recursive problems and distinct multiscale formulations. |
| [Robin MH and three-field MH²M](api/mh.md) | Local Robin and Neumann maps, independent pressure/conormal traces, and tetrahedral assembly. |
| [Darcy in two dimensions](api/darcy.md) | Primal, mixed, analytical and residual-enriched Darcy discretizations. |
| [Darcy in three dimensions](api/darcy3d.md) | Tetrahedral, prismatic and mapped mixed formulations with their explicit geometry and degree contracts. |
| [Flux reconstruction and error estimation](api/reconstruction.md) | Moment reconstruction, conforming potentials and dimension-dependent estimator hypotheses. |
| [Adaptive scalar and elasticity indicators](api/adaptivity.md) | Darcy refinement policies, local error controls, metric remeshing and the primal elasticity indicator. |
| [Stokes, Brinkman and Oseen](api/flow.md) | Incompressible-flow formulations, residual estimators and adaptive policies. |
| [Elasticity](api/elasticity.md) | Displacement, displacement–pressure and weakly symmetric stress formulations. |
| [Transport, reaction and diffusion](api/transport.md) | Stationary and transient scalar problems, conservative transport and stabilization. |
| [Finite element bases](api/elements.md) | Lagrange, BDM and RT bases and their declared local degrees of freedom. |
| [Helmholtz and Maxwell](api/waves.md) | Time-harmonic acoustic and transient electromagnetic formulations. |
| [Solvers and execution backends](api/backends.md) | Finite-element adapters, sparse solvers, parallel execution and separable operators. |

## Section links

The following links retain the earlier section locations and lead to the relevant family.

- <span id="reservoir-data-and-visualization"></span>[Reservoir data and visualization](api/materials.md)
- <span id="mesh-and-skeleton"></span>[Mesh and skeleton](api/geometry.md)
- <span id="local-and-global-hybrid-operators"></span>[Local and global hybrid operators](api/hybrid.md)
- <span id="darcy-and-reconstruction"></span>[Darcy and reconstruction](api/darcy.md)
- <span id="vector-and-scalar-problems"></span>[Vector and scalar problems](api/flow.md)
- <span id="native-finite-element-bases"></span>[Native finite element bases](api/elements.md)
- <span id="optional-backends"></span>[Optional backends](api/backends.md)
- <span id="additional-three-dimensional-and-material-interfaces"></span>[Additional three-dimensional and material interfaces](api/darcy3d.md)
- <span id="three-dimensional-displacementpressure-elasticity"></span>[Three-dimensional displacement–pressure elasticity](api/elasticity.md)
- <span id="residual-petrovgalerkin-enrichment"></span>[Residual Petrov–Galerkin enrichment](api/darcy.md)
- <span id="time-harmonic-acoustics"></span>[Time-harmonic acoustics](api/waves.md)
- <span id="planar-material-interfaces"></span>[Planar material interfaces](api/materials.md)

## Object index

Module, class, function and member links below retain their identifiers across
the family pages. Numerical conventions and supported inputs are documented
with each object.

### Meshes and geometric refinement

<a id="pymhm.mesh"></a>**[pymhm.mesh](api/geometry.md#pymhm.mesh)**

- <a id="pymhm.mesh.TriangleMesh"></a>[`TriangleMesh`](api/geometry.md#pymhm.mesh.TriangleMesh)
- <a id="pymhm.mesh.TriangleMesh.boundary_faces"></a>[`TriangleMesh.boundary_faces`](api/geometry.md#pymhm.mesh.TriangleMesh.boundary_faces)
- <a id="pymhm.mesh.TriangleMesh.normals"></a>[`TriangleMesh.normals`](api/geometry.md#pymhm.mesh.TriangleMesh.normals)
- <a id="pymhm.mesh.TriangleMesh.lengths"></a>[`TriangleMesh.lengths`](api/geometry.md#pymhm.mesh.TriangleMesh.lengths)
- <a id="pymhm.mesh.TriangleMesh.areas"></a>[`TriangleMesh.areas`](api/geometry.md#pymhm.mesh.TriangleMesh.areas)
- <a id="pymhm.mesh.TriangleMesh.__post_init__"></a>[`TriangleMesh.__post_init__`](api/geometry.md#pymhm.mesh.TriangleMesh.__post_init__)
- <a id="pymhm.mesh.TriangleMesh.unit_square"></a>[`TriangleMesh.unit_square`](api/geometry.md#pymhm.mesh.TriangleMesh.unit_square)
- <a id="pymhm.mesh.TriangleMesh.submesh"></a>[`TriangleMesh.submesh`](api/geometry.md#pymhm.mesh.TriangleMesh.submesh)
- <a id="pymhm.mesh.FaceSpace"></a>[`FaceSpace`](api/geometry.md#pymhm.mesh.FaceSpace)
- <a id="pymhm.mesh.FaceSpace.size"></a>[`FaceSpace.size`](api/geometry.md#pymhm.mesh.FaceSpace.size)
- <a id="pymhm.mesh.FaceSpace.__post_init__"></a>[`FaceSpace.__post_init__`](api/geometry.md#pymhm.mesh.FaceSpace.__post_init__)
- <a id="pymhm.mesh.FaceSpace.uniform"></a>[`FaceSpace.uniform`](api/geometry.md#pymhm.mesh.FaceSpace.uniform)
- <a id="pymhm.mesh.FaceSpace.evaluate"></a>[`FaceSpace.evaluate`](api/geometry.md#pymhm.mesh.FaceSpace.evaluate)
- <a id="pymhm.mesh.FaceSpace.constant_coefficients"></a>[`FaceSpace.constant_coefficients`](api/geometry.md#pymhm.mesh.FaceSpace.constant_coefficients)
- <a id="pymhm.mesh.FaceSpace.quadrature"></a>[`FaceSpace.quadrature`](api/geometry.md#pymhm.mesh.FaceSpace.quadrature)
- <a id="pymhm.mesh.SkeletonSpace"></a>[`SkeletonSpace`](api/geometry.md#pymhm.mesh.SkeletonSpace)
- <a id="pymhm.mesh.SkeletonSpace.size"></a>[`SkeletonSpace.size`](api/geometry.md#pymhm.mesh.SkeletonSpace.size)
- <a id="pymhm.mesh.SkeletonSpace.__init__"></a>[`SkeletonSpace.__init__`](api/geometry.md#pymhm.mesh.SkeletonSpace.__init__)
- <a id="pymhm.mesh.SkeletonSpace.dofs"></a>[`SkeletonSpace.dofs`](api/geometry.md#pymhm.mesh.SkeletonSpace.dofs)
- <a id="pymhm.mesh.SkeletonSpace.cell_dofs"></a>[`SkeletonSpace.cell_dofs`](api/geometry.md#pymhm.mesh.SkeletonSpace.cell_dofs)
- <a id="pymhm.mesh.positive_int"></a>[`positive_int`](api/geometry.md#pymhm.mesh.positive_int)

<a id="pymhm.polygon"></a>**[pymhm.polygon](api/geometry.md#pymhm.polygon)**

- <a id="pymhm.polygon.PolygonMesh"></a>[`PolygonMesh`](api/geometry.md#pymhm.polygon.PolygonMesh)
- <a id="pymhm.polygon.PolygonMesh.areas"></a>[`PolygonMesh.areas`](api/geometry.md#pymhm.polygon.PolygonMesh.areas)
- <a id="pymhm.polygon.PolygonMesh.lengths"></a>[`PolygonMesh.lengths`](api/geometry.md#pymhm.polygon.PolygonMesh.lengths)
- <a id="pymhm.polygon.PolygonMesh.normals"></a>[`PolygonMesh.normals`](api/geometry.md#pymhm.polygon.PolygonMesh.normals)
- <a id="pymhm.polygon.PolygonMesh.boundary_faces"></a>[`PolygonMesh.boundary_faces`](api/geometry.md#pymhm.polygon.PolygonMesh.boundary_faces)
- <a id="pymhm.polygon.PolygonMesh.__post_init__"></a>[`PolygonMesh.__post_init__`](api/geometry.md#pymhm.polygon.PolygonMesh.__post_init__)
- <a id="pymhm.polygon.PolygonMesh.submesh"></a>[`PolygonMesh.submesh`](api/geometry.md#pymhm.polygon.PolygonMesh.submesh)
- <a id="pymhm.polygon.solve_darcy_polygons"></a>[`solve_darcy_polygons`](api/geometry.md#pymhm.polygon.solve_darcy_polygons)
- <a id="pymhm.polygon.solve_transport_polygons"></a>[`solve_transport_polygons`](api/geometry.md#pymhm.polygon.solve_transport_polygons)
- <a id="pymhm.polygon.solve_brinkman_polygons"></a>[`solve_brinkman_polygons`](api/geometry.md#pymhm.polygon.solve_brinkman_polygons)
- <a id="pymhm.polygon.solve_mshho_polygons"></a>[`solve_mshho_polygons`](api/geometry.md#pymhm.polygon.solve_mshho_polygons)
- <a id="pymhm.polygon.solve_elasticity_mixed_polygons"></a>[`solve_elasticity_mixed_polygons`](api/geometry.md#pymhm.polygon.solve_elasticity_mixed_polygons)

<a id="pymhm.polyhedral"></a>**[pymhm.polyhedral](api/geometry.md#pymhm.polyhedral)**

- <a id="pymhm.polyhedral.PolyhedralMesh"></a>[`PolyhedralMesh`](api/geometry.md#pymhm.polyhedral.PolyhedralMesh)
- <a id="pymhm.polyhedral.PolyhedralMesh.__init__"></a>[`PolyhedralMesh.__init__`](api/geometry.md#pymhm.polyhedral.PolyhedralMesh.__init__)
- <a id="pymhm.polyhedral.PolyhedralMesh.cubes"></a>[`PolyhedralMesh.cubes`](api/geometry.md#pymhm.polyhedral.PolyhedralMesh.cubes)
- <a id="pymhm.polyhedral.PolyhedralMesh.extrude"></a>[`PolyhedralMesh.extrude`](api/geometry.md#pymhm.polyhedral.PolyhedralMesh.extrude)
- <a id="pymhm.polyhedral.PolyhedralMesh.submesh"></a>[`PolyhedralMesh.submesh`](api/geometry.md#pymhm.polyhedral.PolyhedralMesh.submesh)

<a id="pymhm.tetrahedral"></a>**[pymhm.tetrahedral](api/geometry.md#pymhm.tetrahedral)**

- <a id="pymhm.tetrahedral.TetraMesh"></a>[`TetraMesh`](api/geometry.md#pymhm.tetrahedral.TetraMesh)
- <a id="pymhm.tetrahedral.TetraMesh.__init__"></a>[`TetraMesh.__init__`](api/geometry.md#pymhm.tetrahedral.TetraMesh.__init__)
- <a id="pymhm.tetrahedral.TetraMesh.unit_cube"></a>[`TetraMesh.unit_cube`](api/geometry.md#pymhm.tetrahedral.TetraMesh.unit_cube)
- <a id="pymhm.tetrahedral.TetraMesh.submesh"></a>[`TetraMesh.submesh`](api/geometry.md#pymhm.tetrahedral.TetraMesh.submesh)
- <a id="pymhm.tetrahedral.tetrahedron_quadrature"></a>[`tetrahedron_quadrature`](api/geometry.md#pymhm.tetrahedral.tetrahedron_quadrature)
- <a id="pymhm.tetrahedral.tetra_nodal_space"></a>[`tetra_nodal_space`](api/geometry.md#pymhm.tetrahedral.tetra_nodal_space)
- <a id="pymhm.tetrahedral.tetra_basis"></a>[`tetra_basis`](api/geometry.md#pymhm.tetrahedral.tetra_basis)
- <a id="pymhm.tetrahedral.tetra_tabulate"></a>[`tetra_tabulate`](api/geometry.md#pymhm.tetrahedral.tetra_tabulate)
- <a id="pymhm.tetrahedral.tetra_element_tabulate"></a>[`tetra_element_tabulate`](api/geometry.md#pymhm.tetrahedral.tetra_element_tabulate)
- <a id="pymhm.tetrahedral.scalar_values_3d"></a>[`scalar_values_3d`](api/geometry.md#pymhm.tetrahedral.scalar_values_3d)
- <a id="pymhm.tetrahedral.tensor_values_3d"></a>[`tensor_values_3d`](api/geometry.md#pymhm.tetrahedral.tensor_values_3d)
- <a id="pymhm.tetrahedral.tetra_operators"></a>[`tetra_operators`](api/geometry.md#pymhm.tetrahedral.tetra_operators)

<a id="pymhm.refinement"></a>**[pymhm.refinement](api/geometry.md#pymhm.refinement)**

- <a id="pymhm.refinement.TriangleRefinement"></a>[`TriangleRefinement`](api/geometry.md#pymhm.refinement.TriangleRefinement)
- <a id="pymhm.refinement.validate_submesh"></a>[`validate_submesh`](api/geometry.md#pymhm.refinement.validate_submesh)
- <a id="pymhm.refinement.refine_triangles"></a>[`refine_triangles`](api/geometry.md#pymhm.refinement.refine_triangles)
- <a id="pymhm.refinement.transfer_skeleton"></a>[`transfer_skeleton`](api/geometry.md#pymhm.refinement.transfer_skeleton)

<a id="pymhm.longest_edge"></a>**[pymhm.longest_edge](api/geometry.md#pymhm.longest_edge)**

- <a id="pymhm.longest_edge.refine_longest_edge"></a>[`refine_longest_edge`](api/geometry.md#pymhm.longest_edge.refine_longest_edge)

<a id="pymhm.refinement3d"></a>**[pymhm.refinement3d](api/geometry.md#pymhm.refinement3d)**

- <a id="pymhm.refinement3d.TetraRefinement"></a>[`TetraRefinement`](api/geometry.md#pymhm.refinement3d.TetraRefinement)
- <a id="pymhm.refinement3d.refine_tetrahedra"></a>[`refine_tetrahedra`](api/geometry.md#pymhm.refinement3d.refine_tetrahedra)

### Materials and reservoir data

<a id="pymhm.reservoir"></a>**[pymhm.reservoir](api/materials.md#pymhm.reservoir)**

- <a id="pymhm.reservoir.CartesianCellField"></a>[`CartesianCellField`](api/materials.md#pymhm.reservoir.CartesianCellField)
- <a id="pymhm.reservoir.CartesianCellField.__post_init__"></a>[`CartesianCellField.__post_init__`](api/materials.md#pymhm.reservoir.CartesianCellField.__post_init__)
- <a id="pymhm.reservoir.CartesianCellField.__call__"></a>[`CartesianCellField.__call__`](api/materials.md#pymhm.reservoir.CartesianCellField.__call__)
- <a id="pymhm.reservoir.ReservoirData"></a>[`ReservoirData`](api/materials.md#pymhm.reservoir.ReservoirData)
- <a id="pymhm.reservoir.ReservoirData.__post_init__"></a>[`ReservoirData.__post_init__`](api/materials.md#pymhm.reservoir.ReservoirData.__post_init__)
- <a id="pymhm.reservoir.ReservoirData.layer"></a>[`ReservoirData.layer`](api/materials.md#pymhm.reservoir.ReservoirData.layer)
- <a id="pymhm.reservoir.ReservoirData.to_si"></a>[`ReservoirData.to_si`](api/materials.md#pymhm.reservoir.ReservoirData.to_si)
- <a id="pymhm.reservoir.read_eclipse_properties"></a>[`read_eclipse_properties`](api/materials.md#pymhm.reservoir.read_eclipse_properties)
- <a id="pymhm.reservoir.load_spe10_model2"></a>[`load_spe10_model2`](api/materials.md#pymhm.reservoir.load_spe10_model2)
- <a id="pymhm.reservoir.download_spe10_model2"></a>[`download_spe10_model2`](api/materials.md#pymhm.reservoir.download_spe10_model2)

<a id="pymhm.loads"></a>**[pymhm.loads](api/materials.md#pymhm.loads)**

- <a id="pymhm.loads.split_point_sources"></a>[`split_point_sources`](api/materials.md#pymhm.loads.split_point_sources)
- <a id="pymhm.loads.point_load_vector"></a>[`point_load_vector`](api/materials.md#pymhm.loads.point_load_vector)

<a id="pymhm.cut_cells"></a>**[pymhm.cut_cells](api/materials.md#pymhm.cut_cells)**

- <a id="pymhm.cut_cells.cartesian_trace_values"></a>[`cartesian_trace_values`](api/materials.md#pymhm.cut_cells.cartesian_trace_values)
- <a id="pymhm.cut_cells.cartesian_edge_quadrature"></a>[`cartesian_edge_quadrature`](api/materials.md#pymhm.cut_cells.cartesian_edge_quadrature)
- <a id="pymhm.cut_cells.material_triangle_quadrature"></a>[`material_triangle_quadrature`](api/materials.md#pymhm.cut_cells.material_triangle_quadrature)
- <a id="pymhm.cut_cells.fit_material_faces"></a>[`fit_material_faces`](api/materials.md#pymhm.cut_cells.fit_material_faces)
- <a id="pymhm.cut_cells.fit_material_mesh"></a>[`fit_material_mesh`](api/materials.md#pymhm.cut_cells.fit_material_mesh)
- <a id="pymhm.cut_cells.cartesian_triangle_quadrature"></a>[`cartesian_triangle_quadrature`](api/materials.md#pymhm.cut_cells.cartesian_triangle_quadrature)

<a id="pymhm.planar_material"></a>**[pymhm.planar_material](api/materials.md#pymhm.planar_material)**

- <a id="pymhm.planar_material.PlanarRegion"></a>[`PlanarRegion`](api/materials.md#pymhm.planar_material.PlanarRegion)
- <a id="pymhm.planar_material.PlanarRegion.__post_init__"></a>[`PlanarRegion.__post_init__`](api/materials.md#pymhm.planar_material.PlanarRegion.__post_init__)
- <a id="pymhm.planar_material.PlanarRegion.box"></a>[`PlanarRegion.box`](api/materials.md#pymhm.planar_material.PlanarRegion.box)
- <a id="pymhm.planar_material.PlanarMaterial"></a>[`PlanarMaterial`](api/materials.md#pymhm.planar_material.PlanarMaterial)
- <a id="pymhm.planar_material.PlanarMaterial.dimension"></a>[`PlanarMaterial.dimension`](api/materials.md#pymhm.planar_material.PlanarMaterial.dimension)
- <a id="pymhm.planar_material.PlanarMaterial.planes"></a>[`PlanarMaterial.planes`](api/materials.md#pymhm.planar_material.PlanarMaterial.planes)
- <a id="pymhm.planar_material.PlanarMaterial.tensors"></a>[`PlanarMaterial.tensors`](api/materials.md#pymhm.planar_material.PlanarMaterial.tensors)
- <a id="pymhm.planar_material.PlanarMaterial.__post_init__"></a>[`PlanarMaterial.__post_init__`](api/materials.md#pymhm.planar_material.PlanarMaterial.__post_init__)
- <a id="pymhm.planar_material.PlanarMaterial.region_ids"></a>[`PlanarMaterial.region_ids`](api/materials.md#pymhm.planar_material.PlanarMaterial.region_ids)
- <a id="pymhm.planar_material.PlanarMaterial.__call__"></a>[`PlanarMaterial.__call__`](api/materials.md#pymhm.planar_material.PlanarMaterial.__call__)
- <a id="pymhm.planar_material.PlanarMaterial.trace_values"></a>[`PlanarMaterial.trace_values`](api/materials.md#pymhm.planar_material.PlanarMaterial.trace_values)

<a id="pymhm.planar_fitting"></a>**[pymhm.planar_fitting](api/materials.md#pymhm.planar_fitting)**

- <a id="pymhm.planar_fitting.PlanarFittedMesh"></a>[`PlanarFittedMesh`](api/materials.md#pymhm.planar_fitting.PlanarFittedMesh)
- <a id="pymhm.planar_fitting.PlanarFittedMesh.tensors"></a>[`PlanarFittedMesh.tensors`](api/materials.md#pymhm.planar_fitting.PlanarFittedMesh.tensors)
- <a id="pymhm.planar_fitting.fit_planar_material"></a>[`fit_planar_material`](api/materials.md#pymhm.planar_fitting.fit_planar_material)
- <a id="pymhm.planar_fitting.fit_planar_skeleton"></a>[`fit_planar_skeleton`](api/materials.md#pymhm.planar_fitting.fit_planar_skeleton)
- <a id="pymhm.planar_fitting.planar_face_partitions"></a>[`planar_face_partitions`](api/materials.md#pymhm.planar_fitting.planar_face_partitions)

<a id="pymhm.planar_quadrature"></a>**[pymhm.planar_quadrature](api/materials.md#pymhm.planar_quadrature)**

- <a id="pymhm.planar_quadrature.planar_simplex_quadrature"></a>[`planar_simplex_quadrature`](api/materials.md#pymhm.planar_quadrature.planar_simplex_quadrature)
- <a id="pymhm.planar_quadrature.planar_edge_quadrature"></a>[`planar_edge_quadrature`](api/materials.md#pymhm.planar_quadrature.planar_edge_quadrature)

### Mesh exchange and visualization

<a id="pymhm.meshing"></a>**[pymhm.meshing](api/meshing.md#pymhm.meshing)**

- <a id="pymhm.meshing.MeshData"></a>[`MeshData`](api/meshing.md#pymhm.meshing.MeshData)
- <a id="pymhm.meshing.MeshData.boundary_tags"></a>[`MeshData.boundary_tags`](api/meshing.md#pymhm.meshing.MeshData.boundary_tags)
- <a id="pymhm.meshing.MeshData.__post_init__"></a>[`MeshData.__post_init__`](api/meshing.md#pymhm.meshing.MeshData.__post_init__)
- <a id="pymhm.meshing.read_mesh"></a>[`read_mesh`](api/meshing.md#pymhm.meshing.read_mesh)
- <a id="pymhm.meshing.write_mesh"></a>[`write_mesh`](api/meshing.md#pymhm.meshing.write_mesh)
- <a id="pymhm.meshing.from_gmsh"></a>[`from_gmsh`](api/meshing.md#pymhm.meshing.from_gmsh)
- <a id="pymhm.meshing.unit_square_gmsh"></a>[`unit_square_gmsh`](api/meshing.md#pymhm.meshing.unit_square_gmsh)
- <a id="pymhm.meshing.from_netgen"></a>[`from_netgen`](api/meshing.md#pymhm.meshing.from_netgen)
- <a id="pymhm.meshing.unit_square_netgen"></a>[`unit_square_netgen`](api/meshing.md#pymhm.meshing.unit_square_netgen)

<a id="pymhm.meshing3d"></a>**[pymhm.meshing3d](api/meshing.md#pymhm.meshing3d)**

- <a id="pymhm.meshing3d.TetraMeshData"></a>[`TetraMeshData`](api/meshing.md#pymhm.meshing3d.TetraMeshData)
- <a id="pymhm.meshing3d.TetraMeshData.__post_init__"></a>[`TetraMeshData.__post_init__`](api/meshing.md#pymhm.meshing3d.TetraMeshData.__post_init__)
- <a id="pymhm.meshing3d.read_tetra_mesh"></a>[`read_tetra_mesh`](api/meshing.md#pymhm.meshing3d.read_tetra_mesh)
- <a id="pymhm.meshing3d.write_tetra_mesh"></a>[`write_tetra_mesh`](api/meshing.md#pymhm.meshing3d.write_tetra_mesh)

<a id="pymhm.mesh_exchange"></a>**[pymhm.mesh_exchange](api/meshing.md#pymhm.mesh_exchange)**

- <a id="pymhm.mesh_exchange.VolumeMeshData"></a>[`VolumeMeshData`](api/meshing.md#pymhm.mesh_exchange.VolumeMeshData)
- <a id="pymhm.mesh_exchange.VolumeMeshData.__post_init__"></a>[`VolumeMeshData.__post_init__`](api/meshing.md#pymhm.mesh_exchange.VolumeMeshData.__post_init__)
- <a id="pymhm.mesh_exchange.from_meshio"></a>[`from_meshio`](api/meshing.md#pymhm.mesh_exchange.from_meshio)
- <a id="pymhm.mesh_exchange.read_volume_mesh"></a>[`read_volume_mesh`](api/meshing.md#pymhm.mesh_exchange.read_volume_mesh)
- <a id="pymhm.mesh_exchange.to_meshio"></a>[`to_meshio`](api/meshing.md#pymhm.mesh_exchange.to_meshio)
- <a id="pymhm.mesh_exchange.write_volume_mesh"></a>[`write_volume_mesh`](api/meshing.md#pymhm.mesh_exchange.write_volume_mesh)

<a id="pymhm.meshing_native3d"></a>**[pymhm.meshing_native3d](api/meshing.md#pymhm.meshing_native3d)**

- <a id="pymhm.meshing_native3d.from_gmsh_3d"></a>[`from_gmsh_3d`](api/meshing.md#pymhm.meshing_native3d.from_gmsh_3d)
- <a id="pymhm.meshing_native3d.unit_cube_gmsh"></a>[`unit_cube_gmsh`](api/meshing.md#pymhm.meshing_native3d.unit_cube_gmsh)
- <a id="pymhm.meshing_native3d.from_netgen_3d"></a>[`from_netgen_3d`](api/meshing.md#pymhm.meshing_native3d.from_netgen_3d)
- <a id="pymhm.meshing_native3d.unit_cube_netgen"></a>[`unit_cube_netgen`](api/meshing.md#pymhm.meshing_native3d.unit_cube_netgen)

<a id="pymhm.visualization"></a>**[pymhm.visualization](api/meshing.md#pymhm.visualization)**

- <a id="pymhm.visualization.triangle_grid"></a>[`triangle_grid`](api/meshing.md#pymhm.visualization.triangle_grid)
- <a id="pymhm.visualization.broken_triangle_grid"></a>[`broken_triangle_grid`](api/meshing.md#pymhm.visualization.broken_triangle_grid)
- <a id="pymhm.visualization.macro_edges"></a>[`macro_edges`](api/meshing.md#pymhm.visualization.macro_edges)
- <a id="pymhm.visualization.structured_cell_grid"></a>[`structured_cell_grid`](api/meshing.md#pymhm.visualization.structured_cell_grid)
- <a id="pymhm.visualization.plot_field"></a>[`plot_field`](api/meshing.md#pymhm.visualization.plot_field)

### Hybrid operators and multiscale constructions

<a id="pymhm.hybrid"></a>**[pymhm.hybrid](api/hybrid.md#pymhm.hybrid)**

- <a id="pymhm.hybrid.LocalProblem"></a>[`LocalProblem`](api/hybrid.md#pymhm.hybrid.LocalProblem)
- <a id="pymhm.hybrid.LocalProblem.__init__"></a>[`LocalProblem.__init__`](api/hybrid.md#pymhm.hybrid.LocalProblem.__init__)
- <a id="pymhm.hybrid.LocalProblem.condense"></a>[`LocalProblem.condense`](api/hybrid.md#pymhm.hybrid.LocalProblem.condense)
- <a id="pymhm.hybrid.LocalProblem.condensation_system"></a>[`LocalProblem.condensation_system`](api/hybrid.md#pymhm.hybrid.LocalProblem.condensation_system)
- <a id="pymhm.hybrid.LocalProblem.response_from_solution"></a>[`LocalProblem.response_from_solution`](api/hybrid.md#pymhm.hybrid.LocalProblem.response_from_solution)
- <a id="pymhm.hybrid.LocalProblem.with_load"></a>[`LocalProblem.with_load`](api/hybrid.md#pymhm.hybrid.LocalProblem.with_load)
- <a id="pymhm.hybrid.LocalAssembly"></a>[`LocalAssembly`](api/hybrid.md#pymhm.hybrid.LocalAssembly)
- <a id="pymhm.hybrid.LocalAssembly.__post_init__"></a>[`LocalAssembly.__post_init__`](api/hybrid.md#pymhm.hybrid.LocalAssembly.__post_init__)
- <a id="pymhm.hybrid.LocalResponse"></a>[`LocalResponse`](api/hybrid.md#pymhm.hybrid.LocalResponse)
- <a id="pymhm.hybrid.LocalResponse.retained_basis"></a>[`LocalResponse.retained_basis`](api/hybrid.md#pymhm.hybrid.LocalResponse.retained_basis)
- <a id="pymhm.hybrid.LocalResponse.reconstruct"></a>[`LocalResponse.reconstruct`](api/hybrid.md#pymhm.hybrid.LocalResponse.reconstruct)
- <a id="pymhm.hybrid.LocalResponse.global_load"></a>[`LocalResponse.global_load`](api/hybrid.md#pymhm.hybrid.LocalResponse.global_load)
- <a id="pymhm.hybrid.LocalResponse.global_contribution"></a>[`LocalResponse.global_contribution`](api/hybrid.md#pymhm.hybrid.LocalResponse.global_contribution)
- <a id="pymhm.hybrid.HybridSolution"></a>[`HybridSolution`](api/hybrid.md#pymhm.hybrid.HybridSolution)
- <a id="pymhm.hybrid.HybridSystem"></a>[`HybridSystem`](api/hybrid.md#pymhm.hybrid.HybridSystem)
- <a id="pymhm.hybrid.HybridSystem.__init__"></a>[`HybridSystem.__init__`](api/hybrid.md#pymhm.hybrid.HybridSystem.__init__)
- <a id="pymhm.hybrid.HybridSystem.from_local_factory"></a>[`HybridSystem.from_local_factory`](api/hybrid.md#pymhm.hybrid.HybridSystem.from_local_factory)
- <a id="pymhm.hybrid.HybridSystem.from_responses"></a>[`HybridSystem.from_responses`](api/hybrid.md#pymhm.hybrid.HybridSystem.from_responses)
- <a id="pymhm.hybrid.HybridSystem.mean_constraint"></a>[`HybridSystem.mean_constraint`](api/hybrid.md#pymhm.hybrid.HybridSystem.mean_constraint)
- <a id="pymhm.hybrid.HybridSystem.solve"></a>[`HybridSystem.solve`](api/hybrid.md#pymhm.hybrid.HybridSystem.solve)

<a id="pymhm.offline"></a>**[pymhm.offline](api/hybrid.md#pymhm.offline)**

- <a id="pymhm.offline.LocalFactorCache"></a>[`LocalFactorCache`](api/hybrid.md#pymhm.offline.LocalFactorCache)
- <a id="pymhm.offline.LocalFactorCache.size"></a>[`LocalFactorCache.size`](api/hybrid.md#pymhm.offline.LocalFactorCache.size)
- <a id="pymhm.offline.LocalFactorCache.__init__"></a>[`LocalFactorCache.__init__`](api/hybrid.md#pymhm.offline.LocalFactorCache.__init__)
- <a id="pymhm.offline.LocalFactorCache.condense"></a>[`LocalFactorCache.condense`](api/hybrid.md#pymhm.offline.LocalFactorCache.condense)
- <a id="pymhm.offline.LocalFactorCache.close"></a>[`LocalFactorCache.close`](api/hybrid.md#pymhm.offline.LocalFactorCache.close)
- <a id="pymhm.offline.LocalFactorCache.__enter__"></a>[`LocalFactorCache.__enter__`](api/hybrid.md#pymhm.offline.LocalFactorCache.__enter__)
- <a id="pymhm.offline.LocalFactorCache.__exit__"></a>[`LocalFactorCache.__exit__`](api/hybrid.md#pymhm.offline.LocalFactorCache.__exit__)
- <a id="pymhm.offline.OfflineLocalProblem"></a>[`OfflineLocalProblem`](api/hybrid.md#pymhm.offline.OfflineLocalProblem)
- <a id="pymhm.offline.OfflineLocalProblem.__init__"></a>[`OfflineLocalProblem.__init__`](api/hybrid.md#pymhm.offline.OfflineLocalProblem.__init__)
- <a id="pymhm.offline.OfflineLocalProblem.response"></a>[`OfflineLocalProblem.response`](api/hybrid.md#pymhm.offline.OfflineLocalProblem.response)
- <a id="pymhm.offline.OfflineLocalProblem.close"></a>[`OfflineLocalProblem.close`](api/hybrid.md#pymhm.offline.OfflineLocalProblem.close)
- <a id="pymhm.offline.OfflineLocalProblem.__enter__"></a>[`OfflineLocalProblem.__enter__`](api/hybrid.md#pymhm.offline.OfflineLocalProblem.__enter__)
- <a id="pymhm.offline.OfflineLocalProblem.__exit__"></a>[`OfflineLocalProblem.__exit__`](api/hybrid.md#pymhm.offline.OfflineLocalProblem.__exit__)
- <a id="pymhm.offline.OfflineHybridSystem"></a>[`OfflineHybridSystem`](api/hybrid.md#pymhm.offline.OfflineHybridSystem)
- <a id="pymhm.offline.OfflineHybridSystem.__init__"></a>[`OfflineHybridSystem.__init__`](api/hybrid.md#pymhm.offline.OfflineHybridSystem.__init__)
- <a id="pymhm.offline.OfflineHybridSystem.solve"></a>[`OfflineHybridSystem.solve`](api/hybrid.md#pymhm.offline.OfflineHybridSystem.solve)
- <a id="pymhm.offline.OfflineHybridSystem.solve_many"></a>[`OfflineHybridSystem.solve_many`](api/hybrid.md#pymhm.offline.OfflineHybridSystem.solve_many)
- <a id="pymhm.offline.OfflineHybridSystem.close"></a>[`OfflineHybridSystem.close`](api/hybrid.md#pymhm.offline.OfflineHybridSystem.close)
- <a id="pymhm.offline.OfflineHybridSystem.__enter__"></a>[`OfflineHybridSystem.__enter__`](api/hybrid.md#pymhm.offline.OfflineHybridSystem.__enter__)
- <a id="pymhm.offline.OfflineHybridSystem.__exit__"></a>[`OfflineHybridSystem.__exit__`](api/hybrid.md#pymhm.offline.OfflineHybridSystem.__exit__)
- <a id="pymhm.offline.condense_cached"></a>[`condense_cached`](api/hybrid.md#pymhm.offline.condense_cached)

<a id="pymhm.nested"></a>**[pymhm.nested](api/hybrid.md#pymhm.nested)**

- <a id="pymhm.nested.NestedSolution"></a>[`NestedSolution`](api/hybrid.md#pymhm.nested.NestedSolution)
- <a id="pymhm.nested.NestedLocalProblem"></a>[`NestedLocalProblem`](api/hybrid.md#pymhm.nested.NestedLocalProblem)
- <a id="pymhm.nested.NestedLocalProblem.reconstruct"></a>[`NestedLocalProblem.reconstruct`](api/hybrid.md#pymhm.nested.NestedLocalProblem.reconstruct)
- <a id="pymhm.nested.NestedLocalProblem.moment"></a>[`NestedLocalProblem.moment`](api/hybrid.md#pymhm.nested.NestedLocalProblem.moment)
- <a id="pymhm.nested.nested_trace_map"></a>[`nested_trace_map`](api/hybrid.md#pymhm.nested.nested_trace_map)
- <a id="pymhm.nested.nest_hybrid_system"></a>[`nest_hybrid_system`](api/hybrid.md#pymhm.nested.nest_hybrid_system)

<a id="pymhm.subspaces"></a>**[pymhm.subspaces](api/hybrid.md#pymhm.subspaces)**

- <a id="pymhm.subspaces.restrict_response"></a>[`restrict_response`](api/hybrid.md#pymhm.subspaces.restrict_response)

<a id="pymhm.mshho"></a>**[pymhm.mshho](api/hybrid.md#pymhm.mshho)**

- <a id="pymhm.mshho.MsHHOLocal"></a>[`MsHHOLocal`](api/hybrid.md#pymhm.mshho.MsHHOLocal)
- <a id="pymhm.mshho.MsHHOSolution"></a>[`MsHHOSolution`](api/hybrid.md#pymhm.mshho.MsHHOSolution)
- <a id="pymhm.mshho.MsHHOSolution.l2_error"></a>[`MsHHOSolution.l2_error`](api/hybrid.md#pymhm.mshho.MsHHOSolution.l2_error)
- <a id="pymhm.mshho.MsHHOSolution.flux_l2_error"></a>[`MsHHOSolution.flux_l2_error`](api/hybrid.md#pymhm.mshho.MsHHOSolution.flux_l2_error)
- <a id="pymhm.mshho.solve_mshho"></a>[`solve_mshho`](api/hybrid.md#pymhm.mshho.solve_mshho)

<a id="pymhm.mshho3d"></a>**[pymhm.mshho3d](api/hybrid.md#pymhm.mshho3d)**

- <a id="pymhm.mshho3d.MsHHO3DSolution"></a>[`MsHHO3DSolution`](api/hybrid.md#pymhm.mshho3d.MsHHO3DSolution)
- <a id="pymhm.mshho3d.MsHHO3DSolution.l2_error"></a>[`MsHHO3DSolution.l2_error`](api/hybrid.md#pymhm.mshho3d.MsHHO3DSolution.l2_error)
- <a id="pymhm.mshho3d.MsHHO3DSolution.flux_l2_error"></a>[`MsHHO3DSolution.flux_l2_error`](api/hybrid.md#pymhm.mshho3d.MsHHO3DSolution.flux_l2_error)
- <a id="pymhm.mshho3d.solve_mshho_3d"></a>[`solve_mshho_3d`](api/hybrid.md#pymhm.mshho3d.solve_mshho_3d)

<a id="pymhm.mh2m"></a>**[pymhm.mh2m](api/mh.md#pymhm.mh2m)**

- <a id="pymhm.mh2m.PressureTraceSpace"></a>[`PressureTraceSpace`](api/mh.md#pymhm.mh2m.PressureTraceSpace)
- <a id="pymhm.mh2m.PressureTraceSpace.size"></a>[`PressureTraceSpace.size`](api/mh.md#pymhm.mh2m.PressureTraceSpace.size)
- <a id="pymhm.mh2m.PressureTraceSpace.__post_init__"></a>[`PressureTraceSpace.__post_init__`](api/mh.md#pymhm.mh2m.PressureTraceSpace.__post_init__)
- <a id="pymhm.mh2m.PressureTraceSpace.uniform"></a>[`PressureTraceSpace.uniform`](api/mh.md#pymhm.mh2m.PressureTraceSpace.uniform)
- <a id="pymhm.mh2m.PressureTraceSpace.cell_dofs"></a>[`PressureTraceSpace.cell_dofs`](api/mh.md#pymhm.mh2m.PressureTraceSpace.cell_dofs)
- <a id="pymhm.mh2m.MH2MLocal"></a>[`MH2MLocal`](api/mh.md#pymhm.mh2m.MH2MLocal)
- <a id="pymhm.mh2m.MH2MSolution"></a>[`MH2MSolution`](api/mh.md#pymhm.mh2m.MH2MSolution)
- <a id="pymhm.mh2m.MH2MSolution.conservation_residuals"></a>[`MH2MSolution.conservation_residuals`](api/mh.md#pymhm.mh2m.MH2MSolution.conservation_residuals)
- <a id="pymhm.mh2m.MH2MSolution.trace_moment_residuals"></a>[`MH2MSolution.trace_moment_residuals`](api/mh.md#pymhm.mh2m.MH2MSolution.trace_moment_residuals)
- <a id="pymhm.mh2m.MH2MSolution.local_equation_residuals"></a>[`MH2MSolution.local_equation_residuals`](api/mh.md#pymhm.mh2m.MH2MSolution.local_equation_residuals)
- <a id="pymhm.mh2m.MH2MSolution.l2_error"></a>[`MH2MSolution.l2_error`](api/mh.md#pymhm.mh2m.MH2MSolution.l2_error)
- <a id="pymhm.mh2m.MH2MSolution.gradient_l2_error"></a>[`MH2MSolution.gradient_l2_error`](api/mh.md#pymhm.mh2m.MH2MSolution.gradient_l2_error)
- <a id="pymhm.mh2m.MH2MSolution.flux_l2_error"></a>[`MH2MSolution.flux_l2_error`](api/mh.md#pymhm.mh2m.MH2MSolution.flux_l2_error)
- <a id="pymhm.mh2m.solve_mh2m"></a>[`solve_mh2m`](api/mh.md#pymhm.mh2m.solve_mh2m)

<a id="pymhm.mh"></a>**[pymhm.mh](api/mh.md#pymhm.mh)**

- <a id="pymhm.mh.MHSolution"></a>[`MHSolution`](api/mh.md#pymhm.mh.MHSolution)
- <a id="pymhm.mh.MHSolution.l2_error"></a>[`MHSolution.l2_error`](api/mh.md#pymhm.mh.MHSolution.l2_error)
- <a id="pymhm.mh.MHSolution.flux_l2_error"></a>[`MHSolution.flux_l2_error`](api/mh.md#pymhm.mh.MHSolution.flux_l2_error)
- <a id="pymhm.mh.MHSolution.normal_flux"></a>[`MHSolution.normal_flux`](api/mh.md#pymhm.mh.MHSolution.normal_flux)
- <a id="pymhm.mh.MHSolution.conservation_residuals"></a>[`MHSolution.conservation_residuals`](api/mh.md#pymhm.mh.MHSolution.conservation_residuals)
- <a id="pymhm.mh.solve_mh"></a>[`solve_mh`](api/mh.md#pymhm.mh.solve_mh)

### Darcy in two dimensions

<a id="pymhm.darcy"></a>**[pymhm.darcy](api/darcy.md#pymhm.darcy)**

- <a id="pymhm.darcy.DarcySolution"></a>[`DarcySolution`](api/darcy.md#pymhm.darcy.DarcySolution)
- <a id="pymhm.darcy.DarcySolution.l2_error"></a>[`DarcySolution.l2_error`](api/darcy.md#pymhm.darcy.DarcySolution.l2_error)
- <a id="pymhm.darcy.DarcySolution.flux_l2_error"></a>[`DarcySolution.flux_l2_error`](api/darcy.md#pymhm.darcy.DarcySolution.flux_l2_error)
- <a id="pymhm.darcy.DarcySolution.conservation_residuals"></a>[`DarcySolution.conservation_residuals`](api/darcy.md#pymhm.darcy.DarcySolution.conservation_residuals)
- <a id="pymhm.darcy.DarcySolution.fine_conservation_residuals"></a>[`DarcySolution.fine_conservation_residuals`](api/darcy.md#pymhm.darcy.DarcySolution.fine_conservation_residuals)
- <a id="pymhm.darcy.solve_darcy"></a>[`solve_darcy`](api/darcy.md#pymhm.darcy.solve_darcy)

<a id="pymhm.darcy_mixed"></a>**[pymhm.darcy_mixed](api/darcy.md#pymhm.darcy_mixed)**

- <a id="pymhm.darcy_mixed.BDMDarcySolution"></a>[`BDMDarcySolution`](api/darcy.md#pymhm.darcy_mixed.BDMDarcySolution)
- <a id="pymhm.darcy_mixed.BDMDarcySolution.l2_error"></a>[`BDMDarcySolution.l2_error`](api/darcy.md#pymhm.darcy_mixed.BDMDarcySolution.l2_error)
- <a id="pymhm.darcy_mixed.BDMDarcySolution.flux_l2_error"></a>[`BDMDarcySolution.flux_l2_error`](api/darcy.md#pymhm.darcy_mixed.BDMDarcySolution.flux_l2_error)
- <a id="pymhm.darcy_mixed.BDMDarcySolution.divergence_l2_error"></a>[`BDMDarcySolution.divergence_l2_error`](api/darcy.md#pymhm.darcy_mixed.BDMDarcySolution.divergence_l2_error)
- <a id="pymhm.darcy_mixed.BDMDarcySolution.fine_equilibrium_residuals"></a>[`BDMDarcySolution.fine_equilibrium_residuals`](api/darcy.md#pymhm.darcy_mixed.BDMDarcySolution.fine_equilibrium_residuals)
- <a id="pymhm.darcy_mixed.BDMDarcySolution.fine_conservation_residuals"></a>[`BDMDarcySolution.fine_conservation_residuals`](api/darcy.md#pymhm.darcy_mixed.BDMDarcySolution.fine_conservation_residuals)
- <a id="pymhm.darcy_mixed.BDMDarcySolution.conservation_residuals"></a>[`BDMDarcySolution.conservation_residuals`](api/darcy.md#pymhm.darcy_mixed.BDMDarcySolution.conservation_residuals)
- <a id="pymhm.darcy_mixed.BDMDarcySolution.normal_flux_residuals"></a>[`BDMDarcySolution.normal_flux_residuals`](api/darcy.md#pymhm.darcy_mixed.BDMDarcySolution.normal_flux_residuals)
- <a id="pymhm.darcy_mixed.solve_darcy_bdm"></a>[`solve_darcy_bdm`](api/darcy.md#pymhm.darcy_mixed.solve_darcy_bdm)

<a id="pymhm.darcy_rt"></a>**[pymhm.darcy_rt](api/darcy.md#pymhm.darcy_rt)**

- <a id="pymhm.darcy_rt.RTDarcySolution"></a>[`RTDarcySolution`](api/darcy.md#pymhm.darcy_rt.RTDarcySolution)
- <a id="pymhm.darcy_rt.RTDarcySolution.l2_error"></a>[`RTDarcySolution.l2_error`](api/darcy.md#pymhm.darcy_rt.RTDarcySolution.l2_error)
- <a id="pymhm.darcy_rt.RTDarcySolution.flux_l2_error"></a>[`RTDarcySolution.flux_l2_error`](api/darcy.md#pymhm.darcy_rt.RTDarcySolution.flux_l2_error)
- <a id="pymhm.darcy_rt.RTDarcySolution.divergence_l2_error"></a>[`RTDarcySolution.divergence_l2_error`](api/darcy.md#pymhm.darcy_rt.RTDarcySolution.divergence_l2_error)
- <a id="pymhm.darcy_rt.RTDarcySolution.fine_equilibrium_residuals"></a>[`RTDarcySolution.fine_equilibrium_residuals`](api/darcy.md#pymhm.darcy_rt.RTDarcySolution.fine_equilibrium_residuals)
- <a id="pymhm.darcy_rt.RTDarcySolution.fine_conservation_residuals"></a>[`RTDarcySolution.fine_conservation_residuals`](api/darcy.md#pymhm.darcy_rt.RTDarcySolution.fine_conservation_residuals)
- <a id="pymhm.darcy_rt.RTDarcySolution.conservation_residuals"></a>[`RTDarcySolution.conservation_residuals`](api/darcy.md#pymhm.darcy_rt.RTDarcySolution.conservation_residuals)
- <a id="pymhm.darcy_rt.RTDarcySolution.normal_flux_residuals"></a>[`RTDarcySolution.normal_flux_residuals`](api/darcy.md#pymhm.darcy_rt.RTDarcySolution.normal_flux_residuals)
- <a id="pymhm.darcy_rt.pressure_basis"></a>[`pressure_basis`](api/darcy.md#pymhm.darcy_rt.pressure_basis)
- <a id="pymhm.darcy_rt.rt_operators"></a>[`rt_operators`](api/darcy.md#pymhm.darcy_rt.rt_operators)
- <a id="pymhm.darcy_rt.rt_trace_map"></a>[`rt_trace_map`](api/darcy.md#pymhm.darcy_rt.rt_trace_map)
- <a id="pymhm.darcy_rt.solve_darcy_rt"></a>[`solve_darcy_rt`](api/darcy.md#pymhm.darcy_rt.solve_darcy_rt)
- <a id="pymhm.darcy_rt.solve_darcy_rt_conforming"></a>[`solve_darcy_rt_conforming`](api/darcy.md#pymhm.darcy_rt.solve_darcy_rt_conforming)

<a id="pymhm.conforming"></a>**[pymhm.conforming](api/darcy.md#pymhm.conforming)**

- <a id="pymhm.conforming.ConformingQuadrilateralSolution"></a>[`ConformingQuadrilateralSolution`](api/darcy.md#pymhm.conforming.ConformingQuadrilateralSolution)
- <a id="pymhm.conforming.ConformingQuadrilateralSolution.evaluate"></a>[`ConformingQuadrilateralSolution.evaluate`](api/darcy.md#pymhm.conforming.ConformingQuadrilateralSolution.evaluate)
- <a id="pymhm.conforming.ConformingQuadrilateralSolution.physical_flux"></a>[`ConformingQuadrilateralSolution.physical_flux`](api/darcy.md#pymhm.conforming.ConformingQuadrilateralSolution.physical_flux)
- <a id="pymhm.conforming.solve_conforming_quadrilateral"></a>[`solve_conforming_quadrilateral`](api/darcy.md#pymhm.conforming.solve_conforming_quadrilateral)

<a id="pymhm.analytic"></a>**[pymhm.analytic](api/darcy.md#pymhm.analytic)**

- <a id="pymhm.analytic.AnalyticDarcySpace"></a>[`AnalyticDarcySpace`](api/darcy.md#pymhm.analytic.AnalyticDarcySpace)
- <a id="pymhm.analytic.AnalyticDarcySpace.evaluate"></a>[`AnalyticDarcySpace.evaluate`](api/darcy.md#pymhm.analytic.AnalyticDarcySpace.evaluate)
- <a id="pymhm.analytic.AnalyticDarcySpace.integrate_rt0"></a>[`AnalyticDarcySpace.integrate_rt0`](api/darcy.md#pymhm.analytic.AnalyticDarcySpace.integrate_rt0)
- <a id="pymhm.analytic.AnalyticDarcySolution"></a>[`AnalyticDarcySolution`](api/darcy.md#pymhm.analytic.AnalyticDarcySolution)
- <a id="pymhm.analytic.AnalyticDarcySolution.pressure_update"></a>[`AnalyticDarcySolution.pressure_update`](api/darcy.md#pymhm.analytic.AnalyticDarcySolution.pressure_update)
- <a id="pymhm.analytic.AnalyticDarcySolution.evaluate"></a>[`AnalyticDarcySolution.evaluate`](api/darcy.md#pymhm.analytic.AnalyticDarcySolution.evaluate)
- <a id="pymhm.analytic.AnalyticDarcySolution.errors"></a>[`AnalyticDarcySolution.errors`](api/darcy.md#pymhm.analytic.AnalyticDarcySolution.errors)
- <a id="pymhm.analytic.analytic_darcy_local"></a>[`analytic_darcy_local`](api/darcy.md#pymhm.analytic.analytic_darcy_local)
- <a id="pymhm.analytic.solve_darcy_analytic"></a>[`solve_darcy_analytic`](api/darcy.md#pymhm.analytic.solve_darcy_analytic)

<a id="pymhm.quadrilateral"></a>**[pymhm.quadrilateral](api/darcy.md#pymhm.quadrilateral)**

- <a id="pymhm.quadrilateral.CartesianMacroMesh"></a>[`CartesianMacroMesh`](api/darcy.md#pymhm.quadrilateral.CartesianMacroMesh)
- <a id="pymhm.quadrilateral.CartesianMacroMesh.spacing"></a>[`CartesianMacroMesh.spacing`](api/darcy.md#pymhm.quadrilateral.CartesianMacroMesh.spacing)
- <a id="pymhm.quadrilateral.CartesianMacroMesh.areas"></a>[`CartesianMacroMesh.areas`](api/darcy.md#pymhm.quadrilateral.CartesianMacroMesh.areas)
- <a id="pymhm.quadrilateral.CartesianMacroMesh.lengths"></a>[`CartesianMacroMesh.lengths`](api/darcy.md#pymhm.quadrilateral.CartesianMacroMesh.lengths)
- <a id="pymhm.quadrilateral.CartesianMacroMesh.normals"></a>[`CartesianMacroMesh.normals`](api/darcy.md#pymhm.quadrilateral.CartesianMacroMesh.normals)
- <a id="pymhm.quadrilateral.CartesianMacroMesh.boundary_faces"></a>[`CartesianMacroMesh.boundary_faces`](api/darcy.md#pymhm.quadrilateral.CartesianMacroMesh.boundary_faces)
- <a id="pymhm.quadrilateral.CartesianMacroMesh.__post_init__"></a>[`CartesianMacroMesh.__post_init__`](api/darcy.md#pymhm.quadrilateral.CartesianMacroMesh.__post_init__)
- <a id="pymhm.quadrilateral.CartesianMacroMesh.submesh"></a>[`CartesianMacroMesh.submesh`](api/darcy.md#pymhm.quadrilateral.CartesianMacroMesh.submesh)
- <a id="pymhm.quadrilateral.QuadrilateralDarcySolution"></a>[`QuadrilateralDarcySolution`](api/darcy.md#pymhm.quadrilateral.QuadrilateralDarcySolution)
- <a id="pymhm.quadrilateral.QuadrilateralDarcySolution.evaluate"></a>[`QuadrilateralDarcySolution.evaluate`](api/darcy.md#pymhm.quadrilateral.QuadrilateralDarcySolution.evaluate)
- <a id="pymhm.quadrilateral.QuadrilateralDarcySolution.l2_error"></a>[`QuadrilateralDarcySolution.l2_error`](api/darcy.md#pymhm.quadrilateral.QuadrilateralDarcySolution.l2_error)
- <a id="pymhm.quadrilateral.QuadrilateralDarcySolution.flux_l2_error"></a>[`QuadrilateralDarcySolution.flux_l2_error`](api/darcy.md#pymhm.quadrilateral.QuadrilateralDarcySolution.flux_l2_error)
- <a id="pymhm.quadrilateral.QuadrilateralDarcySolution.conservation_residuals"></a>[`QuadrilateralDarcySolution.conservation_residuals`](api/darcy.md#pymhm.quadrilateral.QuadrilateralDarcySolution.conservation_residuals)
- <a id="pymhm.quadrilateral.quadrilateral_quadrature"></a>[`quadrilateral_quadrature`](api/darcy.md#pymhm.quadrilateral.quadrilateral_quadrature)
- <a id="pymhm.quadrilateral.qk_basis"></a>[`qk_basis`](api/darcy.md#pymhm.quadrilateral.qk_basis)
- <a id="pymhm.quadrilateral.qk_space"></a>[`qk_space`](api/darcy.md#pymhm.quadrilateral.qk_space)
- <a id="pymhm.quadrilateral.quadrilateral_operators"></a>[`quadrilateral_operators`](api/darcy.md#pymhm.quadrilateral.quadrilateral_operators)
- <a id="pymhm.quadrilateral.quadrilateral_trace_coupling"></a>[`quadrilateral_trace_coupling`](api/darcy.md#pymhm.quadrilateral.quadrilateral_trace_coupling)
- <a id="pymhm.quadrilateral.solve_darcy_quadrilateral"></a>[`solve_darcy_quadrilateral`](api/darcy.md#pymhm.quadrilateral.solve_darcy_quadrilateral)

<a id="pymhm.tensor_rt"></a>**[pymhm.tensor_rt](api/darcy.md#pymhm.tensor_rt)**

- <a id="pymhm.tensor_rt.TensorRTDarcySolution"></a>[`TensorRTDarcySolution`](api/darcy.md#pymhm.tensor_rt.TensorRTDarcySolution)
- <a id="pymhm.tensor_rt.TensorRTDarcySolution.evaluate"></a>[`TensorRTDarcySolution.evaluate`](api/darcy.md#pymhm.tensor_rt.TensorRTDarcySolution.evaluate)
- <a id="pymhm.tensor_rt.TensorRTDarcySolution.errors"></a>[`TensorRTDarcySolution.errors`](api/darcy.md#pymhm.tensor_rt.TensorRTDarcySolution.errors)
- <a id="pymhm.tensor_rt.TensorRTDarcySolution.equilibrium_residuals"></a>[`TensorRTDarcySolution.equilibrium_residuals`](api/darcy.md#pymhm.tensor_rt.TensorRTDarcySolution.equilibrium_residuals)
- <a id="pymhm.tensor_rt.TensorRTDarcySolution.normal_flux_residuals"></a>[`TensorRTDarcySolution.normal_flux_residuals`](api/darcy.md#pymhm.tensor_rt.TensorRTDarcySolution.normal_flux_residuals)
- <a id="pymhm.tensor_rt.tensor_rt_basis"></a>[`tensor_rt_basis`](api/darcy.md#pymhm.tensor_rt.tensor_rt_basis)
- <a id="pymhm.tensor_rt.tensor_rt_dofs"></a>[`tensor_rt_dofs`](api/darcy.md#pymhm.tensor_rt.tensor_rt_dofs)
- <a id="pymhm.tensor_rt.solve_darcy_tensor_rt"></a>[`solve_darcy_tensor_rt`](api/darcy.md#pymhm.tensor_rt.solve_darcy_tensor_rt)

<a id="pymhm.pgmhm"></a>**[pymhm.pgmhm](api/darcy.md#pymhm.pgmhm)**

- <a id="pymhm.pgmhm.PGMHMSolution"></a>[`PGMHMSolution`](api/darcy.md#pymhm.pgmhm.PGMHMSolution)
- <a id="pymhm.pgmhm.PGMHMSolution.l2_error"></a>[`PGMHMSolution.l2_error`](api/darcy.md#pymhm.pgmhm.PGMHMSolution.l2_error)
- <a id="pymhm.pgmhm.PGMHMSolution.flux_l2_error"></a>[`PGMHMSolution.flux_l2_error`](api/darcy.md#pymhm.pgmhm.PGMHMSolution.flux_l2_error)
- <a id="pymhm.pgmhm.PGMHMSolution.normal_flux"></a>[`PGMHMSolution.normal_flux`](api/darcy.md#pymhm.pgmhm.PGMHMSolution.normal_flux)
- <a id="pymhm.pgmhm.PGMHMSolution.conservation_residuals"></a>[`PGMHMSolution.conservation_residuals`](api/darcy.md#pymhm.pgmhm.PGMHMSolution.conservation_residuals)
- <a id="pymhm.pgmhm.solve_pgmhm"></a>[`solve_pgmhm`](api/darcy.md#pymhm.pgmhm.solve_pgmhm)

### Darcy in three dimensions

<a id="pymhm.darcy3d"></a>**[pymhm.darcy3d](api/darcy3d.md#pymhm.darcy3d)**

- <a id="pymhm.darcy3d.TriangularSkeleton"></a>[`TriangularSkeleton`](api/darcy3d.md#pymhm.darcy3d.TriangularSkeleton)
- <a id="pymhm.darcy3d.TriangularSkeleton.__init__"></a>[`TriangularSkeleton.__init__`](api/darcy3d.md#pymhm.darcy3d.TriangularSkeleton.__init__)
- <a id="pymhm.darcy3d.TriangularSkeleton.face_partition"></a>[`TriangularSkeleton.face_partition`](api/darcy3d.md#pymhm.darcy3d.TriangularSkeleton.face_partition)
- <a id="pymhm.darcy3d.TriangularSkeleton.face_weights"></a>[`TriangularSkeleton.face_weights`](api/darcy3d.md#pymhm.darcy3d.TriangularSkeleton.face_weights)
- <a id="pymhm.darcy3d.TriangularSkeleton.basis"></a>[`TriangularSkeleton.basis`](api/darcy3d.md#pymhm.darcy3d.TriangularSkeleton.basis)
- <a id="pymhm.darcy3d.TriangularSkeleton.dofs"></a>[`TriangularSkeleton.dofs`](api/darcy3d.md#pymhm.darcy3d.TriangularSkeleton.dofs)
- <a id="pymhm.darcy3d.TriangularSkeleton.cell_dofs"></a>[`TriangularSkeleton.cell_dofs`](api/darcy3d.md#pymhm.darcy3d.TriangularSkeleton.cell_dofs)
- <a id="pymhm.darcy3d.Darcy3DSolution"></a>[`Darcy3DSolution`](api/darcy3d.md#pymhm.darcy3d.Darcy3DSolution)
- <a id="pymhm.darcy3d.Darcy3DSolution.evaluate"></a>[`Darcy3DSolution.evaluate`](api/darcy3d.md#pymhm.darcy3d.Darcy3DSolution.evaluate)
- <a id="pymhm.darcy3d.Darcy3DSolution.l2_error"></a>[`Darcy3DSolution.l2_error`](api/darcy3d.md#pymhm.darcy3d.Darcy3DSolution.l2_error)
- <a id="pymhm.darcy3d.Darcy3DSolution.flux_l2_error"></a>[`Darcy3DSolution.flux_l2_error`](api/darcy3d.md#pymhm.darcy3d.Darcy3DSolution.flux_l2_error)
- <a id="pymhm.darcy3d.Darcy3DSolution.conservation_residuals"></a>[`Darcy3DSolution.conservation_residuals`](api/darcy3d.md#pymhm.darcy3d.Darcy3DSolution.conservation_residuals)
- <a id="pymhm.darcy3d.tetra_trace_coupling"></a>[`tetra_trace_coupling`](api/darcy3d.md#pymhm.darcy3d.tetra_trace_coupling)
- <a id="pymhm.darcy3d.solve_darcy_3d"></a>[`solve_darcy_3d`](api/darcy3d.md#pymhm.darcy3d.solve_darcy_3d)

<a id="pymhm.mapped_rt"></a>**[pymhm.mapped_rt](api/darcy3d.md#pymhm.mapped_rt)**

- <a id="pymhm.mapped_rt.HexMesh"></a>[`HexMesh`](api/darcy3d.md#pymhm.mapped_rt.HexMesh)
- <a id="pymhm.mapped_rt.HexMesh.__post_init__"></a>[`HexMesh.__post_init__`](api/darcy3d.md#pymhm.mapped_rt.HexMesh.__post_init__)
- <a id="pymhm.mapped_rt.HexMesh.geometry"></a>[`HexMesh.geometry`](api/darcy3d.md#pymhm.mapped_rt.HexMesh.geometry)
- <a id="pymhm.mapped_rt.HexMesh.submesh"></a>[`HexMesh.submesh`](api/darcy3d.md#pymhm.mapped_rt.HexMesh.submesh)
- <a id="pymhm.mapped_rt.HexMesh.refined"></a>[`HexMesh.refined`](api/darcy3d.md#pymhm.mapped_rt.HexMesh.refined)
- <a id="pymhm.mapped_rt.HexMesh.unit_cube"></a>[`HexMesh.unit_cube`](api/darcy3d.md#pymhm.mapped_rt.HexMesh.unit_cube)
- <a id="pymhm.mapped_rt.HexMesh.annular_prism"></a>[`HexMesh.annular_prism`](api/darcy3d.md#pymhm.mapped_rt.HexMesh.annular_prism)
- <a id="pymhm.mapped_rt.HexSkeleton"></a>[`HexSkeleton`](api/darcy3d.md#pymhm.mapped_rt.HexSkeleton)
- <a id="pymhm.mapped_rt.HexSkeleton.face_size"></a>[`HexSkeleton.face_size`](api/darcy3d.md#pymhm.mapped_rt.HexSkeleton.face_size)
- <a id="pymhm.mapped_rt.HexSkeleton.size"></a>[`HexSkeleton.size`](api/darcy3d.md#pymhm.mapped_rt.HexSkeleton.size)
- <a id="pymhm.mapped_rt.HexSkeleton.__post_init__"></a>[`HexSkeleton.__post_init__`](api/darcy3d.md#pymhm.mapped_rt.HexSkeleton.__post_init__)
- <a id="pymhm.mapped_rt.HexSkeleton.cell_dofs"></a>[`HexSkeleton.cell_dofs`](api/darcy3d.md#pymhm.mapped_rt.HexSkeleton.cell_dofs)
- <a id="pymhm.mapped_rt.HexSkeleton.evaluate"></a>[`HexSkeleton.evaluate`](api/darcy3d.md#pymhm.mapped_rt.HexSkeleton.evaluate)
- <a id="pymhm.mapped_rt.MappedRTDarcySolution"></a>[`MappedRTDarcySolution`](api/darcy3d.md#pymhm.mapped_rt.MappedRTDarcySolution)
- <a id="pymhm.mapped_rt.MappedRTDarcySolution.evaluate"></a>[`MappedRTDarcySolution.evaluate`](api/darcy3d.md#pymhm.mapped_rt.MappedRTDarcySolution.evaluate)
- <a id="pymhm.mapped_rt.MappedRTDarcySolution.errors"></a>[`MappedRTDarcySolution.errors`](api/darcy3d.md#pymhm.mapped_rt.MappedRTDarcySolution.errors)
- <a id="pymhm.mapped_rt.MappedRTDarcySolution.equilibrium_residuals"></a>[`MappedRTDarcySolution.equilibrium_residuals`](api/darcy3d.md#pymhm.mapped_rt.MappedRTDarcySolution.equilibrium_residuals)
- <a id="pymhm.mapped_rt.cube_quadrature"></a>[`cube_quadrature`](api/darcy3d.md#pymhm.mapped_rt.cube_quadrature)
- <a id="pymhm.mapped_rt.mapped_rt_dofs"></a>[`mapped_rt_dofs`](api/darcy3d.md#pymhm.mapped_rt.mapped_rt_dofs)
- <a id="pymhm.mapped_rt.mapped_rt_basis"></a>[`mapped_rt_basis`](api/darcy3d.md#pymhm.mapped_rt.mapped_rt_basis)
- <a id="pymhm.mapped_rt.solve_darcy_mapped_rt"></a>[`solve_darcy_mapped_rt`](api/darcy3d.md#pymhm.mapped_rt.solve_darcy_mapped_rt)

<a id="pymhm.hdiv3d_family"></a>**[pymhm.hdiv3d_family](api/darcy3d.md#pymhm.hdiv3d_family)**

- <a id="pymhm.hdiv3d_family.HDiv3DFamily"></a>[`HDiv3DFamily`](api/darcy3d.md#pymhm.hdiv3d_family.HDiv3DFamily)
- <a id="pymhm.hdiv3d_family.HDiv3DFamily.face_sizes"></a>[`HDiv3DFamily.face_sizes`](api/darcy3d.md#pymhm.hdiv3d_family.HDiv3DFamily.face_sizes)
- <a id="pymhm.hdiv3d_family.HDiv3DFamily.local_size"></a>[`HDiv3DFamily.local_size`](api/darcy3d.md#pymhm.hdiv3d_family.HDiv3DFamily.local_size)
- <a id="pymhm.hdiv3d_family.HDiv3DFamily.pressure_size"></a>[`HDiv3DFamily.pressure_size`](api/darcy3d.md#pymhm.hdiv3d_family.HDiv3DFamily.pressure_size)
- <a id="pymhm.hdiv3d_family.HDiv3DFamily.interior_size"></a>[`HDiv3DFamily.interior_size`](api/darcy3d.md#pymhm.hdiv3d_family.HDiv3DFamily.interior_size)
- <a id="pymhm.hdiv3d_family.HDiv3DFamily.coefficients"></a>[`HDiv3DFamily.coefficients`](api/darcy3d.md#pymhm.hdiv3d_family.HDiv3DFamily.coefficients)
- <a id="pymhm.hdiv3d_family.HDiv3DFamily.interior_moment_seeds"></a>[`HDiv3DFamily.interior_moment_seeds`](api/darcy3d.md#pymhm.hdiv3d_family.HDiv3DFamily.interior_moment_seeds)
- <a id="pymhm.hdiv3d_family.HDiv3DFamily.__post_init__"></a>[`HDiv3DFamily.__post_init__`](api/darcy3d.md#pymhm.hdiv3d_family.HDiv3DFamily.__post_init__)
- <a id="pymhm.hdiv3d_family.HDiv3DFamily.tabulate"></a>[`HDiv3DFamily.tabulate`](api/darcy3d.md#pymhm.hdiv3d_family.HDiv3DFamily.tabulate)
- <a id="pymhm.hdiv3d_family.reference_vertices"></a>[`reference_vertices`](api/darcy3d.md#pymhm.hdiv3d_family.reference_vertices)
- <a id="pymhm.hdiv3d_family.reference_faces"></a>[`reference_faces`](api/darcy3d.md#pymhm.hdiv3d_family.reference_faces)
- <a id="pymhm.hdiv3d_family.cell_quadrature"></a>[`cell_quadrature`](api/darcy3d.md#pymhm.hdiv3d_family.cell_quadrature)
- <a id="pymhm.hdiv3d_family.face_quadrature"></a>[`face_quadrature`](api/darcy3d.md#pymhm.hdiv3d_family.face_quadrature)
- <a id="pymhm.hdiv3d_family.face_shape"></a>[`face_shape`](api/darcy3d.md#pymhm.hdiv3d_family.face_shape)
- <a id="pymhm.hdiv3d_family.face_polynomials"></a>[`face_polynomials`](api/darcy3d.md#pymhm.hdiv3d_family.face_polynomials)
- <a id="pymhm.hdiv3d_family.face_size"></a>[`face_size`](api/darcy3d.md#pymhm.hdiv3d_family.face_size)

<a id="pymhm.hdiv3d_mesh"></a>**[pymhm.hdiv3d_mesh](api/darcy3d.md#pymhm.hdiv3d_mesh)**

- <a id="pymhm.hdiv3d_mesh.AffineMixedMesh"></a>[`AffineMixedMesh`](api/darcy3d.md#pymhm.hdiv3d_mesh.AffineMixedMesh)
- <a id="pymhm.hdiv3d_mesh.AffineMixedMesh.__init__"></a>[`AffineMixedMesh.__init__`](api/darcy3d.md#pymhm.hdiv3d_mesh.AffineMixedMesh.__init__)
- <a id="pymhm.hdiv3d_mesh.AffineMixedMesh.geometry"></a>[`AffineMixedMesh.geometry`](api/darcy3d.md#pymhm.hdiv3d_mesh.AffineMixedMesh.geometry)
- <a id="pymhm.hdiv3d_mesh.AffineMixedMesh.face_coordinates"></a>[`AffineMixedMesh.face_coordinates`](api/darcy3d.md#pymhm.hdiv3d_mesh.AffineMixedMesh.face_coordinates)
- <a id="pymhm.hdiv3d_mesh.AffineMixedMesh.submesh"></a>[`AffineMixedMesh.submesh`](api/darcy3d.md#pymhm.hdiv3d_mesh.AffineMixedMesh.submesh)
- <a id="pymhm.hdiv3d_mesh.AffineMixedMesh.refined"></a>[`AffineMixedMesh.refined`](api/darcy3d.md#pymhm.hdiv3d_mesh.AffineMixedMesh.refined)
- <a id="pymhm.hdiv3d_mesh.AffineMixedMesh.unit_cube"></a>[`AffineMixedMesh.unit_cube`](api/darcy3d.md#pymhm.hdiv3d_mesh.AffineMixedMesh.unit_cube)
- <a id="pymhm.hdiv3d_mesh.AffineMixedMesh.from_extruded_hexahedra"></a>[`AffineMixedMesh.from_extruded_hexahedra`](api/darcy3d.md#pymhm.hdiv3d_mesh.AffineMixedMesh.from_extruded_hexahedra)
- <a id="pymhm.hdiv3d_mesh.reference_submesh"></a>[`reference_submesh`](api/darcy3d.md#pymhm.hdiv3d_mesh.reference_submesh)
- <a id="pymhm.hdiv3d_mesh.hdiv3d_face_offsets"></a>[`hdiv3d_face_offsets`](api/darcy3d.md#pymhm.hdiv3d_mesh.hdiv3d_face_offsets)
- <a id="pymhm.hdiv3d_mesh.hdiv3d_dofs"></a>[`hdiv3d_dofs`](api/darcy3d.md#pymhm.hdiv3d_mesh.hdiv3d_dofs)
- <a id="pymhm.hdiv3d_mesh.hdiv3d_transform"></a>[`hdiv3d_transform`](api/darcy3d.md#pymhm.hdiv3d_mesh.hdiv3d_transform)
- <a id="pymhm.hdiv3d_mesh.hdiv3d_basis"></a>[`hdiv3d_basis`](api/darcy3d.md#pymhm.hdiv3d_mesh.hdiv3d_basis)

<a id="pymhm.darcy_hdiv3d"></a>**[pymhm.darcy_hdiv3d](api/darcy3d.md#pymhm.darcy_hdiv3d)**

- <a id="pymhm.darcy_hdiv3d.Mixed3DSkeleton"></a>[`Mixed3DSkeleton`](api/darcy3d.md#pymhm.darcy_hdiv3d.Mixed3DSkeleton)
- <a id="pymhm.darcy_hdiv3d.Mixed3DSkeleton.size"></a>[`Mixed3DSkeleton.size`](api/darcy3d.md#pymhm.darcy_hdiv3d.Mixed3DSkeleton.size)
- <a id="pymhm.darcy_hdiv3d.Mixed3DSkeleton.__init__"></a>[`Mixed3DSkeleton.__init__`](api/darcy3d.md#pymhm.darcy_hdiv3d.Mixed3DSkeleton.__init__)
- <a id="pymhm.darcy_hdiv3d.Mixed3DSkeleton.cell_dofs"></a>[`Mixed3DSkeleton.cell_dofs`](api/darcy3d.md#pymhm.darcy_hdiv3d.Mixed3DSkeleton.cell_dofs)
- <a id="pymhm.darcy_hdiv3d.Mixed3DDarcySolution"></a>[`Mixed3DDarcySolution`](api/darcy3d.md#pymhm.darcy_hdiv3d.Mixed3DDarcySolution)
- <a id="pymhm.darcy_hdiv3d.Mixed3DDarcySolution.evaluate"></a>[`Mixed3DDarcySolution.evaluate`](api/darcy3d.md#pymhm.darcy_hdiv3d.Mixed3DDarcySolution.evaluate)
- <a id="pymhm.darcy_hdiv3d.Mixed3DDarcySolution.errors"></a>[`Mixed3DDarcySolution.errors`](api/darcy3d.md#pymhm.darcy_hdiv3d.Mixed3DDarcySolution.errors)
- <a id="pymhm.darcy_hdiv3d.Mixed3DDarcySolution.equilibrium_residuals"></a>[`Mixed3DDarcySolution.equilibrium_residuals`](api/darcy3d.md#pymhm.darcy_hdiv3d.Mixed3DDarcySolution.equilibrium_residuals)
- <a id="pymhm.darcy_hdiv3d.hdiv3d_operators"></a>[`hdiv3d_operators`](api/darcy3d.md#pymhm.darcy_hdiv3d.hdiv3d_operators)
- <a id="pymhm.darcy_hdiv3d.solve_darcy_hdiv3d"></a>[`solve_darcy_hdiv3d`](api/darcy3d.md#pymhm.darcy_hdiv3d.solve_darcy_hdiv3d)

<a id="pymhm.rt3d"></a>**[pymhm.rt3d](api/darcy3d.md#pymhm.rt3d)**

- <a id="pymhm.rt3d.RTTetraFamily"></a>[`RTTetraFamily`](api/darcy3d.md#pymhm.rt3d.RTTetraFamily)
- <a id="pymhm.rt3d.RTTetraFamily.coefficients"></a>[`RTTetraFamily.coefficients`](api/darcy3d.md#pymhm.rt3d.RTTetraFamily.coefficients)
- <a id="pymhm.rt3d.RTTetraFamily.__init__"></a>[`RTTetraFamily.__init__`](api/darcy3d.md#pymhm.rt3d.RTTetraFamily.__init__)
- <a id="pymhm.rt3d.RTTetraFamily.tabulate"></a>[`RTTetraFamily.tabulate`](api/darcy3d.md#pymhm.rt3d.RTTetraFamily.tabulate)
- <a id="pymhm.rt3d.rt3d_interior_tests"></a>[`rt3d_interior_tests`](api/darcy3d.md#pymhm.rt3d.rt3d_interior_tests)

### Flux reconstruction and error estimation

<a id="pymhm.reconstruction"></a>**[pymhm.reconstruction](api/reconstruction.md#pymhm.reconstruction)**

- <a id="pymhm.reconstruction.EquilibratedFlux"></a>[`EquilibratedFlux`](api/reconstruction.md#pymhm.reconstruction.EquilibratedFlux)
- <a id="pymhm.reconstruction.EquilibratedFlux.conservation_residuals"></a>[`EquilibratedFlux.conservation_residuals`](api/reconstruction.md#pymhm.reconstruction.EquilibratedFlux.conservation_residuals)
- <a id="pymhm.reconstruction.EquilibratedFlux.l2_error"></a>[`EquilibratedFlux.l2_error`](api/reconstruction.md#pymhm.reconstruction.EquilibratedFlux.l2_error)
- <a id="pymhm.reconstruction.equilibrate_flux"></a>[`equilibrate_flux`](api/reconstruction.md#pymhm.reconstruction.equilibrate_flux)

<a id="pymhm.reconstruction_moments"></a>**[pymhm.reconstruction_moments](api/reconstruction.md#pymhm.reconstruction_moments)**

- <a id="pymhm.reconstruction_moments.MomentFluxSolution"></a>[`MomentFluxSolution`](api/reconstruction.md#pymhm.reconstruction_moments.MomentFluxSolution)
- <a id="pymhm.reconstruction_moments.MomentFluxSolution.flux_l2_error"></a>[`MomentFluxSolution.flux_l2_error`](api/reconstruction.md#pymhm.reconstruction_moments.MomentFluxSolution.flux_l2_error)
- <a id="pymhm.reconstruction_moments.MomentFluxSolution.divergence_l2_error"></a>[`MomentFluxSolution.divergence_l2_error`](api/reconstruction.md#pymhm.reconstruction_moments.MomentFluxSolution.divergence_l2_error)
- <a id="pymhm.reconstruction_moments.MomentFluxSolution.continuous_moment_residuals"></a>[`MomentFluxSolution.continuous_moment_residuals`](api/reconstruction.md#pymhm.reconstruction_moments.MomentFluxSolution.continuous_moment_residuals)
- <a id="pymhm.reconstruction_moments.MomentFluxSolution.fine_conservation_residuals"></a>[`MomentFluxSolution.fine_conservation_residuals`](api/reconstruction.md#pymhm.reconstruction_moments.MomentFluxSolution.fine_conservation_residuals)
- <a id="pymhm.reconstruction_moments.MomentFluxSolution.conservation_residuals"></a>[`MomentFluxSolution.conservation_residuals`](api/reconstruction.md#pymhm.reconstruction_moments.MomentFluxSolution.conservation_residuals)
- <a id="pymhm.reconstruction_moments.MomentFluxSolution.normal_flux_residuals"></a>[`MomentFluxSolution.normal_flux_residuals`](api/reconstruction.md#pymhm.reconstruction_moments.MomentFluxSolution.normal_flux_residuals)
- <a id="pymhm.reconstruction_moments.MomentFluxSolution.continuous_divergence_projection"></a>[`MomentFluxSolution.continuous_divergence_projection`](api/reconstruction.md#pymhm.reconstruction_moments.MomentFluxSolution.continuous_divergence_projection)
- <a id="pymhm.reconstruction_moments.MomentFluxSolution.projected_divergence_l2_error"></a>[`MomentFluxSolution.projected_divergence_l2_error`](api/reconstruction.md#pymhm.reconstruction_moments.MomentFluxSolution.projected_divergence_l2_error)
- <a id="pymhm.reconstruction_moments.reconstruct_flux_moments"></a>[`reconstruct_flux_moments`](api/reconstruction.md#pymhm.reconstruction_moments.reconstruct_flux_moments)
- <a id="pymhm.reconstruction_moments.reconstruct_darcy_moments"></a>[`reconstruct_darcy_moments`](api/reconstruction.md#pymhm.reconstruction_moments.reconstruct_darcy_moments)

<a id="pymhm.reconstruction3d"></a>**[pymhm.reconstruction3d](api/reconstruction.md#pymhm.reconstruction3d)**

- <a id="pymhm.reconstruction3d.MomentFlux3DSolution"></a>[`MomentFlux3DSolution`](api/reconstruction.md#pymhm.reconstruction3d.MomentFlux3DSolution)
- <a id="pymhm.reconstruction3d.MomentFlux3DSolution.evaluate"></a>[`MomentFlux3DSolution.evaluate`](api/reconstruction.md#pymhm.reconstruction3d.MomentFlux3DSolution.evaluate)
- <a id="pymhm.reconstruction3d.MomentFlux3DSolution.flux_l2_error"></a>[`MomentFlux3DSolution.flux_l2_error`](api/reconstruction.md#pymhm.reconstruction3d.MomentFlux3DSolution.flux_l2_error)
- <a id="pymhm.reconstruction3d.MomentFlux3DSolution.normal_flux_residuals"></a>[`MomentFlux3DSolution.normal_flux_residuals`](api/reconstruction.md#pymhm.reconstruction3d.MomentFlux3DSolution.normal_flux_residuals)
- <a id="pymhm.reconstruction3d.MomentFlux3DSolution.continuous_moment_residuals"></a>[`MomentFlux3DSolution.continuous_moment_residuals`](api/reconstruction.md#pymhm.reconstruction3d.MomentFlux3DSolution.continuous_moment_residuals)
- <a id="pymhm.reconstruction3d.MomentFlux3DSolution.fine_conservation_residuals"></a>[`MomentFlux3DSolution.fine_conservation_residuals`](api/reconstruction.md#pymhm.reconstruction3d.MomentFlux3DSolution.fine_conservation_residuals)
- <a id="pymhm.reconstruction3d.reconstruct_darcy_moments_3d"></a>[`reconstruct_darcy_moments_3d`](api/reconstruction.md#pymhm.reconstruction3d.reconstruct_darcy_moments_3d)

<a id="pymhm.estimator"></a>**[pymhm.estimator](api/reconstruction.md#pymhm.estimator)**

- <a id="pymhm.estimator.ConformingPotential"></a>[`ConformingPotential`](api/reconstruction.md#pymhm.estimator.ConformingPotential)
- <a id="pymhm.estimator.ConformingPotential.l2_error"></a>[`ConformingPotential.l2_error`](api/reconstruction.md#pymhm.estimator.ConformingPotential.l2_error)
- <a id="pymhm.estimator.DarcyEstimator"></a>[`DarcyEstimator`](api/reconstruction.md#pymhm.estimator.DarcyEstimator)
- <a id="pymhm.estimator.DarcyEstimator.local_squared"></a>[`DarcyEstimator.local_squared`](api/reconstruction.md#pymhm.estimator.DarcyEstimator.local_squared)
- <a id="pymhm.estimator.DarcyEstimator.total"></a>[`DarcyEstimator.total`](api/reconstruction.md#pymhm.estimator.DarcyEstimator.total)
- <a id="pymhm.estimator.DarcyEstimator.energy_error"></a>[`DarcyEstimator.energy_error`](api/reconstruction.md#pymhm.estimator.DarcyEstimator.energy_error)
- <a id="pymhm.estimator.recover_potential"></a>[`recover_potential`](api/reconstruction.md#pymhm.estimator.recover_potential)
- <a id="pymhm.estimator.estimate_darcy_error"></a>[`estimate_darcy_error`](api/reconstruction.md#pymhm.estimator.estimate_darcy_error)

<a id="pymhm.weighted_estimator"></a>**[pymhm.weighted_estimator](api/reconstruction.md#pymhm.weighted_estimator)**

- <a id="pymhm.weighted_estimator.WeightedDarcyEstimator"></a>[`WeightedDarcyEstimator`](api/reconstruction.md#pymhm.weighted_estimator.WeightedDarcyEstimator)
- <a id="pymhm.weighted_estimator.WeightedDarcyEstimator.convention"></a>[`WeightedDarcyEstimator.convention`](api/reconstruction.md#pymhm.weighted_estimator.WeightedDarcyEstimator.convention)
- <a id="pymhm.weighted_estimator.WeightedDarcyEstimator.energy_error"></a>[`WeightedDarcyEstimator.energy_error`](api/reconstruction.md#pymhm.weighted_estimator.WeightedDarcyEstimator.energy_error)
- <a id="pymhm.weighted_estimator.PublishedDarcyIndicator"></a>[`PublishedDarcyIndicator`](api/reconstruction.md#pymhm.weighted_estimator.PublishedDarcyIndicator)
- <a id="pymhm.weighted_estimator.PublishedDarcyIndicator.convention"></a>[`PublishedDarcyIndicator.convention`](api/reconstruction.md#pymhm.weighted_estimator.PublishedDarcyIndicator.convention)
- <a id="pymhm.weighted_estimator.PublishedDarcyIndicator.local_squared"></a>[`PublishedDarcyIndicator.local_squared`](api/reconstruction.md#pymhm.weighted_estimator.PublishedDarcyIndicator.local_squared)
- <a id="pymhm.weighted_estimator.recover_dirichlet_potential"></a>[`recover_dirichlet_potential`](api/reconstruction.md#pymhm.weighted_estimator.recover_dirichlet_potential)
- <a id="pymhm.weighted_estimator.estimate_weighted_darcy_error"></a>[`estimate_weighted_darcy_error`](api/reconstruction.md#pymhm.weighted_estimator.estimate_weighted_darcy_error)
- <a id="pymhm.weighted_estimator.estimate_darcy_indicator"></a>[`estimate_darcy_indicator`](api/reconstruction.md#pymhm.weighted_estimator.estimate_darcy_indicator)

<a id="pymhm.estimator3d"></a>**[pymhm.estimator3d](api/reconstruction.md#pymhm.estimator3d)**

- <a id="pymhm.estimator3d.ConformingPotential3D"></a>[`ConformingPotential3D`](api/reconstruction.md#pymhm.estimator3d.ConformingPotential3D)
- <a id="pymhm.estimator3d.ConformingPotential3D.l2_error"></a>[`ConformingPotential3D.l2_error`](api/reconstruction.md#pymhm.estimator3d.ConformingPotential3D.l2_error)
- <a id="pymhm.estimator3d.Darcy3DEstimator"></a>[`Darcy3DEstimator`](api/reconstruction.md#pymhm.estimator3d.Darcy3DEstimator)
- <a id="pymhm.estimator3d.Darcy3DEstimator.local_squared"></a>[`Darcy3DEstimator.local_squared`](api/reconstruction.md#pymhm.estimator3d.Darcy3DEstimator.local_squared)
- <a id="pymhm.estimator3d.Darcy3DEstimator.total"></a>[`Darcy3DEstimator.total`](api/reconstruction.md#pymhm.estimator3d.Darcy3DEstimator.total)
- <a id="pymhm.estimator3d.Darcy3DEstimator.energy_error"></a>[`Darcy3DEstimator.energy_error`](api/reconstruction.md#pymhm.estimator3d.Darcy3DEstimator.energy_error)
- <a id="pymhm.estimator3d.recover_potential_3d"></a>[`recover_potential_3d`](api/reconstruction.md#pymhm.estimator3d.recover_potential_3d)
- <a id="pymhm.estimator3d.estimate_darcy_error_3d"></a>[`estimate_darcy_error_3d`](api/reconstruction.md#pymhm.estimator3d.estimate_darcy_error_3d)

### Adaptive scalar and elasticity indicators

<a id="pymhm.adaptive_darcy"></a>**[pymhm.adaptive_darcy](api/adaptivity.md#pymhm.adaptive_darcy)**

- <a id="pymhm.adaptive_darcy.AdaptiveDarcyResult"></a>[`AdaptiveDarcyResult`](api/adaptivity.md#pymhm.adaptive_darcy.AdaptiveDarcyResult)
- <a id="pymhm.adaptive_darcy.AdaptiveDarcyResult.totals"></a>[`AdaptiveDarcyResult.totals`](api/adaptivity.md#pymhm.adaptive_darcy.AdaptiveDarcyResult.totals)
- <a id="pymhm.adaptive_darcy.mark_dorfler"></a>[`mark_dorfler`](api/adaptivity.md#pymhm.adaptive_darcy.mark_dorfler)
- <a id="pymhm.adaptive_darcy.solve_adaptive_darcy"></a>[`solve_adaptive_darcy`](api/adaptivity.md#pymhm.adaptive_darcy.solve_adaptive_darcy)

<a id="pymhm.adaptive_darcy_balanced"></a>**[pymhm.adaptive_darcy_balanced](api/adaptivity.md#pymhm.adaptive_darcy_balanced)**

- <a id="pymhm.adaptive_darcy_balanced.BalancedDarcyResult"></a>[`BalancedDarcyResult`](api/adaptivity.md#pymhm.adaptive_darcy_balanced.BalancedDarcyResult)
- <a id="pymhm.adaptive_darcy_balanced.solve_balanced_adaptive_darcy"></a>[`solve_balanced_adaptive_darcy`](api/adaptivity.md#pymhm.adaptive_darcy_balanced.solve_balanced_adaptive_darcy)

<a id="pymhm.adaptive_darcy_budget"></a>**[pymhm.adaptive_darcy_budget](api/adaptivity.md#pymhm.adaptive_darcy_budget)**

- <a id="pymhm.adaptive_darcy_budget.DarcyBudgetRefinement"></a>[`DarcyBudgetRefinement`](api/adaptivity.md#pymhm.adaptive_darcy_budget.DarcyBudgetRefinement)
- <a id="pymhm.adaptive_darcy_budget.refine_darcy_budget"></a>[`refine_darcy_budget`](api/adaptivity.md#pymhm.adaptive_darcy_budget.refine_darcy_budget)

<a id="pymhm.adaptive_darcy3d"></a>**[pymhm.adaptive_darcy3d](api/adaptivity.md#pymhm.adaptive_darcy3d)**

- <a id="pymhm.adaptive_darcy3d.AdaptiveDarcy3DResult"></a>[`AdaptiveDarcy3DResult`](api/adaptivity.md#pymhm.adaptive_darcy3d.AdaptiveDarcy3DResult)
- <a id="pymhm.adaptive_darcy3d.AdaptiveDarcy3DResult.totals"></a>[`AdaptiveDarcy3DResult.totals`](api/adaptivity.md#pymhm.adaptive_darcy3d.AdaptiveDarcy3DResult.totals)
- <a id="pymhm.adaptive_darcy3d.solve_adaptive_darcy_3d"></a>[`solve_adaptive_darcy_3d`](api/adaptivity.md#pymhm.adaptive_darcy3d.solve_adaptive_darcy_3d)

<a id="pymhm.darcy_jump_estimator"></a>**[pymhm.darcy_jump_estimator](api/adaptivity.md#pymhm.darcy_jump_estimator)**

- <a id="pymhm.darcy_jump_estimator.DarcyJumpEstimator"></a>[`DarcyJumpEstimator`](api/adaptivity.md#pymhm.darcy_jump_estimator.DarcyJumpEstimator)
- <a id="pymhm.darcy_jump_estimator.DarcyJumpEstimator.total"></a>[`DarcyJumpEstimator.total`](api/adaptivity.md#pymhm.darcy_jump_estimator.DarcyJumpEstimator.total)
- <a id="pymhm.darcy_jump_estimator.estimate_darcy_jumps"></a>[`estimate_darcy_jumps`](api/adaptivity.md#pymhm.darcy_jump_estimator.estimate_darcy_jumps)

<a id="pymhm.darcy_local_error"></a>**[pymhm.darcy_local_error](api/adaptivity.md#pymhm.darcy_local_error)**

- <a id="pymhm.darcy_local_error.LocalDarcyRefinementEstimate"></a>[`LocalDarcyRefinementEstimate`](api/adaptivity.md#pymhm.darcy_local_error.LocalDarcyRefinementEstimate)
- <a id="pymhm.darcy_local_error.LocalDarcyRefinementEstimate.total"></a>[`LocalDarcyRefinementEstimate.total`](api/adaptivity.md#pymhm.darcy_local_error.LocalDarcyRefinementEstimate.total)
- <a id="pymhm.darcy_local_error.estimate_darcy_local_refinement"></a>[`estimate_darcy_local_refinement`](api/adaptivity.md#pymhm.darcy_local_error.estimate_darcy_local_refinement)

<a id="pymhm.metric_adapt"></a>**[pymhm.metric_adapt](api/adaptivity.md#pymhm.metric_adapt)**

- <a id="pymhm.metric_adapt.ResidualMeshSize"></a>[`ResidualMeshSize`](api/adaptivity.md#pymhm.metric_adapt.ResidualMeshSize)
- <a id="pymhm.metric_adapt.residual_mesh_size"></a>[`residual_mesh_size`](api/adaptivity.md#pymhm.metric_adapt.residual_mesh_size)
- <a id="pymhm.metric_adapt.remesh_freefem"></a>[`remesh_freefem`](api/adaptivity.md#pymhm.metric_adapt.remesh_freefem)

<a id="pymhm.elasticity_estimator"></a>**[pymhm.elasticity_estimator](api/adaptivity.md#pymhm.elasticity_estimator)**

- <a id="pymhm.elasticity_estimator.PrimalElasticityIndicator"></a>[`PrimalElasticityIndicator`](api/adaptivity.md#pymhm.elasticity_estimator.PrimalElasticityIndicator)
- <a id="pymhm.elasticity_estimator.PrimalElasticityIndicator.eta"></a>[`PrimalElasticityIndicator.eta`](api/adaptivity.md#pymhm.elasticity_estimator.PrimalElasticityIndicator.eta)
- <a id="pymhm.elasticity_estimator.estimate_primal_elasticity_error"></a>[`estimate_primal_elasticity_error`](api/adaptivity.md#pymhm.elasticity_estimator.estimate_primal_elasticity_error)

### Stokes, Brinkman and Oseen

<a id="pymhm.vector"></a>**[pymhm.vector](api/flow.md#pymhm.vector)**

- <a id="pymhm.vector.VectorSolution"></a>[`VectorSolution`](api/flow.md#pymhm.vector.VectorSolution)
- <a id="pymhm.vector.VectorSolution.l2_error"></a>[`VectorSolution.l2_error`](api/flow.md#pymhm.vector.VectorSolution.l2_error)
- <a id="pymhm.vector.VectorSolution.pressure_l2_error"></a>[`VectorSolution.pressure_l2_error`](api/flow.md#pymhm.vector.VectorSolution.pressure_l2_error)
- <a id="pymhm.vector.VectorSolution.divergence_l2"></a>[`VectorSolution.divergence_l2`](api/flow.md#pymhm.vector.VectorSolution.divergence_l2)
- <a id="pymhm.vector.solve_brinkman"></a>[`solve_brinkman`](api/flow.md#pymhm.vector.solve_brinkman)
- <a id="pymhm.vector.solve_elasticity"></a>[`solve_elasticity`](api/flow.md#pymhm.vector.solve_elasticity)

<a id="pymhm.flow"></a>**[pymhm.flow](api/flow.md#pymhm.flow)**

- <a id="pymhm.flow.solve_flow"></a>[`solve_flow`](api/flow.md#pymhm.flow.solve_flow)

<a id="pymhm.flow_estimator"></a>**[pymhm.flow_estimator](api/flow.md#pymhm.flow_estimator)**

- <a id="pymhm.flow_estimator.FlowEstimator"></a>[`FlowEstimator`](api/flow.md#pymhm.flow_estimator.FlowEstimator)
- <a id="pymhm.flow_estimator.FlowEstimator.eta1"></a>[`FlowEstimator.eta1`](api/flow.md#pymhm.flow_estimator.FlowEstimator.eta1)
- <a id="pymhm.flow_estimator.FlowEstimator.eta2"></a>[`FlowEstimator.eta2`](api/flow.md#pymhm.flow_estimator.FlowEstimator.eta2)
- <a id="pymhm.flow_estimator.FlowEstimator.total"></a>[`FlowEstimator.total`](api/flow.md#pymhm.flow_estimator.FlowEstimator.total)
- <a id="pymhm.flow_estimator.FlowEstimator.mixed_error"></a>[`FlowEstimator.mixed_error`](api/flow.md#pymhm.flow_estimator.FlowEstimator.mixed_error)
- <a id="pymhm.flow_estimator.estimate_flow_error"></a>[`estimate_flow_error`](api/flow.md#pymhm.flow_estimator.estimate_flow_error)

<a id="pymhm.flow_adaptive"></a>**[pymhm.flow_adaptive](api/flow.md#pymhm.flow_adaptive)**

- <a id="pymhm.flow_adaptive.AdaptiveFlowResult"></a>[`AdaptiveFlowResult`](api/flow.md#pymhm.flow_adaptive.AdaptiveFlowResult)
- <a id="pymhm.flow_adaptive.mark_flow_faces"></a>[`mark_flow_faces`](api/flow.md#pymhm.flow_adaptive.mark_flow_faces)
- <a id="pymhm.flow_adaptive.adapt_flow"></a>[`adapt_flow`](api/flow.md#pymhm.flow_adaptive.adapt_flow)

<a id="pymhm.flow_macro_adaptive"></a>**[pymhm.flow_macro_adaptive](api/flow.md#pymhm.flow_macro_adaptive)**

- <a id="pymhm.flow_macro_adaptive.AdaptiveFlowMacroResult"></a>[`AdaptiveFlowMacroResult`](api/flow.md#pymhm.flow_macro_adaptive.AdaptiveFlowMacroResult)
- <a id="pymhm.flow_macro_adaptive.mark_flow_cells"></a>[`mark_flow_cells`](api/flow.md#pymhm.flow_macro_adaptive.mark_flow_cells)
- <a id="pymhm.flow_macro_adaptive.adapt_flow_macros"></a>[`adapt_flow_macros`](api/flow.md#pymhm.flow_macro_adaptive.adapt_flow_macros)

<a id="pymhm.flow3d"></a>**[pymhm.flow3d](api/flow.md#pymhm.flow3d)**

- <a id="pymhm.flow3d.Flow3DSolution"></a>[`Flow3DSolution`](api/flow.md#pymhm.flow3d.Flow3DSolution)
- <a id="pymhm.flow3d.Flow3DSolution.evaluate"></a>[`Flow3DSolution.evaluate`](api/flow.md#pymhm.flow3d.Flow3DSolution.evaluate)
- <a id="pymhm.flow3d.Flow3DSolution.gradient"></a>[`Flow3DSolution.gradient`](api/flow.md#pymhm.flow3d.Flow3DSolution.gradient)
- <a id="pymhm.flow3d.Flow3DSolution.pseudostress"></a>[`Flow3DSolution.pseudostress`](api/flow.md#pymhm.flow3d.Flow3DSolution.pseudostress)
- <a id="pymhm.flow3d.Flow3DSolution.l2_error"></a>[`Flow3DSolution.l2_error`](api/flow.md#pymhm.flow3d.Flow3DSolution.l2_error)
- <a id="pymhm.flow3d.Flow3DSolution.pressure_l2_error"></a>[`Flow3DSolution.pressure_l2_error`](api/flow.md#pymhm.flow3d.Flow3DSolution.pressure_l2_error)
- <a id="pymhm.flow3d.Flow3DSolution.h1_seminorm_error"></a>[`Flow3DSolution.h1_seminorm_error`](api/flow.md#pymhm.flow3d.Flow3DSolution.h1_seminorm_error)
- <a id="pymhm.flow3d.Flow3DSolution.divergence_l2"></a>[`Flow3DSolution.divergence_l2`](api/flow.md#pymhm.flow3d.Flow3DSolution.divergence_l2)
- <a id="pymhm.flow3d.solve_flow_3d"></a>[`solve_flow_3d`](api/flow.md#pymhm.flow3d.solve_flow_3d)

<a id="pymhm.flow3d_forms"></a>**[pymhm.flow3d_forms](api/flow.md#pymhm.flow3d_forms)**

- <a id="pymhm.flow3d_forms.Flow3DOperators"></a>[`Flow3DOperators`](api/flow.md#pymhm.flow3d_forms.Flow3DOperators)
- <a id="pymhm.flow3d_forms.resistance_values_3d"></a>[`resistance_values_3d`](api/flow.md#pymhm.flow3d_forms.resistance_values_3d)
- <a id="pymhm.flow3d_forms.flow_contract_3d"></a>[`flow_contract_3d`](api/flow.md#pymhm.flow3d_forms.flow_contract_3d)
- <a id="pymhm.flow3d_forms.tetra_flow_operators"></a>[`tetra_flow_operators`](api/flow.md#pymhm.flow3d_forms.tetra_flow_operators)

### Elasticity

<a id="pymhm.elasticity"></a>**[pymhm.elasticity](api/elasticity.md#pymhm.elasticity)**

- <a id="pymhm.elasticity.ElasticitySolution"></a>[`ElasticitySolution`](api/elasticity.md#pymhm.elasticity.ElasticitySolution)
- <a id="pymhm.elasticity.ElasticitySolution.gradient"></a>[`ElasticitySolution.gradient`](api/elasticity.md#pymhm.elasticity.ElasticitySolution.gradient)
- <a id="pymhm.elasticity.ElasticitySolution.stress"></a>[`ElasticitySolution.stress`](api/elasticity.md#pymhm.elasticity.ElasticitySolution.stress)
- <a id="pymhm.elasticity.ElasticitySolution.h1_seminorm_error"></a>[`ElasticitySolution.h1_seminorm_error`](api/elasticity.md#pymhm.elasticity.ElasticitySolution.h1_seminorm_error)
- <a id="pymhm.elasticity.ElasticitySolution.stress_l2_error"></a>[`ElasticitySolution.stress_l2_error`](api/elasticity.md#pymhm.elasticity.ElasticitySolution.stress_l2_error)
- <a id="pymhm.elasticity.ElasticitySolution.compressibility_l2"></a>[`ElasticitySolution.compressibility_l2`](api/elasticity.md#pymhm.elasticity.ElasticitySolution.compressibility_l2)
- <a id="pymhm.elasticity.solve_displacement_pressure"></a>[`solve_displacement_pressure`](api/elasticity.md#pymhm.elasticity.solve_displacement_pressure)

<a id="pymhm.elasticity_mixed"></a>**[pymhm.elasticity_mixed](api/elasticity.md#pymhm.elasticity_mixed)**

- <a id="pymhm.elasticity_mixed.MixedElasticitySolution"></a>[`MixedElasticitySolution`](api/elasticity.md#pymhm.elasticity_mixed.MixedElasticitySolution)
- <a id="pymhm.elasticity_mixed.MixedElasticitySolution.displacement_degree"></a>[`MixedElasticitySolution.displacement_degree`](api/elasticity.md#pymhm.elasticity_mixed.MixedElasticitySolution.displacement_degree)
- <a id="pymhm.elasticity_mixed.MixedElasticitySolution.l2_error"></a>[`MixedElasticitySolution.l2_error`](api/elasticity.md#pymhm.elasticity_mixed.MixedElasticitySolution.l2_error)
- <a id="pymhm.elasticity_mixed.MixedElasticitySolution.stress_l2_error"></a>[`MixedElasticitySolution.stress_l2_error`](api/elasticity.md#pymhm.elasticity_mixed.MixedElasticitySolution.stress_l2_error)
- <a id="pymhm.elasticity_mixed.MixedElasticitySolution.divergence_l2_error"></a>[`MixedElasticitySolution.divergence_l2_error`](api/elasticity.md#pymhm.elasticity_mixed.MixedElasticitySolution.divergence_l2_error)
- <a id="pymhm.elasticity_mixed.MixedElasticitySolution.rotation_l2_error"></a>[`MixedElasticitySolution.rotation_l2_error`](api/elasticity.md#pymhm.elasticity_mixed.MixedElasticitySolution.rotation_l2_error)
- <a id="pymhm.elasticity_mixed.MixedElasticitySolution.fine_force_residuals"></a>[`MixedElasticitySolution.fine_force_residuals`](api/elasticity.md#pymhm.elasticity_mixed.MixedElasticitySolution.fine_force_residuals)
- <a id="pymhm.elasticity_mixed.MixedElasticitySolution.weak_symmetry_residuals"></a>[`MixedElasticitySolution.weak_symmetry_residuals`](api/elasticity.md#pymhm.elasticity_mixed.MixedElasticitySolution.weak_symmetry_residuals)
- <a id="pymhm.elasticity_mixed.MixedElasticitySolution.normal_traction_residuals"></a>[`MixedElasticitySolution.normal_traction_residuals`](api/elasticity.md#pymhm.elasticity_mixed.MixedElasticitySolution.normal_traction_residuals)
- <a id="pymhm.elasticity_mixed.MixedElasticitySolution.equilibrium_residuals"></a>[`MixedElasticitySolution.equilibrium_residuals`](api/elasticity.md#pymhm.elasticity_mixed.MixedElasticitySolution.equilibrium_residuals)
- <a id="pymhm.elasticity_mixed.solve_elasticity_mixed"></a>[`solve_elasticity_mixed`](api/elasticity.md#pymhm.elasticity_mixed.solve_elasticity_mixed)

<a id="pymhm.elasticity_primal"></a>**[pymhm.elasticity_primal](api/elasticity.md#pymhm.elasticity_primal)**

- <a id="pymhm.elasticity_primal.PrimalElasticitySolution"></a>[`PrimalElasticitySolution`](api/elasticity.md#pymhm.elasticity_primal.PrimalElasticitySolution)
- <a id="pymhm.elasticity_primal.PrimalElasticitySolution.gradient"></a>[`PrimalElasticitySolution.gradient`](api/elasticity.md#pymhm.elasticity_primal.PrimalElasticitySolution.gradient)
- <a id="pymhm.elasticity_primal.PrimalElasticitySolution.stress"></a>[`PrimalElasticitySolution.stress`](api/elasticity.md#pymhm.elasticity_primal.PrimalElasticitySolution.stress)
- <a id="pymhm.elasticity_primal.PrimalElasticitySolution.stress_l2_error"></a>[`PrimalElasticitySolution.stress_l2_error`](api/elasticity.md#pymhm.elasticity_primal.PrimalElasticitySolution.stress_l2_error)
- <a id="pymhm.elasticity_primal.constitutive_values"></a>[`constitutive_values`](api/elasticity.md#pymhm.elasticity_primal.constitutive_values)
- <a id="pymhm.elasticity_primal.solve_primal_elasticity"></a>[`solve_primal_elasticity`](api/elasticity.md#pymhm.elasticity_primal.solve_primal_elasticity)

<a id="pymhm.elasticity_tensor_rt"></a>**[pymhm.elasticity_tensor_rt](api/elasticity.md#pymhm.elasticity_tensor_rt)**

- <a id="pymhm.elasticity_tensor_rt.TensorRTElasticitySolution"></a>[`TensorRTElasticitySolution`](api/elasticity.md#pymhm.elasticity_tensor_rt.TensorRTElasticitySolution)
- <a id="pymhm.elasticity_tensor_rt.TensorRTElasticitySolution.evaluate"></a>[`TensorRTElasticitySolution.evaluate`](api/elasticity.md#pymhm.elasticity_tensor_rt.TensorRTElasticitySolution.evaluate)
- <a id="pymhm.elasticity_tensor_rt.TensorRTElasticitySolution.errors"></a>[`TensorRTElasticitySolution.errors`](api/elasticity.md#pymhm.elasticity_tensor_rt.TensorRTElasticitySolution.errors)
- <a id="pymhm.elasticity_tensor_rt.TensorRTElasticitySolution.fine_force_residuals"></a>[`TensorRTElasticitySolution.fine_force_residuals`](api/elasticity.md#pymhm.elasticity_tensor_rt.TensorRTElasticitySolution.fine_force_residuals)
- <a id="pymhm.elasticity_tensor_rt.TensorRTElasticitySolution.weak_symmetry_residuals"></a>[`TensorRTElasticitySolution.weak_symmetry_residuals`](api/elasticity.md#pymhm.elasticity_tensor_rt.TensorRTElasticitySolution.weak_symmetry_residuals)
- <a id="pymhm.elasticity_tensor_rt.TensorRTElasticitySolution.normal_traction_residuals"></a>[`TensorRTElasticitySolution.normal_traction_residuals`](api/elasticity.md#pymhm.elasticity_tensor_rt.TensorRTElasticitySolution.normal_traction_residuals)
- <a id="pymhm.elasticity_tensor_rt.TensorRTElasticitySolution.equilibrium_residuals"></a>[`TensorRTElasticitySolution.equilibrium_residuals`](api/elasticity.md#pymhm.elasticity_tensor_rt.TensorRTElasticitySolution.equilibrium_residuals)
- <a id="pymhm.elasticity_tensor_rt.solve_elasticity_tensor_rt"></a>[`solve_elasticity_tensor_rt`](api/elasticity.md#pymhm.elasticity_tensor_rt.solve_elasticity_tensor_rt)

<a id="pymhm.elasticity3d"></a>**[pymhm.elasticity3d](api/elasticity.md#pymhm.elasticity3d)**

- <a id="pymhm.elasticity3d.Elasticity3DSolution"></a>[`Elasticity3DSolution`](api/elasticity.md#pymhm.elasticity3d.Elasticity3DSolution)
- <a id="pymhm.elasticity3d.Elasticity3DSolution.evaluate"></a>[`Elasticity3DSolution.evaluate`](api/elasticity.md#pymhm.elasticity3d.Elasticity3DSolution.evaluate)
- <a id="pymhm.elasticity3d.Elasticity3DSolution.errors"></a>[`Elasticity3DSolution.errors`](api/elasticity.md#pymhm.elasticity3d.Elasticity3DSolution.errors)
- <a id="pymhm.elasticity3d.Elasticity3DSolution.equilibrium_residuals"></a>[`Elasticity3DSolution.equilibrium_residuals`](api/elasticity.md#pymhm.elasticity3d.Elasticity3DSolution.equilibrium_residuals)
- <a id="pymhm.elasticity3d.constitutive_values_3d"></a>[`constitutive_values_3d`](api/elasticity.md#pymhm.elasticity3d.constitutive_values_3d)
- <a id="pymhm.elasticity3d.rigid_modes_3d"></a>[`rigid_modes_3d`](api/elasticity.md#pymhm.elasticity3d.rigid_modes_3d)
- <a id="pymhm.elasticity3d.solve_elasticity_3d"></a>[`solve_elasticity_3d`](api/elasticity.md#pymhm.elasticity3d.solve_elasticity_3d)

<a id="pymhm.elasticity_compliance"></a>**[pymhm.elasticity_compliance](api/elasticity.md#pymhm.elasticity_compliance)**

- <a id="pymhm.elasticity_compliance.stress_compliance_values"></a>[`stress_compliance_values`](api/elasticity.md#pymhm.elasticity_compliance.stress_compliance_values)
- <a id="pymhm.elasticity_compliance.compliance_products"></a>[`compliance_products`](api/elasticity.md#pymhm.elasticity_compliance.compliance_products)

<a id="pymhm.gals3d"></a>**[pymhm.gals3d](api/elasticity.md#pymhm.gals3d)**

- <a id="pymhm.gals3d.GaLS3DSolution"></a>[`GaLS3DSolution`](api/elasticity.md#pymhm.gals3d.GaLS3DSolution)
- <a id="pymhm.gals3d.GaLS3DSolution.evaluate"></a>[`GaLS3DSolution.evaluate`](api/elasticity.md#pymhm.gals3d.GaLS3DSolution.evaluate)
- <a id="pymhm.gals3d.GaLS3DSolution.gradient"></a>[`GaLS3DSolution.gradient`](api/elasticity.md#pymhm.gals3d.GaLS3DSolution.gradient)
- <a id="pymhm.gals3d.GaLS3DSolution.stress"></a>[`GaLS3DSolution.stress`](api/elasticity.md#pymhm.gals3d.GaLS3DSolution.stress)
- <a id="pymhm.gals3d.GaLS3DSolution.l2_error"></a>[`GaLS3DSolution.l2_error`](api/elasticity.md#pymhm.gals3d.GaLS3DSolution.l2_error)
- <a id="pymhm.gals3d.GaLS3DSolution.pressure_l2_error"></a>[`GaLS3DSolution.pressure_l2_error`](api/elasticity.md#pymhm.gals3d.GaLS3DSolution.pressure_l2_error)
- <a id="pymhm.gals3d.GaLS3DSolution.h1_seminorm_error"></a>[`GaLS3DSolution.h1_seminorm_error`](api/elasticity.md#pymhm.gals3d.GaLS3DSolution.h1_seminorm_error)
- <a id="pymhm.gals3d.GaLS3DSolution.stress_l2_error"></a>[`GaLS3DSolution.stress_l2_error`](api/elasticity.md#pymhm.gals3d.GaLS3DSolution.stress_l2_error)
- <a id="pymhm.gals3d.GaLS3DSolution.compressibility_l2"></a>[`GaLS3DSolution.compressibility_l2`](api/elasticity.md#pymhm.gals3d.GaLS3DSolution.compressibility_l2)
- <a id="pymhm.gals3d.solve_elasticity_gals_3d"></a>[`solve_elasticity_gals_3d`](api/elasticity.md#pymhm.gals3d.solve_elasticity_gals_3d)

<a id="pymhm.gals3d_forms"></a>**[pymhm.gals3d_forms](api/elasticity.md#pymhm.gals3d_forms)**

- <a id="pymhm.gals3d_forms.ElasticityPressure3DOperators"></a>[`ElasticityPressure3DOperators`](api/elasticity.md#pymhm.gals3d_forms.ElasticityPressure3DOperators)
- <a id="pymhm.gals3d_forms.elasticity_contract_3d"></a>[`elasticity_contract_3d`](api/elasticity.md#pymhm.gals3d_forms.elasticity_contract_3d)
- <a id="pymhm.gals3d_forms.tetra_elasticity_pressure_operators"></a>[`tetra_elasticity_pressure_operators`](api/elasticity.md#pymhm.gals3d_forms.tetra_elasticity_pressure_operators)

### Transport, reaction and diffusion

<a id="pymhm.transport"></a>**[pymhm.transport](api/transport.md#pymhm.transport)**

- <a id="pymhm.transport.ScalarSolution"></a>[`ScalarSolution`](api/transport.md#pymhm.transport.ScalarSolution)
- <a id="pymhm.transport.ScalarSolution.l2_error"></a>[`ScalarSolution.l2_error`](api/transport.md#pymhm.transport.ScalarSolution.l2_error)
- <a id="pymhm.transport.solve_transport"></a>[`solve_transport`](api/transport.md#pymhm.transport.solve_transport)
- <a id="pymhm.transport.solve_heat"></a>[`solve_heat`](api/transport.md#pymhm.transport.solve_heat)

<a id="pymhm.rad"></a>**[pymhm.rad](api/transport.md#pymhm.rad)**

- <a id="pymhm.rad.solve_rad"></a>[`solve_rad`](api/transport.md#pymhm.rad.solve_rad)

<a id="pymhm.rad3d"></a>**[pymhm.rad3d](api/transport.md#pymhm.rad3d)**

- <a id="pymhm.rad3d.RAD3DSolution"></a>[`RAD3DSolution`](api/transport.md#pymhm.rad3d.RAD3DSolution)
- <a id="pymhm.rad3d.RAD3DSolution.evaluate"></a>[`RAD3DSolution.evaluate`](api/transport.md#pymhm.rad3d.RAD3DSolution.evaluate)
- <a id="pymhm.rad3d.RAD3DSolution.l2_error"></a>[`RAD3DSolution.l2_error`](api/transport.md#pymhm.rad3d.RAD3DSolution.l2_error)
- <a id="pymhm.rad3d.RAD3DSolution.h1_seminorm_error"></a>[`RAD3DSolution.h1_seminorm_error`](api/transport.md#pymhm.rad3d.RAD3DSolution.h1_seminorm_error)
- <a id="pymhm.rad3d.RAD3DSolution.flux_l2_error"></a>[`RAD3DSolution.flux_l2_error`](api/transport.md#pymhm.rad3d.RAD3DSolution.flux_l2_error)
- <a id="pymhm.rad3d.vector_values_3d"></a>[`vector_values_3d`](api/transport.md#pymhm.rad3d.vector_values_3d)
- <a id="pymhm.rad3d.tetra_rad_operators"></a>[`tetra_rad_operators`](api/transport.md#pymhm.rad3d.tetra_rad_operators)
- <a id="pymhm.rad3d.solve_rad_3d"></a>[`solve_rad_3d`](api/transport.md#pymhm.rad3d.solve_rad_3d)
- <a id="pymhm.rad3d.solve_rad_3d_conforming"></a>[`solve_rad_3d_conforming`](api/transport.md#pymhm.rad3d.solve_rad_3d_conforming)

<a id="pymhm.scalar_adaptive"></a>**[pymhm.scalar_adaptive](api/transport.md#pymhm.scalar_adaptive)**

- <a id="pymhm.scalar_adaptive.TransportBounds"></a>[`TransportBounds`](api/transport.md#pymhm.scalar_adaptive.TransportBounds)
- <a id="pymhm.scalar_adaptive.TransportBounds.__post_init__"></a>[`TransportBounds.__post_init__`](api/transport.md#pymhm.scalar_adaptive.TransportBounds.__post_init__)
- <a id="pymhm.scalar_adaptive.TransportBounds.scale"></a>[`TransportBounds.scale`](api/transport.md#pymhm.scalar_adaptive.TransportBounds.scale)
- <a id="pymhm.scalar_adaptive.FaceIndicators"></a>[`FaceIndicators`](api/transport.md#pymhm.scalar_adaptive.FaceIndicators)
- <a id="pymhm.scalar_adaptive.FaceIndicators.total"></a>[`FaceIndicators.total`](api/transport.md#pymhm.scalar_adaptive.FaceIndicators.total)
- <a id="pymhm.scalar_adaptive.FaceIndicators.mark"></a>[`FaceIndicators.mark`](api/transport.md#pymhm.scalar_adaptive.FaceIndicators.mark)
- <a id="pymhm.scalar_adaptive.AdaptiveTransportResult"></a>[`AdaptiveTransportResult`](api/transport.md#pymhm.scalar_adaptive.AdaptiveTransportResult)
- <a id="pymhm.scalar_adaptive.estimate_transport_faces"></a>[`estimate_transport_faces`](api/transport.md#pymhm.scalar_adaptive.estimate_transport_faces)
- <a id="pymhm.scalar_adaptive.refine_skeleton_faces"></a>[`refine_skeleton_faces`](api/transport.md#pymhm.scalar_adaptive.refine_skeleton_faces)
- <a id="pymhm.scalar_adaptive.solve_adaptive_transport"></a>[`solve_adaptive_transport`](api/transport.md#pymhm.scalar_adaptive.solve_adaptive_transport)

<a id="pymhm.scalar_transient"></a>**[pymhm.scalar_transient](api/transport.md#pymhm.scalar_transient)**

- <a id="pymhm.scalar_transient.MacroCoefficient"></a>[`MacroCoefficient`](api/transport.md#pymhm.scalar_transient.MacroCoefficient)
- <a id="pymhm.scalar_transient.MacroCoefficient.for_cell"></a>[`MacroCoefficient.for_cell`](api/transport.md#pymhm.scalar_transient.MacroCoefficient.for_cell)
- <a id="pymhm.scalar_transient.TransientTransportResult"></a>[`TransientTransportResult`](api/transport.md#pymhm.scalar_transient.TransientTransportResult)
- <a id="pymhm.scalar_transient.TransientTransportResult.total_mass"></a>[`TransientTransportResult.total_mass`](api/transport.md#pymhm.scalar_transient.TransientTransportResult.total_mass)
- <a id="pymhm.scalar_transient.solve_transient_transport"></a>[`solve_transient_transport`](api/transport.md#pymhm.scalar_transient.solve_transient_transport)

<a id="pymhm.darcy_transport"></a>**[pymhm.darcy_transport](api/transport.md#pymhm.darcy_transport)**

- <a id="pymhm.darcy_transport.RT0DarcyVelocity"></a>[`RT0DarcyVelocity`](api/transport.md#pymhm.darcy_transport.RT0DarcyVelocity)
- <a id="pymhm.darcy_transport.RT0DarcyVelocity.__init__"></a>[`RT0DarcyVelocity.__init__`](api/transport.md#pymhm.darcy_transport.RT0DarcyVelocity.__init__)
- <a id="pymhm.darcy_transport.RT0DarcyVelocity.locate"></a>[`RT0DarcyVelocity.locate`](api/transport.md#pymhm.darcy_transport.RT0DarcyVelocity.locate)
- <a id="pymhm.darcy_transport.RT0DarcyVelocity.__call__"></a>[`RT0DarcyVelocity.__call__`](api/transport.md#pymhm.darcy_transport.RT0DarcyVelocity.__call__)
- <a id="pymhm.darcy_transport.RT0DarcyVelocity.divergence"></a>[`RT0DarcyVelocity.divergence`](api/transport.md#pymhm.darcy_transport.RT0DarcyVelocity.divergence)
- <a id="pymhm.darcy_transport.HydrodynamicDispersion"></a>[`HydrodynamicDispersion`](api/transport.md#pymhm.darcy_transport.HydrodynamicDispersion)
- <a id="pymhm.darcy_transport.HydrodynamicDispersion.__post_init__"></a>[`HydrodynamicDispersion.__post_init__`](api/transport.md#pymhm.darcy_transport.HydrodynamicDispersion.__post_init__)
- <a id="pymhm.darcy_transport.HydrodynamicDispersion.__call__"></a>[`HydrodynamicDispersion.__call__`](api/transport.md#pymhm.darcy_transport.HydrodynamicDispersion.__call__)
- <a id="pymhm.darcy_transport.HydrodynamicDispersion.divergence"></a>[`HydrodynamicDispersion.divergence`](api/transport.md#pymhm.darcy_transport.HydrodynamicDispersion.divergence)
- <a id="pymhm.darcy_transport.solve_darcy_transport"></a>[`solve_darcy_transport`](api/transport.md#pymhm.darcy_transport.solve_darcy_transport)

<a id="pymhm.unusual"></a>**[pymhm.unusual](api/transport.md#pymhm.unusual)**

- <a id="pymhm.unusual.UnusualParameters"></a>[`UnusualParameters`](api/transport.md#pymhm.unusual.UnusualParameters)
- <a id="pymhm.unusual.UnusualParameters.__post_init__"></a>[`UnusualParameters.__post_init__`](api/transport.md#pymhm.unusual.UnusualParameters.__post_init__)
- <a id="pymhm.unusual.unusual_scale"></a>[`unusual_scale`](api/transport.md#pymhm.unusual.unusual_scale)

<a id="pymhm.polyhedral_rad"></a>**[pymhm.polyhedral_rad](api/transport.md#pymhm.polyhedral_rad)**

- <a id="pymhm.polyhedral_rad.PolygonalSkeleton3D"></a>[`PolygonalSkeleton3D`](api/transport.md#pymhm.polyhedral_rad.PolygonalSkeleton3D)
- <a id="pymhm.polyhedral_rad.PolygonalSkeleton3D.__init__"></a>[`PolygonalSkeleton3D.__init__`](api/transport.md#pymhm.polyhedral_rad.PolygonalSkeleton3D.__init__)
- <a id="pymhm.polyhedral_rad.PolygonalSkeleton3D.dofs"></a>[`PolygonalSkeleton3D.dofs`](api/transport.md#pymhm.polyhedral_rad.PolygonalSkeleton3D.dofs)
- <a id="pymhm.polyhedral_rad.PolygonalSkeleton3D.cell_dofs"></a>[`PolygonalSkeleton3D.cell_dofs`](api/transport.md#pymhm.polyhedral_rad.PolygonalSkeleton3D.cell_dofs)
- <a id="pymhm.polyhedral_rad.PolygonalSkeleton3D.basis"></a>[`PolygonalSkeleton3D.basis`](api/transport.md#pymhm.polyhedral_rad.PolygonalSkeleton3D.basis)
- <a id="pymhm.polyhedral_rad.PolyhedralRADSolution"></a>[`PolyhedralRADSolution`](api/transport.md#pymhm.polyhedral_rad.PolyhedralRADSolution)
- <a id="pymhm.polyhedral_rad.PolyhedralRADSolution.local_meshes"></a>[`PolyhedralRADSolution.local_meshes`](api/transport.md#pymhm.polyhedral_rad.PolyhedralRADSolution.local_meshes)
- <a id="pymhm.polyhedral_rad.PolyhedralRADSolution.values"></a>[`PolyhedralRADSolution.values`](api/transport.md#pymhm.polyhedral_rad.PolyhedralRADSolution.values)
- <a id="pymhm.polyhedral_rad.PolyhedralRADSolution.degree"></a>[`PolyhedralRADSolution.degree`](api/transport.md#pymhm.polyhedral_rad.PolyhedralRADSolution.degree)
- <a id="pymhm.polyhedral_rad.PolyhedralRADSolution.evaluate"></a>[`PolyhedralRADSolution.evaluate`](api/transport.md#pymhm.polyhedral_rad.PolyhedralRADSolution.evaluate)
- <a id="pymhm.polyhedral_rad.PolyhedralRADSolution.l2_error"></a>[`PolyhedralRADSolution.l2_error`](api/transport.md#pymhm.polyhedral_rad.PolyhedralRADSolution.l2_error)
- <a id="pymhm.polyhedral_rad.PolyhedralRADSolution.h1_seminorm_error"></a>[`PolyhedralRADSolution.h1_seminorm_error`](api/transport.md#pymhm.polyhedral_rad.PolyhedralRADSolution.h1_seminorm_error)
- <a id="pymhm.polyhedral_rad.PolyhedralRADSolution.flux_l2_error"></a>[`PolyhedralRADSolution.flux_l2_error`](api/transport.md#pymhm.polyhedral_rad.PolyhedralRADSolution.flux_l2_error)
- <a id="pymhm.polyhedral_rad.polygonal_trace_coupling"></a>[`polygonal_trace_coupling`](api/transport.md#pymhm.polyhedral_rad.polygonal_trace_coupling)
- <a id="pymhm.polyhedral_rad.solve_polyhedral_rad"></a>[`solve_polyhedral_rad`](api/transport.md#pymhm.polyhedral_rad.solve_polyhedral_rad)

### Finite element bases

<a id="pymhm.lagrange"></a>**[pymhm.lagrange](api/elements.md#pymhm.lagrange)**

- <a id="pymhm.lagrange.multiindices"></a>[`multiindices`](api/elements.md#pymhm.lagrange.multiindices)
- <a id="pymhm.lagrange.reference_basis"></a>[`reference_basis`](api/elements.md#pymhm.lagrange.reference_basis)
- <a id="pymhm.lagrange.nodal_space"></a>[`nodal_space`](api/elements.md#pymhm.lagrange.nodal_space)
- <a id="pymhm.lagrange.tabulate"></a>[`tabulate`](api/elements.md#pymhm.lagrange.tabulate)
- <a id="pymhm.lagrange.element_tabulate"></a>[`element_tabulate`](api/elements.md#pymhm.lagrange.element_tabulate)
- <a id="pymhm.lagrange.trace_coupling"></a>[`trace_coupling`](api/elements.md#pymhm.lagrange.trace_coupling)
- <a id="pymhm.lagrange.scalar_operators"></a>[`scalar_operators`](api/elements.md#pymhm.lagrange.scalar_operators)

<a id="pymhm.bdm"></a>**[pymhm.bdm](api/elements.md#pymhm.bdm)**

- <a id="pymhm.bdm.bdm2_dofs"></a>[`bdm2_dofs`](api/elements.md#pymhm.bdm.bdm2_dofs)
- <a id="pymhm.bdm.bdm2_basis"></a>[`bdm2_basis`](api/elements.md#pymhm.bdm.bdm2_basis)
- <a id="pymhm.bdm.bdm2_evaluate"></a>[`bdm2_evaluate`](api/elements.md#pymhm.bdm.bdm2_evaluate)
- <a id="pymhm.bdm.validate_bdm2_trace"></a>[`validate_bdm2_trace`](api/elements.md#pymhm.bdm.validate_bdm2_trace)
- <a id="pymhm.bdm.bdm2_trace_map"></a>[`bdm2_trace_map`](api/elements.md#pymhm.bdm.bdm2_trace_map)

<a id="pymhm.bdm_family"></a>**[pymhm.bdm_family](api/elements.md#pymhm.bdm_family)**

- <a id="pymhm.bdm_family.BDMFamily"></a>[`BDMFamily`](api/elements.md#pymhm.bdm_family.BDMFamily)
- <a id="pymhm.bdm_family.BDMFamily.polynomial_degree"></a>[`BDMFamily.polynomial_degree`](api/elements.md#pymhm.bdm_family.BDMFamily.polynomial_degree)
- <a id="pymhm.bdm_family.BDMFamily.interior_size"></a>[`BDMFamily.interior_size`](api/elements.md#pymhm.bdm_family.BDMFamily.interior_size)
- <a id="pymhm.bdm_family.BDMFamily.local_size"></a>[`BDMFamily.local_size`](api/elements.md#pymhm.bdm_family.BDMFamily.local_size)
- <a id="pymhm.bdm_family.BDMFamily.__post_init__"></a>[`BDMFamily.__post_init__`](api/elements.md#pymhm.bdm_family.BDMFamily.__post_init__)
- <a id="pymhm.bdm_family.BDMFamily.size"></a>[`BDMFamily.size`](api/elements.md#pymhm.bdm_family.BDMFamily.size)
- <a id="pymhm.bdm_family.BDMFamily.dofs"></a>[`BDMFamily.dofs`](api/elements.md#pymhm.bdm_family.BDMFamily.dofs)
- <a id="pymhm.bdm_family.BDMFamily.basis"></a>[`BDMFamily.basis`](api/elements.md#pymhm.bdm_family.BDMFamily.basis)
- <a id="pymhm.bdm_family.BDMFamily.evaluate"></a>[`BDMFamily.evaluate`](api/elements.md#pymhm.bdm_family.BDMFamily.evaluate)
- <a id="pymhm.bdm_family.BDMFamily.validate_trace"></a>[`BDMFamily.validate_trace`](api/elements.md#pymhm.bdm_family.BDMFamily.validate_trace)
- <a id="pymhm.bdm_family.BDMFamily.trace_map"></a>[`BDMFamily.trace_map`](api/elements.md#pymhm.bdm_family.BDMFamily.trace_map)

<a id="pymhm.rt"></a>**[pymhm.rt](api/elements.md#pymhm.rt)**

- <a id="pymhm.rt.rt_degree"></a>[`rt_degree`](api/elements.md#pymhm.rt.rt_degree)
- <a id="pymhm.rt.rt_interior_tests"></a>[`rt_interior_tests`](api/elements.md#pymhm.rt.rt_interior_tests)
- <a id="pymhm.rt.rt_dofs"></a>[`rt_dofs`](api/elements.md#pymhm.rt.rt_dofs)
- <a id="pymhm.rt.rt_basis"></a>[`rt_basis`](api/elements.md#pymhm.rt.rt_basis)
- <a id="pymhm.rt.rt_evaluate"></a>[`rt_evaluate`](api/elements.md#pymhm.rt.rt_evaluate)
- <a id="pymhm.rt.rt_evaluate_points"></a>[`rt_evaluate_points`](api/elements.md#pymhm.rt.rt_evaluate_points)
- <a id="pymhm.rt.rt_interpolate"></a>[`rt_interpolate`](api/elements.md#pymhm.rt.rt_interpolate)

### Helmholtz and Maxwell

<a id="pymhm.helmholtz"></a>**[pymhm.helmholtz](api/waves.md#pymhm.helmholtz)**

- <a id="pymhm.helmholtz.HelmholtzSolution"></a>[`HelmholtzSolution`](api/waves.md#pymhm.helmholtz.HelmholtzSolution)
- <a id="pymhm.helmholtz.HelmholtzSolution.sample"></a>[`HelmholtzSolution.sample`](api/waves.md#pymhm.helmholtz.HelmholtzSolution.sample)
- <a id="pymhm.helmholtz.HelmholtzSolution.l2_error"></a>[`HelmholtzSolution.l2_error`](api/waves.md#pymhm.helmholtz.HelmholtzSolution.l2_error)
- <a id="pymhm.helmholtz.HelmholtzSolution.gradient_l2_error"></a>[`HelmholtzSolution.gradient_l2_error`](api/waves.md#pymhm.helmholtz.HelmholtzSolution.gradient_l2_error)
- <a id="pymhm.helmholtz.HelmholtzSolution.conservation_residuals"></a>[`HelmholtzSolution.conservation_residuals`](api/waves.md#pymhm.helmholtz.HelmholtzSolution.conservation_residuals)
- <a id="pymhm.helmholtz.solve_helmholtz"></a>[`solve_helmholtz`](api/waves.md#pymhm.helmholtz.solve_helmholtz)

<a id="pymhm.helmholtz_spaces"></a>**[pymhm.helmholtz_spaces](api/waves.md#pymhm.helmholtz_spaces)**

- <a id="pymhm.helmholtz_spaces.PolynomialNeumannTrace"></a>[`PolynomialNeumannTrace`](api/waves.md#pymhm.helmholtz_spaces.PolynomialNeumannTrace)
- <a id="pymhm.helmholtz_spaces.PolynomialNeumannTrace.__post_init__"></a>[`PolynomialNeumannTrace.__post_init__`](api/waves.md#pymhm.helmholtz_spaces.PolynomialNeumannTrace.__post_init__)
- <a id="pymhm.helmholtz_spaces.PolynomialNeumannTrace.evaluate"></a>[`PolynomialNeumannTrace.evaluate`](api/waves.md#pymhm.helmholtz_spaces.PolynomialNeumannTrace.evaluate)
- <a id="pymhm.helmholtz_spaces.PolynomialNeumannTrace.coefficients_on"></a>[`PolynomialNeumannTrace.coefficients_on`](api/waves.md#pymhm.helmholtz_spaces.PolynomialNeumannTrace.coefficients_on)
- <a id="pymhm.helmholtz_spaces.OscillatoryFaceSpace"></a>[`OscillatoryFaceSpace`](api/waves.md#pymhm.helmholtz_spaces.OscillatoryFaceSpace)
- <a id="pymhm.helmholtz_spaces.OscillatoryFaceSpace.__post_init__"></a>[`OscillatoryFaceSpace.__post_init__`](api/waves.md#pymhm.helmholtz_spaces.OscillatoryFaceSpace.__post_init__)
- <a id="pymhm.helmholtz_spaces.OscillatoryFaceSpace.evaluate"></a>[`OscillatoryFaceSpace.evaluate`](api/waves.md#pymhm.helmholtz_spaces.OscillatoryFaceSpace.evaluate)
- <a id="pymhm.helmholtz_spaces.OscillatoryFaceSpace.constant_coefficients"></a>[`OscillatoryFaceSpace.constant_coefficients`](api/waves.md#pymhm.helmholtz_spaces.OscillatoryFaceSpace.constant_coefficients)
- <a id="pymhm.helmholtz_spaces.helmholtz_skeleton"></a>[`helmholtz_skeleton`](api/waves.md#pymhm.helmholtz_spaces.helmholtz_skeleton)

<a id="pymhm.maxwell"></a>**[pymhm.maxwell](api/waves.md#pymhm.maxwell)**

- <a id="pymhm.maxwell.MaxwellSolution"></a>[`MaxwellSolution`](api/waves.md#pymhm.maxwell.MaxwellSolution)
- <a id="pymhm.maxwell.MaxwellSolution.sample"></a>[`MaxwellSolution.sample`](api/waves.md#pymhm.maxwell.MaxwellSolution.sample)
- <a id="pymhm.maxwell.MaxwellSolution.l2_errors"></a>[`MaxwellSolution.l2_errors`](api/waves.md#pymhm.maxwell.MaxwellSolution.l2_errors)
- <a id="pymhm.maxwell.MaxwellStepper"></a>[`MaxwellStepper`](api/waves.md#pymhm.maxwell.MaxwellStepper)
- <a id="pymhm.maxwell.MaxwellStepper.__init__"></a>[`MaxwellStepper.__init__`](api/waves.md#pymhm.maxwell.MaxwellStepper.__init__)
- <a id="pymhm.maxwell.MaxwellStepper.initialize"></a>[`MaxwellStepper.initialize`](api/waves.md#pymhm.maxwell.MaxwellStepper.initialize)
- <a id="pymhm.maxwell.MaxwellStepper.modified_energy"></a>[`MaxwellStepper.modified_energy`](api/waves.md#pymhm.maxwell.MaxwellStepper.modified_energy)
- <a id="pymhm.maxwell.MaxwellStepper.advance"></a>[`MaxwellStepper.advance`](api/waves.md#pymhm.maxwell.MaxwellStepper.advance)
- <a id="pymhm.maxwell.MaxwellStepper.solution"></a>[`MaxwellStepper.solution`](api/waves.md#pymhm.maxwell.MaxwellStepper.solution)
- <a id="pymhm.maxwell.MaxwellStepper.close"></a>[`MaxwellStepper.close`](api/waves.md#pymhm.maxwell.MaxwellStepper.close)
- <a id="pymhm.maxwell.MaxwellStepper.__enter__"></a>[`MaxwellStepper.__enter__`](api/waves.md#pymhm.maxwell.MaxwellStepper.__enter__)
- <a id="pymhm.maxwell.MaxwellStepper.__exit__"></a>[`MaxwellStepper.__exit__`](api/waves.md#pymhm.maxwell.MaxwellStepper.__exit__)
- <a id="pymhm.maxwell.field_values"></a>[`field_values`](api/waves.md#pymhm.maxwell.field_values)
- <a id="pymhm.maxwell.volume_load"></a>[`volume_load`](api/waves.md#pymhm.maxwell.volume_load)
- <a id="pymhm.maxwell.solve_maxwell"></a>[`solve_maxwell`](api/waves.md#pymhm.maxwell.solve_maxwell)

<a id="pymhm.maxwell_dg.MaxwellSkeleton"></a>**[pymhm.maxwell_dg.MaxwellSkeleton](api/waves.md#pymhm.maxwell_dg.MaxwellSkeleton)**

- <a id="pymhm.maxwell_dg.MaxwellSkeleton.__init__"></a>[`__init__`](api/waves.md#pymhm.maxwell_dg.MaxwellSkeleton.__init__)
- <a id="pymhm.maxwell_dg.MaxwellSkeleton.dofs"></a>[`dofs`](api/waves.md#pymhm.maxwell_dg.MaxwellSkeleton.dofs)
- <a id="pymhm.maxwell_dg.MaxwellSkeleton.cell_dofs"></a>[`cell_dofs`](api/waves.md#pymhm.maxwell_dg.MaxwellSkeleton.cell_dofs)

### Solvers and execution backends

<a id="pymhm.fenics"></a>**[pymhm.fenics](api/backends.md#pymhm.fenics)**

- <a id="pymhm.fenics.from_ufl"></a>[`from_ufl`](api/backends.md#pymhm.fenics.from_ufl)
- <a id="pymhm.fenics.primal_darcy_forms"></a>[`primal_darcy_forms`](api/backends.md#pymhm.fenics.primal_darcy_forms)
- <a id="pymhm.fenics.mixed_darcy_forms"></a>[`mixed_darcy_forms`](api/backends.md#pymhm.fenics.mixed_darcy_forms)
- <a id="pymhm.fenics.brinkman_forms"></a>[`brinkman_forms`](api/backends.md#pymhm.fenics.brinkman_forms)
- <a id="pymhm.fenics.elasticity_forms"></a>[`elasticity_forms`](api/backends.md#pymhm.fenics.elasticity_forms)
- <a id="pymhm.fenics.usfem_brinkman_forms"></a>[`usfem_brinkman_forms`](api/backends.md#pymhm.fenics.usfem_brinkman_forms)

<a id="pymhm.solvers"></a>**[pymhm.solvers](api/backends.md#pymhm.solvers)**

- <a id="pymhm.solvers.LinearSolveError"></a>[`LinearSolveError`](api/backends.md#pymhm.solvers.LinearSolveError)
- <a id="pymhm.solvers.SolverUnavailableError"></a>[`SolverUnavailableError`](api/backends.md#pymhm.solvers.SolverUnavailableError)
- <a id="pymhm.solvers.LinearFactorization"></a>[`LinearFactorization`](api/backends.md#pymhm.solvers.LinearFactorization)
- <a id="pymhm.solvers.LinearFactorization.matches"></a>[`LinearFactorization.matches`](api/backends.md#pymhm.solvers.LinearFactorization.matches)
- <a id="pymhm.solvers.LinearFactorization.solve"></a>[`LinearFactorization.solve`](api/backends.md#pymhm.solvers.LinearFactorization.solve)
- <a id="pymhm.solvers.LinearFactorization.close"></a>[`LinearFactorization.close`](api/backends.md#pymhm.solvers.LinearFactorization.close)
- <a id="pymhm.solvers.LinearFactorization.__enter__"></a>[`LinearFactorization.__enter__`](api/backends.md#pymhm.solvers.LinearFactorization.__enter__)
- <a id="pymhm.solvers.LinearFactorization.__exit__"></a>[`LinearFactorization.__exit__`](api/backends.md#pymhm.solvers.LinearFactorization.__exit__)
- <a id="pymhm.solvers.validate_invertible"></a>[`validate_invertible`](api/backends.md#pymhm.solvers.validate_invertible)
- <a id="pymhm.solvers.factorize"></a>[`factorize`](api/backends.md#pymhm.solvers.factorize)
- <a id="pymhm.solvers.solve_linear"></a>[`solve_linear`](api/backends.md#pymhm.solvers.solve_linear)

<a id="pymhm.parallel"></a>**[pymhm.parallel](api/backends.md#pymhm.parallel)**

- <a id="pymhm.parallel.map_local"></a>[`map_local`](api/backends.md#pymhm.parallel.map_local)

<a id="pymhm.distributed"></a>**[pymhm.distributed](api/backends.md#pymhm.distributed)**

- <a id="pymhm.distributed.DistributedHybridSolution"></a>[`DistributedHybridSolution`](api/backends.md#pymhm.distributed.DistributedHybridSolution)
- <a id="pymhm.distributed.solve_distributed"></a>[`solve_distributed`](api/backends.md#pymhm.distributed.solve_distributed)

<a id="pymhm.gpu"></a>**[pymhm.gpu](api/backends.md#pymhm.gpu)**

- <a id="pymhm.gpu.BatchedFactorization"></a>[`BatchedFactorization`](api/backends.md#pymhm.gpu.BatchedFactorization)
- <a id="pymhm.gpu.BatchedFactorization.__init__"></a>[`BatchedFactorization.__init__`](api/backends.md#pymhm.gpu.BatchedFactorization.__init__)
- <a id="pymhm.gpu.BatchedFactorization.solve"></a>[`BatchedFactorization.solve`](api/backends.md#pymhm.gpu.BatchedFactorization.solve)
- <a id="pymhm.gpu.BatchedFactorization.close"></a>[`BatchedFactorization.close`](api/backends.md#pymhm.gpu.BatchedFactorization.close)
- <a id="pymhm.gpu.BatchedFactorization.__enter__"></a>[`BatchedFactorization.__enter__`](api/backends.md#pymhm.gpu.BatchedFactorization.__enter__)
- <a id="pymhm.gpu.BatchedFactorization.__exit__"></a>[`BatchedFactorization.__exit__`](api/backends.md#pymhm.gpu.BatchedFactorization.__exit__)
- <a id="pymhm.gpu.assemble_p1_batch"></a>[`assemble_p1_batch`](api/backends.md#pymhm.gpu.assemble_p1_batch)
- <a id="pymhm.gpu.condense_batched"></a>[`condense_batched`](api/backends.md#pymhm.gpu.condense_batched)

<a id="pymhm.block"></a>**[pymhm.block](api/backends.md#pymhm.block)**

- <a id="pymhm.block.SaddleBlockSolver"></a>[`SaddleBlockSolver`](api/backends.md#pymhm.block.SaddleBlockSolver)
- <a id="pymhm.block.SaddleBlockSolver.__init__"></a>[`SaddleBlockSolver.__init__`](api/backends.md#pymhm.block.SaddleBlockSolver.__init__)
- <a id="pymhm.block.SaddleBlockSolver.solve"></a>[`SaddleBlockSolver.solve`](api/backends.md#pymhm.block.SaddleBlockSolver.solve)
- <a id="pymhm.block.SaddleBlockSolver.as_factorization"></a>[`SaddleBlockSolver.as_factorization`](api/backends.md#pymhm.block.SaddleBlockSolver.as_factorization)
- <a id="pymhm.block.SaddleBlockSolver.close"></a>[`SaddleBlockSolver.close`](api/backends.md#pymhm.block.SaddleBlockSolver.close)
- <a id="pymhm.block.SaddleBlockSolver.__enter__"></a>[`SaddleBlockSolver.__enter__`](api/backends.md#pymhm.block.SaddleBlockSolver.__enter__)
- <a id="pymhm.block.SaddleBlockSolver.__exit__"></a>[`SaddleBlockSolver.__exit__`](api/backends.md#pymhm.block.SaddleBlockSolver.__exit__)

<a id="pymhm.separable"></a>**[pymhm.separable](api/backends.md#pymhm.separable)**

- <a id="pymhm.separable.SeparableField"></a>[`SeparableField`](api/backends.md#pymhm.separable.SeparableField)
- <a id="pymhm.separable.SeparableField.__post_init__"></a>[`SeparableField.__post_init__`](api/backends.md#pymhm.separable.SeparableField.__post_init__)
- <a id="pymhm.separable.SeparableField.__call__"></a>[`SeparableField.__call__`](api/backends.md#pymhm.separable.SeparableField.__call__)
- <a id="pymhm.separable.separable_diffusion_operators"></a>[`separable_diffusion_operators`](api/backends.md#pymhm.separable.separable_diffusion_operators)
- <a id="pymhm.separable.solve_separable_diffusion"></a>[`solve_separable_diffusion`](api/backends.md#pymhm.separable.solve_separable_diffusion)

<a id="pymhm.separable_krylov"></a>**[pymhm.separable_krylov](api/backends.md#pymhm.separable_krylov)**

- <a id="pymhm.separable_krylov.TensorDiffusionOperator"></a>[`TensorDiffusionOperator`](api/backends.md#pymhm.separable_krylov.TensorDiffusionOperator)
- <a id="pymhm.separable_krylov.TensorDiffusionOperator.apply"></a>[`TensorDiffusionOperator.apply`](api/backends.md#pymhm.separable_krylov.TensorDiffusionOperator.apply)
- <a id="pymhm.separable_krylov.TensorDiffusionOperator.interior"></a>[`TensorDiffusionOperator.interior`](api/backends.md#pymhm.separable_krylov.TensorDiffusionOperator.interior)
- <a id="pymhm.separable_krylov.TensorDiffusionOperator.assemble"></a>[`TensorDiffusionOperator.assemble`](api/backends.md#pymhm.separable_krylov.TensorDiffusionOperator.assemble)
- <a id="pymhm.separable_krylov.SeparableKrylovSolution"></a>[`SeparableKrylovSolution`](api/backends.md#pymhm.separable_krylov.SeparableKrylovSolution)
- <a id="pymhm.separable_krylov.tensor_diffusion_operator"></a>[`tensor_diffusion_operator`](api/backends.md#pymhm.separable_krylov.tensor_diffusion_operator)
- <a id="pymhm.separable_krylov.solve_separable_krylov"></a>[`solve_separable_krylov`](api/backends.md#pymhm.separable_krylov.solve_separable_krylov)
