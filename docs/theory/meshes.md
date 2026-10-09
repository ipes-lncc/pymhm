# Mesh geometry, material regions and approximation spaces

A multiscale calculation has a macro mesh, independent partitions of its
faces, and one conforming local fine mesh per macrocell. The local fine
elements need not have the same geometry as the macrocell: a polygon can
contain triangles and a polyhedron can contain tetrahedra. The boundary
faces remain the original macrofaces, not the diagonals of the local
triangulation.

## Implemented geometries

| Macrocell | Local geometry | Current formulation scope |
| --- | --- | --- |
| Triangle | Conforming triangles | Primal/mixed Darcy, MH/MH²M/MsHHO/PGMHM, flow, elasticity, transport, dynamics and waves in their stated spaces |
| Cartesian rectangle | Cartesian quadrilaterals or a declared triangular partition | Qk scalar/Helmholtz, tensor RT Darcy/stress and selected TM Maxwell paths |
| Simple planar polygon, including nonconvex polygons | Conforming triangulation | Primal/mixed scalar and stress operators, MH/MH²M/MsHHO/PGMHM and transport paths; each has separate compatibility conditions |
| Tetrahedron | Conforming tetrahedra | Primal/mixed Darcy, flow, primal/GaLS/AFW elasticity, MH/MH²M/MsHHO, scalar transport, elastodynamics and waves |
| Affine prism | Affine prismatic mixed elements | The declared three-dimensional H(div) Darcy families |
| Mapped hexahedron | Trilinear hexahedral mapping | Mapped RT Darcy; reject invalid or orientation-reversing maps |
| Star-shaped polyhedron with planar faces | Conforming positive-volume tetrahedral decomposition | Primal scalar diffusion/RAD and MsHHO with original polygonal face moments; other operators require their own compatible realization |

Geometry acceptance includes nondegenerate cells, valid orientation and a
conforming partition. Nonconvex polyhedra need a certified positive-volume
kernel and a valid tetrahedral decomposition. Planar faces and affine
prismatic maps are actual restrictions; arbitrary curved CAD solids are not
accepted as these algebraic cells.

The polytopal MHM analyses of
[Barrenechea et al. (2020)](https://doi.org/10.1007/s00211-020-01103-5),
[Araya et al. (2024)](https://doi.org/10.1016/j.cma.2024.117089) and
[Chaumont-Frelet et al. (2022)](https://doi.org/10.1051/m2an/2021082)
give formulation-specific regularity and shape assumptions. Supporting a
cell type in a mesh class does not establish those analytical hypotheses
for every form assembled on that class.

## Polynomial and mapped spaces

Basix supplies reference finite element definitions, basis tabulation and
entity transformations. Pk means total-degree polynomials on a simplex;
Qk means tensor-product polynomials on a Cartesian element. RT and BDM
vector elements use the contravariant Piola transformation, preserving
normal-trace integrals and divergence identities. An ordinary componentwise
scalar mapping does not preserve those properties.

On two-dimensional macroedges, `FaceSpace` supports discontinuous polynomial
segments or continuous interpolation within the macroface. Triangular
three-dimensional macrofaces support corresponding independently refined
polynomial spaces through `TriangularSkeleton`. A change of face normal and
a change of face parameterization have separate transformations. The
mesh/space binding constructs these maps; custom interfaces can provide
them explicitly through the same contract.

Restricted H(div) traces must align with fine boundary faces and belong to
the local normal-trace space. Increasing the local interior order does not
increase an unchanged boundary normal order. The tetrahedral and prismatic
enriched families follow the dimension-specific constructions discussed by
[Castro et al. (2016)](https://doi.org/10.1016/j.cma.2016.03.050) and
[Durán et al. (2019)](https://doi.org/10.1016/j.cma.2019.05.013).

## Material interfaces and tags

Material tags identify physical regions; boundary tags identify boundary
data. They are not interchangeable. A piecewise coefficient must be
integrated on its physical regions, even when a material interface crosses
an integration element. Sampling once at a centroid can replace the
intended PDE with a different cellwise material.

`PlanarMaterial` and Cartesian material fields support explicit region or
pixel intersections. Local meshing can fit those cuts; an unfitted fine
element can instead use intersection quadrature where the operator permits
it. Stabilization involving coefficient derivatives requires the stated
piecewise regularity and alignment assumptions. A material jump has no
ordinary derivative obtained by silently setting that derivative to zero.

The unfitted-flux estimate of
[Chaumont-Frelet, Paredes and Valentin (2026)](https://doi.org/10.1016/j.camwa.2026.01.016)
permits a macro mesh crossing material regions while requiring each
skeletal subface to lie within one region. This is a requirement on the
face partition in addition to material-aware volume quadrature.

## Refinement and exchange

Macro refinement changes global topology; face refinement changes exchanged
moments; local refinement resolves operators and coefficients. Longest-edge
triangular refinement follows [Rivara (1984)](https://doi.org/10.1002/nme.1620200412),
with conformity closure rather than hanging edges silently accepted.
Persisted meshes and solution bases must retain material tags, boundary
tags, basis ordering, orientation and coefficient identity for replay.

The [mesh guide](../meshing.md) gives generation, import and export examples
with Gmsh, Netgen and meshio. The [custom interface tutorial](../tutorials/custom-interface.md)
explains explicit maps for users who need to control them.

## References

- Omar Durán, Philippe R. B. Devloo, Sônia M. Gomes, and Frédéric Valentin (2019). *A multiscale hybrid method for Darcy’s problems using mixed finite element local solvers*, Computer Methods in Applied Mechanics and Engineering 354, 213–244. [DOI: 10.1016/j.cma.2019.05.013](https://doi.org/10.1016/j.cma.2019.05.013).

- Théophile Chaumont-Frelet, Alexandre Ern, Simon Lemaire, and Frédéric Valentin (2022). *Bridging the Multiscale Hybrid-Mixed and Multiscale Hybrid High-Order Methods*, ESAIM: M2AN 56, 261–285. [DOI: 10.1051/m2an/2021082](https://doi.org/10.1051/m2an/2021082).

- Théophile Chaumont-Frelet, Diego Paredes, and Frédéric Valentin (2026). *Flux approximation on unfitted meshes and application to multiscale hybrid-mixed methods*, Computers & Mathematics with Applications 209, 16–27. [DOI: 10.1016/j.camwa.2026.01.016](https://doi.org/10.1016/j.camwa.2026.01.016).

- Rodolfo Araya, Fabrice Jaillet, Diego Paredes, and Frédéric Valentin (2024). *Generalizing the multiscale hybrid-mixed method for reactive-advective-diffusive equations*, Computer Methods in Applied Mechanics and Engineering 428, 117089. [DOI: 10.1016/j.cma.2024.117089](https://doi.org/10.1016/j.cma.2024.117089).

- M. Cecilia Rivara (1984). *Algorithms for refining triangular grids suitable for adaptive and multigrid techniques*. International Journal for Numerical Methods in Engineering 20(4), 745–756. [DOI: 10.1002/nme.1620200412](https://doi.org/10.1002/nme.1620200412).

- Douglas A. Castro, Philippe R.B. Devloo, Agnaldo M. Farias, Sônia M. Gomes, Denise de Siqueira, and Omar Durán (2016). *Three dimensional hierarchical mixed finite element approximations with enhanced primal variable accuracy*. Computer Methods in Applied Mechanics and Engineering 306 479-502. [DOI: 10.1016/j.cma.2016.03.050](https://doi.org/10.1016/j.cma.2016.03.050).

- Gabriel R. Barrenechea, Fabrice Jaillet, Diego Paredes, and Frédéric Valentin (2020). *The multiscale hybrid mixed method in general polygonal meshes*. Numerische Mathematik 145(1), 197–237. [DOI: 10.1007/s00211-020-01103-5](https://doi.org/10.1007/s00211-020-01103-5).

- [Basix reference element documentation](https://docs.fenicsproject.org/basix/main/).
