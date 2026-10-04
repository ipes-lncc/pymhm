# Tutorials and notebooks

Start with the small executable tutorials:

- [Scalar formulations](tutorials/scalar.md): primal Galerkin, mixed H(div),
  alternative hybrid formulations, transport and complex fields.
- [Vector formulations](tutorials/vector.md): displacement, mixed stress,
  velocity–pressure and electromagnetic fields.
- [Local providers and assembly](tutorials/providers.md): define local and
  global forms, supply a local solver, and select serial or parallel batches.

Each tutorial states its approximation spaces, interface convention and physical
gauge. The small checks introduce the APIs; the case gallery contains the separate
convergence and reference comparisons.

The notebooks combine small executable numerical checks with readers for the
larger research campaigns. Archived comparison notebooks identify the solver,
spaces, data and reference uncertainty; they do not execute external comparison
programs. Source notebooks remain unexecuted. The notebook task writes executed
copies with their figures and numerical output.

The [case gallery](cases/index.md) provides larger field plots, analytical
references, profiles and interpretation of the expected and measured results.

| Notebook | Main check |
| --- | --- |
| `01_darcy_convergence.ipynb` | Cosine manufactured pressure; separate primal/mixed errors |
| `02_heterogeneous_darcy.ipynb` | Layered permeability with exact constant physical flux |
| `03_skeleton_hp.ipynb` | Independent per-face degrees and partitions |
| `04_mixed_and_reconstructed_flux.ipynb` | RT0 fine-cell balance and primal equilibration |
| `05_stokes_published_problem.ipynb` | Analytical data from Araya et al. (2017), with declared discretization changes |
| `06_brinkman_and_oseen.ipynb` | Consistent USFEM, pressure gauge and transport sign |
| `07_elasticity.ipynb` | Displacement–pressure elasticity, rigid modes and near-incompressible limits |
| `08_transport_and_heat.ipynb` | Robin RAD convention and backward Euler space-time patch |
| `09_local_operators_and_amg.ipynb` | Independent local API, condensation and optional PyAMG |
| `10_visual_gallery.ipynb` | Regenerate and display analytical-reference fields, error maps, profiles and recorded timings |
| `11_published_darcy_2013.ipynb` | Archived pyMHM results against four digitized Harder et al. curves; RT0 potential and extraction uncertainty |
| `12_stokes_pressure_audit.ipynb` | pyMHM/DOLFINx comparison, five-level errors and both pressure traces |
| `13_published_stokes_2017.ipynb` | Recorded pyMHM + DOLFINx/UFL results against Araya et al. curves; unresolved stress-norm differences |
| `14_neopz_hdiv_comparison.ipynb` | Archived NeoPZ/pyMHM RT0 fields, macroface restriction and heterogeneous flux; archive checksums |
| `15_elasticity_reference.ipynb` | Archived MSL GaLS displacement, pressure and stress agreement; field checksums |
| `16_mixed_elasticity.ipynb` | H(div) stress, weak symmetry, projected displacement and material limits |
| `17_darcy_bdm.ipynb` | BDM2/P1 Darcy fields, trace enrichment and fine-cell source moments |
| `18_reconstruction_moments.ipynb` | RT moments, continuous-test conservation and raw/projected divergence |
| `19_native_extensions.ipynb` | High-order native Darcy/flow, variable RAD, outflow layers and heat |
| `20_darcy_estimator.ipynb` | Unit-diffusion energy estimator, conforming Oswald potential and five-level effectivity |
| `21_spe10_pyvista.ipynb` | SPE10 layer identity, units, ordering and native PyVista rendering |
| `22_quarter_five_spot.ipynb` | Point-source sharing, well conservation, Green-series reference and layered fields |
| `23_spe10_mhm_comparison.ipynb` | Native Q2 patch, local Q1 refinement, published pressure and flux figures, Q3/MSL/NeoPZ Darcy flux comparisons, and a refined conforming Taylor–Hood Brinkman baseline |
| `24_mshho.ipynb` | MHM–MsHHO field equivalence, source hypotheses and anisotropy through 1e6 |
| `25_tensor_rt.ipynb` | Independent rectangular RT interior enrichment and published convergence markers |
| `26_adaptive_transient_transport.ipynb` | Adaptive conservative transport, time refinement and Darcy dispersion |
| `27_oseen_adaptivity.ipynb` | Variable-convection Oseen and smooth, boundary-layer and internal-layer adaptation |
| `28_mixed_elasticity_families.ipynb` | BDM, BDM-plus and double-plus weak-symmetry elasticity families |
| `29_darcy3d.ipynb` | Tetrahedral P1–P4 Darcy, face polynomials and five-level exact-field errors |
| `30_polygonal_macro_meshes.ipynb` | Five polygon families, nonconvex patches and published P3/P1 oscillatory RAD |
| `31_weighted_energy_estimator.ipynb` | Material-weighted energy estimation, anisotropy and layered coefficients |
| `32_primal_tensor_elasticity.ipynb` | General Kelvin elasticity tensors, enrichment and five-level displacement/stress errors |
| `33_darcy_rt.ipynb` | Triangular RT0–RT2 Darcy, source moments and conforming classical references |
| `34_rectangular_mixed_elasticity.ipynb` | Rectangular mixed stress spaces, rotation moments and infinite bulk compliance |
| `35_elasticity3d.ipynb` | Three-dimensional elasticity, six rigid modes and tensor-material checks |
| `36_recursive_mhm.ipynb` | Recursive local MHM and equality with the independently assembled leaf system |
| `37_periodic_darcy.ipynb` | Periodic permeability, separate skeletal/local refinement and independently refined Qk references |
| `38_unfitted_darcy.ipynb` | Material interfaces cutting macrocells, exact integration and approximation-space effects |
| `39_adaptive_darcy.ipynb` | Conforming adaptive macro refinement and energy-estimator records |
| `40_spe10_adaptive.ipynb` | SPE10 reconstruction, local energy controls and refined classical RT2 fields |
| `41_rad3d.ipynb` | Published three-dimensional P4/P1 RAD and classical P2/exact-field comparisons |
| `42_darcy_jump_estimator.ipynb` | Published Darcy face-jump estimator, skeletal degrees 0 and 3, and quadrature checks |
| `43_analytical_darcy.ipynb` | Analytical local Darcy lifts, pressure/source reconstructions and classical RT0 distinctions |
| `44_mapped_well.ipynb` | Mapped hexahedral RT1 well, physical Piola fields and five classical refinements |
| `45_rad_conditioning.ipynb` | Selective mixed/primal RAD kernels and five published conditioning cases |
| `46_hexagonal_boundary_layer.ipynb` | Published P3/P1 boundary layer, exact profile and independent macro/face/local studies |
| `47_mapped_oscillatory_well.ipynb` | Oscillatory anisotropic three-dimensional well and classical reference refinement |
| `48_stokes_adaptive.ipynb` | Separate macro/face algorithms, analytical errors and refined Taylor–Hood cavity references |
| `49_polyhedral_rad.ipynb` | Original polygonal faces, polyhedral P4/P1 fields and three five-level geometry families |
| `50_hpc4e_geomechanics.ipynb` | Original geological material samples, published RT1 stress profiles and independently refined mixed-elasticity fields |
| `51_mixed_well_geometries.ipynb` | Tetrahedral/prismatic H(div) families, physical Piola patches and common-domain mixed well campaigns |

[FEniCS](fenics.md) and [meshing](meshing.md) have additional runnable examples
and native integration tests. See [performance](https://github.com/volpatto/pymhm/blob/main/docs/performance.md) for executable
CPU/GPU experiments rather than notebook timing claims.

## Per-face approximation

```python
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
mesh = TriangleMesh.unit_square(2)
faces = tuple(FaceSpace.uniform(degree=i % 2, subdivisions=2)
              for i in range(len(mesh.faces)))
skeleton = SkeletonSpace(mesh, faces)
solution = solve_darcy(mesh, skeleton=skeleton, local_refinement=6,
                       dirichlet=lambda x: 1 + x[:, 0])
```

## Stokes–Brinkman

```python
from pymhm import solve_brinkman
solution = solve_brinkman(mesh, viscosity=1.0, drag=10.0,
                          formulation="usfem", source=(1.0, 0.0),
                          local_refinement=4)
print(solution.divergence_l2())
```

P1/P1 USFEM retains the drag contribution in the residual and modifies both the
bilinear form and source. A pressure-Laplacian term alone is not the implemented
Brinkman stabilization. The global mean pressure is imposed once, after local
condensation.

The same native interface also supports P2/P2 and P3/P3 USFEM, with their
velocity-Laplacian residual and a computed inverse bound. Taylor–Hood uses
P2/P1 or P3/P2. Variable symmetric resistance is evaluated at physical
quadrature points; viscosity is constant in this flow implementation.

## Additional native formulations

The source notebooks remain unexecuted; `pixi run -e notebooks notebooks-run`
writes executed copies under `build/notebooks`. Recent formulation studies are:

| Notebook | Executable verification and recorded results |
|---|---|
| `52_mh2m.ipynb` | Hybrid-hybrid mixed pressure traces |
| `53_unusual.ipynb` | Unusual reaction–diffusion stabilization |
| `54_flow3d.ipynb` | Three-dimensional Stokes–Brinkman and Oseen |
| `55_gals3d.ipynb` | Three-dimensional nearly incompressible elasticity |
| `56_mh.ipynb` | Robin local MH formulation |
| `57_core_extensions.ipynb` | Anisotropic stress profiles, general H(div) orders and MsHHO3D |
| `58_reconstruction3d.ipynb` | Dimension-dependent estimator spaces, adaptive errors and physical flux profiles |
| `59_pgmhm.ipynb` | Residual Petrov–Galerkin enrichment |
| `60_mesh_exchange.ipynb` | Three-dimensional native meshing and file exchange |
| `61_helmholtz.ipynb` | Complex acoustic fields, oscillatory traces and perfectly matched layers |
| `62_maxwell.ipynb` | Time-dependent electric and magnetic fields, energy and CFL controls |
| `63_planar3d.ipynb` | Oblique material transmission with three-dimensional fitted meshes |
| `65_mh_boundary.ipynb` | Physical Neumann data and volume gauges for MH/MH²M on polygons |
| `69_mh3d.ipynb` | Independent tetrahedral Robin and three-field formulations |
| `70_mh2m_heterogeneous.ipynb` | Oscillatory permeability, classical refinement and independent conormal enrichment |

The [quadratic Darcy tutorial](https://github.com/volpatto/pymhm/blob/main/docs/cases/darcy-bdm.md) explains the distinction
between exact flux and projected pressure. The
[elasticity gallery](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity.md) compares GaLS with Taylor–Hood and
the displacement-only baseline; [mixed elasticity](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-elasticity.md)
instead solves for an H(div) stress and an independent rotation.

`solve_transport(..., degree=k, stabilization="supg")` uses the conservative
RAD residual, including the supplied velocity and diffusion divergences.
`solve_heat(..., degree=k)` uses backward Euler with Pk spatial fields.
Continuous face interpolation is selected with `FaceSpace.uniform(...,
continuous=True)` and remains independent of the local degree. These options
require compatible trace/local choices; an arbitrary combination of enriched
spaces is not a stability guarantee.
