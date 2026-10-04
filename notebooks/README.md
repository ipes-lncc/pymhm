# PyMHM examples by problem

Start with an introductory notebook, then use the numbered notebooks for
convergence records, geometry variants and literature comparisons. Local and
global methods appear together under their physical problem.

For user-written forms, begin with the local/global, UFL, vector UFL and
hierarchy notebooks in `foundations/operators`. They use the generic
`LocalEquations`, `Equation` and `MultiscaleProblem` interface. Introductory
physical notebooks put user-written equations before their method-family
comparisons. Those comparisons retain compatibility solvers for established
discretizations; the catalogue does not claim that all cases use the generic
interface.

Instructional examples are notebooks. Reusable Python modules in `examples/`
provide analytical patch data, campaign acquisition, archive readers and figure
renderers. Import them from cells when needed; numerical algorithms remain in
`pymhm`. Process-worker callables remain importable to preserve spawn semantics.

Source notebooks remain unexecuted in Git. The runner writes executed copies
under `build/notebooks`, preserving these folders:

```bash
pixi run --locked -e notebooks notebooks-run darcy/primal_galerkin.ipynb
pixi run --locked -e notebooks notebooks-run flow/introductory_methods.ipynb
pixi run --locked -e notebooks notebooks-run 01
pixi run --locked -e notebooks python scripts/notebook_data.py --notebook darcy --check
```

A folder selects all its notebooks. Some numbered studies require large locally
computed archives and publication figures; the runner checks their declared
inputs before execution. An introductory patch does not replace a convergence
study or a matched paper reproduction. Native UFL cells require the Pixi `fem`
kernel and report explicitly when the optional DOLFINx runtime has not executed.

The machine-readable index is [catalogue.json](catalogue.json).

## convergence

| Notebook | Methods |
| --- | --- |
| [Initial convergence studies](convergence/73_initial_convergence.ipynb) | cross-family convergence catalogue |

## darcy

| Notebook | Methods |
| --- | --- |
| [Start here: Alternative hybrid methods for scalar diffusion](darcy/hybrid_methods.ipynb) | user-written three-field MH2M and MsHHO moment blocks; Robin MH; MH2M; MsHHO; PGMHM |
| [Start here: Darcy with mixed H(div) local problems](darcy/mixed_hdiv.ipynb) | user-written RT0/P0 mixed LocalEquations/Equation; RT0/RT1; BDM/BDM+/BDM++; tensor RT; restricted tetrahedral/prismatic H(div); classical conforming RT1 |
| [Start here: Darcy with primal Galerkin local problems](darcy/primal_galerkin.ipynb) | user-written P2 primal LocalEquations/Equation; primal MHM; local Galerkin P2/Q2; physical Neumann mean gauge |
| [Darcy: primal and mixed convergence](darcy/01_darcy_convergence.ipynb) | primal MHM; mixed RT0/P0 MHM |
| [Darcy across a permeability jump](darcy/02_heterogeneous_darcy.ipynb) | primal MHM; heterogeneous permeability |
| [Independent face partitions and degrees](darcy/03_skeleton_hp.ipynb) | primal MHM; independent hp face traces |
| [Fine-cell conservation in H(div)](darcy/04_mixed_and_reconstructed_flux.ipynb) | primal MHM; mixed RT0/P0 MHM; equilibrated H(div) flux |
| [Published Darcy curves: archived results](darcy/11_published_darcy_2013.ipynb) | primal MHM; mixed RT0/P0 MHM |
| [NeoPZ H(div): archived comparison results](darcy/14_neopz_hdiv_comparison.ipynb) | RT H(div) MHM; restricted NeoPZ reference |
| [BDM2/P1 Darcy: conservative high-order local flux](darcy/17_darcy_bdm.ipynb) | BDM2/P1 mixed Darcy MHM |
| [RT moment reconstruction: normal conformity and continuous-test balance](darcy/18_reconstruction_moments.ipynb) | primal MHM; RT moment reconstruction |
| [Unit-diffusion MHM energy estimator](darcy/20_darcy_estimator.ipynb) | primal MHM; Oswald potential recovery; energy estimator |
| [SPE10 Model 2: physical layers and PyVista](darcy/21_spe10_pyvista.ipynb) | SPE10 material data; PyVista visualization |
| [Quarter-five spot: point wells, layered media and a square obstacle](darcy/22_quarter_five_spot.ipynb) | primal MHM; mixed RT0/P0 MHM |
| [SPE10: published Darcy and Brinkman spaces](darcy/23_spe10_mhm_comparison.ipynb) | quadrilateral primal MHM; USFEM Brinkman |
| [MHM and MsHHO field equivalence](darcy/24_mshho.ipynb) | primal MHM; MsHHO |
| [Enriched rectangular RT elements](darcy/25_tensor_rt.ipynb) | rectangular RT mixed MHM |
| [Three-dimensional tetrahedral MHM](darcy/29_darcy3d.ipynb) | tetrahedral primal MHM |
| [Polygonal macro meshes](darcy/30_polygonal_macro_meshes.ipynb) | polygonal primal MHM |
| [Material-weighted energy estimation](darcy/31_weighted_energy_estimator.ipynb) | primal MHM; material-weighted energy estimator |
| [Raviart–Thomas Darcy: RT0, RT1 and RT2](darcy/33_darcy_rt.ipynb) | RT0; RT1; RT2 |
| [Recursive MHM in identical leaf spaces](darcy/36_recursive_mhm.ipynb) | user-written Q2/P1 NestedEquations and leaf-integral gauges; recursive MHM; direct leaf-space MHM |
| [Periodic Darcy: separate face, local and reference refinement](darcy/37_periodic_darcy.ipynb) | periodic quadrilateral MHM |
| [Material interfaces and skeletal partitions](darcy/38_unfitted_darcy.ipynb) | material-fitted integration; independent skeletal partitions |
| [Adaptive Darcy macro meshes](darcy/39_adaptive_darcy.ipynb) | adaptive primal MHM; weighted energy estimator |
| [Adaptive SPE10 Darcy: published indicators and classical RT2 comparison](darcy/40_spe10_adaptive.ipynb) | adaptive primal MHM; classical RT2/P2 reference |
| [Darcy face-jump indicator](darcy/42_darcy_jump_estimator.ipynb) | primal MHM; pressure jump indicator |
| [Analytical Darcy MHM with full source moments](darcy/43_analytical_darcy.ipynb) | analytical local Darcy MHM |
| [Mapped hexahedral RT1 around a well](darcy/44_mapped_well.ipynb) | mapped hexahedral RT1 MHM |
| [Oscillatory 3D producing-well flow](darcy/47_mapped_oscillatory_well.ipynb) | mapped hexahedral RT1 MHM |
| [Mixed tetrahedral and prismatic Darcy](darcy/51_mixed_well_geometries.ipynb) | tetrahedral and prismatic H(div) Darcy MHM |
| [Three-field multiscale Darcy: MH²M](darcy/52_mh2m.ipynb) | three-field MH2M |
| [Multiscale Hybrid diffusion with Robin local problems](darcy/56_mh.ipynb) | Robin-local Multiscale Hybrid (MH) |
| [Tetrahedral flux reconstruction and adaptive resolution](darcy/58_reconstruction3d.ipynb) | tetrahedral primal MHM; 3D RT moment reconstruction; energy estimator |
| [Petrov–Galerkin MHM on polytopes](darcy/59_pgmhm.ipynb) | Petrov-Galerkin MHM |
| [Material-fitted 3D local meshes and nonuniform face partitions](darcy/63_planar3d.ipynb) | material-fitted tetrahedral MHM; nonuniform face partitions; RT reconstruction |
| [General tetrahedral degree and admissible P5/P2 estimation](darcy/64_tetra_pk.ipynb) | tetrahedral Pk MHM; P5/P2 error estimation |
| [Physical Neumann data and nonconvex macroelements in MH and MH²M](darcy/65_mh_boundary.ipynb) | MH; MH2M; physical Neumann data |
| [Tetrahedral MH and MH²M](darcy/69_mh3d.ipynb) | tetrahedral MH; tetrahedral MH2M |
| [MH²M with independently refined conormals](darcy/70_mh2m_heterogeneous.ipynb) | MH2M; independent pressure and conormal traces |

## elasticity

| Notebook | Methods |
| --- | --- |
| [Start here: Primal and mixed elasticity methods](elasticity/introductory_methods.ipynb) | user-written UFL primal plane elasticity; primal Galerkin; GaLS displacement-pressure; BDM weak-symmetry stress; tensor RT stress |
| [Mixed elasticity and the incompressible limit](elasticity/07_elasticity.ipynb) | user-written P1/P1 GaLS equations and physical rigid/pressure gauges; GaLS elasticity MHM |
| [Independent MSL GaLS elasticity fields](elasticity/15_elasticity_reference.ipynb) | GaLS elasticity MHM; independent MSL reference |
| [Mixed elasticity: H(div) stress and weak symmetry](elasticity/16_mixed_elasticity.ipynb) | weak-symmetry H(div) elasticity MHM |
| [Mixed H(div) elasticity families](elasticity/28_mixed_elasticity_families.ipynb) | BDM; BDM-plus; BDM-double-plus |
| [General-tensor primal elasticity](elasticity/32_primal_tensor_elasticity.ipynb) | general-tensor primal elasticity MHM |
| [Rectangular RT mixed elasticity](elasticity/34_rectangular_mixed_elasticity.ipynb) | rectangular RT mixed elasticity MHM |
| [General-tensor tetrahedral elasticity](elasticity/35_elasticity3d.ipynb) | tetrahedral general-tensor primal elasticity MHM |
| [HPC4e heterogeneous mixed elasticity](elasticity/50_hpc4e_geomechanics.ipynb) | rectangular RT mixed elasticity MHM |
| [Three-dimensional nearly incompressible elasticity](elasticity/55_gals3d.ipynb) | user-written tetrahedral P2/P2 GaLS equations with physical inverse bound; tetrahedral GaLS elasticity MHM |
| [Three-dimensional weak-symmetry mixed elasticity](elasticity/68_mixed_elasticity3d.ipynb) | 3D Arnold-Falk-Winther mixed elasticity MHM |

## flow

| Notebook | Methods |
| --- | --- |
| [Start here: Stokes–Brinkman and Oseen introductory methods](flow/introductory_methods.ipynb) | user-written UFL P2/P1 Taylor-Hood Brinkman; Taylor-Hood; USFEM; Oseen |
| [Native three-dimensional Stokes, Brinkman and Oseen](flow/54_flow3d.ipynb) | user-written tetrahedral P2/P1 velocity/pressure LocalEquations; 3D Stokes MHM; 3D Brinkman MHM; 3D Oseen MHM |

## flow/brinkman_oseen

| Notebook | Methods |
| --- | --- |
| [Brinkman stabilization and Oseen transport](flow/brinkman_oseen/06_brinkman_and_oseen.ipynb) | user-written USFEM drag0/1/100 and skew Taylor–Hood Oseen blocks; USFEM Brinkman; Oseen MHM |
| [Stabilized Oseen flow and two-level adaptivity](flow/brinkman_oseen/27_oseen_adaptivity.ipynb) | stabilized Oseen MHM; macro and local adaptivity |
| [Stokes–Brinkman: two levels of error and two adaptive strategies](flow/brinkman_oseen/48_stokes_adaptive.ipynb) | Stokes-Brinkman MHM; local and macro adaptivity |

## flow/stokes

| Notebook | Methods |
| --- | --- |
| [The polynomial Stokes problem of Araya et al. (2017)](flow/stokes/05_stokes_published_problem.ipynb) | user-written Taylor–Hood and P1/P1 USFEM LocalEquations; Taylor-Hood MHM; USFEM MHM |
| [Stokes pressure: archived convergence and interface jumps](flow/stokes/12_stokes_pressure_audit.ipynb) | Taylor-Hood MHM; USFEM MHM; conforming DOLFINx reference |
| [Published Stokes experiment: archived high-order results](flow/stokes/13_published_stokes_2017.ipynb) | DOLFINx/UFL local USFEM MHM |

## foundations/general

| Notebook | Methods |
| --- | --- |
| [Visual verification gallery](foundations/general/10_visual_gallery.ipynb) | primal and mixed Darcy; Taylor-Hood and USFEM flow; RAD; heat; elasticity |
| [Higher-order local fields and variable operators](foundations/general/19_native_extensions.ipynb) | user-written Darcy, tensor USFEM, conservative SUPG and backward Euler equations; higher-order primal Darcy; USFEM flow; variable-coefficient RAD; heat |
| [Anisotropic stress and three-dimensional mixed spaces](foundations/general/57_core_extensions.ipynb) | anisotropic mixed elasticity; 3D H(div) Darcy; 3D MsHHO |

## foundations/geometry

| Notebook | Methods |
| --- | --- |
| [Three-dimensional mesh exchange](foundations/geometry/60_mesh_exchange.ipynb) | triangle/polygon/tetrahedron/prism/hexahedron/polyhedron geometry; volume mesh exchange |

## foundations/operators

| Notebook | Methods |
| --- | --- |
| [Start here: Local and global forms, providers and ordered batches](foundations/operators/local_global_providers.ipynb) | LocalEquations/Equation; primal and mixed local forms; serial/thread/spawn batches; external local solver |
| [Start here: User-written UFL local and global equations](foundations/operators/ufl_provider.ipynb) | UFL/DOLFINx forms; independent row and column pairings; COMM_SELF local assembly |
| [Start here: Three levels of user-defined equations](foundations/operators/variational_hierarchy.ipynb) | four-block equations; three-level operator recursion; independent full-system comparison |
| [Start here: User-written vector reaction-diffusion equations](foundations/operators/vector_ufl.ipynb) | vector reaction-diffusion; UFL/DOLFINx compilation; oriented trace pairings |
| [Composing local operators and using AMG](foundations/operators/09_local_operators_and_amg.ipynb) | free local operators; hybrid condensation; AMG |

## transport

| Notebook | Methods |
| --- | --- |
| [Start here: Scalar reaction–advection–diffusion methods](transport/introductory_methods.ipynb) | user-written UFL skew Galerkin RAD; RAD Galerkin; SUPG; UNUSUAL |
| [Robin transport and transient heat](transport/08_transport_and_heat.ipynb) | user-written Robin transport and backward Euler LocalEquations; identical-space compatibility controls |
| [Adaptive and Darcy-coupled transient transport](transport/26_adaptive_transient_transport.ipynb) | adaptive RAD; Darcy-coupled backward Euler transport |
| [Three-dimensional conservative RAD: local P4 and face P1](transport/41_rad3d.ipynb) | tetrahedral conservative RAD |
| [Selective RAD kernels and conditioning](transport/45_rad_conditioning.ipynb) | RAD; selective local kernels |
| [Hexagonal boundary-layer problem](transport/46_hexagonal_boundary_layer.ipynb) | hexagonal conservative RAD |
| [Conservative RAD on convex polyhedra](transport/49_polyhedral_rad.ipynb) | polyhedral conservative RAD |
| [Scalar MHM-UNUSUAL reaction–diffusion](transport/53_unusual.ipynb) | scalar reaction-diffusion MHM-UNUSUAL |
| [Nonconvex star-shaped polyhedral RAD](transport/67_star_polyhedra.ipynb) | star-shaped polyhedral conservative RAD |

## waves/elastodynamics

| Notebook | Methods |
| --- | --- |
| [Elastodynamic MHM: spatial and temporal convergence](waves/elastodynamics/71_elastodynamics.ipynb) | Newmark elastodynamic MHM |

## waves/helmholtz

| Notebook | Methods |
| --- | --- |
| [Start here: Complex scalar Helmholtz fields](waves/helmholtz/introductory_methods.ipynb) | user-written UFL real embedding of complex Helmholtz; complex hybrid Helmholtz |
| [Helmholtz MHM: complex fields and PML](waves/helmholtz/61_helmholtz.ipynb) | complex Helmholtz MHM; PML |
| [Marmousi acoustic point source](waves/helmholtz/72_marmousi.ipynb) | Q3 acoustic Helmholtz MHM; classical conforming reference |

## waves/maxwell

| Notebook | Methods |
| --- | --- |
| [Start here: Vector Maxwell fields and a short trajectory](waves/maxwell/introductory_methods.ipynb) | user-written DG mass/curl LocalEquations/Equation stages; DG Maxwell; tangential hybrid traces; staggered time stepping |
| [Time-domain Maxwell MHM in two and three dimensions](waves/maxwell/62_maxwell.ipynb) | 2D and 3D Maxwell MHM |
| [Maxwell photonic device](waves/maxwell/66_maxwell_nanoguide.ipynb) | Cartesian Maxwell MHM; classical numerical reference |
