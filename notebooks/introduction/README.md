# Learning PyMHM through multiscale problems

Read the [API overview](https://ipes-lncc.github.io/pymhm/tutorials/overview/)
and the [rendered introductory course](https://ipes-lncc.github.io/pymhm/tutorials/)
in the documentation. The course includes the code, numerical outputs and plots
from these ten notebooks. Use the notebooks below to run or modify the examples.

These ten notebooks define their problems in executable cells. Each starts
from the differential equations, writes executable UFL weak forms and their
interface couplings, and assembles a global multiscale problem
with `MeshHierarchy`, `LocalContext`, `Equation` and `bind_problem`. Physical
data, boundary conditions and approximation parameters remain visible and
editable. Importable helpers handle evaluation, reference controls, plots,
archives and performance campaigns. Their source is downloaded in a separate companion archive;
helpers do not select a different physical case or call a prepared PDE solver.

Begin with the Darcy convergence notebook. It introduces the relationship
between a mathematical pairing, its matrix representation, a local response
and the global equations. The remaining notebooks apply that workflow to
different spaces and local formulations.

| Notebook | What it teaches |
| --- | --- |
| [Multiscale Darcy and convergence](darcy_multiscale_convergence.ipynb) | Tensor permeability, explicit primal local forms, oriented flux traces, a physical mean gauge and separate pressure and flux convergence |
| [Parallel Darcy: speed-up and scalability](darcy_parallel_scalability.ipynb) | Mesh-size sweep from 200×200 to 1000×1000, matched fine-element counts, classical LU and AMG comparisons, parallel local solves, complete timings and strong/weak scaling |
| [Darcy with spawned processes](darcy_process_scalability.ipynb) | Explicit importable worker operators, process startup and transfer costs, strong scaling at 1000×1000, weak scaling on growing rectangles and classical LU/AMG comparisons |
| [Three-dimensional Darcy: workspaces, LU and AMG](darcy_3d_parallel_scalability.ipynb) | Explicit UFL local forms; reusable native resources; independent material assembly; local LU/AMG; MPI classical baselines; strong and focused weak measurements |
| [Darcy on a SPE10 layer](darcy_spe10_layer.ipynb) | Discontinuous reservoir material data, integration across material cells, pressure boundary data and a refined conforming comparison |
| [Multiscale elasticity](multiscale_elasticity.ipynb) | Vector UFL forms, heterogeneous stiffness, rigid motions and displacement and stress comparisons |
| [MsHHO with an oscillatory coefficient](mshho_multiscale.ipynb) | Cell and face moments, constrained energy reconstruction and the global moment equations |
| [MH²M with an oscillatory coefficient](mh2m_multiscale.ipynb) | Independent potential and conormal trace spaces, local saddle equations and global compatibility |
| [MHM-USFEM for reaction–diffusion layers](mhm_usfem_rad.ipynb) | Explicit Galerkin and stabilized forms, coarse and refined skeletal/local resolution, scalar and flux convergence, and the tradeoff between overshoot and L2 error |
| [Stokes–Brinkman boundary-layer convergence](stokes_brinkman_boundary_layer.ipynb) | Velocity–pressure UFL forms, admissible Taylor–Hood submeshes, the single-triangle P2/P2–P4/P4 USFEM family, pressure gauge and separately measured field convergence |

## Run the notebooks

Install the notebook and plotting dependencies, then run from a writable
working directory using the same Python interpreter:

```bash
python -m pip install 'pymhm[notebooks,visualization]'
jupyter lab darcy_multiscale_convergence.ipynb
```

All ten introductions additionally use the native DOLFINx/UFL backend. Install
the compatible backend described in the
[installation guide](https://ipes-lncc.github.io/pymhm/installation/); the
qualified backend is DOLFINx 0.9. Its native assembly is optional for the
portable PyMHM core. Windows needs a compatible MPI runtime and Visual Studio's
C/C++ compiler and Windows SDK available to FFCx JIT. The three-dimensional
performance notebook's distributed conforming reference sections require
PETSc/MUMPS and the Unix stack. Backend availability is separate from an
execution receipt. No checkout or Pixi installation is required.

The installed distribution contains only `pymhm`. Download a notebook from
its link above and run the first cell: it explicitly acquires a verified
companion ZIP, makes its local `examples` helpers importable and prepares the
declared data. The workspace path is printed. Inspectable support sources and
small configurations live there, separately from the installed library. Large
inputs use the [documented download links](https://ipes-lncc.github.io/pymhm/data/)
and a SHA256-verified cache. Download alone executes no code.

The optional runner is part of the companion. From its workspace, use
`python -m scripts.run_notebooks /path/to/downloaded.ipynb --timeout 7200`.
It uses the active interpreter and preserves the source, writing an executed
copy and receipt under `build/notebooks`. No notebook source or execution
receipt is installed with the library.

The two 2D performance notebooks default to a fresh 200×200 numerical control
and the retained scalar observations from the recorded 2026-10-04 campaign.
Historical figures are displayed when their original payloads are available;
available payloads retain their checksum checks. Missing historical figures do
not prevent the fresh numerical control from executing. Historical timings
identify their original source revision. To acquire complete new strong,
weak and crossover samples on dedicated resources, enable the explicit flag:

```bash
PYMHM_RUN_CAMPAIGN=1 python -m scripts.run_notebooks /path/to/darcy_process_scalability.ipynb --timeout 7200
```

Source notebooks have no saved outputs. The runner preserves them and writes
executed copies under `build/notebooks/introduction`. Numerical records and
figures generated by cells are also written under `build/`. SPE10 uses the
material layer obtained from the documented dataset link, verified by SHA256
and materialized under `examples/results/spe10` in the workspace; its provenance,
units, cell ordering and checksum are described in the data cells.

Running every cell includes the reference and convergence controls. The larger
controls can take over an hour on a CPU. The rendered course provides their
completed outputs; interactive execution lets you start with the problem,
local equations and global solve before running the remaining controls.

The Brinkman introduction runs a three-level single-element family by default
($n=8,16,32$ at each trace degree). Its complete original qualification sequence
is available with `PYMHM_FULL_STUDY=1`; the profile and levels are recorded in
the provenance. The main MHM comparisons, three classical reference meshes and
subface controls run in both profiles.

## Reading the discretizations

`H` denotes the macro mesh scale; `h` denotes the mesh used inside a
macroelement. Face degrees and face subdivisions specify the independent
skeletal approximation. A finer local mesh does not increase the number of
global macroelements. Each notebook reports these choices explicitly.

The main operator cells use UFL expressions close to the mathematical weak
forms. Mesh-associated contexts provide supported numbering, orientation and
native coefficient-order conversions. The physical forms and conventions remain
explicit. Prepared operator functions are introduced afterwards as conveniences,
with a comparison of the assembled blocks where applicable.
Details of low-level quadrature are complementary material. Replace the displayed
form and its data to define a new problem while keeping the local/global contracts.

Field panels retain the macro mesh. Primal scalar fluxes are evaluated as
`-K grad(p)` and are identified separately from an H(div) reconstruction.
Vector examples distinguish velocity or displacement from interface multipliers
and declare the physical gauge. Algebraic diagnostics accompany the field
comparisons.

Each classical reference solves the same operator, material and boundary data.
The notebooks report errors against an analytical solution where available,
or differences between refined reference meshes. Parallel examples distinguish
their current numerical control from larger recorded or optional campaigns.
A numerical reference remains an approximation. Analytical solutions are
identified separately. Measured convergence rates
refer to the displayed refinement sequence and are not transferred from smooth
theorems to unresolved layers or discontinuous materials.

The notebooks cite the literature for their formulations and analytical data.
An introductory application or an analytical benchmark on the notebook's meshes
is identified accordingly; it is not a reproduction of a published figure.

The performance notebook compares 40,000 classical Q1 quadrilaterals with a
10×10 MHM macro mesh containing 20×20 Q1 quadrilaterals per macroelement.
The larger comparison uses a 500×500 classical mesh and the same total of
250,000 local quadrilaterals in MHM, with a wider worker-count sweep.
The size study also compares one million fine quadrilaterals, including a
classical Q1 solve with an AMG-preconditioned conjugate-gradient method. The
skeletal partition is refined in that comparison to control its contribution
to the field error. Geometry shared by local problems is prepared once per
run, and its preparation remains part of the setup time.
Equal element counts specify the amount of fine geometry, rather than equal
global spaces or identical field accuracy. Both field errors and timings are
reported. Strong scaling holds this problem fixed; weak scaling keeps local
work per worker fixed on a growing domain. Timings include the complete declared
solve pipeline and report the worker model and available hardware. Run this
notebook when other computations are idle to obtain useful performance data.

The process companion declares the same oscillatory problem in its cells.
Its displayed worker definitions live in versioned importable modules for
cross-platform `spawn`, without selecting a predefined physical case.
Strong scaling holds one million fine quadrilaterals fixed. Weak scaling keeps
40,000 fine quadrilaterals and 100 macroelements per process on integer-length
rectangles, preserving the coefficient's physical period. Classical LU and AMG
solve the same problem on each domain. A 500×500 comparison provides an
additional problem-size point. Each configuration uses one untimed warmup and
three measured repetitions, including startup, transfers and shutdown.

The performance study compiles and verifies the UFL forms before timing. Both
timed solvers use prepared Basix operators. Each configuration is warmed before
repeated samples; solve times include per-run setup, executor startup, assembly,
synchronization, the global solve and reconstruction. Observed native
initialization with the current compiler cache is recorded separately and is
not presented as a cold-cache measurement.

The three-dimensional tutorial declares its material and local forms before
introducing native workspace reuse. Every macrocell assembles its material
matrix and load and builds its own local factors or AMG hierarchy. Compatible
mesh/space/form resources and this application's material-independent face and
volume-moment geometry can remain resident in each isolated worker. Local LU
and AMG are compared with independent conforming references; the distributed
LU reference uses PETSc/MUMPS. The aligned period 0.1 and nonaligned period
0.137 define separate physical cases. Equal fine-cell counts do not establish
equal field accuracy.

The optional one/two-GPU section describes local condensation of original
assembled operators. Its native small controls establish availability, while
no accepted large GPU condensation or complete GPU-workflow speedup is
available. GPU setup, transfer, synchronization and all physical checks remain
part of that separately scoped experiment.

## Custom spaces and manual control

The advanced [custom-interface notebook](../foundations/operators/custom_interface.ipynb)
defines its own numbering, dense nonorthogonal face basis and orientation maps.
It compares transformed operators and physical fields with canonical coordinates.
`LocalEquations` and `MultiscaleProblem` remain available for complete manual
control, sharing the same assembly and solver owners as the introductory path.
