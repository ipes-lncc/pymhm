# Gallery

Explore the same numerical cases by their **physical problem** or their
**spatial dimension**. Each link leads to the canonical case page, its figures,
the executed approximation spaces and reproducible numerical records.

<div class="grid cards" markdown>

-   **Browse by problem**

    Darcy flow, incompressible flow, scalar transport, solid mechanics and waves.

    [Choose a physical problem](#physical-problems)

-   **Browse by dimension**

    Find planar cases or volume problems without changing the physical model.

    [Two dimensions](two-dimensional.md) · [Three dimensions](three-dimensional.md)

-   **Learn to build a formulation**

    Follow the mesh → local equations → global equations → assembly → solve workflow.

    [Start with the API overview](../tutorials/overview.md)

-   **Choose an execution environment**

    Run the same heterogeneous Darcy problem with CPU workers, MPI or CUDA.

    [Execution guides](../guides/index.md)

</div>

## Physical problems

| Problem | Cases and fields |
| --- | --- |
| [Darcy](darcy.md) | Heterogeneous permeability, SPE10, mixed H(div), multilevel methods, wells and CPU/GPU scaling |
| [Stokes–Brinkman and Oseen](flow.md) | Velocity, physical pressure, incompressibility and boundary layers |
| [Steady transport](transport.md) | Advection–reaction–diffusion and stabilized boundary layers |
| [Transient transport](transient-transport.md) | Concentration and time-step refinement |
| [Elasticity](elasticity.md) | Displacement, physical Cauchy stress, weak rotation and nearly incompressible materials |
| [Elastodynamics](elastodynamics.md) | Displacement, velocity, stress and time-step refinement |
| [Helmholtz and acoustics](acoustics.md) | Analytical plane wave and explicitly cropped Marmousi II pilot |
| [Maxwell](maxwell.md) | Electric and magnetic fields with an independent DG device control |

## Read the evidence

Analytical cases measure each physical field against an independently defined
exact solution. A numerical reference has its own refinement check. A
successive-solution increment is labelled as an increment; it is not an exact
error or a certified reference. Short refinement series remain initial studies
unless their stated approximation hypotheses and asymptotic rates have been
verified.

The [numerical evidence guide](../cases/index.md) explains these conventions and
the [current convergence catalogue](../cases/minimal-convergence.md) records
all 31 accepted initial study groups. The larger performance campaigns retain
setup, transfer, synchronization, reconstruction and solver costs.

The original publications are cited on their case pages and in the
[bibliography](../literature.md). Analytical verification on a newly constructed
mesh is distinguished from a matched reproduction of a publication.
