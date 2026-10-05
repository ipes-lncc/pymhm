# Multiscale Darcy: thread execution

[The introductory notebook](../../../../notebooks/introduction/darcy_parallel_scalability.ipynb)
is the reproducible recipe, using the locked Pixi `introduction` environment.
These records contain 54 strong, 18 million-element crossover and 21 weak
samples for an analytical Darcy problem.

$$
K(x,y)=\exp\!\left[\sin(2\pi x/0.1)\sin(2\pi y/0.1)\right],
\qquad p_\star(x,y)=\sin(\pi x)\sin(\pi y).
$$

Both methods use $f=-\nabla\cdot(K\nabla p_\star)$ and zero exterior pressure.
Classical conforming Q1 and MHM have the same total quadrilateral count.
MHM uses 10 × 10 macroelements, one retained constant per macroelement,
and continuous P1 traces within each independent macroface.

| Classical grid | Fine elements, either method | Local MHM grid | Trace segments per face |
| --- | ---: | --- | ---: |
| 200 × 200 | 40,000 | 20 × 20 | 4 |
| 500 × 500 | 250,000 | 50 × 50 | 4 |
| 1000 × 1000 | 1,000,000 | 100 × 100 | 8 |

The 1000 case refines both the local grid and trace partition. The Darcy
flux is $\boldsymbol q=-K\nabla p$; macro conservation is distinguished
from fine-cell H(div) conservation.

## Measurement procedure

Executed UFL forms verify the Basix/SciPy operators used in solve timings.
Native initialization/JIT is separate: local50/local100 observations were
0.056/0.150 s with an existing compiler cache, so they are not cold-cache
measurements. Classical/thread configurations have one recorded warm-up;
serial MHM paths are warmed by the earlier scientific validation runs.
Each configuration then has three fresh solves in seeded, randomized order.

Complete timers include mesh/provider setup, geometry snapshots and trace
templates, executor startup, dispatch/transfer, local assembly and
factorization, worker Schur contributions, synchronization, ordered global
assembly, global solve and full pressure reconstruction. Physical operators
and factors are rebuilt every run. Only geometric trace transport is reused
within that run, with setup included. Imports, JIT, warm-ups, field diagnostics,
error integration, plots and archives are excluded equally. Package original
residual checks remain inside solves. Raw samples and observed ranges are
retained.

The Linux Xeon Silver 4216 host has 64 logical CPUs / 32 physical cores in
affinity; recorded cgroup controls show no quota. Numerical libraries use
one thread per worker. MHM counts are 1/2/4/8/16/32/64 (1000: 1/4/8/16).
Other project numerical jobs were paused, but the operating system was not
exclusively reserved; load snapshots are recorded. Classical solves use one
numerical thread; a distributed classical comparison is outside this study.

## Results

| Grid | Classical baseline | Classical median [s] | Best MHM threads / median [s] | Classical / MHM |
| --- | --- | ---: | --- | ---: |
| 200 × 200 | LU | 1.339 | 1 / 2.358 | 0.568 |
| 500 × 500 | LU | 8.484 | 8 / 9.468 | 0.896 |
| 1000 × 1000 | LU | 48.290 | 8 / 32.951 | 1.466 |
| 1000 × 1000 | AMG-preconditioned CG | 20.813 | 8 / 32.951 | 0.632 |

Threads show no gain against classical 200/500, and beat LU while losing
to AMG at 1000. MHM thread1-to-thread8 speedup at 1000 is 2.125.
Analytical MHM/classical error ratios are 1.002511 for pressure and 1.000047
for vector Darcy flux, with original global residual $1.86\times10^{-16}$.

Weak scaling keeps 40,000 fine elements per worker on
$(0,p)\times(0,1)$, a 10$p$ × 10 macrogrid and 20 × 20 locals.
Median time rises from 2.326 s at one worker to 292.479 s at 64
(2.56 million elements; range 290.518–294.469 s).
Weak efficiency, one-worker time divided by $p$-worker time, is 0.795% at 64:
poor measured thread scaling. Physical errors per square-root domain area
and residuals are recorded at every count.

## Provenance and files

```sh
pixi run --locked -e introduction python scripts/run_notebooks.py notebooks/introduction/darcy_parallel_scalability.ipynb --timeout 3600
```

The two JSON records and six figures are byte-identical acquisition copies;
`SHA256SUMS` identifies every payload. JSON includes package-source hashes,
versions, affinity, seeds, timings, physical errors and basis/archive digests.
The notebook regenerates coefficient and executed-basis matrices, verifies
replay with BLAS1/2, and checks classical refinement, quadrature, physical
mean constraints and all four trace orientations. Replay requires the
recorded basis matrix as part of the coefficient contract.

- Notebook SHA256: `fc90ba07181ac4b2a156796473d1ffb78bdcf1081f9527b0ee5586751fbe4b18`.
- Lockfile SHA256: `c54e433e408e8345516023d96526b305d09452a55d53c018ea6ea529535df9fd`.

Figures show reference refinement, 200/500/1000 pressure and Darcy flux,
strong/weak scaling, and cost/accuracy crossover. Spatial panels include
the actual macro mesh and independent one-sided reconstructed values.

Gomes, Pereira, Valentin and Paredes,
[*On the Implementation of a Scalable Simulator for Multiscale Hybrid-Mixed
Methods*](https://arxiv.org/abs/1703.10435v1), motivate the scalability discussion.
This analytical benchmark does not reproduce their discretization, data or
hardware.
