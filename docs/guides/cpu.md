# Serial, threads and processes

Choose the execution policy after defining `problem`. These settings change
how local work is scheduled; the local equations, materials, moments and
interface remain the same. The [API overview](../tutorials/overview.md) supplies
a complete problem definition.

## Run sequentially

```python
from pymhm import ExecutionConfig, assemble

system = assemble(problem, execution=ExecutionConfig(backend="serial"))
solution = system.solve()
```

The coordinator constructs, condenses and adds one macrocell before requesting
the next. Use this policy for initial correctness checks and small problems.

## Use independent workers

```python
policy = ExecutionConfig(
    backend="process",
    workers=8,
    batch_size=16,
    pipeline=True,
    native_threads=1,
)
system = assemble(problem, execution=policy)
solution = system.solve()
```

| Option | Effect |
| --- | --- |
| `backend="process"` | Independent spawned Python processes |
| `backend="thread"` | Threads within one process |
| `workers` | Maximum local workers; respect the scheduler's CPU allocation |
| `batch_size` | Maximum submitted work window; `None` uses the worker count |
| `pipeline=True` | Refill the ordered window as contributions are consumed |
| `native_threads=1` | Limit each worker's supported BLAS/OpenMP pools |

Workers compute reduced local blocks. The coordinator adds their shared-face
entries in input order; neighboring cells do not write concurrently to the global
matrix. Increasing workers beyond the number of macrocells exposes no additional
local parallel work.

Thread providers must keep concurrent mutable native workspaces independent.
Gmsh uses process-global state and must be serialized within a process. Numba's
compiled numerical kernels release the GIL, but that alone does not make every
external provider thread safe.

## Make process callbacks importable

Processes use `spawn` on every platform. Put providers, mesh factories and
coefficient callables in an importable Python module. Construct native meshes,
forms and solver resources inside the worker; return portable numerical data.
Protect the launch in a script:

```python
from pymhm import ExecutionConfig, assemble
from my_formulation import make_problem

if __name__ == "__main__":
    problem = make_problem()
    system = assemble(problem, execution=ExecutionConfig(backend="process", workers=8))
    solution = system.solve()
```

`my_formulation` is your module containing the mathematical definition, not a
PyMHM model module. Notebook-defined closures are suitable for serial execution;
use an importable companion module when selecting spawned workers.

## Select the numerical solver independently

```python
from pymhm import SolverConfig

system = assemble(
    problem,
    execution=policy,
    solvers=SolverConfig(local_solver="pypardiso", global_solver="scipy"),
)
```

This PARDISO choice requires its installed native runtime. Mean-constrained
Neumann locals are saddle systems, so an SPD-only solver cannot be selected
without the appropriate projected formulation. See [linear solvers](../solvers.md).

For reusable worker resources, implement `prepare_runtime()` and `close()`
as described in [execution and repeated solves](../execution.md).
For timing, warm the actual forms and kernels first, then include process startup,
serialization, reduction, global solution and reconstruction in the complete
clock. The [Gallery](../gallery/index.md) contains measured parallel applications.
