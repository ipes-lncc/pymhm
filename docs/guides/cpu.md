# Serial, threads and processes

Start with the [common heterogeneous Darcy problem](heterogeneous-darcy.md).
The local equations, fine meshes, normal-flux traces, physical moments and
global boundary data remain identical in every mode.

## 1. Run serially

```python
from pymhm import ExecutionConfig, assemble

system = assemble(problem, execution=ExecutionConfig(backend="serial"))
solution = system.solve()
```

The coordinator constructs and condenses one local problem at a time, then
adds its small reduced blocks to the global matrix. This is the reference
execution for coefficient and field comparisons.

## 2. Select independent CPU workers

```python
policy = ExecutionConfig(
    backend="process",
    workers=8,
    batch_size=16,
    pipeline=True,
    native_threads=1,
)
parallel_system = assemble(problem, execution=policy)
parallel_solution = parallel_system.solve()
```

`workers` is the available process count; `native_threads=1` bounds nested
BLAS/OpenMP work in each local solve. A larger rolling window lets the workers
continue while the coordinator reduces an earlier cell. The coordinator adds
shared-face contributions in deterministic cell order, so adjacent cells do
not race to write the global matrix.

Use `backend="thread"` with the same fields for a thread pool. Native FEM
assembly and mutable workspaces must be thread safe; a provider must not share
one mutable native mesh/space among concurrent calls. The common guide creates
the native resources inside each invocation. Gmsh uses process-global state
and must not be called concurrently by threads.

## 3. Keep spawn entry points importable

The package uses `spawn` on every platform. Define the provider and mesh
callables in an importable module. In a Python script, guard execution:

```python
from examples.guides.heterogeneous_execution import declared_problem, run_cpu

if __name__ == "__main__":
    problem = declared_problem(n=2, refinement=8)
    system, solution, report = run_cpu(problem, backend="process", workers=2)
    print(report)
```

Run the file inside the locked FEM/notebook environment:

```bash
pixi run --locked -e introduction python cpu_darcy.py
```

The notebook can use the importable provider directly from its kernel. Native
UFL/DOLFINx objects, PETSc factors and CUDA arrays stay in their owning worker;
only the compiled numerical equations and reconstruction data cross processes.

## 4. Choose a solver independently

```python
from pymhm import SolverConfig

system = assemble(
    problem,
    execution=policy,
    solvers=SolverConfig(local_solver="pypardiso", global_solver="scipy"),
)
solution = system.solve()
```

Use `introduction-intel` for that PARDISO example. The constrained Neumann
operator is a saddle system; select a compatible indefinite/general solver,
rather than an SPD-only method. See [solver requirements](../solvers.md).

## 5. Compare correctness before timing

Compare each macrocell's independent coefficients and physical errors against
serial execution, using the same meshes, coefficient/source quadrature and
error integration. The guide performs this check for two threads and two spawn
workers. Increasing workers beyond the number of macrocells cannot expose more
local parallel work in this four-cell example.

For performance measurements, exclude form-compilation cold starts through an
explicit warmup and report startup, serialization, global solution and
reconstruction costs. The [3D scaling campaign](../cases/darcy-3d-scalability.md)
provides measured strong speedup and efficiency curves; this small execution
guide makes no speedup claim. See [detailed execution policies](../execution.md)
for resource closure and failure propagation.
