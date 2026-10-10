# External and custom local providers

A provider is an ordinary callable. It connects your local formulation or
simulation tool to PyMHM's checked condensation and global assembly. Choose the
interface according to what your external tool already produces; a provider
need not inherit from a framework class or select a built-in physical model.

| What the tool supplies | Interface to use |
| --- | --- |
| Weak forms on a local mesh | A `LocalContext` callback returning `local.equations(...)` |
| Assembled local matrices and trace maps | Explicit `LocalEquations` and `MultiscaleProblem` |
| A solve of an augmented local matrix | `SolverConfig(local_solver=callable)` |
| A complete nested multiscale problem | A local `MultiscaleProblem` with declared restrictions |

The [UFL guide](../fenics.md) covers weak-form providers. This page explains the
numerical contracts needed for another FEM library, a simulator or a learned
local solver.

## Return the local and global pairings explicitly

The four declared blocks mean

$$
\begin{aligned}
A_Ku_K+B_K\lambda_K&=f_K,\\
C_Ku_K+D_K\lambda_K&=g_K.
\end{aligned}
$$

Rows are test coordinates, columns are trial coordinates. `B` and `C` are
independent mathematical pairings; their signs are not inferred by transposition.
Convert your external tool's numbering, units and basis into one declared
coefficient contract before returning the data:

```python
from pymhm import LocalEquations

forms = LocalEquations(
    a=local_matrix,
    L=local_load,
    b=trial_trace_pairing,
    c=test_trace_pairing,
    d=direct_trace_matrix,
    g=direct_trace_load,
    dofs=global_trial_indices,
    test_dofs=global_test_indices,
    kernel=local_kernel_coefficients,
    moments=physical_moment_columns,
    metadata=reconstruction_data,
)
```

The symbolic names above are arrays supplied by your adapter, in its executed
local basis. An invertible local operator can omit `kernel` and `moments`.
A singular Neumann operator needs its actual left/right kernels and physical
constraints; retaining a nearly null mode uses `coarse_basis` rather than
claiming it is an exact kernel. Consult the [variational reference](../variational.md)
for distinct trial/test bases and moment layouts.

Return these records from `provider(cell)` and connect them to your explicit
global layout through `MultiscaleProblem`. If your provider instead receives
`LocalContext`, use `local.equations` so the bound interface supplies geometric
transport. Prepared blocks whose transport is already included must declare
`coordinates="global"`.

## Replace only the local numerical solver

To keep your forms and use an external linear solver, pass a callable:

```python
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from pymhm import SolverConfig, assemble


def dense_response(matrix: spmatrix, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve all augmented local RHS columns in their declared basis."""
    return np.linalg.solve(matrix.toarray(), rhs)


system = assemble(problem, solvers=SolverConfig(local_solver=dense_response))
solution = system.solve()
```

The callable receives owned copies of the augmented sparse matrix and **all**
source, trace and retained-mode RHS columns. Return a finite real array with
exactly the same shape as `rhs`, including physical-moment multipliers. A solver
that returns only the pressure entries does not implement this contract.
PyMHM checks the untouched original augmented equations independently for each
column at the existing relative criterion. The augmented matrix also undergoes
its numerical rank check. These checks do not establish discretization accuracy
or a mathematical inf-sup estimate.

For another library, replace the dense solve with that library's factorization
and multiple-RHS routine. Map its result back to the declared basis before
returning and release native factor resources. A different coefficient matrix
needs its own setup; sharing a compiled form kernel does not make two material
operators interchangeable.

## Connect a learned local response

A trained model can implement the same callable contract when its inputs encode
the complete local operator, coefficient basis, boundary data and constraints.
For example, an adapter can pass matrix and RHS arrays to a user-supplied model:

```python
prediction = model(matrix, rhs)
response_columns = np.asarray(prediction, dtype=np.float64)
```

Wrap those steps in the solver callable above and return `response_columns`.
`model` is your trained callable, not a built-in PyMHM object. Its coefficients
must satisfy the same original equations and moment constraints as a numerical
solver; approximate predictions outside that budget are rejected. A coarse
physical error or an algebraic residual alone does not validate extrapolation
to unseen materials. PyMHM supplies the interface and checks, not model training
or a claim of verified accuracy for an arbitrary learned solver.

## Keep worker resources explicit

Process providers and solvers must be importable and picklable. Construct
native meshes, factors, device arrays and model sessions within their owning
worker. Return portable coefficient arrays and reconstruction metadata.
An optional idempotent `prepare_runtime()` hook loads native libraries before
thread limits are applied; `close()` releases reusable worker state. Thread
providers must separate mutable workspaces across concurrent calls.
See [CPU execution](cpu.md) and [execution details](../execution.md).

For a custom face basis, arbitrary numbering or external trace conventions,
follow [custom interfaces](custom-interface.md). Persist the executed basis
matrix, its digest and orientation maps alongside solution coefficients so
replayed fields use the same representation.

## Continue with a complete workflow

The [API overview](../tutorials/overview.md) defines a bound problem from meshes
through postprocessing. The
[provider notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/local_global_providers.ipynb)
compares explicit primal and mixed provider declarations; the
[UFL notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/ufl_provider.ipynb)
uses native weak forms. Their companion modules are user-written example code,
separate from the installed package.

## References

- Christopher Harder and Frédéric Valentin (2016). *Foundations of the MHM Method*.
  [DOI: 10.1007/978-3-319-41640-3_13](https://doi.org/10.1007/978-3-319-41640-3_13).
