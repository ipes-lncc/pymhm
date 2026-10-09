# Recursive local–global problems

[Open notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/variational_hierarchy.ipynb) · [Download](https://raw.githubusercontent.com/ipes-lncc/pymhm/main/notebooks/foundations/operators/variational_hierarchy.ipynb)

An operator at one scale can be another `MultiscaleProblem`. This small
coefficient example declares every A, B, C and D block. It verifies recursive
field reconstruction against a separately written full matrix; it is an
algebraic example, not a convergence study for a physical PDE.

At the leaf, $u=(u_1,u_2)$ and the trace is $\xi$. Middle and top traces
are $\eta$ and $\lambda$. Their equations are

$$
\begin{aligned}
\begin{pmatrix}2&-1\\-1&2\end{pmatrix}u
  +\begin{pmatrix}1\\0\end{pmatrix}\xi &= \begin{pmatrix}1\\3\end{pmatrix},\\
-u_1+3\xi+2\eta &= 2,\\
-3\xi+4\eta+\lambda &= 1,\\
-2\eta+6\lambda &= 3.
\end{aligned}
$$

No symmetry relation between B and C is assumed. Intermediate operators are
the actually assembled finer systems; the final solution retains their tree.


The PyMHM distribution contains only the library. The first cell explicitly
downloads a checksum-verified companion archive and prepares the declared data
using its local Python helpers. You can inspect these support files in `ROOT`.
Complete studies and historical replay retain their existing opt-in flags.


```python
from pathlib import Path
import os
import sys
from pymhm.io.workspace import workspace_from_archive

# Download verified support files; this operation does not execute them.
COMPANION_URL = "https://ipes-lncc.github.io/pymhm/downloads/3c8b96fb992c085f07624e9ab588746ea6e324f186886e0d414e2b7e7f071866/variational_hierarchy-companion.zip"
COMPANION_SHA256 = "3c8b96fb992c085f07624e9ab588746ea6e324f186886e0d414e2b7e7f071866"
WORKSPACE = Path(
    os.environ.get("PYMHM_WORKSPACE", Path.cwd() / ".pymhm-companions" / COMPANION_SHA256)
)
ROOT = workspace_from_archive(COMPANION_URL, sha256=COMPANION_SHA256, directory=WORKSPACE)
os.environ["PYMHM_WORKSPACE"] = str(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Explicitly prepare declared data with the downloaded Python helpers.
from scripts.notebook_reproduction import notebook_workspace

ROOT = notebook_workspace("foundations/operators/variational_hierarchy.ipynb", directory=ROOT)
root = ROOT
print("Workspace:", ROOT)

import numpy as np

```

```python
from pymhm import Equation, LocalEquations, MultiscaleProblem, assemble

def leaf_equations(item):
    """Supply the finest two coefficients and their single interface balance."""
    return LocalEquations([[2.0, -1.0], [-1.0, 2.0]], [1.0, 3.0],
                          [[1.0], [0.0]], [[-1.0, 0.0]], [0], d=[[3.0]], g=[2.0])

leaf = MultiscaleProblem(Equation(0, 0), leaf_equations, [0], 1, (0,))

def middle_equations(item):
    """Use the leaf's reduced operator as this level's local A block."""
    return LocalEquations(leaf, 0, [[2.0]], [[-3.0]], [0], d=[[4.0]], g=[1.0])

middle = MultiscaleProblem(Equation(0, 0), middle_equations, [0], 1, (0,))

def top_equations(item):
    """Use the middle system without flattening or recomputing its leaf basis."""
    return LocalEquations(middle, 0, [[1.0]], [[-2.0]], [0], d=[[6.0]], g=[3.0])

problem = MultiscaleProblem(Equation(0, 0), top_equations, [0], 1, (0,))
system = assemble(problem)
solution = system.solve()
child = solution.children[0]
finest = child.children[0]
recovered = np.r_[finest.fields[0], finest.trace, child.trace, solution.trace]

monolithic = np.array([[2., -1., 1., 0., 0.], [-1., 2., 0., 0., 0.],
                       [-1., 0., 3., 2., 0.], [0., 0., -3., 4., 1.],
                       [0., 0., 0., -2., 6.]])
rhs = np.array([1., 3., 2., 1., 3.])
reference = np.linalg.solve(monolithic, rhs)
np.testing.assert_allclose(recovered, reference, atol=2e-14, rtol=2e-14)
print({"reconstructed_coefficients": recovered.tolist(),
       "monolithic_equation_residual": float(np.linalg.norm(monolithic @ recovered - rhs)),
       "relative_residuals_by_level": [solution.raw_residual, child.raw_residual,
                                        finest.raw_residual]})

```

## Physical interpretation and acceptance

The executed algebraic example verifies nested condensation against the same full monolithic equations. It is a hierarchy contract test, not a spatial PDE convergence study. A physical recursive MHM provider declares a child `bind_problem` and returns `local.nested(child_problem)`. Automatic restriction currently covers supported planar normal-trace skeletons; custom interfaces supply an exact `boundary_dofs`/`trace_map` restriction. Child source, outer boundary treatment, physical mean/gauge and injectivity remain explicit. An unrepresentable parent trace is rejected rather than projected silently. Refinement estimates require the hypotheses at each scale.

The [custom-interface tutorial](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/custom_interface.ipynb) shows manual control when the user intentionally owns the coordinates.

## References

- Christopher Harder and Frédéric Valentin (2016). *Foundations of the MHM Method*, in *Building Bridges: Connections and Challenges in Modern Approaches to Numerical Partial Differential Equations*, Lecture Notes in Computational Science and Engineering 114, Springer. [DOI: 10.1007/978-3-319-41640-3_13](https://doi.org/10.1007/978-3-319-41640-3_13).
