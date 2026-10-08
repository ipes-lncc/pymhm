"""User-defined local equations compiled for streamed acquisition and original-row checks.

These small spawn-safe data records select the adjacent mathematical providers.
Their callbacks compile explicitly written A/B/C/D through the generic API; no
prepared method factory is invoked or hidden in an assembly wrapper.
"""

from dataclasses import dataclass
from typing import Any

from pymhm import compile_local_equations
from pymhm.core.contracts import LocalAssembly

from .cartesian_darcy import local_equations as cartesian_equations
from .darcy import local_equations as darcy_equations


@dataclass(frozen=True)
class DarcyLocalFactory:
    """Declare one local Darcy form and preserve the original streamed metadata layout."""

    mesh: Any
    skeleton: Any
    permeability: Any
    source: Any
    point_parts: Any
    local_refinement: int
    supplied_meshes: Any
    degree: int
    formulation: str
    quadrature_order: int
    element_backend: str = "basix"

    def __call__(self, cell: int) -> LocalAssembly:
        """Compile the explicit Neumann/normal-moment equations in their literal coordinates."""
        if self.element_backend not in ("basix", "portable"):
            raise ValueError("element_backend must be basix or portable")
        equation = darcy_equations(
            cell,
            mesh=self.mesh,
            skeleton=self.skeleton,
            permeability=self.permeability,
            source=self.source,
            point_parts=self.point_parts,
            degree=self.degree,
            formulation=self.formulation,
            local_refinement=self.local_refinement,
            local_meshes=self.supplied_meshes,
            quadrature_order=self.quadrature_order,
        )
        compiled = compile_local_equations(equation)
        return LocalAssembly(compiled.problem, compiled.metadata)


@dataclass(frozen=True)
class CartesianTask:
    """Literal local Cartesian Qk geometry, coefficients and numerical integration inputs."""

    mesh: Any
    cell: int
    refinement: Any
    skeleton: Any
    degree: int
    permeability: Any
    source: Any
    order: int


def assemble_cartesian_task(task: CartesianTask) -> LocalAssembly:
    """Compile the adjacent user-declared Qk equations for streamed local acquisition."""
    equations = cartesian_equations(
        task.cell,
        mesh=task.mesh,
        skeleton=task.skeleton,
        degree=task.degree,
        refinement=task.refinement,
        permeability=task.permeability,
        source=task.source,
        order=task.order,
    )
    compiled = compile_local_equations(equations)
    return LocalAssembly(compiled.problem, compiled.metadata)
