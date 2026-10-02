"""Analytic incompressible three-dimensional fields for Stokes, Brinkman and Oseen."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Flow3DData:
    """Smooth manufactured velocity/pressure on the unit cube with fixed physical units.

    u=(sin(pi*z)+cos(pi*y), sin(pi*x)+cos(pi*z), sin(pi*y)+cos(pi*x)) is
    divergence-free. Its Laplacian is -pi^2*u. Pressure is
    (x-1/2)(y-1/2)+(z-1/2), with exactly zero volume mean.
    """

    kind: str = "stokes"

    def __post_init__(self) -> None:
        """Reject undefined material/transport families."""
        if self.kind not in ("stokes", "brinkman", "oseen"):
            raise ValueError("kind must be stokes, brinkman or oseen")

    @property
    def viscosity(self) -> float:
        """Return the constant physical viscosity for the selected family."""
        return {"stokes": 1.0, "brinkman": 0.2, "oseen": 0.1}[self.kind]

    def velocity(self, points: np.ndarray) -> np.ndarray:
        """Evaluate all three incompressible velocity components."""
        sine, cosine = np.sin(np.pi * points), np.cos(np.pi * points)
        return np.column_stack(
            (sine[:, 2] + cosine[:, 1], sine[:, 0] + cosine[:, 2], sine[:, 1] + cosine[:, 0])
        )

    def gradient(self, points: np.ndarray) -> np.ndarray:
        """Differentiate velocity analytically, with axes (point, component, derivative)."""
        sine, cosine = np.sin(np.pi * points), np.cos(np.pi * points)
        result = np.zeros((len(points), 3, 3))
        result[:, 0, 1], result[:, 0, 2] = -np.pi * sine[:, 1], np.pi * cosine[:, 2]
        result[:, 1, 0], result[:, 1, 2] = np.pi * cosine[:, 0], -np.pi * sine[:, 2]
        result[:, 2, 0], result[:, 2, 1] = -np.pi * sine[:, 0], np.pi * cosine[:, 1]
        return result

    def pressure(self, points: np.ndarray) -> np.ndarray:
        """Evaluate zero-mean pressure, including a quadratic mixed derivative."""
        return (points[:, 0] - 0.5) * (points[:, 1] - 0.5) + points[:, 2] - 0.5

    def pressure_gradient(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the exact gradient of the prescribed pressure."""
        return np.column_stack((points[:, 1] - 0.5, points[:, 0] - 0.5, np.ones(len(points))))

    def resistance(self, points: np.ndarray) -> np.ndarray:
        """Return zero, a variable SPD tensor, or a constant Oseen resistance."""
        if self.kind == "stokes":
            return np.zeros((len(points), 3, 3))
        if self.kind == "oseen":
            return np.broadcast_to(2 * np.eye(3), (len(points), 3, 3))
        tensor = np.array([[2.0, 0.2, 0.1], [0.2, 3.0, 0.4], [0.1, 0.4, 4.0]])
        return (1 + 0.5 * points[:, 0] + 0.25 * points[:, 2])[:, None, None] * tensor

    def advection(self, points: np.ndarray) -> np.ndarray:
        """Return affine Oseen convection with divergence three, or zero."""
        return points + [1.0, 0.5, 0.25] if self.kind == "oseen" else np.zeros_like(points)

    def source(self, points: np.ndarray) -> np.ndarray:
        """Evaluate -nu*Delta(u)+beta.grad(u)+Gamma*u+grad(p) directly."""
        velocity = self.velocity(points)
        return (
            self.viscosity * np.pi**2 * velocity
            + np.einsum("nij,nj->ni", self.gradient(points), self.advection(points))
            + np.einsum("nij,nj->ni", self.resistance(points), velocity)
            + self.pressure_gradient(points)
        )

    def options(self) -> dict:
        """Return picklable physical inputs without choosing finite-element spaces."""
        result = dict(viscosity=self.viscosity, source=self.source, dirichlet=self.velocity)
        if self.kind == "brinkman":
            result["drag"] = self.resistance
        elif self.kind == "oseen":
            result.update(
                drag=2.0, advection=self.advection, advection_divergence=3.0, advection_bound=3.0
            )
        return result
