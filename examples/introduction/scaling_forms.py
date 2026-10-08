"""Declared two-dimensional Darcy forms shared by thread and spawn tutorials.

The physical flux is -K grad(p). Local Q1 pressures have a declared constant
kernel and physical-volume moments; the normal multiplier uses continuous P1
segments. These are public equation declarations, not calls to a PDE solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl

from pymhm import LocalContext
from pymhm.core.equations import LocalEquations, columns
from pymhm.fem.scalar.quadrilateral import quadrilateral_operators, quadrilateral_trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.cartesian import CartesianMacroMesh

EPSILON = 0.1


@dataclass(frozen=True)
class PeriodicDarcyData:
    """Declare a positive material period for coordinates with final axis (x,y).

    Pressure is sin(pi*x) sin(pi*y), physical flux is -K grad(p), and f is
    differentiated analytically. Integer rectangular lengths retain zero data.
    """

    period: float = 0.1

    def __post_init__(self) -> None:
        """Reject nonfinite and nonpositive material periods."""
        if not np.isfinite(self.period) or self.period <= 0:
            raise ValueError("period must be finite and positive")

    def permeability(self, points: np.ndarray) -> np.ndarray:
        """Return scalar isotropic permeability at points with final axis (x,y)."""
        x, y = points[..., 0], points[..., 1]
        frequency = 2 * np.pi / self.period
        return np.exp(np.sin(frequency * x) * np.sin(frequency * y))

    def source(self, points: np.ndarray) -> np.ndarray:
        """Evaluate f=-div(K grad(p_exact)) from independent analytic derivatives."""
        x, y = points[..., 0], points[..., 1]
        frequency = 2 * np.pi / self.period
        material = self.permeability(points)
        grad_material = (
            frequency
            * material[..., None]
            * np.stack(
                (
                    np.cos(frequency * x) * np.sin(frequency * y),
                    np.sin(frequency * x) * np.cos(frequency * y),
                ),
                axis=-1,
            )
        )
        return 2 * np.pi**2 * material * exact_pressure(points) - np.sum(
            grad_material * exact_gradient(points), axis=-1
        )


DEFAULT_DATA = PeriodicDarcyData()


def permeability(points: np.ndarray) -> np.ndarray:
    """Evaluate the fixed period-0.1 material used by complete CPU campaigns."""
    return DEFAULT_DATA.permeability(points)


def exact_pressure(points: np.ndarray) -> np.ndarray:
    """Return the manufactured pressure with homogeneous boundary data."""
    return np.sin(np.pi * points[..., 0]) * np.sin(np.pi * points[..., 1])


def exact_gradient(points: np.ndarray) -> np.ndarray:
    """Return the two physical derivatives of manufactured pressure."""
    x, y = points[..., 0], points[..., 1]
    return np.pi * np.stack(
        (np.cos(np.pi * x) * np.sin(np.pi * y), np.sin(np.pi * x) * np.cos(np.pi * y)), axis=-1
    )


def source(points: np.ndarray) -> np.ndarray:
    """Evaluate the fixed period-0.1 forcing used by complete CPU campaigns."""
    return DEFAULT_DATA.source(points)


def define_ufl_local_equations(
    local: LocalContext, data: PeriodicDarcyData = DEFAULT_DATA
) -> LocalEquations:
    """Write local volume and interface forms without numbering or orientation code."""
    binding = local.native_space(degree=1)
    domain, V = binding.mesh, binding.space
    p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x, y = ufl.SpatialCoordinate(domain)
    frequency = 2 * np.pi / data.period
    K = ufl.exp(ufl.sin(frequency * x) * ufl.sin(frequency * y))
    p_exact = ufl.sin(np.pi * x) * ufl.sin(np.pi * y)
    grad_K = (
        frequency
        * K
        * ufl.as_vector(
            (
                ufl.cos(frequency * x) * ufl.sin(frequency * y),
                ufl.sin(frequency * x) * ufl.cos(frequency * y),
            )
        )
    )
    grad_exact = np.pi * ufl.as_vector(
        (ufl.cos(np.pi * x) * ufl.sin(np.pi * y), ufl.sin(np.pi * x) * ufl.cos(np.pi * y))
    )
    f = 2 * np.pi**2 * K * p_exact - ufl.dot(grad_K, grad_exact)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})
    a = K * ufl.inner(ufl.grad(p), ufl.grad(v)) * dx
    L = f * v * dx

    # Write both mathematical pairings explicitly. The interface adapter supplies
    # their basis, geometric support and outward-normal transport.
    b = local.trace_pairings(lambda phi, ds: phi * v * ds)
    c = local.trace_pairings(lambda phi, ds: phi * p * ds, axis="rows")
    return local.equations(
        a=a,
        L=L,
        b=b,
        c=c,
        kernel=np.ones((len(binding.mapping), 1)),
        moments=columns(v * dx),
        metadata={"native_to_cartesian": np.argsort(binding.mapping)},
    )


def face_length_snapshot(macro: CartesianMacroMesh) -> np.ndarray:
    """Compute immutable physical face lengths once for a declared mesh."""
    lengths = macro.lengths
    lengths.setflags(write=False)
    return lengths


def translated_interface(
    template: np.ndarray,
    template_ends: np.ndarray,
    macro: CartesianMacroMesh,
    cell: int,
    face_lengths: np.ndarray,
    face_space: FaceSpace,
) -> np.ndarray:
    """Transport uniform continuous P1 face pairings by length, orientation and sign.

    Within each size, local Cartesian grids share their declared nodal order.
    The supplied FaceSpace determines each side's column count. Reversing a
    uniform nodal-P1 parameter reverses those columns. Face lengths are an
    immutable per-run snapshot; material, source and local factors are never
    reused. Each worker receives an owned coupling matrix.
    """
    if not face_space.continuous or any(p != 1 for p in face_space.degrees):
        raise ValueError("This translation helper requires continuous piecewise P1 traces")
    if not np.allclose(
        face_space.breaks, np.linspace(0, 1, len(face_space.breaks)), rtol=0, atol=1e-14
    ):
        raise ValueError("This translation helper requires uniform face segments")
    width = face_space.size
    assert template.shape[1] == 4 * width
    coupling = np.empty_like(template)
    for side, face in enumerate(macro.cell_faces[cell]):
        tangent = macro.points[macro.faces[face, 1]] - macro.points[macro.faces[face, 0]]
        reference_tangent = template_ends[side, 1] - template_ends[side, 0]
        columns = slice(width * side, width * (side + 1))
        block = template[:, columns]
        if tangent @ reference_tangent < 0:
            block = block[:, ::-1]
        coupling[:, columns] = (
            macro.signs[cell, side] * face_lengths[face] / np.linalg.norm(reference_tangent)
        ) * block
    return coupling


def refined_interface_template(
    spacing: np.ndarray,
    refinement: int,
    face_space: FaceSpace,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare the actual local-grid pairing with the supplied trace partition."""
    macro = CartesianMacroMesh(1, 1, (0, float(spacing[0]), 0, float(spacing[1])))
    skeleton = SkeletonSpace(macro, tuple(face_space for _ in macro.faces))
    template = quadrilateral_trace_coupling(macro, 0, macro.submesh(0, refinement), skeleton, 1)
    ends = macro.points[macro.faces[macro.cell_faces[0]]]
    template.setflags(write=False)
    ends.setflags(write=False)
    return template, ends


def interface_template(spacing: np.ndarray, face_space: FaceSpace) -> tuple[np.ndarray, np.ndarray]:
    """Prepare the introductory local20 pairing for its declared trace space."""
    return refined_interface_template(spacing, 20, face_space)


@dataclass(frozen=True)
class LocalProvider:
    """Supply the verified weak forms through prepared portable assembly conveniences."""

    macro: CartesianMacroMesh
    skeleton: SkeletonSpace
    coupling_template: np.ndarray
    template_endpoints: np.ndarray
    face_lengths: np.ndarray
    face_space: FaceSpace
    refinement: int = 20
    quadrature_order: int = 4
    data: PeriodicDarcyData = DEFAULT_DATA

    def local_mesh(self, cell: int) -> CartesianMacroMesh:
        """Describe each fine mesh so it is created within its owning worker."""
        return self.macro.submesh(cell, self.refinement)

    def __call__(self, local: LocalContext) -> LocalEquations:
        """Declare A p+B lambda=f and C=B.T with the physical volume mean."""
        cell, fine = local.cell, local.mesh
        A, mass, f = quadrilateral_operators(
            fine,
            1,
            permeability=self.data.permeability,
            source=self.data.source,
            order=self.quadrature_order,
        )
        B = translated_interface(
            self.coupling_template,
            self.template_endpoints,
            self.macro,
            cell,
            self.face_lengths,
            self.face_space,
        )
        constant = np.ones((len(f), 1))
        return local.equations(
            a=A,
            L=f,
            b=B,
            c=B.T,
            coordinates="global",
            kernel=constant,
            moments=mass @ constant,
            metadata={"mesh": fine},
        )


def verify_source(data: PeriodicDarcyData = DEFAULT_DATA) -> None:
    """Check f=-div(K grad(p)) independently at interior physical points."""
    probe = np.array([[0.237, 0.419], [0.613, 0.728], [0.832, 0.147]])

    step = 2e-6

    divergence = np.zeros(len(probe))

    for axis in range(2):
        shift = np.eye(2)[axis] * step
        plus = data.permeability(probe + shift) * exact_gradient(probe + shift)[:, axis]
        minus = data.permeability(probe - shift) * exact_gradient(probe - shift)[:, axis]
        divergence += (plus - minus) / (2 * step)

    np.testing.assert_allclose(data.source(probe), -divergence, rtol=2e-8, atol=2e-8)
