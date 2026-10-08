"""Portable physical Piola fields with literal executed polynomial and moment matrices."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray, positive_int, real_array
from pymhm.fem.geometry import pullback_points
from pymhm.fem.hdiv.bdm_family import BDMFamily
from pymhm.fem.hdiv.family_3d import HDiv3DFamily
from pymhm.fem.hdiv.mapped import mapped_reference_coefficients, mapped_rt_dofs
from pymhm.fem.hdiv.moments_3d import tetrahedral_candidate_coefficients
from pymhm.fem.hdiv.rt import rt_dofs, rt_moment_coefficients
from pymhm.fem.hdiv.tensor_rt import tensor_rt_dofs, tensor_rt_moment_coefficients
from pymhm.fem.reference import (
    ReferenceElementSpec,
    bernstein_basis_matrix,
    create_reference_element,
    monomial_tabulation,
    tabulate_archived_basis,
)
from pymhm.meshes.mixed import hdiv3d_dofs, hdiv3d_transform
from pymhm.postprocessing.fields import FieldDefinition


def _immutable(value: Any, dtype: Any = float) -> Any:
    """Own an executed numeric map independently of a native cache or caller buffer."""
    if dtype is float:
        result = real_array(value, "executed basis map")
    else:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iu" or np.any(raw < 0):
            raise ValueError("executed DOF map must contain nonnegative integers")
        result = np.array(raw, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReferenceVectorBasis:
    """Executed reference values/divergence before orientation and physical Piola.

    Native mode archives Basix's coefficient_matrix and the actual moment map.
    Monomial mode archives explicitly ordered component powers and the actual
    moment matrix. Tensor Bernstein mode additionally retains its executed
    triangle and interval matrices for each vector component. Evaluation never
    recomputes a dual basis or arbitrarily oriented nullspace.
    """

    cell: str
    degree: int
    dimension: int
    matrix: Any
    transform: Any
    mode: str = "native"
    powers: tuple[Any, ...] = ()
    factors: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """Archive literal arrays through construction and pickle reconstruction."""
        object.__setattr__(self, "matrix", _immutable(self.matrix))
        object.__setattr__(self, "transform", _immutable(self.transform))
        object.__setattr__(self, "factors", tuple(_immutable(v) for v in self.factors))
        positive_int(self.degree, "basis degree", 0)
        positive_int(self.dimension, "basis dimension")
        if self.mode not in {"native", "monomial", "prism_bernstein"}:
            raise ValueError("reference vector basis mode must be declared explicitly")
        if self.matrix.ndim != 2 or self.transform.ndim != 2:
            raise ValueError("reference polynomial and moment maps must be matrices")
        if self.mode == "prism_bernstein" and (self.dimension != 3 or len(self.factors) != 6):
            raise ValueError("prism Bernstein mode requires three pairs of factor matrices")

    def __reduce__(self) -> Any:
        """Replay archived arrays instead of native interpolation and moment solves."""
        return type(self), (
            self.cell,
            self.degree,
            self.dimension,
            self.matrix,
            self.transform,
            self.mode,
            self.powers,
            self.factors,
        )

    @property
    def digest(self) -> str:
        """Identify every executed polynomial/moment matrix and its coordinate convention."""
        digest = sha256(
            repr((self.cell, self.degree, self.dimension, self.mode, self.powers)).encode()
        )
        for array in (self.matrix, self.transform, *self.factors):
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def __call__(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return (q,basis,reference_component) values and (q,basis) divergence."""
        if self.mode == "native":
            table = tabulate_archived_basis(
                self.cell, self.degree, self.matrix, points, components=self.dimension, nderiv=1
            )
            values = table[0]
            divergence = sum(table[a + 1, :, :, a] for a in range(self.dimension))
        elif self.mode == "monomial":
            table = monomial_tabulation(points, self.powers, nderiv=1)
            values = np.concatenate(
                [table[0, :, :, None] * np.eye(self.dimension)[a] for a in range(self.dimension)],
                axis=1,
            )
            divergence = np.concatenate([table[a + 1] for a in range(self.dimension)], axis=1)
        else:
            values_list, divergence_list = [], []
            for axis in range(3):
                triangle_degree = self.degree + int(axis < 2)
                interval_degree = self.degree + int(axis == 2)
                triangle = tabulate_archived_basis(
                    "triangle", triangle_degree, self.factors[2 * axis], points[:, :2], nderiv=1
                )
                interval = tabulate_archived_basis(
                    "interval", interval_degree, self.factors[2 * axis + 1], points[:, 2:], nderiv=1
                )
                value = np.einsum("qi,qj->qij", triangle[0], interval[0]).reshape(len(points), -1)
                derivative = np.einsum(
                    "qi,qj->qij",
                    triangle[axis + 1] if axis < 2 else triangle[0],
                    interval[0] if axis < 2 else interval[1],
                ).reshape(len(points), -1)
                values_list.append(value[..., None] * np.eye(3)[axis])
                divergence_list.append(derivative)
            values, divergence = (
                np.concatenate(values_list, axis=1),
                np.concatenate(divergence_list, axis=1),
            )
        if self.mode == "native":
            for factor in self.factors:
                values = np.einsum("qia,ij->qja", values, factor)
                divergence = divergence @ factor
        return np.einsum("qia,ij->qja", values, self.transform), divergence @ self.transform


@dataclass(frozen=True)
class _PiolaEvaluator:
    """Apply literal cell maps and contravariant Piola to an archived reference basis."""

    reference_basis: ReferenceVectorBasis
    dofs: Any
    orientation: Any
    components: int
    divergence: bool = False

    def __post_init__(self) -> None:
        """Own the executed topology and orientation independently of regenerated caches."""
        object.__setattr__(self, "dofs", _immutable(self.dofs, np.int64))
        object.__setattr__(self, "orientation", _immutable(self.orientation))

    def __reduce__(self) -> Any:
        """Retain the full basis contract across spawn and persisted-field replay."""
        return type(self), (
            self.reference_basis,
            self.dofs,
            self.orientation,
            self.components,
            self.divergence,
        )

    def __call__(self, mesh: Any, coefficients: Any, points: Any, *, cells: Any = None) -> Any:
        """Evaluate physical vectors/tensors or their physical divergence at one-sided points."""
        owners, reference, jacobian, determinant = pullback_points(mesh, points, cells=cells)
        coefficients = real_array(coefficients, "Piola coefficients")
        size = int(self.dofs.max()) + 1
        if coefficients.shape != (size * self.components,):
            raise ValueError("Piola coefficients must match the executed moment map")
        values, divergence = self.reference_basis(reference)
        local = coefficients.reshape(size, self.components)[self.dofs[owners]]
        orient = self.orientation[owners]
        if self.divergence:
            output = np.einsum("qi,qij,qjc->qc", divergence, orient, local) / determinant[:, None]
            return output[:, 0] if self.components == 1 else output
        values = (
            np.einsum("qab,qib,qij->qja", jacobian, values, orient) / determinant[:, None, None]
        )
        output = np.einsum("qia,qic->qca", values, local)
        return output[:, 0] if self.components == 1 else output


def piola_field(
    name: str,
    mesh: Any,
    *,
    reference_basis: ReferenceVectorBasis,
    dofs: Any,
    orientation: Any,
    components: int = 1,
    reconstruction: Any = None,
    divergence: bool = False,
    trace_reconstruction: Any = None,
    trace_dofs: Any = None,
    offset: Any = None,
) -> FieldDefinition:
    """Declare a physical vector/tensor from explicitly supplied moment and Piola maps.

    The reference basis, global DOF map and per-cell orientation matrix are
    independent mathematical inputs. Components interleave per scalar H(div)
    DOF; one component returns a physical vector, d components row-wise tensor.
    ``divergence=True`` declares its scalar/vector divergence using the very same
    archived basis and maps. No continuity or physical PDE is inferred from
    successful algebraic evaluation. Names such as flux/stress are caller data.
    """
    components = positive_int(components, "components")
    evaluator = _PiolaEvaluator(reference_basis, dofs, orientation, components, divergence)
    if (
        evaluator.dofs.ndim != 2
        or evaluator.dofs.shape[0] != len(mesh.cells)
        or evaluator.dofs.shape[1] != reference_basis.transform.shape[1]
        or evaluator.orientation.shape
        != (len(mesh.cells), evaluator.dofs.shape[1], evaluator.dofs.shape[1])
        or reference_basis.dimension != mesh.points.shape[1]
    ):
        raise ValueError("Piola basis, cell DOFs, orientation and physical dimension must agree")
    digest = sha256(reference_basis.digest.encode())
    digest.update(evaluator.dofs.tobytes())
    digest.update(evaluator.orientation.tobytes())
    digest.update(repr((components, divergence)).encode())
    return FieldDefinition(
        name,
        mesh=mesh,
        evaluator=evaluator,
        reconstruction=reconstruction,
        basis_id=digest.hexdigest(),
        trace_reconstruction=trace_reconstruction,
        trace_dofs=trace_dofs,
        offset=offset,
    )


def _native_basis(cell: str, family: str, degree: int, transform: Any) -> ReferenceVectorBasis:
    """Capture the same native matrix consumed by the shared volume basis owner."""
    element = create_reference_element(
        ReferenceElementSpec(family, cell, degree, lagrange_variant="legendre")
    )
    return ReferenceVectorBasis(
        cell, degree, element.cell_dimension, element.basis_matrix, transform
    )


def hdiv_field(
    name: str,
    mesh: Any,
    family: Any = "RT",
    *,
    degree: int = 0,
    enrichment: int = 0,
    components: int = 1,
    reconstruction: Any = None,
    divergence: bool = False,
) -> FieldDefinition:
    """Declare a built-in moment H(div) field in its actual executed coordinate basis.

    ``family`` is BDMFamily, HDiv3DFamily, or 'RT', 'tensor-RT', 'mapped-RT'.
    Family-specific degree conventions remain those of their volume owners.
    Literal native/moment/orientation matrices are archived once, and replay
    uses them for both physical evaluation and divergence. Pressure and broken
    kinematic polynomial fields have separate scalar/nodal definitions.
    """
    if isinstance(family, HDiv3DFamily):
        coefficients = family.coefficients
        p = family.pressure_degree
        classic = family.normal_degree == 1 and p in (
            (1, 2) if family.kind == "tetrahedron" else (1,)
        )
        if classic:
            powers = tuple(
                e
                for e in product(range(p + 2), repeat=3)
                if (sum(e) <= p + 1 if family.kind == "tetrahedron" else e[0] + e[1] <= p + 1)
            )
            reference = ReferenceVectorBasis(
                family.kind, p + 1, 3, np.empty((0, 0)), coefficients, "monomial", powers
            )
        elif family.kind == "tetrahedron":
            native = _native_basis("tetrahedron", "BDM", p + 1, coefficients)
            reference = ReferenceVectorBasis(
                "tetrahedron",
                p + 1,
                3,
                native.matrix,
                coefficients,
                factors=(tetrahedral_candidate_coefficients(p),),
            )
        else:
            factors = []
            for axis in range(3):
                for dimension, k in ((2, p + int(axis < 2)), (1, p + int(axis == 2))):
                    exponents = tuple(
                        e for e in product(range(k + 1), repeat=dimension + 1) if sum(e) == k
                    )
                    factors.append(bernstein_basis_matrix(dimension, k, exponents))
            reference = ReferenceVectorBasis(
                "prism",
                p,
                3,
                np.empty((0, 0)),
                coefficients,
                "prism_bernstein",
                factors=tuple(factors),
            )
        dofs, orientation = hdiv3d_dofs(mesh, family), hdiv3d_transform(mesh, family)
    else:
        if isinstance(family, BDMFamily):
            reference = _native_basis(
                "triangle", "BDM", family.polynomial_degree, family.reference_coefficients
            )
            dofs, normal, sides = family.dofs(mesh), family.degree, 3
        elif family == "RT":
            reference = _native_basis("triangle", "RT", degree + 1, rt_moment_coefficients(degree))
            dofs, normal, sides = rt_dofs(mesh, degree), degree, 3
        elif family == "tensor-RT":
            reference = _native_basis(
                "quadrilateral",
                "RT",
                degree + enrichment + 1,
                tensor_rt_moment_coefficients(degree, degree + enrichment),
            )
            dofs, normal, sides = tensor_rt_dofs(mesh, degree, enrichment), degree, 4
        elif family == "mapped-RT":
            reference = _native_basis(
                "hexahedron", "RT", degree + 1, mapped_reference_coefficients(degree)
            )
            dofs, normal, sides = mapped_rt_dofs(mesh, degree), degree, 6
        else:
            raise ValueError("unknown moment H(div) family")
        width = dofs.shape[1]
        orientation = np.broadcast_to(np.eye(width), (len(mesh.cells), width, width)).copy()
        if family == "mapped-RT":
            mapped_exponents = np.array(list(product(range(degree + 1), repeat=2)))
            count = (degree + 1) ** 2
            for side in range(6):
                linear = mesh.face_transforms[:, side, 1:]
                mapped_powers = np.einsum("tab,ib->tia", abs(linear), mapped_exponents).astype(int)
                permutation = mapped_powers[..., 0] * (degree + 1) + mapped_powers[..., 1]
                phase = (
                    np.prod(linear.sum(axis=1)[:, None, :] ** mapped_exponents[None], axis=-1)
                    * mesh.signs[:, side, None]
                )
                indices = side * count + np.arange(count)
                orientation[:, indices[:, None], indices] = 0
                for cell in range(len(mesh.cells)):
                    orientation[cell, side * count + permutation[cell], indices] = phase[cell]
        else:
            phase = (mesh.signs[:, :, None] ** np.arange(1, normal + 2)).reshape(
                len(mesh.cells), sides * (normal + 1)
            )
            indices = np.arange(phase.shape[1])
            orientation[:, indices, indices] = phase
    return piola_field(
        name,
        mesh,
        reference_basis=reference,
        dofs=dofs,
        orientation=orientation,
        components=components,
        reconstruction=reconstruction,
        divergence=divergence,
    )
