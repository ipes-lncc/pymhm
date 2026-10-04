"""Optional DOLFINx/UFL assembly of user-defined local variational problems.

The adapter separates finite-element assembly from the global skeleton. It does
not construct a trace space or infer boundary conditions, kernels, stable mixed
pairs, or macroface orientation. Local meshes must live on one MPI rank.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np
from scipy import sparse

from pymhm.hybrid import LocalProblem
from pymhm.mesh import FloatArray
from pymhm.variational import LocalForm, compile_local_forms


def _require(module: str) -> ModuleType:
    """Load optional finite-element modules only when an adapter is requested."""
    try:
        return import_module(module)
    except (ImportError, OSError) as exc:
        raise ImportError(
            f"Cannot load {module}. Install compatible DOLFINx, Basix and UFL "
            "packages, or use the Pixi fem environment."
        ) from exc


def _assemble_linear(form: Any, fem: ModuleType, space: Any, size: int) -> FloatArray:
    """Assemble one matching linear form, including UFL's simplified zero form."""
    arguments = form.arguments()
    if not arguments and all(integral.integrand() == 0 for integral in form.integrals()):
        return np.zeros(size)
    if len(arguments) != 1 or arguments[0].ufl_function_space() != space:
        raise ValueError("every load, trace and constraint form must use the same test space")
    vector = fem.assemble_vector(fem.form(form))
    if np.iscomplexobj(vector.array):
        raise ValueError("the MHM local adapter currently requires real-valued forms")
    return np.array(vector.array, dtype=float, copy=True)


def from_ufl(
    a: Any,
    load: Any,
    trace_forms: Iterable[Any],
    trace_dofs: Any,
    *,
    kernel: Any = None,
    constraint_forms: Iterable[Any] | None = None,
    coarse_basis: Any = None,
) -> LocalProblem:
    """Assemble UFL forms as ``A u + B lambda = f`` on a serial local mesh.

    Parameters
    ----------
    a
        Bilinear UFL form with trial and test arguments in the same space.
        Scalar, vector, mixed, and H(div) spaces are supported by the assembler.
    load
        Linear UFL right-hand side in that test space.
    trace_forms
        One signed linear form per global trace coefficient. Column ``j`` of B
        is its assembled vector. Include outward/global-normal signs yourself.
    trace_dofs
        Distinct global skeleton indices, one per trace form.
    kernel
        Explicit real coefficient array shaped ``(local_dofs, kernel_size)``.
        Supply both the trial/test nullspace; it is checked by ``LocalProblem``.
    constraint_forms
        One linear form per retained basis vector defining physical moments C.
        Omitting these uses coefficient-space orthogonality C=Z. For a retained
        scalar constant use ``[v * dx]`` to enforce an integral-zero local lift.
    coarse_basis
        Explicit real coefficient array shaped ``(local_dofs, coarse_size)``
        retaining arbitrary coarse modes, including nearly null modes, in the
        global system. Mutually exclusive with ``kernel``; these vectors need
        not be null vectors. Constraint moments must pair nonsingularly with
        the retained basis, as checked by ``LocalProblem``.

    Notes
    -----
    Assembly uses DOLFINx's native CSR interface, without a PETSc requirement.
    ``MPI.COMM_SELF`` is the recommended mesh communicator; any communicator
    containing exactly one rank is accepted. MPI-distributed local meshes and
    complex forms are rejected. Local problems may subsequently be sent to CPU
    workers because their arrays no longer hold DOLFINx or MPI objects.

    No essential boundary condition is applied. In an H(div) formulation, a
    prescribed normal velocity is an essential condition and must be expressed
    with an appropriate augmented form or eliminated before local condensation.
    Merely attaching boundary pressure forms changes the hybrid formulation.
    """
    fem = _require("dolfinx.fem")
    arguments = a.arguments()
    if len(arguments) != 2:
        raise ValueError("a must be a bilinear UFL form")
    space = arguments[0].ufl_function_space()
    if arguments[1].ufl_function_space() != space:
        raise ValueError("trial and test spaces must be the same")
    compiled = fem.form(a)
    if compiled.mesh.comm.size != 1:
        raise ValueError("local DOLFINx meshes must use a single-rank communicator (COMM_SELF)")
    assembled = fem.assemble_matrix(compiled)
    assembled.scatter_reverse()
    matrix = sparse.csr_matrix(assembled.to_scipy(), copy=True)
    if np.iscomplexobj(matrix.data):
        raise ValueError("the MHM local adapter currently requires real-valued forms")
    size = matrix.shape[0]
    traces = list(trace_forms)
    coupling = (
        np.column_stack([_assemble_linear(form, fem, space, size) for form in traces])
        if traces
        else np.empty((size, 0))
    )
    constraints = None
    if constraint_forms is not None:
        forms = list(constraint_forms)
        constraints = (
            np.column_stack([_assemble_linear(form, fem, space, size) for form in forms])
            if forms
            else np.empty((size, 0))
        )
    dofs = np.asarray(trace_dofs)
    if dofs.size == 0:
        dofs = dofs.astype(np.int64)
    return LocalProblem(
        matrix,
        coupling,
        _assemble_linear(load, fem, space, size),
        dofs,
        kernel=kernel,
        constraints=constraints,
        coarse_basis=coarse_basis,
    )


def _apply_coefficient(ufl: ModuleType, coefficient: Any, vector: Any) -> Any:
    """Apply a scalar isotropic or rank-two tensor coefficient to a vector."""
    coefficient = ufl.as_ufl(coefficient)
    return coefficient * vector if coefficient.ufl_shape == () else ufl.dot(coefficient, vector)


def assemble_local_forms(forms: LocalForm) -> LocalProblem:
    """Compile a ``LocalForm`` through the native DOLFINx/UFL adapter.

    Signed trace forms and literal retained coefficients are passed unchanged
    to ``from_ufl``; physical ``moment_forms`` become its constraint forms.
    The existing real-valued, same trial/test space, single-rank restrictions
    apply. No boundary elimination, trace space, gauge or stabilization is
    inferred. Native modules are imported only when this function executes.
    """
    return compile_local_forms(forms, _compile_local_forms)


def _compile_local_forms(forms: LocalForm) -> LocalProblem:
    """Pass the declared forms to the existing owner of native assembly."""
    return from_ufl(
        forms.a,
        forms.L,
        forms.trace_forms,
        forms.trace_dofs,
        kernel=forms.kernel,
        constraint_forms=forms.moment_forms,
        coarse_basis=forms.coarse_basis,
    )


def primal_darcy_forms(space: Any, permeability: Any, source: Any) -> tuple[Any, Any]:
    """Return primal Darcy forms ``(K grad p,grad v)=(f,v)``.

    Pressure is the primal unknown, Darcy velocity is ``-K grad p`` and the
    continuity equation is ``div(velocity)=source``. Supply a positive scalar
    or positive-definite tensor permeability. Boundary flux terms are separate.
    """
    ufl = _require("ufl")
    pressure, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    measure = ufl.dx(domain=space.ufl_domain())
    return (
        ufl.inner(_apply_coefficient(ufl, permeability, ufl.grad(pressure)), ufl.grad(test))
        * measure,
        source * test * measure,
    )


def mixed_darcy_forms(space: Any, inverse_permeability: Any, source: Any) -> tuple[Any, Any]:
    """Return symmetric mixed Darcy forms for H(div) velocity and L2 pressure.

    The equations are ``K^-1 u + grad p=0`` and ``div u=source``. The scalar
    test equation is multiplied by minus one: the bilinear form is
    ``(K^-1 u,v)-(p,div v)-(q,div u)`` and the load is ``-(source,q)``.
    Choose an inf-sup stable pair such as Raviart-Thomas/DG. Pressure boundary
    data contribute ``-<p_D,v.n>`` to the load; normal velocity is essential.
    This helper only defines volume forms and does not impose MHM trace data.
    """
    ufl = _require("ufl")
    velocity, pressure = ufl.TrialFunctions(space)
    test_velocity, test_pressure = ufl.TestFunctions(space)
    measure = ufl.dx(domain=space.ufl_domain())
    return (
        (
            ufl.inner(_apply_coefficient(ufl, inverse_permeability, velocity), test_velocity)
            - pressure * ufl.div(test_velocity)
            - test_pressure * ufl.div(velocity)
        )
        * measure,
        -source * test_pressure * measure,
    )


def brinkman_forms(
    space: Any,
    viscosity: Any,
    resistance: Any,
    force: Any,
    divergence_source: Any = 0.0,
) -> tuple[Any, Any]:
    """Return symmetric Stokes-Brinkman velocity/pressure volume forms.

    The strong momentum operator is ``-div(2 mu sym(grad u))+R u+grad p`` and
    ``div u=g``. The pressure test equation is negated to obtain a symmetric
    saddle operator. An inf-sup stable H1 velocity/pressure pair, for example
    Taylor-Hood P2/P1, is required. No stabilization or boundary conditions are
    inferred. ``resistance=0`` gives Stokes; positive resistance gives Brinkman.
    """
    ufl = _require("ufl")
    velocity, pressure = ufl.TrialFunctions(space)
    test_velocity, test_pressure = ufl.TestFunctions(space)
    measure = ufl.dx(domain=space.ufl_domain())
    bilinear = (
        2 * viscosity * ufl.inner(ufl.sym(ufl.grad(velocity)), ufl.sym(ufl.grad(test_velocity)))
        + ufl.inner(_apply_coefficient(ufl, resistance, velocity), test_velocity)
        - pressure * ufl.div(test_velocity)
        - test_pressure * ufl.div(velocity)
    ) * measure
    linear = (ufl.inner(force, test_velocity) - divergence_source * test_pressure) * measure
    return bilinear, linear


def elasticity_forms(
    space: Any, lame_lambda: Any, shear_modulus: Any, force: Any
) -> tuple[Any, Any]:
    """Return small-strain isotropic elasticity displacement volume forms.

    Stress is ``2 mu sym(grad u)+lambda div(u) I``. In two dimensions these
    Lamé parameters describe plane strain; a plane-stress reduction must be
    supplied explicitly by the caller. Tractions and rigid-motion constraints
    are assembled separately and are not inferred from the displacement space.
    """
    ufl = _require("ufl")
    displacement, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    measure = ufl.dx(domain=space.ufl_domain())
    return (
        (
            2 * shear_modulus * ufl.inner(ufl.sym(ufl.grad(displacement)), ufl.sym(ufl.grad(test)))
            + lame_lambda * ufl.div(displacement) * ufl.div(test)
        )
        * measure,
        ufl.inner(force, test) * measure,
    )


def usfem_brinkman_forms(
    space: Any,
    viscosity: Any,
    resistance: Any,
    force: Any,
    *,
    smallest_resistance: Any,
    divergence_source: Any = 0.0,
    inverse_estimate: float = 1 / 3,
) -> tuple[Any, Any]:
    """Return the symmetric-test version of the 2017 MHM-Brinkman USFEM forms.

    This formulation uses the vector Laplacian and pseudostress, with volume
    form ``nu grad(u):grad(v)+Theta u.v-p div(v)-q div(u)``. Its stabilization
    is **negative** ``-kappa R(u,p).R(v,q)`` with the matching source term
    ``-kappa f.R(v,q)``, where ``R(u,p)=-nu div(grad(u))+Theta u+grad(p)``.
    The pressure test has been negated relative to the paper to obtain the
    symmetric saddle convention used by this package.

    ``smallest_resistance`` is the nonnegative smallest eigenvalue of Theta;
    it must be supplied explicitly for heterogeneous or anisotropic media.
    ``inverse_estimate`` is m=min(1/3,C_inverse)>0. The default m=1/3 is valid
    for the piecewise-linear pair; higher degrees require a justified inverse
    estimate. With cell diameter h, ``kappa=h²/(max(gamma*h²,4*nu/m)+4*nu/m)``.
    For Stokes (gamma=0, nu>0), this gives ``kappa=m*h²/(8*nu)``.

    Viscosity must be constant per cell for this strong residual to represent
    the diffusion operator. Continuous equal-order velocity/pressure spaces
    are intended. Boundary pseudotractions and a global pressure gauge must
    be handled consistently by the caller; this helper does not impose them.
    """
    if not np.isfinite(inverse_estimate) or not 0 < inverse_estimate <= 1 / 3:
        raise ValueError("inverse_estimate must lie in (0, 1/3]")
    ufl = _require("ufl")
    velocity, pressure = ufl.TrialFunctions(space)
    test_velocity, test_pressure = ufl.TestFunctions(space)
    measure = ufl.dx(domain=space.ufl_domain())
    diameter = ufl.CellDiameter(space.ufl_domain())
    viscous_scale = 4 * viscosity / inverse_estimate
    kappa = diameter**2 / (
        ufl.max_value(smallest_resistance * diameter**2, viscous_scale) + viscous_scale
    )
    residual = (
        -viscosity * ufl.div(ufl.grad(velocity))
        + _apply_coefficient(ufl, resistance, velocity)
        + ufl.grad(pressure)
    )
    test_residual = (
        -viscosity * ufl.div(ufl.grad(test_velocity))
        + _apply_coefficient(ufl, resistance, test_velocity)
        + ufl.grad(test_pressure)
    )
    bilinear = (
        viscosity * ufl.inner(ufl.grad(velocity), ufl.grad(test_velocity))
        + ufl.inner(_apply_coefficient(ufl, resistance, velocity), test_velocity)
        - pressure * ufl.div(test_velocity)
        - test_pressure * ufl.div(velocity)
        - kappa * ufl.inner(residual, test_residual)
    ) * measure
    linear = (
        ufl.inner(force, test_velocity)
        - divergence_source * test_pressure
        - kappa * ufl.inner(force, test_residual)
    ) * measure
    return bilinear, linear
