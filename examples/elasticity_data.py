"""Exact elasticity fields with bounded loads near incompressibility.

On the unit square let ``phi=128*x²*(1-x)²*y²*(1-y)²`` and
``u0=(-phi_y, phi_x)``. For ``beta=mu/(lambda+mu)`` and amplitude ``a``,
the displacement ``u=u0-a*beta*grad(phi)`` and Herrmann pressure
``p=a*lambda*beta*Delta(phi)`` satisfy ``div(u)+p/lambda=0`` for positive,
finite lambda; at infinite lambda the displacement is divergence-free.
The pressure has zero mean, and the displacement vanishes on the boundary.

The convention is ``sigma=2*mu*epsilon(u)-p*I`` and ``-div(sigma)=f``.
Its load is ``f=-mu*Delta(u0)+a*mu*(1+beta)*grad(Delta(phi))``;
it stays bounded as lambda tends to infinity. Amplitude zero isolates a
solenoidal displacement independent of lambda; amplitude one also exercises
nonzero pressure in the incompressible limit. This polynomial family is an
original manufactured example. A separate trigonometric family below provides
consistent data for the MSL and printed-displacement benchmark conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_BUBBLE = Polynomial([0.0, 0.0, 1.0, -2.0, 1.0])


def _potential(points: FloatArray, dx: int, dy: int) -> FloatArray:
    """Evaluate the indicated partial derivative of the polynomial potential."""
    x, y = np.asarray(points).T
    return 128 * _BUBBLE.deriv(dx)(x) * _BUBBLE.deriv(dy)(y)


@dataclass(frozen=True)
class ElasticityData:
    """Provide compatible displacement, pressure, stress and body-force fields.

    Parameters
    ----------
    lame_lambda
        Nonnegative first Lame modulus; positive infinity denotes the exact
        incompressible limit. At zero, the pressure vanishes identically.
    lame_mu
        Positive, finite shear modulus.
    pressure_amplitude
        Finite amplitude of the pressure-carrying gradient correction. Zero
        gives an exactly solenoidal displacement for every first Lame modulus.

    Notes
    -----
    Array gradients follow ``gradient[q, component, spatial_direction]``.
    Stable modulus ratios avoid both infinity times zero and cancellation of
    a small pressure coefficient when lambda is much smaller than mu.
    """

    lame_lambda: float = 1.0
    lame_mu: float = 1.0
    pressure_amplitude: float = 1.0
    _beta: float = field(init=False, repr=False)
    _pressure_factor: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate material data and compute stable compressibility factors."""
        if np.isnan(self.lame_lambda) or self.lame_lambda < 0:
            raise ValueError("lame_lambda must be nonnegative or positive infinity")
        if not np.isfinite(self.lame_mu) or self.lame_mu <= 0:
            raise ValueError("lame_mu must be positive and finite")
        if not np.isfinite(self.pressure_amplitude):
            raise ValueError("pressure_amplitude must be finite")
        if np.isinf(self.lame_lambda):
            beta, pressure_factor = 0.0, self.lame_mu
        elif self.lame_lambda <= self.lame_mu:
            ratio = self.lame_lambda / self.lame_mu
            beta = 1.0 / (1.0 + ratio)
            pressure_factor = self.lame_lambda / (1.0 + ratio)
        else:
            ratio = self.lame_mu / self.lame_lambda
            beta = ratio / (1.0 + ratio)
            pressure_factor = self.lame_mu / (1.0 + ratio)
        object.__setattr__(self, "_beta", beta)
        object.__setattr__(self, "_pressure_factor", pressure_factor)

    def displacement(self, points: FloatArray) -> FloatArray:
        """Evaluate the boundary-vanishing displacement on an array of points."""
        px, py = _potential(points, 1, 0), _potential(points, 0, 1)
        correction = self.pressure_amplitude * self._beta
        return np.column_stack((-py - correction * px, px - correction * py))

    def gradient(self, points: FloatArray) -> FloatArray:
        """Evaluate the exact displacement gradient without finite differences."""
        pxx = _potential(points, 2, 0)
        pxy = _potential(points, 1, 1)
        pyy = _potential(points, 0, 2)
        correction = self.pressure_amplitude * self._beta
        return np.stack(
            (
                np.column_stack((-pxy - correction * pxx, -pyy - correction * pxy)),
                np.column_stack((pxx - correction * pxy, pxy - correction * pyy)),
            ),
            axis=1,
        )

    def divergence(self, points: FloatArray) -> FloatArray:
        """Evaluate divergence without subtracting two nearly equal gradients."""
        laplacian = _potential(points, 2, 0) + _potential(points, 0, 2)
        return -self.pressure_amplitude * self._beta * laplacian

    def pressure(self, points: FloatArray) -> FloatArray:
        """Evaluate zero-mean Herrmann pressure, including its finite limit."""
        laplacian = _potential(points, 2, 0) + _potential(points, 0, 2)
        return self.pressure_amplitude * self._pressure_factor * laplacian

    def stress(self, points: FloatArray) -> FloatArray:
        """Evaluate symmetric Cauchy stress using the Herrmann pressure."""
        gradient = self.gradient(points)
        return self.lame_mu * (gradient + gradient.swapaxes(1, 2)) - self.pressure(points)[
            :, None, None
        ] * np.eye(2)

    def source(self, points: FloatArray) -> FloatArray:
        """Evaluate the bounded body force from the exact momentum equation."""
        dx = _potential(points, 3, 0) + _potential(points, 1, 2)
        dy = _potential(points, 2, 1) + _potential(points, 0, 3)
        solenoidal_force = self.lame_mu * np.column_stack((dy, -dx))
        correction = self.pressure_amplitude * self.lame_mu * (1.0 + self._beta)
        return solenoidal_force + correction * np.column_stack((dx, dy))


@dataclass(frozen=True)
class TrigonometricElasticityData(ElasticityData):
    """Provide the consistent trigonometric family used in MSL elasticity cases.

    With ``beta=mu/(lambda+mu)=1-2*nu`` and ``a=pressure_amplitude``, use
    ``u=((cos(2*pi*x)-1)*sin(2*pi*y), (1-cos(2*pi*y))*sin(2*pi*x))``
    plus ``a*beta*sin(pi*x)*sin(pi*y)*(1,1)``. The default amplitude one
    corresponds to the MSL manufactured family. Amplitude one half gives the
    displacement printed in section 6 of arXiv:2403.16890, with pressure and
    force derived consistently from that displacement rather than combining
    it with the different pressure and force coefficients printed there.

    All displacement boundary values and the pressure mean vanish. The force
    stays bounded at infinite lambda, and ``stress`` uses the inherited
    symmetric-gradient constitutive law. The Lame conventions and material
    validation are the same as in :class:`ElasticityData`.
    """

    def displacement(self, points: FloatArray) -> FloatArray:
        """Evaluate the trigonometric displacement with its chosen amplitude."""
        x, y = np.asarray(points).T
        pi = np.pi
        solenoidal = np.column_stack(
            (
                (np.cos(2 * pi * x) - 1) * np.sin(2 * pi * y),
                (1 - np.cos(2 * pi * y)) * np.sin(2 * pi * x),
            )
        )
        correction = self.pressure_amplitude * self._beta * np.sin(pi * x) * np.sin(pi * y)
        return solenoidal + correction[:, None]

    def gradient(self, points: FloatArray) -> FloatArray:
        """Evaluate the derivative of both displacement components exactly."""
        x, y = np.asarray(points).T
        pi = np.pi
        sine_x, sine_y = np.sin(2 * pi * x), np.sin(2 * pi * y)
        cosine_x, cosine_y = np.cos(2 * pi * x), np.cos(2 * pi * y)
        solenoidal = (
            2
            * pi
            * np.stack(
                (
                    np.column_stack((-sine_x * sine_y, (cosine_x - 1) * cosine_y)),
                    np.column_stack(((1 - cosine_y) * cosine_x, sine_x * sine_y)),
                ),
                axis=1,
            )
        )
        correction = (
            self.pressure_amplitude
            * self._beta
            * pi
            * np.column_stack((np.cos(pi * x) * np.sin(pi * y), np.sin(pi * x) * np.cos(pi * y)))
        )
        return solenoidal + correction[:, None, :]

    def divergence(self, points: FloatArray) -> FloatArray:
        """Evaluate the exact divergence without cancelling solenoidal terms."""
        x, y = np.asarray(points).T
        return self.pressure_amplitude * self._beta * np.pi * np.sin(np.pi * (x + y))

    def pressure(self, points: FloatArray) -> FloatArray:
        """Evaluate the pressure imposed by the chosen displacement family."""
        x, y = np.asarray(points).T
        return -self.pressure_amplitude * self._pressure_factor * np.pi * np.sin(np.pi * (x + y))

    def source(self, points: FloatArray) -> FloatArray:
        """Evaluate momentum balance for the actual displacement amplitude."""
        x, y = np.asarray(points).T
        pi = np.pi
        solenoidal_force = (
            4
            * self.lame_mu
            * pi**2
            * np.column_stack(
                (
                    (2 * np.cos(2 * pi * x) - 1) * np.sin(2 * pi * y),
                    (1 - 2 * np.cos(2 * pi * y)) * np.sin(2 * pi * x),
                )
            )
        )
        correction = (
            self.pressure_amplitude
            * self.lame_mu
            * pi**2
            * (2 * self._beta * np.sin(pi * x) * np.sin(pi * y) - np.cos(pi * (x + y)))
        )
        return solenoidal_force + correction[:, None]
