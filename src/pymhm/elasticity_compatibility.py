"""Scale-invariant boundary compatibility for incompressible elasticity."""


def require_compatible_displacement_flux(flux: float, absolute_moments: float) -> None:
    """Reject net volume flux relative to the uncancelled physical boundary moments.

    Both arguments have volume-displacement units and use the same assembled
    boundary moments. The relative tolerance is independent of load amplitude;
    homogeneous data give both arguments zero.
    """
    if abs(flux) > 1e-10 * absolute_moments:
        raise ValueError("incompatible incompressible displacement boundary data")
