"""Unit coercion at the public API boundary.

XRISALIS accepts astropy :class:`~astropy.units.Quantity` and
:class:`~astropy.coordinates.SkyCoord` objects wherever a physical quantity is
expected, and converts them here to the plain-float internal units so the
numeric core keeps working on floats/ndarrays (not Quantity objects):

    positions -> degrees      angular sizes -> arcsec
    energies  -> keV          times         -> seconds

Bare numbers (and NumPy arrays) are passed through unchanged and interpreted in
those default units, so existing float-based code keeps working.
"""

from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


def _to(value, unit, equivalencies=None):
    """Coerce ``value`` to a plain float/ndarray in ``unit``.

    A :class:`~astropy.units.Quantity` is converted; anything else is returned
    as a float (scalars) or float ndarray (sequences) unchanged.
    """
    if value is None:
        return None
    if isinstance(value, u.Quantity):
        return value.to_value(unit, equivalencies=equivalencies or [])
    if np.isscalar(value):
        return float(value)
    return np.asarray(value, dtype=float)


def to_deg(value):
    """Angle -> degrees (Quantity or bare number in degrees)."""
    return _to(value, u.deg)


def to_arcsec(value):
    """Angle -> arcsec (Quantity or bare number in arcsec)."""
    return _to(value, u.arcsec)


def to_uas(value):
    """Angle -> micro-arcsec (Quantity or bare number in micro-arcsec)."""
    return _to(value, u.uas)


def to_rad(value):
    """Angle -> radians.

    A :class:`~astropy.units.Quantity` is converted from its own unit; a bare
    number is interpreted in **degrees** (the package's sky-coordinate
    convention) and converted to radians. ``None`` passes through.
    """
    if value is None:
        return None
    if isinstance(value, u.Quantity):
        return value.to_value(u.rad)
    return to_deg(value) * (np.pi / 180.0)


def to_kev(value):
    """Energy -> keV (Quantity or bare number in keV).

    Uses the spectral equivalency, so wavelength/frequency Quantities are also
    accepted and converted to photon energy in keV.
    """
    return _to(value, u.keV, equivalencies=u.spectral())


def to_seconds(value):
    """Time interval -> seconds (Quantity or bare number in seconds)."""
    return _to(value, u.s)


def to_radec_deg(position, dec=None):
    """Resolve a sky position to ``(ra_deg, dec_deg)`` as plain floats/arrays.

    ``position`` may be:

    * a :class:`~astropy.coordinates.SkyCoord` (scalar or array) — ``dec`` is
      then ignored;
    * an RA angle (``Quantity`` or bare degrees) with ``dec`` given separately.

    Returns ``(ra_deg, dec_deg)`` where each is a float (scalar input) or an
    ndarray (array input), always in the ICRS frame for a SkyCoord.
    """
    if isinstance(position, SkyCoord):
        icrs = position.icrs
        ra, dec_out = icrs.ra.to_value(u.deg), icrs.dec.to_value(u.deg)
        if position.isscalar:
            return float(ra), float(dec_out)
        return np.asarray(ra, float), np.asarray(dec_out, float)
    return to_deg(position), to_deg(dec)


def is_skycoord(obj):
    return isinstance(obj, SkyCoord)
