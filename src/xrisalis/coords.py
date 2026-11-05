"""Sky <-> tangent-plane coordinates (gnomonic projection).

All off-axis, fringe and imaging maths works in small offsets on the tangent
plane at the nominal pointing ``(ra0, dec0)``: ``xi`` towards East, ``eta``
towards North [rad]. Source positions are true sky ``(ra, dec)`` [deg] and are
projected here, so RA differences carry their ``cos(dec)`` and wrap at 0/360.
The formulas take differences in degrees first and use
``1 - cos x = 2 sin^2(x/2)``, so sub-nas offsets survive in double precision.
There is no automatic re-pointing: the nominal pointing is whatever the user
defines (default ``(0, 0)``), and a source outside the field is not seen.
"""
import numpy as np

from .config import DEG2RAD, RAD2DEG

def wrap_deg(x):
    """Wrap an angle difference [deg] into (-180, 180]."""
    x = np.asarray(x, float)
    return x - 360.0 * np.ceil((x - 180.0) / 360.0)


def sky_to_tangent(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """``(ra, dec)`` [deg] -> ``(xi, eta)`` [rad] about ``(ra0, dec0)``."""
    da = wrap_deg(np.asarray(ra_deg, float) - ra0_deg) * DEG2RAD
    dd = (np.asarray(dec_deg, float) - dec0_deg) * DEG2RAD
    d0, cd = dec0_deg * DEG2RAD, np.cos(np.asarray(dec_deg, float) * DEG2RAD)
    h = 2.0 * np.sin(0.5 * da) ** 2                       # 1 - cos(d_alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosc = np.cos(dd) - np.cos(d0) * cd * h
        return (cd * np.sin(da) / cosc,
                (np.sin(dd) + np.sin(d0) * cd * h) / cosc)


def tangent_to_sky(xi, eta, ra0_deg, dec0_deg, wrap=False):
    """``(xi, eta)`` [rad] -> ``(ra, dec)`` [deg]; RA continuous about ``ra0``
    unless ``wrap`` (then in [0, 360))."""
    s0, c0 = np.sin(dec0_deg * DEG2RAD), np.cos(dec0_deg * DEG2RAD)
    x = c0 - np.asarray(eta, float) * s0
    ra = ra0_deg + np.arctan2(xi, x) * RAD2DEG
    dec = np.arctan2(s0 + np.asarray(eta, float) * c0, np.hypot(xi, x)) * RAD2DEG
    return (np.mod(ra, 360.0) if wrap else ra), dec
