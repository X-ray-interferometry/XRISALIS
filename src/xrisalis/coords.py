"""Sky positions <-> tangent-plane offsets.

Sources are specified by absolute (RA, Dec). Everything downstream (off-axis
angles, fringes, attitude, images) works in small angles on the tangent plane at
the observation's **nominal pointing** ``(ra_pnt, dec_pnt)``: ``east`` (xi) and
``north`` (eta), in radians, from the gnomonic projection

    cos c = sin d0 sin d + cos d0 cos d cos(a - a0)
    xi    = cos d sin(a - a0) / cos c
    eta   = (cos d0 sin d - sin d0 cos d cos(a - a0)) / cos c

so an RA difference counts as ``(a - a0) cos d`` and RA wraps at 0/360 deg.
Positions on the far hemisphere (``cos c <= 0``) have no projection and return NaN.
"""
from __future__ import annotations

import numpy as np

DEG2RAD = np.pi / 180.0


def tangent_offsets(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Gnomonic offsets ``(east, north)`` [rad] of (ra, dec) from (ra0, dec0) [deg]."""
    a = np.asarray(ra_deg, float) * DEG2RAD
    d = np.asarray(dec_deg, float) * DEG2RAD
    a0, d0 = float(ra0_deg) * DEG2RAD, float(dec0_deg) * DEG2RAD
    da = a - a0
    cos_c = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(da)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_c = np.where(cos_c > 0, cos_c, np.nan)
        east = np.cos(d) * np.sin(da) / cos_c
        north = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(da)) / cos_c
    return east, north


def tangent_to_sky(east_rad, north_rad, ra0_deg, dec0_deg):
    """Inverse of :func:`tangent_offsets`: ``(ra_deg, dec_deg)``, RA in [0, 360)."""
    x = np.asarray(east_rad, float)
    y = np.asarray(north_rad, float)
    a0, d0 = float(ra0_deg) * DEG2RAD, float(dec0_deg) * DEG2RAD
    denom = np.cos(d0) - y * np.sin(d0)
    a = a0 + np.arctan2(x, denom)
    d = np.arctan2(np.sin(d0) + y * np.cos(d0), np.hypot(x, denom))
    return np.mod(a / DEG2RAD, 360.0), d / DEG2RAD


def sky_centroid(ra_deg, dec_deg, weights=None):
    """Weighted mean direction ``(ra_deg, dec_deg)`` (wrap-safe, via unit vectors)."""
    a = np.atleast_1d(np.asarray(ra_deg, float)) * DEG2RAD
    d = np.atleast_1d(np.asarray(dec_deg, float)) * DEG2RAD
    w = np.ones_like(a) if weights is None else np.atleast_1d(np.asarray(weights, float))
    v = np.array([np.sum(w * np.cos(d) * np.cos(a)), np.sum(w * np.cos(d) * np.sin(a)),
                  np.sum(w * np.sin(d))])
    if not np.any(v):
        return 0.0, 0.0
    ra = np.mod(np.arctan2(v[1], v[0]) / DEG2RAD, 360.0)
    dec = np.arctan2(v[2], np.hypot(v[0], v[1])) / DEG2RAD
    return float(ra), float(dec)


def nominal_pointing(source):
    """Default nominal pointing ``(ra_deg, dec_deg)`` for a source.

    An xrimodel source's reference position; otherwise the centre (unweighted
    mean direction) of the source's positions: components, sub-sources, or a
    catalogue's ``ra_deg``/``dec_deg``; (0, 0) if the source has no positions.
    """
    ref = getattr(source, "reference_deg", None)
    if ref is not None:
        return float(ref[0]), float(ref[1])
    subs = getattr(source, "sources", None)
    if subs:
        pts = np.array([nominal_pointing(s) for s in subs])
        return sky_centroid(pts[:, 0], pts[:, 1])
    if hasattr(source, "positions_deg"):
        try:
            pos = np.atleast_2d(source.positions_deg())
            if pos.size:
                return sky_centroid(pos[:, 0], pos[:, 1])
        except (TypeError, NotImplementedError):
            pass
    if hasattr(source, "ra_deg") and np.size(source.ra_deg):
        return sky_centroid(source.ra_deg, source.dec_deg)
    return 0.0, 0.0


def sky_offset_deg(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Boresight ``(pitch, yaw)`` [deg] that points at (ra, dec) for a nominal
    pointing (ra0, dec0): the (north, east) tangent offset, for
    :func:`xrisalis.pointing.constant_pointing` and friends."""
    east, north = tangent_offsets(ra_deg, dec_deg, ra0_deg, dec0_deg)
    return north / DEG2RAD, east / DEG2RAD
