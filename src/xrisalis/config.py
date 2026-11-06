"""Central units, constants and conventions for XRISALIS.

Unit convention (the single source of truth for the FITS boundary):

* Sky coordinates on disk (FITS): **degrees**.
* Energies on disk (FITS): **keV**.
* Detector positions on disk (FITS): **meters**.
* Times on disk (FITS): **seconds**.

Internally, the physics layer works in SI (radians, Joules, meters, seconds).
Conversions happen only at the io/ boundary, never scattered through the code.
"""

import numpy as np
import astropy.constants as const
import astropy.units as u

# fundamental constants (SI) — sourced from astropy.constants (CODATA), kept as
# plain float SI values so the numeric core works on floats/ndarrays, not
# Quantity objects.
H = const.h.to_value(u.J * u.s)  # Planck constant    [J s]
C = const.c.to_value(u.m / u.s)  # speed of light     [m/s]
EV = (1.0 * u.eV).to_value(u.J)  # electron volt      [J]
KEV = 1e3 * EV  # kilo-electron volt [J]

# angle conversions
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
ARCSEC2RAD = np.pi / (180.0 * 3600.0)
RAD2ARCSEC = (180.0 * 3600.0) / np.pi
UAS2RAD = ARCSEC2RAD * 1e-6
RAD2UAS = RAD2ARCSEC * 1e6


def energy_to_wavelength(energy_j):
    """Photon energy [J] -> wavelength [m]."""
    return H * C / energy_j


def kev_to_joule(e_kev):
    """Photon energy [keV] -> [J]."""
    return np.asarray(e_kev, dtype=float) * KEV


def joule_to_kev(e_j):
    """Photon energy [J] -> [keV]."""
    return np.asarray(e_j, dtype=float) / KEV
