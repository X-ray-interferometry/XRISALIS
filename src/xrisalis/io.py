"""FITS I/O — the single I/O boundary of XRISALIS.

PhotonList and EventList data products, OGIP ARF/RMF and SIMPUT readers, and
shared FITS helpers. astropy is the only FITS backend. Sky coords are degrees,
energies keV, detector positions meters, times seconds on the FITS boundary."""

from __future__ import annotations
import datetime as _dt
import os
import re
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from dataclasses import dataclass, field
import numpy as np
from .config import KEV, H, C
from .coords import tangent_offsets


# ==========================================================================
# from fits_base.py
# ==========================================================================

try:
    from .version import __version__ as XRI_VERSION
except Exception:  # pragma: no cover - version file may be absent in dev
    XRI_VERSION = "0.0.dev"

TELESCOP = "XRI"
INSTRUME = "CIS"


def provenance_header(history=None, extra=None, seed=None):
    """Return a fits.Header with standard provenance keywords.

    Parameters
    ----------
    history : str or list[str]
        HISTORY line(s) describing the call that produced the product.
    extra : dict
        Additional keyword cards.
    seed : int, optional
        PRNG seed used, for reproducibility (PRNGSEED).
    """
    h = fits.Header()
    h["ORIGIN"] = ("XRISALIS", "X-ray interferometry sim & analysis")
    h["XRIVERS"] = (str(XRI_VERSION), "XRISALIS version")
    h["DATE"] = (
        _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "File creation UTC",
    )
    h["TELESCOP"] = TELESCOP
    h["INSTRUME"] = INSTRUME
    if seed is not None:
        h["PRNGSEED"] = (int(seed), "PRNG seed for reproducibility")
    if extra:
        for k, v in extra.items():
            h[k] = v
    if history:
        for line in [history] if isinstance(history, str) else history:
            h.add_history(line)
    return h


def write_hdulist(hdus, path, overwrite=True):
    """Write a list of HDUs (a PrimaryHDU is prepended if absent)."""
    if not isinstance(hdus[0], fits.PrimaryHDU):
        hdus = [fits.PrimaryHDU(header=provenance_header())] + list(hdus)
    fits.HDUList(hdus).writeto(path, overwrite=overwrite)


def col(name, fmt, array, unit=None):
    """Thin wrapper around fits.Column with an optional unit."""
    return fits.Column(name=name, format=fmt, array=array, unit=unit)


# ==========================================================================
# from photon_list.py
# ==========================================================================


@dataclass
class PhotonList:
    time: np.ndarray  # [s]
    energy_kev: np.ndarray  # [keV]
    ra_deg: np.ndarray  # [deg]
    dec_deg: np.ndarray  # [deg]
    src_id: np.ndarray = None  # [-]
    baseline_id: np.ndarray = None
    exposure: float = None
    ra_pnt: float = 0.0  # nominal pointing RA  [deg] (tangent point)
    dec_pnt: float = 0.0  # nominal pointing Dec [deg]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.time = np.asarray(self.time, float)
        n = self.time.size
        self.energy_kev = np.asarray(self.energy_kev, float)
        self.ra_deg = np.asarray(self.ra_deg, float)
        self.dec_deg = np.asarray(self.dec_deg, float)
        if self.src_id is None:
            self.src_id = np.zeros(n, dtype=np.int32)
        if self.baseline_id is None:
            self.baseline_id = np.full(n, -1, dtype=np.int16)

    def __len__(self):
        return self.time.size

    # --- physics-layer views (SI) -------------------------------------------
    @property
    def energy_j(self):
        return self.energy_kev * KEV

    @property
    def loc_rad(self):
        """(N,2) tangent-plane offset [rad] as [north, east] from the nominal
        pointing ``(ra_pnt, dec_pnt)`` (gnomonic, see :mod:`xrisalis.coords`), on
        the same axes as the boresight ``(pitch, yaw)``."""
        east, north = tangent_offsets(self.ra_deg, self.dec_deg,
                                      self.ra_pnt, self.dec_pnt)
        return np.column_stack((north, east))

    # --- io ------------------------------------------------------------------
    def to_hdu(self):
        cols = [
            col("TIME", "D", self.time, "s"),
            col("ENERGY", "E", self.energy_kev, "keV"),
            col("RA", "D", self.ra_deg, "deg"),
            col("DEC", "D", self.dec_deg, "deg"),
            col("SRC_ID", "J", self.src_id),
            col("BASELINE_ID", "I", self.baseline_id),
        ]
        hdr = provenance_header(
            history="PhotonList (generate stage)", seed=self.meta.get("seed")
        )
        hdr["EXTNAME"] = "PHOTONS"
        if self.exposure is not None:
            hdr["EXPOSURE"] = (float(self.exposure), "Exposure time [s]")
        hdr["RA_PNT"] = (float(self.ra_pnt), "Nominal pointing RA [deg]")
        hdr["DEC_PNT"] = (float(self.dec_pnt), "Nominal pointing DEC [deg]")
        return fits.BinTableHDU.from_columns(cols, header=hdr)

    def write(self, path, overwrite=True):
        write_hdulist([self.to_hdu()], path, overwrite=overwrite)
        return path

    @classmethod
    def read(cls, path):
        with fits.open(path) as hdul:
            h = hdul["PHOTONS"]
            d = h.data
            obj = cls(
                time=d["TIME"],
                energy_kev=d["ENERGY"],
                ra_deg=d["RA"],
                dec_deg=d["DEC"],
                src_id=d["SRC_ID"] if "SRC_ID" in d.names else None,
                baseline_id=d["BASELINE_ID"] if "BASELINE_ID" in d.names else None,
                exposure=h.header.get("EXPOSURE"),
                ra_pnt=h.header.get("RA_PNT", 0.0),
                dec_pnt=h.header.get("DEC_PNT", 0.0),
            )
        return obj


# ==========================================================================
# from event_list.py
# ==========================================================================


@dataclass
class EventList:
    time: np.ndarray  # read-out time [s]
    energy_kev: np.ndarray  # channel-mapped energy [keV]
    dety: np.ndarray  # detector fringe-axis position [m]
    detz: np.ndarray  # detector cross-axis position [m]
    baseline_id: np.ndarray  # [-]
    pi: np.ndarray = None  # detector channel [-]
    true_energy_kev: np.ndarray = None  # simulation ground truth [keV] (optional)
    ra_deg: np.ndarray = None
    dec_deg: np.ndarray = None
    pointing: np.ndarray = None  # (T,3) pitch,yaw,roll [rad]; row = int(time/time_step)
    #   pitch/yaw are tangent-plane offsets (North, East) from (ra_pnt, dec_pnt)
    ra_pnt: float = 0.0  # nominal pointing RA  [deg] (tangent point)
    dec_pnt: float = 0.0  # nominal pointing Dec [deg]
    time_step: float = 1.0  # attitude sampling interval [s] (POINTING row spacing)
    ebounds: dict = None  # {'CHANNEL','E_MIN','E_MAX'}
    exposure: float = None
    fov: float = None  # derived imaging field of view [rad] (half-angle)
    slat_id: np.ndarray = None  # slatted-mirror comb patch index per photon (-1 = none)
    # multi-chip focal-plane coordinates (optional; see xrisalis.detector).
    # Image/ds9 convention: x_img <- DETY (fringe axis), y_img <- DETZ (cross
    # axis), matching the spacecraft/optical design. So CHIPX/FOCX run with DETY
    # and CHIPY/FOCY run with DETZ.
    chip_id: np.ndarray = None  # chip each photon landed on (-1 if none)
    chipx: np.ndarray = None  # within-chip pixel column [pix]  (x_img, from DETY)
    chipy: np.ndarray = None  # within-chip pixel row [pix]     (y_img, from DETZ)
    chipx_m: np.ndarray = None  # within-chip position from corner [m] (DETY)
    chipy_m: np.ndarray = None  # within-chip position from corner [m] (DETZ)
    focx: np.ndarray = None  # total focal-plane x (fringe/DETY axis) [m]
    focy: np.ndarray = None  # total focal-plane y (cross/DETZ axis) [m]
    header_extra: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.time = np.asarray(self.time, float)
        n = self.time.size
        self.energy_kev = np.asarray(self.energy_kev, float)
        self.dety = np.asarray(self.dety, float)
        self.detz = (
            np.asarray(self.detz, float) if self.detz is not None else np.zeros(n)
        )
        self.baseline_id = np.asarray(self.baseline_id, np.int16)
        if self.pi is None:
            self.pi = np.full(n, -1, dtype=np.int32)
        if self.ra_deg is None:
            self.ra_deg = np.zeros(n)
        if self.dec_deg is None:
            self.dec_deg = np.zeros(n)
        if self.true_energy_kev is not None:
            self.true_energy_kev = np.asarray(self.true_energy_kev, float)
        self.time_step = float(self.time_step) if self.time_step else 1.0

    @property
    def true_energy_j(self):
        """True photon energy [J] (ideal-reconstruction / diagnostics)."""
        return None if self.true_energy_kev is None else self.true_energy_kev * KEV

    def __len__(self):
        return self.time.size

    # --- physics-layer views -------------------------------------------------
    @property
    def energy_j(self):
        return self.energy_kev * KEV

    @property
    def roll_per_event(self):
        """Roll angle [rad] per event, looked up from the POINTING stream."""
        return self.roll_per_event_from(None)

    # --- attitude: what the events flew vs. what we *think* they flew --------
    #
    # The POINTING stream stored on an EventList is the attitude the simulation
    # actually flew -- the TRUTH. Real reconstruction never has that; it has an
    # attitude-determination *estimate* (star tracker + gyro, possibly refined
    # by fringe self-calibration), which differs from the truth by a
    # time-varying error. ``attitude_stream`` is the seam between the two: pass
    # an estimated stream to interpret the events with, and the difference
    # between it and the stored truth is exactly the pointing error the fringes
    # have to survive.

    def attitude_stream(self, attitude=None):
        """The ``(T, 3)`` attitude stream [rad] to interpret these events with.

        Parameters
        ----------
        attitude : None, ndarray, Pointing, callable, path or tuple
            None (default) returns the stored POINTING stream -- the true
            attitude, i.e. perfect knowledge. Anything else is resolved to a
            stream of the same length: an ``(T, 2)`` or ``(T, 3)`` array is
            taken as-is, and any other spec goes through
            :func:`xrisalis.pointing.as_pointing` and is sampled on the same
            integer time-step grid the POINTING stream uses. A stream carrying
            only ``(pitch, yaw)`` inherits the stored roll column.
        """
        if attitude is None:
            return self.pointing
        arr = attitude
        if not isinstance(arr, np.ndarray):
            from .pointing import as_pointing

            p = as_pointing(attitude)
            if p is None:
                return self.pointing
            n_t = (
                self.pointing.shape[0]
                if self.pointing is not None
                else int(np.max(self.time) / self.time_step) + 2
            )
            arr = p(np.arange(n_t, dtype=float) * self.time_step)
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        if arr.shape[1] == 2:  # no roll supplied -> keep ours
            roll = (
                self.pointing[: arr.shape[0], 2]
                if self.pointing is not None
                else np.zeros(arr.shape[0])
            )
            arr = np.column_stack([arr, roll])
        if arr.shape[1] != 3:
            raise ValueError(
                "attitude must have 2 (pitch, yaw) or 3 "
                f"(pitch, yaw, roll) columns; got {arr.shape[1]}"
            )
        return arr

    def _attitude_per_event(self, stream):
        """Attitude ``(N, 3)`` per event, interpolated at the event times."""
        from .pointing import attitude_at
        return attitude_at(stream, self.time_step, self.time)

    def roll_per_event_from(self, attitude=None):
        """Roll [rad] per event from ``attitude`` (see :meth:`attitude_stream`),
        interpolated at each event's time."""
        stream = self.attitude_stream(attitude)
        if stream is None:
            return np.zeros(len(self))
        return self._attitude_per_event(stream)[:, 2]

    def boresight_per_event(self, attitude=None):
        """``(N, 2)`` boresight ``(pitch, yaw)`` [rad] per event.

        The companion of :attr:`roll_per_event` for the two *translational*
        attitude axes, which reconstruction needs in order to phase-reference
        every photon to a common phase center (see
        :func:`xrisalis.reconstruct.compute_uv` and :meth:`mean_boresight`).
        """
        stream = self.attitude_stream(attitude)
        if stream is None:
            return np.zeros((len(self), 2))
        return self._attitude_per_event(stream)[:, :2]

    def mean_boresight(self):
        """Mean recorded boresight ``(pitch, yaw)`` [rad] over the events.

        Like the stream itself, this is a tangent-plane offset (North, East)
        from the nominal pointing ``(ra_pnt, dec_pnt)``;
        :meth:`mean_boresight_radec` gives the sky position.

        The time-average pointing of the observation, taken from the *stored*
        POINTING stream (never from an attitude estimate passed to
        :meth:`attitude_stream`). It is the default **phase center** of
        :func:`xrisalis.reconstruct.reconstruct`: the sky point the image is
        centered on. Averaging over the events rather than over the stream's
        rows weights each attitude by the photons that saw it.

        A constant stream returns its exact value (no floating-point averaging
        residue), so phase referencing a fixed stare is bit-for-bit a no-op.
        """
        bore = self.boresight_per_event()
        if bore.shape[0] == 0:
            return np.zeros(2)
        if np.all(bore == bore[0]):
            return bore[0].copy()
        return bore.mean(axis=0)

    def offsets_to_radec(self, pitch_yaw):
        """Tangent-plane ``(pitch, yaw)`` [rad] -> sky ``(ra_deg, dec_deg)``."""
        from .coords import tangent_to_sky

        p = np.asarray(pitch_yaw, float)
        ra, dec = tangent_to_sky(p[..., 1], p[..., 0], self.ra_pnt, self.dec_pnt)
        return ra, dec

    def radec_to_offsets(self, ra_deg, dec_deg):
        """Sky ``(ra_deg, dec_deg)`` -> tangent-plane ``(pitch, yaw)`` [rad]."""
        from .coords import sky_to_tangent

        xi, eta = sky_to_tangent(ra_deg, dec_deg, self.ra_pnt, self.dec_pnt)
        return np.stack((np.asarray(eta, float), np.asarray(xi, float)), axis=-1)

    def mean_boresight_radec(self):
        """Sky position ``(ra_deg, dec_deg)`` of :meth:`mean_boresight`."""
        ra, dec = self.offsets_to_radec(self.mean_boresight())
        return float(ra), float(dec)

    # --- selection -----------------------------------------------------------
    def select(self, mask):
        """A new EventList holding only the events picked out by ``mask``.

        ``mask`` is any boolean or integer index array over the events. Every
        per-event column is indexed; the observation-level metadata (the
        POINTING stream, EBOUNDS, EXPOSURE, the cached FoV) is carried over
        unchanged, so the result reconstructs exactly like the parent.

        This is what makes *phase-* or *time-resolved* imaging a one-liner —
        split on any column and hand each slice to ``reconstruct``::

            phase = (events.time % period) / period
            early = events.select(phase < 0.5)

        Note that ``EXPOSURE`` is deliberately *not* rescaled: a subset spans the
        same observation, so rates computed from a slice (e.g. a phase bin)
        must be divided by the live time of that slice, not by ``exposure``.
        Energy and baseline cuts do not need this — ``reconstruct`` takes
        ``energy_kev=`` and ``baseline_ids=`` directly.
        """
        import copy as _copy

        n = len(self)
        idx = np.asarray(mask)
        if idx.dtype == bool and idx.size != n:
            raise ValueError(f"boolean mask has {idx.size} entries for {n} events")
        out = _copy.copy(self)
        for name in (
            "time",
            "energy_kev",
            "dety",
            "detz",
            "baseline_id",
            "pi",
            "true_energy_kev",
            "ra_deg",
            "dec_deg",
            "slat_id",
            "chip_id",
            "chipx",
            "chipy",
            "chipx_m",
            "chipy_m",
            "focx",
            "focy",
        ):
            col_ = getattr(self, name, None)
            if col_ is not None and np.ndim(col_) == 1 and np.size(col_) == n:
                setattr(out, name, np.asarray(col_)[idx])
        out.meta = dict(self.meta)
        out.meta["selected_from"] = n
        return out

    # --- io ------------------------------------------------------------------
    def _events_hdu(self):
        cols = [
            col("TIME", "D", self.time, "s"),
            col("PI", "J", self.pi, "chan"),
            col("ENERGY", "E", self.energy_kev, "keV"),
            col("DETY", "E", self.dety, "m"),
            col("DETZ", "E", self.detz, "m"),
            col("BASELINE_ID", "I", self.baseline_id),
            col("RA", "D", self.ra_deg, "deg"),
            col("DEC", "D", self.dec_deg, "deg"),
        ]
        if self.true_energy_kev is not None:
            cols.append(col("TRUE_ENERGY", "E", self.true_energy_kev, "keV"))
        if self.slat_id is not None:
            cols.append(col("SLAT_ID", "J", self.slat_id))
        if self.chip_id is not None:  # multi-chip focal plane
            cols.append(col("CHIP_ID", "J", self.chip_id))
            cols.append(col("CHIPX", "E", self.chipx, "pix"))
            cols.append(col("CHIPY", "E", self.chipy, "pix"))
            cols.append(col("CHIPX_M", "E", self.chipx_m, "m"))
            cols.append(col("CHIPY_M", "E", self.chipy_m, "m"))
            cols.append(col("FOCX", "E", self.focx, "m"))
            cols.append(col("FOCY", "E", self.focy, "m"))
        hdr = provenance_header(
            history="EventList (redistribute stage)", seed=self.meta.get("seed")
        )
        hdr["EXTNAME"] = "EVENTS"
        if self.exposure is not None:
            hdr["EXPOSURE"] = (float(self.exposure), "Exposure time [s]")
        if self.fov is not None and np.isfinite(self.fov):
            hdr["FOV"] = (float(self.fov), "Derived imaging FoV, half-angle [rad]")
        hdr["TIMESTEP"] = (float(self.time_step), "Attitude row spacing [s]")
        hdr["RA_PNT"] = (float(self.ra_pnt), "Nominal pointing RA [deg]")
        hdr["DEC_PNT"] = (float(self.dec_pnt), "Nominal pointing DEC [deg]")
        for k, v in self.header_extra.items():
            hdr[k] = v
        return fits.BinTableHDU.from_columns(cols, header=hdr)

    def _pointing_hdu(self):
        p = self.pointing
        t = np.arange(p.shape[0], dtype=float) * float(self.time_step)
        cols = [
            col("TIME", "D", t, "s"),
            col("PITCH", "E", p[:, 0], "rad"),
            col("YAW", "E", p[:, 1], "rad"),
            col("ROLL", "E", p[:, 2], "rad"),
        ]
        hdr = fits.Header()
        hdr["EXTNAME"] = "POINTING"
        hdr["HDUCLAS1"] = "TEMPORALDATA"
        hdr["TIMESTEP"] = (float(self.time_step), "Attitude row spacing [s]")
        hdr.add_comment("Spacecraft attitude stream (attitude-sensor analogue).")
        return fits.BinTableHDU.from_columns(cols, header=hdr)

    def _ebounds_hdu(self):
        eb = self.ebounds
        cols = [
            col("CHANNEL", "J", np.asarray(eb["CHANNEL"])),
            col("E_MIN", "E", np.asarray(eb["E_MIN"]), "keV"),
            col("E_MAX", "E", np.asarray(eb["E_MAX"]), "keV"),
        ]
        hdr = fits.Header()
        hdr["EXTNAME"] = "EBOUNDS"
        hdr["HDUCLAS1"] = "RESPONSE"
        hdr["HDUCLAS2"] = "EBOUNDS"
        return fits.BinTableHDU.from_columns(cols, header=hdr)

    def write(self, path, overwrite=True):
        hdus = [self._events_hdu()]
        if self.pointing is not None:
            hdus.append(self._pointing_hdu())
        if self.ebounds is not None:
            hdus.append(self._ebounds_hdu())
        write_hdulist(hdus, path, overwrite=overwrite)
        return path

    @classmethod
    def read(cls, path):
        with fits.open(path) as hdul:
            d = hdul["EVENTS"].data
            hdr = hdul["EVENTS"].header
            pointing = None
            time_step = float(hdr.get("TIMESTEP", 1.0))
            if "POINTING" in hdul:
                ph = hdul["POINTING"]
                pd = ph.data
                pointing = np.column_stack((pd["PITCH"], pd["YAW"], pd["ROLL"]))
                if "TIMESTEP" in ph.header:
                    time_step = float(ph.header["TIMESTEP"])
                elif len(pd["TIME"]) > 1:  # infer from the TIME column
                    time_step = float(pd["TIME"][1] - pd["TIME"][0])
            ebounds = None
            if "EBOUNDS" in hdul:
                ed = hdul["EBOUNDS"].data
                ebounds = {
                    "CHANNEL": np.array(ed["CHANNEL"]),
                    "E_MIN": np.array(ed["E_MIN"]),
                    "E_MAX": np.array(ed["E_MAX"]),
                }
            names = d.names
            return cls(
                time=d["TIME"],
                energy_kev=d["ENERGY"],
                dety=d["DETY"] if "DETY" in names else np.zeros(len(d["TIME"])),
                detz=d["DETZ"] if "DETZ" in names else None,
                baseline_id=d["BASELINE_ID"],
                pi=d["PI"] if "PI" in names else None,
                true_energy_kev=d["TRUE_ENERGY"] if "TRUE_ENERGY" in names else None,
                ra_deg=d["RA"] if "RA" in names else None,
                dec_deg=d["DEC"] if "DEC" in names else None,
                pointing=pointing, 
                time_step=time_step, 
                ebounds=ebounds,
                exposure=hdr.get("EXPOSURE"), 
                fov=hdr.get("FOV"),
                ra_pnt=hdr.get("RA_PNT", 0.0), 
                dec_pnt=hdr.get("DEC_PNT", 0.0),
                slat_id=d["SLAT_ID"] if "SLAT_ID" in names else None,
                chip_id=d["CHIP_ID"] if "CHIP_ID" in names else None,
                chipx=d["CHIPX"] if "CHIPX" in names else None,
                chipy=d["CHIPY"] if "CHIPY" in names else None,
                chipx_m=d["CHIPX_M"] if "CHIPX_M" in names else None,
                chipy_m=d["CHIPY_M"] if "CHIPY_M" in names else None,
                focx=d["FOCX"] if "FOCX" in names else None,
                focy=d["FOCY"] if "FOCY" in names else None,
            )


# ==========================================================================
# from responses.py
# ==========================================================================


# --------------------------------------------------------------------- ARF ---
@dataclass
class ARF:
    """Per-baseline ancillary response (effective area)."""

    energ_lo: np.ndarray  # [keV]
    energ_hi: np.ndarray  # [keV]
    total: np.ndarray  # total effective area [cm^2]
    per_baseline: list  # list of (D_meters, specresp[cm^2])

    @property
    def energy_mid(self):
        return 0.5 * (self.energ_lo + self.energ_hi)

    @property
    def baseline_lengths(self):
        return np.array([D for D, _ in self.per_baseline])

    @property
    def n_baselines(self):
        return len(self.per_baseline)

    @classmethod
    def read(cls, path):
        with fits.open(path) as hdul:
            lo = np.array(hdul["SPECRESP"].data["ENERG_LO"], float)
            hi = np.array(hdul["SPECRESP"].data["ENERG_HI"], float)
            total = np.array(hdul["SPECRESP"].data["SPECRESP"], float)
            per = []
            i = 1
            while f"SPECRESP_BL{i}" in hdul:
                ext = hdul[f"SPECRESP_BL{i}"]
                D = float(ext.header["BASELINE"])
                resp = np.array(ext.data[f"SPECRESP_BL{i}"], float)
                per.append((D, resp))
                i += 1
        return cls(lo, hi, total, per)


# --------------------------------------------------------------------- RMF ---
@dataclass
class RMF:
    energ_lo: np.ndarray
    energ_hi: np.ndarray
    matrix: np.ndarray  # (n_energy, n_channel) redistribution matrix
    f_chan: np.ndarray
    n_chan: np.ndarray
    channel: np.ndarray
    e_min: np.ndarray  # channel lower energy [keV]
    e_max: np.ndarray  # channel upper energy [keV]

    @property
    def channel_energy(self):
        return 0.5 * (self.e_min + self.e_max)

    @property
    def ebounds_dict(self):
        return {"CHANNEL": self.channel, "E_MIN": self.e_min, "E_MAX": self.e_max}

    @classmethod
    def read(cls, path):
        with fits.open(path) as hdul:
            m = hdul["MATRIX"].data
            eb = hdul["EBOUNDS"].data
            # MATRIX may be fixed-width (n_chan array) or OGIP variable-length.
            mat = np.array([np.asarray(row, float) for row in m["MATRIX"]])
            return cls(
                energ_lo=np.array(m["ENERG_LO"], float),
                energ_hi=np.array(m["ENERG_HI"], float),
                matrix=mat,
                f_chan=np.array(m["F_CHAN"]),
                n_chan=np.array(m["N_CHAN"]),
                channel=np.array(eb["CHANNEL"]),
                e_min=np.array(eb["E_MIN"], float),
                e_max=np.array(eb["E_MAX"], float),
            )


# ==========================================================================
# from source_readers.py — universal FITS source ingestion
# ==========================================================================


# --------------------------------------------------------------- utilities ---
def _as_str(value):
    """Decode a FITS cell (str, bytes, or char-array) to a plain string."""
    if isinstance(value, bytes):
        return value.decode("ascii", "ignore")
    if isinstance(value, np.ndarray):
        return "".join(np.asarray(value, dtype=str).ravel())
    return str(value)


def energy_to_kev(values, unit):
    """Convert spectral-axis ``values`` given in ``unit`` to keV.

    Handles the units XRISALIS is likely to meet on a FITS boundary: keV/eV/MeV,
    Joule (astropy normalizes a WCS ``ENER`` axis to J), Hz, and wavelength
    (Angstrom/nm/m). Unknown units warn and are assumed keV.
    """
    v = np.asarray(values, float)
    u = (unit or "keV").strip().lower()
    if u in ("kev",):
        return v
    if u in ("ev",):
        return v * 1e-3
    if u in ("mev",):
        return v * 1e3
    if u in ("j", "joule", "joules"):
        return v / KEV
    if u in ("hz",):
        return H * v / KEV
    if u in ("khz", "mhz", "ghz"):
        scale = {"khz": 1e3, "mhz": 1e6, "ghz": 1e9}[u]
        return H * v * scale / KEV
    if u in ("angstrom", "a", "ang", "aa"):
        return (H * C / (v * 1e-10)) / KEV
    if u in ("nm",):
        return (H * C / (v * 1e-9)) / KEV
    if u in ("m",):
        return (H * C / v) / KEV
    warnings.warn(f"Unknown spectral unit {unit!r}; assuming keV.")
    return v


# ------------------------------------------------------- reference resolver ---
# SIMPUT "extended filename" reference:  file.fits[EXT,ver][ROWFILTER]
_REF_RE = re.compile(
    r"^\s*(?P<file>[^\[\]]+?)?\s*"
    r"(?:\[(?P<ext>[^\],]+?)(?:,(?P<ver>[^\]]+))?\])?"
    r"(?:\[(?P<filt>[^\]]+)\])?\s*$"
)
_NAME_FILTER_RE = re.compile(r"""NAME\s*==\s*['"]?([^'"\]]+)['"]?""", re.IGNORECASE)


def _within_subtree(path, base_dir):
    """True if ``path`` resolves inside ``base_dir`` (external-ref safety)."""
    base = os.path.realpath(base_dir)
    tgt = os.path.realpath(path)
    return tgt == base or tgt.startswith(base + os.sep)


def _detach_hdu(hdu):
    """Return an in-memory copy of ``hdu`` that survives its file being closed.

    ``BinTableHDU.copy`` cannot reconstruct variable-length-array heaps, so tables
    are rebuilt column-by-column (reading each field while the file is still open
    materializes the heap into ordinary arrays)."""
    if getattr(hdu, "is_image", False):
        return fits.ImageHDU(data=np.array(hdu.data), header=hdu.header.copy())
    out = []
    for c in hdu.columns:
        arr = hdu.data[c.name]
        vla = ("P" in c.format.upper()) or ("Q" in c.format.upper())
        out.append(
            fits.Column(
                name=c.name,
                format=c.format,
                unit=c.unit,
                array=np.array(list(arr), dtype=object) if vla else np.array(arr),
            )
        )
    return fits.BinTableHDU.from_columns(out, header=hdu.header.copy())


def resolve_reference(ref, hdul, base_dir, cache=None):
    """Resolve a SIMPUT reference to an in-memory HDU + optional row index.

    Parameters
    ----------
    ref : str | bytes | char-array
        A SIMPUT reference, e.g. ``"[SPECTRUM,1][NAME=='spec_soft']"``,
        ``"[IMG1,1]"`` or ``"spectrum.fits[SPECTRUM,1][NAME=='x']"``. ``""`` /
        ``"NULL"`` mean "no reference".
    hdul : fits.HDUList
        The open catalog file (source of same-file references).
    base_dir : str
        Directory of the catalog file; external references are opened relative
        to it and only if they resolve inside its subtree.
    cache : dict, optional
        Shared between calls for one catalog: each referenced HDU is detached
        once and NAME filters use a cached name -> row index.

    Returns
    -------
    (hdu, row) : (fits HDU copy, int | None) or (None, None)
        ``hdu`` is detached from any external file (safe after close). ``row`` is
        the row selected by a ``[NAME=='..']`` filter (or an integer row filter),
        else None. Warns and returns ``(None, None)`` when unresolvable.
    """
    s = _as_str(ref).strip()
    if s == "" or s.upper() == "NULL":
        return None, None
    m = _REF_RE.match(s)
    if not m:
        warnings.warn(f"Unparseable SIMPUT reference {s!r}; ignoring.")
        return None, None
    fname, ext, ver, filt = (
        m.group("file"),
        m.group("ext"),
        m.group("ver"),
        m.group("filt"),
    )

    # optional per-catalog cache: each referenced HDU is detached once and
    # NAME filters use a prebuilt name -> row index (O(1) per reference
    # instead of copying + scanning the table for every source)
    ckey = None
    if cache is not None:
        ckey = (
            os.path.realpath(os.path.join(base_dir, fname)) if fname else None,
            ext,
            ver,
        )
        if ckey in cache:
            return cache[ckey], _filter_row(cache[ckey], filt, ext, cache, ckey)

    owner = None
    try:
        if fname:
            path = os.path.join(base_dir, fname)
            if not (os.path.exists(path) and _within_subtree(path, base_dir)):
                warnings.warn(
                    f"External reference {fname!r} not found within "
                    f"the source directory; ignoring."
                )
                return None, None
            owner = fits.open(path)
            src = owner
        else:
            src = hdul

        try:
            if ext is None:
                warnings.warn(f"Reference {s!r} has no extension; ignoring.")
                return None, None
            key = (ext, int(ver)) if (ver and ver.strip().isdigit()) else ext
            hdu = _detach_hdu(src[key])  # detach from the file
        except (KeyError, IndexError):
            warnings.warn(f"Reference target [{ext}] not found; ignoring.")
            return None, None
        if cache is not None:
            cache[ckey] = hdu

        return hdu, _filter_row(hdu, filt, ext, cache, ckey)
    finally:
        if owner is not None:
            owner.close()


def _filter_row(hdu, filt, ext, cache=None, ckey=None):
    """Row index selected by a ``[NAME=='..']`` / ``[n]`` filter, else None."""
    if not filt:
        return None
    nm = _NAME_FILTER_RE.search(filt)
    if nm and hdu.data is not None and "NAME" in getattr(hdu.columns, "names", []):
        index = None if cache is None else cache.get(("names", ckey))
        if index is None:
            index = {}
            for i, x in enumerate(hdu.data["NAME"]):
                index.setdefault(_as_str(x).strip(), i)  # first match wins
            if cache is not None:
                cache[("names", ckey)] = index
        row = index.get(nm.group(1).strip())
        if row is None:
            warnings.warn(f"No NAME=={nm.group(1)!r} row in [{ext}]; using first row.")
            row = 0
        return row
    if filt.strip().isdigit():
        return int(filt)
    return None


# ---------------------------------------------------------- image reading -----
def read_fits_image(hdu_or_path, scale_arcsec=None, ext=0):
    """Read a 2-D FITS image -> (image, scale_arcsec, center_deg, rotation_deg).

    Angular scale is resolved WCS -> header (``CDELT``/``IMGSCAL``) ->
    ``scale_arcsec`` kwarg. Raises if none of the three is available. Center is
    the WCS ``CRVAL`` when present, else ``(0, 0)``.
    """
    own = None
    if isinstance(hdu_or_path, str):
        own = fits.open(hdu_or_path)
        hdu = _first_image_hdu(own, ext)
    else:
        hdu = hdu_or_path
    try:
        img = np.asarray(hdu.data, float)
        if img.ndim != 2:
            raise ValueError(f"Expected a 2-D image, got shape {img.shape}.")
        hdr = hdu.header
        scale, center, rot = _image_scale_center(hdr, img.shape, scale_arcsec)
        if _east_left(hdr):
            img = img[:, ::-1]  # xrisalis convention: columns -> +RA
        return img, scale, center, rot
    finally:
        if own is not None:
            own.close()


def _first_image_hdu(hdul, ext=0):
    if ext:
        return hdul[ext]
    for h in hdul:
        if getattr(h, "is_image", False) and h.data is not None:
            return h
    raise ValueError("No image HDU with data found.")


def _wcs_celestial(hdr):
    """Return a celestial WCS if the header has one, else None (quietly)."""
    try:
        w = WCS(hdr)
        if w.has_celestial:
            return w.celestial
    except Exception:
        pass
    return None


def _image_scale_center(hdr, shape, scale_arcsec):
    """(scale_arcsec, center_deg, rotation_deg) via WCS -> header -> kwarg."""
    ny, nx = shape
    w = _wcs_celestial(hdr)
    if w is not None:
        px = proj_plane_pixel_scales(w) * 3600.0  # deg/pix -> arcsec
        scale = float(np.max(px)) * max(nx, ny)  # largest-axis span
        try:
            cen = w.wcs_pix2world([[(nx - 1) / 2.0, (ny - 1) / 2.0]], 0)[0]
            center = (float(cen[0]), float(cen[1]))
        except Exception:
            center = (0.0, 0.0)
        rot = _wcs_rotation_deg(w, flip=_east_left(hdr))
        return scale, center, rot
    # header fallback: CDELT (deg) or IMGSCAL (arcsec/pix, SIMPUT convention)
    cdelt = hdr.get("CDELT1") or hdr.get("CDELT2")
    if cdelt:
        scale = abs(float(cdelt)) * 3600.0 * max(nx, ny)
        return (
            scale,
            (float(hdr.get("CRVAL1", 0.0)), float(hdr.get("CRVAL2", 0.0))),
            0.0,
        )
    if hdr.get("IMGSCAL"):
        scale = abs(float(hdr["IMGSCAL"])) * max(nx, ny)
        return scale, (0.0, 0.0), 0.0
    if scale_arcsec is not None:
        return float(scale_arcsec), (0.0, 0.0), 0.0
    raise ValueError(
        "Image has no WCS, no CDELT/IMGSCAL header, and no scale_arcsec was "
        "given; cannot determine the angular scale."
    )


def _wcs_rotation_deg(w, flip=False):
    """Rotation angle [deg] of a celestial WCS (0 if not rotated).

    ``flip`` = the image's columns are mirrored to run towards +RA (see
    :func:`_east_left`); the angle is then measured on the mirrored matrix.
    """
    try:
        cd = np.array(w.pixel_scale_matrix, float)
        if flip:
            cd[:, 0] = -cd[:, 0]
        # ImageMorphology rotates the array with scipy.ndimage.rotate, whose
        # positive sense is opposite to the WCS angle measured here
        return float(-np.degrees(np.arctan2(cd[1, 0], cd[1, 1])))
    except Exception:
        return 0.0


def _east_left(hdr):
    """True if the image's column axis runs towards decreasing RA (East left).

    That is the standard FITS orientation (``CDELT1 < 0`` / negative CD
    determinant). xrisalis' :class:`~xrisalis.sources.ImageMorphology` maps
    columns to +RA, so such images are mirrored on read to keep the sky right.
    """
    w = _wcs_celestial(hdr)
    if w is not None:
        try:
            return bool(np.linalg.det(w.pixel_scale_matrix) < 0)
        except Exception:
            return False
    cdelt = hdr.get("CDELT1")
    return bool(cdelt is not None and float(cdelt) < 0)


# ----------------------------------------------------------- cube reading -----
def read_fits_cube(path):
    """Read a spectral cube -> (energies_kev, images, scale_arcsec, center_deg).

    ``images`` has shape ``(n_energy, ny, nx)``. Per-slice energies come from a
    spectral WCS third axis (``CTYPE3`` ENER/FREQ/WAVE, normalized to keV) when
    present, else an ``ENERGIES`` table extension. Angular scale/center from the
    celestial WCS (falling back to CDELT/IMGSCAL like :func:`read_fits_image`).
    Slices are returned in xrisalis' sky orientation (columns -> +RA, rows ->
    +Dec): East-left images are mirrored and a WCS rotation is applied to the
    pixels, so the returned cube needs no further rotation.
    """
    with fits.open(path) as hdul:
        cube_hdu = None
        for h in hdul:
            if (
                getattr(h, "is_image", False)
                and h.data is not None
                and np.asarray(h.data).ndim == 3
            ):
                cube_hdu = h
                break
        if cube_hdu is None:
            raise ValueError("No 3-D image HDU found for a spectral cube.")
        cube = np.asarray(cube_hdu.data, float)  # (nE, ny, nx)
        hdr = cube_hdu.header
        n_e, ny, nx = cube.shape

        scale, center, rot = _image_scale_center(hdr, (ny, nx), None)
        if _east_left(hdr):
            cube = cube[:, :, ::-1]  # xrisalis convention: columns -> +RA
        if rot:
            # bake the WCS rotation into the slices, exactly as ImageMorphology
            # does for a single image (scipy sense, bilinear, flux >= 0)
            from scipy.ndimage import rotate as _rotate  # lazy

            cube = np.clip(
                _rotate(
                    cube,
                    rot,
                    axes=(2, 1),
                    reshape=False,
                    order=1,
                    mode="constant",
                    cval=0.0,
                ),
                0,
                None,
            )

        energies = _cube_energies(hdr, hdul, n_e)
        return energies, cube, scale, center


def _cube_energies(hdr, hdul, n_e):
    """Per-slice energies [keV]: spectral WCS axis 3, else an ENERGIES table."""
    ctype3 = str(hdr.get("CTYPE3", "")).upper()
    if ctype3.startswith(("ENER", "FREQ", "WAVE", "AWAV")):
        try:
            w = WCS(hdr)
            pix = np.zeros((n_e, w.naxis))
            pix[:, 2] = np.arange(n_e)
            world = w.wcs_pix2world(pix, 0)[:, 2]
            unit = (
                w.wcs.cunit[2].to_string() if w.wcs.cunit[2] else hdr.get("CUNIT3", "")
            )
            return energy_to_kev(world, unit)
        except Exception:
            warnings.warn(
                "Spectral WCS on axis 3 could not be evaluated; "
                "falling back to an ENERGIES table."
            )
    if "ENERGIES" in hdul:
        et = hdul["ENERGIES"]
        col = "ENERGY" if "ENERGY" in et.columns.names else et.columns.names[0]
        unit = et.columns[col].unit or "keV"
        return energy_to_kev(np.asarray(et.data[col], float), unit)
    warnings.warn(
        "Cube has neither a spectral WCS nor an ENERGIES table; "
        "assuming 1 keV per slice."
    )
    return np.ones(n_e)


# --------------------------------------------------------- timing reading -----
def _time_unit_to_seconds(hdr):
    """Seconds-per-``TIMEUNIT`` for a timing HDU (default 1 = seconds)."""
    unit = str(hdr.get("TIMEUNIT", "s")).strip().lower()
    if unit in ("s", "sec", "secs", "second", "seconds", ""):
        return 1.0
    if unit in ("d", "day", "days"):
        return 86400.0
    if unit in ("ks",):
        return 1e3
    warnings.warn(f"Unknown TIMEUNIT {unit!r}; assuming seconds.")
    return 1.0


def read_timing(hdu):
    """Parse a SIMPUT TIMING extension into a light-curve descriptor dict.

    Returns ``{'kind': 'binned', 'time': [s], 'flux': [-]}`` for a
    ``(TIME, FLUX/RATE)`` light curve, or ``{'kind': 'periodic', 'phase': [-],
    'flux': [-], 'period': [s]}`` for a ``(PHASE, FLUX)`` table with a ``PERIOD``
    keyword. Returns ``None`` (with a warning) if it cannot be interpreted.

    Time-system keywords are honored: ``TIMEUNIT`` (s/d/ks) scales the ``TIME``
    axis and ``PERIOD`` to seconds, and ``TIMEZERO`` shifts the axis. The
    resulting ``time`` array is anchored to its own start (so an absolute
    ``MJDREF``-based reference is applied as a shape from the exposure start).
    """
    if hdu is None or hdu.data is None:
        return None
    names = [n.upper() for n in hdu.columns.names]
    hdr = hdu.header
    flux_col = next(
        (c for c in hdu.columns.names if c.upper() in ("FLUX", "RATE")), None
    )
    if "PHASE" in names and flux_col is not None:
        period = hdr.get("PERIOD")
        if period is None:
            warnings.warn(
                "Periodic TIMING extension has no PERIOD keyword; treating as steady."
            )
            return None
        return {
            "kind": "periodic",
            "phase": np.asarray(
                hdu.data[hdu.columns.names[names.index("PHASE")]], float
            ),
            "flux": np.asarray(hdu.data[flux_col], float),
            "period": float(period) * _time_unit_to_seconds(hdr),
            "phase0": float(hdr.get("PHASE0", 0.0)),
        }
    if "TIME" in names and flux_col is not None:
        tcol = hdu.columns.names[names.index("TIME")]
        scale = _time_unit_to_seconds(hdr)
        t = np.asarray(hdu.data[tcol], float) + float(hdr.get("TIMEZERO", 0.0))
        t = t * scale
        t = t - t.min()  # anchor to the light-curve start [s]
        return {
            "kind": "binned",
            "time": t,
            "flux": np.asarray(hdu.data[flux_col], float),
        }
    warnings.warn("Unrecognized TIMING extension; ignoring.")
    return None


# ------------------------------------------------------------------ SIMPUT ---
@dataclass
class SimputEntry:
    """One resolved SIMPUT source: position, spectrum, optional image + timing."""

    src_id: int
    ra_deg: float
    dec_deg: float
    energy_kev: np.ndarray
    fluxdensity: np.ndarray
    flux: float = None  # SRC_CAT FLUX [erg/s/cm^2] over [e_min,e_max]
    e_min: float = None
    e_max: float = None
    image: np.ndarray = None  # brightness array (extended sources)
    image_scale_arcsec: float = None
    image_center_deg: tuple = None
    image_rotation_deg: float = 0.0
    timing: dict = None  # descriptor from read_timing (or None)


@dataclass
class SimputSource:
    """SIMPUT catalog reader.

    ``entries`` holds the fully-resolved per-source records (positions, spectra,
    optional images + light curves). The ``ra_deg``/``dec_deg``/``spectra``
    fields and ``n_sources`` are kept for backward compatibility with the
    point-catalog fast path in :func:`xrisalis.simulate.generate_photons`.
    """

    src_id: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    spectra: list  # list of (energy_kev, fluxdensity) per source
    entries: list = field(default_factory=list)

    @property
    def n_sources(self):
        return len(self.src_id)

    @classmethod
    def read(cls, path):
        """Read a SIMPUT catalog, resolving SPECTRUM/IMAGE/TIMING references.

        SPECTRUM/IMAGE/TIMING columns are resolved via
        :func:`resolve_reference` (by NAME / ``[EXT]`` / external ``file[EXT]``),
        so spectra map to the right source regardless of table order, extended
        (IMAGE) sources keep their morphology, and variable (TIMING) sources
        keep their light curve. Missing/unresolved references warn and fall back
        (flat spectrum / steady / point).
        """
        base_dir = os.path.dirname(os.path.abspath(path))
        entries = []
        cache = {}  # referenced HDUs, resolved once
        with fits.open(path) as hdul:
            cat = hdul["SRC_CAT"].data
            cols = cat.columns.names

            def cell(row, key, default=None):
                return row[key] if key in cols else default

            for row in cat:
                sid = int(cell(row, "SRC_ID", 0))
                ra = float(cell(row, "RA", 0.0))
                dec = float(cell(row, "DEC", 0.0))
                e_min = _f(cell(row, "E_MIN"))
                e_max = _f(cell(row, "E_MAX"))
                flux = _f(cell(row, "FLUX"))

                E, F = _resolve_spectrum(cell(row, "SPECTRUM"), hdul, base_dir, cache)
                img = scale = center = None
                rot = 0.0
                imref = cell(row, "IMAGE")
                if imref is not None and _as_str(imref).strip().upper() not in (
                    "",
                    "NULL",
                ):
                    img, scale, center, rot = _resolve_image(
                        imref,
                        hdul,
                        base_dir,
                        ra,
                        dec,
                        _f(cell(row, "IMGSCAL"), 1.0),
                        _f(cell(row, "IMGROTA"), 0.0),
                    )
                timing = None
                tmref = cell(row, "TIMING")
                if tmref is not None and _as_str(tmref).strip().upper() not in (
                    "",
                    "NULL",
                ):
                    thdu, _ = resolve_reference(tmref, hdul, base_dir, cache)
                    timing = read_timing(thdu)

                entries.append(
                    SimputEntry(
                        src_id=sid,
                        ra_deg=ra,
                        dec_deg=dec,
                        energy_kev=E,
                        fluxdensity=F,
                        flux=flux,
                        e_min=e_min,
                        e_max=e_max,
                        image=img,
                        image_scale_arcsec=scale,
                        image_center_deg=center,
                        image_rotation_deg=rot,
                        timing=timing,
                    )
                )

        src_id = np.array([e.src_id for e in entries])
        ra = np.array([e.ra_deg for e in entries], float)
        dec = np.array([e.dec_deg for e in entries], float)
        spectra = [(e.energy_kev, e.fluxdensity) for e in entries]
        return cls(src_id, ra, dec, spectra, entries)


def _f(value, default=None):
    """Best-effort float of a FITS cell, else ``default``."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_spectrum(ref, hdul, base_dir, cache=None):
    """(energy_kev, fluxdensity) for a SPECTRUM reference; flat + warn on miss."""
    hdu, row = resolve_reference(ref, hdul, base_dir, cache)
    if hdu is None or hdu.data is None:
        warnings.warn(
            "Source has no resolvable SPECTRUM; using a flat 1-10 keV spectrum."
        )
        E = np.linspace(0.5, 10.0, 100)
        return E, np.ones_like(E)
    r = 0 if row is None else row
    e_unit = _col_unit(hdu, "ENERGY") or "keV"
    E = energy_to_kev(np.asarray(hdu.data["ENERGY"][r], float), e_unit)
    F = np.asarray(hdu.data["FLUXDENSITY"][r], float)
    return E, F


def _resolve_image(ref, hdul, base_dir, ra, dec, imgscal, imgrota):
    """(image, scale_arcsec, center_deg, rotation_deg) for an IMAGE reference.

    The source's catalog position sets the on-sky center; ``IMGSCAL`` multiplies
    the image's own angular size and ``IMGROTA`` adds to any WCS rotation."""
    hdu, _ = resolve_reference(ref, hdul, base_dir)
    if hdu is None or hdu.data is None:
        warnings.warn(
            "Source IMAGE reference did not resolve; treating as a point source."
        )
        return None, None, None, 0.0
    img, scale, _wcs_center, rot = read_fits_image(hdu)
    scale = scale * float(imgscal if imgscal else 1.0)
    return img, scale, (ra, dec), float(rot) + float(imgrota or 0.0)


def _col_unit(hdu, name):
    try:
        return hdu.columns[name].unit
    except Exception:
        return None
