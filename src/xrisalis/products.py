"""Data products from the event list: spectra and light curves.

Both are written in the community **OGIP FITS** layouts (PHA-style spectrum,
RATE light curve) so they load in external tools, but nothing here depends on
those tools — computation is pure NumPy and the writer reuses the FITS helpers
in :mod:`xrisalis.io`. Plotting lives in ``viz`` (kept separate); these classes
only compute and read/write.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

from .io import col, provenance_header, write_hdulist


# ============================================================================
# Spectrum  (OGIP PHA)
# ============================================================================
@dataclass
class Spectrum:
    """Counts-per-channel spectrum (OGIP ``SPECTRUM``/PHA)."""

    channel: np.ndarray  # channel numbers
    counts: np.ndarray  # counts per channel
    e_min: np.ndarray = None  # channel lower energy [keV] (optional)
    e_max: np.ndarray = None  # channel upper energy [keV]
    exposure: float = None
    baseline_id: int = None  # None = all baselines
    respfile: str = None
    ancrfile: str = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.channel = np.asarray(self.channel)
        self.counts = np.asarray(self.counts)

    @property
    def energy(self):
        """Channel mid energy [keV] (None if EBOUNDS not attached)."""
        return (
            None
            if self.e_min is None
            else 0.5 * (np.asarray(self.e_min) + np.asarray(self.e_max))
        )

    @property
    def rate(self):
        """Counts/s per channel (None if no exposure)."""
        return None if not self.exposure else self.counts / self.exposure

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_events(cls, events, baseline_id=None, respfile=None, ancrfile=None):
        mask = np.ones(len(events), bool)
        if baseline_id is not None:
            mask &= events.baseline_id == baseline_id
        pi = np.asarray(events.pi)[mask]
        eb = getattr(events, "ebounds", None)
        if eb is not None:
            channel = np.asarray(eb["CHANNEL"])
            e_min = np.asarray(eb["E_MIN"], float)
            e_max = np.asarray(eb["E_MAX"], float)
            nchan = len(channel)
            chan0 = int(channel.min()) if nchan else 0
        else:
            valid = pi[pi >= 0]
            nchan = int(valid.max()) + 1 if valid.size else 1
            channel = np.arange(nchan)
            e_min = e_max = None
            chan0 = 0
        good = pi >= 0
        counts = np.bincount(pi[good] - chan0, minlength=nchan)[:nchan]
        return cls(
            channel=channel,
            counts=counts.astype(np.int64),
            e_min=e_min,
            e_max=e_max,
            exposure=events.exposure,
            baseline_id=baseline_id,
            respfile=respfile,
            ancrfile=ancrfile,
        )

    # -- io -------------------------------------------------------------------
    def _spectrum_hdu(self):
        cols = [
            col("CHANNEL", "J", self.channel),
            col("COUNTS", "J", self.counts, "count"),
        ]
        h = provenance_header(history="Spectrum (products stage)")
        h["EXTNAME"] = "SPECTRUM"
        h["HDUCLASS"] = "OGIP"
        h["HDUCLAS1"] = "SPECTRUM"
        h["HDUCLAS2"] = "TOTAL"
        h["HDUCLAS3"] = "COUNT"
        h["HDUVERS"] = "1.2.1"
        h["CHANTYPE"] = "PI"
        h["DETCHANS"] = len(self.channel)
        h["POISSERR"] = True
        h["AREASCAL"] = 1.0
        h["BACKSCAL"] = 1.0
        h["CORRSCAL"] = 1.0
        h["BACKFILE"] = "NONE"
        h["CORRFILE"] = "NONE"
        h["RESPFILE"] = self.respfile or "NONE"
        h["ANCRFILE"] = self.ancrfile or "NONE"
        if len(self.channel):
            h["TLMIN1"] = int(np.min(self.channel))
            h["TLMAX1"] = int(np.max(self.channel))
        if self.exposure is not None:
            h["EXPOSURE"] = float(self.exposure)
        if self.baseline_id is not None:
            h["BASELINE"] = (int(self.baseline_id), "XRI baseline index (-1=all)")
        return fits.BinTableHDU.from_columns(cols, header=h)

    def _ebounds_hdu(self):
        cols = [
            col("CHANNEL", "J", self.channel),
            col("E_MIN", "E", self.e_min, "keV"),
            col("E_MAX", "E", self.e_max, "keV"),
        ]
        h = fits.Header()
        h["EXTNAME"] = "EBOUNDS"
        h["HDUCLASS"] = "OGIP"
        h["HDUCLAS1"] = "RESPONSE"
        h["HDUCLAS2"] = "EBOUNDS"
        return fits.BinTableHDU.from_columns(cols, header=h)

    def write(self, path, overwrite=True):
        hdus = [self._spectrum_hdu()]
        if self.e_min is not None:
            hdus.append(self._ebounds_hdu())
        write_hdulist(hdus, path, overwrite=overwrite)
        return path

    @classmethod
    def read(cls, path):
        with fits.open(path) as hdul:
            d = hdul["SPECTRUM"].data
            hdr = hdul["SPECTRUM"].header
            e_min = e_max = None
            if "EBOUNDS" in hdul:
                ed = hdul["EBOUNDS"].data
                e_min = np.asarray(ed["E_MIN"], float)
                e_max = np.asarray(ed["E_MAX"], float)
            bl = hdr.get("BASELINE")
            return cls(
                channel=np.asarray(d["CHANNEL"]),
                counts=np.asarray(d["COUNTS"]),
                e_min=e_min,
                e_max=e_max,
                exposure=hdr.get("EXPOSURE"),
                baseline_id=None if bl in (None, -1) else int(bl),
                respfile=hdr.get("RESPFILE"),
                ancrfile=hdr.get("ANCRFILE"),
            )


# ============================================================================
# LightCurve  (OGIP RATE)
# ============================================================================
@dataclass
class LightCurve:
    """Binned light curve (OGIP ``RATE``)."""

    time: np.ndarray  # bin-center time [s]
    rate: np.ndarray  # counts/s
    error: np.ndarray  # counts/s (Poisson)
    timedel: float  # bin width [s]
    exposure: float = None
    energy_kev: tuple = None
    baseline_bounds: tuple = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_events(
        cls, events, dt, energy_kev=None, baseline_bounds=None, tstart=0.0, tstop=None
    ):
        mask = np.ones(len(events), bool)
        if energy_kev is not None:
            ek = events.energy_kev
            mask &= (ek > energy_kev[0]) & (ek < energy_kev[1])
        if baseline_bounds is not None:
            mask &= (events.baseline_id >= baseline_bounds[0]) & (
                events.baseline_id <= baseline_bounds[1]
            )
        t = np.asarray(events.time)[mask]
        if tstop is None:
            tstop = (
                float(events.exposure)
                if events.exposure
                else (float(t.max()) if t.size else dt)
            )
        edges = np.arange(tstart, tstop + dt, dt)
        counts, _ = np.histogram(t, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return cls(
            time=centers,
            rate=counts / dt,
            error=np.sqrt(counts) / dt,
            timedel=float(dt),
            exposure=float(tstop - tstart),
            energy_kev=energy_kev,
            baseline_bounds=baseline_bounds,
        )

    def _rate_hdu(self):
        cols = [
            col("TIME", "D", self.time, "s"),
            col("RATE", "E", self.rate, "count/s"),
            col("ERROR", "E", self.error, "count/s"),
        ]
        h = provenance_header(history="LightCurve (products stage)")
        h["EXTNAME"] = "RATE"
        h["HDUCLASS"] = "OGIP"
        h["HDUCLAS1"] = "LIGHTCURVE"
        h["HDUCLAS2"] = "ALL"
        h["HDUVERS"] = "1.1.0"
        h["TIMEDEL"] = float(self.timedel)
        h["TIMEUNIT"] = "s"
        h["TIMESYS"] = "TT"
        if self.exposure is not None:
            h["EXPOSURE"] = float(self.exposure)
        if self.energy_kev is not None:
            h["E_MIN"] = float(self.energy_kev[0])
            h["E_MAX"] = float(self.energy_kev[1])
        return fits.BinTableHDU.from_columns(cols, header=h)

    def write(self, path, overwrite=True):
        write_hdulist([self._rate_hdu()], path, overwrite=overwrite)
        return path

    @classmethod
    def read(cls, path):
        with fits.open(path) as hdul:
            d = hdul["RATE"].data
            hdr = hdul["RATE"].header
            ek = None
            if "E_MIN" in hdr and "E_MAX" in hdr:
                ek = (hdr["E_MIN"], hdr["E_MAX"])
            return cls(
                time=np.asarray(d["TIME"], float),
                rate=np.asarray(d["RATE"], float),
                error=np.asarray(d["ERROR"], float),
                timedel=hdr.get("TIMEDEL"),
                exposure=hdr.get("EXPOSURE"),
                energy_kev=ek,
            )


# ============================================================================
# convenience API
# ============================================================================
def spectrum(events, baseline_id=None, **kw):
    """Total (or single-baseline) spectrum from an EventList."""
    return Spectrum.from_events(events, baseline_id=baseline_id, **kw)


def spectra_per_baseline(events, **kw):
    """dict {baseline_id: Spectrum} for every populated baseline."""
    return {
        int(b): Spectrum.from_events(events, baseline_id=int(b), **kw)
        for b in np.unique(events.baseline_id)
    }


def lightcurve(events, dt, energy_kev=None, baseline_bounds=None, **kw):
    """Binned light curve from an EventList (bin width ``dt`` seconds)."""
    return LightCurve.from_events(
        events, dt, energy_kev=energy_kev, baseline_bounds=baseline_bounds, **kw
    )
