"""Image evaluation: compare a reconstructed image to the true source model.

The diagram's *Image evaluation* box. Given a reconstructed image ``I(alpha_d,
delta_d)`` and knowledge of the true sources (positions + fluxes, e.g. from a
SimputSource), this computes interpretable quality metrics:

* astrometry   — recovered peak positions and their separation vs. the truth;
* dynamic range — peak brightness over the background RMS (image cleanliness);
* photometry   — recovered flux ratio of the two brightest peaks vs. the truth;
* fidelity     — cross-correlation and residual RMS against a model image
                 rendered at the *reconstruction's own resolution* (so a
                 perfect-but-blurred image scores well, not penalised for the
                 finite beam).

Pure NumPy — no astropy needed (it reads plain arrays / a SimputSource's public
attributes), so it is testable without a FITS backend. Coordinate convention
matches :func:`xrisalis.reconstruct.reconstruct`: ``image[row, col]`` with
alpha along columns, delta along rows, centre at the array centre, span in
micro-arcseconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .units import to_uas
from .coords import nominal_pointing, tangent_offsets

RAD2UAS = 180.0 / np.pi * 3600e6


# --------------------------------------------------------------------- coords
def _pix_scale(span_uas, npix):
    return span_uas / npix


def _rc_to_uas(row, col, span_uas, npix):
    pix = _pix_scale(span_uas, npix)
    c0 = (npix - 1) / 2.0
    return (col - c0) * pix, (row - c0) * pix        # (alpha, delta) [uas]


def _uas_to_rc(alpha, delta, span_uas, npix):
    pix = _pix_scale(span_uas, npix)
    c0 = (npix - 1) / 2.0
    return c0 + delta / pix, c0 + alpha / pix         # (row, col)


# ------------------------------------------------------------- true sources
def _offsets_uas(ra_deg, dec_deg, center_deg):
    east, north = tangent_offsets(ra_deg, dec_deg, *center_deg)
    return np.column_stack((east * RAD2UAS, north * RAD2UAS))


def true_sources(source, center_deg=None):
    """Extract (positions_uas (m,2)=[alpha,delta], fluxes (m,)) from a source.

    ``source`` may be a :class:`xrisalis.sources.SourceModel` (uses component
    centroids + integrated fluxes), a SimputSource (ra_deg/dec_deg + spectra), or
    a plain ``(positions_uas, fluxes)`` tuple. Positions are tangent-plane
    offsets (East, North) from ``center_deg`` = ``(ra, dec)`` [deg], the image's
    phase centre (``meta['phase_center_deg']`` from
    :func:`xrisalis.reconstruct.reconstruct`); default: the source's nominal
    pointing (:func:`xrisalis.coords.nominal_pointing`), which is also the
    default image centre.
    """
    if isinstance(source, tuple):
        pos, flux = source
        return np.asarray(pos, float), np.asarray(flux, float)
    if center_deg is None:
        center_deg = nominal_pointing(source)
    if hasattr(source, "positions_deg") and hasattr(source, "fluxes"):
        pos_deg = np.atleast_2d(source.positions_deg())    # (m,2) [ra,dec] deg
        return (_offsets_uas(pos_deg[:, 0], pos_deg[:, 1], center_deg),
                np.asarray(source.fluxes(), float))
    pos = _offsets_uas(source.ra_deg, source.dec_deg, center_deg)
    spectra = getattr(source, "spectra", None)
    if spectra:
        # trapezoidal integral of each spectrum (np.trapz renamed in numpy 2.x)
        flux = np.array([0.5 * np.sum((np.asarray(f)[1:] + np.asarray(f)[:-1])
                                      * np.diff(np.asarray(e))) for e, f in spectra],
                        float)
    else:
        flux = np.ones(len(pos))
    return pos, flux


# ------------------------------------------------------------- peak finding
def find_peaks(image, span_uas, n=2, min_sep_uas=None):
    """Return the ``n`` brightest, mutually separated peaks.

    Returns (positions_uas (k,2), amplitudes (k,)) sorted by brightness.
    """
    im = np.maximum(np.asarray(image, float), 0.0)
    npix = im.shape[0]
    pix = _pix_scale(span_uas, npix)
    guard = max(1, int((min_sep_uas / pix) if min_sep_uas else npix // 12))
    work = im.copy()
    rows, cols, amps = [], [], []
    rr, cc = np.mgrid[0:npix, 0:npix]
    for _ in range(n):
        idx = np.unravel_index(np.argmax(work), work.shape)
        if work[idx] <= 0:
            break
        rows.append(idx[0]); cols.append(idx[1]); amps.append(im[idx])
        work[(np.abs(rr - idx[0]) <= guard) & (np.abs(cc - idx[1]) <= guard)] = 0
    pos = np.array([_rc_to_uas(r, c, span_uas, npix) for r, c in zip(rows, cols)])
    return pos, np.array(amps)


def _match(recovered, truth):
    """Greedy nearest-neighbour match truth->recovered; return per-truth error."""
    errs = []
    used = set()
    for t in truth:
        d = np.array([np.hypot(*(r - t)) if i not in used else np.inf
                      for i, r in enumerate(recovered)])
        j = int(np.argmin(d))
        used.add(j)
        errs.append(float(d[j]))
    return np.array(errs)


# ------------------------------------------------------------- model image
def _estimate_fwhm_uas(image, span_uas):
    """FWHM [uas] of the brightest peak from its second moment (local window)."""
    im = np.maximum(np.asarray(image, float), 0.0)
    npix = im.shape[0]
    pix = _pix_scale(span_uas, npix)
    r0, c0 = np.unravel_index(np.argmax(im), im.shape)
    w = max(3, npix // 12)
    r1, r2 = max(0, r0 - w), min(npix, r0 + w + 1)
    c1, c2 = max(0, c0 - w), min(npix, c0 + w + 1)
    sub = im[r1:r2, c1:c2]
    rr, cc = np.mgrid[r1:r2, c1:c2]
    tot = sub.sum()
    if tot <= 0:
        return 2 * pix
    sig_r = np.sqrt(max((sub * (rr - r0) ** 2).sum() / tot, 0.0))
    sig_c = np.sqrt(max((sub * (cc - c0) ** 2).sum() / tot, 0.0))
    sigma_pix = 0.5 * (sig_r + sig_c)
    return max(sigma_pix, 0.75) * pix * 2.3548          # sigma->FWHM


def model_image(positions_uas, fluxes, span_uas, npix, fwhm_uas):
    """Render flux-weighted Gaussians at source positions onto the recon grid."""
    sigma = fwhm_uas / 2.3548
    img = np.zeros((npix, npix), float)
    ax = (np.arange(npix) - (npix - 1) / 2.0) * _pix_scale(span_uas, npix)
    X, Y = np.meshgrid(ax, ax)                          # X=alpha(col), Y=delta(row)
    for (a, d), f in zip(positions_uas, np.atleast_1d(fluxes)):
        img += f * np.exp(-((X - a) ** 2 + (Y - d) ** 2) / (2 * sigma ** 2))
    return img


# ------------------------------------------------------------- result
@dataclass
class EvaluationResult:
    n_peaks: int
    peaks_uas: np.ndarray            # (k,2) recovered [alpha, delta]
    peak_amp: np.ndarray             # (k,) recovered amplitudes
    separation_uas: float            # recovered (two brightest), nan if <2
    true_separation_uas: float
    position_error_uas: np.ndarray   # per true source (nan if truth unknown)
    dynamic_range: float
    flux_ratio: float                # recovered faint/bright
    true_flux_ratio: float
    fidelity: float                  # cross-correlation with model image
    residual_rms: float
    fwhm_uas: float
    meta: dict = field(default_factory=dict)

    def summary(self):
        L = [f"Image evaluation ({self.n_peaks} peaks, beam FWHM "
             f"{self.fwhm_uas:.0f} uas):"]
        if np.isfinite(self.separation_uas):
            L.append(f"  separation : {self.separation_uas:.0f} uas "
                     f"(truth {self.true_separation_uas:.0f})")
        if np.isfinite(self.position_error_uas).any():
            L.append(f"  astrometry : mean pos. error "
                     f"{np.nanmean(self.position_error_uas):.0f} uas")
        L.append(f"  dyn. range : {self.dynamic_range:.1f}")
        if np.isfinite(self.flux_ratio):
            L.append(f"  flux ratio : {self.flux_ratio:.2f} "
                     f"(truth {self.true_flux_ratio:.2f})")
        if np.isfinite(self.fidelity):
            L.append(f"  fidelity   : {self.fidelity:.3f} "
                     f"(residual RMS {self.residual_rms:.3f})")
        return "\n".join(L)


# ------------------------------------------------------------- main entry
def evaluate(image, span_uas, source=None, n_sources=2, min_sep_uas=None,
             center_deg=None):
    """Evaluate a reconstructed image against the true source model.

    Parameters
    ----------
    image : 2-D ndarray            reconstructed image (as from reconstruct()).
    span_uas : float               angular span of the image [micro-arcsec].
    source : SimputSource | (positions_uas, fluxes) | None
        The truth. If None, only source-agnostic metrics (peaks, separation,
        dynamic range) are returned; truth-relative fields are NaN.
    n_sources : int                number of peaks to recover.
    center_deg : (ra, dec) or None  image phase centre [deg] (see true_sources).

    Returns
    -------
    EvaluationResult
    """
    span_uas = to_uas(span_uas)                      # accept an angle Quantity
    im = np.asarray(image, float)
    npix = im.shape[0]
    peaks, amps = find_peaks(im, span_uas, n=n_sources, min_sep_uas=min_sep_uas)

    sep = (np.hypot(*(peaks[0] - peaks[1])) if len(peaks) >= 2 else np.nan)
    flux_ratio = (min(amps[:2]) / max(amps[:2]) if len(amps) >= 2 else np.nan)

    # dynamic range: peak / RMS of the background (outside source regions)
    imc = np.maximum(im, 0.0)
    pix = _pix_scale(span_uas, npix)
    rr, cc = np.mgrid[0:npix, 0:npix]
    guard = max(1, npix // 12)
    srcmask = np.zeros((npix, npix), bool)
    for (a, d) in peaks:
        r, c = _uas_to_rc(a, d, span_uas, npix)
        srcmask |= (np.abs(rr - r) <= guard) & (np.abs(cc - c) <= guard)
    bg = imc[~srcmask]
    bg_rms = np.sqrt(np.mean(bg ** 2)) if bg.size else np.nan
    dyn = (imc.max() / bg_rms) if bg_rms and bg_rms > 0 else np.inf

    # truth-relative metrics
    true_sep = np.nan; true_ratio = np.nan
    pos_err = np.full(len(peaks), np.nan)
    fidelity = np.nan; resid = np.nan
    fwhm = _estimate_fwhm_uas(im, span_uas)
    if source is not None:
        tpos, tflux = true_sources(source, center_deg)
        if len(tpos) >= 2:
            order = np.argsort(tflux)[::-1]
            true_sep = float(np.hypot(*(tpos[order[0]] - tpos[order[1]])))
            true_ratio = float(min(tflux[order[:2]]) / max(tflux[order[:2]]))
        pos_err = _match(peaks, tpos) if len(peaks) else pos_err
        # fidelity: model rendered at the recon resolution
        model = model_image(tpos, tflux, span_uas, npix, fwhm)
        a = imc / (imc.max() or 1.0)
        b = model / (model.max() or 1.0)
        fidelity = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        resid = float(np.sqrt(np.mean((a - b) ** 2)))

    return EvaluationResult(
        n_peaks=len(peaks), peaks_uas=peaks, peak_amp=amps,
        separation_uas=sep, true_separation_uas=true_sep,
        position_error_uas=pos_err, dynamic_range=float(dyn),
        flux_ratio=float(flux_ratio), true_flux_ratio=true_ratio,
        fidelity=fidelity, residual_rms=resid, fwhm_uas=float(fwhm))
