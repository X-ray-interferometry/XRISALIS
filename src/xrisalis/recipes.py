"""Recipes for observing xrimodel models: simulate, reconstruct, and compare with the model.

The pipeline these wrap, step by step::

    source = xs.XrimodelSource(model)                               # xrimodel draws the photons
    photons = xs.generate_photons(source, arf, exposure)            # ... assigned to baselines
    events = xs.redistribute(photons, instrument, rmf)              # energies, fringes, attitude
    image, span, uv, meta = xs.reconstruct(events, instrument, fov_arcsec=...)

:func:`observe` runs it as a real observation would be analysed: images from the detected
energies, with ``reconstruct``'s matched-filter deconvolution of the chromatic decoherence
(``deconvolve`` undoes the energy-resolution smearing of the fringes, not the uv-coverage
sidelobes), on ~15 uas pixels. Compare a reconstruction with the model convolved with the
dirty beam (:func:`dirty_beam`, :func:`model_image`, :func:`predicted`), not with the model.

The instrument is a :class:`Setup` (ARF, RMF, interferometer); by default the 10-baseline
ARF and the RMF bundled with XRISALIS, with a spacecraft that rolls through half a turn
over the observation. Needs xrimodel (``import xrisalis.recipes``).
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np
import xrimodel as xm

from .instrument import Interferometer, smooth_roll
from .io import ARF, RMF
from .reconstruct import reconstruct, resolving_power
from .simulate import generate_photons, redistribute
from .sources import XrimodelSource

PIXEL_UAS = 15.0


@dataclass(frozen=True)
class Setup:
    """An instrument to observe with: effective areas per baseline, energy response and
    the interferometer (baselines and roll)."""

    arf: ARF
    rmf: RMF
    instrument: Interferometer

    @classmethod
    def from_files(cls, arf, rmf, roller=smooth_roll):
        a, r = ARF.read(str(arf)), RMF.read(str(rmf))
        return cls(a, r, Interferometer.from_arf(a, roller=roller))


@lru_cache(maxsize=1)
def bundled():
    """The 10-baseline ARF (0.05-1 m) and RMF bundled with XRISALIS, rolling half a turn."""
    resp = files("xrisalis") / "data" / "responses"
    return Setup.from_files(resp / "arf_10bl.fits", resp / "rmf.fits")


def pixels(fov_arcsec, pixel_uas=PIXEL_UAS):
    """``samples`` for ``reconstruct``: ~15 uas pixels (even count). Coarser pixels cannot
    sample the finest fringe (``reconstruct`` raises an IndexError); the default 256 samples
    on fields below ~1.5 mas need many GB for the IFFT window."""
    n = int(np.ceil(fov_arcsec * 1e6 / pixel_uas / 2)) * 2
    return (n, n)


def subset(model, keep):
    """A copy of ``model`` with only the layers whose names pass ``keep(name)``."""
    out = xm.Model(ra=model.ra, dec=model.dec, mjdref=model.mjdref)
    for d in model.drivers.values():
        out.add_driver(d)
    for layer in model.layers:
        if keep(layer.name):
            out.add(copy.deepcopy(layer))
    return out


def scaled(model, total_flux):
    """A copy of ``model`` with every layer scaled so the total is ``total_flux``
    photons/s/cm^2 (relative brightnesses kept)."""
    out = copy.deepcopy(model)
    f = total_flux / sum(layer.mean_photon_flux() for layer in out.layers)
    for layer in out.layers:
        if hasattr(layer, "maps"):
            layer.maps["NORM"] = layer.maps["NORM"] * f
        elif hasattr(layer, "spectrum"):
            layer.spectrum.values = layer.spectrum.values * f
        else:
            raise TypeError(f"cannot scale a {layer.kind} layer")
    return out


def observe(model, exposure, fov_arcsec, total_flux=None, seed=0, setup=None,
            **reconstruct_kw):
    """Simulate ``exposure`` seconds of ``model`` and reconstruct an image.

    ``total_flux`` (photons/s/cm^2) rescales the model first; by default its own fluxes are
    used. By default the image is reconstructed as from a real observation: from the
    detected energies, with ``deconvolve`` set to the detector's resolving power at the
    events' median energy, on ~15 uas pixels (:func:`pixels`). Any ``reconstruct`` option
    overrides these; ``use_true_energy=True`` gives the ideal image without detector
    smearing (and no deconvolution), as a diagnostic only.

    Returns a dict with ``image``, ``span`` (uas), ``uv`` (the sampled (u, v), rad^-1, both
    signs), ``events``, ``photons``, ``meta``, ``fov_arcsec``, the (scaled) ``model``, the
    ``setup`` and the ``reconstruct`` options used.
    """
    s = bundled() if setup is None else setup
    m = model if total_flux is None else scaled(model, total_flux)
    photons = generate_photons(XrimodelSource(m), s.arf, exposure, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # vignetting of photons beyond the field of view
        events = redistribute(photons, s.instrument, s.rmf, seed=seed + 1)
    kw = dict(reconstruct_kw)
    kw.setdefault("use_true_energy", False)
    if not kw["use_true_energy"] and "fringe_taper" not in kw:
        kw.setdefault("deconvolve", resolving_power(s.rmf, float(np.median(events.energy_kev))))
    kw.setdefault("samples", pixels(fov_arcsec))
    image, span, uv, meta = reconstruct(events, s.instrument, fov_arcsec=fov_arcsec, **kw)
    return {"image": image, "span": span, "uv": uv, "events": events, "photons": photons,
            "meta": meta, "fov_arcsec": fov_arcsec, "model": m, "setup": s,
            "reconstruct_kw": kw}


def total_spectrum(model, edges=None):
    """The model's spectrum summed over its separable and SpecMap layers (photons/s/cm^2/keV)."""
    edges = xm.energy_grid() if edges is None else edges
    tot = np.zeros(edges.size - 1)
    for layer in model.layers:
        if hasattr(layer, "maps"):
            tot += xm.specmap_projection(layer, edges=edges)[0].values
        elif hasattr(layer, "spectrum"):
            sp = layer.spectrum
            cum = np.concatenate([[0.0], np.cumsum(sp.values * sp.widths)])
            tot += np.diff(np.interp(edges, sp.edges, cum)) / np.diff(edges)
    return xm.Spectrum(edges, tot, "total")


def point_source_like(model, total_flux, spectrum=None):
    """A point source at the model's position with ``total_flux`` photons/s/cm^2 and the
    model's total spectrum (or ``spectrum``): observed like the model, it gives the dirty
    beam, which depends on the spectrum (a line's beam keeps its ring sidelobes, a
    continuum's washes them out)."""
    pt = xm.Model(ra=model.ra, dec=model.dec)
    sp = total_spectrum(model) if spectrum is None else spectrum
    pt.add(xm.SeparableLayer("pt", sp))
    return scaled(pt, total_flux)


def dirty_beam(obs, exposure, total_flux=0.1, seed=100, spectrum=None):
    """The dirty beam for ``obs``: a bright point source (so its noise is small) with the
    observed model's spectrum (or ``spectrum``), observed with the same setup and
    reconstructed the same way (field, pixels, energies, deconvolution), normalised to unit
    peak. Not to unit sum: without zero-length baselines the dirty beam integrates to about
    zero, and can be negative."""
    pt = observe(point_source_like(obs["model"], total_flux, spectrum), exposure,
                 obs["fov_arcsec"], seed=seed, setup=obs["setup"], **obs["reconstruct_kw"])
    return pt["image"] / pt["image"].max()


def _bin_onto(img, grid, x, y, w):
    """Add weights ``w`` at sky offsets (x East, y North; uas) to the pixels of ``grid``
    (column index increasing towards the West), dropping those outside."""
    ix = np.floor((grid.x0 - np.asarray(x, float)) / grid.scale + grid.nx / 2).astype(int)
    iy = np.floor((np.asarray(y, float) - grid.y0) / grid.scale + grid.ny / 2).astype(int)
    ok = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    np.add.at(img, (iy[ok], ix[ok]), np.broadcast_to(w, ix.shape)[ok])


def pixel_rates(layer, lo=0.5, hi=12.0, nbin=60, setup=None):
    """A layer's detected count rate (counts/s) per layer pixel, absorbed as the sampler
    absorbs it and weighted by the ARF (summed over baselines) in ``nbin`` bins between
    ``lo`` and ``hi`` keV. Returns (x, y, rate): pixel centres (uas, x East) and rates;
    a point layer is one pixel at its position."""
    arf = (bundled() if setup is None else setup).arf
    edges = np.linspace(lo, hi, nbin + 1)
    area = np.interp(0.5 * (edges[1:] + edges[:-1]), arf.energy_mid, arf.total)
    if hasattr(layer, "maps"):
        rate = np.zeros(layer.grid.n)
        for k in range(nbin):
            rate += xm.specmap_projection(layer, edges=edges[k:k + 2])[1] * area[k]
        return (*layer.grid.centres(), rate)
    sp = layer.spectrum
    cum = np.concatenate([[0.0], np.cumsum(sp.values * sp.widths)])
    r = float(np.sum(np.diff(np.interp(edges, sp.edges, cum)) * area))
    if layer.grid is not None and getattr(layer.grid, "n", 1) > 1:
        w = np.asarray(layer.image, float)
        return (*layer.grid.centres(), r * w / w.sum())
    return np.array([layer.x]), np.array([layer.y]), np.array([r])


def model_image(model, grid, lo=0.5, hi=12.0, nbin=60, setup=None, layers=None):
    """The model's detected count rate per pixel (counts/s, :func:`pixel_rates`) on
    ``grid`` (a ``RegularGrid``, East to the left; for :func:`predicted`). Each layer pixel
    (quadtree leaf, regular pixel or point) adds its rate to the ``grid`` pixel containing
    its centre, so layers on any grid can be imaged on any other; pixels larger than
    ``grid``'s land in one pixel. ``layers`` optionally selects layers by name (a callable
    ``keep(name)``)."""
    img = np.zeros((grid.ny, grid.nx))
    for layer in model.layers:
        if layers is None or layers(layer.name):
            _bin_onto(img, grid, *pixel_rates(layer, lo, hi, nbin, setup))
    return img


def on_grid(image, half_uas, obs):
    """Resample a model ``image`` spanning +-``half_uas`` (East to the left, i.e. column 0
    is East) onto the pixel grid of ``obs`` (nearest pixel; zero outside the image)."""
    n = obs["image"].shape[0]
    c = (np.arange(n) - n / 2) * obs["span"] / n    # pixel centres (uas, East = +): the
    # reconstruction's phase centre is pixel n // 2 (the IFFT's zero), not a pixel corner
    ny, nx = image.shape
    ix = np.round((half_uas - c) / (2 * half_uas) * nx - 0.5).astype(int)
    iy = np.round((c + half_uas) / (2 * half_uas) * ny - 0.5).astype(int)
    out = np.zeros((n, n))
    okx, oky = (ix >= 0) & (ix < nx), (iy >= 0) & (iy < ny)
    out[np.ix_(oky, okx)] = np.nan_to_num(image)[np.ix_(iy[oky], ix[okx])]
    return out


def predicted(image, half_uas, obs, beam):
    """``image`` (a model, see :func:`on_grid`) convolved with ``beam``: what the
    reconstruction should show without photon noise. The beam is centred on its peak (the
    source's pixel), not on the array's middle: on the even-sized grids of
    :func:`pixels` those differ by up to a pixel."""
    from scipy.signal import fftconvolve
    img = on_grid(image, half_uas, obs)
    full = fftconvolve(img, beam, mode="full")
    py, px = np.unravel_index(np.argmax(beam), beam.shape)
    return full[py:py + img.shape[0], px:px + img.shape[1]]


def show(ax, obs, unit=1.0, lim=None, gamma=0.5, label=None, cmap="magma"):
    """Plot a reconstruction (East to the left), normalised to its peak, with a power-law
    stretch. ``unit`` divides the axes (1e3 for mas); ``lim`` crops to +-lim."""
    from matplotlib.colors import PowerNorm

    img, span = obs["image"], obs["span"]
    h, p = span / 2 / unit, span / img.shape[0] / unit
    # pixel k is centred at (k - n/2) * pixel, so the phase centre is a pixel centre
    ext = [-h - p / 2, h - p / 2, -h - p / 2, h - p / 2]
    im = ax.imshow(np.clip(img / img.max(), 0, None), origin="lower", extent=ext,
                   cmap=cmap, norm=PowerNorm(gamma, vmin=0, vmax=1))
    ax.invert_xaxis()
    if lim is not None:
        ax.set_xlim(lim, -lim)
        ax.set_ylim(-lim, lim)
    if label:
        ax.text(0.04, 0.93, label, transform=ax.transAxes, color="w")
    u = "mas" if unit == 1e3 else "µas"
    ax.set_xlabel(f"East offset ({u})")
    return im


__all__ = ["PIXEL_UAS", "Setup", "bundled", "dirty_beam", "model_image", "observe", "on_grid",
           "pixel_rates", "pixels", "point_source_like", "predicted", "scaled", "show", "subset",
           "total_spectrum"]
