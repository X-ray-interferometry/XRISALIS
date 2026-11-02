"""Source models — the input side of the flowchart, F(alpha, delta, E, t).

A source model is a **spatial morphology** x **spectrum** (x optional **light
curve**). Every model reduces to a list of :class:`Component` objects that the
*generate* stage folds through the ARF, so all input kinds compose uniformly:

* point sources / catalogs        -> :class:`PointSource` (point morphology)
* images (+ a spectrum, timing)   -> :class:`ImageSource` (image morphology)
* data cubes F(alpha,delta,E)     -> :class:`CubeSource` (one component per
                                     energy slice, each an image morphology)
* any combination                 -> ``a + b`` -> :class:`CompositeSource`

Spectra are ``(energy_kev, fluxdensity)`` arrays [photon/s/cm^2/keV]; light
curves are callables ``rate(t)`` on ``[0, exposure]`` (None = steady). Core
NumPy (+ Pillow for non-FITS image files, imported lazily). The FITS loaders
(``from_simput``/``from_fits_image``/``from_fits_cube``/``load_source``) use
:mod:`xrisalis.io`, which depends on astropy (a required dependency).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .io import SimputSource, read_fits_image, read_fits_cube
from .units import to_deg, to_arcsec, to_kev, to_radec_deg, is_skycoord

DEG_PER_ARCSEC = 1.0 / 3600.0


def _center(center_deg):
    """Resolve an image centre to a plain ``(ra_deg, dec_deg)`` float tuple.

    Accepts a :class:`~astropy.coordinates.SkyCoord`, or an ``(ra, dec)`` pair of
    angle Quantities or bare degrees.
    """
    if is_skycoord(center_deg):
        return to_radec_deg(center_deg)
    ra, dec = center_deg
    return to_radec_deg(ra, dec)


# ============================================================================
# spectra
# ============================================================================
def powerlaw(gamma=2.0, norm=1.0, e_range=(0.5, 10.0), n=200):
    """Photon power law F(E) = norm * E**(-gamma) [photon/s/cm^2/keV]."""
    E = np.linspace(e_range[0], e_range[1], n)
    return E, norm * E ** (-gamma)


def flat_spectrum(norm=1.0, e_range=(0.5, 10.0), n=200):
    E = np.linspace(e_range[0], e_range[1], n)
    return E, np.full_like(E, float(norm))


def monochromatic(energy_kev, flux=1.0, width_frac=0.01):
    """A narrow line at ``energy_kev`` (float keV or an energy Quantity)."""
    energy_kev = to_kev(energy_kev)
    dE = energy_kev * width_frac
    E = np.array([energy_kev - dE, energy_kev + dE])
    return E, np.array([flux / (2 * dE), flux / (2 * dE)])


def spectrum_from_file(path):
    """Two-column text file: energy[keV]  fluxdensity."""
    d = np.loadtxt(path)
    return np.asarray(d[:, 0], float), np.asarray(d[:, 1], float)


def spectrum_from_model(model, e_range=(0.5, 10.0), n=200):
    """Sample an :mod:`astropy.modeling` 1-D model as ``(energy_kev, F)``.

    ``model`` is any callable of energy (e.g. ``astropy.modeling.models.PowerLaw1D``
    or ``BlackBody`` or a compound model). ``e_range`` bounds may be bare keV or
    energy Quantities. The returned flux density is a plain float array
    (photon/s/cm^2/keV by convention).
    """
    lo, hi = to_kev(e_range[0]), to_kev(e_range[1])
    E = np.linspace(lo, hi, n)
    return E, _strip(model(E))


def _is_lmfit_model(obj):
    """Duck-type an lmfit / nDspec ``Model`` (also matches composite models)."""
    return (hasattr(obj, "eval") and hasattr(obj, "independent_vars")
            and hasattr(obj, "make_params"))


def _eval_lmfit(model, E, params=None, indep_var=None):
    """Evaluate an lmfit/nDspec model on energy grid ``E`` -> flux array."""
    iv = model.independent_vars
    name = indep_var or (iv[0] if iv else "energ")
    F = model.eval(params, **{name: E}) if params is not None \
        else model.eval(**{name: E})
    return _strip(F)


def spectrum_from_lmfit(model, params=None, e_range=(0.5, 10.0), n=200,
                        indep_var=None):
    """Sample an lmfit / **nDspec** model as ``(energy_kev, fluxdensity)``.

    nDspec spectral models are ``lmfit.Model`` objects (single or composite),
    evaluated on an energy axis whose name is the model's first independent
    variable (``'energ'`` in nDspec). ``params`` is an lmfit ``Parameters``
    object (from ``model.make_params(...)``); if omitted the model's default
    parameters are used. ``e_range`` bounds may be bare keV or Quantities. The
    model's output is taken as the photon flux density [photon/s/cm^2/keV].
    """
    lo, hi = to_kev(e_range[0]), to_kev(e_range[1])
    E = np.linspace(lo, hi, n)
    return E, _eval_lmfit(model, E, params=params, indep_var=indep_var)


def _as_spectrum(spectrum, e_grid_kev=None):
    """Normalise any spectrum form to ``(energy_kev, fluxdensity)`` arrays.

    Accepts:

    * an ``(E, F)`` tuple/array — ``E`` may be an energy :class:`~astropy.units.Quantity`;
    * a callable ``f(E) -> F`` (a plain function **or an** :mod:`astropy.modeling`
      1-D model);
    * an **lmfit / nDspec** ``Model`` (evaluated with its default parameters), or
      a ``(model, params)`` tuple to supply an lmfit ``Parameters`` object;

    all evaluated on ``e_grid_kev`` (default 0.5-10 keV, 200 pts). ``F`` is
    returned as a plain float array (photon/s/cm^2/keV; a Quantity is stripped).
    """
    grid = (np.linspace(0.5, 10.0, 200) if e_grid_kev is None
            else np.asarray(e_grid_kev, float))
    if _is_lmfit_model(spectrum):                       # nDspec / lmfit model
        return grid, _eval_lmfit(spectrum, grid)
    if (isinstance(spectrum, tuple) and len(spectrum) == 2
            and _is_lmfit_model(spectrum[0])):          # (model, params)
        return grid, _eval_lmfit(spectrum[0], grid, params=spectrum[1])
    if callable(spectrum):
        return grid, _strip(spectrum(grid))
    E, F = spectrum
    return to_kev(E), _strip(F)


def _strip(x):
    """Plain float ndarray from a value that may be an astropy Quantity."""
    return np.asarray(getattr(x, "value", x), float)


# ============================================================================
# morphologies (sample sky positions in degrees)
# ============================================================================
class Morphology:
    def sample(self, n, rng):
        raise NotImplementedError

    @property
    def centroid_deg(self):
        """(ra, dec) flux centroid in degrees (for evaluation/reporting)."""
        raise NotImplementedError


class PointMorphology(Morphology):
    def __init__(self, ra_deg, dec_deg):
        self.ra_deg = float(ra_deg)
        self.dec_deg = float(dec_deg)

    def sample(self, n, rng):
        return np.full(n, self.ra_deg), np.full(n, self.dec_deg)

    @property
    def centroid_deg(self):
        return self.ra_deg, self.dec_deg


class ImageMorphology(Morphology):
    """Sky brightness from a 2-D image sampled as a probability map.

    ``scale_arcsec`` is the angular size of the image's largest axis; ``center``
    is the (RA, Dec) of the image centre in degrees. Columns map to +RA (alpha),
    rows to +Dec (delta).
    """
    def __init__(self, image, scale_arcsec, center_deg=(0.0, 0.0),
                 rotation_deg=0.0):
        img = np.asarray(image, float)
        img = np.clip(img, 0, None)
        if rotation_deg:
            from scipy.ndimage import rotate as _rotate      # lazy
            img = np.clip(_rotate(img, float(rotation_deg), reshape=False,
                                  order=1, mode="constant", cval=0.0), 0, None)
        if img.sum() <= 0:
            raise ValueError("Image has no positive flux.")
        self.rotation_deg = float(rotation_deg)
        self.image = img
        self.ny, self.nx = img.shape
        self.pmf = (img / img.sum()).ravel()
        self.pix_arcsec = scale_arcsec / max(self.ny, self.nx)
        self.center_deg = (float(center_deg[0]), float(center_deg[1]))

    def sample(self, n, rng):
        idx = rng.choice(self.pmf.size, size=n, p=self.pmf)
        rows, cols = np.unravel_index(idx, (self.ny, self.nx))
        # sub-pixel jitter for a smooth distribution
        jr = rng.uniform(-0.5, 0.5, n)
        jc = rng.uniform(-0.5, 0.5, n)
        alpha_as = (cols + jc - (self.nx - 1) / 2.0) * self.pix_arcsec
        delta_as = (rows + jr - (self.ny - 1) / 2.0) * self.pix_arcsec
        ra = self.center_deg[0] + alpha_as * DEG_PER_ARCSEC
        dec = self.center_deg[1] + delta_as * DEG_PER_ARCSEC
        return ra, dec

    @property
    def centroid_deg(self):
        ys, xs = np.mgrid[0:self.ny, 0:self.nx]
        w = self.image / self.image.sum()
        col = (w * xs).sum(); row = (w * ys).sum()
        a = (col - (self.nx - 1) / 2.0) * self.pix_arcsec
        d = (row - (self.ny - 1) / 2.0) * self.pix_arcsec
        return (self.center_deg[0] + a * DEG_PER_ARCSEC,
                self.center_deg[1] + d * DEG_PER_ARCSEC)


# ============================================================================
# component + source models
# ============================================================================
@dataclass
class Component:
    """One spectral-spatial piece of a source (what the generator consumes)."""
    energy_kev: np.ndarray
    fluxdensity: np.ndarray            # photon/s/cm^2/keV
    morphology: Morphology
    lightcurve: object = None          # callable rate(t) or None (steady)
    src_id: int = 0


class SourceModel:
    """Base: a source that yields spectral-spatial components."""
    def components(self):
        raise NotImplementedError

    def _as_list(self):
        return [self]

    def __add__(self, other):
        return CompositeSource(self._as_list() + other._as_list())

    # convenience for evaluation / reporting
    def positions_deg(self):
        """(m,2) flux centroids [ra_deg, dec_deg] of the components."""
        return np.array([c.morphology.centroid_deg for c in self.components()])

    def fluxes(self):
        """Integrated photon flux per component (relative)."""
        out = []
        for c in self.components():
            E, F = c.energy_kev, c.fluxdensity
            out.append(0.5 * np.sum((F[1:] + F[:-1]) * np.diff(E)) if len(E) > 1
                       else float(F[0]))
        return np.array(out)


class PointSource(SourceModel):
    def __init__(self, ra_deg, dec_deg, spectrum, lightcurve=None, src_id=0):
        # positions accept a SkyCoord (passed as ra_deg) or angle Quantities
        self.ra_deg, self.dec_deg = to_radec_deg(ra_deg, dec_deg)
        self.E, self.F = _as_spectrum(spectrum)
        self.lightcurve = lightcurve
        self.src_id = src_id

    def components(self):
        return [Component(self.E, self.F,
                          PointMorphology(self.ra_deg, self.dec_deg),
                          self.lightcurve, self.src_id)]


class ImageSource(SourceModel):
    """Extended source: a brightness image with a (separable) spectrum."""
    def __init__(self, image, spectrum, scale_arcsec, center_deg=(0.0, 0.0),
                 lightcurve=None, src_id=0, rotation_deg=0.0):
        self.morph = ImageMorphology(image, to_arcsec(scale_arcsec),
                                     _center(center_deg),
                                     rotation_deg=to_deg(rotation_deg))
        self.E, self.F = _as_spectrum(spectrum)
        self.lightcurve = lightcurve
        self.src_id = src_id

    def components(self):
        return [Component(self.E, self.F, self.morph, self.lightcurve,
                          self.src_id)]


class CubeSource(SourceModel):
    """Data cube F(alpha, delta, E): one component per energy slice.

    ``energies_kev`` (m,) and ``images`` (m, ny, nx); each slice's total flux
    weights a monochromatic line at its energy, so the ARF folding naturally
    weights the energy-dependent morphology.
    """
    def __init__(self, energies_kev, images, scale_arcsec, center_deg=(0.0, 0.0),
                 norm=1.0, width_frac=0.01, lightcurve=None, src_id=0):
        self.energies = np.atleast_1d(to_kev(energies_kev))
        self.images = np.clip(np.asarray(images, float), 0, None)
        self.scale_arcsec = to_arcsec(scale_arcsec)
        self.center_deg = _center(center_deg)
        self.norm = float(norm)          # total photon flux [photon/s/cm^2]
        self.width_frac = width_frac
        self.lightcurve = lightcurve
        self.src_id = src_id

    def components(self):
        # per-slice flux = norm * (slice pixel sum) / (whole-cube pixel sum), so
        # the cube integrates to `norm` and pixel counts are not treated as flux
        total = self.images.sum()
        if total <= 0:
            return []
        comps = []
        for E_i, img in zip(self.energies, self.images):
            w = img.sum()
            if w <= 0:
                continue
            E, F = monochromatic(E_i, flux=self.norm * w / total,
                                 width_frac=self.width_frac)
            comps.append(Component(
                E, F, ImageMorphology(img, self.scale_arcsec, self.center_deg),
                self.lightcurve, self.src_id))
        return comps


class CompositeSource(SourceModel):
    """A combination of source models (catalog / mixed inputs)."""
    def __init__(self, sources):
        self.sources = list(sources)

    def _as_list(self):
        return self.sources

    def components(self):
        out = []
        for k, s in enumerate(self.sources):
            for c in s.components():
                out.append(Component(c.energy_kev, c.fluxdensity, c.morphology,
                                     c.lightcurve, src_id=k))
        return out


# ============================================================================
# convenience constructors
# ============================================================================
def point_source(ra_deg, dec_deg=None, spectrum=None, lightcurve=None):
    """A single point source.

    Call as ``point_source(ra, dec, spectrum)`` (angles as bare degrees or
    Quantities) or ``point_source(skycoord, spectrum)`` with a scalar
    :class:`~astropy.coordinates.SkyCoord`.
    """
    if is_skycoord(ra_deg) and spectrum is None:
        spectrum, dec_deg = dec_deg, None          # point_source(coord, spectrum)
    return PointSource(ra_deg, dec_deg, spectrum, lightcurve)


def points(ra_deg, dec_deg=None, spectra=None, lightcurves=None):
    """Catalog of point sources from positions + per-source spectra.

    Positions may be given as arrays of bare degrees / angle Quantities plus
    ``dec_deg``, or as a single array :class:`~astropy.coordinates.SkyCoord`
    (in which case ``dec_deg`` carries the ``spectra`` list).
    """
    if is_skycoord(ra_deg) and spectra is None:
        spectra, dec_deg = dec_deg, None           # points(skycoord_array, spectra)
    if is_skycoord(ra_deg):
        ra_arr, dec_arr = to_radec_deg(ra_deg)
    else:
        ra_arr, dec_arr = to_radec_deg(ra_deg, dec_deg)
    ra_arr, dec_arr = np.atleast_1d(ra_arr), np.atleast_1d(dec_arr)
    n = len(ra_arr)
    lcs = lightcurves or [None] * n
    return CompositeSource([PointSource(ra_arr[i], dec_arr[i], spectra[i], lcs[i])
                            for i in range(n)])


def _table_column_deg(table, col):
    """A table column as degrees, honouring an astropy unit if the column has one."""
    c = table[col]
    unit = getattr(c, "unit", None)
    if unit is not None:
        import astropy.units as u
        return (np.asarray(c) * unit).to_value(u.deg)
    return np.asarray(c, float)


def points_from_table(table, ra_col="ra", dec_col="dec", spectrum=None,
                      flux_col=None, spectra=None, lightcurves=None):
    """Point-source catalog from a table (e.g. an ``astroquery``/VizieR result).

    Works with any :class:`astropy.table.Table` (or mapping of columns) — the
    query is run by the caller, so there is no runtime dependency on astroquery.

    Parameters
    ----------
    table : astropy Table | mapping
        Must contain the position columns; angular units on the columns are
        honoured (defaults to degrees).
    ra_col, dec_col : str            column names for the sky position.
    spectrum : optional              a single template spectrum (tuple / callable /
        astropy or lmfit model) applied to every source; defaults to a power law.
    flux_col : str, optional         per-row flux used to scale the template
        spectrum's normalisation (relative brightness across the catalog).
    spectra : list, optional         explicit per-source spectra (overrides
        ``spectrum``/``flux_col``).
    """
    ra = _table_column_deg(table, ra_col)
    dec = _table_column_deg(table, dec_col)
    n = len(np.atleast_1d(ra))
    if spectra is None:
        if spectrum is None:
            spectrum = powerlaw()
        if flux_col is not None:
            fl = np.asarray(table[flux_col], float)
            E, F = _as_spectrum(spectrum)
            spectra = [(E, F * float(fl[i])) for i in range(n)]
        else:
            spectra = [spectrum] * n
    return points(ra, dec, spectra, lightcurves)


def _disc_image(radius_frac=0.8, npix=201):
    ax = np.linspace(-1, 1, npix)
    X, Y = np.meshgrid(ax, ax)
    return (X ** 2 + Y ** 2 <= radius_frac ** 2).astype(float)


def _gaussian_image(sigma_frac=0.25, npix=201):
    ax = np.linspace(-1, 1, npix)
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / (2 * sigma_frac ** 2))


def disc(spectrum, diameter_arcsec, center_deg=(0.0, 0.0), npix=201,
         lightcurve=None):
    """A uniform disc of the given angular diameter.

    The disc fills its ``npix`` x ``npix`` image edge to edge, so the image span
    equals the disc diameter. (Before 2026-10-16 the disc was drawn at 0.8 of
    the image span, i.e. its true diameter was ``0.8 * diameter_arcsec``.)
    """
    return ImageSource(_disc_image(radius_frac=1.0, npix=npix), spectrum,
                       diameter_arcsec, center_deg, lightcurve)


def gaussian_source(spectrum, fwhm_arcsec, center_deg=(0.0, 0.0), npix=201,
                    lightcurve=None):
    """A circular Gaussian blob of the given FWHM."""
    sigma_frac = 0.2                                   # in image [-1,1] units
    img = _gaussian_image(sigma_frac=sigma_frac, npix=npix)
    # image full-width (2 units) maps to `span` arcsec; FWHM = 2.3548*sigma_frac
    # in those units, i.e. (2.3548*sigma_frac/2)*span arcsec. Solve for span:
    span = fwhm_arcsec * 2.0 / (2.3548 * sigma_frac)
    return ImageSource(img, spectrum, span, center_deg, lightcurve)


def image_source(image, spectrum, scale_arcsec, center_deg=(0.0, 0.0),
                 lightcurve=None, rotation_deg=0.0):
    """Extended source from an in-memory brightness array."""
    return ImageSource(image, spectrum, scale_arcsec, center_deg, lightcurve,
                       rotation_deg=rotation_deg)


def model_image_source(model2d, spectrum, extent_arcsec, npix=201,
                       center_deg=(0.0, 0.0), lightcurve=None, rotation_deg=0.0):
    """Extended source whose morphology is an :mod:`astropy.modeling` 2-D model.

    ``model2d`` (e.g. ``Gaussian2D``, ``Sersic2D``, ``Disk2D``, or a compound
    model) is evaluated on an ``npix x npix`` grid spanning ``extent_arcsec``
    (largest axis), with ``(x, y)`` in **arcsec offsets from the centre** — so
    define the model's centre/scale parameters in arcsec. Negative values are
    clipped to zero. ``extent_arcsec`` may be a Quantity; ``center_deg`` may be
    a SkyCoord.
    """
    extent = to_arcsec(extent_arcsec)
    ax = np.linspace(-extent / 2.0, extent / 2.0, npix)
    X, Y = np.meshgrid(ax, ax)
    img = np.clip(np.nan_to_num(np.asarray(model2d(X, Y), float)), 0, None)
    return ImageSource(img, spectrum, extent, center_deg=center_deg,
                       lightcurve=lightcurve, rotation_deg=rotation_deg)


def from_image_file(path, spectrum, scale_arcsec, center_deg=(0.0, 0.0),
                    lightcurve=None):
    """Extended source from an image file (PNG/JPG/…) read as grayscale."""
    from PIL import Image                      # lazy
    img = np.asarray(Image.open(path).convert("L"), float)
    return ImageSource(img, spectrum, scale_arcsec, center_deg, lightcurve)


def cube_source(energies_kev, images, scale_arcsec, center_deg=(0.0, 0.0),
                norm=1.0, lightcurve=None):
    return CubeSource(energies_kev, images, scale_arcsec, center_deg,
                      norm=norm, lightcurve=lightcurve)


KEV2ERG = 1.602176634e-9        # 1 keV in erg (for SIMPUT flux normalisation)


def _lightcurve_from_timing(timing):
    """Turn a TIMING descriptor (from io.read_timing) into a callable rate(t).

    The callable is a *relative* modulation (mean ~1 over its span), so it
    reshapes the arrival-time distribution without changing the total flux set
    by the spectrum. Returns None for a steady source.
    """
    if not timing:
        return None
    if timing["kind"] == "binned":
        t, f = np.asarray(timing["time"], float), np.asarray(timing["flux"], float)
        f = np.clip(f, 0, None)
        mean = f.mean() if f.size and f.mean() > 0 else 1.0
        return lambda tt: np.interp(tt, t, f / mean, left=f[0] / mean,
                                    right=f[-1] / mean)
    if timing["kind"] == "periodic":
        ph, f = np.asarray(timing["phase"], float), np.asarray(timing["flux"], float)
        f = np.clip(f, 0, None)
        mean = f.mean() if f.size and f.mean() > 0 else 1.0
        period, phase0 = timing["period"], timing.get("phase0", 0.0)
        # wrap phase table for periodic interpolation
        php = np.concatenate(([ph[-1] - 1.0], ph, [ph[0] + 1.0]))
        fp = np.concatenate(([f[-1]], f, [f[0]])) / mean

        def rate(tt):
            phase = ((np.asarray(tt, float) / period) + phase0) % 1.0
            return np.interp(phase, php, fp)
        return rate
    return None


def _normalize_to_flux(E, F, flux, e_min, e_max):
    """Scale photon flux-density F so its band energy flux equals SIMPUT FLUX.

    SIMPUT FLUX is the energy flux [erg/s/cm^2] over ``[e_min, e_max]`` keV;
    F is [photon/s/cm^2/keV]. Scale F by FLUX / (current band energy flux). No-op
    (with a warning suppressed) when FLUX or the band is missing/degenerate.
    """
    if flux is None or e_min is None or e_max is None or e_max <= e_min:
        return F
    E = np.asarray(E, float); F = np.asarray(F, float)
    m = (E >= e_min) & (E <= e_max)
    if m.sum() < 2:
        return F
    Eb, Fb = E[m], F[m]
    energy_flux_kev = 0.5 * np.sum((Eb[1:] * Fb[1:] + Eb[:-1] * Fb[:-1])
                                   * np.diff(Eb))          # keV/s/cm^2
    cur = energy_flux_kev * KEV2ERG                        # erg/s/cm^2
    if cur <= 0:
        return F
    return F * (float(flux) / cur)


def from_simput(path):
    """Build a source model from a SIMPUT catalog.

    Point entries become :class:`PointSource`; entries with a resolved ``IMAGE``
    reference become :class:`ImageSource` (extended morphology, ``IMGSCAL`` /
    ``IMGROTA`` applied); ``TIMING`` references attach a light curve; each
    source's spectrum is normalised to its ``FLUX``. All entries compose into a
    :class:`CompositeSource`.
    """
    s = SimputSource.read(path)
    models = []
    for e in s.entries:
        F = _normalize_to_flux(e.energy_kev, e.fluxdensity, e.flux,
                               e.e_min, e.e_max)
        lc = _lightcurve_from_timing(e.timing)
        if e.image is not None:
            models.append(ImageSource(
                e.image, (e.energy_kev, F), e.image_scale_arcsec,
                center_deg=e.image_center_deg or (e.ra_deg, e.dec_deg),
                lightcurve=lc, rotation_deg=e.image_rotation_deg))
        else:
            models.append(PointSource(e.ra_deg, e.dec_deg, (e.energy_kev, F),
                                      lightcurve=lc))
    return CompositeSource(models)


def from_fits_image(path, spectrum=None, scale_arcsec=None, center_deg=None,
                    lightcurve=None):
    """Extended source from a standalone FITS image (2-D).

    Angular scale/center come from the image WCS (or ``CDELT``/``IMGSCAL``, or
    the ``scale_arcsec`` kwarg). A FITS image carries no spectrum, so one must be
    supplied; if omitted, a flat spectrum is used (with a warning).
    """
    img, scale, wcs_center, rot = read_fits_image(path, scale_arcsec=scale_arcsec)
    if spectrum is None:
        warnings.warn("from_fits_image: no spectrum given; using a flat "
                      "0.5-10 keV spectrum.")
        spectrum = flat_spectrum()
    center = center_deg if center_deg is not None else wcs_center
    return ImageSource(img, spectrum, scale, center_deg=center,
                       lightcurve=lightcurve, rotation_deg=rot)


def from_fits_cube(path, norm=1.0, center_deg=None, scale_arcsec=None,
                   lightcurve=None):
    """Data-cube source F(alpha, delta, E) from a 3-D FITS cube.

    Per-slice energies come from a spectral WCS axis or an ``ENERGIES`` table;
    each slice contributes an energy-weighted image morphology (see
    :class:`CubeSource`). ``norm`` sets the total integrated photon flux.
    """
    energies, cube, scale, wcs_center = read_fits_cube(path)
    if scale_arcsec is not None:
        scale = scale_arcsec
    center = center_deg if center_deg is not None else wcs_center
    return CubeSource(energies, cube, scale, center_deg=center, norm=norm,
                      lightcurve=lightcurve)


def load_source(path, kind="auto", **overrides):
    """Load any reasonable FITS source into a :class:`SourceModel`.

    Detects the input type and routes to the right builder:

    * a ``SRC_CAT`` extension       -> SIMPUT catalog  (:func:`from_simput`)
    * a 3-D image HDU               -> spectral cube   (:func:`from_fits_cube`)
    * a 2-D image HDU               -> extended image  (:func:`from_fits_image`)

    Parameters
    ----------
    path : str
        FITS file to load.
    kind : {'auto', 'simput', 'image', 'cube'}
        Force a reader instead of auto-detecting.
    **overrides
        Forwarded to the image/cube builders (``spectrum``, ``scale_arcsec``,
        ``center_deg``, ``lightcurve``, ``norm``). Ignored for SIMPUT.

    Returns
    -------
    SourceModel
        Flows unchanged through :func:`xrisalis.simulate.generate_photons`.
    """
    from astropy.io import fits

    if kind == "auto":
        with fits.open(path) as hdul:
            names = [h.name for h in hdul]
            if "SRC_CAT" in names:
                kind = "simput"
            else:
                ndim = 0
                for h in hdul:
                    if getattr(h, "is_image", False) and h.data is not None:
                        ndim = np.asarray(h.data).ndim
                        if ndim >= 2:
                            break
                if ndim >= 3:
                    kind = "cube"
                elif ndim == 2:
                    kind = "image"
                else:
                    raise ValueError(
                        f"Could not identify a source in {path!r}; found HDUs "
                        f"{names}. Pass kind='simput'|'image'|'cube' to force.")

    if kind == "simput":
        if overrides:
            warnings.warn(f"load_source: ignoring {list(overrides)} for a "
                          "SIMPUT catalog (its parameters come from the file).")
        return from_simput(path)
    if kind == "cube":
        return from_fits_cube(path, **overrides)
    if kind == "image":
        return from_fits_image(path, **overrides)
    raise ValueError(f"Unknown kind {kind!r}; use auto/simput/image/cube.")
