"""Simulation stages.

Section 1 (fringe physics): Fresnel diffraction + fringe sampling.
Section 2 (generate): source + ARF + exposure -> Photon List (Poisson photons).
Section 3 (redistribute): Photon List + RMF + optics -> Event List."""

from __future__ import annotations
import warnings
import numpy as np
import scipy.special as sps
import scipy.stats as sampler
from .config import KEV, H, C
from .io import PhotonList, EventList
from .units import to_seconds, to_radec_deg, is_skycoord
from .pointing import attitude_index


# ==========================================================================
# from fringes.py
# ==========================================================================

def fresnel_diffraction(wavelength, baseline, y_pos=None, samples=1):
    """Fresnel diffraction intensity for a slit of a given baseline.

    Parameters
    ----------
    wavelength : array_like
        Photon wavelength(s) [m].
    baseline : Baseline
        Baseline providing slit width ``W`` and combining length ``L``.
    y_pos : array_like, optional
        Detector positions [m] to evaluate at. If None, sample the slit.
    samples : int
        Number of samples if ``y_pos`` is None.

    Returns
    -------
    intensity, u : ndarray, ndarray
    """
    u_0 = baseline.W * np.sqrt(2 / (wavelength * baseline.L))

    def u_1(u):
        return u - u_0 / 2

    def u_2(u):
        return u + u_0 / 2

    if y_pos is None:
        y_pos = np.linspace(-baseline.W / 2, baseline.W / 2, int(samples))
    u = y_pos * np.sqrt(2 / (wavelength * baseline.L))

    s1, c1 = sps.fresnel(u_1(u))
    s2, c2 = sps.fresnel(u_2(u))
    amp = (c2 - c1) + 1j * (s2 - s1)
    intensity = np.abs(amp * np.conjugate(amp))
    return intensity, u


def sample_detector_positions(baseline, energies_j, thetas, rng=None,
                              pure_diffraction=False, pure_fringes=False,
                              max_iter=5000, on_unplaceable="drop",
                              return_slat=False):
    """Accept-reject sample detector y-positions [m] for photons in one baseline.

    Parameters
    ----------
    baseline : Baseline
    energies_j : ndarray
        Photon energies [J].
    thetas : ndarray
        Off-axis angle [rad] per photon (same length as energies).
    rng : np.random.Generator, optional
    pure_diffraction, pure_fringes : bool
        Diagnostic switches, as in the legacy code.
    max_iter : int
        Cap on accept-reject iterations. Photons still unplaced after this many
        rounds have a fringe packet that cannot land on the detector (source
        beyond the field of view); rather than loop forever they are handled per
        ``on_unplaceable``. Normal (in-FoV) photons place in far fewer rounds.
    on_unplaceable : {'drop', 'raise'}
        'drop' (default) marks unplaceable photons with ``NaN`` (the caller
        vignettes them — see :func:`redistribute`); 'raise' raises RuntimeError.

    return_slat : bool
        If True, also return the per-photon slat index (which comb patch each
        photon landed on; -1 for unplaced photons).

    Returns
    -------
    y_pos : ndarray
        Detector positions [m], one per input photon; ``NaN`` for any photon
        that could not be placed (only when ``on_unplaceable='drop'``).
    slat_id : ndarray of int, optional
        Only if ``return_slat=True`` — the slat/comb-patch index per photon
        (0..num_pairs-1; -1 if unplaced).

    Notes
    -----
    Slatted-mirror comb: the ``num_pairs`` slats (width ``W``, aperture pitch
    ``slat_pitch``) tile the detector into a comb of single-slit envelopes at
    pitch ``slat_pitch``, each filled by the *global* two-beam fringe (period
    ``lambda/beam_angle``, shifted by the source term ``D*sin(theta)``). Keeping
    the fringe global preserves the interferometric phase that reconstruction
    reads. ``slat_pitch = W`` gives a contiguous comb (the legacy behavior);
    ``slat_pitch > W`` opens dark gaps between patches.
    """
    if getattr(baseline, "combiner", "slatted") == "arbitrary":
        return sample_arbitrary_positions(baseline, energies_j, thetas, rng=rng,
                                          return_slat=return_slat)
    rng = np.random.default_rng() if rng is None else rng
    thetas = np.asarray(thetas, dtype=float)
    energies_j = np.ravel(np.asarray(energies_j, dtype=float))
    n = len(thetas)
    y_pos = np.zeros(n)
    slat_ids = np.full(n, -1, dtype=np.int32)
    indices = np.arange(n)
    wavelengths = H * C / energies_j
    N = baseline.num_pairs
    W = baseline.W
    p_det = baseline.slat_pitch                     # detector comb pitch (mag 1)

    # accept-reject until every photon has a position. Photons whose fringe
    # packet falls outside the detector (source beyond the field of view) can
    # never be accepted and would loop forever, so the loop is bounded by
    # ``max_iter`` and leftovers are dropped (NaN) or raised — see the guard
    # after the loop. In-FoV photons place in far fewer than ``max_iter`` rounds.
    it = 0
    warned_off_fov = False          # the off-FoV notice is raised once per call
    while len(indices) > 0 and it < max_iter:
        it += 1
        m = len(indices)

        # illuminated strip = the full comb span (num_pairs * slat_pitch),
        # shifted by the geometric beam displacement with off-axis angle.
        half_span = N * p_det / 2.0
        width_box = np.array([(-half_span) - baseline.L * thetas,
                              (half_span) - baseline.L * thetas])

        # box height: just above the Fresnel maximum (empirical 2.7x factor)
        height_top = 2.7 * fresnel_diffraction(wavelengths[0], baseline,
                                               y_pos=np.array([0.0]))[0][0]

        fringe_spacing = wavelengths / baseline.beam_angle
        num_fringes = np.ceil(
            (width_box[1] - width_box[0]) * baseline.num_pairs / fringe_spacing)

        off_set_fringes = -(baseline.D * np.sin(thetas)) / (
            2 * np.sin(baseline.beam_angle / 2))

        left_fringe = off_set_fringes - (
            np.ceil((off_set_fringes - baseline.L * thetas + half_span)
                    / fringe_spacing) * fringe_spacing)

        rand_cos = (0.5 * fringe_spacing
                    * sampler.cosine.rvs(loc=2 * np.pi * left_fringe / fringe_spacing,
                                         size=m, random_state=rng) / np.pi)

        if pure_diffraction:
            rand_pos = rng.uniform(width_box[0], width_box[1], size=m)
        else:
            rand_pos = rand_cos + (rng.integers(low=0, high=num_fringes + 1, size=m)
                                   * fringe_spacing)

        if not warned_off_fov and np.logical_or(
                (off_set_fringes < width_box[0]).any(),
                (off_set_fringes > width_box[1]).any()):
            # Once per call: this is an accept-reject loop, so an unconditional
            # warning here fires on every iteration and buries the output of a
            # long run (thousands of identical lines) instead of informing it.
            warned_off_fov = True
            warnings.warn("There are fringe centers outside the FoV!")

        valid = np.logical_and(rand_pos >= width_box[0], rand_pos <= width_box[1])

        # SLAT COMB: assign each position to its nearest slat patch (centers at
        # (k-(N-1)/2)*p_det, k=0..N-1) and reject positions in the inter-slat
        # gaps (|local| > W/2). local is the within-slat coordinate.
        y_rel = rand_pos + baseline.L * thetas
        kf = y_rel / p_det + (N - 1) / 2.0
        k = np.clip(np.round(kf).astype(int), 0, N - 1)
        local = y_rel - (k - (N - 1) / 2.0) * p_det
        in_slat = np.abs(local) <= (W / 2.0)

        diffraction_value, _ = fresnel_diffraction(wavelengths, baseline, y_pos=local)
        diffraction_value = np.where(in_slat, diffraction_value, 0.0)  # gaps -> reject
        rand_value = rng.uniform(0, height_top, size=m)
        if pure_fringes:
            accept = in_slat
        else:
            accept = rand_value <= diffraction_value

        keep = accept & valid
        y_pos[indices[keep]] = rand_pos[keep]
        slat_ids[indices[keep]] = k[keep]

        thetas = thetas[~keep]
        wavelengths = wavelengths[~keep]
        indices = indices[~keep]

    if len(indices) > 0:                    # unplaceable: beyond the field of view
        if on_unplaceable == "raise":
            raise RuntimeError(
                f"sample_detector_positions: {len(indices)} photon(s) could not "
                f"be placed after {max_iter} iterations — fringe packet outside "
                f"the field of view (|theta| too large for this baseline).")
        y_pos[indices] = np.nan            # caller vignettes NaNs
        slat_ids[indices] = -1

    return (y_pos, slat_ids) if return_slat else y_pos


def arbitrary_fringe_intensity(y, baseline, wavelength, theta):
    """Interferogram of the ``'arbitrary'`` combiner (unnormalized).

    ``I(y) = 1 + V cos(2 pi (y - y0) / dy)`` inside the overlap window
    ``|y + L theta| <= overlap_span/2`` and 0 outside, with fringe period
    ``dy = lambda / beam_angle`` and fringe-maximum offset
    ``y0 = -D sin(theta) / (2 sin(beam_angle/2))`` (~ ``-D theta/beam_angle``,
    the same phase convention as the slatted sampler, so reconstruction is
    unchanged). No diffraction envelope.
    """
    y = np.asarray(y, float)
    lam = np.asarray(wavelength, float)
    theta = np.asarray(theta, float)
    period = lam / baseline.beam_angle
    y0 = -(baseline.D * np.sin(theta)) / (2 * np.sin(baseline.beam_angle / 2))
    center = -baseline.L * theta
    inside = np.abs(y - center) <= baseline.overlap_span / 2.0
    val = 1.0 + baseline.visibility * np.cos(2 * np.pi * (y - y0) / period)
    return np.where(inside, val, 0.0)


def sample_arbitrary_positions(baseline, energies_j, thetas, rng=None,
                               max_iter=10000, return_slat=False):
    """Detector DETY positions [m] for an ``'arbitrary'`` beam combiner.

    Idealized two-beam combiner: fringes of period ``lambda/beam_angle`` and
    contrast ``baseline.visibility``, NO diffraction. The interferogram is cut
    off at the edges of the overlap window (width ``baseline.overlap_span``,
    centered at ``-L theta`` -- the geometric beam walk), and **every** photon is
    placed inside it: there is no vignetting and no photon is wasted. See
    :func:`arbitrary_fringe_intensity` for the pattern.

    Sampling is accept-reject from a uniform proposal over the window with
    acceptance ``(1 + V cos)/(1 + V)`` (>= 1/2 on average for a window spanning
    many fringes), so it always terminates for realistic geometries.

    Returns ``y_pos`` (and ``slat_id``, all 0 -- a single overlap patch -- if
    ``return_slat``).
    """
    rng = np.random.default_rng() if rng is None else rng
    thetas = np.ravel(np.asarray(thetas, dtype=float))
    energies_j = np.ravel(np.asarray(energies_j, dtype=float))
    n = thetas.size
    if energies_j.size != n:
        raise ValueError("energies_j and thetas must have the same length")
    lam_all = H * C / energies_j
    half = baseline.overlap_span / 2.0
    V = baseline.visibility
    y_pos = np.empty(n)
    todo = np.arange(n)
    it = 0
    while todo.size and it < max_iter:
        it += 1
        th, lam = thetas[todo], lam_all[todo]
        y = -baseline.L * th + rng.uniform(-half, half, todo.size)
        p = arbitrary_fringe_intensity(y, baseline, lam, th) / (1.0 + V)
        keep = rng.uniform(0.0, 1.0, todo.size) <= p
        y_pos[todo[keep]] = y[keep]
        todo = todo[~keep]
    if todo.size:
        raise RuntimeError(
            f"sample_arbitrary_positions: {todo.size} photon(s) not placed after "
            f"{max_iter} rounds -- overlap window ({baseline.overlap_span:g} m) "
            f"is much narrower than a fringe and sits on a dark fringe.")
    slat_ids = np.zeros(n, dtype=np.int32)
    return (y_pos, slat_ids) if return_slat else y_pos


def off_axis_angles_2d(pointing, toa, loc):
    """Off-axis angle components ``(theta_par, theta_perp)`` [rad] per photon.

    Decomposes the sky offset (source relative to the boresight) into the
    baseline/roll frame: ``theta_par`` along the fringe (DETY, interferometric)
    axis and ``theta_perp`` along the cross (DETZ) axis. ``loc`` is the (N,2)
    tangent-plane offset ``[eta, xi]`` [rad] of each photon from the nominal
    pointing (:attr:`~xrisalis.io.PhotonList.loc_rad`), and ``pointing`` the
    attitude stream whose ``(pitch, yaw)`` are offsets on the same plane, so
    the difference below is a true small angle (cos dec and the RA wrap are
    handled by the projection); ``toa`` is the integer time-step index.
    """
    pos_rel = pointing[toa, :2] - loc
    r = np.sqrt(pos_rel[:, 0] ** 2 + pos_rel[:, 1] ** 2)
    ang = pointing[toa, 2] - np.arctan2(pos_rel[:, 0], pos_rel[:, 1])
    return np.cos(ang) * r, np.sin(ang) * r


def off_axis_angles(pointing, toa, loc):
    """Off-axis angle theta [rad] along the fringe axis (see off_axis_angles_2d)."""
    return off_axis_angles_2d(pointing, toa, loc)[0]


# ---------------------------------------------------------- z (DETZ) sampling --
def _fresnel_slit_intensity(y, aperture, wavelength, dist):
    """1-D Fresnel diffraction intensity of a slit of full width ``aperture``.

    ``y`` is the detector position [m] relative to the slit center, ``dist`` the
    propagation distance [m]. Returns the (unnormalized) intensity.
    """
    s = np.sqrt(2.0 / (wavelength * dist))
    u1 = (y - aperture / 2.0) * s
    u2 = (y + aperture / 2.0) * s
    s1, c1 = sps.fresnel(u1)
    s2, c2 = sps.fresnel(u2)
    return (c2 - c1) ** 2 + (s2 - s1) ** 2


def _fresnel_slit_sample(aperture, wavelengths, dist, rng):
    """Accept-reject sample of the single-slit Fresnel pattern (rare z regime)."""
    wavelengths = np.ravel(np.asarray(wavelengths, float))
    n = wavelengths.size
    span = aperture * 1.6
    ygrid = np.linspace(-span / 2, span / 2, 512)[:, None]
    hmax = _fresnel_slit_intensity(ygrid, aperture, wavelengths[None, :], dist).max() * 1.05
    out = np.zeros(n)
    todo = np.arange(n)
    while todo.size:
        m = todo.size
        y = rng.uniform(-span / 2, span / 2, m)
        val = _fresnel_slit_intensity(y, aperture, wavelengths[todo], dist)
        acc = rng.uniform(0, hmax, m) <= val
        out[todo[acc]] = y[acc]
        todo = todo[~acc]
    return out


def sample_detz_positions(baseline, energies_j, theta_perp, rng=None,
                          distance=None, geometric_nf=10.0):
    """Detector cross-axis positions ``DETZ`` [m] for one baseline.

    The z axis carries NO interference (the two beams are separated along the
    baseline/DETY direction), so DETZ is the single-aperture illumination of the
    mirror's z-extent ``Wz`` at the detector plane, shifted by the source's
    perpendicular off-axis angle (the z direction is unfocused — flat mirrors —
    so the shift is ``-theta_perp * Z``).

    For cm-scale mirrors at the ~10 m detector plane the Fresnel number
    ``N_F = (Wz/2)^2/(lambda Z)`` is >> 1, so this is a **geometric shadow**: a
    top-hat of width ``Wz`` (wavelength-independent). Only if ``N_F`` drops below
    ``geometric_nf`` (much longer propagation / smaller mirror) does it fall back
    to sampling the wavelength-dependent Fresnel pattern.
    """
    rng = np.random.default_rng() if rng is None else rng
    e = np.ravel(np.asarray(energies_j, float))
    tperp = np.broadcast_to(np.asarray(theta_perp, float), e.shape)
    Z = baseline.L if distance is None else float(distance)
    Wz = baseline.Wz
    lam = H * C / e
    NF = (Wz / 2.0) ** 2 / (lam * Z)
    center = -tperp * Z                       # geometric lateral beam shift [m]
    detz = np.empty(e.size)
    geo = NF >= geometric_nf
    if geo.any():                              # geometric shadow: uniform top-hat
        ng = int(geo.sum())
        detz[geo] = center[geo] + rng.uniform(-Wz / 2.0, Wz / 2.0, ng)
    if (~geo).any():                           # Fresnel regime (unusual here)
        idx = np.where(~geo)[0]
        detz[idx] = center[idx] + _fresnel_slit_sample(Wz, lam[idx], Z, rng)
    return detz


def simulate_detz(baseline_id, energy_j, theta_perp, baselines, rng=None):
    """Detector cross-axis positions ``DETZ`` [m], per photon, per baseline."""
    rng = np.random.default_rng() if rng is None else rng
    detz = np.zeros(baseline_id.size)
    for k in np.unique(baseline_id):
        sel = baseline_id == k
        detz[sel] = sample_detz_positions(
            baselines[int(k)], energy_j[sel], theta_perp[sel], rng=rng)
    return detz

# ==========================================================================
# from generate.py
# ==========================================================================

def _draw_arrival_times(n, exposure, rng, lightcurve=None):
    """Arrival times for a homogeneous (or lightcurve-modulated) Poisson process.

    For a constant rate the arrival times of a fixed number of Poisson events
    are uniform on [0, exposure]; that is the steady-state default. If
    ``lightcurve`` (a callable rate(t) >= 0 on [0, exposure]) is given, times are
    inverse-CDF sampled from its normalized cumulative, enabling source
    variability without changing the count model.
    """
    if lightcurve is None:
        return rng.uniform(0.0, exposure, n)
    t = np.linspace(0.0, exposure, 4096)
    rate = np.clip(np.asarray(lightcurve(t), float), 0, None)
    cdf = np.cumsum(rate)
    if cdf[-1] <= 0:
        return rng.uniform(0.0, exposure, n)
    cdf = cdf / cdf[-1]
    return np.interp(rng.uniform(0, 1, n), cdf, t)


def sample_photons(energ_lo, energ_hi, per_baseline_resp, spectra, coords,
                   exposure, rng=None, lightcurve=None):
    """Pure-NumPy photon sampler.

    Parameters
    ----------
    energ_lo, energ_hi : (nE,) ndarray
        ARF energy bin edges [keV].
    per_baseline_resp : list[(D_m, resp_cm2)]
        Effective area [cm^2] on the ARF grid, per baseline.
    spectra : list[(energy_kev, fluxdensity)]
        One source spectrum per source (flux density [photon/s/cm^2/keV]).
    coords : (ra_deg, dec_deg)
        Arrays of source sky positions [deg].
    exposure : float
        Exposure time [s].
    rng : np.random.Generator
    lightcurve : callable, optional
        rate(t) for source variability; None = steady state.

    Returns
    -------
    dict with keys time, energy_kev, ra_deg, dec_deg, src_id, baseline_id.
    """
    rng = np.random.default_rng() if rng is None else rng
    ra_all, dec_all = np.asarray(coords[0], float), np.asarray(coords[1], float)
    emid_arf = 0.5 * (energ_lo + energ_hi)

    times, energies, ras, decs, sids, bids = [], [], [], [], [], []
    for i, (_, resp) in enumerate(per_baseline_resp):
        for j, (src_e, src_f) in enumerate(spectra):
            src_e = np.asarray(src_e, float)
            # ARF effective area interpolated onto the source energy grid
            resp_interp = np.interp(src_e, emid_arf, resp, left=0.0, right=0.0)
            # integrate over each bin of the SOURCE's own energy grid (not the
            # ARF grid, which need not match the spectrum's sampling)
            binw = np.gradient(src_e) if src_e.size > 1 else np.array([1.0])
            counts_per_bin_per_s = src_f * resp_interp * binw
            expected = counts_per_bin_per_s * exposure
            counts = rng.poisson(np.clip(expected, 0, None))
            if counts.sum() == 0:
                continue
            e = np.repeat(src_e, counts)
            n = e.size
            times.append(_draw_arrival_times(n, exposure, rng, lightcurve))
            energies.append(e)
            ras.append(np.full(n, ra_all[j]))
            decs.append(np.full(n, dec_all[j]))
            sids.append(np.full(n, j, dtype=np.int32))
            bids.append(np.full(n, i, dtype=np.int16))

    if not energies:
        empty = np.array([])
        return {"time": empty, "energy_kev": empty, "ra_deg": empty,
                "dec_deg": empty, "src_id": empty.astype(np.int32),
                "baseline_id": empty.astype(np.int16)}

    return {
        "time": np.concatenate(times),
        "energy_kev": np.concatenate(energies),
        "ra_deg": np.concatenate(ras),
        "dec_deg": np.concatenate(decs),
        "src_id": np.concatenate(sids),
        "baseline_id": np.concatenate(bids),
    }


def sample_photons_components(arf, components, exposure, rng=None):
    """Generalized photon sampler over source **components** (any morphology).

    Each ``component`` is duck-typed with ``energy_kev``, ``fluxdensity``,
    ``morphology.sample(n, rng) -> (ra_deg, dec_deg)``, ``lightcurve`` and
    ``src_id`` (see :mod:`xrisalis.sources`). This is what lets point, image and
    cube inputs share one generate path. Returns the same dict as
    :func:`sample_photons`.
    """
    rng = np.random.default_rng() if rng is None else rng
    emid = 0.5 * (arf.energ_lo + arf.energ_hi)
    times, energies, ras, decs, sids, bids = [], [], [], [], [], []
    for i, (_, resp) in enumerate(arf.per_baseline):
        for comp in components:
            E = np.asarray(comp.energy_kev, float)
            F = np.asarray(comp.fluxdensity, float)
            binw = np.gradient(E) if E.size > 1 else np.array([1.0])
            resp_i = np.interp(E, emid, resp, left=0.0, right=0.0)
            expected = F * resp_i * binw * exposure
            counts = rng.poisson(np.clip(expected, 0, None))
            if counts.sum() == 0:
                continue
            e = np.repeat(E, counts)
            n = e.size
            ra, dec = comp.morphology.sample(n, rng)
            times.append(_draw_arrival_times(n, exposure, rng, comp.lightcurve))
            energies.append(e)
            ras.append(np.asarray(ra, float))
            decs.append(np.asarray(dec, float))
            sids.append(np.full(n, comp.src_id, dtype=np.int32))
            bids.append(np.full(n, i, dtype=np.int16))
    if not energies:
        z = np.array([])
        return {"time": z, "energy_kev": z, "ra_deg": z, "dec_deg": z,
                "src_id": z.astype(np.int32), "baseline_id": z.astype(np.int16)}
    return {"time": np.concatenate(times),
            "energy_kev": np.concatenate(energies),
            "ra_deg": np.concatenate(ras), "dec_deg": np.concatenate(decs),
            "src_id": np.concatenate(sids),
            "baseline_id": np.concatenate(bids)}


def generate_photons(source, arf, exposure, rng=None, lightcurve=None,
                     pointing=(0.0, 0.0), seed=None):
    """High-level generate stage: -> :class:`~xrisalis.io.PhotonList`.

    ``source`` may be any :class:`xrisalis.sources.SourceModel` (point, image,
    cube, or a combination via ``a + b``) — routed through
    :func:`sample_photons_components` — or a SimputSource / object exposing
    ``ra_deg``, ``dec_deg``, ``spectra`` (the point-catalog fast path).

    ``pointing`` is the **nominal pointing** ``(ra, dec)`` (bare degrees,
    Quantities or a SkyCoord): the tangent point all off-axis/fringe maths is
    projected about, stored as ``PhotonList.ra_pnt/dec_pnt``. The attitude
    stream (:mod:`xrisalis.pointing`) gives small offsets from it. There is no
    automatic re-pointing: a source far from it falls outside the field of
    view and is vignetted.
    """
    exposure = to_seconds(exposure)                  # accept a time Quantity
    if rng is None:
        rng = np.random.default_rng(seed)
    if hasattr(source, "components"):                 # a SourceModel
        d = sample_photons_components(arf, source.components(), exposure, rng=rng)
    else:                                             # SimputSource / points
        d = sample_photons(arf.energ_lo, arf.energ_hi, arf.per_baseline,
                           source.spectra, (source.ra_deg, source.dec_deg),
                           exposure, rng=rng, lightcurve=lightcurve)
    if is_skycoord(pointing):
        ra_pnt, dec_pnt = to_radec_deg(pointing)
    else:
        ra_pnt, dec_pnt = to_radec_deg(pointing[0], pointing[1])
    return PhotonList(
        time=d["time"], energy_kev=d["energy_kev"],
        ra_deg=d["ra_deg"], dec_deg=d["dec_deg"],
        src_id=d["src_id"], baseline_id=d["baseline_id"],
        exposure=exposure, ra_pnt=float(ra_pnt), dec_pnt=float(dec_pnt),
        meta={"seed": seed},
    )

# ==========================================================================
# from redistribute.py
# ==========================================================================

def redistribute_rmf(true_energy_kev, rmf, rng=None, chunk=50_000):
    """Vectorized RMF redistribution.

    Parameters
    ----------
    true_energy_kev : (N,) ndarray
    rmf : RMF
        Provides ``energ_lo/energ_hi`` (energy bins), ``matrix`` (nE x nChan,
        full-width rows starting at channel 0) and channel energy bounds.
    rng : np.random.Generator

    Returns
    -------
    pi : (M,) int channel per surviving photon
    detected_kev : (M,) float channel-mapped energy [keV]
    valid : (N,) bool mask of photons that fell inside the RMF energy range
    """
    rng = np.random.default_rng() if rng is None else rng
    e = np.asarray(true_energy_kev, float)

    # RMF energy bin per photon
    bin_idx = np.searchsorted(rmf.energ_hi, e)
    valid = (bin_idx < rmf.energ_lo.size) & (e >= rmf.energ_lo[0])
    bin_idx = bin_idx[valid]

    # per-energy-row channel CDF (full-width rows: channel index == column index)
    row_sums = rmf.matrix.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    cdf = np.cumsum(np.clip(rmf.matrix, 0, None), axis=1) / safe   # (nE, nChan)

    pi = np.empty(bin_idx.size, dtype=np.int32)
    for s in range(0, bin_idx.size, chunk):
        b = bin_idx[s:s + chunk]
        u = rng.uniform(0, 1, b.size)[:, None]
        # inverse-CDF: channel = number of cdf entries below u
        pi[s:s + chunk] = (u > cdf[b]).sum(axis=1)

    pi = np.clip(pi, 0, rmf.channel.size - 1)
    detected_kev = rmf.channel_energy[pi]
    return pi, detected_kev, valid


def simulate_positions(baseline_id, energy_j, theta, baselines, rng=None):
    """Detector fringe-axis positions ``DETY`` [m] + slat index, per photon.

    Returns ``(dety, slat_id)``; ``slat_id`` is the comb-patch index each photon
    landed on (see :func:`sample_detector_positions`), -1 for unplaced photons.
    """
    rng = np.random.default_rng() if rng is None else rng
    dety = np.zeros(baseline_id.size)
    slat_id = np.full(baseline_id.size, -1, dtype=np.int32)
    for k in np.unique(baseline_id):
        sel = baseline_id == k
        y, s = sample_detector_positions(
            baselines[int(k)], energy_j[sel], theta[sel], rng=rng,
            return_slat=True)
        dety[sel] = y
        slat_id[sel] = s
    return dety, slat_id


def redistribute(photons, instrument, rmf, rng=None, seed=None,
                 readout=None, pointing=None):
    """High-level redistribute stage: -> :class:`~xrisalis.io.EventList`.

    ``photons`` is a PhotonList; ``instrument`` an Interferometer (baselines +
    pointing model); ``rmf`` an RMF. ``readout`` optionally maps arrival time to
    read-out time (frame/dead time); default is identity (t_d = t_s).

    ``pointing`` optionally overrides the instrument's boresight attitude for
    this call (any spec accepted by :func:`xrisalis.pointing.as_pointing`: a
    :class:`~xrisalis.pointing.Pointing`, a callable ``f(t)``, a file path, or a
    constant ``(pitch, yaw[, roll])``). If None, ``instrument.pointing`` is used.
    The boresight ``(pitch, yaw)`` are tangent-plane offsets (North, East) from
    the photons' nominal pointing ``(ra_pnt, dec_pnt)``; the target itself is set
    in :func:`generate_photons`.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # attitude stream + per-photon off-axis angle (fringe axis + cross axis)
    t_s = np.asarray(photons.time, dtype=float)          # arrival times [s]
    attitude = instrument.gen_pointing(float(t_s.max()) if t_s.size else 1.0,
                                       pointing=pointing)
    # attitude rows are spaced by instrument.time_step SECONDS, so the row for a
    # photon is int(t / time_step) -- int(t) is only right at time_step = 1.
    toa = attitude_index(t_s, instrument.time_step, attitude.shape[0])
    theta, theta_perp = off_axis_angles_2d(attitude, toa, photons.loc_rad)

    # 1) RMF energy redistribution (vectorized)
    pi, detected_kev, valid = redistribute_rmf(photons.energy_kev, rmf, rng=rng)

    # keep only photons that survived the RMF energy range
    bidx = photons.baseline_id[valid]
    true_kev = photons.energy_kev[valid]
    theta_v = theta[valid]
    theta_perp_v = theta_perp[valid]
    time_v = photons.time[valid]
    ra_v, dec_v = photons.ra_deg[valid], photons.dec_deg[valid]

    # 2) fringe positioning -> DETY [m]; DETZ cross-axis reserved (0 for v1).
    # PHYSICS: the fringe spacing is set by the photon's TRUE wavelength (the
    # actual light that interferes), so positions are sampled with the true
    # energy. Reconstruction, however, only knows the DETECTED (channel) energy.
    # Because the fringe frequency scales with 1/wavelength, this true<->detected
    # gap produces a genuine chromatic decoherence set by the detector energy
    # resolution (the RMF). It is a real effect, not hidden: recover the image
    # with the `fringe_taper` (down-weight decohered high-order fringes) or the
    # sub-band synthesis options in xrisalis.reconstruct. The TRUE_ENERGY column
    # is stored so the ideal (true-energy) reconstruction can be compared against.
    # VIGNETTE: beyond a baseline's geometric (detection) FoV the fringe packet
    # misses the detector, so those photons are never placed (the sampler alone
    # does not enforce this edge and stalls far outside it); the 'arbitrary'
    # combiner receives every photon along the fringe axis by design. Across
    # it (DETZ) the field is set by the beam angle, |theta_perp| <= beta/2.
    # Placements the sampler cannot make come back as NaN too; all are dropped.
    fov = np.array([(np.inf if b.combiner == "arbitrary"
                     else b.field_of_view(which="detection"),
                     b.field_of_view(which="cross")) for b in instrument.baselines])
    inside = ((np.abs(theta_v) <= fov[bidx, 0])
              & (np.abs(theta_perp_v) <= fov[bidx, 1]))
    dety, slat_id = np.full(bidx.size, np.nan), np.full(bidx.size, -1, np.int32)
    dety[inside], slat_id[inside] = simulate_positions(
        bidx[inside], true_kev[inside] * KEV, theta_v[inside],
        instrument.baselines, rng=rng)
    on = np.isfinite(dety)
    n_out = int((~on).sum())
    if n_out:
        frac = 100.0 * n_out / on.size
        warnings.warn(
            f"{n_out} of {on.size} photons ({frac:.1f}%) fall outside the field "
            f"of view and were vignetted. Check the source position and "
            f"pointing/boresight relative to the instrument FoV "
            f"(Interferometer.field_of_view).")
    if not on.any():
        raise ValueError(
            "all photons fall outside the field of view — the source (with the "
            "current pointing/boresight) is not within the instrument's FoV.")
    dety = dety[on]
    slat_id = slat_id[on]
    bidx, true_kev = bidx[on], true_kev[on]
    time_v = time_v[on]
    ra_v, dec_v = ra_v[on], dec_v[on]
    pi = pi[on]
    detected_kev = detected_kev[on]
    theta_perp_v = theta_perp_v[on]

    # 2b) cross-axis position -> DETZ [m]: single-aperture illumination of the
    # mirror z-extent (geometric shadow for cm mirrors at the detector plane;
    # Fresnel only over ~km propagation). Sampled with the TRUE wavelength, as
    # for DETY.
    detz = simulate_detz(bidx, true_kev * KEV, theta_perp_v,
                         instrument.baselines, rng=rng)

    t_d = time_v if readout is None else readout(time_v)

    # 3) focal-plane placement: map each baseline's (DETY, DETZ) onto the
    # (possibly multi-chip) detector so the event list also carries chip id,
    # chip pixel/meter coordinates and total focal-plane coordinates. Uses the
    # instrument's detector layout if it defines chips, else a default tiled
    # layout (one chip per baseline).
    from .detector import Detector
    det = instrument.detector
    if det is None or not getattr(det, "chips", None):
        px = det.pixel_m if det is not None else 1e-5
        det = Detector.tiled_layout(instrument, pixel_m=px)
    place = det.place(bidx, dety, detz)

    # cache the derived imaging FoV (low-energy edge, min over baselines) so
    # reconstruct can default to it without being handed the RMF again.
    fov_img = None
    try:
        if detected_kev.size:
            e_lo = float(np.percentile(detected_kev, 1))
            e_hi = float(np.percentile(detected_kev, 99))
            fov_img = instrument.field_of_view(
                energy_kev=(e_lo, e_hi), rmf=rmf, which="imaging",
                reduce="low_edge", combine="min")
    except Exception:                                    # never block redistribute
        fov_img = None

    return EventList(
        time=t_d, energy_kev=detected_kev, true_energy_kev=true_kev,
        dety=dety, detz=detz,
        baseline_id=bidx, pi=pi, ra_deg=ra_v, dec_deg=dec_v,
        pointing=attitude, time_step=float(instrument.time_step),
        ra_pnt=photons.ra_pnt, dec_pnt=photons.dec_pnt,
        ebounds=rmf.ebounds_dict,
        exposure=photons.exposure, fov=fov_img, slat_id=slat_id,
        chip_id=place["chip_id"], chipx=place["chipx"], chipy=place["chipy"],
        chipx_m=place["chipx_m"], chipy_m=place["chipy_m"],
        focx=place["focx"], focy=place["focy"],
        meta={"seed": seed},
    )
