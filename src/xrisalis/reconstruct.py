"""Image reconstruction.

Section 1 (transforms): gridded IFFT and direct IFT (pure physics).
Section 2 (uv): filter events + compute uv-coordinates/visibilities + taper.
Section 3 (drivers): reconstruct() and reconstruct_subbands().
Section 4 (color): dirty beams, beam matching and reconstruct_rgb()."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .config import ARCSEC2RAD, KEV, RAD2ARCSEC, RAD2UAS, C, H
from .coords import tangent_offsets, tangent_to_sky
from .units import to_arcsec, to_kev, to_radec_deg


def _resolve_fov_arcsec(
    fov_arcsec,
    events,
    instrument,
    energy_kev=None,
    baseline_bounds=None,
    baseline_ids=None,
):
    """Return a FoV in arcsec, deriving it from the instrument/events if None.

    Precedence when ``fov_arcsec is None``: the imaging FoV cached on the
    EventList at redistribute time (includes the coherence term), else
    :meth:`Interferometer.field_of_view` (geometry only, no RMF here), else a
    legacy 1.6 mas fallback (with a warning). An explicit ``fov_arcsec`` (number
    or Quantity) is always honored.
    """
    if fov_arcsec is not None:
        return to_arcsec(fov_arcsec)
    fov_rad = getattr(events, "fov", None)
    if fov_rad is None or not np.isfinite(fov_rad):
        try:
            fov_rad = instrument.field_of_view(
                energy_kev=energy_kev,
                baseline_bounds=baseline_bounds,
                baseline_ids=baseline_ids,
                which="imaging",
            )
        except Exception:
            fov_rad = None
    if fov_rad is None or not np.isfinite(fov_rad):
        warnings.warn(
            "could not derive a field of view from the instrument or "
            "events; falling back to 1.6 mas. Pass fov_arcsec "
            "explicitly to silence this."
        )
        return 1.6e-3
    return float(fov_rad) * RAD2ARCSEC


def _grid_safe_fov_arcsec(
    fov_arcsec, max_energy_kev, max_baseline_D, n_samples, safety=0.9
):
    """Cap the FoV so an ``n_samples``-pixel grid still samples the finest fringe.

    The reconstruction pixel scale is ``fov / n_samples``; it must stay below the
    finest fringe scale ``λ_min / D_max`` or the gridded IFFT overflows (and the
    image is aliased). Returns ``min(fov, safety·n_samples·λ_min/D_max)`` in
    arcsec, warning when it caps. Increase ``samples``/``npix`` to image wider.
    """
    lam_min = H * C / (float(max_energy_kev) * KEV)
    cap_arcsec = (
        safety * int(n_samples) * lam_min / float(max_baseline_D)
    ) * RAD2ARCSEC
    if fov_arcsec > cap_arcsec:
        warnings.warn(
            f"field of view capped to {cap_arcsec * 1e3:.3g} mas so the "
            f"{int(n_samples)}-pixel grid samples the finest fringe "
            f"(≤{max_energy_kev:.2g} keV, D≤{max_baseline_D:.3g} m); pass a "
            f"larger samples/npix to image a wider field."
        )
        return cap_arcsec
    return fov_arcsec


# ==========================================================================
# from transform.py
# ==========================================================================

# optional acceleration, silent fallback to numpy
try:  # pragma: no cover
    import pyfftw.interfaces.numpy_fft as _fft

    _HAVE_PYFFTW = True
except Exception:  # pragma: no cover
    import numpy.fft as _fft

    _HAVE_PYFFTW = False


def _grid_params(fov, samples, min_baseline_D, max_wavelength, error):
    """Sample spacing d and window size n for the gridded IFFT (legacy formula)."""
    d = (fov) / np.max(samples)
    n = np.ceil(1 / (2 * d * (min_baseline_D / max_wavelength) * error)).astype(int)
    return d, int(n)


def ifft_image(
    uv,
    vis,
    fov,
    energies_j,
    min_baseline_D,
    samples=(512, 512),
    error=0.1,
    coherence=None,
    deconv_epsilon=0.1,
    verbose=False,
):
    """Gridded inverse-FFT reconstruction.

    Parameters
    ----------
    uv : (N,2) ndarray      uv-coordinates [wavelengths]
    vis : (N,) complex      visibilities
    fov : float             field of view [rad] (largest axis)
    energies_j : ndarray    event energies [J] (for wavelength range)
    min_baseline_D : float  shortest baseline length [m]
    samples : (int,int)     output pixel grid
    error : float           grid oversampling control (legacy)
    coherence : (N,) float, optional
        Recovery **option 2** (uv-domain matched-filter Wiener). Per-photon
        chromatic coherence factors ``D(N)`` from :func:`compute_uv`
        (``coherence_R=R``). When given, each uv cell is estimated by the
        maximum-likelihood matched filter for phasors scattered as
        ``z_k ~ CN(V * D_k, 1)``:

            V_est(cell) = sum_k D_k z_k / ( sum_k D_k^2 + reg ),
            reg = deconv_epsilon * max_cell( sum_k D_k^2 )

        i.e. each photon is weighted by its own coherence (down-weighting the
        chromatically scrambled high-fringe-order photons rather than amplifying
        them), then the cell is normalized by its accumulated squared coherence.
        This is the correct place to undo chromatic decoherence — per-cell after
        binning, never per-photon. As ``deconv_epsilon`` grows it converges to an
        optimal D-shaped taper; as it shrinks it deconvolves more aggressively at
        the cost of reintroducing the (physically irrecoverable) high-N phase
        noise. The scrambled long-baseline phase is genuinely lost, so this does
        not out-resolve the fringe-order taper — it recovers a clean image with a
        principled, RMF-derived weighting.
    deconv_epsilon : float  Wiener regularization (fraction of peak sum D^2).

    Returns
    -------
    image : (ny,nx) ndarray  real reconstructed image
    pixel_scale_uas : float  pixel scale [micro-arcsec]
    meta : dict
    """
    samples = np.asarray(samples)
    wavelengths = H * C / energies_j
    max_wl = np.max(wavelengths)

    d, n = _grid_params(fov, samples, min_baseline_D, max_wl, error)
    if verbose:
        print(f"IFFT grid: sample spacing d={d:.3e} rad, window n={n}")

    u_conv = np.around(uv[:, 0] * (n * d)).astype(int)
    v_conv = np.around(uv[:, 1] * (n * d)).astype(int)

    if coherence is None:
        grid = np.zeros((n, n), dtype=np.complex128)
        np.add.at(grid, (u_conv, v_conv), vis)
    else:
        # gridded matched-filter Wiener: numerator sum D_k z_k, denom sum D_k^2
        cwt = np.asarray(coherence, float)
        grid_num = np.zeros((n, n), dtype=np.complex128)
        grid_den = np.zeros((n, n), dtype=np.float64)
        np.add.at(grid_num, (u_conv, v_conv), cwt * vis)
        np.add.at(grid_den, (u_conv, v_conv), cwt**2)
        reg = deconv_epsilon * grid_den.max()
        if reg <= 0:
            reg = 1.0
        grid = grid_num / (grid_den + reg)
        if verbose:
            filled = int(np.count_nonzero(grid_den))
            print(
                f"IFFT matched-filter deconvolution: {filled} uv cells, reg {reg:.3g}"
            )

    img = np.real(_fft.ifft2(grid))
    img = np.roll(img, int(round(img.shape[0] / 2)), axis=0)
    img = np.roll(img, int(round(img.shape[1] / 2)), axis=1)

    # crop to requested output size
    def _bounds(npix, shape):
        lo = int(round(shape / 2 - 0.5 * npix))
        hi = int(round(shape / 2 + 0.5 * npix))
        lo = max(lo, 0)
        hi = min(hi, shape - 1)
        return lo, hi

    xl, xh = _bounds(samples[0], img.shape[0])
    yl, yh = _bounds(samples[1], img.shape[1])
    cropped = img[np.ix_(np.arange(xl, xh), np.arange(yl, yh))]

    # angular span of the *cropped* image: n_pix_out * pixel_scale, pixel scale = d
    crop_span_uas = cropped.shape[0] * d * RAD2UAS
    full_grid_span_uas = n * d * RAD2UAS
    meta = {
        "d": d,
        "n": n,
        "method": "IFFT",
        "pyfftw": _HAVE_PYFFTW,
        "full_grid_span_uas": full_grid_span_uas,
        "pixel_scale_uas": d * RAD2UAS,
    }
    return cropped, crop_span_uas, meta


def ift_image(uv, vis, fov, npix=256, mem_budget_mb=256):
    """Direct (ungridded) inverse Fourier transform.

    I(x,y) = sum_k V_k exp(+2i pi (u_k x + v_k y)), evaluated on an npix x npix
    grid spanning ``fov`` [rad]. Exact for the sampled visibilities; O(Npix*Nvis)
    but chunked over visibilities so the intermediate never exceeds
    ``mem_budget_mb``. Returns image, crop_span_uas, meta (same contract as
    ifft_image).
    """
    half = fov / 2.0
    ax = np.linspace(-half, half, npix)
    # row coordinate pairs with u, column coordinate pairs with v, matching the
    # gridded IFFT convention np.add.at(grid,(u_idx,v_idx)) -> ifft2 (axis0=u).
    row_coord = ax[:, None] * np.ones((1, npix))  # varies down rows
    col_coord = np.ones((npix, 1)) * ax[None, :]  # varies across columns
    flat_row = row_coord.ravel()[None, :]  # pairs with u
    flat_col = col_coord.ravel()[None, :]  # pairs with v
    npix2 = npix * npix

    # size chunk so (chunk x npix^2) complex128 stays under the budget
    chunk = max(1, int(mem_budget_mb * 1e6 / (npix2 * 16)))
    acc = np.zeros(npix2, dtype=np.complex128)
    N = uv.shape[0]
    for s in range(0, N, chunk):
        u = uv[s : s + chunk, 0][:, None]
        v = uv[s : s + chunk, 1][:, None]
        V = vis[s : s + chunk][:, None]
        phase = np.exp(2j * np.pi * (u * flat_row + v * flat_col))
        acc += (V * phase).sum(axis=0)

    img = np.real(acc).reshape(npix, npix)
    crop_span_uas = fov * RAD2UAS
    meta = {
        "method": "IFT",
        "npix": npix,
        "chunk": chunk,
        "pixel_scale_uas": fov * RAD2UAS / npix,
    }
    return img, crop_span_uas, meta


# ==========================================================================
# from uvgrid.py
# ==========================================================================


def filter_events(
    energies_j,
    positions,
    baseline_indices,
    roll,
    energy_kev=None,
    baseline_bounds=None,
    baseline_ids=None,
):
    """Boolean mask + filtered views for an energy band and a baseline subset.

    ``baseline_bounds=(lo, hi)`` keeps a contiguous *inclusive range* of
    baseline indices; ``baseline_ids=[...]`` keeps an arbitrary explicit set of
    them. Both may be given, in which case both must hold.
    """
    mask = np.ones(len(energies_j), dtype=bool)
    if energy_kev is not None:
        e_kev = energies_j / KEV
        mask &= (e_kev > energy_kev[0]) & (e_kev < energy_kev[1])
    if baseline_bounds is not None:
        mask &= (baseline_indices >= baseline_bounds[0]) & (
            baseline_indices <= baseline_bounds[1]
        )
    if baseline_ids is not None:
        mask &= np.isin(baseline_indices, np.asarray(baseline_ids, int))
    return (mask, energies_j[mask], positions[mask], baseline_indices[mask], roll[mask])


def make_fringe_taper(spec):
    """Return a callable w(N) -> weight for the fringe-order taper (option 4).

    ``spec`` may be:
      * None                         -> no taper (weights = 1)
      * a number ``n_cut``           -> Gaussian taper exp(-(N/n_cut)**2)
      * a callable w(N)              -> used directly

    ``N`` is the fringe order of a photon (|detector position| / fringe spacing),
    i.e. how many fringes it sits from the white-light (zero path difference)
    position. Chromatic decoherence grows with N, so down-weighting large-N
    photons suppresses the scrambled fringes and recovers the coherent core.
    """
    if spec is None:
        return None
    if callable(spec):
        return spec
    n_cut = float(spec)
    return lambda N: np.exp(-((N / n_cut) ** 2))


def coherence_envelope(N, R):
    """Chromatic coherence factor D(N) = exp(-2 pi^2 N^2 / R^2).

    N is the fringe order; R = E / sigma_E is the detector resolving power. This
    is how much the finite energy resolution attenuates a fringe's visibility.
    """
    return np.exp(-2 * np.pi**2 * np.asarray(N, float) ** 2 / R**2)


def resolving_power(rmf, energy_kev=6.0):
    """Detector resolving power R = E / sigma_E near ``energy_kev`` from an RMF.

    ``sigma_E`` is estimated from the FWHM of the **photopeak** (the main
    redistribution lobe), not the whole row — an OGIP RMF usually has a broad
    noise floor / escape features whose spread would otherwise dominate and give
    a misleadingly low R. sigma_E = FWHM / 2.3548.
    """
    emid = 0.5 * (rmf.energ_lo + rmf.energ_hi)
    i = int(np.argmin(np.abs(emid - energy_kev)))
    row = np.clip(np.asarray(rmf.matrix[i], float), 0, None)
    if row.sum() <= 0:
        return np.inf
    E_ch = np.asarray(rmf.channel_energy, float)
    pk = int(np.argmax(row))
    half = 0.5 * row[pk]

    def _crossing(a, b):
        # linear-interpolate the energy where row == half between channels a,b
        if row[b] == row[a]:
            return E_ch[a]
        f = (half - row[a]) / (row[b] - row[a])
        return E_ch[a] + f * (E_ch[b] - E_ch[a])

    lo = pk
    while lo > 0 and row[lo - 1] >= half:
        lo -= 1
    e_left = _crossing(lo - 1, lo) if lo > 0 else E_ch[lo]
    hi = pk
    while hi < row.size - 1 and row[hi + 1] >= half:
        hi += 1
    e_right = _crossing(hi + 1, hi) if hi < row.size - 1 else E_ch[hi]

    fwhm = abs(e_right - e_left)
    if fwhm <= 0:
        fwhm = abs(E_ch[1] - E_ch[0])
    sigma = fwhm / 2.3548
    return float(emid[i] / sigma)


def compute_uv(
    energies_j,
    positions,
    baseline_indices,
    roll,
    baselines,
    fringe_taper=None,
    coherence_R=None,
    boresight=None,
    verbose=False,
):
    """Compute uv-coordinates [wavelengths] and complex visibilities.

    Parameters
    ----------
    energies_j : (N,) ndarray     event energies [J]
    positions : (N,) ndarray      detector fringe-axis position [m]
    baseline_indices : (N,) int   baseline index per event
    roll : (N,) ndarray           roll angle [rad] per event
    baselines : list[Baseline]
    boresight : (N, 2) ndarray, optional
        Per-event boresight ``(pitch, yaw)`` [rad] **relative to the phase
        center** ``c`` — i.e. ``P - c``, where ``P`` is the *known or estimated*
        attitude at the photon's arrival time (see
        :meth:`xrisalis.io.EventList.boresight_per_event`). Callers normally
        get this from :func:`reconstruct`, which subtracts ``c`` itself.

        The simulator places a source at off-axis angle ``theta_par`` at
        ``dety = -D theta_par / beam_angle`` (see
        :func:`xrisalis.simulate.sample_detector_positions`), and
        ``theta_par (D/lam) = (u,v) . (P - L)`` with ``L`` the source's sky
        position. The raw visibility ``exp(-2 pi i f dety)`` is therefore
        ``exp(-2 pi i (u,v) . (L - P))``: the image shows the source at
        ``L - P``, *relative to the boresight*. The boresight enters as a pure
        **translation**, identically on every baseline, so a *static* offset
        only shifts the image while a *time-varying* one smears it (the
        attitude analogue of tip-tilt jitter).

        Passing ``boresight = P - c`` multiplies each visibility by
        ``exp(-2 pi i (u,v) . (P - c))``, giving ``exp(-2 pi i (u,v) . (L - c))``
        for every photon: the source lands at ``L - c`` whatever the attitude
        was when that photon arrived. ``c`` must be close to the field (a few
        FoV at most) — referencing to the sky origin when the target sits at a
        real RA/Dec would ask the grid to represent a shift of ~1e8 fields and
        destroys the image.

        This is also the hook a fringe self-calibration writes its solution
        into: any per-event phase screen expressible as an apparent boresight
        offset can be applied here. Note only the *fringe*-axis component
        matters — the cross (DETZ) axis carries no interferometric phase.
    coherence_R : float, optional
        Resolving power R = E/sigma_E. When given, a third array of per-photon
        chromatic coherence factors ``D(N) = exp(-2 pi^2 N^2 / R^2)`` is
        returned alongside the *raw* phasors (no ``fringe_taper`` folded in),
        for the gridded matched-filter Wiener of :func:`ifft_image` (recovery
        option 2). In this mode ``fringe_taper`` is ignored — the matched filter
        supplies its own (optimal, D-shaped) fringe weighting.

    Returns
    -------
    uv : (2N,2) ndarray, vis : (2N,) complex
        Includes the negative-frequency Hermitian mirror.
    coh : (2N,) float
        Only when ``coherence_R`` is given: the matching per-photon D(N).
    """
    taper = None if coherence_R is not None else make_fringe_taper(fringe_taper)
    # One array per baseline, concatenated once at the end. Accumulating row by
    # row into a Python list (or repeatedly np.append-ing) costs O(N) objects and
    # O(n_baselines * N) copies, which at ten million events means gigabytes of
    # object overhead and an out-of-memory kill rather than a slow run.
    uv_blocks = []
    vis_blocks = []
    coh_blocks = []

    for k in np.unique(baseline_indices):
        sel = baseline_indices == k
        bl = baselines[int(k)]

        lam = H * C / energies_j[sel]
        freq_baseline = bl.D / lam
        fourier_freq = bl.beam_angle / lam  # = 1 / (lam / beam_angle)

        r = roll[sel] % (2 * np.pi)
        u = freq_baseline * np.sin(r)
        v = freq_baseline * np.cos(r)
        pairs = np.array([u, v]).T

        # positive and negative (conjugate) frequencies interleaved
        uv_blocks.append(np.column_stack((pairs, -pairs)).reshape(-1, pairs.shape[1]))

        fval = np.exp(-2j * np.pi * fourier_freq * positions[sel])
        N = np.abs(positions[sel] * fourier_freq)  # fringe order per photon

        # attitude (boresight) phase referencing: undo the translation the
        # spacecraft's own pointing imprinted on this photon's fringe phase.
        # fval = exp(-2 pi i uv.(L - P)); multiplying by exp(-2 pi i uv.(P - c))
        # leaves exp(-2 pi i uv.(L - c)). The sign follows the simulator's
        # placement dety = -D theta/beam_angle -- flipping it doubles the
        # offset instead of removing it.
        if boresight is not None:
            bs = np.asarray(boresight)[sel]
            fval = fval * np.exp(-2j * np.pi * (u * bs[:, 0] + v * bs[:, 1]))

        # option 4: fringe-order taper (only when not deconvolving)
        if taper is not None:
            fval = fval * taper(N)

        vis_blocks.append(np.ravel([fval, np.conjugate(fval)], "F"))

        if coherence_R is not None:
            dval = coherence_envelope(N, coherence_R)
            coh_blocks.append(np.ravel([dval, dval], "F"))

        if verbose:
            print(f"  baseline {int(k) + 1}: {sel.sum()} events")

    uv = (np.concatenate(uv_blocks) if uv_blocks else np.zeros((0, 2))).reshape(-1, 2)
    vis = np.concatenate(vis_blocks) if vis_blocks else np.array([], dtype=complex)
    if coherence_R is not None:
        coh = np.concatenate(coh_blocks) if coh_blocks else np.array([], dtype=float)
        return uv, vis, coh
    return uv, vis


# ==========================================================================
# from reconstruct.py
# ==========================================================================


def _resolve_phase_center(events, phase_center=None):
    """Phase center ``(pitch, yaw)`` [rad] for an EventList.

    ``None`` (default) is the observation's mean recorded pointing
    (:meth:`~xrisalis.io.EventList.mean_boresight`). Otherwise a sky position
    in the same convention as source positions: ``(ra_deg, dec_deg)`` (bare
    numbers are degrees, Quantities are converted) or a SkyCoord, projected
    onto the tangent plane at the nominal pointing (``events.ra_pnt``,
    ``events.dec_pnt``) like the sources and the boresight.
    """
    if phase_center is None:
        return events.mean_boresight()
    if isinstance(phase_center, (tuple, list)) and len(phase_center) == 2:
        ra, dec = to_radec_deg(phase_center[0], phase_center[1])
    else:
        ra, dec = to_radec_deg(phase_center)
    east, north = tangent_offsets(ra, dec, events.ra_pnt, events.dec_pnt)
    if not (np.isfinite(east) and np.isfinite(north)):
        raise ValueError("phase_center is on the far side of the sky from the "
                         "nominal pointing")
    return np.array([float(north), float(east)])          # (pitch, yaw)


def _boresight_offsets(events, stream, correct_boresight, phase_center):
    """Per-event boresight relative to the phase center, and the center itself.

    Returns ``(offsets, center)``: ``offsets`` is the ``(N, 2)`` array
    ``P - c`` [rad] to hand to :func:`compute_uv` (``None`` when there is
    nothing to correct), ``center`` the phase center ``c`` as
    ``(pitch, yaw)`` [rad].

    The center is resolved from the *recorded* pointing, never from the
    attitude estimate in ``stream``: a constant bias in the estimate must show
    up as an image shift, as it would in real data, rather than cancel out.
    """
    center = _resolve_phase_center(events, phase_center)
    if not correct_boresight:
        return None, center
    off = events.boresight_per_event(stream) - center
    if not np.any(off):
        return None, center               # fixed stare on the phase centre
    return off, center


def _phase_center_deg(center, events):
    """``(pitch, yaw)`` [rad] -> ``(ra_deg, dec_deg)`` for metadata."""
    ra, dec = tangent_to_sky(center[1], center[0], events.ra_pnt, events.dec_pnt)
    return float(ra), float(dec)


def reconstruct(events, 
                instrument, 
                method="IFFT", 
                fov_arcsec=None,
                energy_kev=None, 
                baseline_bounds=None, 
                baseline_ids=None,
                samples=(256, 256), 
                npix=256, 
                fringe_taper=None,
                deconvolve=None, 
                deconv_epsilon=0.5,
                use_true_energy=False, 
                attitude=None, 
                correct_boresight=True,
                phase_center=None, 
                verbose=False):
    """Reconstruct an image from an EventList.

    Parameters
    ----------
    events : EventList
        Provides ``energy_j``, ``dety`` [m], ``baseline_id`` and the attitude
        (roll + boresight) per event, from the POINTING stream.
    attitude : optional
        The attitude stream to *interpret* the events with — i.e. what the
        attitude-determination system believes the spacecraft was doing. If
        None (default) the events' own POINTING stream is used, which is the
        attitude the simulation actually flew: **perfect knowledge**. Pass an
        estimated stream (an ``(T, 2|3)`` array, a
        :class:`~xrisalis.pointing.Pointing`, or any spec ``as_pointing``
        accepts) to reconstruct with a realistic attitude *error* and see what
        it costs the fringes. See
        :meth:`xrisalis.io.EventList.attitude_stream`.
    correct_boresight : bool
        Apply the per-event boresight phase referencing of
        :func:`compute_uv` (default True): every photon is referred to the
        ``phase_center`` using the attitude at its own arrival time, which
        removes the smear of a drifting or jittering boresight. Set False to
        see the raw, uncorrected image, in which the source sits at its
        position *relative to the boresight*. For a fixed stare at the phase
        center (the default for any constant pointing) the two are identical.
    phase_center : optional
        The sky position the image is centered on, as ``(ra_deg, dec_deg)``
        (bare numbers are degrees; Quantities and a SkyCoord are accepted).
        Image axes are true tangent-plane offsets from it (East along the
        columns' ``alpha``, North along the rows' ``delta``), so RA offsets
        include their ``cos(dec)``.
        Default None: the observation's mean recorded pointing
        (:meth:`~xrisalis.io.EventList.mean_boresight`), so the image is
        centered where the telescope looked and a source at offset ``dL`` from
        it appears at ``dL``. Pass the target's coordinates to center on the
        target instead — e.g. to undo a deliberate or erroneous *static*
        pointing offset, which the data alone cannot distinguish from an
        intended one. Must lie within a few fields of the pointing.
        Returned in ``meta['phase_center_deg']``.
    instrument : Interferometer
    method : {'IFFT', 'IFT'}
        The 'bin uv?' branch: gridded inverse FFT, or direct inverse DFT.
    fov_arcsec : float or None
        Field of view (largest axis) in arcseconds. If None (default), it is
        derived from the instrument/detector optics — the imaging FoV cached on
        the EventList at redistribute time, else
        :meth:`Interferometer.field_of_view` (see docs/fov_from_instrument.md).
        Pass a number or Quantity to override.
    energy_kev : optional (e_lo, e_hi) detected-energy band [keV].
    baseline_bounds : optional (lo, hi) inclusive range of baseline indices.
    baseline_ids : optional sequence of baseline indices to use — an arbitrary
        subset, where ``baseline_bounds`` can only take a contiguous range.
    samples : (int,int)  output grid for IFFT.
    npix : int           output grid for IFT.
    fringe_taper : None | float | callable
        Recovery **option 4**. Down-weights high fringe-order (chromatically
        decohered) photons. A number is the Gaussian cutoff ``n_cut`` in fringe
        orders; see :func:`~xrisalis.reconstruct.uvgrid.make_fringe_taper`.
    deconvolve : float, optional
        Recovery **option 2** (IFFT only): the detector resolving power
        ``R = E/sigma_E`` (see :func:`resolving_power`). Turns on the gridded
        matched-filter Wiener deconvolution (see :func:`ifft_image`), which
        weights each binned photon by its own chromatic coherence ``D(N)`` and
        normalizes per uv-cell — undoing chromatic decoherence in the visibility
        domain without amplifying individual decohered photons. Supersedes
        ``fringe_taper`` (ignored when set). Ignored for method='IFT'.
    deconv_epsilon : float
        Wiener regularization for ``deconvolve`` (fraction of the peak per-cell
        sum D^2). Larger = closer to an optimal taper (cleaner); smaller =
        more aggressive deconvolution (noisier). Default 0.5.
    use_true_energy : bool
        Diagnostic only. If True, reconstruct with the stored TRUE_ENERGY (the
        ideal, decoherence-free image) instead of the detected energy. Requires
        ``events.true_energy_kev``.

    Returns
    -------
    image : 2-D ndarray
    span_uas : float          angular span of the image [micro-arcsec]
    uv : (M,2) ndarray        uv sampling used
    meta : dict
    """
    _fov_derived = fov_arcsec is None
    fov_arcsec = _resolve_fov_arcsec(
        fov_arcsec, events, instrument, energy_kev, baseline_bounds, baseline_ids
    )
    fov = to_arcsec(fov_arcsec) * ARCSEC2RAD
    if energy_kev is not None:  # accept keV or Quantities
        energy_kev = (float(to_kev(energy_kev[0])), float(to_kev(energy_kev[1])))
    if use_true_energy:
        if events.true_energy_j is None:
            raise ValueError("use_true_energy=True needs a TRUE_ENERGY column.")
        energies_j = events.true_energy_j
    else:
        energies_j = events.energy_j
    stream = events.attitude_stream(attitude)
    roll = events.roll_per_event_from(stream)
    bore, center = _boresight_offsets(events, stream, correct_boresight, phase_center)

    mask, e, p, b, r = filter_events(
        energies_j,
        events.dety,
        events.baseline_id,
        roll,
        energy_kev=energy_kev,
        baseline_bounds=baseline_bounds,
        baseline_ids=baseline_ids,
    )
    bore = None if bore is None else bore[mask]
    if e.size == 0:
        raise ValueError("No events left after filtering.")

    # when the FoV was derived, keep the grid able to sample the finest fringe
    # (avoids IFFT overflow / aliasing when the FoV is large vs the pixel grid).
    # An explicit user FoV is left as given.
    if _fov_derived:
        _n = int(np.max(samples)) if method.upper() == "IFFT" else int(npix)
        _maxD = max(instrument.baselines[int(k)].D for k in np.unique(b))
        fov = (
            _grid_safe_fov_arcsec(fov * RAD2ARCSEC, e.max() / KEV, _maxD, _n)
            * ARCSEC2RAD
        )

    deconvolving = deconvolve is not None and method.upper() == "IFFT"
    if deconvolve is not None and method.upper() != "IFFT":
        raise ValueError("deconvolve is only supported for method='IFFT'.")

    if deconvolving:
        uv, vis, coh = compute_uv(
            e,
            p,
            b,
            r,
            instrument.baselines,
            fringe_taper=fringe_taper,
            boresight=bore,
            coherence_R=float(deconvolve),
            verbose=verbose,
        )
    else:
        uv, vis = compute_uv(
            e,
            p,
            b,
            r,
            instrument.baselines,
            fringe_taper=fringe_taper,
            boresight=bore,
            verbose=verbose,
        )
        coh = None

    if method.upper() == "IFFT":
        min_D = min(instrument.baselines[int(k)].D for k in np.unique(b))
        img, span, meta = ifft_image(
            uv,
            vis,
            fov,
            e,
            min_D,
            samples=samples,
            coherence=coh,
            deconv_epsilon=deconv_epsilon,
            verbose=verbose,
        )
    elif method.upper() == "IFT":
        img, span, meta = ift_image(uv, vis, fov, npix=npix)
    else:
        raise ValueError(f"Unknown method {method!r}; use 'IFFT' or 'IFT'.")

    meta["n_events"] = int(e.size)
    meta["deconvolved"] = bool(deconvolving)
    meta["phase_center_deg"] = _phase_center_deg(center, events)
    return img, span, uv, meta


def _pad_to(img, shape):
    """Center-pad a 2-D image with zeros to ``shape`` (same pixel scale)."""
    out = np.zeros(shape, dtype=img.dtype)
    r0 = (shape[0] - img.shape[0]) // 2
    c0 = (shape[1] - img.shape[1]) // 2
    r0, c0 = max(r0, 0), max(c0, 0)
    r1 = min(r0 + img.shape[0], shape[0])
    c1 = min(c0 + img.shape[1], shape[1])
    out[r0:r1, c0:c1] = img[: r1 - r0, : c1 - c0]
    return out


def reconstruct_subbands(
    events,
    instrument,
    bands,
    method="IFFT",
    fov_arcsec=None,
    baseline_bounds=None,
    baseline_ids=None,
    samples=(256, 256),
    npix=256,
    fringe_taper=None,
    combine="sum",
    normalize_bands=False,
    attitude=None,
    correct_boresight=True,
    phase_center=None,
    verbose=False,
):
    """Recovery **option 3**: energy sub-band (multi-frequency) synthesis.

    Reconstructs each detected-energy band separately and combines the images on
    a common grid. Each narrow band has a cleaner, less chromatic uv-sampling, so
    the combined image has a better-behaved beam than one broadband pass; it also
    yields per-band (spectral) images for free. Compose with ``fringe_taper`` to
    additionally suppress chromatic decoherence within each band.

    Parameters
    ----------
    bands : list[(e_lo, e_hi)]      detected-energy bands [keV].
    combine : {'sum', 'mean'}        how to combine the per-band images.
    normalize_bands : bool           peak-normalize each band before combining.
    attitude, correct_boresight, phase_center
        As for :func:`reconstruct`. The default phase center is taken from the
        whole event list, so every band is centered on the same sky point.

    Returns
    -------
    combined : 2-D ndarray
    span_uas : float
        Angular span of the **common (zero-padded) grid** the returned images
        live on — not of any one band's own crop, which may be smaller.
    band_images : list[2-D ndarray]  per-band images (padded to a common grid)
    meta : dict                      per-band event counts, bands
    """
    # resolve once (over the full band span) so every sub-band shares one grid
    _fov_derived = fov_arcsec is None
    _band_lo = min(float(b[0]) for b in bands)
    _band_hi = max(float(b[1]) for b in bands)
    fov_arcsec = _resolve_fov_arcsec(
        fov_arcsec,
        events,
        instrument,
        (_band_lo, _band_hi),
        baseline_bounds,
        baseline_ids,
    )
    if _fov_derived:  # cap to grid-safe; explicit FoV passes through
        _e_kev = events.energy_kev
        _max_e = float(min(_band_hi, _e_kev.max())) if _e_kev.size else _band_hi
        _maxD = float(np.max(instrument.baseline_lengths))
        _n = int(np.max(samples)) if method.upper() == "IFFT" else int(npix)
        fov_arcsec = _grid_safe_fov_arcsec(fov_arcsec, _max_e, _maxD, _n)
    target = tuple(samples) if method.upper() == "IFFT" else (npix, npix)
    band_images, spans, counts = [], [], []
    for e_lo, e_hi in bands:
        try:
            img, span, _, m = reconstruct(
                events,
                instrument,
                method=method,
                fov_arcsec=fov_arcsec,
                energy_kev=(e_lo, e_hi),
                baseline_bounds=baseline_bounds,
                baseline_ids=baseline_ids,
                samples=samples,
                npix=npix,
                fringe_taper=fringe_taper,
                attitude=attitude,
                correct_boresight=correct_boresight,
                phase_center=phase_center,
                verbose=verbose,
            )
        except ValueError:
            img = np.zeros(target)
            span = fov_arcsec * 1e6
            m = {"n_events": 0}
        img = _pad_to(img, target)
        if normalize_bands:
            peak = np.abs(img).max()
            img = img / peak if peak > 0 else img
        band_images.append(img)
        spans.append(span)
        counts.append(m["n_events"])
        if verbose:
            print(f"  band {e_lo}-{e_hi} keV: {m['n_events']} events")

    stack = np.stack(band_images)
    combined = stack.sum(0) if combine == "sum" else stack.mean(0)
    # The span of the *common* grid, not spans[0]: the returned images are all
    # zero-padded to ``target``, and band 0's own crop can be smaller than that
    # (its internal grid n depends on wavelength, the pixel scale d does not).
    fov_uas = fov_arcsec * ARCSEC2RAD * RAD2UAS
    span_uas = (
      fov_uas if method.upper() == "IFT" else target[0] * fov_uas / np.max(samples)
    )
    meta = {"bands": list(bands), 
            "counts": counts, 
            "method": method.upper(),
            "combine": combine, 
            "band_spans_uas": spans,
            "pixel_scale_uas": span_uas / target[0],
            "phase_center_deg": _phase_center_deg(
                _resolve_phase_center(events, phase_center), events)
           }
    return combined, span_uas, band_images, meta


# ==========================================================================
# Section 4: false-color (RGB) reconstruction
# ==========================================================================


def dirty_beam(uv, vis, fov, energies_j, min_baseline_D, samples=(256, 256), **kw):
    """The dirty beam (point-source response) belonging to a uv-sampling.

    A point source at the phase center has unimodular visibilities, so the beam
    of a *weighted* reconstruction is the transform of the weights alone — and
    ``|vis|`` is exactly the per-sample weight :func:`compute_uv` applied (the
    fringe taper, or 1 with no taper).  Passing the same ``uv``, ``fov``,
    ``energies_j``, ``min_baseline_D`` and ``samples`` as the image guarantees
    the beam lands on the same grid at the same pixel scale, so the two can be
    compared and convolved directly.

    Returns ``(beam, span_uas, meta)`` — the same contract as :func:`ifft_image`.
    """
    return ifft_image(
        uv,
        np.abs(vis).astype(complex),
        fov,
        energies_j,
        min_baseline_D,
        samples=samples,
        **kw,
    )


def beam_sigma_uas(beam, pixel_scale_uas):
    """Gaussian-equivalent sigma of a dirty beam's central lobe [micro-arcsec].

    Measured from the azimuthally averaged profile around the peak: the radius
    at which it first drops below half the maximum is the HWHM, and
    ``sigma = HWHM / sqrt(2 ln 2)``.  Preferred over a 2-D Gaussian fit, which
    the sidelobe structure of a dirty beam makes unstable.

    The half-maximum crossing is interpolated linearly between the two
    bracketing radius bins.  Without that the answer is quantized to whole
    pixels, and on a coarse grid two genuinely different beams can measure
    identical — whereupon the beam matching they feed silently does nothing.
    """
    b = np.asarray(beam, float)
    peak = b.max()
    if not np.isfinite(peak) or peak <= 0:
        return float("nan")
    b = b / peak
    cy, cx = np.unravel_index(int(np.argmax(b)), b.shape)
    yy, xx = np.mgrid[0 : b.shape[0], 0 : b.shape[1]]
    rr = np.hypot(yy - cy, xx - cx)
    nb = int(min(b.shape) // 2)
    # A bin covers radii [k, k+1) but holds more pixels at its outer edge, so
    # its representative radius is the mean radius of its pixels, not k + 0.5.
    masks = [(rr >= k) & (rr < k + 1) for k in range(nb)]
    prof = np.array([b[m].mean() for m in masks])
    prof_r = np.array([rr[m].mean() for m in masks])
    below = np.nonzero(prof < 0.5)[0]
    if not below.size:
        r_half = float(nb)
    elif below[0] == 0:
        r_half = float(prof_r[0])  # unresolved: narrower than one pixel
    else:
        k = int(below[0])
        y0, y1 = prof[k - 1], prof[k]
        frac = (y0 - 0.5) / (y0 - y1) if y0 > y1 else 0.0
        r_half = float(
            prof_r[k - 1] + np.clip(frac, 0.0, 1.0) * (prof_r[k] - prof_r[k - 1])
        )
    return float(r_half / np.sqrt(2 * np.log(2)) * pixel_scale_uas)


def match_resolution(images, sigmas_uas, pixel_scale_uas, target_sigma_uas=None):
    """Convolve images down to a common (the coarsest) resolution.

    Bands imaged with the same array still have *different* beams, because the
    uv-sampling scales with wavelength and the effective area is energy
    dependent.  Compositing them raw paints an instrumental color gradient
    onto every source — a colored core inside a differently colored halo —
    which is indistinguishable, by eye, from a real spectral gradient.  Blurring
    each image by ``sqrt(sigma_ref^2 - sigma^2)`` puts them all on the coarsest
    beam, after which color is spectral shape.

    Returns ``(images, sigma_ref_uas)``.
    """
    from scipy.ndimage import gaussian_filter  # lazy import

    sig = np.asarray(sigmas_uas, float)
    s_ref = float(np.nanmax(sig) if target_sigma_uas is None else target_sigma_uas)
    out = []
    for img, s in zip(images, sig):
        img = np.asarray(img, float)
        extra = np.sqrt(max(s_ref**2 - float(s) ** 2, 0.0)) / pixel_scale_uas
        out.append(
            gaussian_filter(img, extra, mode="constant")
            if np.isfinite(extra) and extra > 0.05
            else img
        )
    return out, s_ref


def _per_channel(spec, n=3):
    """Broadcast an option: one value for all channels, or one per channel.

    A flat spec (``None``, ``(7, 9)``, ``[0, 1, 2]``) applies to all ``n``
    channels; a length-``n`` sequence whose entries are themselves sequences (or
    ``None``) is taken as one spec per channel.
    """
    if spec is None:
        return [None] * n
    seq = list(spec)
    if len(seq) == n and all(s is None or np.ndim(s) >= 1 for s in seq):
        return seq
    return [spec] * n


@dataclass
class RGBImage:
    """Three co-registered band images destined for the R, G and B channels.

    This is the *data* product: images on a common grid, their dirty beams and
    the numbers needed to interpret them.  Turning it into pixels you can look
    at — black point, white balance, stretch — is a display choice and lives in
    :mod:`xrisalis.viz`; :meth:`rgb` and :meth:`plot` are shortcuts to it.
    """

    channels: list  # 3 images (R, G, B), common grid
    span_uas: float  # angular span of that grid
    pixel_scale_uas: float
    bands: list  # 3 (e_lo, e_hi) in keV
    counts: list  # events per band
    beams: list = field(default_factory=list)  # 3 dirty beams, or []
    beam_sigmas_uas: list = field(default_factory=list)
    matched_sigma_uas: float = None  # None if the beams were not matched
    channels_raw: list = field(default_factory=list)  # before beam matching
    meta: dict = field(default_factory=dict)

    # -- display shortcuts (implemented in viz) ------------------------------
    def rgb(self, **kw):
        """Composite to an ``(ny, nx, 3)`` array — see :func:`viz.rgb_composite`."""
        from .viz import rgb_composite

        return rgb_composite(self, **kw)

    def plot(self, **kw):
        """Plot the composite — see :func:`viz.plot_rgb`."""
        from .viz import plot_rgb

        return plot_rgb(self, **kw)

    # -- FITS ----------------------------------------------------------------
    def write(self, path, overwrite=True, beams=True):
        """Write the three band images (and their beams) to a FITS file.

        One ``ImageHDU`` per channel — ``EXTNAME = R, G, B`` — each carrying its
        energy band, event count and measured beam width, plus a linear WCS in
        degrees so the file opens correctly in DS9/astropy.  With ``beams`` the
        dirty beams follow as ``BEAM_R``/``BEAM_G``/``BEAM_B``.
        """
        from astropy.io import fits

        from .io import write_hdulist

        hdus = []
        for i, name in enumerate("RGB"):
            lo, hi = self.bands[i]
            h = fits.Header()
            h["EXTNAME"] = name
            h["BAND_LO"] = (float(lo), "band lower edge [keV]")
            h["BAND_HI"] = (float(hi), "band upper edge [keV]")
            h["NEVENTS"] = (int(self.counts[i]), "events in this band")
            if i < len(self.beam_sigmas_uas) and np.isfinite(self.beam_sigmas_uas[i]):
                h["BEAMSIG"] = (
                    float(self.beam_sigmas_uas[i]),
                    "dirty-beam sigma [uas]",
                )
            if self.matched_sigma_uas is not None:
                h["MATCHSIG"] = (
                    float(self.matched_sigma_uas),
                    "common beam sigma after matching [uas]",
                )
            h["SPANUAS"] = (float(self.span_uas), "image span [uas]")
            h["PIXUAS"] = (float(self.pixel_scale_uas), "pixel scale [uas]")
            for k, v in (self.meta or {}).items():
                if isinstance(v, (int, float, str, bool)) and len(k) <= 8:
                    h[k.upper()] = v
            ny, nx = np.shape(self.channels[i])
            deg = self.pixel_scale_uas / 3.6e9
            h["CTYPE1"], h["CTYPE2"] = "RA---SIN", "DEC--SIN"
            h["CUNIT1"], h["CUNIT2"] = "deg", "deg"
            # phase center at pixel (n//2, n//2); columns run East, rows North
            h["CRPIX1"], h["CRPIX2"] = nx // 2 + 1, ny // 2 + 1
            pc = (self.meta or {}).get("phase_center_deg") or (0.0, 0.0)
            h["CRVAL1"], h["CRVAL2"] = float(pc[0]), float(pc[1])
            h["CDELT1"], h["CDELT2"] = deg, deg
            hdus.append(
                fits.ImageHDU(np.asarray(self.channels[i], np.float32), header=h)
            )
        if beams and self.beams:
            for i, name in enumerate("RGB"):
                if self.beams[i] is None:
                    continue
                bh = fits.Header()
                bh["EXTNAME"] = f"BEAM_{name}"
                hdus.append(
                    fits.ImageHDU(np.asarray(self.beams[i], np.float32), header=bh)
                )
        write_hdulist(hdus, path, overwrite=overwrite)
        return path

    @classmethod
    def read(cls, path):
        """Read back a file written by :meth:`write`."""
        from astropy.io import fits

        with fits.open(path) as hdul:
            ch, bands, counts, sig, beams = [], [], [], [], []
            for name in "RGB":
                hdu = hdul[name]
                ch.append(np.asarray(hdu.data, float))
                bands.append((hdu.header["BAND_LO"], hdu.header["BAND_HI"]))
                counts.append(int(hdu.header.get("NEVENTS", 0)))
                sig.append(float(hdu.header.get("BEAMSIG", np.nan)))
                key = f"BEAM_{name}"
                beams.append(np.asarray(hdul[key].data, float) if key in hdul else None)
            h0 = hdul["R"].header
            return cls(
                channels=ch,
                span_uas=float(h0["SPANUAS"]),
                pixel_scale_uas=float(h0["PIXUAS"]),
                bands=bands,
                counts=counts,
                beams=beams,
                beam_sigmas_uas=sig,
                matched_sigma_uas=(float(h0["MATCHSIG"]) if "MATCHSIG" in h0 else None),
                meta={"source": str(path)},
            )


def reconstruct_rgb(
    events,
    instrument,
    bands,
    baseline_ids=None,
    baseline_bounds=None,
    fov_arcsec=None,
    samples=(256, 256),
    fringe_taper=None,
    match_beams=True,
    with_beams=True,
    use_true_energy=False,
    attitude=None,
    correct_boresight=True,
    phase_center=None,
    verbose=False,
):
    """Reconstruct three energy bands onto a common grid for a color image.

    The interferometric version of the familiar false-color composite: one
    reconstruction per detected-energy band, mapped to red, green and blue, so
    that hue carries spectral shape while brightness carries flux.

    Parameters
    ----------
    events : EventList
    instrument : Interferometer
    bands : sequence of exactly three ``(e_lo, e_hi)`` in keV
        Assigned to R, G and B in order — conventionally soft to hard, so that
        soft sources come out red and hard ones blue.  They need not be
        contiguous or equal in width; widening the hard band to buy counts is
        usually the right trade.
    baseline_ids : optional
        Which baselines to image with, as an explicit sequence of baseline
        indices.  Pass one sequence to use the same baselines for all three
        channels, or a list of three sequences to choose them per channel.
        ``None`` uses every baseline present in the events.
    baseline_bounds : optional
        The contiguous alternative: an inclusive ``(lo, hi)`` index range, again
        either one for all channels or a list of three.  Combined with
        ``baseline_ids`` if both are given.

        Selecting long baselines to sharpen the beam is tempting and often
        wrong: effective area is energy dependent, and an array whose long
        baselines have little hard-band area will simply leave the blue channel
        empty.  Check the per-baseline ARF (:func:`viz.plot_arf`) first.
    fov_arcsec : float          field of view (largest axis).
    samples : (int,int)         output grid; all channels are padded to it.
    fringe_taper : None | float | callable
        Fringe-order taper, applied identically to every band.  Strongly
        recommended here: chromatic decoherence hits the hard band hardest —
        fewest counts, longest fringe orders — and without a taper the blue
        channel is often pure speckle.
    match_beams : bool
        Convolve the sharper bands down to the coarsest beam (default), so hue
        reflects the source rather than the instrument.  See
        :func:`match_resolution`.
    with_beams : bool           compute the per-band dirty beams at all.
    use_true_energy : bool      diagnostic; reconstruct with TRUE_ENERGY.
    attitude, correct_boresight, phase_center
        As for :func:`reconstruct`; one phase center for all three channels.

    Returns
    -------
    RGBImage
        Call ``.rgb()`` for an ``(ny, nx, 3)`` display array, ``.plot()`` to
        draw it, or ``.write(path)`` for FITS.

    Notes
    -----
    Padding every band to a common grid is exact, not an approximation: the
    pixel scale ``d = fov / max(samples)`` does not depend on energy, so the
    bands already share it and only the internal grid size differs.
    """
    bands = [tuple(b) for b in bands]
    if len(bands) != 3:
        raise ValueError(
            f"reconstruct_rgb needs exactly three bands (R, G, B); got {len(bands)}."
        )
    _fov_derived = fov_arcsec is None
    _band_lo = min(float(b[0]) for b in bands)
    _band_hi = max(float(b[1]) for b in bands)
    fov_arcsec = _resolve_fov_arcsec(
        fov_arcsec, events, instrument, (_band_lo, _band_hi)
    )
    if _fov_derived:  # cap to grid-safe (finest fringe); explicit left as-is
        _e_kev = events.energy_kev
        _max_e = float(min(_band_hi, _e_kev.max())) if _e_kev.size else _band_hi
        _maxD = float(np.max(instrument.baseline_lengths))
        fov_arcsec = _grid_safe_fov_arcsec(
            fov_arcsec, _max_e, _maxD, int(np.max(samples))
        )
    fov = fov_arcsec * ARCSEC2RAD
    target = tuple(samples)

    if use_true_energy:
        if events.true_energy_j is None:
            raise ValueError("use_true_energy=True needs a TRUE_ENERGY column.")
        energies_j = events.true_energy_j
    else:
        energies_j = events.energy_j
    _stream = events.attitude_stream(attitude)
    roll = events.roll_per_event_from(_stream)
    _bore, _center = _boresight_offsets(
        events, _stream, correct_boresight, phase_center
    )

    ids_per = _per_channel(baseline_ids)
    bounds_per = _per_channel(baseline_bounds)

    pixel_scale_uas = (fov / np.max(samples)) * RAD2UAS
    span_uas = target[0] * pixel_scale_uas

    channels, beams, counts, sigmas, used = [], [], [], [], []
    for (lo, hi), ids, bounds in zip(bands, ids_per, bounds_per):
        lo, hi = float(to_kev(lo)), float(to_kev(hi))
        _m, e, p, b, r = filter_events(
            energies_j,
            events.dety,
            events.baseline_id,
            roll,
            energy_kev=(lo, hi),
            baseline_bounds=bounds,
            baseline_ids=ids,
        )
        bore = None if _bore is None else _bore[_m]
        if e.size == 0:
            channels.append(np.zeros(target))
            beams.append(None)
            counts.append(0)
            sigmas.append(np.nan)
            used.append([])
            if verbose:
                print(f"  band {lo}-{hi} keV: NO EVENTS")
            continue

        uv, vis = compute_uv(
            e, p, b, r, instrument.baselines, fringe_taper=fringe_taper, boresight=bore
        )
        keys = np.unique(b)
        min_D = min(instrument.baselines[int(k)].D for k in keys)
        img, _, _ = ifft_image(uv, vis, fov, e, min_D, samples=samples)
        channels.append(_pad_to(img, target))
        counts.append(int(e.size))
        used.append([int(k) for k in keys])

        if with_beams:
            bm, _, _ = dirty_beam(uv, vis, fov, e, min_D, samples=samples)
            bm = _pad_to(bm, target)
            beams.append(bm)
            sigmas.append(beam_sigma_uas(bm, pixel_scale_uas))
        else:
            beams.append(None)
            sigmas.append(np.nan)
        if verbose:
            print(
                f"  band {lo}-{hi} keV: {counts[-1]} events, "
                f"beam sigma {sigmas[-1]:.1f} uas"
            )

    raw = [c.copy() for c in channels]
    matched_sigma = None
    if match_beams and with_beams and np.all(np.isfinite(sigmas)):
        channels, matched_sigma = match_resolution(channels, sigmas, pixel_scale_uas)
        if verbose:
            print(f"  matched to sigma {matched_sigma:.1f} uas")
    elif match_beams and verbose:
        print("  beams NOT matched (an empty band, or with_beams=False)")

    meta = {"method": "IFFT", 
            "fov_arcsec": float(fov_arcsec),
            "fringe_taper": (fringe_taper if isinstance(fringe_taper,
                                                        (int, float))
                             else str(fringe_taper)),
            "baselines_used": used, "matched": matched_sigma is not None,
            "phase_center_deg": _phase_center_deg(_centre, events)}
    return RGBImage(channels=channels, 
                    span_uas=span_uas,
                    pixel_scale_uas=pixel_scale_uas, 
                    bands=bands,
                    counts=counts, 
                    beams=beams, 
                    beam_sigmas_uas=sigmas,
                    matched_sigma_uas=matched_sigma, 
                    channels_raw=raw,
                    meta=meta)
