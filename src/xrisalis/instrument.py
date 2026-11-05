"""Instrument model: Baseline geometry and the Interferometer (baselines +
attitude/pointing model + field of view)."""

import math
import numpy as np

from .config import H, C, KEV
from .pointing import as_pointing


# ==========================================================================
# from baseline.py
# ==========================================================================

class Baseline:
    """A single slatted-mirror baseline of an X-ray interferometer.

    Exactly one of the geometry pairs must be supplied so the rest can be
    derived (D<->F, W<->L via beam_angle). Units are SI on the way in except
    ``W`` which is given in micrometres for convenience and stored in metres.
    """

    def __init__(self, num_pairs, D=None, L=None, W=None, beam_angle=None,
                 F=None, grazing_angle=None, bench_length=None,
                 mirr_reflec=None, Wz=1e-2, slat_pitch=None,
                 combiner="slatted", overlap_width=None, visibility=1.0):
        # --- beam angle ---
        if beam_angle is not None:
            self.beam_angle = beam_angle
        else:
            try:
                self.beam_angle = W * 1e-6 / L
            except TypeError:
                raise ValueError(
                    "Define either beam_angle, or both W (um) and L (m).")
        # --- baseline length D <-> focal length F ---
        if D is not None:
            self.D = D
        else:
            try:
                self.D = F * self.beam_angle
            except TypeError:
                raise ValueError("Define either D or F.")
        self.F = F if F is not None else self.D / self.beam_angle
        # --- beam width W <-> combining length L ---
        if W is not None:
            self.W = W * 1e-6
        else:
            try:
                self.W = self.beam_angle * L
            except TypeError:
                raise ValueError("Define either W (um) or L (m).")
        self.L = L if L is not None else self.W / self.beam_angle
        if bench_length is not None:
            self.bench_length = bench_length
        # --- grazing angle ---
        if grazing_angle is not None:
            self.grazing_angle = grazing_angle
        else:
            try:
                self.grazing_angle = np.arctan(self.D / 2 / self.bench_length) / 2
            except (TypeError, AttributeError):
                self.grazing_angle = None

        self.num_pairs = num_pairs
        self.mirr_reflec = mirr_reflec
        # mirror aperture along the CROSS (z / DETZ) axis [m] — the "height" of
        # the slat, order cm. Unlike the fringe (DETY) axis, there is no
        # interference here: the two beams are separated along the baseline (y)
        # direction, so z just carries the single-aperture illumination of the
        # mirror. Sets the DETZ spread (see xrisalis.simulate.sample_detz_positions).
        self.Wz = Wz

        # SLATTED-MIRROR APERTURE (fringe / DETY axis). ``num_pairs`` slats, each
        # of width ``W``, at aperture pitch ``slat_pitch`` (>= W; default = W,
        # i.e. contiguous, no gap). The two interfering arms are separated by the
        # baseline ``D`` (two-arm offset). On the detector the beams tile into a
        # COMB of ``num_pairs`` single-slit envelopes at pitch ``slat_pitch``
        # (magnification 1 for the flat combiner), each filled by the global
        # two-beam fringe of period ``lambda/beam_angle``; a gap
        # ``slat_pitch - W`` opens dark bands between patches. See
        # xrisalis.simulate.sample_detector_positions. Placeholder geometry —
        # refine with the real slat pitch/gap/steering later.
        if slat_pitch is None:
            slat_pitch = self.W
        if slat_pitch < self.W - 1e-18:
            raise ValueError("slat_pitch must be >= slat width W (slats cannot "
                             "overlap); use slat_pitch=W for a contiguous comb.")
        self.slat_pitch = slat_pitch

        # BEAM COMBINER MODEL (fringe / DETY axis).
        # * 'slatted' (default): the slatted-mirror comb above -- Fresnel
        #   single-slat envelopes x global two-beam fringe (accept-reject in
        #   xrisalis.simulate.sample_detector_positions).
        # * 'arbitrary': an idealised combiner (e.g. a beamsplitter, as in the
        #   Accretion Explorer Interferometer). Two-beam fringes of period
        #   lambda/beam_angle and contrast ``visibility``, NO diffraction: the
        #   interferogram is a top-hat window of width ``overlap_width`` (the
        #   region where the beams overlap; default = comb_span = W*num_pairs)
        #   with a hard cut-off at its edges. Every photon lands inside the
        #   window -- none are lost to diffraction wings or non-overlap regions.
        #   See xrisalis.simulate.sample_arbitrary_positions.
        combiner = str(combiner).lower()
        if combiner not in ("slatted", "arbitrary"):
            raise ValueError("combiner must be 'slatted' or 'arbitrary', "
                             f"got {combiner!r}")
        self.combiner = combiner
        if overlap_width is not None and not overlap_width > 0:
            raise ValueError("overlap_width must be > 0 (metres)")
        self.overlap_width = None if overlap_width is None else float(overlap_width)
        if not 0.0 <= float(visibility) <= 1.0:
            raise ValueError("visibility must be in [0, 1]")
        self.visibility = float(visibility)

        # physical consistency checks (isclose guards against binary rounding)
        if not math.isclose(self.D, self.F * self.beam_angle):
            raise ValueError("D, F and beam_angle are inconsistent.")
        if not math.isclose(self.W, self.beam_angle * self.L):
            raise ValueError("W, L and beam_angle are inconsistent.")
        if not isinstance(self.num_pairs, (int, np.integer)) or self.num_pairs <= 0:
            raise ValueError("num_pairs must be a positive integer.")

    @classmethod
    def willingale(cls, D, Wz=1e-2, slat_pitch=None):
        """Preset baseline from Willingale (2004): 30 pairs, L=10, W=300um, bench=7m.

        ``Wz`` is the mirror z-aperture (cross/DETZ axis), default 1 cm.
        ``slat_pitch`` is the aperture slat pitch (default = W, contiguous comb).
        """
        return cls(num_pairs=30, D=D, L=10, W=300, bench_length=7, Wz=Wz,
                   slat_pitch=slat_pitch)

    @property
    def slat_gap(self):
        """Dark gap between adjacent slat patches on the detector [m]."""
        return self.slat_pitch - self.W

    @property
    def comb_span(self):
        """Full DETY extent of the slat comb [m] = num_pairs * slat_pitch."""
        return self.num_pairs * self.slat_pitch

    def fresnel_number_z(self, energy_kev, distance=None):
        """Fresnel number ``N_F = (Wz/2)^2 / (lambda * Z)`` for the z aperture.

        ``Z`` is the propagation distance to the detector (defaults to the
        combining length ``L``). ``N_F >> 1`` -> geometric shadow (the DETZ
        profile is the mirror's geometric image, ~wavelength-independent);
        ``N_F ~ 1`` -> Fresnel; ``N_F << 1`` -> Fraunhofer. For cm-scale mirrors
        at the ~10 m detector plane this is in the hundreds-to-thousands
        (geometric); it only approaches Fresnel over ~km propagation.
        """
        lam = H * C / (float(energy_kev) * KEV)
        Z = self.L if distance is None else float(distance)
        return (self.Wz / 2.0) ** 2 / (lam * Z)

    @property
    def overlap_span(self):
        """Full DETY extent of the interferogram [m].

        ``overlap_width`` for an arbitrary combiner if set, else the slat comb
        span ``num_pairs * slat_pitch``.
        """
        if self.combiner == "arbitrary" and self.overlap_width is not None:
            return self.overlap_width
        return self.comb_span

    def __repr__(self):
        extra = ""
        if self.combiner != "slatted":
            extra = (f", combiner={self.combiner!r}, "
                     f"overlap={self.overlap_span * 1e3:.4g} mm")
        return (f"Baseline(D={self.D:.4g} m, L={self.L:.4g} m, "
                f"W={self.W * 1e6:.4g} um, num_pairs={self.num_pairs}{extra})")

    # -- field of view --------------------------------------------------------
    def field_of_view(self, energy_kev=None, R=None, detector=None,
                      which="imaging", coherence_factor=1.0):
        """Half-angle field of view [rad] set by this baseline's optics/detector.

        Four mechanisms (see docs/fov_from_instrument.md); the FoV is the
        minimum of those that apply:

        * **geometry** — the fringe packet must land on the illuminated region
          (``W * num_pairs``, capped by the detector's physical extent):
          ``θ_geo ≈ half / (D/β − L)`` with ``β = beam_angle``,
          ``half = min(W·num_pairs/2, detector.half_extent)``. This is the hard
          edge enforced by the fringe sampler (a source beyond it cannot be
          placed), so it is what redistribute vignettes against. For an
          ``'arbitrary'`` combiner ``half = overlap_span/2``; beyond this angle
          the white-light (zero-OPD) fringe leaves the window, but photons are
          NOT vignetted (the window receives every photon) -- what remains are
          high-order fringes that decohere under the detector resolution.
        * **coherence** — bandwidth/chromatic decoherence limits the usable
          fringe count to ``≈ R`` periods (``R = E/σ_E``, the detector resolving
          power): ``θ_coh ≈ coherence_factor · R · (λ/D)``. Theoretical scaling
          ``θ_coh ~ R·λ/D``; ``coherence_factor`` (default 1) sets where along
          the ``exp(-2π²N²/R²)`` envelope the cut is taken.
        * **diffraction** — single-slat envelope ``θ_env ≈ λ/W`` (informational;
          rarely binds).

        Parameters
        ----------
        energy_kev : float, optional
            Energy at which λ-dependent terms (coherence, diffraction) are
            evaluated. If None, only the (energy-independent) geometry applies.
        R : float, optional
            Detector resolving power ``E/σ_E`` at ``energy_kev`` (e.g. from
            :func:`xrisalis.reconstruct.resolving_power`). Needed for coherence.
        detector : Detector, optional
            Caps the illuminated region by the physical extent.
        which : {'imaging','detection','geometry','coherence','diffraction','cross'}
            'detection' → geometry only (used for vignetting); 'imaging' (the
            default / 'auto') → min(geometry, coherence); 'cross' → the
            non-interferometric (DETZ) axis, half the beam angle ``β/2``.
        coherence_factor : float
            Prefactor on the coherence term.

        Returns
        -------
        float
            Half-angle field of view [rad] (``np.inf`` if the selected
            mechanism does not apply, e.g. coherence without ``R``).
        """
        if which == "auto":
            which = "imaging"
        half = (self.overlap_span if self.combiner == "arbitrary"
                else self.W * self.num_pairs) / 2.0
        if detector is not None and detector.fringe_half_extent_m() is not None:
            half = min(half, detector.fringe_half_extent_m())
        denom = self.D / self.beam_angle - self.L
        fov_geo = np.inf if denom <= 0 else half / denom

        fov_coh = np.inf
        fov_env = np.inf
        if energy_kev is not None:
            lam = H * C / (float(energy_kev) * KEV)
            fov_env = lam / self.W
            if R is not None and np.isfinite(R) and R > 0:
                fov_coh = coherence_factor * R * lam / self.D

        table = {
            "geometry": fov_geo,
            "detection": fov_geo,
            "coherence": fov_coh,
            "diffraction": fov_env,
            "imaging": min(fov_geo, fov_coh),
            "cross": self.beam_angle / 2.0,
        }
        if which not in table:
            raise ValueError(f"unknown FoV kind {which!r}; use one of {list(table)}")
        return float(table[which])

# ==========================================================================
# from interferometer.py
# ==========================================================================

def smooth_roll(inst, pointing):
    """Continuous roll at constant rate spanning pi over the exposure.

    ``roll_speed`` is a genuine *rate* [rad/s], so the sweep spans pi over the
    exposure whatever ``time_step`` is (at ``time_step=1`` this is identical to
    the historical ``roll_speed = pi/n``).
    """
    n = pointing[:, 2].size
    inst.roll_speed = np.pi / (n * inst.time_step)
    pointing[:, 2] = (np.arange(n) * inst.roll_speed * inst.time_step) + inst.roll_init
    return pointing


def no_roll(inst, pointing):
    return pointing


class Interferometer:
    def __init__(self, time_step=1, roller=None, roll_init=0.0,
                 wobble=None, wobble_sigma=0.0, fov=None, pointing=None,
                 detector=None):
        self.baselines = []
        self.time_step = time_step
        self.roller = roller
        self.roll_speed = 0.0
        self.roll_init = roll_init
        self.wobble = wobble
        self.wobble_sigma = wobble_sigma
        # radians (largest-axis field of view). If set explicitly it overrides
        # the derived value; None -> derive from optics/detector (field_of_view).
        self.fov = fov
        self.detector = detector  # xrisalis.detector.Detector, or None
        # boresight attitude over the observation (see xrisalis.pointing). None
        # keeps the boresight on-axis (0,0); roll comes from `roller` unless the
        # pointing itself supplies a roll column.
        self.pointing = as_pointing(pointing)

    def set_pointing(self, pointing):
        """Set the boresight attitude model (see :mod:`xrisalis.pointing`)."""
        self.pointing = as_pointing(pointing)
        return self

    # -- construction ---------------------------------------------------------
    def add_baseline(self, *args, **kwargs):
        self.baselines.append(Baseline(*args, **kwargs))
        return self

    def add_willingale_baseline(self, D):
        self.baselines.append(Baseline.willingale(D))
        return self

    @classmethod
    def from_arf(cls, arf, roller=None, fov_arcsec=None, baseline_kwargs=None,
                 **kwargs):
        """Build an interferometer whose baselines match an ARF's per-baseline
        lengths. ``arf`` is an xrisalis.io.ARF object.

        With ``baseline_kwargs=None`` each baseline gets the Willingale (2004)
        geometry. Otherwise ``baseline_kwargs`` is a dict of
        :class:`Baseline` arguments (everything except ``D``) applied to every
        baseline, e.g. an arbitrary-combiner geometry::

            Interferometer.from_arf(arf, baseline_kwargs=dict(
                num_pairs=1, W=26e3, beam_angle=4.85e-6, combiner="arbitrary"))
        """
        fov = None if fov_arcsec is None else fov_arcsec * (3.141592653589793 /
                                                            (180.0 * 3600.0))
        inst = cls(roller=roller, fov=fov, **kwargs)
        for D in arf.baseline_lengths:
            if baseline_kwargs is None:
                inst.add_willingale_baseline(float(D))
            else:
                inst.add_baseline(D=float(D), **baseline_kwargs)
        return inst

    def clear_baselines(self):
        self.baselines.clear()

    # -- attitude -------------------------------------------------------------
    def gen_pointing(self, t_exp, pointing=None):
        """Attitude ``(T, 3)`` of ``(pitch, yaw, roll)`` [rad], one row per step.

        ``t_exp`` is the observation length in **seconds**; the stream is
        sampled every ``self.time_step`` seconds, so it has
        ``T = int(t_exp / time_step) + 2`` rows spanning the exposure. (With the
        default ``time_step = 1`` this is the historical ``int(t_exp) + 2``.)
        Sub-second ``time_step`` is what an attitude study needs: a jitter or
        gyro-residual stream carrying content near 1 Hz has to be sampled well
        below a second, while the photons still arrive over the full exposure.
        Use :func:`~xrisalis.pointing.attitude_index` to map photon times onto
        rows — never ``int(t)``, which is only correct at ``time_step = 1``.

        The boresight (pitch, yaw) columns come from the pointing model
        (``pointing`` argument if given, else ``self.pointing``; see
        :mod:`xrisalis.pointing`). Roll is then swept by ``self.roller`` unless
        the pointing model itself supplies a roll column
        (``provides_roll=True``). Finally ``self.wobble`` is applied if set.
        With no pointing model the boresight stays at (0, 0).
        """
        T = int(float(t_exp) / float(self.time_step)) + 2
        attitude = np.zeros((T, 3))
        source = self.pointing if pointing is None else as_pointing(pointing)
        provides_roll = False
        if source is not None:
            t = np.arange(T) * self.time_step
            att = source(t)
            attitude[:, :att.shape[1]] = att
            provides_roll = source.provides_roll
        if self.roller is not None and not provides_roll:
            attitude = self.roller(self, attitude)
        if self.wobble is not None and self.wobble_sigma > 0:
            attitude = self.wobble(self, attitude)
        return attitude

    @property
    def n_baselines(self):
        return len(self.baselines)

    @property
    def baseline_lengths(self):
        return np.array([b.D for b in self.baselines])

    # -- field of view --------------------------------------------------------
    def _selected_baseline_indices(self, baseline_bounds=None, baseline_ids=None):
        idx = np.arange(len(self.baselines))
        if baseline_bounds is not None:
            lo, hi = baseline_bounds
            idx = idx[(idx >= lo) & (idx <= hi)]
        if baseline_ids is not None:
            idx = idx[np.isin(idx, np.asarray(baseline_ids, int))]
        return idx

    def field_of_view(self, energy_kev=None, rmf=None, baseline_bounds=None,
                      baseline_ids=None, which="imaging", combine="min",
                      reduce="low_edge", coherence_factor=1.0,
                      resolving_power=None):
        """Field of view [rad] derived from the selected baselines' optics/detector.

        Applies :meth:`Baseline.field_of_view` over the selected baselines and
        combines them. A source must lie within every used baseline's FoV, so
        the imaging FoV is the **minimum** across baselines (``combine='min'``,
        the default; ``'max'`` also available).

        Parameters
        ----------
        energy_kev : float or (lo, hi), optional
            Energy [keV] for the λ-dependent terms. A ``(lo, hi)`` band is
            reduced per ``reduce``; a scalar is used directly; None gives the
            energy-independent geometry FoV.
        rmf : RMF, optional
            Used to evaluate the detector resolving power ``R`` at the chosen
            energy (for the coherence term). Ignored if ``resolving_power`` is
            given. Without either, the coherence term is dropped (geometry only).
        which, coherence_factor
            Passed through to :meth:`Baseline.field_of_view`.
        combine : {'min','max'}
            How to combine the per-baseline FoVs. Default 'min'.
        reduce : {'low_edge','high_edge','min'}
            For a ``(lo, hi)`` band: evaluate at the low edge (default — the
            largest λ, hence the most inclusive imaging window here), the high
            edge, or take the min over both.
        resolving_power : callable, optional
            ``R(energy_kev)`` override (e.g. a cached function), used instead of
            computing from ``rmf``.

        Returns
        -------
        float
            Combined half-angle field of view [rad].
        """
        idx = self._selected_baseline_indices(baseline_bounds, baseline_ids)
        if idx.size == 0:
            raise ValueError("no baselines selected for field_of_view")

        # choose the energy/energies at which to evaluate
        if energy_kev is None:
            energies = [None]
        elif np.ndim(energy_kev) == 0:
            energies = [float(energy_kev)]
        else:
            lo, hi = float(energy_kev[0]), float(energy_kev[1])
            energies = {"low_edge": [lo], "high_edge": [hi],
                        "min": [lo, hi]}.get(reduce, [lo])

        def R_at(E):
            if E is None:
                return None
            if resolving_power is not None:
                return float(resolving_power(E))
            if rmf is not None:
                from .reconstruct import resolving_power as _rp
                return float(_rp(rmf, E))
            return None

        vals = []
        for E in energies:
            R = R_at(E)
            per_bl = [self.baselines[int(i)].field_of_view(
                energy_kev=E, R=R, detector=self.detector, which=which,
                coherence_factor=coherence_factor) for i in idx]
            combined = min(per_bl) if combine == "min" else max(per_bl)
            vals.append(combined)
        # over the evaluated energies, take the tightest (min) as the safe FoV
        return float(min(vals))

    def geometric_fov(self, baseline_bounds=None, baseline_ids=None):
        """Per-baseline hard geometric FoV [rad] (used for vignetting), indexed
        by baseline id (np.inf for baselines not selected)."""
        fovs = np.full(len(self.baselines), np.inf)
        idx = self._selected_baseline_indices(baseline_bounds, baseline_ids)
        for i in idx:
            fovs[i] = self.baselines[int(i)].field_of_view(
                which="detection", detector=self.detector)
        return fovs
