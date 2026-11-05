"""Instrument model: Baseline geometry and the Interferometer (baselines +
attitude/pointing model + field of view)."""

import math
from abc import ABC, abstractmethod

import numpy as np

from .config import KEV, C, H
from .pointing import as_pointing

class Baseline(ABC):
    """One baseline (pair of interfering beams) of an X-ray interferometer.

    Abstract base: every beam-combiner architecture is a subclass that decides
    where photons land on the detector (:meth:`sample_dety`) and how large the
    pattern is (:attr:`footprint`). The base fixes what reconstruction relies
    on — baseline ``D`` [m], beam angle ``beam_angle`` [rad], fringe spacing
    ``lambda/beam_angle`` and the fringe phase convention
    (:meth:`fringe_offset`) — so any subclass that honors it reconstructs
    unchanged. ``L`` [m] is the propagation distance that sets the geometric
    beam walk (``-L theta``) and the DETZ shadow; ``Wz`` [m] is the mirror
    aperture along the cross (DETZ) axis, which carries no interference.

    Subclasses: :class:`SlattedMirrorBaseline` (Willingale 2004),
    :class:`ArbitraryCombinerBaseline` (idealized beamsplitter),
    :class:`PellicleBaseline` (placeholder).
    """
    combiner = None          # short label of the architecture
    vignettes = True         # photons beyond the geometric FoV are dropped

    def __init__(self, D, beam_angle, L, Wz=1e-2):
        self.D, self.beam_angle, self.L, self.Wz = D, beam_angle, L, Wz

    # -- what a subclass must provide -------------------------------------------
    @property
    @abstractmethod
    def footprint(self):
        """Full DETY extent [m] of the pattern on the detector."""

    @abstractmethod
    def sample_dety(self, energies_j, thetas, rng):
        """DETY positions [m] and a per-photon patch index (slat, port, ...)
        for photons of energies ``energies_j`` [J] at fringe-axis off-axis
        angles ``thetas`` [rad]. Unplaceable photons get NaN / -1."""

    def _diffraction_fov(self, lam):
        """Diffraction-envelope FoV [rad]; none unless a subclass has one."""
        return np.inf

    # -- shared physics -----------------------------------------------------------
    def fringe_spacing(self, lam):
        """Fringe spacing ``dy = lambda / beam_angle`` [m]."""
        return lam / self.beam_angle

    def fringe_offset(self, theta):
        """DETY of the white-light fringe maximum for off-axis angle ``theta``:
        ``-D sin(theta) / (2 sin(beam_angle/2))`` (~ ``-D theta/beam_angle``)."""
        return -(self.D * np.sin(theta)) / (2 * np.sin(self.beam_angle / 2))

    def fresnel_number_z(self, energy_kev, distance=None):
        """Fresnel number ``(Wz/2)^2 / (lambda Z)`` of the cross-axis aperture
        (``Z`` defaults to ``L``): >> 1 geometric shadow, ~1 Fresnel."""
        lam = H * C / (float(energy_kev) * KEV)
        Z = self.L if distance is None else float(distance)
        return (self.Wz / 2.0) ** 2 / (lam * Z)

    def field_of_view(self, energy_kev=None, R=None, detector=None,
                      which="imaging", coherence_factor=1.0):
        """Half-angle field of view [rad] (see docs/fov_from_instrument.md).

        * **geometry** — the white-light fringe must land on the footprint
          (capped by the detector extent): ``half / (D/beta - L)``.
        * **detection** — geometry if this architecture vignettes
          (``vignettes``), else ``inf``; used by redistribute to drop photons.
        * **coherence** — chromatic decoherence: ``coherence_factor R lambda/D``
          (``R = E/sigma_E``, needs ``energy_kev`` and ``R``).
        * **diffraction** — single-aperture envelope, if the subclass has one.
        * **imaging** (default, 'auto') — min(geometry, coherence).
        * **cross** — non-interferometric DETZ axis: ``beta/2``.
        """
        if which == "auto":
            which = "imaging"
        half = self.footprint / 2.0
        if detector is not None and detector.fringe_half_extent_m() is not None:
            half = min(half, detector.fringe_half_extent_m())
        denom = self.D / self.beam_angle - self.L
        fov_geo = np.inf if denom <= 0 else half / denom
        fov_coh = fov_env = np.inf
        if energy_kev is not None:
            lam = H * C / (float(energy_kev) * KEV)
            fov_env = self._diffraction_fov(lam)
            if R is not None and np.isfinite(R) and R > 0:
                fov_coh = coherence_factor * R * lam / self.D
        table = {"geometry": fov_geo,
                 "detection": fov_geo if self.vignettes else np.inf,
                 "coherence": fov_coh, "diffraction": fov_env,
                 "imaging": min(fov_geo, fov_coh), "cross": self.beam_angle / 2.0}
        if which not in table:
            raise ValueError(f"unknown FoV kind {which!r}; use one of {list(table)}")
        return float(table[which])


class SlattedMirrorBaseline(Baseline):
    """Slatted-mirror baseline (Willingale 2004).

    ``num_pairs`` slats of width ``W`` at aperture pitch ``slat_pitch`` (>= W;
    default W, a contiguous comb) tile the detector into a comb of single-slit
    Fresnel envelopes, each filled by the global two-beam fringe of spacing
    ``lambda/beam_angle``; ``slat_pitch > W`` opens dark gaps. Sampled by
    :func:`xrisalis.simulate.sample_detector_positions`.

    Exactly one of each geometry pair must be supplied so the rest can be
    derived (D<->F, W<->L via beam_angle). Units are SI except ``W``, given in
    micrometers and stored in meters.
    """
    combiner = "slatted"

    def __init__(
        self,
        num_pairs,
        D=None,
        L=None,
        W=None,
        beam_angle=None,
        F=None,
        grazing_angle=None,
        bench_length=None,
        mirr_reflec=None,
        Wz=1e-2,
        slat_pitch=None,
        combiner="slatted",
        overlap_width=None,
        visibility=1.0,
    ):
        if beam_angle is None:
            if W is None or L is None:
                raise ValueError("Define either beam_angle, or both W (um) and L (m).")
            beam_angle = W * 1e-6 / L
        if D is None:
            if F is None:
                raise ValueError("Define either D or F.")
            D = F * beam_angle
        if W is not None:
            self.W = W * 1e-6
        elif L is not None:
            self.W = beam_angle * L
        else:
            raise ValueError("Define either W (um) or L (m).")
        super().__init__(D, beam_angle, L if L is not None else self.W / beam_angle, Wz)
        self.F = F if F is not None else D / beam_angle
        if bench_length is not None:
            self.bench_length = bench_length
        if grazing_angle is None and bench_length is not None:
            grazing_angle = np.arctan(D / 2 / bench_length) / 2
        self.grazing_angle = grazing_angle
        self.num_pairs = num_pairs
        self.mirr_reflec = mirr_reflec
        self.slat_pitch = self.W if slat_pitch is None else slat_pitch
        if self.slat_pitch < self.W - 1e-18:
            raise ValueError("slat_pitch must be >= slat width W (slats cannot "
                             "overlap); use slat_pitch=W for a contiguous comb.")
        # physical consistency checks (isclose guards against binary rounding)
        if not math.isclose(self.D, self.F * self.beam_angle):
            raise ValueError("D, F and beam_angle are inconsistent.")
        if not math.isclose(self.W, self.beam_angle * self.L):
            raise ValueError("W, L and beam_angle are inconsistent.")
        if not isinstance(num_pairs, (int, np.integer)) or num_pairs <= 0:
            raise ValueError("num_pairs must be a positive integer.")

    @classmethod
    def willingale(cls, D, Wz=1e-2, slat_pitch=None):
        """Preset from Willingale (2004): 30 pairs, L=10 m, W=300 um, bench 7 m."""
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

    @property
    def footprint(self):
        return self.W * self.num_pairs

    def sample_dety(self, energies_j, thetas, rng):
        from .simulate import sample_detector_positions
        return sample_detector_positions(self, energies_j, thetas, rng=rng,
                                         return_slat=True)

    def _diffraction_fov(self, lam):
        return lam / self.W

    def __repr__(self):
        return (f"SlattedMirrorBaseline(D={self.D:.4g} m, L={self.L:.4g} m, "
                f"W={self.W * 1e6:.4g} um, num_pairs={self.num_pairs})")


class ArbitraryCombinerBaseline(Baseline):
    """Idealized two-beam combiner (e.g. an AEI-type beamsplitter).

    Fringes of spacing ``lambda/beam_angle`` and contrast ``visibility`` (0-1),
    NO diffraction: the interferogram is a top-hat window of width
    ``overlap_width`` [m] (where the beams overlap), centered at ``-L theta``
    and cut off hard at its edges. Every photon lands inside the window — none
    are lost to diffraction wings or non-overlapping beam regions, and there
    is no fringe-axis vignetting. ``L`` defaults to ``overlap_width/beam_angle``.
    Sampled by :func:`xrisalis.simulate.sample_arbitrary_positions`.
    """
    combiner = "arbitrary"
    vignettes = False

    def __init__(self, D, beam_angle, overlap_width, visibility=1.0, Wz=1e-2,
                 L=None):
        if not overlap_width > 0:
            raise ValueError("overlap_width must be > 0 (meters)")
        if not 0.0 <= visibility <= 1.0:
            raise ValueError("visibility must be in [0, 1]")
        super().__init__(D, beam_angle,
                         overlap_width / beam_angle if L is None else L, Wz)
        self.overlap_width, self.visibility = float(overlap_width), float(visibility)

    @property
    def footprint(self):
        return self.overlap_width

    def sample_dety(self, energies_j, thetas, rng):
        from .simulate import sample_arbitrary_positions
        return sample_arbitrary_positions(self, energies_j, thetas, rng=rng,
                                          return_slat=True)

    def __repr__(self):
        return (f"ArbitraryCombinerBaseline(D={self.D:.4g} m, "
                f"beam_angle={self.beam_angle:.4g} rad, "
                f"overlap={self.overlap_width * 1e3:.4g} mm, V={self.visibility:.2f})")


class PellicleBaseline(Baseline):
    """PLACEHOLDER — pellicle (beamsplitter) combiner, e.g. the AEI X3 design.

    Not implemented yet; constructing it works, simulating with it raises
    NotImplementedError. Start from :class:`ArbitraryCombinerBaseline`, which
    already gives fringes in a hard-edged overlap window, and add e.g.:

    * two output ports (reflected / transmitted arm) with complementary
      fringes (phase shifted by pi); return the port as the patch index of
      :meth:`sample_dety` and give :attr:`footprint` the extent per port;
    * energy-dependent fringe visibility (pellicle figure, path matching);
    * per-port throughput (~30 % per arm for AEI). A throughput common to all
      photons belongs in the ARF rather than here.

    Keep the base-class conventions — fringe maxima at
    :meth:`fringe_offset`, spacing :meth:`fringe_spacing` — and reconstruction
    works unchanged. Extra design parameters are kept in ``self.parameters``.
    """
    combiner = "pellicle"

    def __init__(self, D, beam_angle, L, Wz=1e-2, **parameters):
        super().__init__(D, beam_angle, L, Wz)
        self.parameters = parameters

    @property
    def footprint(self):
        raise NotImplementedError("PellicleBaseline.footprint is a placeholder.")

    def sample_dety(self, energies_j, thetas, rng):
        raise NotImplementedError(
            "PellicleBaseline is a placeholder; implement sample_dety (see the "
            "class docstring) or use ArbitraryCombinerBaseline.")



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
    def __init__(
        self,
        time_step=1,
        roller=None,
        roll_init=0.0,
        wobble=None,
        wobble_sigma=0.0,
        fov=None,
        pointing=None,
        detector=None,
    ):
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
    def add_baseline(self, baseline=None, **kwargs):
        """Append a :class:`Baseline` instance, or build a
        :class:`SlattedMirrorBaseline` from ``kwargs``."""
        self.baselines.append(SlattedMirrorBaseline(**kwargs) if baseline is None
                              else baseline)
        return self

    def add_willingale_baseline(self, D):
        return self.add_baseline(SlattedMirrorBaseline.willingale(D))

    @classmethod
    def from_arf(cls, arf, roller=None, fov_arcsec=None, baseline=None, **kwargs):
        """Build an interferometer with one baseline per ARF baseline length.

        ``baseline`` is a callable ``D -> Baseline`` (default: the Willingale
        slatted mirror). E.g. an AEI-type idealized combiner::

            Interferometer.from_arf(arf, baseline=lambda D: ArbitraryCombinerBaseline(
                D, beam_angle=4.85e-6, overlap_width=0.026))
        """
        fov = (
            None
            if fov_arcsec is None
            else fov_arcsec * (3.141592653589793 / (180.0 * 3600.0))
        )
        inst = cls(roller=roller, fov=fov, **kwargs)
        make = SlattedMirrorBaseline.willingale if baseline is None else baseline
        for D in arf.baseline_lengths:
            inst.add_baseline(make(float(D)))
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
        Use :func:`~xrisalis.pointing.attitude_at` to get the attitude at photon
        times (linear interpolation between rows).

        The boresight (pitch, yaw) columns come from the pointing model
        (``pointing`` argument if given, else ``self.pointing``; see
        :mod:`xrisalis.pointing`). Roll is then swept by ``self.roller`` unless
        the pointing model itself supplies a roll column
        (``provides_roll=True``). Finally ``self.wobble`` is applied if set.
        With no pointing model the boresight stays at (0, 0): the nominal
        pointing of the photon list (see :mod:`xrisalis.coords`).
        """
        T = int(float(t_exp) / float(self.time_step)) + 2
        attitude = np.zeros((T, 3))
        source = self.pointing if pointing is None else as_pointing(pointing)
        provides_roll = False
        if source is not None:
            t = np.arange(T) * self.time_step
            att = source(t)
            attitude[:, : att.shape[1]] = att
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

    def field_of_view(
        self,
        energy_kev=None,
        rmf=None,
        baseline_bounds=None,
        baseline_ids=None,
        which="imaging",
        combine="min",
        reduce="low_edge",
        coherence_factor=1.0,
        resolving_power=None,
    ):
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
            energies = {"low_edge": [lo], "high_edge": [hi], "min": [lo, hi]}.get(
                reduce, [lo]
            )

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
            per_bl = [
                self.baselines[int(i)].field_of_view(
                    energy_kev=E,
                    R=R,
                    detector=self.detector,
                    which=which,
                    coherence_factor=coherence_factor,
                )
                for i in idx
            ]
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
                which="detection", detector=self.detector
            )
        return fovs