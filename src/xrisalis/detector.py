"""Detector model.

For now the :class:`Detector` carries only what the field-of-view derivation
needs — the pixel pitch and the physical extent along the fringe (DETY) axis —
plus a shape label. It is deliberately a small, explicit object so it can later
grow the rest of the detector physics without changing call sites:

* full pixel geometry (pitch, gaps, tiling) and detector shape;
* per-event **positional reconstruction via charge-cloud modeling** (splitting
  charge across neighboring pixels, then estimating the sub-pixel impact
  position) — the hook is :meth:`reconstruct_position`, not implemented yet;
* Nyquist checks that the pixel pitch samples the fringe spacing
  ``Δy_f = λ / beam_angle``.

Lengths are meters (the package's internal detector unit); astropy length
Quantities are accepted and converted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import astropy.units as u
import numpy as np


def _to_m(value):
    """Length -> meters. Quantity converted; bare number assumed meters."""
    if value is None:
        return None
    if isinstance(value, u.Quantity):
        return float(value.to_value(u.m))
    return float(value)


@dataclass
class Chip:
    """One detector chip on the focal plane.

    A chip is a rectangular pixel array placed at ``(cx_m, cy_m)`` (its center)
    in focal-plane meters. Focal-plane axes: ``x`` runs along the fringe (DETY)
    direction, ``y`` along the cross (DETZ) direction. Pixel indexing has its
    origin at the chip's lower-left corner (col along +x, row along +y).

    ``baseline_ids`` lists which physical baselines land on this chip (usually
    one), so photons can be attributed to a baseline from the chip they hit.
    """

    chip_id: int
    cx_m: float
    cy_m: float
    nx: int
    ny: int
    pixel_m: float
    baseline_ids: tuple = ()

    @property
    def width_m(self):
        return self.nx * self.pixel_m

    @property
    def height_m(self):
        return self.ny * self.pixel_m

    @property
    def x0_m(self):
        """Lower-left corner focal-plane x [m]."""
        return self.cx_m - 0.5 * self.width_m

    @property
    def y0_m(self):
        """Lower-left corner focal-plane y [m]."""
        return self.cy_m - 0.5 * self.height_m


@dataclass
class Detector:
    """A pixelated detector.

    Parameters
    ----------
    pixel_m : float
        Pixel pitch [m] (astropy length Quantity accepted). Reserved for future
        Nyquist / charge-cloud use; not required by the field-of-view geometry.
    half_extent_m : float, optional
        Half-size of the sensitive area along the fringe (DETY) axis [m]. When
        given, it caps the illuminated region used in the geometric field of
        view (the fringe packet must land on the physical detector). If None,
        the baseline's ``footprint`` sets the limit.
    half_extent_cross_m : float, optional
        Half-size along the cross (DETZ) axis [m]. Reserved (DETZ is 0 in v1).
    shape : str
        Detector shape label ('rect' for now).
    name : str
        Optional identifier.
    """

    pixel_m: float
    half_extent_m: float = None
    half_extent_cross_m: float = None
    shape: str = "rect"
    name: str = ""
    chips: list = None  # list[Chip]; a segmented multi-chip focal plane

    def __post_init__(self):
        self.pixel_m = _to_m(self.pixel_m)
        self.half_extent_m = _to_m(self.half_extent_m)
        self.half_extent_cross_m = _to_m(self.half_extent_cross_m)
        self._bl2chip = {}
        if self.chips:
            for c in self.chips:
                for b in c.baseline_ids:
                    self._bl2chip[int(b)] = c

    # -- multi-chip focal plane ----------------------------------------------
    @classmethod
    def tiled_layout(
        cls, instrument, pixel_m=1e-5, margin=1.3, gap_m=2e-3, axis="y", name=""
    ):
        """Build a default focal plane: one chip per baseline, tiled in a strip.

        Each baseline's pattern (fringe extent ``footprint`` along DETY, shadow
        extent ``Wz`` along DETZ) gets its own chip. Chips are uniform (sized to
        the largest baseline, times ``margin``) and tiled along the cross axis
        (``axis='y'``, the default) with ``gap_m`` between them, centered on the
        fringe axis. Baseline ``i`` -> chip ``i``. This is a convenience default;
        supply an explicit ``chips=[...]`` list for a real focal-plane layout.
        """
        pixel_m = _to_m(pixel_m)
        gap_m = _to_m(gap_m)
        bls = instrument.baselines
        # uniform chip size from the widest pattern (+ margin)
        wy = margin * max(b.footprint for b in bls)      # fringe (x) extent
        wz = margin * max(getattr(b, "Wz", 1e-2) for b in bls)  # cross (y) extent
        nx = int(math.ceil(wy / pixel_m))
        ny = int(math.ceil(wz / pixel_m))
        pitch = ny * pixel_m + gap_m  # chip-to-chip step
        n = len(bls)
        chips = []
        for i in range(n):
            if axis == "y":
                cx, cy = 0.0, (i - (n - 1) / 2.0) * pitch
            else:  # tile along x
                cx, cy = (i - (n - 1) / 2.0) * (nx * pixel_m + gap_m), 0.0
            chips.append(
                Chip(
                    chip_id=i,
                    cx_m=cx,
                    cy_m=cy,
                    nx=nx,
                    ny=ny,
                    pixel_m=pixel_m,
                    baseline_ids=(i,),
                )
            )
        return cls(pixel_m=pixel_m, chips=chips, name=name)

    def chip_for_baseline(self, baseline_id):
        return self._bl2chip.get(int(baseline_id))

    def place(self, baseline_id, dety, detz):
        """Map baseline-local ``(dety, detz)`` [m] onto the focal plane.

        Returns a dict of arrays: ``chip_id``; ``chipx``/``chipy`` (pixel col/row
        within the chip, origin at lower-left); ``chipx_m``/``chipy_m`` (meters
        from the chip corner); ``focx``/``focy`` (total focal-plane meters). The
        baseline's pattern is centered on its chip, so ``focx = chip.cx + dety``,
        ``focy = chip.cy + detz``. Photons on a baseline with no chip get
        ``chip_id = -1`` and NaN coordinates.
        """
        baseline_id = np.asarray(baseline_id)
        dety = np.asarray(dety, float)
        detz = np.asarray(detz, float)
        n = dety.size
        chip_id = np.full(n, -1, dtype=np.int32)
        focx = np.full(n, np.nan)
        focy = np.full(n, np.nan)
        chipx = np.full(n, np.nan)
        chipy = np.full(n, np.nan)
        chipx_m = np.full(n, np.nan)
        chipy_m = np.full(n, np.nan)
        if not self.chips:
            return dict(
                chip_id=chip_id,
                chipx=chipx,
                chipy=chipy,
                chipx_m=chipx_m,
                chipy_m=chipy_m,
                focx=focx,
                focy=focy,
            )
        for c in self.chips:
            sel = np.isin(baseline_id, np.asarray(c.baseline_ids, int))
            if not sel.any():
                continue
            fx = c.cx_m + dety[sel]
            fy = c.cy_m + detz[sel]
            chip_id[sel] = c.chip_id
            focx[sel] = fx
            focy[sel] = fy
            lx = fx - c.x0_m
            ly = fy - c.y0_m  # meters from chip corner
            chipx_m[sel] = lx
            chipy_m[sel] = ly
            chipx[sel] = lx / c.pixel_m  # pixel col/row
            chipy[sel] = ly / c.pixel_m
        return dict(
            chip_id=chip_id,
            chipx=chipx,
            chipy=chipy,
            chipx_m=chipx_m,
            chipy_m=chipy_m,
            focx=focx,
            focy=focy,
        )

    def fringe_half_extent_m(self):
        """Half-extent along the fringe axis [m], or None if unbounded."""
        return self.half_extent_m

    def fringe_spacing_m(self, wavelength_m, beam_angle):
        """Fringe spacing on the detector, ``Δy_f = λ / beam_angle`` [m]."""
        return float(wavelength_m) / float(beam_angle)

    def samples_fringe(self, wavelength_m, beam_angle, min_samples=2.0):
        """True if the pixel pitch Nyquist-samples the fringe (≥ ``min_samples``/period)."""
        dyf = self.fringe_spacing_m(wavelength_m, beam_angle)
        return (dyf / self.pixel_m) >= min_samples

    def reconstruct_position(self, *args, **kwargs):
        """Placeholder for charge-cloud sub-pixel position reconstruction.

        Not implemented yet — the detector currently reports the exact impact
        position (see :func:`xrisalis.simulate.simulate_positions`). This is the
        intended hook for splitting the charge cloud across pixels and
        estimating the reconstructed position (with its bias/blur).
        """
        raise NotImplementedError(
            "charge-cloud positional reconstruction is not implemented yet"
        )

    def __repr__(self):
        s = f"Detector(pixel={self.pixel_m * 1e6:.3g} um"
        if self.half_extent_m is not None:
            s += f", half_extent={self.half_extent_m * 1e3:.3g} mm"
        return s + f", shape={self.shape!r})"
