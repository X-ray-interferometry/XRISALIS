"""Plotting for XRISALIS — kept separate from physics, I/O and products.

Every plotting function renders under the packaged Matplotlib style
(``xri_plot_white.mplstyle``, bundled in ``xrisalis/styles/``) so figures share
one look. Call :func:`set_style` once to make it the global default too; pass
``tex=False`` to use the bundled no-LaTeX variant on machines without a TeX
install.

Image plots default to the house colormap ``xri`` — perceptually uniform and
chroma-symmetric in CAM02-UCS (see :mod:`xrisalis.colormaps`). The previous
hand-picked map is still available as ``xri_legacy``, either per call
(``cmap="legacy"``) or globally via :func:`set_cmap`.

Functions take the restructured objects (or plain arrays) and an optional
``ax``; each returns the Matplotlib ``Axes`` (or ``Figure``) so plots compose.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from .colormaps import xri_cmap, xri_cmap_r, xri_legacy

# ---------------------------------------------------------------- style ------
_STYLE_DIR = os.path.join(os.path.dirname(__file__), "styles")
_STYLES = {
    "white": os.path.join(_STYLE_DIR, "xri_plot_white.mplstyle"),
    "white_notex": os.path.join(_STYLE_DIR, "xri_plot_white_notex.mplstyle"),
}
# module-level active style path (default = the real white style, usetex on)
_ACTIVE = _STYLES["white"]


def style_path(tex=True):
    """Path to the bundled style file (``tex=False`` -> no-LaTeX variant)."""
    return _STYLES["white" if tex else "white_notex"]


def set_style(tex=True, apply_global=True):
    """Select the packaged style for all XRISALIS plots.

    ``tex=True`` uses ``xri_plot_white.mplstyle`` (Computer Modern via LaTeX);
    ``tex=False`` uses the bundled no-LaTeX variant. With ``apply_global`` the
    style is also applied to the current Matplotlib session (``plt.style.use``),
    so your own plots pick it up too.
    """
    global _ACTIVE
    _ACTIVE = style_path(tex)
    if apply_global:
        plt.style.use(_ACTIVE)
    return _ACTIVE


def _ax(ax, **fig_kw):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(**fig_kw)
    return fig, ax


# ---------------------------------------------------------------- colormap ---
# The maps themselves live in :mod:`xrisalis.colormaps` and are re-exported
# here, so ``from xrisalis.viz import xri_cmap`` keeps working. ``xri_cmap`` is
# the default everywhere; the previous hand-picked map is kept as
# ``xri_legacy`` and can be selected per call (``cmap=...``) or globally
# (:func:`set_cmap`).
_CMAPS = {
    "xri": xri_cmap,
    "xri_r": xri_cmap_r,
    "legacy": xri_legacy,
    "xri_legacy": xri_legacy,
}
# module-level active colormap (default = the uniform, chroma-symmetric map)
_ACTIVE_CMAP = xri_cmap


def _resolve_cmap(cmap):
    """``None`` -> the active default; a name -> the named map; else itself."""
    if cmap is None:
        return _ACTIVE_CMAP
    if isinstance(cmap, str):
        return _CMAPS.get(cmap, cmap)  # fall through to Matplotlib
    return cmap


def set_cmap(cmap="xri"):
    """Select the colormap XRISALIS image plots use by default.

    Accepts ``"xri"`` (the perceptually uniform, chroma-symmetric house map;
    the default), ``"xri_r"``, ``"legacy"``/``"xri_legacy"`` (the previous
    hand-picked black-to-blue map), any Matplotlib colormap name, or a
    ``Colormap`` instance. Returns the resolved colormap.
    """
    global _ACTIVE_CMAP
    _ACTIVE_CMAP = plt.get_cmap(_resolve_cmap(cmap))
    return _ACTIVE_CMAP


def get_cmap():
    """The colormap image plots currently default to."""
    return _ACTIVE_CMAP


# ---------------------------------------------------------------- images -----
def plot_image(
    image,
    span_uas,
    ax=None,
    cmap=None,
    normalize=True,
    colorbar=False,
    title=None,
    markers=None,
):
    """Reconstructed image ``I(alpha_d, delta_d)`` with axes in micro-arcsec.

    ``cmap=None`` uses the active default (:func:`set_cmap`, initially the
    uniform, chroma-symmetric ``xri`` map); pass ``cmap="legacy"`` for the old
    one. ``markers`` is an optional list of (label, (N,2) positions_uas,
    style-dict).
    """
    with plt.style.context(_ACTIVE):
        fig, ax = _ax(ax, figsize=(4.4, 4))
        cmap = _resolve_cmap(cmap)
        img = np.asarray(image, float)
        if normalize:
            m = np.abs(img).max()
            img = img / m if m else img
        half = span_uas / 2.0
        im = ax.imshow(
            img,
            origin="lower",
            extent=[-half, half, -half, half],
            cmap=cmap,
            vmin=0,
            vmax=1 if normalize else None,
        )
        ax.set_xlabel(r"$\alpha_d$ ($\mu$as)")
        ax.set_ylabel(r"$\delta_d$ ($\mu$as)")
        if title:
            ax.set_title(title)
        for label, pos, style in markers or []:
            pos = np.atleast_2d(pos)
            ax.scatter(pos[:, 0], pos[:, 1], label=label, **style)
        if markers:
            ax.legend(fontsize=8, loc="upper right")
        if colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity (norm.)")
        return ax


def plot_evaluation(
    image, span_uas, result, ax=None, title="Image evaluation", cmap=None
):
    """Reconstructed image with recovered peaks (x) and, if known, truth (+)."""
    markers = [("recovered", result.peaks_uas, dict(marker="x", c="lime", s=90, lw=2))]
    ax = plot_image(image, span_uas, ax=ax, title=title, markers=markers, cmap=cmap)
    return ax


# ---------------------------------------------------------------- uv ---------
def plot_uv(uv, ax=None, lim_glambda=None, color="crimson"):
    """uv-plane sampling, axes in giga-lambda."""
    with plt.style.context(_ACTIVE):
        fig, ax = _ax(ax, figsize=(4.4, 4.4))
        p = np.asarray(uv) / 1e9
        ax.plot(p[:, 0], p[:, 1], ".", ms=1, color=color, alpha=0.5)
        if lim_glambda:
            ax.set_xlim(-lim_glambda, lim_glambda)
            ax.set_ylim(-lim_glambda, lim_glambda)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$u$ (G$\lambda$)")
        ax.set_ylabel(r"$v$ (G$\lambda$)")
        ax.set_title(r"$uv$-plane sampling")
        return ax


def plot_resolution_hist(uv, ax=None, bins=200, color="darkred"):
    """Distribution of interferometric resolution 1/(2|uv|) in micro-arcsec."""
    with plt.style.context(_ACTIVE):
        fig, ax = _ax(ax, figsize=(5.5, 4))
        uv_abs = np.hypot(np.asarray(uv)[:, 0], np.asarray(uv)[:, 1])
        uv_abs = uv_abs[uv_abs > 0]
        res_uas = (1.0 / (2 * uv_abs)) * (180 * 3600 / np.pi) * 1e6
        b = np.logspace(np.log10(res_uas.min()), np.log10(res_uas.max()), bins)
        ax.hist(res_uas, bins=b, color=color, alpha=0.8, histtype="stepfilled")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Interferometric resolution ($\mu$as)")
        ax.set_ylabel(r"$\#$ visibilities")
        return ax


def compare_image_uv(image, span_uas, uv, figsize=(10, 4)):
    """Side-by-side reconstructed image and uv-plane sampling."""
    with plt.style.context(_ACTIVE):
        fig, axs = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.15, 1]}
        )
        plot_image(
            image, span_uas, ax=axs[0], colorbar=True, title="Reconstructed image"
        )
        plot_uv(uv, ax=axs[1], lim_glambda=6)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------- spectra ----
def plot_spectrum(spec, ax=None, per_baseline=None, baselines=(0,), color="k"):
    """Spectrum as counts/s vs energy. ``spec`` is a products.Spectrum.

    ``per_baseline`` (a dict {id: Spectrum} from products.spectra_per_baseline)
    overplots the listed ``baselines``.
    """
    with plt.style.context(_ACTIVE):
        fig, ax = _ax(ax, figsize=(6, 4))
        y = spec.rate if spec.rate is not None else spec.counts
        x = spec.energy if spec.energy is not None else spec.channel
        ax.step(x, y, where="mid", color=color, lw=1.5, label="all")
        if per_baseline:
            for b in baselines:
                s = per_baseline[b]
                yb = s.rate if s.rate is not None else s.counts
                xb = s.energy if s.energy is not None else s.channel
                ax.step(xb, yb, where="mid", lw=1, alpha=0.7, label=f"BL{b}")
        ax.set_yscale("log")
        if spec.energy is not None:
            ax.set_xscale("log")
            ax.set_xlabel("Energy (keV)")
        else:
            ax.set_xlabel("Channel")
        ax.set_ylabel("cts/s" + ("/chan" if spec.rate is not None else ""))
        ax.legend(fontsize=8)
        return ax


def plot_lightcurve(lc, ax=None, color="darkred"):
    """Light curve (rate vs time with Poisson errors). ``lc`` is a LightCurve."""
    with plt.style.context(_ACTIVE):
        fig, ax = _ax(ax, figsize=(6, 4))
        ax.errorbar(lc.time, lc.rate, yerr=lc.error, fmt="o-", ms=3, color=color)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("cts/s")
        return ax


# ---------------------------------------------------------------- ARF --------
def plot_arf(arf, ax=None, baseline_mode=True, scale="log"):
    """Effective area vs energy for an io.ARF (total + per-baseline)."""
    with plt.style.context(_ACTIVE):
        fig, ax = _ax(ax, figsize=(7, 4))
        E = arf.energy_mid
        ax.plot(E, arf.total, color="mediumvioletred", lw=2, zorder=10, label="Total")
        if baseline_mode:
            cmap = plt.get_cmap("plasma")
            n = arf.n_baselines
            for i, (D, resp) in enumerate(arf.per_baseline):
                ax.plot(
                    E,
                    resp,
                    lw=1,
                    zorder=2,
                    color=cmap(i / max(n - 1, 1)),
                    label=rf"$D={D:.2f}\,$m",
                )
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_xlim(np.min(E), np.max(E))
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel(r"Effective area (cm$^2$)")
        ax.grid(which="both", ls="--", lw=0.5, alpha=0.5)
        ax.legend(fontsize=8, ncol=1, framealpha=1.0)
        return ax


# ---------------------------------------------------------------- color -----
def _as_cube(rgb):
    """(H, W, 3) float cube from an RGBImage or any 3-sequence of images.

    Also returns the span and the bands when they are known (an RGBImage
    carries them; a bare list of arrays does not).
    """
    channels = getattr(rgb, "channels", None)
    if channels is None:
        channels = list(rgb)
        span = None
        bands = None
    else:
        span = rgb.span_uas
        bands = rgb.bands
    if len(channels) != 3:
        raise ValueError(f"need exactly three channels (R, G, B); got {len(channels)}.")
    shapes = {np.shape(c) for c in channels}
    if len(shapes) != 1:
        raise ValueError(f"the three channels must share a shape; got {shapes}.")
    cube = np.stack([np.asarray(c, float) for c in channels], axis=-1)
    return cube, span, bands


def white_balance(cube, mode="total", arf=None, bands=None):
    """Per-channel *linear* gains — a deliberate choice of white point.

    A linear gain tints every pixel identically, so hue *differences* (the
    band ratios that carry the physics) survive it. A per-channel non-linear
    stretch would not: it makes hue depend on brightness. That is why the
    stretch in :func:`rgb_composite` is applied once, to all three channels
    together, and only the gains are per channel.

    Modes
    -----
    ``"total"``
        Equalize the integrated flux of the three bands, so the *field-average*
        spectrum comes out white and sources are colored by how they differ
        from it. This is the default, and it is only meaningful once a black
        point has removed each band's sidelobe floor — otherwise it equalizes
        noise, not signal.
    ``"peak"``
        Equalize the band peaks, which makes the brightest source white by
        construction. Useful when one source is the reference.
    ``"flux"``
        Divide by the ARF-integrated effective area of each band, so a flat
        photon spectrum is white. Needs ``arf`` and ``bands``; this is the only
        mode whose white point is instrumental rather than empirical.
    ``"none"``
        Unit gains.

    Whichever is used, say so in the caption: the white point is a choice, and
    the colors are not interpretable without it.
    """
    if mode == "none":
        return np.ones(3)
    if mode == "total":
        ref = np.array([cube[..., c].sum() for c in range(3)], float)
    elif mode == "peak":
        ref = np.array([cube[..., c].max() for c in range(3)], float)
    elif mode == "flux":
        if arf is None or bands is None:
            raise ValueError(
                'white_balance(mode="flux") needs both arf and '
                "bands (pass an RGBImage, which carries the "
                "bands, plus arf=...)."
            )
        g = []
        for lo, hi in bands:
            m = (arf.energy_mid >= lo) & (arf.energy_mid < hi)
            area = np.trapezoid(arf.total[m], arf.energy_mid[m]) if m.any() else 0.0
            g.append(1.0 / area if area > 0 else 0.0)
        g = np.asarray(g, float)
        return g / g.max() if g.max() > 0 else np.ones(3)
    else:
        raise ValueError(
            f"unknown white-balance mode {mode!r}; expected "
            f'"total", "peak", "flux" or "none".'
        )
    ref[ref <= 0] = 1.0
    return ref.max() / ref


def rgb_composite(
    rgb, stretch="asinh", a=0.2, percentile=100.0, gains="total", black=90.0, arf=None
):
    """Composite three band images into a displayable (H, W, 3) RGB array.

    ``rgb`` is an :class:`~xrisalis.reconstruct.RGBImage` (from
    :func:`~xrisalis.reconstruct.reconstruct_rgb`) or any sequence of three
    equally-shaped arrays, ordered R, G, B.

    The steps, in the order they have to happen:

    1. **Clip negatives.** Dirty-image sidelobes go negative, and a negative
       value is not a color.
    2. **Subtract a per-band black point** (``black``, a percentile). Clipping
       at zero leaves a *positive* floor whose level differs from band to band;
       the eye reads that as a color, and the stretch turns it into speckle.
       Subtracting a constant is an offset, not a stretch, so it leaves hue
       alone. Set it honestly — a high black point also hides real faint
       structure, so quote the value whenever you show the image.
    3. **Apply per-channel linear gains** (``gains``; see
       :func:`white_balance`).
    4. **Normalize once**, on the whole cube, to ``percentile`` (100 = the
       maximum).
    5. **Stretch once**, on the whole cube: ``"asinh"`` (soft parameter ``a``;
       smaller = more aggressive), ``"sqrt"`` or ``"linear"``.

    Steps 4 and 5 are deliberately *not* per channel: doing them per channel
    would make hue a function of brightness and destroy the band ratios.

    Returns an array in [0, 1], ready for ``imshow``.
    """
    cube, _, bands = _as_cube(rgb)
    cube = np.maximum(cube, 0.0)
    if black:
        for c in range(3):
            cube[..., c] = np.maximum(
                cube[..., c] - np.percentile(cube[..., c], black), 0.0
            )
    if isinstance(gains, str):
        g = white_balance(cube, gains, arf=arf, bands=bands)
    else:
        g = np.asarray(gains, float)
    cube = cube * g
    hi = cube.max() if percentile >= 100 else np.percentile(cube, percentile)
    x = cube / (hi if hi > 0 else 1.0)
    if stretch == "asinh":
        x = np.arcsinh(x / a) / np.arcsinh(1.0 / a)
    elif stretch == "sqrt":
        x = np.sqrt(np.clip(x, 0.0, None))
    elif stretch not in ("linear", None):
        raise ValueError(
            f'unknown stretch {stretch!r}; expected "asinh", "sqrt" or "linear".'
        )
    return np.clip(x, 0.0, 1.0)


def plot_rgb(rgb, span_uas=None, ax=None, title=None, markers=None, legend=True, **kw):
    """False-color image with axes in micro-arcsec and a band legend.

    ``rgb`` is an :class:`~xrisalis.reconstruct.RGBImage` or three arrays;
    remaining keywords go to :func:`rgb_composite` (``stretch``, ``a``,
    ``gains``, ``black``, ``percentile``, ``arf``).
    """
    cube, span, bands = _as_cube(rgb)
    span_uas = span if span_uas is None else span_uas
    if span_uas is None:
        raise ValueError(
            "pass span_uas — a bare list of arrays does not "
            "carry one (an RGBImage does)."
        )
    img = rgb_composite(rgb, **kw)
    with plt.style.context(_ACTIVE):
        fig, ax = _ax(ax, figsize=(4.6, 4.4))
        half = span_uas / 2.0
        ax.imshow(img, origin="lower", extent=[-half, half, -half, half])
        ax.set_xlabel(r"$\alpha_d$ ($\mu$as)")
        ax.set_ylabel(r"$\delta_d$ ($\mu$as)")
        if title:
            ax.set_title(title)
        for label, pos, style in markers or []:
            pos = np.atleast_2d(pos)
            ax.scatter(pos[:, 0], pos[:, 1], label=label, **style)
        if bands is not None and legend:
            from matplotlib.lines import Line2D

            handles = [
                Line2D(
                    [],
                    [],
                    marker="s",
                    ls="",
                    ms=8,
                    color=c,
                    label=rf"{lo:g}$-${hi:g} keV",
                )
                for c, (lo, hi) in zip("rgb", bands)
            ]
            handles += [
                Line2D(
                    [],
                    [],
                    marker=s.get("marker", "o"),
                    ls="",
                    color=s.get("c", "w"),
                    label=lab,
                )
                for lab, _, s in (markers or [])
            ]
            ax.legend(
                handles=handles,
                fontsize=8,
                loc="upper right",
                framealpha=0.35,
                labelcolor="w",
            )
        elif markers and legend:
            ax.legend(fontsize=8, loc="upper right")
        return ax


def plot_rgb_channels(rgb, figsize=(11, 3.4), cmap=None, **kw):
    """The three band images side by side, plus the composite — a diagnostic.

    Use it when a color looks wrong: it shows whether the cause is one band's
    calibration or the compositing.
    """
    cube, span, bands = _as_cube(rgb)
    with plt.style.context(_ACTIVE):
        fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
        names = ["R", "G", "B"]
        for c in range(3):
            lab = names[c]
            if bands is not None:
                lab += rf" ({bands[c][0]:g}$-${bands[c][1]:g} keV)"
            plot_image(cube[..., c], span, ax=axes[c], cmap=cmap, title=lab)
        plot_rgb(rgb, span_uas=span, ax=axes[3], title="composite", legend=False, **kw)
        for a in axes[1:]:
            a.set_ylabel("")
        return fig
