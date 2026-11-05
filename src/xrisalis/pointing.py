"""Observation pointing / attitude models.

The simulator consumes an attitude stream: a ``(T, 3)`` array of
``(pitch, yaw, roll)`` in **radians**, indexed by integer time-step (built by
:meth:`xrisalis.instrument.Interferometer.gen_pointing`). ``pitch`` and ``yaw``
are the boresight offsets along the two tangent-plane axes (the same axes as a
source's sky offset, i.e. ``pitch`` runs with declination and ``yaw`` with right
ascension); ``roll`` is the rotation about the boresight that the aperture
sweeps to build uv coverage.

Both are **tangent-plane offsets** [rad] from the observation's nominal
pointing ``(ra_pnt, dec_pnt)`` (set in
:func:`xrisalis.simulate.generate_photons`; see :mod:`xrisalis.coords`):
``pitch`` towards North, ``yaw`` towards East, true angles on the sky (no
``cos(dec)`` factor needed). ``(0, 0)`` stares at the nominal pointing; a dither
or drift is a small offset from it.

A :class:`Pointing` supplies the **boresight** ``(pitch, yaw)`` as a function of
time. Roll is filled in by the interferometer's ``roller`` (e.g.
:func:`~xrisalis.instrument.smooth_roll`) *unless* the :class:`Pointing` also
supplies a roll column (``provides_roll=True``), in which case it takes over the
roll and the roller is skipped.

Three ways to specify a pointing:

* :func:`constant_pointing` — a fixed stare (the default is on-axis, ``0, 0``).
* :func:`pointing_from_function` — ``f(t_seconds)`` returning the boresight
  (and optionally roll) over time; use for a smooth analytic drift/dither.
* :func:`pointing_from_file` — a tabulated timeline (FITS table or text),
  linearly interpolated onto the integer time-steps.

Angles are given as astropy :class:`~astropy.units.Quantity` objects, or as bare
numbers in **degrees** (the package's sky-coordinate convention); they are
converted to radians internally.
"""
from __future__ import annotations

import os
import numpy as np

from .units import to_rad


class Pointing:
    """Callable attitude source: ``t_seconds -> (T, 2|3)`` array [rad].

    Parameters
    ----------
    func : callable
        Maps a 1-D array of time-step times [s] to a ``(T, 2)`` boresight
        ``(pitch, yaw)`` or ``(T, 3)`` full ``(pitch, yaw, roll)`` array, in
        radians.
    provides_roll : bool
        If True, ``func`` returns a roll column and the interferometer's roller
        is bypassed. If False, only ``(pitch, yaw)`` are meaningful and roll is
        supplied by the roller.
    label : str
        Short human-readable description (used in ``repr``).
    """

    def __init__(self, func, provides_roll=False, label="pointing"):
        self._func = func
        self.provides_roll = bool(provides_roll)
        self.label = label

    def attitude(self, t):
        """Attitude ``(T, 2 or 3)`` [rad] at time-step times ``t`` [s]."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        arr = np.asarray(self._func(t), dtype=float)
        if arr.ndim == 1:                      # a single (2,)/(3,) triple
            arr = np.tile(arr.reshape(1, -1), (t.size, 1))
        if arr.shape[0] != t.size and arr.shape[1] == t.size:
            arr = arr.T                        # accept (ncol, T) too
        if arr.shape[0] != t.size:
            raise ValueError(
                f"pointing returned {arr.shape[0]} rows for {t.size} time-steps")
        if arr.shape[1] not in (2, 3):
            raise ValueError("pointing must yield 2 (pitch,yaw) or 3 "
                             "(pitch,yaw,roll) columns")
        if self.provides_roll and arr.shape[1] != 3:
            raise ValueError("provides_roll=True but no roll column produced")
        return arr

    __call__ = attitude

    def __repr__(self):
        roll = "+roll" if self.provides_roll else ""
        return f"Pointing({self.label}{roll})"


# --------------------------------------------------------------------------
# constructors
# --------------------------------------------------------------------------
def constant_pointing(pitch=0.0, yaw=0.0, roll=None):
    """A fixed stare: boresight held at ``(pitch, yaw)`` for the whole exposure.

    ``roll`` is optional; if given, this pointing also fixes the roll (which
    collapses uv coverage to a line — useful for a single-roll snapshot or a
    test, not for imaging). If ``roll`` is None (default) the interferometer's
    roller sweeps roll as usual. Angles are Quantities or bare degrees.
    """
    p, y = to_rad(pitch), to_rad(yaw)
    if roll is None:
        def f(t):
            return np.column_stack([np.full(t.shape, p), np.full(t.shape, y)])
        return Pointing(f, provides_roll=False, label="constant")
    r = to_rad(roll)

    def f(t):
        return np.column_stack([np.full(t.shape, p), np.full(t.shape, y),
                                np.full(t.shape, r)])
    return Pointing(f, provides_roll=True, label="constant")


def pointing_from_function(func, degrees=True, provides_roll=None):
    """Pointing from a user callable ``func(t_seconds)``.

    ``func`` receives a 1-D array of times [s] and must return either

    * a 2- or 3-tuple/list of components ``(pitch, yaw[, roll])`` — each a scalar
      (broadcast over time) or an array of length ``T``; or
    * an array shaped ``(T, 2)`` / ``(T, 3)`` (or its transpose).

    Values are interpreted in **degrees** by default (``degrees=False`` for
    radians). A 3-component return supplies roll and bypasses the roller.
    """
    conv = (np.pi / 180.0) if degrees else 1.0

    def f(t):
        out = func(t)
        if isinstance(out, (tuple, list)):
            cols = [np.broadcast_to(np.asarray(c, float), t.shape) for c in out]
            arr = np.column_stack(cols)
        else:
            arr = np.asarray(out, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[0] != t.size and arr.shape[1] == t.size:
                arr = arr.T
        return arr * conv

    if provides_roll is None:                  # probe the output width
        probe = f(np.zeros(1))
        provides_roll = (probe.shape[1] == 3)
    return Pointing(f, provides_roll=provides_roll, label="function")


def _read_pointing_table(path):
    """Return ``(columns_dict_upper, units_dict_upper)`` from a FITS or text table."""
    path = str(path)
    low = path.lower()
    if low.endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        from astropy.io import fits
        with fits.open(path) as hdul:
            hdu = None
            for name in ("POINTING", "ATTITUDE"):
                if name in hdul:
                    hdu = hdul[name]
                    break
            if hdu is None:
                hdu = next(h for h in hdul
                           if getattr(h, "data", None) is not None
                           and hasattr(h, "columns"))
            data = hdu.data
            cols = {c.upper(): np.asarray(data[c], dtype=float)
                    for c in data.columns.names}
            units = {c.upper(): (data.columns[c].unit or "")
                     for c in data.columns.names}
        return cols, units
    # text / CSV
    delim = "," if low.endswith(".csv") else None
    arr = np.genfromtxt(path, names=True, delimiter=delim, dtype=float)
    if arr.dtype.names is not None:
        cols = {n.upper(): np.asarray(arr[n], dtype=float) for n in arr.dtype.names}
    else:                                       # headerless -> positional columns
        raw = np.atleast_2d(np.genfromtxt(path, delimiter=delim, dtype=float))
        names = ["TIME", "PITCH", "YAW", "ROLL"][:raw.shape[1]]
        cols = {n: raw[:, i] for i, n in enumerate(names)}
    return cols, {}


def _pick(cols, names, required=True):
    for n in names:
        if n in cols:
            return n
    if required:
        raise ValueError(f"pointing table is missing one of columns {names}; "
                         f"found {list(cols)}")
    return None


def pointing_from_file(path, degrees=True):
    """Pointing from a tabulated timeline, interpolated onto the time-steps.

    Accepts a FITS binary table (a ``POINTING``/``ATTITUDE`` HDU, else the first
    table HDU) or a whitespace/CSV text file, with a time column and boresight
    columns (and an optional roll column):

    * time  : ``TIME`` (or ``T``), in seconds;
    * pitch : ``PITCH`` (or ``DEC``);  yaw : ``YAW`` (or ``RA``);
    * roll  : ``ROLL`` (optional; if present it supplies roll and bypasses the
      roller).

    Angles are interpreted in **degrees** by default. For a FITS table, a column
    ``TUNIT`` of ``rad``/``radian`` is honoured automatically. Times outside the
    tabulated range are clamped to the first/last sample.
    """
    cols, units = _read_pointing_table(path)
    tkey = _pick(cols, ["TIME", "T"])
    pkey = _pick(cols, ["PITCH", "DEC", "DELTA"])
    ykey = _pick(cols, ["YAW", "RA", "ALPHA"])
    rkey = _pick(cols, ["ROLL", "PA"], required=False)

    t = cols[tkey]
    order = np.argsort(t)
    t = t[order]

    def _conv(key):
        unit = str(units.get(key, "")).lower()
        c = 1.0 if unit in ("rad", "radian", "radians") else \
            (np.pi / 180.0 if degrees else 1.0)
        return cols[key][order] * c

    pitch, yaw = _conv(pkey), _conv(ykey)
    roll = _conv(rkey) if rkey else None

    def f(tt):
        p = np.interp(tt, t, pitch)
        y = np.interp(tt, t, yaw)
        if roll is not None:
            r = np.interp(tt, t, roll)
            return np.column_stack([p, y, r])
        return np.column_stack([p, y])

    label = f"file:{os.path.basename(str(path))}"
    return Pointing(f, provides_roll=bool(rkey), label=label)


def as_pointing(spec):
    """Coerce a user ``pointing`` spec into a :class:`Pointing` (or None).

    Accepts: None (no boresight control), a :class:`Pointing`, a callable
    (``-> pointing_from_function``), a path string/``os.PathLike``
    (``-> pointing_from_file``), or a 2-/3-element array-like constant boresight
    ``(pitch, yaw[, roll])`` (``-> constant_pointing``, bare numbers in degrees).
    """
    if spec is None or isinstance(spec, Pointing):
        return spec
    if isinstance(spec, (str, bytes, os.PathLike)):
        return pointing_from_file(spec)
    if callable(spec):
        return pointing_from_function(spec)
    arr = np.ravel(np.asarray(spec, dtype=float))
    if arr.size not in (2, 3):
        raise ValueError("constant pointing must have 2 (pitch,yaw) or 3 "
                         "(pitch,yaw,roll) values")
    return constant_pointing(*arr)


def attitude_index(times, time_step=1.0, n_rows=None):
    """Row of a ``time_step``-sampled attitude stream for each photon time.

    The attitude stream built by
    :meth:`xrisalis.instrument.Interferometer.gen_pointing` has one row every
    ``time_step`` seconds, so row ``k`` is the attitude at ``t = k*time_step``.
    Photon times are in **seconds**, hence the row is ``int(t / time_step)`` —
    *not* ``int(t)``, which silently compresses the whole timeline whenever
    ``time_step != 1``.

    Parameters
    ----------
    times : array-like        photon times [s]
    time_step : float         attitude sampling interval [s]
    n_rows : int, optional    clip the result to ``[0, n_rows - 1]``

    Returns
    -------
    (N,) int ndarray
    """
    ts = float(time_step)
    if not ts > 0:
        raise ValueError(f"time_step must be positive; got {time_step}")
    idx = np.floor(np.asarray(times, dtype=float) / ts).astype(int)
    if n_rows is not None:
        idx = np.clip(idx, 0, int(n_rows) - 1)
    else:
        idx = np.maximum(idx, 0)
    return idx
