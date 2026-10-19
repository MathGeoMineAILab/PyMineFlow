"""Slope definitions: validation, interpolation by azimuth, steep envelope and per-block discretization.

A slope definition is a number (constant slope) or a list of ``(azimuth, slope)`` pairs in degrees:
azimuth 0 = +y (north), 90 = +x (east), clockwise; slope measured from the horizontal, in (0, 90).
Between pairs the slope is interpolated linearly in azimuth, as in MineFlow.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "SlopeSpec",
    "discretize_slopes",
    "normalize_slopes",
    "slope_at_azimuth",
    "slopes_differ",
    "steep_envelope",
]

Sector = tuple[float, float]
SlopeSpec = float | Sequence[Sector]

_TAU = 2.0 * np.pi
_ROUNDING: dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
    "nearest": lambda t: np.floor(t + 0.5),  # halves go up
    "down": np.floor,
    "up": np.ceil,
}


def _wrap(azimuth: float) -> float:
    """Azimuth in [0, 360): ``-1e-14 % 360`` gives 360.0, the same direction as 0."""
    a = azimuth % 360.0
    return 0.0 if a >= 360.0 else a


def normalize_slopes(slopes: SlopeSpec) -> list[Sector]:
    """Validate a slope definition and return it as a list of ``(azimuth, slope)`` floats.

    Examples
    --------
    >>> normalize_slopes(45)
    [(0.0, 45.0)]
    >>> normalize_slopes([(0, 40), (180, 50)])
    [(0.0, 40.0), (180.0, 50.0)]
    """
    if isinstance(slopes, bool):
        raise TypeError("slopes must be a number or a list of (azimuth, slope) pairs, got a bool")
    try:
        scalar = np.ndim(slopes) == 0
        sectors = [(0.0, float(slopes))] if scalar else [(float(a), float(s)) for a, s in slopes]  # type: ignore[arg-type, union-attr]
    except (TypeError, ValueError):
        raise TypeError(
            f"slopes must be a number or a list of (azimuth, slope) pairs in degrees, got {slopes!r}"
        ) from None
    if not sectors:
        raise ValueError("a slope definition needs at least one (azimuth, slope) pair")
    for azimuth, slope in sectors:
        if not np.isfinite(azimuth) or not np.isfinite(slope) or not 0.0 < slope < 90.0:
            raise ValueError(f"slopes must be in (0, 90) degrees with finite azimuths, got ({azimuth}, {slope})")
    if len({round(_wrap(a), 12) % 360.0 for a, _ in sectors}) != len(sectors):
        raise ValueError(f"repeated azimuth (modulo 360) in {sectors}: use distinct azimuths")
    return sectors


def _sorted_radians(sectors: Sequence[Sector]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pairs = sorted((np.radians(_wrap(a)), np.radians(s)) for a, s in sectors)
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def slope_at_azimuth(azimuth_rad: ArrayLike, sectors: Sequence[Sector]) -> NDArray[np.float64]:
    """Slope (radians) at each azimuth (radians, any range), with circular linear interpolation."""
    a, s = _sorted_radians(sectors)
    az = np.mod(np.asarray(azimuth_rad, dtype=np.float64), _TAU)
    if a.size == 1:
        return np.full(az.shape, s[0])
    right = np.searchsorted(a, az, side="left") % a.size
    left = (right - 1) % a.size
    to_left = np.where(a[left] > az, _TAU - a[left] + az, az - a[left])
    to_right = np.where(a[right] < az, _TAU - az + a[right], a[right] - az)
    total = to_left + to_right
    weight = np.divide(to_left, total, out=np.zeros_like(total), where=total > 0)
    return s[left] + (s[right] - s[left]) * weight


def steep_envelope(domain_slopes: Sequence[SlopeSpec]) -> list[Sector]:
    """The steepest slope of all domains at every azimuth.

    Evaluated at the union of all domains' azimuths: on each interval the maximum of linear functions
    is convex, so the linear interpolation between those points lies above every domain. Its cone is
    contained in the cone of every domain.
    """
    lists = [normalize_slopes(s) for s in domain_slopes]
    azimuths = sorted({_wrap(a) for sectors in lists for a, _ in sectors})
    steepest = np.max([slope_at_azimuth(np.radians(azimuths), sectors) for sectors in lists], axis=0)
    return [(float(a), float(np.degrees(s))) for a, s in zip(azimuths, steepest, strict=True)]


def slopes_differ(domain_slopes: Sequence[SlopeSpec]) -> bool:
    """True when at least two domains have different slopes at some azimuth.

    The profiles are piecewise linear in azimuth with breaks at the sector azimuths, so comparing them
    at the union of all those azimuths is exact.
    """
    if len(domain_slopes) < 2:
        return False
    lists = [normalize_slopes(s) for s in domain_slopes]
    az = np.radians(sorted({_wrap(a) for sectors in lists for a, _ in sectors}))
    profiles = np.array([slope_at_azimuth(az, sectors) for sectors in lists])
    return bool(np.ptp(profiles, axis=0).max() > 1e-12)


def discretize_slopes(
    slopes: ArrayLike,
    azimuths: ArrayLike | None = None,
    step: float = 1.0,
    rounding: str = "nearest",
) -> tuple[NDArray[np.int32], list[list[Sector]]]:
    """Continuous per-block slopes -> (domains, domain_slopes).

    Parameters
    ----------
    slopes : ``(n,)`` one slope per block (same in every direction), or ``(n, k)`` one slope per
        azimuth in ``azimuths``. Degrees from the horizontal.
    azimuths : ``(k,)`` degrees; only with ``(n, k)`` slopes.
    step : resolution in degrees. Each distinct rounded row becomes one domain (one pattern).
    rounding : ``"nearest"`` (halves round up), ``"down"`` (flatter: conservative) or ``"up"``.

    Examples
    --------
    >>> domains, domain_slopes = discretize_slopes([44.6, 45.4, 40.2])
    >>> domains.tolist(), domain_slopes
    ([1, 1, 0], [[(0.0, 40.0)], [(0.0, 45.0)]])
    """
    if not step > 0:
        raise ValueError(f"step must be positive, got {step}")
    if rounding not in _ROUNDING:
        raise ValueError(f"rounding must be one of {sorted(_ROUNDING)}, got {rounding!r}")
    values, az = _slope_table(slopes, azimuths)
    ratio = values / step
    snapped = np.rint(ratio)
    ratio = np.where(np.abs(ratio - snapped) < 1e-9, snapped, ratio)  # 47.3 / 0.1 = 472.99999...
    steps = _ROUNDING[rounding](ratio).astype(np.int64)
    rounded = steps * step
    if (rounded <= 0).any() or (rounded >= 90).any():
        bad = float(values[(rounded <= 0) | (rounded >= 90)][0])
        raise ValueError(f"slope {bad} rounds outside (0, 90) degrees with step={step}: use a smaller step")
    unique, inverse = np.unique(steps, axis=0, return_inverse=True)
    domains = np.asarray(inverse).ravel().astype(np.int32)
    domain_slopes = [[(float(a), round(float(s * step), 10)) for a, s in zip(az, row, strict=True)] for row in unique]
    return domains, domain_slopes


def _slope_table(slopes: ArrayLike, azimuths: ArrayLike | None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(slopes, dtype=np.float64)
    if values.ndim == 1:
        if azimuths is not None:
            raise ValueError("azimuths are only used with (n, k) slopes; got (n,) slopes")
        values, az = values[:, None], np.array([0.0])
    elif values.ndim == 2:
        if azimuths is None:
            raise ValueError("(n, k) slopes need azimuths of length k")
        az = np.asarray(azimuths, dtype=np.float64).ravel()
        if az.size != values.shape[1]:
            raise ValueError(f"expected {values.shape[1]} azimuths (one per column), got {az.size}")
        if len({_wrap(float(a)) for a in az}) != az.size:
            raise ValueError(f"repeated azimuth (modulo 360) in {az.tolist()}")
    else:
        raise ValueError(f"slopes must have shape (n,) or (n, k), got {values.shape}")
    if values.shape[0] == 0 or not np.isfinite(values).all() or not np.isfinite(az).all():
        raise ValueError("slopes and azimuths must be non-empty and finite (no NaN)")
    return values, az
