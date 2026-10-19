"""Block coordinates or integer indices -> regular grid, robust to numerical noise.

Either the indices are correct and verified, or a clear error is raised. Assumes an axis-aligned
regular grid (no rotation, no sub-blocks).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Grid",
    "cells_to_rows",
    "coords_to_grid",
    "domains_to_cells",
    "empty_cells_below_blocks",
    "indices_to_grid",
    "values_to_cells",
]

BlockSize = tuple[float, float, float]
_MAX_CELLS = 2**62


@dataclass(frozen=True, slots=True)
class Grid:
    """Regular grid of a block model. Per-block arrays follow the order of the input rows.

    Attributes
    ----------
    ix, iy, iz : zero-based integer indices of each block.
    nx, ny, nz : grid dimensions (z grows upwards).
    block_size : (dx, dy, dz).
    origin : minimum block centre when built from coordinates, ``None`` when built from indices.
    base : value subtracted from the input indices (``(0, 0, 0)`` for coordinates).
    linear : linear cell index of each block, ``(iz*ny + iy)*nx + ix`` (int32 when it fits).
    n_cells : ``nx * ny * nz``.
    density : blocks per cell (close to 1 for a dense grid).
    """

    ix: NDArray[np.int32]
    iy: NDArray[np.int32]
    iz: NDArray[np.int32]
    nx: int
    ny: int
    nz: int
    block_size: BlockSize
    origin: tuple[float, float, float] | None
    base: tuple[int, int, int]
    linear: NDArray[np.integer]
    n_cells: int
    density: float


def _check_block_size(block_size: ArrayLike | None) -> BlockSize:
    size = np.asarray(block_size if block_size is not None else [], dtype=np.float64).ravel()
    if size.size == 1:  # one number: cubic blocks
        size = np.repeat(size, 3)
    if size.size != 3 or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ValueError(f"block_size must be three positive numbers (dx, dy, dz), got {block_size!r}")
    return (float(size[0]), float(size[1]), float(size[2]))


def _build(
    ix: NDArray[np.int64],
    iy: NDArray[np.int64],
    iz: NDArray[np.int64],
    block_size: BlockSize,
    origin: tuple[float, float, float] | None,
    base: tuple[int, int, int],
) -> Grid:
    nx, ny, nz = int(np.max(ix)) + 1, int(np.max(iy)) + 1, int(np.max(iz)) + 1
    n_cells = nx * ny * nz
    if n_cells >= _MAX_CELLS:
        raise ValueError(f"the blocks span a {nx} x {ny} x {nz} grid, too large for int64 (outlier indices?)")
    linear = (iz * ny + iy) * nx + ix
    linear = linear.astype(np.int32) if n_cells < 2**31 else linear
    _check_duplicates(linear, n_cells)
    return Grid(
        ix.astype(np.int32),
        iy.astype(np.int32),
        iz.astype(np.int32),
        nx,
        ny,
        nz,
        block_size,
        origin,
        base,
        linear,
        n_cells,
        ix.size / n_cells,
    )


_MAX_INDEX = 2.0**40  # far beyond any real model; keeps index arithmetic exact in int64


def _check_duplicates(linear: NDArray[np.integer], n_cells: int) -> None:
    n = linear.size
    if n_cells <= 4 * n:
        repeated = int(np.count_nonzero(np.bincount(linear, minlength=n_cells) > 1))
    else:
        repeated = n - np.unique(linear).size
    if repeated:
        raise ValueError(f"duplicate blocks: {repeated} cells hold more than one row")


def _estimate_step(c: NDArray[np.float64], decimals: int) -> float | None:
    """Block size along one axis: smallest gap between distinct coordinates, refined by the median."""
    unique = np.unique(np.round(c, decimals))
    if unique.size < 2:
        return None
    gaps = np.diff(unique)
    single = gaps[gaps < 1.5 * np.min(gaps)]  # one-step gaps (ignores jumps over empty cells)
    return float(np.median(single))


def _snap(c: NDArray[np.float64], step: float, tol: float, axis: str) -> NDArray[np.int64]:
    t = (c - np.min(c)) / step
    t = t - np.median(t - np.rint(t))  # robust origin: the minimum may be noisy, the median is not
    index = np.rint(t)
    if np.abs(index).max() > _MAX_INDEX:
        raise ValueError(
            f"axis {axis}: coordinates span more than {_MAX_INDEX:.0e} blocks; check for missing-data "
            f"sentinels (e.g. -999 or 1e30) or a wrong block size"
        )
    error = float(np.abs(t - index).max())
    if error > tol:
        raise ValueError(
            f"axis {axis}: coordinates do not lie on a regular grid with step {step:.6g} "
            f"(max deviation {error:.3g} blocks > tol={tol}); likely sub-blocks, a rotated model, "
            f"rounded coordinates or a wrong step. Pass block_size=(dx, dy, dz) or raise tol."
        )
    return index.astype(np.int64)


def coords_to_grid(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    block_size: ArrayLike | None = None,
    tol: float = 1e-3,
    decimals: int = 6,
    z_up: bool = True,
) -> Grid:
    """Grid from block-centre coordinates.

    Parameters
    ----------
    x, y, z : block centres, one per row.
    block_size : (dx, dy, dz); estimated from the data when omitted. Passing it is recommended: the
        estimate is the smallest gap between coordinates rounded to ``decimals``, so noisy coordinates
        need a smaller ``decimals``, and a rotated model may pass as a regular grid.
    tol : maximum distance, as a fraction of the block size, between a coordinate and its cell centre.
    decimals : rounding used only to estimate the block size.
    z_up : ``False`` when z grows downwards (depth).

    Examples
    --------
    >>> g = coords_to_grid([5.0, 15.0], [0.0, 0.0], [2.5, 2.5], block_size=(10, 10, 5))
    >>> (g.nx, g.ny, g.nz), g.ix.tolist()
    ((2, 1, 1), [0, 1])
    """
    if not 0 < tol < 0.5:
        raise ValueError(f"tol must be in (0, 0.5) block sizes, got {tol}")
    c = [np.asarray(v, dtype=np.float64).ravel() for v in (x, y, z)]
    n = c[0].size
    if n == 0 or not c[1].size == c[2].size == n:
        raise ValueError(f"x, y and z must have the same non-zero length, got {[v.size for v in c]}")
    if not all(np.isfinite(v).all() for v in c):
        raise ValueError("coordinates contain NaN or infinite values")
    if not z_up:
        c[2] = -c[2]
    if block_size is not None:
        size = _check_block_size(block_size)
    else:
        steps = [_estimate_step(v, decimals) for v in c]
        if any(s is None for s in steps):
            raise ValueError("an axis has a single distinct coordinate: pass block_size=(dx, dy, dz)")
        size = _check_block_size(steps)
    origin = (float(np.min(c[0])), float(np.min(c[1])), float(np.min(c[2])))
    ix, iy, iz = (_snap(v, s, tol, a) for v, s, a in zip(c, size, "xyz", strict=True))
    return _build(ix, iy, iz, size, origin, (0, 0, 0))


def _as_index(v: ArrayLike, name: str) -> NDArray[np.int64]:
    a = np.asarray(v)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {a.shape}")
    if a.dtype.kind == "f":
        if not np.isfinite(a).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        rounded = np.rint(a)
        if np.abs(a - rounded).max() > 1e-9:
            raise ValueError(f"{name} has non-integer values; for coordinates use coords_to_grid")
        a = rounded
    elif a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be numeric, got dtype {a.dtype}")
    if np.abs(a.astype(np.float64)).max() > _MAX_INDEX:
        raise ValueError(f"{name} has indices beyond {_MAX_INDEX:.0e}; check for missing-data sentinels")
    return a.astype(np.int64)


def indices_to_grid(ix: ArrayLike, iy: ArrayLike, iz: ArrayLike, block_size: ArrayLike) -> Grid:
    """Grid from integer indices (IX, IY, IZ) plus the block size.

    The block size is required (slopes need it). Indices may start at any value, even negative:
    the minima are subtracted and kept in ``Grid.base``.

    Examples
    --------
    >>> g = indices_to_grid([10, 11], [3, 3], [-2, -2], block_size=(10, 10, 5))
    >>> g.base, g.ix.tolist()
    ((10, 3, -2), [0, 1])
    """
    size = _check_block_size(block_size)
    a, b, c = _as_index(ix, "ix"), _as_index(iy, "iy"), _as_index(iz, "iz")
    if not a.size == b.size == c.size:
        raise ValueError(f"ix, iy and iz must have the same length, got {a.size}, {b.size}, {c.size}")
    base = (int(np.min(a)), int(np.min(b)), int(np.min(c)))
    return _build(a - base[0], b - base[1], c - base[2], size, None, base)


def values_to_cells(grid: Grid, values: ArrayLike) -> NDArray[np.int64]:
    """Per-row values -> per-cell values (empty cells are 0)."""
    cells = np.zeros(grid.n_cells, dtype=np.int64)
    cells[grid.linear] = values
    return cells


def cells_to_rows(grid: Grid, cell_array: ArrayLike) -> NDArray[np.generic]:
    """Per-cell array -> per-row array, in the order of the input rows."""
    return np.asarray(cell_array)[grid.linear]


def _fill_empty_columns(top: NDArray[np.int64]) -> NDArray[np.int64]:
    """Columns without blocks (-1) take the domain of the nearest column with blocks
    (growth in 8-neighbourhood layers, i.e. Chebyshev distance; ties broken in a fixed order)."""
    top = top.copy()
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while (top < 0).any():
        padded = np.pad(top, 1, constant_values=-1)
        grown = top.copy()
        for dy, dx in neighbours:
            v = padded[1 + dy : 1 + dy + top.shape[0], 1 + dx : 1 + dx + top.shape[1]]
            grown = np.where((grown < 0) & (v >= 0), v, grown)
        if (grown == top).all():
            raise ValueError("the grid has no column with blocks")
        top = grown
    return top


def domains_to_cells(grid: Grid, domains: ArrayLike, air_domain: str | int = "column") -> NDArray[np.int64]:
    """Per-row domains -> per-cell domains, including empty cells.

    Empty cells always get a domain (and therefore precedences): with a limited ``n_benches`` the
    constraints beyond that bench travel in a chain, and the chain passes through the air.

    Parameters
    ----------
    air_domain : ``"column"``: the air of a column takes the domain of its highest block, a hole takes
        the domain of the nearest block above it, and columns without blocks take the domain of the
        nearest column with blocks. An integer assigns that domain to every empty cell.
    """
    cells = np.full(grid.n_cells, -1, dtype=np.int64)
    cells[grid.linear] = np.asarray(domains, dtype=np.int64)
    if not isinstance(air_domain, str):
        if int(air_domain) < 0:
            raise ValueError(f"air_domain must be >= 0, got {air_domain}")
        cells[cells < 0] = int(air_domain)
        return cells
    if air_domain != "column":
        raise ValueError(f"air_domain must be 'column' or an integer here, got {air_domain!r}")
    view = cells.reshape(grid.nz, grid.ny, grid.nx)  # [z, y, x]
    occupied = view >= 0
    has_blocks = occupied.any(axis=0)
    z_top = grid.nz - 1 - np.argmax(occupied[::-1], axis=0)
    top = np.where(has_blocks, np.take_along_axis(view, z_top[None], axis=0)[0], -1)
    current = _fill_empty_columns(top)
    for z in range(grid.nz - 1, -1, -1):  # top to bottom
        current = np.where(view[z] >= 0, view[z], current)
        view[z] = current
    return cells


def empty_cells_below_blocks(grid: Grid) -> int:
    """Number of empty cells with at least one block above them in the same (x, y) column.

    These are internal holes, not air above the topography. They behave as zero-value blocks;
    the count is useful to review the block model.
    """
    occupied = np.zeros(grid.n_cells, dtype=bool)
    occupied[grid.linear] = True
    o = occupied.reshape(grid.nz, grid.ny, grid.nx)
    any_from_z = np.logical_or.accumulate(o[::-1], axis=0)[::-1]  # a block at z or above
    above = np.zeros_like(o)
    above[:-1] = any_from_z[1:]  # a block strictly above
    return int((~o & above).sum())
