"""High-level API: a block model given by rows -> ultimate pit."""

from __future__ import annotations

import itertools
import os
import threading
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._mineflow import Pit
from .grid import Grid, cells_to_rows, coords_to_grid, domains_to_cells, indices_to_grid, values_to_cells
from .slopes import (
    Sector,
    SlopeSpec,
    discretize_slopes,
    normalize_slopes,
    slopes_differ,
    steep_envelope,
)

__all__ = ["PitModel", "PitResult", "bench_discount_weights", "scale_values", "ultimate_pit"]

_SUM_LIMIT = 2**62  # margin below 2**63, checked in float64
_MAX_FULL_OFFSETS = 20_000  # useful (inside the grid) offsets per domain of the full pattern
_MAX_MINSEARCH_WINDOW = 2e9  # bytes of the search window MinSearch allocates
_MAX_CONE_POINTS = 200_000  # cone points on the top bench (cost of building the pattern)
_FACTORS = (10**6, 10**5, 10**4, 10**3, 10**2, 10, 1)  # scale_values: the largest one that fits
_TONNAGE_NAMES = ("TON", "TONS", "TONNES", "TONNAGE", "TONN", "TONNS", "TONELAJE")


def _check_sum(a: NDArray[np.float64]) -> None:
    if np.abs(a).sum(dtype=np.float64) >= _SUM_LIMIT:
        raise OverflowError("sum of |values| exceeds int64: use a smaller scale")


def bench_discount_weights(iz: ArrayLike, rate: float) -> NDArray[np.float64]:
    """Top-down bench discounting: the weight ``(1 + rate) ** -k`` of every block, where ``k`` is the
    number of benches between the top bench of the model (the highest block, ``k = 0``) and the bench
    of the block. ``iz`` are the bench indices of the blocks, growing upwards.

    Blocks deeper in the deposit are mined later, so their value counts less. With an annual rate and
    an average sinking rate, the rate per bench is ``annual_rate / benches_per_year`` (e.g. 10 % and
    8 benches a year: 0.0125). The whole value of a block is discounted, positive or negative.
    Because the weights are a power of one number, any other horizontal reference multiplies every
    value by the same constant: the pits are the same, only the reported values change.

    Examples
    --------
    >>> bench_discount_weights([5, 4, 3], 0.25).tolist()
    [1.0, 0.8, 0.64]
    """
    if isinstance(rate, (bool, np.bool_, str, bytes)):
        raise TypeError(f"bench_discount must be a number, got {rate!r}")
    try:
        r = float(rate)
    except (TypeError, ValueError):
        raise TypeError(f"bench_discount must be a number, got {rate!r}") from None
    if not np.isfinite(r) or r < 0:
        raise ValueError(f"bench_discount must be finite and >= 0, got {rate!r}")
    levels = np.asarray(iz)
    if levels.size == 0:
        return np.ones(0)
    k = (levels.max() - levels).astype(np.float64)
    return np.power(1.0 + r, -k)


def scale_values(values: ArrayLike) -> tuple[NDArray[np.int64], int]:
    """Real values -> (int64 values, scale factor).

    The factor is 1 when the values are already integers (within 1e-6); otherwise the largest power
    of ten up to 10**6 that keeps every value below 2**53 and the sum of |values| within int64. The pit
    value is returned in that scale: ``real_value = result.value / factor``.

    Examples
    --------
    >>> values, factor = scale_values([1.0, -2.0])
    >>> values.tolist(), factor
    ([1, -2], 1)
    >>> values, factor = scale_values([1.25, -0.0049])
    >>> values.tolist(), factor
    ([1250000, -4900], 1000000)
    """
    a = np.asarray(values)
    if a.ndim != 1:
        raise ValueError(f"values must be a 1-D array, got shape {a.shape}")
    if a.dtype.kind in "iu":
        _check_sum(a.astype(np.float64))
        if a.dtype.kind == "u" and a.size and a.max() > np.iinfo(np.int64).max:
            raise OverflowError("unsigned values exceed int64")
        return a.astype(np.int64), 1
    a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError("values contain NaN or infinite values")
    if np.allclose(a, np.rint(a), rtol=0, atol=1e-6):
        factor = 1
    else:
        largest = float(np.abs(a).max())
        total = float(np.abs(a).sum(dtype=np.float64))
        fits = [f for f in _FACTORS if largest * f <= 2**53 and total * f < _SUM_LIMIT]
        if not fits:
            raise ValueError("values too large to scale exactly: divide them by a power of ten first")
        factor = fits[0]
    scaled = np.rint(a * factor)
    if scaled.size and np.abs(scaled).max() > 2**53:
        raise ValueError("values exceed 2**53, beyond exact float64 integers: divide them by a power of ten first")
    _check_sum(scaled)
    return scaled.astype(np.int64), factor


def _integer_values(values: ArrayLike, n: int) -> NDArray[np.int64]:
    v = np.asarray(values)
    if v.dtype.kind not in "iu":
        raise TypeError(f"values must be integers (int64), got dtype {v.dtype}: scale them with scale_values()")
    if v.ndim != 1 or v.size != n:
        raise ValueError(f"values must be a 1-D array with one entry per block ({n}), got shape {v.shape}")
    if v.dtype.kind == "u" and v.size and v.max() > np.iinfo(np.int64).max:
        raise OverflowError("unsigned values exceed int64")
    return v.astype(np.int64, copy=False)  # the binding checks int64 overflow


@dataclass(frozen=True, slots=True)
class PitResult:
    """Result of a solve, with reports.

    Attributes
    ----------
    mask : bool per block, in the order of the input rows (True = inside the pit).
    value : pit value, in the integer scale of the solved values (see ``scale_values``).
    n_blocks : number of blocks inside the pit.
    seconds : solver time.
    grid : the regular grid built from the input.
    pattern : precedence pattern used: ``"1:5"``, ``"minsearch"``, ``"full"`` or ``"explicit"`` (arcs as given).
    n_benches : benches covered by the pattern (``None`` for ``"1:5"``).
    slopes : slope definition of each domain, as ``(azimuth, slope)`` pairs (``None`` for ``"1:5"``).
    values : the integer value of each block (input rows).
    factor : scale of the values: ``real_value = value / factor``.
    table : the source columns when the model came from ``solve`` (used by the reports).
    value_column : name of the value column in ``table``.
    destination, destinations : with ``economics``, the process of each block (index into
        ``destinations``; -1 = waste) and the process names.

    Methods
    -------
    summary() : a detailed text report. benches() : the pit bench by bench. to_csv(path) : the
    model with a ``PIT`` column (1 inside, 0 outside).
    """

    mask: NDArray[np.bool_]
    value: int
    n_blocks: int
    seconds: float
    grid: Grid
    pattern: str
    n_benches: int | None = None
    slopes: list[list[Sector]] | None = field(default=None, repr=False)
    values: NDArray[np.int64] | None = field(default=None, repr=False)
    factor: int = 1
    table: dict[str, NDArray[Any]] | None = field(default=None, repr=False)
    value_column: str | None = None
    destination: NDArray[np.int16] | None = field(default=None, repr=False)
    destinations: tuple[str, ...] | None = None
    tons: NDArray[np.float64] | None = field(default=None, repr=False)

    @property
    def real_value(self) -> float:
        """Pit value in the units of the original values (``value / factor``)."""
        return self.value / self.factor

    def with_source(self, table: dict[str, NDArray[Any]], factor: int, value_column: str) -> PitResult:
        """The same result, carrying its source table for the reports."""
        return replace(self, table=table, factor=int(factor), value_column=value_column)

    def _tonnage(self) -> tuple[str, NDArray[np.float64]] | None:
        """(column, tonnage of every block): that of the economics when there are economics; otherwise the one
        tonnage column of the table (none when there is none, or several: the reports then leave it out)."""
        if self.tons is not None:
            return "economics", self.tons
        if self.table is None:
            return None
        upper = {c.strip().upper(): c for c in self.table}
        found = [upper[n] for n in _TONNAGE_NAMES if n in upper]
        if len(found) != 1:
            return None
        name = found[0]
        tons = np.asarray(self.table[name])
        return (name, tons.astype(np.float64)) if tons.dtype.kind in "iuf" else None

    def benches(self) -> dict[str, NDArray[Any]]:
        """The pit bench by bench, from the top: level (0 = lowest level of the model), blocks,
        value (real units) and, when the model has a tonnage column, tonnage."""
        levels = np.asarray(self.grid.iz)[self.mask].astype(np.int64)
        order = np.arange(self.grid.nz - 1, -1, -1)
        counts = np.bincount(levels, minlength=self.grid.nz)
        keep = order[counts[order] > 0]
        out: dict[str, NDArray[Any]] = {"level": keep, "blocks": counts[keep]}
        if self.values is not None:
            value = np.bincount(levels, weights=self.values[self.mask].astype(np.float64), minlength=self.grid.nz)
            out["value"] = value[keep] / self.factor
        tonnage = self._tonnage()
        if tonnage is not None:
            tons = np.bincount(levels, weights=tonnage[1][self.mask], minlength=self.grid.nz)
            out["tonnage"] = tons[keep]
        return out

    def summary(self) -> str:
        """A detailed text report: the model, its economics, the slopes and the pit.

        Ore and waste are told apart by the sign of the block value (ore: value > 0, the blocks
        worth processing); the strip ratio is waste tonnage / ore tonnage.
        """
        from . import __version__

        g = self.grid
        n = int(self.mask.size)
        size = " x ".join(f"{s:g}" for s in g.block_size)
        tonnage = self._tonnage()
        out = [f"PyMineFlow {__version__} - ultimate pit", "=" * 60, "MODEL"]
        out.append(_row("blocks", f"{n:,}"))
        out.append(_row("grid", f"{g.nx} x {g.ny} x {g.nz} cells ({g.density:.0%} filled)"))
        out.append(_row("block size", size))
        if self.value_column is not None:
            scale = "" if self.factor == 1 else f" (scaled x{self.factor:,} to integers)"
            out.append(_row("value column", f"{self.value_column}{scale}"))
        if self.values is not None:
            out += _economics(self.values, None, tonnage, self.factor, self.destination, self.destinations)
        out += ["", "SLOPES"]
        if self.pattern == "explicit":
            out.append(_row("precedence", "as given (explicit arcs between blocks)"))
        elif self.slopes is None:
            out.append(_row("pattern", "1:5 (the block above and its 4 neighbours)"))
        else:
            for k, sectors in enumerate(self.slopes):
                label = "slope" if len(self.slopes) == 1 else f"domain {k}"
                text = (
                    f"{sectors[0][1]:g} deg"
                    if len(sectors) == 1
                    else ", ".join(f"az {a:g}: {s:g} deg" for a, s in sectors)
                )
                out.append(_row(label, text))
            height = " (full height)" if self.n_benches is not None and self.n_benches >= g.nz - 1 else ""
            pattern = "minimum" if self.pattern == "minsearch" else self.pattern
            out.append(_row("pattern", f"{pattern}, {self.n_benches} benches{height}"))
        out += ["", "ULTIMATE PIT"]
        out.append(_row("value", _number(self.real_value)))
        out.append(_row("blocks", f"{self.n_blocks:,} ({self.n_blocks / max(n, 1):.2%} of the model)"))
        if self.values is not None and self.n_blocks:
            out += _economics(self.values, self.mask, tonnage, self.factor, self.destination, self.destinations)
        if self.n_blocks:
            levels = np.asarray(g.iz)[self.mask]
            top, bottom = int(np.max(levels)), int(np.min(levels))
            out.append(_row("benches", f"{top - bottom + 1} (levels {top} down to {bottom}; 0 = lowest)"))
        out += ["", _row("solver time", f"{self.seconds:.3f} s"), "=" * 60]
        return "\n".join(out)

    def plot(self, **kwargs: Any) -> Any:
        """3-D view of the pit with PyVista; see ``pymineflow.plot_pit`` for the options."""
        from .plot import plot_pit

        return plot_pit(self, **kwargs)

    def to_csv(self, path: str | Path, column: str = "PIT", only_pit: bool = False) -> Path:
        """Write the model with a ``column`` of 1 (inside the pit) / 0; returns the path.
        ``only_pit=True`` writes only the blocks of the pit.

        Without a source table (``ultimate_pit``), the grid indices ``IX, IY, IZ`` (from 0) are written.
        """
        from .readers import write_table

        if self.table is not None:
            table = dict(self.table)
        else:
            table = {"IX": np.asarray(self.grid.ix), "IY": np.asarray(self.grid.iy), "IZ": np.asarray(self.grid.iz)}
        if column in table:
            raise ValueError(f"the table already has a column {column!r}: pass column=")
        table[column] = self.mask.astype(np.int8)
        if only_pit:
            table = {k: np.asarray(v)[self.mask] for k, v in table.items()}
        return write_table(table, path)


def _number(x: float) -> str:
    """Thousands separators; two decimals only when the number has them."""
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"


def _row(label: str, text: str) -> str:
    return f"  {label:<22}{text}"


def _economics(
    values: NDArray[np.int64],
    mask: NDArray[np.bool_] | None,
    tonnage: tuple[str, NDArray[np.float64]] | None,
    factor: int,
    destination: NDArray[np.int16] | None = None,
    destinations: tuple[str, ...] | None = None,
) -> list[str]:
    """Ore and waste: blocks, value and, with a tonnage column, tonnes and strip ratio.

    Ore is what goes to a process when the destinations are known (``economics``), otherwise the
    blocks of positive value.
    """
    v = values if mask is None else values[mask]
    if destination is not None:
        d = destination if mask is None else destination[mask]
        ore, waste, label = d >= 0, d < 0, ("ore (processed)", "waste")
    else:
        d = None
        ore, waste, label = v > 0, v <= 0, ("ore (value > 0)", "waste (value <= 0)")
    rows = [
        _row(label[0], f"{int(ore.sum()):,} blocks, worth {_number(int(v[ore].sum()) / factor)}"),
        _row(label[1], f"{int(waste.sum()):,} blocks, costing {_number(-int(v[waste].sum()) / factor)}"),
    ]
    if tonnage is not None:
        name, tons = tonnage
        t = tons if mask is None else tons[mask]
        ore_t, waste_t = float(t[ore].sum()), float(t[waste].sum())
        rows.append(_row(f"tonnage ({name})", f"{_number(ore_t + waste_t)}"))
        rows.append(_row("  ore", _number(ore_t)))
        if d is not None and destinations is not None and len(destinations) > 2:
            for k, process in enumerate(destinations[:-1]):
                rows.append(_row(f"    {process}", _number(float(t[d == k].sum()))))
        rows.append(_row("  waste", _number(waste_t)))
        ratio = f"{waste_t / ore_t:.4f}" if ore_t > 0 else "no ore"
        rows.append(_row("strip ratio", f"{ratio} (waste / ore tonnage)"))
    return rows


class PitModel:
    """Block model prepared ONCE (grid, slopes, precedences) and solved for one or more value vectors.

    Parameters
    ----------
    x, y, z : block-centre coordinates, one per row. Use either these or ``ix, iy, iz``.
    ix, iy, iz : integer block indices, one per row.
    block_size : (dx, dy, dz), always required.
    slopes : ``None`` -> classic 1:5 pattern; a number -> constant slope (degrees); a list of
        ``(azimuth, slope)`` pairs -> sectors (azimuth 0 = north/+y, clockwise).
    domains, domain_slopes : one integer per row plus one slope definition per domain.
    block_slopes, azimuths, slope_step : continuous slopes per row, ``(n,)`` or ``(n, k)`` with
        ``azimuths`` of length k, discretized every ``slope_step`` degrees (see ``discretize_slopes``).
    air_domain : domain of the empty cells, which always have precedences. ``"envelope"`` (default):
        the steepest slope of all domains; with ``n_benches`` >= model height every block gets exactly
        its own cone. ``"column"``: the domain of the column's highest block; conservative, it may
        require extra blocks at domain boundaries. An integer: that domain.
    full_pattern : ``None`` = automatic (True when domains have different slopes, since the minimum
        pattern loses constraints at their boundaries); forcing ``False`` then warns. ``True`` gives
        every block all the offsets of its own cone inside the model: exact, but slow on tall models.
    n_benches : benches covered by each block's precedence pattern. ``None`` (default) = the full model
        height, so the slope holds over the whole height by construction. An integer is used as is:
        fewer benches is faster, but above that height constraints only arrive in a chain and the pit
        may violate the slope. A value above the height equals ``None``.

    The model is solved on its own grid: no cells are added around it. With slopes that vary by
    azimuth, the minimum pattern reaches part of the cone through chains of offsets, and near the
    side edges of the model a chain can need cells outside it; then a few constraints may be lost
    there. ``full_pattern=True`` avoids it on models of moderate height.
    tol : coordinate tolerance as a fraction of the block size (coordinates only).
    z_up : ``False`` when z (or iz) grows downwards (depth).
    min_density : warn when the grid has fewer blocks per cell than this.

    Examples
    --------
    >>> import numpy as np
    >>> model = PitModel(ix=[0, 1, 1], iy=[0, 0, 0], iz=[0, 0, 1], block_size=(10, 10, 10), slopes=45)
    >>> model.solve(np.array([5, -1, -1])).value
    4
    """

    def __init__(
        self,
        *,
        x: ArrayLike | None = None,
        y: ArrayLike | None = None,
        z: ArrayLike | None = None,
        ix: ArrayLike | None = None,
        iy: ArrayLike | None = None,
        iz: ArrayLike | None = None,
        block_size: ArrayLike,
        slopes: SlopeSpec | None = None,
        domains: ArrayLike | None = None,
        domain_slopes: Sequence[SlopeSpec] | None = None,
        block_slopes: ArrayLike | None = None,
        azimuths: ArrayLike | None = None,
        slope_step: float = 1.0,
        air_domain: str | int = "envelope",
        full_pattern: bool | None = None,
        n_benches: int | None = None,
        tol: float = 1e-3,
        z_up: bool = True,
        min_density: float = 0.2,
    ) -> None:
        self.grid: Grid = _make_grid(x, y, z, ix, iy, iz, block_size, tol, z_up)
        _warn_if_sparse(self.grid, min_density)
        self.domains, self.domain_slopes, differ = _resolve_slopes(
            self.n_rows, slopes, domains, domain_slopes, block_slopes, azimuths, slope_step
        )
        self.n_benches: int = _resolve_n_benches(n_benches, self.grid.nz)
        self.full_pattern: bool = _resolve_full_pattern(full_pattern, self.domain_slopes, differ)
        if self.domain_slopes is not None:
            _check_pattern_size(self.grid, self.domain_slopes, self.n_benches, self.full_pattern)
        self.pattern: str = "1:5" if self.domain_slopes is None else "full" if self.full_pattern else "minsearch"
        self._air_domain = air_domain
        self._pit_kwargs = self._make_pit_kwargs(air_domain)
        self._pit: Pit | None = None

    @property
    def n_rows(self) -> int:
        """Number of blocks (input rows)."""
        return int(self.grid.linear.size)

    def _make_pit_kwargs(self, air_domain: str | int) -> dict[str, Any]:
        if isinstance(air_domain, str) and air_domain not in ("envelope", "column"):
            raise ValueError(f"air_domain must be 'envelope', 'column' or an integer, got {air_domain!r}")
        if self.domain_slopes is None:
            return {"pattern": "1:5"}
        kwargs: dict[str, Any] = {
            "pattern": "minsearch",
            "n_benches": self.n_benches,
            "block_size": self.grid.block_size,
            "full_pattern": self.full_pattern,
        }
        if self.domains is None:
            sectors = self.domain_slopes[0]
            if len(sectors) == 1:
                kwargs["slope_deg"] = sectors[0][1]
            else:
                kwargs["sectors"] = sectors
            return kwargs
        slope_lists = list(self.domain_slopes)
        if air_domain == "envelope":
            slope_lists.append(steep_envelope(slope_lists))  # extra domain, only for empty cells
            air_domain = len(slope_lists) - 1
        elif not isinstance(air_domain, str) and int(air_domain) >= len(slope_lists):
            raise ValueError(f"air_domain {air_domain} out of range [0, {len(slope_lists)})")
        kwargs["domains"] = domains_to_cells(self.grid, self.domains, air_domain)
        kwargs["domain_slopes"] = slope_lists
        return kwargs

    def _cell_domains(self) -> NDArray[np.int64] | None:
        """The slope domain of every cell, as given to ``Pit`` (``None`` with a single slope definition)."""
        domains: NDArray[np.int64] | None = self._make_pit_kwargs(self._air_domain).get("domains")
        return domains

    def solve(self, values: ArrayLike) -> PitResult:
        """Solve for these integer values (one per row).

        The first call builds the grid precedences; later calls reuse them and only replace the values
        (the solve itself starts from scratch).
        """
        v = _integer_values(values, self.n_rows)
        cell_mask, value, seconds = self._solve_cells(v)
        return self._result(v, cells_to_rows(self.grid, cell_mask).astype(bool, copy=False), int(value), seconds)

    def _solve_cells(self, v: NDArray[np.int64]) -> tuple[NDArray[np.bool_], int, float]:
        """Pit over every cell of the grid (empty cells included): (cell mask, value, seconds)."""
        cells = values_to_cells(self.grid, v)
        if self._pit is None:
            g = self.grid
            self._pit = Pit(g.nx, g.ny, g.nz, cells, **self._pit_kwargs)  # reads the values once
            self._pit_kwargs.pop("domains", None)
            del cells  # free before solving (peak memory)
            cell_mask, value, _, seconds = self._pit.solve()
        else:
            cell_mask, value, _, seconds = self._pit.solve_with_values(cells)
        return cell_mask, int(value), float(seconds)

    def _result(self, v: NDArray[np.int64], mask: NDArray[np.bool_], value: int, seconds: float) -> PitResult:
        n_benches = None if self.domain_slopes is None else self.n_benches
        return PitResult(
            mask, int(value), int(mask.sum()), float(seconds), self.grid, self.pattern, n_benches,
            self.domain_slopes, v.copy(),
        )  # fmt: skip

    def nested_pits(self, values_list: Iterable[ArrayLike]) -> list[PitResult]:
        """Solve several value scenarios (e.g. price or revenue factors). Results come in the same order.

        When the scenarios can be ordered so that no block loses value from one to the next (as with
        increasing price factors), their pits are nested and are solved with ``nested_pit_numbers``
        (bisection on rings), with the same result as solving each scenario on its own, much faster.
        Other scenarios are solved one by one. For hundreds of scenarios use ``nested_pit_numbers``,
        which keeps one array instead of one mask and one value copy per pit.
        """
        values = [_integer_values(v, self.n_rows) for v in values_list]
        order = _nesting_order(values)
        if order is None or len(values) < 3:
            return [self.solve(v) for v in values]

        def scenario(k: int, rows: NDArray[np.intp] | None) -> NDArray[np.int64]:
            v = values[order[k]]
            return v if rows is None else v[rows]

        numbers, pit_values, seconds = self.nested_pit_numbers(scenario, len(order))
        results: dict[int, PitResult] = {}
        for k, index in enumerate(order):
            mask = (numbers >= 1) & (numbers <= k + 1)
            results[index] = self._result(values[index], mask, pit_values[k], seconds[k])
        return [results[k] for k in range(len(values))]

    def nested_pit_numbers(
        self,
        scenario: Callable[[int, NDArray[np.intp] | None], ArrayLike],
        count: int,
        progress: Callable[[dict[str, Any]], None] | None = None,
        threads: int = 0,
    ) -> tuple[NDArray[np.int32], list[int], list[float]]:
        """Nested pits of ``count`` scenarios given in increasing order, with little memory.

        ``scenario(k, rows)`` returns the integer values of scenario ``k`` for the given row indices
        (every row when ``rows`` is ``None``). No block may lose value from scenario ``k`` to ``k + 1``
        (e.g. increasing price or revenue factors), so that the pits are nested. The largest pit is
        solved on the whole grid and every other one only on the ring between two pits already solved,
        by bisection; each scenario is evaluated once, on the rows of the outer pit that bounds it.

        Returns ``(pit_number, values, seconds)``: per row, the 1-based index of the smallest pit that
        contains it (0 outside every pit), so that pit ``k`` is ``(pit_number >= 1) & (pit_number <= k + 1)``;
        and per pit, its value and its solver time.

        ``progress``, if given, is called after every pit with a dict: ``order`` (1 for the first pit
        solved), ``pit`` (1-based), ``inner`` and ``outer`` (the pits that bound its ring, 0 = none),
        ``cells`` (cells solved), ``blocks``, ``value`` and ``seconds``.

        The two halves of every bisection are independent, so they are solved at the same time on
        ``threads`` threads (0 = every core, 1 = one after the other); the result does not depend on it.
        ``scenario`` and ``progress`` are then called from those threads, one at a time.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or threads < 0:
            raise ValueError(f"threads must be an integer >= 0, got {threads!r}")
        workers = int(threads) or (os.cpu_count() or 1)
        g = self.grid
        numbers = np.zeros(self.n_rows, dtype=np.int32)
        pit_values = [0] * count
        seconds = [0.0] * count
        row_of_cell = np.full(g.n_cells, -1, dtype=np.int64)
        row_of_cell[g.linear] = np.arange(self.n_rows)

        top = _integer_values(scenario(count - 1, None), self.n_rows)
        top_cells, top_value, top_seconds = self._solve_cells(top)
        top_rows = cells_to_rows(g, top_cells).astype(bool)
        numbers[top_rows] = count
        pit_values[-1], seconds[-1] = int(top_value), float(top_seconds)
        solved = [0]
        lock = threading.Lock()  # scenario and progress: one thread at a time

        def report(k: int, inner: int, outer: int, cells: int, blocks: int) -> None:
            with lock:
                solved[0] += 1
                if progress is not None:
                    progress(
                        {"order": solved[0], "pit": k + 1, "inner": inner, "outer": outer, "cells": cells,
                         "blocks": blocks, "value": pit_values[k], "seconds": seconds[k]}
                    )  # fmt: skip

        report(count - 1, 0, 0, g.n_cells, int(top_rows.sum()))
        del top
        assert self._pit is not None
        pit = self._pit

        def ring(
            k: int, a: int, b: int, inner: NDArray[np.bool_], outer: NDArray[np.bool_]
        ) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
            """Pit k on the cells of the outer pit that are not in the inner one: (its cells, the rows it adds)."""
            outer_cells = np.flatnonzero(outer)  # the inner pit is inside the outer one
            outer_rows = row_of_cell[outer_cells]
            occupied = outer_rows >= 0
            with lock:
                v = np.asarray(scenario(k, outer_rows[occupied]))
            if v.shape != (int(occupied.sum()),) or v.dtype.kind not in "iu":
                raise ValueError(f"scenario({k}, rows) must return one integer per row asked for")
            values = np.zeros(outer_cells.size, dtype=np.int64)  # this scenario, on the cells of the outer pit
            values[occupied] = v
            in_ring = ~inner[outer_cells]
            cells = outer_cells[in_ring]
            mask, _, _, solve_seconds = pit.solve_subset(cells, values[in_ring])
            pit_cells = inner.copy()
            pit_cells[cells[mask]] = True
            rows = outer_rows[in_ring]
            mined = mask & (rows >= 0)
            pit_values[k] = int(values[~in_ring].sum() + values[in_ring][mask].sum())
            seconds[k] = float(solve_seconds)
            report(k, a + 1, b + 1, int(cells.size), int(occupied[~in_ring].sum() + mined.sum()))
            return pit_cells, rows[mined]

        Task = tuple[int, int, NDArray[np.bool_], NDArray[np.bool_]]

        def between(a: int, b: int, inner: NDArray[np.bool_], outer: NDArray[np.bool_]) -> list[Task]:
            """Solve the middle scenario strictly between a and b (both solved or sentinels); return the two
            halves left, independent of each other."""
            if b - a < 2:
                return []
            k = (a + b) // 2
            middle, rows = ring(k, a, b, inner, outer)
            numbers[rows] = k + 1  # the smaller pits, solved later, overwrite it; the two halves write disjoint rows
            return [(a, k, inner, middle), (k, b, middle, outer)]

        first: Task = (-1, count - 1, np.zeros(g.n_cells, dtype=bool), top_cells)
        if workers == 1:
            stack = [first]
            while stack:
                stack.extend(reversed(between(*stack.pop())))
        else:
            from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

            # At most one task per thread, depth first (the most recent halves first, as the serial order): the
            # pits kept for the pending halves stay few, so the memory is about that of the serial bisection.
            pending: list[Task] = [first]
            running: set[Future[list[Task]]] = set()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                while pending or running:
                    while pending and len(running) < workers:
                        running.add(pool.submit(between, *pending.pop()))
                    done, running = wait(running, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.extend(reversed(future.result()))
        return numbers, pit_values, seconds


def _nesting_order(values: list[NDArray[np.int64]]) -> list[int] | None:
    """Scenario indices from the lowest to the highest values when every block's value never decreases
    along that order (then the pits are nested), else ``None``."""
    order = sorted(range(len(values)), key=lambda k: int(values[k].sum()))
    for a, b in itertools.pairwise(order):
        if not np.all(values[a] <= values[b]):
            return None
    return order


def _make_grid(
    x: ArrayLike | None,
    y: ArrayLike | None,
    z: ArrayLike | None,
    ix: ArrayLike | None,
    iy: ArrayLike | None,
    iz: ArrayLike | None,
    block_size: ArrayLike,
    tol: float,
    z_up: bool,
) -> Grid:
    coords, indices = (x, y, z), (ix, iy, iz)
    uses_coords = any(c is not None for c in coords)
    uses_indices = any(c is not None for c in indices)
    if uses_coords == uses_indices:
        raise ValueError("pass exactly one input mode: coordinates (x, y, z) or indices (ix, iy, iz)")
    chosen = coords if uses_coords else indices
    if any(c is None for c in chosen):
        raise ValueError("missing components: pass all three of (x, y, z) or (ix, iy, iz)")
    if block_size is None:
        raise ValueError("block_size=(dx, dy, dz) is required")
    if uses_coords:
        return coords_to_grid(x, y, z, block_size=block_size, tol=tol, z_up=z_up)
    depth = iz if z_up else -np.asarray(iz).astype(np.int64)  # unsigned indices would wrap
    return indices_to_grid(ix, iy, depth, block_size)


def _warn_if_sparse(grid: Grid, min_density: float) -> None:
    if grid.density < min_density:
        warnings.warn(
            f"sparse grid ({grid.density:.2f} blocks per cell, {grid.n_cells:,} cells for "
            f"{grid.linear.size:,} blocks): use ExplicitPit or crop the model to save memory",
            RuntimeWarning,
            stacklevel=4,
        )


def _resolve_slopes(
    n: int,
    slopes: SlopeSpec | None,
    domains: ArrayLike | None,
    domain_slopes: Sequence[SlopeSpec] | None,
    block_slopes: ArrayLike | None,
    azimuths: ArrayLike | None,
    slope_step: float,
) -> tuple[NDArray[np.integer] | None, list[list[Sector]] | None, bool]:
    """-> (per-row domains or None, slope list per domain or None, whether domains differ)."""
    if sum(v is not None for v in (slopes, domains, block_slopes)) > 1 or (
        domain_slopes is not None and domains is None
    ):
        raise ValueError("pass only one of: slopes, (domains + domain_slopes) or block_slopes")
    if block_slopes is not None:
        if np.asarray(block_slopes).shape[0] != n:
            raise ValueError(f"block_slopes needs one row per block ({n}), got {np.asarray(block_slopes).shape[0]}")
        domains, domain_slopes = discretize_slopes(block_slopes, azimuths, slope_step)
    if domains is not None:
        if domain_slopes is None:
            raise ValueError("domains needs domain_slopes (one slope definition per domain)")
        d = np.asarray(domains)
        if d.dtype.kind not in "iu" or d.shape != (n,):
            raise ValueError(f"domains must hold one integer per block ({n}), got dtype {d.dtype}, shape {d.shape}")
        lists = [normalize_slopes(s) for s in domain_slopes]
        if d.min() < 0 or d.max() >= len(lists):
            raise ValueError(
                f"domains must be in [0, {len(lists)}) (one domain_slopes entry each), got range [{d.min()}, {d.max()}]"
            )
        return d, lists, slopes_differ(lists)
    if slopes is not None:
        return None, [normalize_slopes(slopes)], False
    return None, None, False


def _resolve_n_benches(n_benches: int | None, nz: int) -> int:
    height = max(1, nz - 1)  # benches that can lie above a block
    if n_benches is None:
        return height
    if isinstance(n_benches, bool) or not isinstance(n_benches, (int, np.integer)) or n_benches < 1:
        raise ValueError(f"n_benches must be None or an integer >= 1, got {n_benches!r}")
    return min(int(n_benches), height)  # more benches than the height add nothing


def _resolve_full_pattern(full_pattern: bool | None, domain_slopes: list[list[Sector]] | None, differ: bool) -> bool:
    if domain_slopes is None:
        if full_pattern:
            raise ValueError("full_pattern needs slopes (the 1:5 pattern has no cone)")
        return False
    if full_pattern is None:
        return differ
    if differ and not full_pattern:
        warnings.warn(
            "domains with different slopes and full_pattern=False: the minimum pattern loses constraints "
            "at domain boundaries and the pit may be larger than valid",
            RuntimeWarning,
            stacklevel=4,
        )
    return bool(full_pattern)


def _check_pattern_size(grid: Grid, domain_slopes: list[list[Sector]], n_benches: int, full_pattern: bool) -> None:
    """Cheap checks that fail with a clear error before memory is allocated.

    MinSearch searches the pattern in a (2cx+1)(2cy+1)(n_benches+1)-byte window that does not depend
    on the grid (it grows with n_benches^3 for flat slopes and thin blocks). The full pattern has
    about sum_k (cone points inside the grid) offsets, and the solver keeps one pointer per arc of
    every visited node.
    """
    dx, dy, dz = grid.block_size
    flattest = min(s for sectors in domain_slopes for _, s in sectors)
    radius = n_benches * dz / np.tan(np.radians(flattest))
    window = (2 * np.ceil(radius / dx) + 1) * (2 * np.ceil(radius / dy) + 1) * (n_benches + 1)
    points = np.pi * radius * radius / (dx * dy)
    if window > _MAX_MINSEARCH_WINDOW or points > _MAX_CONE_POINTS:
        raise ValueError(
            f"n_benches={n_benches} with flattest slope {flattest} and blocks {grid.block_size}: the cone spans "
            f"~{points:,.0f} blocks per bench and building the pattern would need ~{window / 2**30:.1f} GB: "
            f"use a smaller n_benches"
        )
    if full_pattern:
        radii = np.arange(1, n_benches + 1) * dz / np.tan(np.radians(flattest))
        in_cone = np.pi * radii * radii / (dx * dy) + 2 * radii * (1 / dx + 1 / dy) + 1
        in_grid = (2 * grid.nx - 1) * (2 * grid.ny - 1)
        estimate = int(np.minimum(in_cone, in_grid).sum())
        if estimate > _MAX_FULL_OFFSETS:
            raise ValueError(
                f"full_pattern with n_benches={n_benches} gives ~{estimate:,} offsets per domain "
                f"(limit {_MAX_FULL_OFFSETS:,}): use a smaller n_benches (e.g. 8-24) or full_pattern=False"
            )


def ultimate_pit(
    values: ArrayLike,
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    z: ArrayLike | None = None,
    ix: ArrayLike | None = None,
    iy: ArrayLike | None = None,
    iz: ArrayLike | None = None,
    block_size: ArrayLike,
    slopes: SlopeSpec | None = None,
    domains: ArrayLike | None = None,
    domain_slopes: Sequence[SlopeSpec] | None = None,
    block_slopes: ArrayLike | None = None,
    azimuths: ArrayLike | None = None,
    slope_step: float = 1.0,
    air_domain: str | int = "envelope",
    full_pattern: bool | None = None,
    n_benches: int | None = None,
    tol: float = 1e-3,
    z_up: bool = True,
    min_density: float = 0.2,
) -> PitResult:
    """Ultimate pit of a block model given by rows. See ``PitModel`` for the parameters.

    ``values`` are integers per row (int64); scale real values first with ``scale_values``, the
    returned value is in that scale. The mask follows the order of the input rows.

    Examples
    --------
    >>> import numpy as np
    >>> r = ultimate_pit(np.array([5, -1, -1]), ix=[0, 1, 1], iy=[0, 0, 0], iz=[0, 0, 1],
    ...                  block_size=(10, 10, 10), slopes=45)
    >>> r.value, r.mask.tolist()
    (4, [True, False, True])
    """
    model = PitModel(
        x=x, y=y, z=z, ix=ix, iy=iy, iz=iz, block_size=block_size, slopes=slopes, domains=domains,
        domain_slopes=domain_slopes, block_slopes=block_slopes, azimuths=azimuths, slope_step=slope_step,
        air_domain=air_domain, full_pattern=full_pattern, n_benches=n_benches, tol=tol, z_up=z_up,
        min_density=min_density,
    )  # fmt: skip
    return model.solve(values)
