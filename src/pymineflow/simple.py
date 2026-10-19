"""The shortest path: a block model file (or table) -> ultimate pit or nested pits, in one call.

>>> pit = solve("model.csv", block_size=(10, 10, 5), slopes=45)   # doctest: +SKIP
>>> print(pit.summary())                                          # doctest: +SKIP
>>> pit.to_csv("pit.csv")                                         # doctest: +SKIP

The columns are found by their usual names (case does not matter): indices ``IX, IY, IZ`` or
``I, J, K``; block-centre coordinates ``X, Y, Z``, ``XC, YC, ZC``, ``XCENTRE, ...`` or
``EAST, NORTH, RL``; the value ``VALUE``, ``BLOCK_VALUE``, ``BVALUE``, ``VALOR`` or ``PROFIT``.
Pass ``value=``, ``indices=`` or ``coords=`` when the names differ, or ``economics=`` to compute
the block values from grades, prices and costs (see ``pymineflow.Economics``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._report import check_kind, describe, kinds, phase_report, report_text
from .compare import PitComparison, compare_nested
from .economics import Contaminant, Economics, Metal, Param, Process, economics_from_dict, read_economics
from .grid import coords_to_grid
from .model import _FACTORS, _SUM_LIMIT, PitModel, PitResult, _make_grid, bench_discount_weights, scale_values
from .npv import PitByPit, pit_by_pit
from .oes import OES, ultimate_pit_oes
from .readers import read_blocks
from .slopes import SlopeSpec

__all__ = ["NestedPits", "nested_pits", "solve", "value_blocks"]

_INDEX_NAMES = (("IX", "IY", "IZ"), ("I", "J", "K"))
_COORD_NAMES = (
    ("X", "Y", "Z"),
    ("XC", "YC", "ZC"),
    ("XCENTRE", "YCENTRE", "ZCENTRE"),
    ("XCENTER", "YCENTER", "ZCENTER"),
    ("EAST", "NORTH", "RL"),
    ("EASTING", "NORTHING", "ELEVATION"),
    ("ESTE", "NORTE", "COTA"),
)
_WHEN = {
    "end": "the end of each year",
    "mid": "mid-year",
    "start": "the start of each year",
    "continuous": "the moment each block is mined, exp(-rate x time)",
}


def _percent(rate: float) -> str:
    """A rate as a percentage without rounding it away: 0.075 -> '7.5 %'."""
    return f"{rate * 100:.6g} %"


def _capacities(cases: PitByPit) -> str:
    """The discount rate and the capacities of a pit-by-pit analysis, in words."""
    parts = [f"{_percent(cases.discount_rate)} a year"]
    if cases.processing_rate is not None:
        plant = cases.processing_rate
        parts.append(
            ", ".join(f"{k} {v:,.10g} t" for k, v in plant.items()) + " processed a year"
            if isinstance(plant, dict)
            else f"{plant:,.10g} t of ore a year"
        )
    if cases.mining_rate is not None:
        parts.append(f"{cases.mining_rate:,.10g} t mined a year")
    if cases.fixed_cost:
        parts.append(f"fixed cost {cases.fixed_cost:,.10g} a year")
    return ", ".join(parts)


_VALUE_NAMES = ("VALUE", "BLOCK_VALUE", "BLOCKVALUE", "BVALUE", "VALOR", "PROFIT")


def _find(table: dict[str, NDArray[Any]], names: Sequence[str]) -> str | None:
    """The column of ``table`` whose name matches one of ``names`` (case-insensitive)."""
    upper = {c.strip().upper(): c for c in table}
    return next((upper[n] for n in names if n in upper), None)


def _columns(table: dict[str, NDArray[Any]], wanted: Sequence[str], what: str) -> list[str]:
    missing = [c for c in wanted if c not in table]
    if missing:
        raise ValueError(f"{what} columns {missing} not found; available: {list(table)}")
    return list(wanted)


def _value_column(table: dict[str, NDArray[Any]], value: str | None) -> str:
    if value is None:
        upper = {str(c).strip().upper(): c for c in table}
        found = [upper[n] for n in _VALUE_NAMES if n in upper]
        if not found:
            raise ValueError(
                f"no value column found (looked for {list(_VALUE_NAMES)}): pass value= or economics=; "
                f"available: {list(table)}"
            )
        if len(found) > 1:
            raise ValueError(f"several value columns {found}: pass value= with the one to use")
        return str(found[0])
    if value not in table:
        raise ValueError(f"value column {value!r} not found; available: {list(table)}")
    return value


def _positions(
    table: dict[str, NDArray[Any]], indices: Sequence[str] | None, coords: Sequence[str] | None
) -> dict[str, Any]:
    """The position keyword arguments of ``PitModel``: ``ix, iy, iz`` or ``x, y, z``."""
    if indices is not None and coords is not None:
        raise ValueError("pass indices= or coords=, not both")
    if indices is not None:
        a, b, c = _columns(table, indices, "index")
        return {"ix": table[a], "iy": table[b], "iz": table[c]}
    if coords is not None:
        a, b, c = _columns(table, coords, "coordinate")
        return {"x": table[a], "y": table[b], "z": table[c]}
    for keys, candidates in ((("ix", "iy", "iz"), _INDEX_NAMES), (("x", "y", "z"), _COORD_NAMES)):
        for names in candidates:
            found = [_find(table, [n]) for n in names]
            if all(found):
                return {k: table[str(c)] for k, c in zip(keys, found, strict=True)}
    raise ValueError(
        f"no block positions found (indices IX, IY, IZ or coordinates X, Y, Z): pass indices= or coords=; "
        f"available: {list(table)}"
    )


def _check_value_options(economics: object, value: str | None) -> None:
    if economics is not None and value is not None:
        raise ValueError("pass value= (a value column) or economics= (values from grades and prices), not both")


def solve(
    source: str | Path | Any,
    *,
    block_size: ArrayLike | None = None,
    slopes: SlopeSpec | None = None,
    value: str | None = None,
    economics: Economics | Mapping[str, Any] | str | Path | None = None,
    price_factor: float = 1.0,
    indices: Sequence[str] | None = None,
    coords: Sequence[str] | None = None,
    n_benches: int | None = None,
    z_up: bool = True,
    bench_discount: float = 0.0,
    precedence: str | Path | tuple[ArrayLike, ArrayLike] | None = None,
    **read_options: Any,
) -> PitResult:
    """Ultimate pit of a block model file or table, in one call.

    Parameters
    ----------
    source : a file (CSV or any delimited text, Parquet, Feather; see ``read_blocks``) or a table
        (pandas/polars DataFrame, pyarrow Table, dict of columns).
    block_size : (dx, dy, dz), or one number for cubic blocks.
    slopes : slope in degrees from the horizontal (e.g. ``45``), or ``[(azimuth, slope), ...]``
        sectors with azimuth 0 = north, clockwise.
    value : value column; found by name when ``None``. Real numbers are scaled automatically.
    economics : instead of a value column, compute the values from grades, prices and costs: a
        ``pymineflow.Economics``, a dict with its fields or a JSON file; ``price_factor`` multiplies
        every price. ``value_blocks`` adds such values to a table in one call.
    indices, coords : the three position columns, found by name when ``None``.
    n_benches : benches of the precedence pattern; ``None`` = the full height (exact slope).
    z_up : ``False`` when z grows downwards (depth).
    bench_discount : top-down bench discounting, rate per bench (e.g. 0.0125 = 10 % a year over 8
        benches a year): every block value is multiplied by ``(1 + rate) ** -k``, ``k`` benches below
        the top bench (see ``bench_discount_weights``). The pit maximises the discounted value.
    precedence : instead of ``slopes``, the precedences as given: a MineLib ``.prec`` file or arcs
        ``(src, dst)`` (block ``src[k]`` requires ``dst[k]``). Blocks are identified by the ``ID`` column when
        the table has one, by row (from 0) otherwise. ``block_size`` is then optional: with coordinates it is
        the spacing of the block centres, with indices 1 x 1 x 1 (it sets the benches of ``bench_discount``).
    read_options : passed to ``read_blocks`` (``sep``, ``header``, ``decimal``, ``encoding``, ``engine``).

    Returns
    -------
    PitResult with ``mask`` (one bool per row), ``real_value``, ``summary()``, ``benches()``,
    ``to_csv(path)`` and ``plot()``.

    Examples
    --------
    >>> table = {"IX": [0, 1, 1], "IY": [0, 0, 0], "IZ": [0, 0, 1], "VALUE": [5.5, -1.0, -1.25]}
    >>> pit = solve(table, block_size=(10, 10, 10), slopes=45)
    >>> pit.mask.tolist(), pit.real_value
    ([True, False, True], 4.25)
    """
    _check_value_options(economics, value)
    if economics is None and price_factor != 1.0:
        raise ValueError("price_factor multiplies the prices of economics=: without economics, scale the values")
    if precedence is None and (block_size is None or slopes is None):
        raise ValueError("give block_size and slopes, or precedence= (a MineLib .prec file or arcs (src, dst))")
    if precedence is not None and (slopes is not None or n_benches is not None):
        raise ValueError("precedence= replaces slopes and n_benches: give one or the other")
    economics = _as_economics(economics)
    table = read_blocks(source, **read_options)
    where = _positions(table, indices, coords)
    destination = None
    if economics is None:
        value_column = _value_column(table, value)
        real = np.asarray(table[value_column])
    else:
        real, destination = economics.block_values(table, price_factor=price_factor)
        value_column = "value from economics"
    if precedence is None:
        model = PitModel(block_size=block_size, slopes=slopes, n_benches=n_benches, z_up=z_up, **where)
        grid = model.grid
    else:
        if block_size is None and "x" in where:  # coordinates: the block size is the spacing of the centres
            grid = coords_to_grid(where["x"], where["y"], where["z"], block_size=None, tol=1e-3, z_up=z_up)
        else:  # indices: unit blocks unless given (the size only matters for the views and the benches)
            grid = _make_grid(block_size=(1.0, 1.0, 1.0) if block_size is None else block_size, z_up=z_up,
                              tol=1e-3, **{k: where.get(k) for k in ("x", "y", "z", "ix", "iy", "iz")})  # fmt: skip
    weights = bench_discount_weights(grid.iz, bench_discount)
    if bench_discount:
        real = np.asarray(real, dtype=np.float64) * weights
        value_column = f"{value_column}, discounted {bench_discount:.4%} per bench"
    values, factor = scale_values(real)
    result = model.solve(values) if precedence is None else _solve_explicit(table, grid, values, precedence)
    result = result.with_source(table, factor, value_column)
    if economics is not None:
        result = replace(result, destination=destination, destinations=tuple(economics.destination_names()),
                         tons=economics.tonnages(table))  # fmt: skip
    return result


def _solve_explicit(
    table: dict[str, NDArray[Any]],
    grid: Any,
    values: NDArray[np.int64],
    precedence: str | Path | tuple[ArrayLike, ArrayLike],
) -> PitResult:
    """The pit with the precedences as given (arcs between block ids or rows)."""
    from ._mineflow import ExplicitPit
    from .readers import read_minelib_precedence

    n = values.size
    if isinstance(precedence, (str, Path)):
        src, dst = read_minelib_precedence(precedence)
        where = Path(precedence).name
    else:
        if not isinstance(precedence, (tuple, list)) or len(precedence) != 2:
            raise ValueError("precedence must be a .prec file or a pair of arrays (src, dst)")
        src, dst = (np.asarray(a) for a in precedence)
        where = "precedence"
        if (
            src.shape != dst.shape
            or src.ndim != 1
            or (src.size and (src.dtype.kind not in "iu" or dst.dtype.kind not in "iu"))
        ):
            raise ValueError("precedence arcs (src, dst) must be two 1-D integer arrays of the same length")
        src, dst = src.astype(np.int64), dst.astype(np.int64)
    ids_column = next((c for c in table if c.upper() in ("ID", "BLOCK_ID", "BLOCKID")), None)
    if ids_column is None:
        rows_src, rows_dst = src, dst
        both = np.concatenate([src, dst])
        if both.size and (int(np.min(both)) < 0 or int(np.max(both)) >= n):
            raise ValueError(f"{where}: block numbers must be rows from 0 to {n - 1} (or give an ID column)")
    else:
        ids = np.asarray(table[ids_column])
        if ids.dtype.kind not in "iu" or np.unique(ids).size != n:
            raise ValueError(f"column {ids_column!r} must hold one distinct integer id per block")
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]

        def rows(a: NDArray[np.int64]) -> NDArray[np.int64]:
            at = np.minimum(np.searchsorted(sorted_ids, a), n - 1)
            missing = sorted_ids[at] != a
            if missing.any():
                raise ValueError(f"{where}: block id {int(a[np.argmax(missing)])} is not in column {ids_column!r}")
            return order[at].astype(np.int64)

        rows_src, rows_dst = rows(src), rows(dst)
    mask, value, n_blocks, seconds = ExplicitPit(n, rows_src, rows_dst, values).solve()
    return PitResult(np.asarray(mask, dtype=bool), int(value), int(n_blocks), float(seconds), grid, "explicit",
                     None, None, values.copy())  # fmt: skip


@dataclass(frozen=True)
class NestedPits:
    """Nested pits for increasing factors (see ``nested_pits``); light in memory.

    The pits are kept as one array: ``pit_number()`` gives, per block, the smallest pit that contains
    it. ``pits[k]`` (a ``PitResult``) is rebuilt on demand.
    """

    factors: list[float]
    scale: str
    numbers: NDArray[np.int32] = field(repr=False)
    values: list[float]
    seconds: list[float]
    model: PitModel = field(repr=False)
    table: dict[str, NDArray[Any]] = field(repr=False)
    factor: int
    value_column: str
    scenario: Callable[[int, NDArray[np.intp] | None], NDArray[np.int64]] = field(repr=False)
    destination: Callable[[int, NDArray[np.intp]], NDArray[np.int16]] | None = field(default=None, repr=False)
    destinations: tuple[str, ...] | None = None
    base: Callable[[], NDArray[np.float64]] | None = field(default=None, repr=False)
    economics: Economics | None = field(default=None, repr=False)
    bench_discount: float = 0.0
    discount_rate: float = 0.0
    benches_per_year: float | None = None
    ore_per_year: float | None = None

    def __len__(self) -> int:
        return len(self.factors)

    def __getitem__(self, k: int) -> PitResult:
        k = range(len(self))[k]
        mask = (self.numbers >= 1) & (self.numbers <= k + 1)
        v = self.scenario(k, None)
        result = self.model._result(v, mask, int(v[mask].sum()), self.seconds[k])
        result = result.with_source(self.table, self.factor, self.value_column)
        if self.destination is not None:
            rows = np.arange(mask.size)
            result = replace(result, destination=self.destination(k, rows), destinations=self.destinations)
        if self.economics is not None:
            result = replace(result, tons=self._economics_tons)
        return result

    @cached_property
    def _economics_tons(self) -> NDArray[np.float64]:
        assert self.economics is not None
        return self.economics.tonnages(self.table)

    def __iter__(self) -> Iterator[PitResult]:
        return (self[k] for k in range(len(self)))

    def first_factor(self) -> NDArray[np.float64]:
        """Per block: the factor of the smallest pit that contains it, 0 outside all."""
        factors = np.asarray(self.factors, dtype=np.float64)
        return np.where(self.numbers > 0, factors[np.maximum(self.numbers, 1) - 1], 0.0)

    def compare(self, reference: ArrayLike, **kwargs: Any) -> PitComparison:
        """These pits against a reference, block by block and pit by pit: ``reference`` gives, per block,
        the factor of the first reference pit that holds it (0 = none), e.g. ``read_reference_pits``. See
        ``pymineflow.compare.compare_nested``."""
        return compare_nested(self, reference, **kwargs)

    def pit_number(self) -> NDArray[np.int32]:
        """Per block: 1 for the smallest pit that contains it, 2 for the next one, ..., 0 outside all."""
        return self.numbers.copy()

    def table_by_pit(self) -> dict[str, NDArray[Any]]:
        """One row per pit: factor, blocks, new blocks, value (at its factor and at factor 1) and, with a
        tonnage column, tonnage, ore and waste tonnage (ore: processed or of positive value) and strip ratio."""
        n = len(self)
        count = np.bincount(self.numbers, minlength=n + 1)[1:]
        blocks = np.cumsum(count)
        out: dict[str, NDArray[Any]] = {
            "pit": np.arange(1, n + 1),
            "factor": np.asarray(self.factors, dtype=np.float64),
            "blocks": blocks,
            "new_blocks": count,
            "value": np.asarray(self.values, dtype=np.float64),
        }
        if self.base is not None:
            weights = self.base()
            out["value_at_factor_1"] = np.cumsum(np.bincount(self.numbers, weights=weights, minlength=n + 1)[1:])
        tonnage = self[n - 1]._tonnage()
        if tonnage is not None:
            tons = tonnage[1]
            order = np.argsort(self.numbers, kind="stable")
            first = int(np.searchsorted(self.numbers[order], 1))
            ore_t = np.zeros(n)
            total_t = np.cumsum(np.bincount(self.numbers, weights=tons, minlength=n + 1)[1:])
            for k in range(n):
                rows = order[first : first + int(blocks[k])]
                ore = self.destination(k, rows) >= 0 if self.destination is not None else self.scenario(k, rows) > 0
                ore_t[k] = float(tons[rows][ore].sum())
            out["tonnage"] = total_t
            out["ore_tonnage"] = ore_t
            out["waste_tonnage"] = total_t - ore_t
            with np.errstate(divide="ignore", invalid="ignore"):
                out["strip_ratio"] = np.where(ore_t > 0, (total_t - ore_t) / np.where(ore_t > 0, ore_t, 1), np.nan)
        return out

    def summary(self) -> str:
        """A report with one line per pit."""
        t = self.table_by_pit()
        label = {"revenue": "revenue factor", "price": "price factor", "profit": "profit factor"}[self.scale]
        objective = "discounted value" if self.bench_discount else "value"
        head = f"{'pit':>5} {'factor':>8} {'blocks':>10} {'new':>8} {objective:>18}"
        if "value_at_factor_1" in t:
            head += f" {'value (factor 1)':>18}"
        if "tonnage" in t:
            head += f" {'tonnage':>16} {'strip ratio':>12}"
        distinct = int((t["new_blocks"] > 0).sum())
        title = f"Nested pits: {len(self)} {label}s, {distinct} distinct pits"
        if self.bench_discount:
            title += f", top-down bench discounting {self.bench_discount:.4%} per bench"
        lines = [title, head, "-" * len(head)]
        for k in range(len(self)):
            line = f"{t['pit'][k]:>5} {t['factor'][k]:>8.4f} {t['blocks'][k]:>10,} {t['new_blocks'][k]:>8,}"
            line += f" {t['value'][k]:>18,.0f}"
            if "value_at_factor_1" in t:
                line += f" {t['value_at_factor_1'][k]:>18,.0f}"
            if "tonnage" in t:
                line += f" {t['tonnage'][k]:>16,.0f} {t['strip_ratio'][k]:>12.4f}"
            lines.append(line)
        lines.append(f"solver time {sum(self.seconds):.3f} s for {len(self)} pits")
        return "\n".join(lines)

    def to_csv(self, path: str | Path, column: str = "PIT_NUMBER", pit: int | str | None = None) -> Path:
        """Write the model with the pit number of each block (1 = smallest pit, 0 = outside).

        ``pit`` writes only the blocks of one pit: ``"final"`` (the largest), ``"best"`` (the highest best
        case NPV, with ``discount_rate`` and ``ore_per_year``) or a pit number from 1. With economics, the
        value and destination of every block at factor 1 are written too (``VALUE``, ``DESTINATION``),
        unless the model already has columns with those names (they are then kept as they are).
        """
        from .readers import write_table

        table = dict(self.table)
        if column in table:
            raise ValueError(f"the table already has a column {column!r}: pass column=")
        table[column] = self.numbers
        if self.economics is not None:
            values, destination = self.economics.block_values(self.table)
            names = np.asarray(self.economics.destination_names())
            table.setdefault("VALUE", values)
            table.setdefault("DESTINATION", names[destination])  # -1 (waste) is the last name
        if pit is not None:
            k = self._pit_index(pit)
            keep = (self.numbers >= 1) & (self.numbers <= k)
            table = {name: np.asarray(v)[keep] for name, v in table.items()}
        return write_table(table, path)

    def _pit_index(self, pit: int | str) -> int:
        """The number (from 1) of ``"final"``, ``"best"`` or a pit number."""
        n = len(self)
        if pit == "final":
            return n
        if pit == "best":
            if not self._with_npv():
                raise ValueError("pit='best' needs discount_rate and ore_per_year in nested_pits")
            return self.npv().best_pit("best")
        if isinstance(pit, (int, np.integer)) and not isinstance(pit, bool) and 1 <= int(pit) <= n:
            return int(pit)
        raise ValueError(f"pit must be 'final', 'best' or a pit number from 1 to {n}, got {pit!r}")

    def plot(self, **kwargs: Any) -> Any:
        """3-D view of the nested pits coloured by pit number (jet), with a slider from the first pit to
        the last (``remaining=True``: the final pit, from which the slider removes the pits in order);
        ``screenshot="pits.png"`` saves the image (see ``pymineflow.plot.plot_nested``).

        ``color_by=`` (a column, ``"value"``, ``"ore"`` or ``"level"``) draws the largest pit instead,
        as ``plot_pit``.
        """
        if kwargs.get("color_by", "pit number") == "pit number":
            from .plot import plot_nested

            kwargs.pop("color_by", None)
            return plot_nested(self, **kwargs)
        from .plot import plot_pit

        return plot_pit(self[len(self) - 1], **kwargs)

    def npv(self, **kwargs: Any) -> PitByPit:
        """Discounted value of every pit as the final pit, in the best, worst and specified cases:
        ``pits.npv(discount_rate=0.10, mining_rate=..., processing_rate=...)`` (see
        ``pymineflow.pit_by_pit`` for the options). The discount rate and the ore per year given to
        ``nested_pits`` are the defaults: the ore per year is the processing capacity whenever
        ``processing_rate`` is not given, also together with a ``mining_rate``."""
        if self.discount_rate and kwargs.get("discount_rate") is None:
            kwargs["discount_rate"] = self.discount_rate
        if self.ore_per_year and kwargs.get("processing_rate") is None:
            kwargs["processing_rate"] = self.ore_per_year
        return pit_by_pit(self, **kwargs)

    def oes(self, **kwargs: Any) -> OES:
        """The ultimate pit OES: a block-by-block extraction sequence of the final pit, phase by phase and
        bench by bench, ordered inside every phase-bench by proximity or lookahead values, and its NPV (see
        ``pymineflow.ultimate_pit_oes`` for the options). The discount rate and the ore per year given to
        ``nested_pits`` are the defaults."""
        return ultimate_pit_oes(self, **kwargs)

    def _with_npv(self, npv_options: Mapping[str, Any] | None = None) -> bool:
        """An NPV can be computed: a discount rate (0 included when given) and a capacity. NPV options that cannot
        be used are an error, never ignored."""
        import inspect

        o = npv_options or {}
        allowed = set(inspect.signature(pit_by_pit).parameters) - {"pits", "timing"}
        unknown = sorted(set(o) - allowed)
        if unknown:
            raise TypeError(f"unknown NPV options {unknown}; use {sorted(allowed)}")
        rate = o.get("discount_rate") is not None or bool(self.discount_rate)
        capacity = bool(self.ore_per_year) or o.get("mining_rate") is not None or o.get("processing_rate") is not None
        if o and not (rate and capacity):
            need = "a discount rate (discount_rate=)" if not rate else "a capacity (mining_rate= or processing_rate=)"
            raise ValueError(f"the NPV options {sorted(o)} need {need}, here or in nested_pits")
        return rate and capacity

    def report_table(
        self, timing: str = "continuous", kind: str = "cumulative", **npv_options: Any
    ) -> dict[str, NDArray[Any]]:
        """One row per phase (a factor whose pit is larger than the previous one): ``phase``, ``factor``,
        ``profit`` (undiscounted, at factor 1); with a tonnage, ``profit_per_t``, ``rock``, ``ore``,
        ``strip_ratio`` and ``waste`` (ore: processed at factor 1).

        ``kind="cumulative"`` (default): every phase as the final pit, everything mined up to it; with a discount
        rate and a capacity, also ``npv_best`` and ``npv_worst``. ``kind="incremental"``: the blocks every phase
        adds to the previous one, with the increment of the best case NPV (``npv``). The NPV is that of
        ``npv(timing=timing, **npv_options)``: the discount rate and ore per year of ``nested_pits`` by default,
        every block discounted when it is mined (``timing="continuous"``), and any option of ``pit_by_pit``
        (``mining_rate``, ``processing_rate``, ``fixed_cost``...)."""
        return self._report(timing, kind, npv_options)[0]

    def _report(
        self, timing: str, kind: str, npv_options: Mapping[str, Any]
    ) -> tuple[dict[str, NDArray[Any]], PitByPit | None]:
        from .npv import _TIMINGS, _block_data

        check_kind(kind)
        if timing not in _TIMINGS:
            raise ValueError(
                f"timing must be one of {', '.join(map(repr, _TIMINGS))}, got {timing!r} (the kind of report is kind=)"
            )
        with_npv = self._with_npv(npv_options)
        tons: NDArray[np.float64] | None
        try:
            values, destination, tons, _ = _block_data(self, None, None)
        except ValueError as error:
            if "needs the tonnage" not in str(error) or with_npv:
                raise
            values, destination, _, _ = _block_data(self, 1.0, None)  # no tonnage column: values only
            tons = None
        cases = self.npv(timing=timing, **npv_options) if with_npv else None
        factors = np.asarray(self.factors, dtype=np.float64)
        table = phase_report(self.numbers, factors, values, tons, destination >= 0,
                             None if cases is None else cases.table, kind)  # fmt: skip
        return table, cases

    def report(self, timing: str = "continuous", kind: str = "cumulative", **npv_options: Any) -> str:
        """The phases of the nested pits, one line each, like a pit optimisation report (``kind``:
        ``"cumulative"`` or ``"incremental"``; ``timing`` and ``npv_options`` as in ``report_table``)."""
        t, cases = self._report(timing, kind, npv_options)
        label = {"revenue": "revenue factor", "price": "price factor", "profit": "profit factor"}[self.scale]
        lines = [f"Nested pits: {len(self)} {label}s from {self.factors[0]:g} to {self.factors[-1]:g}, "
                 f"{len(t['phase'])} phases ({describe(kind)})"]  # fmt: skip
        if self.bench_discount:
            line = f"Top-down bench discounting: {self.bench_discount:.4%} per bench"
            if self.benches_per_year:
                line += f" ({_percent(self.discount_rate)} a year, {self.benches_per_year:g} benches a year)"
            lines.append(line)
        if cases is not None:
            lines.append(f"NPV: {_capacities(cases)}, cash at {_WHEN.get(timing, timing)}; best case = the phases in "
                         "order, worst case = the final pit bench by bench")  # fmt: skip
        return report_text(t, lines)

    def export(
        self,
        folder: str | Path,
        *,
        files: Sequence[str] = ("pits", "final_pit", "report", "plan", "npv", "gif"),
        timing: str = "continuous",
        report: str | None = "both",
        npv_options: Mapping[str, Any] | None = None,
        **plot_options: Any,
    ) -> list[Path]:
        """Write a study into ``folder`` and return the files written. ``files`` chooses among ``"pits"``
        (``pits.csv``: the whole model with its pit number), ``"final_pit"`` (``final_pit.csv``: only the
        blocks of the final pit), ``"report"`` (``report_cumulative.csv`` and ``report_incremental.csv``, or only
        the one chosen by ``report=``), ``"plan"``
        (``plan.png``: plan view coloured by pit number), ``"npv"`` (``npv.png``: best and worst case
        NPV, only with an annual discount rate and ore per year) and ``"gif"`` (``nested_pits.gif``: the pit
        growing, and ``nested_pits_remaining.gif``: the final pit mined pit after pit). ``timing`` and
        ``npv_options`` (a dict of ``pit_by_pit`` options) are those of the NPV, as in ``report_table``;
        ``plot_options`` change the picture (``view``, ``rotation``, ``title``, ``cmap``, ``outline``,
        ``axes``, ``window_size``; by default a plan view without the model box)."""
        from .readers import write_table

        kinds(report)
        npv_options = dict(npv_options or {})
        self._with_npv(npv_options)  # NPV options that cannot be used are an error before anything is written
        if plot_options.get("view", "top") not in ("iso", "below", "top", "side"):
            raise ValueError(f"view must be 'iso', 'below', 'top' or 'side', got {plot_options['view']!r}")
        unknown = set(files) - {"pits", "final_pit", "report", "plan", "npv", "gif"}
        if unknown:
            raise ValueError(
                f"files must be among 'pits', 'final_pit', 'report', 'plan', 'npv' and 'gif', got {sorted(unknown)}"
            )
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        if "pits" in files:
            written.append(self.to_csv(folder / "pits.csv"))
        if "final_pit" in files:
            written.append(self.to_csv(folder / "final_pit.csv", pit="final"))
        for kind in kinds(report) if "report" in files else ():
            written.append(write_table(self.report_table(timing, kind, **npv_options), folder / f"report_{kind}.csv"))
        if "plan" in files:
            options = {"view": "top", "outline": False, **plot_options}
            self.plot(screenshot=folder / "plan.png", **options).close()
            written.append(folder / "plan.png")
        if "npv" in files and self._with_npv(npv_options):
            label = {"revenue": "revenue factor", "price": "price factor", "profit": "profit factor"}[self.scale]
            self.npv(timing=timing, **npv_options).plot(folder / "npv.png", factor_label=label, show=False)
            written.append(folder / "npv.png")
        if "gif" in files:
            options = {"view": "top", "outline": False, **plot_options}
            step = max(1, -(-len(self) // 100))  # at most about 100 frames
            for name, remaining in (("nested_pits.gif", False), ("nested_pits_remaining.gif", True)):
                written.append(self.animate(folder / name, step=step, remaining=remaining, **options))
        return written

    def animate(self, path: str | Path = "nested_pits.gif", **kwargs: Any) -> Path:
        """An animated GIF of the pit growing from the first pit to the last or, with ``remaining=True``, of
        the final pit mined pit after pit (see ``pymineflow.plot.animate_nested`` for the options)."""
        from .plot import animate_nested

        return animate_nested(self, path, **kwargs)


def _common_scale(top: NDArray[np.float64], exact: bool = True) -> int:
    """The integer scale of every scenario, chosen on the largest one (it bounds every other one).

    ``exact``: every scenario is integral when this one is (integer factors): the rule of ``scale_values``.
    Otherwise (fractional factors, whose scenarios are fractional even for integer values) the largest power of
    ten up to 10**6 that fits, so that no scenario is rounded to whole units."""
    if not top.size:
        return 1
    if exact:
        return scale_values(top)[1]
    a = np.abs(np.asarray(top, dtype=np.float64))
    largest, total = float(a.max()), float(a.sum(dtype=np.float64))
    fits = [f for f in _FACTORS if largest * f <= 2**53 and total * f < _SUM_LIMIT]
    if not fits:
        raise OverflowError("the largest factor makes the values exceed int64: use smaller factors or values")
    return fits[0]


def nested_pits(
    source: str | Path | Any,
    *,
    block_size: ArrayLike,
    slopes: SlopeSpec,
    factors: Sequence[float],
    scale: str = "revenue",
    value: str | None = None,
    economics: Economics | Mapping[str, Any] | str | Path | None = None,
    indices: Sequence[str] | None = None,
    coords: Sequence[str] | None = None,
    n_benches: int | None = None,
    z_up: bool = True,
    verbose: bool = False,
    bench_discount: float = 0.0,
    discount_rate: float = 0.0,
    benches_per_year: float | None = None,
    ore_per_year: float | None = None,
    threads: int = 0,
    **read_options: Any,
) -> NestedPits:
    """Nested pits of a block model for many factors (hundreds are fine), in one call.

    Parameters
    ----------
    factors : the factors, in any order (the pits come sorted by factor).
    scale : ``"revenue"`` (default): with a value column, each factor multiplies the positive block
        values (a profit factor; waste keeps its cost); with ``economics``, it multiplies the revenue
        net of selling costs and every block goes again to its best destination (the usual price
        factors of pit optimisation). ``"profit"``: the positive block values at factor 1 times each
        factor, destinations fixed (profit factors; the same as ``"revenue"`` on a value column).
        ``"price"``: each factor multiplies every metal price (needs ``economics``), so that the best
        destination of each block (process or waste) is chosen again at every price.
    economics : compute the values from grades, prices and costs (``pymineflow.Economics``).
    verbose : print every pit as it is solved: the ring it was solved on, its blocks and its time (in the
        order they finish: the two halves of every bisection are solved at the same time).
    threads : threads for the nested pits (0 = every core, 1 = one after the other); the pits are the same.
    bench_discount : top-down bench discounting, rate per bench (see ``solve``); applied to the block
        values of every factor. ``value_at_factor_1`` in the reports stays undiscounted.
    discount_rate, benches_per_year : the same from an annual rate (e.g. 0.10) and an average sinking
        rate: ``bench_discount = discount_rate / benches_per_year``. Optional, as everything here.
    ore_per_year : ore processed per year; with ``discount_rate``, ``report()``, ``export()`` and ``npv()``
        give the best and worst case NPV of every pit.
    The other arguments are those of ``solve``.

    No block loses value as the factor grows, so the pits are nested: the largest is solved on the
    whole model and every other one only on the ring between two pits already solved, by bisection.
    The result is the same as solving each factor alone, much faster, and the pits are kept as one
    array (``NestedPits.pit_number``).

    Examples
    --------
    >>> table = {"IX": [0, 1, 1], "IY": [0, 0, 0], "IZ": [0, 0, 1], "VALUE": [5.0, -1.0, -3.0]}
    >>> pits = nested_pits(table, block_size=(10, 10, 10), slopes=45, factors=[0.5, 1.0])
    >>> [p.n_blocks for p in pits], pits.pit_number().tolist()  # at 0.5 the rich block does not pay
    ([0, 2], [2, 0, 2])
    """
    _check_value_options(economics, value)
    economics = _as_economics(economics)
    if scale not in ("revenue", "price", "profit"):
        raise ValueError(f"scale must be 'revenue', 'price' or 'profit', got {scale!r}")
    if scale == "price" and economics is None:
        raise ValueError("scale='price' needs economics= (grades, prices and costs)")
    if len(factors) == 0:
        raise ValueError("factors must not be empty")
    if any(isinstance(f, (bool, np.bool_, str, bytes)) for f in factors):
        raise TypeError(f"factors must be numbers, got {list(factors)}")
    if any(not np.isfinite(f) or f < 0 for f in factors):
        raise ValueError(f"factors must be finite and >= 0, got {list(factors)}")
    ordered = sorted(float(f) for f in factors)
    integral = all(f == int(f) for f in ordered)  # integer factors keep integer values integral
    table = read_blocks(source, **read_options)
    where = _positions(table, indices, coords)
    model = PitModel(block_size=block_size, slopes=slopes, n_benches=n_benches, z_up=z_up, **where)
    for name, x in (("discount_rate", discount_rate), ("benches_per_year", benches_per_year),
                    ("ore_per_year", ore_per_year)):  # fmt: skip
        if isinstance(x, (bool, np.bool_, str, bytes)):
            raise TypeError(f"{name} must be a number, got {x!r}")
        if x is not None and (not np.isfinite(float(x)) or float(x) < 0 or (name != "discount_rate" and float(x) == 0)):
            raise ValueError(f"{name} must be finite and {'>= 0' if name == 'discount_rate' else '> 0'}, got {x!r}")
    if benches_per_year is not None and discount_rate:
        if bench_discount:
            raise ValueError("give bench_discount, or discount_rate with benches_per_year, not both")
        bench_discount = float(discount_rate) / float(benches_per_year)
    weights = bench_discount_weights(model.grid.iz, bench_discount)  # all ones without discounting

    destination: Callable[[int, NDArray[np.intp]], NDArray[np.int16]] | None = None
    if economics is None or scale == "profit":
        if economics is None:
            value_column = _value_column(table, value)
            raw = np.asarray(table[value_column])
        else:  # profit factors with economics: the values and destinations at factor 1
            raw, fixed = economics.block_values(table)
            value_column = "value from economics"

            def destination(k: int, rows: NDArray[np.intp]) -> NDArray[np.int16]:
                return fixed[rows]

        discounted = np.asarray(raw * weights if bench_discount else raw, dtype=np.float64)
        if not np.isfinite(discounted).all():
            raise ValueError(f"the values ({value_column}) contain NaN or infinite values")
        top_factor = ordered[-1]
        factor = _common_scale(np.where(discounted > 0, discounted * max(top_factor, 1.0), discounted), integral)
        base_ints = np.rint(discounted * factor).astype(np.int64)

        def scenario(k: int, rows: NDArray[np.intp] | None) -> NDArray[np.int64]:
            b = base_ints if rows is None else base_ints[rows]
            return np.where(b > 0, np.rint(b * ordered[k]), b).astype(np.int64)

        def base() -> NDArray[np.float64]:
            return np.asarray(raw, dtype=np.float64) if bench_discount else base_ints / factor

        top = scenario(len(ordered) - 1, None).astype(np.float64)
        if np.abs(top).sum() >= _SUM_LIMIT:
            raise OverflowError("the largest factor makes the sum of |values| exceed int64: use smaller factors")
    else:
        eco = economics
        keyword = "price_factor" if scale == "price" else "revenue_factor"
        value_column = "value from economics"

        def real(k: int, rows: NDArray[np.intp] | None = None) -> tuple[NDArray[np.float64], NDArray[np.int16]]:
            f = ordered[k]
            if keyword == "price_factor":
                return eco.block_values(table, price_factor=f, rows=rows)
            return eco.block_values(table, revenue_factor=f, rows=rows)

        factor = _common_scale(real(len(ordered) - 1)[0] * weights, integral)

        def scenario(k: int, rows: NDArray[np.intp] | None) -> NDArray[np.int64]:
            w = weights if rows is None else weights[rows]
            return np.rint(real(k, rows)[0] * w * factor).astype(np.int64)

        def destination(k: int, rows: NDArray[np.intp]) -> NDArray[np.int16]:
            return real(k, rows)[1]

        def base() -> NDArray[np.float64]:
            return eco.block_values(table)[0]

    progress = None
    if verbose:
        kind = {"revenue": "revenue factor", "price": "price factor", "profit": "profit factor"}[scale]
        print(f"Solving {len(ordered)} nested pits by {kind}: the largest on the whole model, then by bisection,")
        print("each one only on the ring between two pits already solved (inner pit mined, outer pit as limit).")
        print(
            f"{'order':>6} {'pit':>5} {kind:>14} {'ring between':>14} {'cells solved':>13} {'blocks':>10} "
            f"{'value':>18} {'time':>10}"
        )

        def progress(info: dict[str, Any]) -> None:
            ring = "whole model" if info["outer"] == 0 else f"{info['inner']} and {info['outer']}"
            print(
                f"{info['order']:>6} {info['pit']:>5} {ordered[info['pit'] - 1]:>14.4f} {ring:>14} "
                f"{info['cells']:>13,} {info['blocks']:>10,} {info['value'] / factor:>18,.0f} "
                f"{info['seconds'] * 1000:>7.1f} ms"
            )

    numbers, pit_values, seconds = model.nested_pit_numbers(scenario, len(ordered), progress, threads=threads)
    return NestedPits(
        factors=ordered,
        scale=scale,
        numbers=numbers,
        values=[v / factor for v in pit_values],
        seconds=seconds,
        model=model,
        table=table,
        factor=factor,
        value_column=value_column,
        scenario=scenario,
        destination=destination,
        destinations=None if economics is None else tuple(economics.destination_names()),
        base=base,
        economics=economics,
        bench_discount=float(bench_discount),
        discount_rate=float(discount_rate),
        benches_per_year=None if benches_per_year is None else float(benches_per_year),
        ore_per_year=None if ore_per_year is None else float(ore_per_year),
    )


_TONNAGE_NAMES = ("TON", "TONS", "TONNES", "TONNAGE", "TONN", "TONNS", "TONELAJE")


def _as_economics(economics: Economics | Mapping[str, Any] | str | Path | None) -> Economics | None:
    """An ``Economics``, a dict with the same fields, or a JSON file."""
    if economics is None or isinstance(economics, Economics):
        return economics
    if isinstance(economics, (str, Path)):
        return read_economics(economics)
    return economics_from_dict(economics)


def _metal(name: str, spec: Mapping[str, Any]) -> tuple[Metal, Any]:
    """A metal from a simple dict: price, grade_unit, price_unit, recovery, payable, selling_cost."""
    known = {"price", "grade_unit", "price_unit", "recovery", "payable", "selling_cost"}
    unknown = set(spec) - known
    if unknown:
        raise ValueError(f"metal {name}: unknown keys {sorted(unknown)}; use {sorted(known)}")
    if "price" not in spec:
        raise ValueError(f"metal {name}: a price is needed")
    grade_unit = spec.get("grade_unit", "g/t")
    price_unit = spec.get("price_unit", "lb" if grade_unit == "%" else "oz")
    metal = Metal(
        name,
        price=spec["price"],
        grade_unit=grade_unit,
        price_unit=price_unit,
        payable=spec.get("payable", 1.0),
        selling_cost=spec.get("selling_cost", 0.0),
    )
    return metal, spec.get("recovery", 1.0)  # a number, a column, one value per block or a dict by domain


def value_blocks(
    source: str | Path | Any,
    *,
    metals: Mapping[str, Mapping[str, Any]] | None = None,
    processing_cost: Param | None = None,
    mining_cost: Param | None = None,
    economics: Economics | Mapping[str, Any] | str | Path | None = None,
    tonnage: str | None = None,
    density: str | float | None = None,
    block_size: ArrayLike | None = None,
    tonnage_unit: str = "t",
    contaminants: Mapping[str, Mapping[str, Any]] | None = None,
    domain: str | None = None,
    price_factor: float = 1.0,
    revenue_factor: float = 1.0,
    column: str = "VALUE",
    **read_options: Any,
) -> dict[str, NDArray[Any]]:
    """Block values from grades, prices and costs: the model plus a value and a destination column.

    The simple way: one process, and the metals as ``{grade column: {...}}`` with the keys ``price``,
    ``grade_unit`` (``"g/t"`` by default, ``"ppm"``, ``"ppb"``, ``"%"`` or ``"oz/t"``), ``price_unit``
    (``"oz"`` by default, ``"lb"`` for ``"%"``; also ``"g"``, ``"kg"``, ``"t"``), ``recovery``
    (a fraction, 1 by default), ``payable`` and ``selling_cost``. For several processes or more
    control, pass ``economics`` (an ``Economics``, a dict with its fields, or a JSON file) instead.

    Parameters
    ----------
    source : a file or a table (see ``read_blocks``).
    metals, processing_cost, mining_cost : the simple way (costs per tonnage unit). Every number (price,
        recovery, payable, selling cost, processing and mining cost) may also be a column name, an array with
        one value per block, or a dict by domain (with ``domain=``), as in ``Economics``.
    economics : the full way (``pymineflow.Economics``).
    tonnage : tonnage column; found by name (``TON``, ``TONNES``, ``TONNAGE``...) when ``None``.
    density : instead of a tonnage column, the density (tonnage per unit volume): a column, a number, an array
        with one value per block or a dict by domain; with ``block_size``, the tonnage is ``density * dx * dy * dz``.
    tonnage_unit : ``"t"`` (metric tonne), ``"st"`` (short ton) or ``"lt"`` (long ton).
    contaminants : ``{grade column: {"penalty": ..., "threshold": ..., "limit": ...}}`` (simple way).
    domain : the domain column of the dicts by domain (simple way; ``Economics(domain=...)`` otherwise).
    price_factor, revenue_factor : multiply every price, or the revenue.
    column : name of the value column added; ``DESTINATION`` holds the process or ``"waste"``.

    Returns the table (``{column: numpy array}``), ready for ``solve`` or ``nested_pits``.

    Examples
    --------
    >>> table = {"IX": [0, 1], "IY": [0, 0], "IZ": [0, 0], "AU": [0.0, 2.0], "TON": [1000.0, 1000.0]}
    >>> out = value_blocks(table, metals={"AU": {"price": 1800, "recovery": 0.9}},
    ...                    processing_cost=20, mining_cost=2)
    >>> np.round(out["VALUE"], 2).tolist(), out["DESTINATION"].tolist()
    ([-2000.0, 82168.42], ['waste', 'process'])
    """
    simple = metals is not None or processing_cost is not None or mining_cost is not None or contaminants is not None
    if economics is not None and simple:
        raise ValueError("pass economics= or the simple way (metals=, processing_cost=, mining_cost=), not both")
    if economics is not None and domain is not None:
        raise ValueError("with economics=, the domain is that of the economics (Economics(domain=...))")
    if economics is not None and tonnage_unit != "t":
        raise ValueError("with economics=, the tonnage unit is that of the economics (Economics(tonnage_unit=...))")
    table = dict(read_blocks(source, **read_options))
    if column in table:
        raise ValueError(f"the table already has a column {column!r}: pass another column= name")
    if density is not None:
        if tonnage is not None:
            raise ValueError("pass tonnage= or density=, not both")
        if block_size is None:
            raise ValueError("density= needs block_size=(dx, dy, dz) to compute the tonnage")
        # a column, a number, one value per block or a dict by domain, as in Economics
        given = _as_economics(economics)
        reader = Economics(metals=[], processes=[Process("tonnage", 0.0, {})], mining_cost=0.0, density=density,
                           block_size=block_size,  # type: ignore[arg-type]
                           domain=domain if given is None else given.domain)  # fmt: skip
        tons = reader.tonnages(table)
        tonnage = "TONNAGE" if "TONNAGE" not in table else "TONNAGE_FROM_DENSITY"
        table[tonnage] = tons
    eco = _as_economics(economics)
    if eco is None:
        if metals is None or processing_cost is None or mining_cost is None:
            raise ValueError("value_blocks needs metals=, processing_cost= and mining_cost= (or economics=)")
        if tonnage is None:
            tonnage = _find(table, _TONNAGE_NAMES)
            if tonnage is None:
                raise ValueError(
                    f"no tonnage column found (looked for {list(_TONNAGE_NAMES)}): pass tonnage= or density="
                )
        parsed = [_metal(name, spec) for name, spec in metals.items()]
        eco = Economics(
            metals=[m for m, _ in parsed],
            processes=[Process("process", processing_cost, {m.grade: r for m, r in parsed})],
            mining_cost=mining_cost,
            tonnage=tonnage,
            tonnage_unit=tonnage_unit,
            contaminants=[Contaminant(name, **spec) for name, spec in (contaminants or {}).items()],
            domain=domain,
        )
    elif tonnage is not None:
        eco = replace(eco, tonnage=tonnage)
    values, destination = eco.block_values(table, price_factor=price_factor, revenue_factor=revenue_factor)
    table[column] = values
    names = np.asarray(eco.destination_names(), dtype=object)
    table["DESTINATION" if "DESTINATION" not in table else f"{column}_DESTINATION"] = names[destination]
    return table
