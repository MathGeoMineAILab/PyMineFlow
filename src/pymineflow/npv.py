"""Discounted value of nested pits: the pit-by-pit analysis (best, worst and specified cases).

Every nested pit is taken in turn as the final pit, mined in a fixed sequence at the given capacities,
and its cash flow is discounted period by period:

- **best case**: the nested pits are the phases (pit 1, then the ring from pit 1 to pit 2, ...), each
  one mined bench by bench from the top.
- **worst case**: the final pit mined bench by bench from the top, each bench complete.
- **specified case**: the pushbacks you choose (pit numbers), then the final pit, each one bench by bench.

Capacities hold per period, as in strategic mine-planning software and in MineLib's scheduling
problems: in every period the sequence is mined as far as it can go without exceeding the mining
capacity (all tonnage) or the processing capacity of any process (its ore tonnage); waste can still be
mined while the plant is full. A bench can be split between periods; its tonnage, ore and value are
taken in proportion. There is no stockpile.

By default every block is discounted when it is mined, by ``exp(-rate * t)`` with ``t`` the time elapsed in
periods (``timing="continuous"``). By period instead, cash of period ``k`` (1, 2, ...) is discounted by
``(1 + rate) ** -k`` (``"end"``), ``(1 + rate) ** -(k - 0.5)`` (``"mid"``) or ``(1 + rate) ** -(k - 1)``
(``"start"``, MineLib's convention). Every sequence respects the precedences: a block's predecessors are in
the same or an earlier phase and, within a phase, on a higher bench. Both cases are therefore feasible schedules, so
both are lower bounds of the best NPV a schedule can reach with these capacities; "best" and "worst"
name the two extreme ways of using the nested pits, not bounds of every schedule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._mineflow import _oes_schedule

__all__ = ["PitByPit", "pit_by_pit"]

_SHIFT = {"end": 0.0, "mid": 0.5, "start": 1.0}  # cash of period k is discounted (1 + rate) ** -(k - shift)
_TIMINGS = (*_SHIFT, "continuous")  # continuous: every piece discounted when it is mined, exp(-rate * t)


def _number(name: str, value: Any, low: float, strict: bool) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, got {value!r}") from None
    if not np.isfinite(x) or x < low or (strict and x == low):
        sign = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {sign} {low:g}, got {value!r}")
    return x


def _at(cum: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative quantity at fractional positions ``x`` (one per row) of piecewise-linear sequences."""
    u = cum.shape[1] - 1
    i = np.minimum(np.floor(x).astype(np.int64), u - 1)
    rows = np.arange(cum.shape[0])
    return cum[rows, i] + (x - i) * (cum[rows, i + 1] - cum[rows, i])


def _fill(cums: list[NDArray[np.float64]], caps: list[float]) -> NDArray[np.float64]:
    """Period boundaries of sequences mined as far as every capacity allows, period after period.

    ``cums[c]`` is the cumulative use of capacity ``c`` along each sequence (rows), with a leading 0;
    position ``x`` in ``[0, U]`` is a fraction of the units. Returns the boundaries, ``(rows, periods + 1)``,
    from 0 to ``U`` (padded with ``U`` once a sequence is finished).
    """
    rows, u1 = cums[0].shape
    u = float(u1 - 1)
    x = np.zeros(rows)
    bounds = [x]
    # every period but the last fills one capacity: periods <= sum of total use / capacity + 1
    guard = int(sum(float(np.max(c[:, -1]) / cap) for c, cap in zip(cums, caps, strict=True))) + len(caps) + 2
    for _ in range(guard):
        if not (x < u).any():
            break
        new = np.full(rows, u)
        for cum, cap in zip(cums, caps, strict=True):
            target = _at(cum, x) + cap
            over = cum > (target + 1e-9 * cap)[:, None]  # what this period cannot reach
            has = over.any(axis=1)
            i = np.maximum(np.argmax(over, axis=1), 1)
            r = np.arange(rows)
            low, high = cum[r, i - 1], cum[r, i]
            reach = (i - 1) + np.clip((target - low) / np.where(high > low, high - low, 1.0), 0.0, 1.0)
            new = np.where(has, np.minimum(new, reach), new)
        x = np.where(x < u, np.maximum(new, x), x)
        bounds.append(x)
    else:
        raise RuntimeError("the period filling did not finish")  # cannot happen: every period uses a capacity
    return np.stack(bounds, axis=1)


@dataclass(frozen=True)
class _Units:
    """Tonnage, value and capacity use of every unit (increment x bench, top first)."""

    nz: int
    ton: NDArray[np.float64]  # (n, nz)
    value: NDArray[np.float64]  # (n, nz)
    ore: NDArray[np.float64]  # (processes, n, nz)
    uses: list[NDArray[np.float64]]  # one (n, nz) array per capacity
    caps: list[float]
    prefix: list[NDArray[np.float64]]  # value, ton, ore of each process, uses: sums over pits 1..k, (n + 1, nz)


def _sequence(units: _Units, phases: Sequence[int], final: int) -> list[NDArray[np.float64]]:
    """Per-unit arrays (value, tonnage, ore of each process, then each capacity use) of the pushbacks
    ``phases`` (pit numbers below ``final``) and the final pit, each one bench by bench."""
    limits = [p for p in phases if p < final] + [final]
    lows = [0, *limits[:-1]]
    return [np.concatenate([pre[b] - pre[low] for low, b in zip(lows, limits, strict=True)]) for pre in units.prefix]


def _continuous(units: _Units, phases: Sequence[int], final: int, rate: float, fixed: float) -> dict[str, Any]:
    """Schedule of the pushbacks ``phases`` and the final pit, unit by unit, every piece discounted when it is
    mined (``exp(-rate * t)``, the time running with the binding capacity), the fixed cost at a constant rate."""
    arrays = _sequence(units, phases, final)
    k = 2 + units.ore.shape[0]
    uses = np.maximum(np.stack(arrays[k:]), 0.0)  # differences of prefix sums: no rounding below zero
    track = np.maximum(np.stack(arrays[1:k]), 0.0)
    return _oes_schedule(np.ascontiguousarray(arrays[0]), np.ascontiguousarray(uses),
                         np.asarray(units.caps, dtype=np.float64), np.ascontiguousarray(track), rate, 0.0, fixed,
                         True)  # fmt: skip


def _evaluate(
    value: NDArray[np.float64], uses: list[NDArray[np.float64]], caps: list[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Boundaries, cash per period and capacity fraction of the last period for sequences (rows)."""
    zero = np.zeros((value.shape[0], 1))
    cums = [np.concatenate([zero, np.cumsum(np.maximum(w, 0.0), axis=1)], axis=1) for w in uses]
    bounds = _fill(cums, caps)
    cv = np.concatenate([zero, np.cumsum(value, axis=1)], axis=1)
    at = np.stack([_at(cv, bounds[:, k]) for k in range(bounds.shape[1])], axis=1)
    return bounds, np.diff(at, axis=1), np.stack([np.stack([_at(c, bounds[:, k]) for c in cums]) for k in
                                                  range(bounds.shape[1])], axis=2)  # fmt: skip


@dataclass(frozen=True)
class PitByPit:
    """Discounted value of every nested pit as the final pit (see ``pit_by_pit``).

    ``table`` has one row per pit: ``pit``, ``factor``, ``tonnage``, ``ore_tonnage``, ``value``
    (undiscounted, without fixed costs), then ``life_best`` and ``npv_best``, ``life_worst`` and
    ``npv_worst`` and, with pushbacks, ``life_specified`` and ``npv_specified``. Lives are in periods
    (the last one counts by the fraction of the binding capacity it uses).
    """

    table: dict[str, NDArray[Any]]
    discount_rate: float
    mining_rate: float | None
    processing_rate: dict[str, float] | float | None
    fixed_cost: float
    timing: str
    pushbacks: tuple[int, ...] | None
    processes: tuple[str, ...]
    _units: _Units = field(repr=False)

    def best_pit(self, case: str = "best") -> int:
        """The pit number (1-based) with the highest NPV in ``case`` (best, worst or specified)."""
        column = f"npv_{case}"
        if column not in self.table:
            raise ValueError(f"case must be one of {[c[4:] for c in self.table if c.startswith('npv_')]}")
        return int(np.argmax(self.table[column])) + 1

    def schedule(self, pit: int, case: str = "best") -> dict[str, NDArray[Any]]:
        """Period by period for pit ``pit`` (1-based) in ``case``: ``period``, ``tonnage``, ``ore_tonnage``
        (and one column per process when there are several), ``value``, ``discount_factor``,
        ``fixed_cost`` and ``discounted`` (value minus fixed cost, discounted)."""
        n = len(self.table["pit"])
        if not 1 <= int(pit) <= n or int(pit) != pit:
            raise ValueError(f"pit must be a pit number from 1 to {n}, got {pit!r}")
        if case == "best":
            phases: Sequence[int] = range(1, n + 1)
        elif case == "worst":
            phases = ()
        elif case == "specified" and self.pushbacks is not None:
            phases = self.pushbacks
        else:
            raise ValueError(f"case must be one of {[c[4:] for c in self.table if c.startswith('npv_')]}")
        u = self._units
        if self.timing == "continuous":
            return self._continuous_schedule(phases, int(pit))
        arrays = [a[None, :] for a in _sequence(u, phases, int(pit))]
        value, ton = arrays[0], arrays[1]
        ore = arrays[2 : 2 + len(self.processes)]
        bounds, cash, _ = _evaluate(value, arrays[2 + len(self.processes) :], u.caps)
        used = np.diff(bounds[0]) > 0
        periods = int(used.sum())
        b = bounds[:, : periods + 1]

        def per_period(a: NDArray[np.float64]) -> NDArray[np.float64]:
            c = np.concatenate([[[0.0]], np.cumsum(a, axis=1)], axis=1)
            return np.diff(np.stack([_at(c, b[:, k]) for k in range(periods + 1)], axis=1)[0])

        k = np.arange(1, periods + 1, dtype=np.float64)
        factor = (1.0 + self.discount_rate) ** -(k - _SHIFT[self.timing])
        life = float(self.table[f"life_{case}"][int(pit) - 1])
        share = np.minimum(np.maximum(life - (k - 1), 0.0), 1.0)
        out: dict[str, NDArray[Any]] = {"period": k.astype(np.int64), "tonnage": per_period(ton)}
        ores = [per_period(o) for o in ore]
        out["ore_tonnage"] = np.sum(ores, axis=0)
        if len(self.processes) > 1:
            for name, o in zip(self.processes, ores, strict=True):
                out[f"ore_tonnage_{name}"] = o
        out["value"] = cash[0, :periods]
        out["discount_factor"] = factor
        out["fixed_cost"] = self.fixed_cost * share
        out["discounted"] = (out["value"] - out["fixed_cost"]) * factor
        return out

    def _continuous_schedule(self, phases: Sequence[int], pit: int) -> dict[str, NDArray[Any]]:
        s = _continuous(self._units, phases, pit, self.discount_rate, self.fixed_cost)
        count = s["period_value"].size
        k = np.arange(count, dtype=np.float64)
        life = float(s["time_end"][-1]) if s["time_end"].size else 0.0
        share = np.clip(life - k, 0.0, 1.0)
        rate, fixed = self.discount_rate, self.fixed_cost
        track = list(s["period_track"])
        out: dict[str, NDArray[Any]] = {"period": (k + 1).astype(np.int64), "tonnage": track[0]}
        out["ore_tonnage"] = np.sum(track[1:], axis=0)
        if len(self.processes) > 1:
            for name, o in zip(self.processes, track[1:], strict=True):
                out[f"ore_tonnage_{name}"] = o
        value = s["period_value"]
        out["value"] = value
        mean = np.exp(-rate * (k + 0.5 * share))
        out["discount_factor"] = np.divide(s["period_discounted"], value, out=mean, where=np.abs(value) > 0)
        out["fixed_cost"] = fixed * share
        paid = _continuous_cost(fixed, rate, np.minimum(k + 1, life)) - _continuous_cost(fixed, rate, k)
        out["discounted"] = s["period_discounted"] - paid
        return out

    def summary(self) -> str:
        """A report with one line per pit and the pit of highest NPV in every case."""
        t = self.table
        if self.processing_rate is None:
            plant = "unlimited"
        elif isinstance(self.processing_rate, dict):
            plant = ", ".join(f"{k} {v:,.10g}" for k, v in self.processing_rate.items())
        else:
            plant = f"{self.processing_rate:,.10g}"
        mine = "unlimited" if self.mining_rate is None else f"{self.mining_rate:,.10g}"
        cases = ["best", "worst"] + (["specified"] if self.pushbacks is not None else [])
        if self.timing == "continuous":
            cash = "every block discounted when it is mined, exp(-rate x time)"
        else:
            cash = f"cash at the {self.timing} of each period"
        lines = [
            f"Pit by pit: discount rate {self.discount_rate * 100:.2f} % per period, {cash}",
            f"Capacities per period: mining {mine}, processing {plant}; fixed cost {self.fixed_cost:,.10g} per period",
        ]
        if self.pushbacks is not None:
            lines.append(f"Specified pushbacks: pits {', '.join(map(str, self.pushbacks))} and the final pit")
        head = f"{'pit':>5} {'factor':>8} {'tonnage':>16} {'ore tonnage':>16} {'value':>18}"
        for c in cases:
            head += f" {'life ' + c:>15} {'NPV ' + c:>18}"
        lines += [head, "-" * len(head)]
        for k in range(len(t["pit"])):
            line = (
                f"{t['pit'][k]:>5} {t['factor'][k]:>8.4f} {t['tonnage'][k]:>16,.0f} {t['ore_tonnage'][k]:>16,.0f}"
                f" {t['value'][k]:>18,.0f}"
            )
            for c in cases:
                line += f" {t['life_' + c][k]:>15.2f} {t['npv_' + c][k]:>18,.0f}"
            lines.append(line)
        for c in cases:
            k = self.best_pit(c) - 1
            lines.append(
                f"Highest NPV, {c} case: pit {k + 1} (factor {t['factor'][k]:.4f}), NPV {t['npv_' + c][k]:,.0f}, "
                f"{t['life_' + c][k]:.2f} periods"
            )
        return "\n".join(lines)

    def to_csv(self, path: str | Path) -> Path:
        """Write the table (one row per pit) to a CSV file."""
        from .readers import write_table

        return write_table(self.table, path)

    def plot(
        self,
        path: str | Path | None = None,
        *,
        title: str | None = None,
        factor_label: str = "factor",
        tonnage: bool = True,
        show: bool | None = None,
    ) -> Any:
        """The pit-by-pit chart: NPV of the best and worst cases (and the specified one) against the
        factor of each pit, the pit of highest NPV marked on every curve, and the ore and waste tonnage
        of each pit as bars on a second axis (``tonnage=False`` to leave them out).

        ``path`` saves the image (PNG, SVG, PDF...); ``show`` opens the window (by default only when
        no ``path`` is given). Needs matplotlib (installed with PyVista). Returns the figure.
        """
        from matplotlib.figure import Figure
        from matplotlib.ticker import FuncFormatter

        if show is None:
            show = path is None
        t = self.table
        f = np.asarray(t["factor"], dtype=np.float64)
        if show:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(11, 6.2), dpi=120)
        else:
            fig = Figure(figsize=(11, 6.2), dpi=120)  # no window, no pyplot state
        ax = fig.subplots()
        styles = {"best": ("#1565c0", "o", "Best case"), "worst": ("#c62828", "s", "Worst case"),
                  "specified": ("#2e7d32", "^", "Specified case")}  # fmt: skip
        cases = ["best", "worst"] + (["specified"] if "npv_specified" in t else [])
        # only the pits with blocks: low factors that give empty pits would flatten the chart
        tons = np.asarray(t["tonnage"], dtype=np.float64)
        filled = np.flatnonzero((tons > 0) | (np.asarray(t["npv_best"]) != 0))
        shown = np.arange(filled[0], f.size) if filled.size else np.arange(f.size)
        fs = f[shown]
        big = max((float(np.nanmax(np.abs(np.asarray(t[f"npv_{c}"])[shown]))) for c in cases), default=0.0)
        scale, unit = (1e6, "M") if big >= 1e6 else (1.0, "")
        curves = {c: np.asarray(t[f"npv_{c}"], dtype=np.float64)[shown] / scale for c in cases}
        width = 0.02 * (fs.max() - fs.min() if fs.size > 1 and fs.max() > fs.min() else 1.0)
        widths: Any = width
        if fs.size > 1:
            gaps = np.diff(fs)
            gaps = np.where(gaps > 0, gaps, np.inf)
            near = np.minimum(np.r_[gaps[0], gaps], np.r_[gaps, gaps[-1]])  # each bar as wide as its spacing
            widths = 0.85 * np.where(np.isfinite(near), near, width)
            width = float(np.min(widths))
        ax.set_xlim(float(fs.min()) - width, float(fs.max()) + width)
        # NPV axis from the lowest to the highest value of every case (zero always in view), with a margin
        low = min(0.0, *(float(np.nanmin(v)) for v in curves.values()))
        high = max(0.0, *(float(np.nanmax(v)) for v in curves.values()))
        span = high - low if high > low else 1.0
        y0, y1 = low - 0.05 * span, high + 0.08 * span
        ax.set_ylim(y0, y1)
        if tonnage:
            ax2 = ax.twinx()
            ore = np.asarray(t["ore_tonnage"], dtype=np.float64)[shown] / 1e6
            waste = tons[shown] / 1e6 - ore
            ax2.bar(fs, ore, width=widths, color="#f2b134", alpha=0.35, label="ore tonnage", zorder=1)
            ax2.bar(fs, waste, width=widths, bottom=ore, color="#9e9e9e", alpha=0.25, label="waste tonnage", zorder=1)
            ax2.set_ylabel("tonnage of the pit (millions)")
            top = float(np.max(ore + waste, initial=0.0)) or 1.0
            # the bars fill about the lower 60 % of the chart, and their zero is the zero of the NPV axis
            zero = -y0 / (y1 - y0)
            r1 = top / max(0.6 - zero, 0.3) * (1 - zero) if zero < 0.6 else top / 0.3
            ax2.set_ylim(-zero / (1 - zero) * r1, r1)
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 0 else ""))
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
        ax.axhline(0.0, color="black", lw=0.6, alpha=0.5, zorder=2)
        for c in cases:
            color, marker, label = styles[c]
            y = curves[c]
            k = int(np.nanargmax(y))
            pit = int(shown[k]) + 1
            ax.plot(fs, y, color=color, lw=2.0, zorder=3,
                    label=f"{label}: highest NPV {y[k]:,.1f}{unit} at pit {pit} (factor {fs[k]:.3g})")  # fmt: skip
            ax.scatter([fs[k]], [y[k]], s=80, color=color, marker=marker, edgecolor="black", zorder=4)
        ax.set_xlabel(factor_label)
        ax.set_ylabel(f"discounted NPV ({unit or 'currency'})" if unit else "discounted NPV")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(True, alpha=0.3)
        mine = "unlimited" if self.mining_rate is None else f"{self.mining_rate:,.10g}"
        plant = self.processing_rate if not isinstance(self.processing_rate, dict) else "per process"
        plant_text = "unlimited" if plant is None else (plant if isinstance(plant, str) else f"{plant:,.10g}")
        ax.set_title(
            title
            or f"Pit by pit: discount rate {self.discount_rate * 100:g} %, mining {mine}, "
            f"processing {plant_text} per period",
            fontsize=11,
        )
        handles, labels = ax.get_legend_handles_labels()
        if tonnage:
            h2, l2 = ax2.get_legend_handles_labels()
            handles, labels = handles + h2, labels + l2
        # below the chart: it never hides a curve or a bar, whatever the shape of the case
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=9, frameon=False)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, bbox_inches="tight")
        if show:
            plt.show()
        return fig


def _block_data(
    pits: Any, tonnage: str | float | ArrayLike | None, ore: str | ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], tuple[str, ...]]:
    """Value at factor 1, destination (process index, -1 = waste), tonnage and process names per row."""
    table = pits.table
    economics = getattr(pits, "economics", None)
    if economics is not None:
        if ore is not None:
            raise ValueError(
                "ore= is for models without economics: with economics every block goes to its best process"
            )
        values, destination = economics.block_values(table)
        names = tuple(p.name for p in economics.processes)  # by destination index (waste is -1)
    else:
        if pits.base is None:
            raise ValueError("these nested pits have no block values")
        values = np.asarray(pits.base(), dtype=np.float64)
        if ore is None:
            processed = values > 0  # positive value: processed
        else:
            if isinstance(ore, str):
                if ore not in table:
                    raise ValueError(f"ore column {ore!r} not found; available: {list(table)}")
                column = np.asarray(table[ore])
            else:
                column = np.asarray(ore)
            if column.shape != values.shape or column.dtype.kind not in "biuf":
                raise ValueError(
                    f"ore must be a column or an array of 0/1 (or booleans), one per block ({values.size})"
                )
            if column.dtype.kind == "f" and not np.isfinite(column).all():
                raise ValueError("ore has missing values")
            processed = column != 0
        destination = np.where(processed, 0, -1)
        names = ("ore",)
    rows = values.size
    if tonnage is None:
        if economics is not None:
            tonnage = economics.tonnages(table)
        else:
            found = pits[len(pits) - 1]._tonnage()
            if found is None:
                raise ValueError(
                    "pit_by_pit needs the tonnage of the blocks (no tonnage column, or several): tonnage='column' or a "
                    "number per block"
                )
            tonnage = found[0]
    if isinstance(tonnage, str):
        if tonnage not in table:
            raise ValueError(f"tonnage column {tonnage!r} not found; available: {list(table)}")
        tons = np.asarray(table[tonnage], dtype=np.float64)
    else:
        tons = np.asarray(tonnage, dtype=np.float64)
        if tons.ndim == 0:
            tons = np.full(rows, float(tons))
    if tons.shape != (rows,) or not np.isfinite(tons).all() or (tons < 0).any():
        raise ValueError(f"tonnage must be finite and >= 0, one value per block ({rows})")
    return np.asarray(values, dtype=np.float64), np.asarray(destination, dtype=np.int64), tons, names


def pit_by_pit(
    pits: Any,
    *,
    discount_rate: float,
    mining_rate: float | None = None,
    processing_rate: float | Mapping[str, float] | None = None,
    fixed_cost: float = 0.0,
    pushbacks: Sequence[int] | None = None,
    timing: str = "continuous",
    tonnage: str | float | ArrayLike | None = None,
    ore: str | ArrayLike | None = None,
) -> PitByPit:
    """Discounted value (NPV) of every nested pit as the final pit: best, worst and specified cases.

    Parameters
    ----------
    pits : the result of ``nested_pits``. Blocks are valued at factor 1 (the real prices and costs);
        with ``economics``, every block goes to its best process.
    discount_rate : per period (e.g. 0.10 for 10 % a year when the capacities are per year).
    mining_rate : tonnage mined per period, ore and waste (``None``: unlimited).
    processing_rate : ore tonnage processed per period: one number for all processes, or a dict
        ``{process name: capacity}`` (processes left out are unlimited). At least one capacity is needed.
    fixed_cost : cost per period (e.g. general and administrative) over the life of the mine; the last
        period pays the fraction of the binding capacity it uses.
    pushbacks : pit numbers (1-based, increasing) of the pushbacks for the specified case.
    timing : ``"continuous"`` (default): every block discounted when it is mined, by ``exp(-rate * t)`` with
        ``t`` the time elapsed in periods (the convention of strategic mine-planning reports), the fixed cost
        paid at a constant rate; or the cash of each period at its ``"end"``, ``"mid"`` or ``"start"``
        (``"start"`` is MineLib's convention: the first period is not discounted).
    tonnage : column, number or array with the tonnage of each block (default: the economics'
        tonnage column, or a column called TON, TONNES, TONNAGE...).
    ore : without economics, which blocks are processed: a column or an array of 0/1 (e.g. MineLib's
        DESTINATION). By default, the blocks of positive value. The process is called ``"ore"``.

    Examples
    --------
    >>> import math, pymineflow
    >>> table = {"IX": [0], "IY": [0], "IZ": [0], "VALUE": [1000.0], "TON": [100.0]}
    >>> pits = pymineflow.nested_pits(table, block_size=(10, 10, 10), slopes=45, factors=[1.0])
    >>> r = pit_by_pit(pits, discount_rate=0.10, mining_rate=50)  # 2 periods, every block discounted when mined
    >>> round(float(r.table["npv_worst"][0]), 2), round(500 * (1 - math.exp(-0.2)) / 0.1, 2)
    (906.35, 906.35)
    >>> r = pit_by_pit(pits, discount_rate=0.10, mining_rate=50, timing="end")  # 500 at the end of each period
    >>> round(float(r.table["npv_worst"][0]), 2), round(500 / 1.1 + 500 / 1.1**2, 2)
    (867.77, 867.77)
    """
    rate = _number("discount_rate", discount_rate, 0.0, strict=False)
    fixed = _number("fixed_cost", fixed_cost, 0.0, strict=False)
    if timing not in _TIMINGS:
        raise ValueError(f"timing must be 'end', 'mid', 'start' or 'continuous', got {timing!r}")
    mine = None if mining_rate is None else _number("mining_rate", mining_rate, 0.0, strict=True)
    n = len(pits)
    values, destination, tons, names = _block_data(pits, tonnage, ore)
    plant: dict[str, float] | float | None
    if processing_rate is None:
        plant = None
    elif isinstance(processing_rate, Mapping):
        unknown = [k for k in processing_rate if k not in names]
        if unknown or not processing_rate:
            raise ValueError(
                f"processing_rate keys must be among the processes {list(names)}, got {list(processing_rate)}"
            )
        plant = {k: _number(f"processing_rate[{k!r}]", v, 0.0, strict=True) for k, v in processing_rate.items()}
    else:
        plant = _number("processing_rate", processing_rate, 0.0, strict=True)
    if mine is None and plant is None:
        raise ValueError("give mining_rate, processing_rate or both: they set how long each pit takes")
    phases: tuple[int, ...] | None = None
    if pushbacks is not None:
        phases = tuple(int(p) for p in pushbacks)
        if any(p != q for p, q in zip(phases, pushbacks, strict=True)) or any(p < 1 or p > n for p in phases):
            raise ValueError(f"pushbacks must be pit numbers from 1 to {n}, got {list(pushbacks)}")
        if any(b <= a for a, b in pairwise(phases)):
            raise ValueError(f"pushbacks must be increasing, got {list(pushbacks)}")

    # units: increment k (the blocks of pit k + 1 that are not in pit k) x bench (0 = top)
    numbers = np.asarray(pits.numbers)
    grid = pits.model.grid
    inside = numbers > 0
    nz = grid.nz
    unit = (numbers[inside].astype(np.int64) - 1) * nz + (nz - 1 - np.asarray(grid.iz)[inside].astype(np.int64))

    def per_unit(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.bincount(unit, weights=weights[inside], minlength=n * nz).reshape(n, nz)

    ton_u = per_unit(tons)
    ore_u = np.stack([per_unit(np.where(destination == p, tons, 0.0)) for p in range(len(names))])
    uses, caps = [], []
    if mine is not None:
        uses.append(ton_u)
        caps.append(mine)
    if isinstance(plant, dict):
        for name, r in plant.items():
            uses.append(ore_u[names.index(name)])
            caps.append(r)
    elif plant is not None:
        uses.append(ore_u.sum(axis=0))
        caps.append(plant)
    value_u = per_unit(values)
    zero_row = np.zeros((1, nz))
    prefix = [np.concatenate([zero_row, np.cumsum(a, axis=0)]) for a in [value_u, ton_u, *ore_u, *uses]]
    units = _Units(nz=nz, ton=ton_u, value=value_u, ore=ore_u, uses=uses, caps=caps, prefix=prefix)
    shift = _SHIFT.get(timing, 0.0)

    def npv_and_life(
        bounds: NDArray[np.float64], cash: NDArray[np.float64], used: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        k = np.arange(1, bounds.shape[1], dtype=np.float64)
        factor = (1.0 + rate) ** -(k - shift)
        width = np.diff(bounds, axis=1) > 0
        periods = width.sum(axis=1)
        last = np.maximum(periods, 1)
        rows = np.arange(bounds.shape[0])
        share = np.max((used[:, rows, last] - used[:, rows, last - 1]) / np.asarray(caps)[:, None], axis=0)
        share = np.where(periods > 0, np.minimum(share, 1.0), 0.0)
        life = np.maximum(periods - 1, 0) + share
        npv = (cash * factor).sum(axis=1)
        if fixed:
            full = np.where(k[None, :] < periods[:, None], 1.0, 0.0)
            full[rows, last - 1] = np.where(periods > 0, share, 0.0)
            npv = npv - fixed * (full * factor).sum(axis=1)
        return npv, life

    def evaluate_rows(phase_list: Sequence[int], finals: Sequence[int]) -> tuple[NDArray[Any], NDArray[Any]]:
        seqs = [_sequence(units, phase_list, f) for f in finals]
        width = len(seqs[0][0])
        if any(len(s[0]) != width for s in seqs):
            raise AssertionError("sequences of one group must have the same length")
        stacked = [np.stack([s[i] for s in seqs]) for i in range(len(seqs[0]))]
        value = stacked[0]
        use = stacked[2 + len(names) :]
        bounds, cash, used = _evaluate(value, use, caps)
        return npv_and_life(bounds, cash, used)

    def by_groups(phase_list: Sequence[int]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Every final pit: pits between two pushbacks share the sequence length, one vectorised fill."""
        npv, life = np.zeros(n), np.zeros(n)
        edges = [0, *phase_list, n]
        for a, b in pairwise(edges):
            finals = list(range(a + 1, b + 1))
            if finals:
                npv[a:b], life[a:b] = evaluate_rows(phase_list, finals)
        return npv, life

    if timing == "continuous":
        return _continuous_table(pits, units, n, phases, rate, mine, plant, fixed, names)

    # best case: one sequence (phase after phase), cut at the end of every pit
    value_seq = units.value.reshape(1, -1)
    use_seq = [w.reshape(1, -1) for w in uses]
    bounds, cash, _ = _evaluate(value_seq, use_seq, caps)
    ends = np.arange(1, n + 1, dtype=np.float64) * nz
    b = bounds[0]
    period = np.maximum(np.searchsorted(b, ends - 1e-9 * nz, side="left"), 1)  # the period where pit j ends
    zero = np.zeros((1, 1))
    cv = np.concatenate([zero, np.cumsum(value_seq, axis=1)], axis=1)[0]
    cums = [np.concatenate([zero, np.cumsum(np.maximum(w, 0.0), axis=1)], axis=1)[0] for w in use_seq]
    start = b[period - 1]
    factor_all = (1.0 + rate) ** -(np.arange(1, b.size, dtype=np.float64) - shift)
    done = np.concatenate([[0.0], np.cumsum(cash[0] * factor_all)])  # discounted cash of whole periods

    def at1(c: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
        return _at(np.broadcast_to(c, (x.size, c.size)), x)

    last_cash = at1(cv, ends) - at1(cv, start)
    npv_best = done[period - 1] + last_cash * factor_all[period - 1]
    share = np.max([(at1(c, ends) - at1(c, start)) / cap for c, cap in zip(cums, caps, strict=True)], axis=0)
    share = np.minimum(share, 1.0)
    life_best = (period - 1) + share
    if fixed:
        whole = np.concatenate([[0.0], np.cumsum(factor_all)])
        npv_best = npv_best - fixed * (whole[period - 1] + share * factor_all[period - 1])

    npv_worst, life_worst = by_groups(())
    value_c, ton_c, ore_c = np.cumsum(units.value, axis=0), np.cumsum(ton_u, axis=0), np.cumsum(ore_u, axis=1)
    out: dict[str, NDArray[Any]] = {
        "pit": np.arange(1, n + 1),
        "factor": np.asarray(pits.factors, dtype=np.float64),
        "tonnage": ton_c.sum(axis=1),
        "ore_tonnage": ore_c.sum(axis=(0, 2)),
        "value": value_c.sum(axis=1),
        "life_best": life_best,
        "npv_best": npv_best,
        "life_worst": life_worst,
        "npv_worst": npv_worst,
    }
    if phases is not None:
        out["npv_specified"], out["life_specified"] = by_groups(phases)
        out = {**{k: v for k, v in out.items() if k not in ("npv_specified", "life_specified")},
               "life_specified": out["life_specified"], "npv_specified": out["npv_specified"]}  # fmt: skip
    return PitByPit(
        table=out,
        discount_rate=rate,
        mining_rate=mine,
        processing_rate=plant,
        fixed_cost=fixed,
        timing=timing,
        pushbacks=phases,
        processes=names,
        _units=units,
    )


def _continuous_cost(fixed: float, rate: float, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Discounted cost of ``fixed`` per period paid at a constant rate from 0 to ``t``."""
    return fixed * t if rate == 0 else fixed * (1.0 - np.exp(-rate * t)) / rate


def _continuous_table(
    pits: Any,
    units: _Units,
    n: int,
    phases: tuple[int, ...] | None,
    rate: float,
    mine: float | None,
    plant: dict[str, float] | float | None,
    fixed: float,
    names: tuple[str, ...],
) -> PitByPit:
    """``pit_by_pit`` with every block discounted when it is mined."""
    nz = units.nz
    best = _continuous(units, range(1, n + 1), n, rate, fixed)
    ends = np.arange(1, n + 1) * nz - 1  # the last unit of every pit, in the phase order
    out: dict[str, NDArray[Any]] = {
        "pit": np.arange(1, n + 1),
        "factor": np.asarray(pits.factors, dtype=np.float64),
        "tonnage": np.cumsum(units.ton, axis=0).sum(axis=1),
        "ore_tonnage": np.cumsum(units.ore, axis=1).sum(axis=(0, 2)),
        "value": np.cumsum(units.value, axis=0).sum(axis=1),
        "life_best": best["time_end"][ends],
        "npv_best": best["npv_stop"][ends],
    }
    cases: list[tuple[str, tuple[int, ...]]] = [("worst", ())]
    if phases is not None:
        cases.append(("specified", phases))
    for case, plist in cases:
        npv, life = np.zeros(n), np.zeros(n)
        for final in range(1, n + 1):
            s = _continuous(units, plist, final, rate, fixed)
            npv[final - 1], life[final - 1] = s["npv_stop"][-1], s["time_end"][-1]
        out[f"life_{case}"], out[f"npv_{case}"] = life, npv
    return PitByPit(table=out, discount_rate=rate, mining_rate=mine, processing_rate=plant, fixed_cost=fixed,
                    timing="continuous", pushbacks=phases, processes=names, _units=units)  # fmt: skip
