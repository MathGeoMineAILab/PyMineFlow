"""Ultimate pit OES: the optimal extraction sequence of the final pit, block by block.

The stage that strategic mine-planning software runs after the (top-down discounted) ultimate pit and
before the pushbacks. It is built on the nested pits, which already rank the blocks by value:

1. **Phases.** Every nested pit with new blocks is a phase (or the pits you choose), mined in order.
2. **Inside a phase**, the benches from the top down; in every phase-bench the block of highest key first,
   then repeatedly the remaining block that maximises ``key - c * distance`` to the last block
   (*proximity values*): low-value blocks next to rich ones move up and the sequence grows in contiguous
   groups. The key is the block value (``search="proximity"``) or its *lookahead value*
   (``search="lookahead"``): its value plus the discounted value of the pit blocks in its downward cone,
   ``lookahead_benches`` deep. The nested pits are closed and every precedence points upwards, so a
   block's predecessors are in an earlier phase or on a higher bench: the slopes hold by construction.
   ``bench_order=False`` instead takes any *available* block of the phase (every slope predecessor
   mined), whatever its bench: it goes down early where the key is high, at the price of an irregular
   sequence.

3. **Schedule.** The sequence is mined period after period at the given capacities (mining, processing,
   or both), blocks split between periods when they do not fit, and every period's cash is discounted.
   The NPV of the sequence, and the NPV had mining stopped after any block, follow.

``OES.check`` validates a sequence independently (completeness, order, slopes against the precedence
pattern, value, capacities). The ordering, the lookahead values, the schedule and the check run in C++.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._mineflow import _oes_check, _oes_lookahead, _oes_merge_small, _oes_order, _oes_order_free, _oes_schedule
from ._report import check_kind, describe, kinds, phase_pits, phase_report, report_text
from .npv import _SHIFT, _TIMINGS, PitByPit, _block_data, _continuous_cost, _number, pit_by_pit
from .plot import _surface_cells

_FACTOR_LABEL = {"revenue": "revenue factor", "price": "price factor", "profit": "profit factor"}

__all__ = ["OES", "OESCheck", "ultimate_pit_oes"]


@dataclass(frozen=True)
class OESCheck:
    """Validation of an OES (see ``OES.check``); ``summary()`` lists every item and ``ok`` sums them up.

    complete : every block of the final pit is in the sequence exactly once, and no other block.
    ordered : phases never go back and, with the bench order, inside a phase the benches go from the top down.
    violations : blocks mined before one of their slope predecessors (0 = the slopes hold), checked
        independently against the grid's precedence pattern (constraints through empty cells included);
        ``first_violation`` is the cell of the first one (-1 = none).
    value_error : |value of the sequence - value of the final pit| (undiscounted, at factor 1), and
        ``value_scale`` the sum of |values| of the final pit.
    capacity_use : highest use of a capacity in a period, as a fraction of the capacity (<= 1).
    best_case : per phase, (NPV of the sequence at the end of the phase - pit-by-pit best case NPV of that
        phase's pit) / |best case|; the best case mines the same phases with every bench as a whole (empty
        when the phases are not every nested pit, or when the continuity moved blocks between phases).
    """

    complete: bool
    ordered: bool
    violations: int
    first_violation: int
    value_error: float
    value_scale: float
    capacity_use: float
    best_case: NDArray[np.float64] = field(repr=False)
    bench_order: bool = True

    @property
    def ok(self) -> bool:
        """Every check passed (the best case comparison is information, not a check)."""
        return (self.complete and self.ordered and self.violations == 0 and self.capacity_use <= 1 + 1e-9
                and self.value_error <= 1e-9 * max(1.0, self.value_scale))  # fmt: skip

    def summary(self) -> str:
        def mark(good: bool) -> str:
            return "ok  " if good else "FAIL"

        b = self.best_case
        lines = [
            f"{mark(self.complete)} complete: every block of the final pit exactly once",
            f"{mark(self.ordered)} ordered: phase by phase"
            + (", bench by bench from the top" if self.bench_order else ""),
            f"{mark(self.violations == 0)} slopes: {self.violations:,} blocks mined before a predecessor"
            + (f" (first at cell {self.first_violation})" if self.violations else ""),
            f"{mark(self.value_error <= 1e-9 * max(1.0, self.value_scale))} value: the sequence holds the value of "
            f"the final pit (difference {self.value_error:.3g})",
            f"{mark(self.capacity_use <= 1 + 1e-9)} capacities: highest use in a period {self.capacity_use:.9f} "
            "of the capacity",
        ]
        if b.size:
            low, high = float(np.min(b)), float(np.max(b))
            lines.append(
                f"info NPV at the end of each phase vs the pit-by-pit best case: {low:+.4%} to {high:+.4%} "
                "(same phases; the best case mines every bench as a whole)"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class _Frames:
    """The frames of an animation as cumulative "nested pits" (what ``plot._NestedScene`` draws)."""

    numbers: NDArray[np.int32]
    model: Any
    count: int

    def __len__(self) -> int:
        return self.count


@dataclass(frozen=True)
class OES:
    """The ultimate pit OES (see ``ultimate_pit_oes``).

    Per block (input rows; 0 or -1 outside the final pit): ``rank`` (position in the sequence, from 0),
    ``phase`` (from 1), ``period`` (first period in which it is mined, from 1), ``discounted`` (its
    discounted cash) and ``key`` (the value it was ordered by). Per position of the sequence:
    ``order`` (the row), ``npv_curve`` (NPV if mining stopped after that block) and ``time`` (periods
    elapsed). ``periods`` and ``phase_table()`` give the schedule period by period and phase by phase.
    """

    pits: Any = field(repr=False)
    order: NDArray[np.int64] = field(repr=False)
    phase: NDArray[np.int64] = field(repr=False)
    phase_pits: NDArray[np.int64]
    key: NDArray[np.float64] = field(repr=False)
    period: NDArray[np.int64] = field(repr=False)
    discounted: NDArray[np.float64] = field(repr=False)
    npv_curve: NDArray[np.float64] = field(repr=False)
    time: NDArray[np.float64] = field(repr=False)
    periods: dict[str, NDArray[Any]] = field(repr=False)
    capacity_use: list[NDArray[np.float64]] = field(repr=False)
    capacities: list[float] = field(repr=False)
    values: NDArray[np.float64] = field(repr=False)
    tonnage: NDArray[np.float64] = field(repr=False)
    destination: NDArray[np.int64] = field(repr=False)
    processes: tuple[str, ...]
    search: str
    bench_order: bool
    min_component: int
    moved_groups: int
    moved_blocks: int
    proximity: float
    proximity_weight: float
    lookahead_benches: int
    lookahead_radius: float
    lookahead_discount: float
    discount_rate: float
    mining_rate: float | None
    processing_rate: float | dict[str, float] | None
    fixed_cost: float
    timing: str
    seconds: dict[str, float] = field(repr=False)

    @property
    def npv(self) -> float:
        """NPV of the whole sequence (the final pit mined to the end)."""
        return float(self.npv_curve[-1]) if self.npv_curve.size else 0.0

    @property
    def rank(self) -> NDArray[np.int64]:
        """Per block: its position in the sequence (from 0), -1 outside the final pit."""
        out = np.full(self.phase.size, -1, dtype=np.int64)
        out[self.order] = np.arange(self.order.size)
        return out

    def best_stop(self) -> int:
        """Position in the sequence after which mining should stop for the highest NPV (the last one when
        the whole final pit pays): a smaller pit than the final one can have a higher NPV."""
        return int(np.argmax(self.npv_curve)) if self.npv_curve.size else -1

    def phase_table(self) -> dict[str, NDArray[Any]]:
        """One row per phase: ``phase``, ``pit`` (the nested pit that closes it), ``factor``, ``blocks``,
        ``tonnage``, ``ore_tonnage``, ``value`` (undiscounted), ``start`` and ``end`` (periods elapsed
        when it starts and ends) and ``npv`` (NPV had mining stopped at its end)."""
        n = self.phase_pits.size
        ph = self.phase[self.order]
        ends = np.searchsorted(ph, np.arange(1, n + 1), side="right") - 1  # last position of each phase
        starts = np.searchsorted(ph, np.arange(1, n + 1), side="left")
        rows = self.phase > 0

        def per_phase(w: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.bincount(self.phase[rows], weights=w[rows], minlength=n + 1)[1:]

        before = np.concatenate([[0.0], self.time])
        factors = np.asarray(self.pits.factors, dtype=np.float64)
        return {
            "phase": np.arange(1, n + 1),
            "pit": self.phase_pits,
            "factor": factors[self.phase_pits - 1],
            "blocks": np.bincount(self.phase[rows], minlength=n + 1)[1:],
            "tonnage": per_phase(self.tonnage),
            "ore_tonnage": per_phase(np.where(self.destination >= 0, self.tonnage, 0.0)),
            "value": per_phase(self.values),
            "start": before[starts],
            "end": self.time[ends],
            "npv": self.npv_curve[ends],
        }

    def spatial(self) -> dict[str, float]:
        """Contiguity of the order: mean distance between consecutive blocks of one phase (on one bench with the
        bench order; in 3-D otherwise), in block widths, and the fraction of those steps longer than 1.5 widths."""
        g = self.pits.model.grid
        o = self.order
        if o.size < 2:
            return {"mean_step": 0.0, "jumps": 0.0}
        dx, dy, dz = g.block_size
        x = np.asarray(g.ix, dtype=np.float64)[o] * dx
        y = np.asarray(g.iy, dtype=np.float64)[o] * dy
        z = np.asarray(g.iz, dtype=np.float64)[o] * dz
        same = self.phase[o][1:] == self.phase[o][:-1]
        if self.bench_order:
            same &= z[1:] == z[:-1]
        if not same.any():
            return {"mean_step": 0.0, "jumps": 0.0}
        step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)[same] / float(np.sqrt(dx * dy))
        return {"mean_step": float(step.mean()), "jumps": float((step > 1.5).mean())}

    def summary(self) -> str:
        """A detailed report: the method, the NPV, the best place to stop, phase by phase and period by period."""
        t = self.phase_table()
        p = self.periods
        n_benches = len(np.unique(np.asarray(self.pits.model.grid.iz)[self.order])) if self.order.size else 0
        if self.processing_rate is None:
            plant = "unlimited"
        elif isinstance(self.processing_rate, dict):
            plant = ", ".join(f"{k} {v:,.10g}" for k, v in self.processing_rate.items())
        else:
            plant = f"{self.processing_rate:,.10g}"
        mine = "unlimited" if self.mining_rate is None else f"{self.mining_rate:,.10g}"
        if self.search == "proximity":
            key = "block value"
        else:
            key = (f"lookahead value ({self.lookahead_benches} benches, cone radius {self.lookahead_radius:,.4g} "
                   f"per bench, discount {self.lookahead_discount:.4%} per bench)")  # fmt: skip
        walk = f"key - {self.proximity_weight:,.6g} x distance to the last block (proximity {self.proximity:g})"
        if self.bench_order:
            order = (f"Order: phase by phase, bench by bench from the top; in every phase-bench the block of highest "
                     f"{key} first, then the block that maximises {walk}")  # fmt: skip
        else:
            order = ("Order: phase by phase; inside a phase the available block (every slope predecessor mined) of "
                     f"highest {key} first, then the available block that maximises {walk}, "
                     "whatever its bench")  # fmt: skip
        life = float(self.time[-1]) if self.time.size else 0.0
        if self.timing == "continuous":
            cash = "every block discounted when it is mined, exp(-rate x time)"
        else:
            cash = f"cash at the {self.timing} of each period"
        lines = [
            f"Ultimate pit OES: {self.order.size:,} blocks, {t['phase'].size} phases, {n_benches} benches",
            order,
            f"Schedule: discount rate {self.discount_rate:.2%} per period, {cash}; capacities per period: mining "
            f"{mine}, processing {plant}; fixed cost {self.fixed_cost:,.10g} per period",
            f"NPV of the whole sequence {self.npv:,.0f} over {life:.2f} periods "
            f"(undiscounted value {self.values[self.order].sum():,.0f})",
        ]
        stop = self.best_stop()
        if 0 <= stop < self.order.size - 1:
            ph = int(self.phase[self.order[stop]])
            lines.append(
                f"Highest NPV if mining stops early: {self.npv_curve[stop]:,.0f} after block {stop + 1:,} of "
                f"{self.order.size:,} (phase {ph}, pit {int(self.phase_pits[ph - 1])}, {self.time[stop]:.2f} periods)"
            )
        else:
            lines.append("Highest NPV: mining the whole final pit")
        steps = self.spatial()
        lines.append(
            f"Spatial: consecutive blocks of a {'phase-bench' if self.bench_order else 'phase'} are "
            f"{steps['mean_step']:.2f} block widths apart on average; {steps['jumps']:.2%} of the steps jump more "
            "than 1.5 block widths"
        )
        if self.min_component > 1:
            lines.append(
                f"Continuity: {self.moved_groups:,} isolated groups of fewer than {self.min_component:,} blocks "
                f"({self.moved_blocks:,} blocks) moved to the phase that first needs them"
            )
        lines.append("Times: " + ", ".join(f"{k} {v:.2f} s" for k, v in self.seconds.items()))
        head = (f"{'phase':>5} {'pit':>5} {'factor':>8} {'blocks':>9} {'tonnage':>15} {'ore':>15} {'value':>17} "
                f"{'start':>7} {'end':>7} {'NPV if stop':>17}")  # fmt: skip
        lines += ["", head, "-" * len(head)]
        for i in range(t["phase"].size):
            lines.append(
                f"{t['phase'][i]:>5} {t['pit'][i]:>5} {t['factor'][i]:>8.4f} {t['blocks'][i]:>9,} "
                f"{t['tonnage'][i]:>15,.0f} {t['ore_tonnage'][i]:>15,.0f} {t['value'][i]:>17,.0f} "
                f"{t['start'][i]:>7.2f} {t['end'][i]:>7.2f} {t['npv'][i]:>17,.0f}"
            )
        head = (f"{'period':>6} {'tonnage':>15} {'ore':>15} {'value':>17} {'factor':>8} {'fixed':>12} "
                f"{'discounted':>17}")  # fmt: skip
        lines += ["", head, "-" * len(head)]
        for i in range(p["period"].size):
            lines.append(
                f"{p['period'][i]:>6} {p['tonnage'][i]:>15,.0f} {p['ore_tonnage'][i]:>15,.0f} {p['value'][i]:>17,.0f} "
                f"{p['discount_factor'][i]:>8.4f} {p['fixed_cost'][i]:>12,.0f} {p['discounted'][i]:>17,.0f}"
            )
        return "\n".join(lines)

    def table(self) -> dict[str, NDArray[Any]]:
        """The sequence, one row per block in extraction order: ``rank``, ``row`` (input row), the block's
        indices, ``phase``, ``pit``, ``period``, ``value``, ``tonnage``, ``destination``, ``key``,
        ``discounted``, ``npv`` (if mining stopped after it) and ``time``."""
        g = self.pits.model.grid
        o = self.order
        names = np.asarray([*self.processes, "waste"])
        return {
            "rank": np.arange(o.size),
            "row": o,
            "IX": np.asarray(g.ix)[o] + g.base[0],
            "IY": np.asarray(g.iy)[o] + g.base[1],
            "IZ": np.asarray(g.iz)[o] + g.base[2],
            "phase": self.phase[o],
            "pit": np.asarray(self.pits.numbers)[o],
            "period": self.period[o],
            "value": self.values[o],
            "tonnage": self.tonnage[o],
            "destination": names[self.destination[o]],
            "key": self.key[o],
            "discounted": self.discounted[o],
            "npv": self.npv_curve,
            "time": self.time,
        }

    def to_csv(self, path: str | Path, pit: str | None = None) -> Path:
        """Write a CSV. ``pit=None`` (default): the sequence, one row per block in extraction order (``table()``).
        ``pit="whole"``: the whole model with the answer of every block; ``pit="final"``: only the blocks of the
        final pit, with the same columns (see ``model_table``)."""
        from .readers import write_table

        if pit is None:
            return write_table(self.table(), path)
        if pit not in ("whole", "final"):
            raise ValueError(f"pit must be 'whole', 'final' or None, got {pit!r}")
        return write_table(self.model_table(final_only=pit == "final"), path)

    def model_table(self, final_only: bool = False) -> dict[str, NDArray[Any]]:
        """The model with the answer of every block: the model's own columns, ``VALUE`` and ``DESTINATION`` (at
        factor 1, unless the model has them), ``PIT_NUMBER`` (1 = smallest nested pit, 0 = outside),
        ``FINAL_PIT`` (1 = inside the final pit), ``OES_RANK`` (position in the sequence from 0, -1 outside),
        ``OES_PHASE`` and ``OES_PERIOD`` (0 outside). ``final_only=True``: only the blocks of the final pit."""
        table = dict(self.pits.table)
        names = np.asarray([*self.processes, "waste"])
        table.setdefault("VALUE", self.values)
        table.setdefault("DESTINATION", names[self.destination])
        new = {"PIT_NUMBER": np.asarray(self.pits.numbers), "FINAL_PIT": (self.phase > 0).astype(np.int8),
               "OES_RANK": self.rank, "OES_PHASE": self.phase, "OES_PERIOD": self.period}  # fmt: skip
        for name, column in new.items():
            if name in table:
                raise ValueError(f"the model already has a column {name!r}")
            table[name] = column
        if final_only:
            keep = self.phase > 0
            table = {name: np.asarray(v)[keep] for name, v in table.items()}
        return table

    def cases(self) -> PitByPit:
        """Every nested pit as the final pit, mined with the discount rate, capacities, fixed cost and timing of
        the sequence: the best case (the phases in order) and the worst case (bench by bench) of
        ``pymineflow.pit_by_pit``."""
        ore = None if self.pits.economics is not None else self.destination >= 0
        return pit_by_pit(self.pits, discount_rate=self.discount_rate, mining_rate=self.mining_rate,
                          processing_rate=self.processing_rate, fixed_cost=self.fixed_cost, timing=self.timing,
                          tonnage=self.tonnage, ore=ore)  # fmt: skip

    def report_table(self, kind: str = "cumulative", *, cases: PitByPit | None = None) -> dict[str, NDArray[Any]]:
        """One row per phase of the nested pits, as ``NestedPits.report_table`` but with the discount rate,
        capacities and timing of the sequence, and the sequence itself.

        ``kind="cumulative"`` (default): every phase as the final pit: ``phase``, ``factor``, ``profit``,
        ``profit_per_t``, ``npv_best``, ``npv_worst``, ``oes_npv`` (NPV of the sequence stopped at the end of
        the phase), ``oes_end`` (periods elapsed then), ``rock``, ``ore``, ``strip_ratio`` and ``waste``.
        ``kind="incremental"``: what every phase adds: ``npv`` (best case) and ``oes_npv`` are increments, and
        ``oes_start`` and ``oes_end`` say when the sequence mines the phase. The OES columns are empty (NaN) in
        the rows of a nested pit that does not end a phase of the sequence (merged with ``phases=``, or whose
        blocks all wait for a later phase for continuity). With continuity (``min_component``) a phase of the
        sequence may hold a few blocks of earlier pits or lack a few of its own, so the OES columns of a row
        describe the phase of the sequence that ends there, not exactly the blocks of the row. ``cases``:
        ``cases()`` if already computed."""
        check_kind(kind)
        cases = self.cases() if cases is None else cases
        numbers = np.asarray(self.pits.numbers)
        factors = np.asarray(self.pits.factors, dtype=np.float64)
        tons = self.tonnage if bool(np.any(self.tonnage > 0)) else None
        t = phase_report(numbers, factors, self.values, tons, self.destination >= 0, cases.table, kind)
        rows = phase_pits(numbers, len(self.pits))  # the pit that closes every row
        p = self.phase_table()
        at = np.minimum(np.searchsorted(p["pit"], rows), p["pit"].size - 1)
        # the rows that end a phase of the sequence: not a pit merged into the next one (phases=) or whose blocks
        # all wait for a later phase (min_component)
        match = p["pit"][at] == rows

        def column(values: NDArray[np.float64]) -> NDArray[np.float64]:
            out = np.full(rows.size, np.nan)
            out[match] = values[at[match]]
            return out

        oes: dict[str, NDArray[Any]] = {}
        if kind == "cumulative":
            oes["oes_npv"] = column(p["npv"])
        else:
            oes["oes_npv"] = column(np.diff(p["npv"], prepend=0.0))
            oes["oes_start"] = column(p["start"])
        oes["oes_end"] = column(p["end"])
        keys = list(t)
        split = keys.index("rock") if "rock" in keys else len(keys)
        return {**{k: t[k] for k in keys[:split]}, **oes, **{k: t[k] for k in keys[split:]}}

    def report(self, kind: str = "cumulative") -> str:
        """The phases as text, one line each (see ``report_table``): the best and worst case of every phase and
        the sequence, with the same discount rate, capacities and timing."""
        t = self.report_table(kind)
        when = {"continuous": "the moment each block is mined, exp(-rate x time)", "end": "the end of each period",
                "mid": "mid-period", "start": "the start of each period"}[self.timing]  # fmt: skip
        lines = [
            f"Ultimate pit OES ({self.search}): {len(t['phase'])} phases ({describe(kind)})",
            f"NPV: {self.discount_rate * 100:.6g} % per period, cash at {when}; best case = the phases in order, "
            "worst case = the final pit bench by bench, OES = the sequence; OES start and end in periods",
        ]
        return report_text(t, lines)

    def plot_cases(self, path: str | Path | None = None, *, show: bool | None = None, **kwargs: Any) -> Any:
        """The best case against the worst case NPV of every nested pit as the final pit (``cases().plot``,
        same options): saved to ``path`` (PNG, SVG, PDF...), and shown in a window by default when there is no
        path. Returns the figure."""
        kwargs.setdefault("factor_label", _FACTOR_LABEL[self.pits.scale])
        return self.cases().plot(path, show=show, **kwargs)

    def plot(
        self,
        color_by: str = "rank",
        *,
        frames: int = 200,
        remaining: bool = False,
        surface: bool = False,
        slider: bool = True,
        view: str = "top",
        rotation: float = 0.0,
        title: str | None = None,
        outline: bool = False,
        axes: bool = True,
        cmap: str = "jet",
        screenshot: str | Path | None = None,
        window_size: tuple[int, int] = (1280, 800),
        show: bool | None = None,
    ) -> Any:
        """3-D view of the sequence (jet, with its legend) and a slider that walks along it, as ``NestedPits.plot``.

        The sequence is cut into ``frames`` steps of equal time. ``remaining=False`` draws what is mined (the pit
        growing); ``remaining=True`` the whole final pit, from which the slider removes what is mined, what is left
        staying on screen; ``surface=True`` the ground surface, which the slider digs along the sequence (coloured
        by elevation; best with ``view="iso"``). ``color_by``: ``"rank"`` (the position in the sequence),
        ``"period"`` or ``"phase"``. ``screenshot`` saves the image of the whole pit; ``show`` opens the window
        (by default only without a ``screenshot``). The other options are those of ``plot_pit``. Returns the
        ``pyvista.Plotter``."""
        if show is None:
            show = screenshot is None
        options: dict[str, Any] = {"frames": frames, "color_by": color_by, "remaining": remaining,
                                   "surface": surface, "view": view,
                                   "rotation": rotation, "title": title, "outline": outline, "axes": axes,
                                   "cmap": cmap, "window_size": window_size}  # fmt: skip
        if screenshot is not None:
            scene = self._scene(off_screen=True, **options)
            if surface:
                scene.show(scene.high)  # the image: the final pit dug
            scene.plotter.show(screenshot=str(screenshot), auto_close=False)
            if not show:
                return scene.plotter
            scene.plotter.close()
        scene = self._scene(off_screen=not show, **options)
        plotter = scene.plotter
        if slider and scene.high > scene.low:
            plotter.add_slider_widget(
                scene.show, [scene.low, scene.high], value=scene.low if scene.remaining else scene.high,
                title="step", fmt="%.0f", pointa=(0.25, 0.06), pointb=(0.70, 0.06), style="modern",
                interaction_event="always",
            )  # fmt: skip
        if show:
            plotter.show()
        return plotter

    def plot_npv(self, path: str | Path | None = None, *, show: bool | None = None) -> Any:
        """NPV had mining stopped after every block, against the tonnage mined, with the ends of the phases and
        the best place to stop. ``path`` saves the image; ``show`` opens the window (by default only without
        ``path``). Returns the figure."""
        from matplotlib.figure import Figure
        from matplotlib.ticker import FuncFormatter

        if show is None:
            show = path is None
        if show:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(11, 6), dpi=120)
        else:
            fig = Figure(figsize=(11, 6), dpi=120)
        ax = fig.subplots()
        tons = np.cumsum(self.tonnage[self.order]) / 1e6
        scale, unit = (1e6, "M") if np.max(np.abs(self.npv_curve), initial=0) >= 1e6 else (1.0, "")
        ax.plot(tons, self.npv_curve / scale, color="#1565c0", lw=1.6, label="NPV if mining stops here")
        t = self.phase_table()
        ends = np.searchsorted(self.phase[self.order], t["phase"], side="right") - 1
        ax.scatter(tons[ends], t["npv"] / scale, s=12, color="#c62828", zorder=3, label="end of a phase")
        k = self.best_stop()
        if k >= 0:
            ax.scatter([tons[k]], [self.npv_curve[k] / scale], s=90, marker="o", color="#1565c0", edgecolor="black",
                       zorder=4, label=f"highest NPV {self.npv_curve[k] / scale:,.1f}{unit}")  # fmt: skip
        ax.set_xlabel("tonnage mined along the sequence (millions)")
        ax.set_ylabel(f"NPV ({unit})" if unit else "NPV")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(True, alpha=0.3)
        rate = f"{self.discount_rate * 100:.6g} %"
        ax.set_title(f"Ultimate pit OES ({self.search}): discount rate {rate}, NPV {self.npv:,.0f}", fontsize=11)
        low, high = (
            min(0.0, float(np.min(self.npv_curve, initial=0.0))),
            max(0.0, float(np.max(self.npv_curve, initial=0.0))),
        )
        span = (high - low) / scale if high > low else 1.0
        ax.set_ylim(low / scale - 0.05 * span, high / scale + 0.08 * span)
        if tons.size:
            ax.set_xlim(0.0, float(tons[-1]) * 1.01 if tons[-1] > 0 else 1.0)
        ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
        # below the chart: it never hides the curve, whatever its shape
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=9, frameon=False)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def animate(
        self,
        path: str | Path = "oes.gif",
        *,
        frames: int = 60,
        color_by: str = "period",
        remaining: bool = False,
        surface: bool = False,
        view: str = "top",
        rotation: float = 0.0,
        title: str | None = None,
        outline: bool = False,
        axes: bool = True,
        cmap: str = "jet",
        seconds_per_frame: float = 0.12,
        hold: float = 2.5,
        window_size: tuple[int, int] = (960, 600),
    ) -> Path:
        """An animated GIF of the final pit mined along the sequence, in ``frames`` steps of equal time, with the
        time, the blocks mined and the NPV so far: what is mined growing or, with ``remaining=True``, the whole pit
        from which the mined part disappears, or with ``surface=True`` the ground surface dug along the sequence.
        The options are those of ``plot``. Returns the path."""
        from .plot import _write_gif

        scene = self._scene(frames=frames, color_by=color_by, remaining=remaining, surface=surface, view=view,
                            rotation=rotation,
                            title=title, outline=outline, axes=axes, cmap=cmap, window_size=window_size,
                            off_screen=True)  # fmt: skip
        return _write_gif(scene, scene.steps(), Path(path), seconds_per_frame, hold)

    def _scene(
        self,
        *,
        frames: int,
        color_by: str,
        remaining: bool,
        view: str,
        surface: bool = False,
        rotation: float,
        title: str | None,
        outline: bool,
        axes: bool,
        cmap: str,
        window_size: tuple[int, int],
        off_screen: bool,
    ) -> Any:
        """The sequence as a cumulative scene: every block belongs to the step of equal time in which it is mined."""
        from .plot import _NestedScene, _SurfaceScene

        if isinstance(frames, bool) or not isinstance(frames, (int, np.integer)) or frames < 1:
            raise ValueError(f"frames must be an integer >= 1, got {frames!r}")
        if color_by not in ("rank", "period", "phase"):
            raise ValueError(f"color_by must be 'rank', 'period' or 'phase', got {color_by!r}")
        frames = int(frames)
        n = self.order.size
        life = float(self.time[-1])
        if life > 0:
            step = np.clip(np.ceil(self.time / life * frames - 1e-9), 1, frames).astype(np.int32)
        else:  # nothing takes any time (no tonnage): one step
            step = np.full(n, frames, dtype=np.int32)
        numbers = np.zeros(self.phase.size, dtype=np.int32)
        numbers[self.order] = step  # the step in which every block is mined
        last = np.searchsorted(step, np.arange(frames + 1), side="right") - 1  # last position mined after each step
        clock = np.arange(frames + 1) / frames * life
        # every block in its own colour: its position in the sequence, its period or its phase
        colors = {"rank": self.rank, "period": self.period, "phase": self.phase}[color_by].astype(np.float64)
        npv = self.npv_curve

        def label(k: int) -> str:
            j = int(last[k])
            mined = j + 1
            text = f"period {clock[k]:.2f} of {life:.2f}: {mined:,} of {n:,} blocks mined"
            if remaining or surface:
                text += f", {n - mined:,} still to mine"
            return text + f", NPV so far {npv[j] if j >= 0 else 0.0:,.0f}"

        kind = _SurfaceScene if surface else _NestedScene
        return kind(
            _Frames(numbers=numbers, model=self.pits.model, count=frames), view=view, rotation=rotation,
            title=f"Ultimate pit OES ({self.search}): NPV {self.npv:,.0f}" if title is None else title,
            outline=outline, axes=axes, cmap=cmap, window_size=window_size, off_screen=off_screen,
            remaining=remaining, legend=f"OES {color_by}", colors=colors, label=label,
        )  # fmt: skip

    def export(
        self,
        folder: str | Path,
        *,
        report: str | None = "both",
        pit: str | None = "whole",
        chart: bool = True,
        gif: bool = True,
        details: bool = False,
        view: str = "top",
        rotation: float = 0.0,
    ) -> list[Path]:
        """Write the results into ``folder`` and return the files written. Every part is optional:

        - ``report``: ``"both"`` (default: ``report_cumulative.csv`` and ``report_incremental.csv``),
          ``"cumulative"``, ``"incremental"`` or ``None`` (see ``report_table``);
        - ``pit``: ``"whole"`` (default: ``model.csv``, the whole model with the answer of every block),
          ``"final"`` (``final_pit.csv``, only the blocks of the final pit), ``"both"`` or ``None`` (see
          ``model_table``);
        - ``chart``: ``npv.png``, the best case against the worst case NPV of every phase (``plot_cases``);
        - ``gif``: ``oes.gif``, the whole final pit mined away in the order of the sequence, coloured by the
          position in it (``animate(remaining=True, color_by="rank")``, with ``view`` and ``rotation``), and
          ``oes_surface.gif``, the ground surface dug along the sequence (``animate(surface=True)``; not for a
          vertical section, which has no surface), both in the same ``view`` (a plan view by default);
        - ``details``: also ``oes.csv`` (the sequence block by block), ``oes_periods.csv``, ``oes_report.txt``
          (the summary and the check) and ``oes_npv.png`` (the NPV along the sequence)."""
        from .readers import write_table

        reports = kinds(report)
        if pit not in ("whole", "final", "both", None):
            raise ValueError(f"pit must be 'whole', 'final', 'both' or None, got {pit!r}")
        if view not in ("iso", "below", "top", "side"):  # checked before anything is written
            raise ValueError(f"view must be 'iso', 'below', 'top' or 'side', got {view!r}")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        cases = self.cases() if reports or chart else None
        for kind in reports:
            written.append(write_table(self.report_table(kind, cases=cases), folder / f"report_{kind}.csv"))
        if pit in ("whole", "both"):
            written.append(self.to_csv(folder / "model.csv", pit="whole"))
        if pit in ("final", "both"):
            written.append(self.to_csv(folder / "final_pit.csv", pit="final"))
        if chart and cases is not None:
            cases.plot(folder / "npv.png", factor_label=_FACTOR_LABEL[self.pits.scale], show=False)
            written.append(folder / "npv.png")
        if gif:
            written.append(self.animate(folder / "oes.gif", remaining=True, color_by="rank", view=view,
                                        rotation=rotation))  # fmt: skip
            if _surface_cells(self.pits.model.grid).any():  # a vertical section (one row of columns) has none
                written.append(self.animate(folder / "oes_surface.gif", surface=True, view=view, rotation=rotation))
        if details:
            written.append(self.to_csv(folder / "oes.csv"))
            written.append(write_table(self.periods, folder / "oes_periods.csv"))
            text = folder / "oes_report.txt"
            text.write_text(self.summary() + "\n\nCHECK\n" + self.check().summary() + "\n", encoding="utf-8")
            written.append(text)
            self.plot_npv(folder / "oes_npv.png", show=False)
            written.append(folder / "oes_npv.png")
        return written

    def check(self, threads: int = 0) -> OESCheck:
        """Validate the sequence (see ``OESCheck``): completeness, order, slopes (an independent check against
        the precedence pattern), value, capacities and the comparison with the pit-by-pit best case."""
        pits = self.pits
        model = pits.model
        g = model.grid
        n = len(pits)
        numbers = np.asarray(pits.numbers)
        in_pit = (numbers >= 1) & (numbers <= n)
        o = self.order
        complete = bool(o.size == in_pit.sum() and np.unique(o).size == o.size and in_pit[o].all())
        ph, z = self.phase[o], np.asarray(g.iz)[o]
        ordered = bool(np.all(np.diff(ph) >= 0))
        if self.bench_order:
            ordered = ordered and bool(np.all((np.diff(ph) > 0) | (np.diff(z) <= 0)))
        linear = np.asarray(g.linear)
        rank_cells = np.full(g.n_cells, -1, dtype=np.int64)
        rank_cells[linear] = -2
        rank_cells[linear[o]] = np.arange(o.size)
        pit = model._pit
        offsets = [pit.offsets(d) for d in range(pit.n_domains)]
        violations, first = _oes_check(g.nx, g.ny, g.nz, rank_cells, offsets, model._cell_domains(), threads)
        value_error = abs(float(self.values[o].sum()) - float(self.values[in_pit].sum()))
        use = max((float(np.max(u, initial=0.0)) / c for u, c in zip(self.capacity_use, self.capacities, strict=True)),
                  default=0.0)  # fmt: skip
        best = np.zeros(0)
        every_pit = np.flatnonzero(np.bincount(numbers[in_pit], minlength=n + 1)[1:]) + 1
        if np.array_equal(self.phase_pits, every_pit) and self.moved_blocks == 0:
            ore = None if pits.economics is not None else self.destination >= 0
            ref = pit_by_pit(pits, discount_rate=self.discount_rate, mining_rate=self.mining_rate,
                             processing_rate=self.processing_rate, fixed_cost=self.fixed_cost, timing=self.timing,
                             tonnage=self.tonnage, ore=ore)  # fmt: skip
            expected = ref.table["npv_best"][self.phase_pits - 1]
            best = (self.phase_table()["npv"] - expected) / np.where(np.abs(expected) > 0, np.abs(expected), 1.0)
        scale = float(np.abs(self.values[in_pit]).sum())
        return OESCheck(complete, ordered, int(violations), int(first), value_error, scale, use, best, self.bench_order)


def ultimate_pit_oes(
    pits: Any,
    *,
    search: str = "proximity",
    proximity: float = 1.0,
    bench_order: bool | None = None,
    lookahead_benches: int = 6,
    lookahead_slope: float | None = None,
    lookahead_discount: float | None = None,
    lookahead_scope: str = "pit",
    phases: Sequence[int] | None = None,
    min_component: int | None = None,
    discount_rate: float | None = None,
    mining_rate: float | None = None,
    processing_rate: float | Mapping[str, float] | None = None,
    fixed_cost: float = 0.0,
    timing: str = "continuous",
    tonnage: str | float | ArrayLike | None = None,
    ore: str | ArrayLike | None = None,
    threads: int = 0,
) -> OES:
    """The ultimate pit OES of nested pits: a block-by-block extraction sequence of the final pit and its NPV.

    Parameters
    ----------
    pits : the result of ``nested_pits`` (with top-down discounting, if wanted: ``discount_rate=`` and
        ``benches_per_year=``). Its last pit is the final pit; the blocks are valued at factor 1.
    search : ``"proximity"`` (default): the blocks of every phase-bench ordered by their value and their
        distance to the last block. ``"lookahead"``: by their lookahead value (the block plus the discounted
        value of the pit blocks in its downward cone) and their distance.
    proximity : weight of the distance, >= 0. A step of one block width costs ``proximity`` times the mean
        absolute key of the pit blocks whose key is not zero; 0 orders by key alone, a large value walks to the
        nearest block (most contiguous). The distance is horizontal with the bench order and 3-D otherwise.
    bench_order : ``True``: bench by bench inside every phase; ``False``: any available block. By default
        ``True`` (the bench order of mine-planning software, for both searches).
    lookahead_benches : depth of the lookahead cone, in benches (``search="lookahead"``; 0 = the block value).
    lookahead_slope : slope of the cone in degrees (default: the flattest slope of the model; with the 1:5
        pattern, one block per bench).
    lookahead_discount : discount per bench inside the cone (default: the top-down rate of ``pits``, else 0).
    phases : pit numbers that close each phase, increasing (default: every nested pit with new blocks). The
        final pit is always the last phase.
    min_component : continuity. A group of blocks of one phase (touching through faces, edges or corners)
        with fewer blocks than this, which touches nothing mined before, is not mined alone: it moves to the
        phase that first needs it (the lowest phase of the blocks that require it or of its later
        neighbours), and is mined with its surroundings. The slopes still hold. By default 27 (a 3 x 3 x 3
        cube: single rich blocks near the surface that pay for themselves at the lowest factors); 0 turns
        it off. Larger values also postpone real small bodies and move the sequence away from the nested
        pits.
    discount_rate, mining_rate, processing_rate, fixed_cost, timing, tonnage, ore : as in ``pit_by_pit``
        (``timing="continuous"`` discounts every block when it is mined).
        The discount rate and the ore per year of ``nested_pits`` are the defaults (the ore per year is the
        processing capacity unless ``processing_rate`` is given). ``timing``: ``"continuous"`` (default: every
        block discounted when it is mined, ``exp(-rate * t)``, as strategic mine-planning reports), or cash at
        the ``"mid"``, ``"end"`` or ``"start"`` of each period.
    threads : threads for the lookahead values and the check (0 = every core).

    Returns an ``OES``: ``summary()``, ``phase_table()``, ``periods``, ``table()``, ``to_csv()``,
    ``export()``, ``plot()``, ``plot_npv()``, ``check()``.

    Examples
    --------
    >>> import pymineflow
    >>> table = {"IX": [0, 1, 2, 1], "IY": [0, 0, 0, 0], "IZ": [1, 1, 1, 0],
    ...          "VALUE": [-1.0, -1.0, -1.0, 10.0], "TON": [1.0, 1.0, 1.0, 1.0]}
    >>> pits = pymineflow.nested_pits(table, block_size=(10, 10, 10), slopes=45, factors=[1.0])
    >>> oes = ultimate_pit_oes(pits, discount_rate=0.1, mining_rate=2)
    >>> oes.order.tolist(), oes.period[oes.order].tolist()  # the bench on top first, two blocks a period
    ([0, 1, 2, 3], [1, 1, 2, 2])
    >>> oes.check().ok
    True
    """
    seconds: dict[str, float] = {}
    if search not in ("proximity", "lookahead"):
        raise ValueError(f"search must be 'proximity' or 'lookahead', got {search!r}")
    if lookahead_scope not in ("phase", "pit"):
        raise ValueError(f"lookahead_scope must be 'phase' or 'pit', got {lookahead_scope!r}")
    prox = _number("proximity", proximity, 0.0, strict=False)
    if (
        isinstance(lookahead_benches, bool)
        or not isinstance(lookahead_benches, (int, np.integer))
        or lookahead_benches < 0
    ):
        raise ValueError(f"lookahead_benches must be an integer >= 0, got {lookahead_benches!r}")
    if timing not in _TIMINGS:
        raise ValueError(f"timing must be 'end', 'mid', 'start' or 'continuous', got {timing!r}")
    if discount_rate is None and not pits.discount_rate:
        raise ValueError("give discount_rate (or discount_rate= in nested_pits)")
    rate = float(pits.discount_rate) if discount_rate is None else _number("discount_rate", discount_rate, 0.0, False)
    fixed = _number("fixed_cost", fixed_cost, 0.0, strict=False)
    mine = None if mining_rate is None else _number("mining_rate", mining_rate, 0.0, strict=True)
    if processing_rate is None and pits.ore_per_year:
        processing_rate = float(pits.ore_per_year)
    values, destination, tons, names = _block_data(pits, tonnage, ore)
    plant: float | dict[str, float] | None
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
        raise ValueError("give mining_rate, processing_rate or both (or ore_per_year in nested_pits)")

    n = len(pits)
    numbers = np.asarray(pits.numbers, dtype=np.int64)
    inside = (numbers >= 1) & (numbers <= n)
    if not inside.any():
        raise ValueError("the final pit is empty: there is nothing to sequence")
    if phases is None:
        closing = np.flatnonzero(np.bincount(numbers[inside], minlength=n + 1)[1:]) + 1  # pits with new blocks
    else:
        try:
            items = list(phases)
            closing = np.array([int(p) for p in items], dtype=np.int64)
        except (TypeError, ValueError):
            raise ValueError(f"phases must be increasing pit numbers from 1 to {n}, got {phases!r}") from None
        if (
            any(isinstance(p, (bool, np.bool_)) or p != q for p, q in zip(items, closing, strict=True))
            or (closing.ndim != 1)
            or np.any(closing < 1)
            or np.any(closing > n)
            or np.any(np.diff(closing) <= 0)
        ):
            raise ValueError(f"phases must be increasing pit numbers from 1 to {n}, got {list(phases)}")
        if closing.size == 0 or closing[-1] != n:
            closing = np.append(closing, n)
    phase = np.zeros(numbers.size, dtype=np.int64)
    phase[inside] = np.searchsorted(closing, numbers[inside], side="left") + 1
    model = pits.model
    g = model.grid
    linear = np.asarray(g.linear, dtype=np.int64)
    offsets = [model._pit.offsets(d) for d in range(model._pit.n_domains)]
    if min_component is None:
        min_component = 27
    if isinstance(min_component, bool) or not isinstance(min_component, (int, np.integer)) or min_component < 0:
        raise ValueError(f"min_component must be None or an integer >= 0, got {min_component!r}")
    moved = (0, 0)
    if min_component > 1:
        start = time.perf_counter()
        domains = model._cell_domains()
        merged, groups, blocks = _oes_merge_small(g.nx, g.ny, g.nz, linear[inside], phase[inside], offsets, domains,
                                                  linear[~inside], int(min_component))  # fmt: skip
        phase[inside] = merged
        moved = (int(groups), int(blocks))
        seconds["continuity"] = time.perf_counter() - start
    used = np.unique(phase[inside])
    if used.size < closing.size:  # drop the phases without blocks
        renumber = np.zeros(closing.size + 1, dtype=np.int64)
        renumber[used] = np.arange(1, used.size + 1)
        phase = renumber[phase]
        closing = closing[used - 1]
    # every phase is labelled with the last nested pit that adds blocks to it (a pit that adds none is no phase)
    last_pit = np.zeros(closing.size + 1, dtype=np.int64)
    np.maximum.at(last_pit, phase[inside], numbers[inside])
    closing = last_pit[1:]

    rows = np.flatnonzero(inside)
    ix = np.asarray(g.ix, dtype=np.int64)[rows]
    iy = np.asarray(g.iy, dtype=np.int64)[rows]
    iz = np.asarray(g.iz, dtype=np.int64)[rows]
    dx, dy, dz = (float(s) for s in g.block_size)
    radius = _cone_radius(model, lookahead_slope, dx, dy, dz)
    bench_rate = (float(pits.bench_discount) if lookahead_discount is None
                  else _number("lookahead_discount", lookahead_discount, 0.0, strict=False))  # fmt: skip
    key = np.ascontiguousarray(values[rows], dtype=np.float64)
    if search == "lookahead" and lookahead_benches > 0:
        start = time.perf_counter()
        k = np.arange(1, int(lookahead_benches) + 1, dtype=np.float64)
        radii, weights = list(k * radius), list((1.0 + bench_rate) ** -k)
        if lookahead_scope == "pit":
            key = _oes_lookahead(ix, iy, iz, key, dx, dy, radii, weights, threads)
        else:  # only the blocks of the same phase: those that the search can reach
            ph = phase[rows]
            by_phase = np.argsort(ph, kind="stable")
            edges = np.searchsorted(ph[by_phase], np.arange(1, closing.size + 2))
            out = np.empty_like(key)
            for a, b in itertools.pairwise(edges):
                sel = by_phase[a:b]
                out[sel] = _oes_lookahead(ix[sel], iy[sel], iz[sel], key[sel], dx, dy, radii, weights, threads)
            key = out
        seconds["lookahead"] = time.perf_counter() - start
    nonzero = np.abs(key[key != 0])
    weight = prox * float(nonzero.mean()) / float(np.sqrt(dx * dy)) if nonzero.size else 0.0
    bench = True if bench_order is None else bool(bench_order)
    start = time.perf_counter()
    if bench:
        local = _oes_order(ix, iy, iz, phase[rows], key, weight, dx, dy)
    else:
        local = _oes_order_free(g.nx, g.ny, g.nz, linear[rows], ix, iy, iz, phase[rows], key, weight, dx, dy, dz,
                                offsets, model._cell_domains(), linear[~inside])  # fmt: skip
    seconds["order"] = time.perf_counter() - start
    order = rows[local]
    keys = np.full(numbers.size, np.nan)
    keys[rows] = key

    uses, caps = [], []
    if mine is not None:
        uses.append(tons)
        caps.append(mine)
    if isinstance(plant, dict):
        for name, cap in plant.items():
            uses.append(np.where(destination == names.index(name), tons, 0.0))
            caps.append(cap)
    elif plant is not None:
        uses.append(np.where(destination >= 0, tons, 0.0))
        caps.append(plant)
    tracked = [tons, np.where(destination >= 0, tons, 0.0)]
    if len(names) > 1:
        tracked += [np.where(destination == p, tons, 0.0) for p in range(len(names))]
    start = time.perf_counter()
    s = _oes_schedule(
        np.ascontiguousarray(values[order]),
        np.ascontiguousarray(np.stack([u[order] for u in uses]), dtype=np.float64),
        np.asarray(caps, dtype=np.float64),
        np.ascontiguousarray(np.stack([t[order] for t in tracked]), dtype=np.float64),
        rate, _SHIFT.get(timing, 0.0), fixed, timing == "continuous",
    )  # fmt: skip
    seconds["schedule"] = time.perf_counter() - start

    period = np.zeros(numbers.size, dtype=np.int64)
    period[order] = s["first_period"]
    discounted = np.zeros(numbers.size)
    discounted[order] = s["discounted"]
    count = s["period_value"].size
    share = np.clip(float(s["time_end"][-1]) - np.arange(count), 0.0, 1.0)  # the last period pays in proportion
    track = list(s["period_track"])
    periods: dict[str, NDArray[Any]] = {"period": np.arange(1, count + 1), "tonnage": track[0], "ore_tonnage": track[1]}
    if len(names) > 1:
        for name, t in zip(names, track[2:], strict=True):
            periods[f"ore_tonnage_{name}"] = t
    periods["value"] = s["period_value"]
    if timing == "continuous":  # every block discounted when it is mined; the fixed cost at a constant rate
        k = np.arange(count, dtype=np.float64)
        life = float(s["time_end"][-1])
        paid = _continuous_cost(fixed, rate, np.minimum(k + 1, life)) - _continuous_cost(fixed, rate, k)
        mean = np.exp(-rate * (k + 0.5 * share))
        value = s["period_value"]
        periods["discount_factor"] = np.divide(s["period_discounted"], value, out=mean, where=np.abs(value) > 0)
        periods["fixed_cost"] = fixed * share
        periods["discounted"] = s["period_discounted"] - paid
    else:
        periods["discount_factor"] = s["period_factor"]
        periods["fixed_cost"] = fixed * share
        periods["discounted"] = (s["period_value"] - fixed * share) * s["period_factor"]
    return OES(
        pits=pits, order=order, phase=phase, phase_pits=closing, key=keys, period=period, discounted=discounted,
        npv_curve=s["npv_stop"], time=s["time_end"], periods=periods, capacity_use=list(s["period_use"]),
        capacities=list(caps), values=values, tonnage=tons, destination=destination, processes=names,
        search=search, bench_order=bench, min_component=int(min_component), moved_groups=moved[0],
        moved_blocks=moved[1], proximity=prox, proximity_weight=weight,
        lookahead_benches=int(lookahead_benches), lookahead_radius=radius, lookahead_discount=bench_rate,
        discount_rate=rate, mining_rate=mine, processing_rate=plant, fixed_cost=fixed, timing=timing, seconds=seconds,
    )  # fmt: skip


def _cone_radius(model: Any, slope: float | None, dx: float, dy: float, dz: float) -> float:
    """Horizontal radius of the lookahead cone per bench (centre to centre, boundary included)."""
    if slope is None:
        if model.domain_slopes is None:  # 1:5 pattern: one block per bench
            return min(dx, dy)
        slope = min(s for sectors in model.domain_slopes for _, s in sectors)
    s = _number("lookahead_slope", slope, 0.0, strict=True)
    if s >= 90:
        raise ValueError(f"lookahead_slope must be in (0, 90) degrees, got {slope!r}")
    return float(dz / np.tan(np.radians(s) - 1e-9))
