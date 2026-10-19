"""Phase reports of nested pits and of the OES, cumulative or incremental (shared by ``simple`` and ``oes``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

KINDS = ("cumulative", "incremental")


def check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be 'cumulative' or 'incremental', got {kind!r}")
    return kind


def kinds(report: str | None) -> tuple[str, ...]:
    """The reports asked for by ``report=``: ``"cumulative"``, ``"incremental"``, ``"both"`` or ``None``."""
    if report is None:
        return ()
    if report == "both":
        return KINDS
    if isinstance(report, str) and report in KINDS:
        return (report,)
    raise ValueError(f"report must be 'cumulative', 'incremental', 'both' or None, got {report!r}")


def phase_pits(numbers: NDArray[Any], n: int) -> NDArray[np.int64]:
    """The pits (from 1) with new blocks: the phases."""
    return np.flatnonzero(np.bincount(numbers, minlength=n + 1)[1 : n + 1] > 0) + 1


def phase_report(
    numbers: NDArray[Any],
    factors: NDArray[np.float64],
    values: NDArray[np.float64],
    tons: NDArray[np.float64] | None,
    ore: NDArray[np.bool_],
    npv: Mapping[str, NDArray[Any]] | None,
    kind: str,
) -> dict[str, NDArray[Any]]:
    """One row per phase. ``cumulative``: every phase as the final pit (everything mined up to it), with the
    best and worst case NPV; ``incremental``: the blocks the phase adds to the previous one, with the increment
    of the best case NPV (``npv``)."""
    n = factors.size
    k = phase_pits(numbers, n) - 1

    def rows(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        total = np.cumsum(np.bincount(numbers, weights=weights, minlength=n + 1)[1 : n + 1])[k]
        return total if kind == "cumulative" else np.diff(total, prepend=0.0)

    profit = rows(values)
    out: dict[str, NDArray[Any]] = {"phase": np.arange(1, k.size + 1), "factor": factors[k], "profit": profit}
    if tons is not None:
        rock = rows(tons)
        ore_t = rows(np.where(ore, tons, 0.0))
        out["profit_per_t"] = profit / np.where(rock > 0, rock, np.nan)
    if npv is not None:
        if kind == "cumulative":
            out["npv_best"] = np.asarray(npv["npv_best"], dtype=np.float64)[k]
            out["npv_worst"] = np.asarray(npv["npv_worst"], dtype=np.float64)[k]
        else:
            out["npv"] = np.diff(np.asarray(npv["npv_best"], dtype=np.float64)[k], prepend=0.0)
    if tons is not None:
        out["rock"], out["ore"] = rock, ore_t
        out["strip_ratio"] = (rock - ore_t) / np.where(ore_t > 0, ore_t, np.nan)
        out["waste"] = rock - ore_t
    return out


_COLUMNS = (
    ("phase", "Phase", 5, "{:>5}"),
    ("factor", "Factor", 7, "{:>7.2%}"),
    ("profit", "Profit", 16, "{:>16,.0f}"),
    ("profit_per_t", "Profit/t", 9, "{:>9.2f}"),
    ("npv_best", "NPV best case", 16, "{:>16,.0f}"),
    ("npv_worst", "NPV worst case", 16, "{:>16,.0f}"),
    ("npv", "NPV", 16, "{:>16,.0f}"),
    ("oes_npv", "OES NPV", 16, "{:>16,.0f}"),
    ("oes_start", "OES start", 9, "{:>9.2f}"),
    ("oes_end", "OES end", 9, "{:>9.2f}"),
    ("rock", "Rock", 14, "{:>14,.0f}"),
    ("ore", "Ore", 14, "{:>14,.0f}"),
    ("strip_ratio", "Strip ratio", 11, "{:>11.3f}"),
    ("waste", "Waste", 14, "{:>14,.0f}"),
)


def report_text(table: Mapping[str, NDArray[Any]], head_lines: Sequence[str]) -> str:
    """The report as text: the head lines, one line per phase and, when cumulative, the phase of highest NPV."""
    shown = [c for c in _COLUMNS if c[0] in table]
    head = " ".join(f"{title:>{width}}" for _, title, width, _ in shown)
    lines = [*head_lines, head, "-" * len(head)]
    for i in range(len(table["phase"])):
        lines.append(" ".join(fmt.format(table[key][i]) for key, _, _, fmt in shown))
    t = table
    if "npv_best" in t and len(t["phase"]):
        b, w = int(np.nanargmax(t["npv_best"])), int(np.nanargmax(t["npv_worst"]))
        lines.append(
            f"Highest NPV: best case phase {t['phase'][b]} ({t['factor'][b]:.2%}) {t['npv_best'][b]:,.0f}; "
            f"worst case phase {t['phase'][w]} ({t['factor'][w]:.2%}) {t['npv_worst'][w]:,.0f}"
        )
    return "\n".join(lines)


def describe(kind: str) -> str:
    return {
        "cumulative": "cumulative: every phase as the final pit, everything mined up to it",
        "incremental": "incremental: the blocks every phase adds to the previous one",
    }[kind]
