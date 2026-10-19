"""Compare nested pits with a reference (e.g. another program), block by block and pit by pit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["PitComparison", "compare_nested"]


@dataclass(frozen=True)
class PitComparison:
    """Nested pits against a reference (see ``NestedPits.compare``).

    ``table`` has one row per factor of the reference: the blocks of both pits (``blocks`` and
    ``blocks_reference``), the blocks they share and the ones only in each, the Jaccard index, and for
    both pits the tonnage, ore and waste tonnage, strip ratio, value at factor 1, lowest bench and
    footprint (number of block columns). ``same_first`` blocks of ``compared`` enter their first pit at
    the same factor in both.
    """

    table: dict[str, NDArray[Any]]
    same_first: int
    compared: int

    @property
    def identical(self) -> int:
        """Number of reference pits with exactly the same blocks."""
        return int(((self.table["only_pits"] + self.table["only_reference"]) == 0).sum())

    def summary(self) -> str:
        """A report: the block-by-block agreement and one line per reference pit."""
        t = self.table
        n = len(t["factor"])
        lines = [
            f"Block by block: {self.same_first:,} of {self.compared:,} blocks enter their first pit at the same "
            f"factor ({self.same_first / max(1, self.compared):.3%})",
            f"Pit by pit: {self.identical} of {n} pits identical",
            f"{'factor':>8} {'blocks':>9} {'ref':>9} {'only':>7} {'only ref':>8} {'jaccard':>8} {'tonnage':>14} "
            f"{'ref':>14} {'ore':>13} {'ref':>13} {'SR':>7} {'ref':>7} {'value':>16} {'ref':>16}",
        ]
        for k in range(n):
            lines.append(
                f"{t['factor'][k]:>8.4f} {t['blocks'][k]:>9,} {t['blocks_reference'][k]:>9,} {t['only_pits'][k]:>7,} "
                f"{t['only_reference'][k]:>8,} {t['jaccard'][k]:>8.4f} {t['tonnage'][k]:>14,.0f} "
                f"{t['tonnage_reference'][k]:>14,.0f} {t['ore'][k]:>13,.0f} {t['ore_reference'][k]:>13,.0f} "
                f"{t['strip_ratio'][k]:>7.3f} {t['strip_ratio_reference'][k]:>7.3f} {t['value'][k]:>16,.0f} "
                f"{t['value_reference'][k]:>16,.0f}"
            )
        return "\n".join(lines)

    def to_csv(self, path: str | Path) -> Path:
        """Write the table (one row per reference pit) to a CSV file."""
        from .readers import write_table

        return write_table(self.table, path)


def compare_nested(
    pits: Any,
    reference: ArrayLike,
    *,
    ore: str | ArrayLike | None = None,
    tonnage: str | float | ArrayLike | None = None,
) -> PitComparison:
    """Compare nested pits with a reference given, per block, as the factor of the first reference pit that
    holds it (0 = in no pit), e.g. from ``read_reference_pits``. Each reference pit is compared with the pit of
    the largest factor of ``pits`` that does not exceed its factor. ``ore`` and ``tonnage`` are those of
    ``NestedPits.npv`` (ore: processed at factor 1)."""
    from .npv import _block_data

    ref = np.asarray(reference, dtype=np.float64)
    numbers = np.asarray(pits.numbers)
    if ref.shape != numbers.shape or not np.isfinite(ref).all() or (ref < 0).any():
        raise ValueError(f"reference must be one finite factor >= 0 per block ({numbers.size}), 0 = in no pit")
    values, destination, tons, _ = _block_data(pits, tonnage, ore)
    processed = destination >= 0
    factors = np.asarray(pits.factors, dtype=np.float64)
    first = np.where(numbers > 0, factors[np.maximum(numbers, 1) - 1], 0.0)
    union = (first > 0) | (ref > 0)
    same = union & np.isclose(first, ref, rtol=0, atol=1e-9)
    g = pits.model.grid
    columns = np.asarray(g.iy, dtype=np.int64) * (int(np.max(g.ix)) + 1) + np.asarray(g.ix, dtype=np.int64)
    iz = np.asarray(g.iz)

    def describe(mask: NDArray[np.bool_]) -> list[float]:
        rock = float(tons[mask].sum())
        ore_t = float(tons[mask & processed].sum())
        return [
            int(mask.sum()), rock, ore_t, rock - ore_t, (rock - ore_t) / ore_t if ore_t > 0 else np.nan,
            float(values[mask].sum()), int(iz[mask].min()) if mask.any() else -1, len(np.unique(columns[mask])),
        ]  # fmt: skip

    keys = ["blocks", "tonnage", "ore", "waste", "strip_ratio", "value", "lowest_bench", "footprint"]
    rows: dict[str, list[float]] = {"factor": [], "shared": [], "only_pits": [], "only_reference": [], "jaccard": []}
    for key in keys:
        rows[key], rows[f"{key}_reference"] = [], []
    for f in np.unique(ref[ref > 0]):
        theirs = (ref > 0) & (ref <= f + 1e-9)
        k = int(np.searchsorted(factors, f + 1e-9, side="right")) - 1  # largest factor of pits <= f
        ours = (numbers >= 1) & (numbers <= k + 1) if k >= 0 else np.zeros(numbers.size, dtype=bool)
        rows["factor"].append(float(f))
        rows["shared"].append(int((ours & theirs).sum()))
        rows["only_pits"].append(int((ours & ~theirs).sum()))
        rows["only_reference"].append(int((theirs & ~ours).sum()))
        rows["jaccard"].append(float((ours & theirs).sum() / max(1, (ours | theirs).sum())))
        for key, a, b in zip(keys, describe(ours), describe(theirs), strict=True):
            rows[key].append(a)
            rows[f"{key}_reference"].append(b)
    order = ["factor", "blocks", "blocks_reference", "shared", "only_pits", "only_reference", "jaccard"]
    order += [k for key in keys[1:] for k in (key, f"{key}_reference")]
    table = {k: np.asarray(rows[k]) for k in order}
    return PitComparison(table=table, same_first=int(same.sum()), compared=int(union.sum()))
