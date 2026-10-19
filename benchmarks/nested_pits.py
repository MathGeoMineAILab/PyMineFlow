"""Hundreds of nested pits on MineLib's McLaughlin model (2.1 million blocks), by profit and by price.

The block values at a price factor come from MineLib's documented economics for McLaughlin:
$900/oz gold, 90 % recovery, $19/ton processing and $1.32/ton mining (short tons, grades in oz/ton),
with the best destination (mill or waste) chosen again at every price.

Every pit is checked to be nested in the next one, and a sample of pits is solved again on its own
(``ultimate_pit``) and compared block by block.

Usage::

    python benchmarks/nested_pits.py --download            # fetch mclaughlin.blocks (7.7 MB zip)
    python benchmarks/nested_pits.py --pits 1000 --check 40
"""

from __future__ import annotations

import argparse
import io
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

import pymineflow
from pymineflow import Economics, Metal, Process

URL = "https://minelib.org/v1/data/mclaughlin.blocks.zip"
COLUMNS = ["ID", "X", "Y", "Z", "BLOCK_VALUE", "TON", "DESTINATION", "AU"]
SIZE = (7.62, 7.62, 6.096)  # 25 x 25 x 20 ft
ECONOMICS = Economics(
    metals=[Metal("AU", price=900, grade_unit="oz/t", price_unit="oz")],
    processes=[Process("mill", cost=19, recovery={"AU": 0.9})],
    mining_cost=1.32,
    tonnage="TON",
    tonnage_unit="st",
)


def download(data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as r, zipfile.ZipFile(io.BytesIO(r.read())) as z:
        (data / "mclaughlin.blocks").write_bytes(z.read("mclaughlin.blocks"))


def run(table: dict, scale: str, factors: np.ndarray, n_benches: int | None, check: int) -> None:
    options = {"economics": ECONOMICS} if scale == "price" else {"value": "BLOCK_VALUE"}
    start = time.perf_counter()
    pits = pymineflow.nested_pits(
        table, block_size=SIZE, slopes=45, factors=factors, scale=scale, n_benches=n_benches,
        indices=("X", "Y", "Z"), **options,
    )  # fmt: skip
    elapsed = time.perf_counter() - start
    t = pits.table_by_pit()
    numbers = pits.pit_number()
    assert (np.diff(t["blocks"]) >= 0).all(), "pits are not nested"
    print(
        f"\n{scale} factor, {len(factors)} pits from {factors[0]:g} to {factors[-1]:g}, n_benches={n_benches or 'full'}"
    )
    print(f"  time {elapsed:.1f} s ({elapsed / len(factors) * 1000:.1f} ms per pit), solver {sum(pits.seconds):.1f} s")
    print(f"  distinct pits {(t['new_blocks'] > 0).sum()}, blocks {t['blocks'][0]:,} to {t['blocks'][-1]:,}")
    sample = np.unique(np.linspace(0, len(factors) - 1, check).astype(int))
    start = time.perf_counter()
    for k in sample:
        v = pits.scenario(int(k), None)
        alone = pymineflow.ultimate_pit(
            v, ix=table["X"], iy=table["Y"], iz=table["Z"], block_size=SIZE, slopes=45, n_benches=n_benches
        )
        mask = (numbers >= 1) & (numbers <= k + 1)
        if alone.value != int(v[mask].sum()) or not np.array_equal(alone.mask, mask):
            raise AssertionError(f"pit {k + 1} differs from solving its factor alone")
    print(f"  {sample.size} pits solved again alone: identical values and blocks ({time.perf_counter() - start:.0f} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "minelib")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--pits", type=int, default=1000)
    parser.add_argument("--low", type=float, default=0.3)
    parser.add_argument("--high", type=float, default=1.3)
    parser.add_argument("--benches", type=int, default=None, help="default: full height")
    parser.add_argument("--check", type=int, default=20, help="pits solved again alone")
    args = parser.parse_args()
    if args.download:
        download(args.data)
    table = pymineflow.read_blocks(args.data / "mclaughlin.blocks", names=COLUMNS)
    factors = np.linspace(args.low, args.high, args.pits)
    for scale in ("revenue", "price"):
        run(table, scale, factors, args.benches, args.check)


if __name__ == "__main__":
    main()
