"""Nested pits of MineLib's McLaughlin gold mine with and without top-down bench discounting, with the
exact 45-degree cone, their NPV, and the files of the study.

The block values come from MineLib's documented economics for McLaughlin: $900/oz gold, 90 %
recovery, $19/ton processing and $1.32/ton mining (short tons, grades in oz/ton). The NPV uses
MineLib's scheduling data: 15 % a year and a mill of 3.3 million tons a year; the sinking rate of
8 benches a year is an assumption. Run it from the repository root:

    python examples/mclaughlin_discounted_nested_pits.py

The model (7.7 MB zip) is downloaded from https://minelib.org on the first run.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

import pymineflow
from pymineflow import Economics, Metal, Process

blocks = Path("mclaughlin.blocks")
if not blocks.exists():
    with urllib.request.urlopen("https://minelib.org/v1/data/mclaughlin.blocks.zip") as r:
        blocks.write_bytes(zipfile.ZipFile(io.BytesIO(r.read())).read("mclaughlin.blocks"))

model = pymineflow.read_blocks(blocks, names=["ID", "IX", "IY", "IZ", "BLOCK_VALUE", "TON", "DESTINATION", "AU"])
economics = Economics(
    metals=[Metal("AU", price=900, grade_unit="oz/t", price_unit="oz")],
    processes=[Process("mill", cost=19, recovery={"AU": 0.9})],
    mining_cost=1.32,
    tonnage="TON",
    tonnage_unit="st",  # short tons
)

for rate in (0.0, 0.15):  # without and with discounting
    pits = pymineflow.nested_pits(
        model,
        block_size=(7.62, 7.62, 6.096),  # 25 x 25 x 20 ft
        slopes=45,  # exact cone over the full height
        economics=economics,
        scale="price",
        factors=np.arange(1, 101) / 100,  # price factors from 1 % to 100 %
        discount_rate=rate,
        benches_per_year=8,  # 15 % / 8 = 1.875 % per bench (assumed sinking rate)
        ore_per_year=3_300_000,  # for the best and worst case NPV of every phase
    )
    print(pits.report())
    print()
    folder = Path("mclaughlin_discounted" if rate else "mclaughlin_undiscounted")
    pits.export(folder, rotation=-90)  # pits.csv, final_pit.csv, reports, plan.png (north to the right), npv.png, GIFs

# only the blocks of the final pit, or of the pit of highest NPV
pits.to_csv("mclaughlin_final_pit.csv", pit="final")
pits.to_csv("mclaughlin_best_npv_pit.csv", pit="best")
pits.plot(view="top", rotation=-90, outline=False)  # 3-D window; the slider walks through the pits
