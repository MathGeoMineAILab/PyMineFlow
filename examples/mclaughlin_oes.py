"""Ultimate pit OES of MineLib's McLaughlin gold mine: the extraction sequence of the top-down discounted
final pit, block by block, with the proximity and the lookahead searches, its NPV and its check.

The block values come from MineLib's documented economics for McLaughlin: $900/oz gold, 90 %
recovery, $19/ton processing and $1.32/ton mining (short tons, grades in oz/ton). The NPV uses
MineLib's scheduling data: 15 % a year and a mill of 3.3 million tons a year; the sinking rate of
8 benches a year is an assumption. Run it from the repository root:

    python examples/mclaughlin_oes.py

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
pits = pymineflow.nested_pits(
    model,
    block_size=(7.62, 7.62, 6.096),  # 25 x 25 x 20 ft
    slopes=45,  # exact cone over the full height
    economics=economics,
    scale="price",
    factors=np.arange(1, 101) / 100,  # price factors from 1 % to 100 %: the phases
    discount_rate=0.15,  # top-down discounting: 15 % / 8 benches a year = 1.875 % per bench
    benches_per_year=8,
    ore_per_year=3_300_000,  # the mill
)

for search in ("proximity", "lookahead"):
    oes = pits.oes(search=search)  # the ultimate pit OES
    print(oes.summary())
    print(oes.check().summary())  # completeness, order, slopes, value and capacities
    print()
    print(oes.report())  # every phase: best and worst case NPV and the sequence
    # reports (cumulative and incremental), the whole model with the answer, the NPV chart and the two GIFs of the
    # README: the final pit mined away (oes.gif) and the ground surface dug (oes_surface.gif), in plan view
    oes.export(f"mclaughlin_oes_{search}", report="both", pit="whole", rotation=-90)
oes.plot(remaining=True, view="iso", rotation=-90)  # 3-D window: the slider mines the final pit in order
oes.plot(surface=True, view="iso", rotation=-90)  # 3-D window: the slider digs the ground surface
