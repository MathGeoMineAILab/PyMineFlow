"""100 nested pits of MineLib's McLaughlin gold mine by price factor, with the exact 45-degree cone.

The block values come from MineLib's documented economics for McLaughlin: $900/oz gold, 90 %
recovery, $19/ton processing and $1.32/ton mining (short tons, grades in oz/ton). At every price
each block goes to the mill or to waste, whichever pays more. The slope holds over the full height
(``n_benches=None``, the default). Run it from the repository root:

    python examples/mclaughlin_price_factors.py

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
    factors=np.linspace(0.30, 1.30, 100),  # gold from $270 to $1,170 per ounce
)
print(pits.summary())
pits.to_csv("mclaughlin_price_pits.csv")
pits.plot(
    view="top",
    rotation=-90,  # plan view, north to the right
    outline=False,
    title="McLaughlin: 100 nested pits, price factor 0.3 to 1.3, exact 45 deg cone",
    screenshot="mclaughlin_price_pits.png",
)
