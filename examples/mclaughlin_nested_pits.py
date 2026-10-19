"""100 nested pits of MineLib's McLaughlin gold mine (2.1 million blocks), as MineLib solves it.

MineLib's setup: 25 x 25 x 20 ft blocks, 45 degrees, precedences over 8 levels. The pit of factor
1.00 is MineLib's published optimum, 1,495,886,962. Run it from the repository root:

    python examples/mclaughlin_nested_pits.py

The model (7.7 MB zip) is downloaded from https://minelib.org on the first run.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

import pymineflow

blocks = Path("mclaughlin.blocks")
if not blocks.exists():
    with urllib.request.urlopen("https://minelib.org/v1/data/mclaughlin.blocks.zip") as r:
        blocks.write_bytes(zipfile.ZipFile(io.BytesIO(r.read())).read("mclaughlin.blocks"))

model = pymineflow.read_blocks(blocks, names=["ID", "IX", "IY", "IZ", "BLOCK_VALUE", "TON", "DESTINATION", "AU"])

pits = pymineflow.nested_pits(
    model,
    block_size=(7.62, 7.62, 6.096),  # 25 x 25 x 20 ft
    slopes=45,
    n_benches=8,  # MineLib: 45 degrees, 8 levels
    factors=np.linspace(0.30, 1.00, 100),  # 100 nested pits (profit factor)
)
print(pits[-1].summary())  # the ultimate pit (factor 1.00)
print(pits.summary())  # one line per pit
pits.to_csv("mclaughlin_nested.csv")
pits.plot(view="top", screenshot="mclaughlin_100_pits.png")  # plan view
