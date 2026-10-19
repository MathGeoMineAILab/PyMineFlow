"""MineLib instance sm2.

The ultimate pit as MineLib publishes it.

Run it from anywhere: ``python sm2.py``. The data are downloaded from https://minelib.org the first time
(see ../minelib_data.py); the results go to results/ next to this file.
"""

import sys
from pathlib import Path

import numpy as np

import pymineflow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from minelib_data import load

here = Path(__file__).resolve().parent
results = here / "results"
results.mkdir(exist_ok=True)
csv, prec, solution = load("sm2")

# 1. The ultimate pit with MineLib's precedences (sm2.prec) and values (VALUE).
#    Published optimum: 2,743,603,730
pit = pymineflow.solve(csv, precedence=prec, value="VALUE")
print(pit.summary())
published = np.zeros(pit.mask.size, dtype=bool)
published[np.loadtxt(solution, dtype=np.int64, ndmin=1)] = True
print("same pit as MineLib's published solution:", bool(np.array_equal(pit.mask, published)))
pit.to_csv(results / "sm2_pit.csv")  # the whole model, PIT = 1 inside the final pit
pit.to_csv(results / "sm2_pit_only.csv", only_pit=True)  # only the blocks of the final pit
(results / "sm2_pit_report.txt").write_text(pit.summary(), encoding="utf-8")

# MineLib gives no block size or slope for this instance: the nested pits and the sequence, which build the
# slopes from the geometry, do not apply here.
pit.plot(view="top", screenshot=results / "sm2_pit.png", show=False).close()  # plan view of the final pit
pit.plot()  # 3-D view of the final pit
