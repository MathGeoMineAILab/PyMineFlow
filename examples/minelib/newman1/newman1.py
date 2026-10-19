"""MineLib instance newman1.

The ultimate pit as MineLib publishes it,
and a full study: nested pits with top-down discounting and the ultimate pit OES.

Run it from anywhere: ``python newman1.py``. The data are downloaded from https://minelib.org the first time
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
csv, prec, solution = load("newman1")

# 1. The ultimate pit with MineLib's precedences (newman1.prec) and values (VALUE).
#    Published optimum: 26,086,899
pit = pymineflow.solve(csv, precedence=prec, value="VALUE")
print(pit.summary())
published = np.zeros(pit.mask.size, dtype=bool)
published[np.loadtxt(solution, dtype=np.int64, ndmin=1)] = True
print("same pit as MineLib's published solution:", bool(np.array_equal(pit.mask, published)))
pit.to_csv(results / "newman1_pit.csv")  # the whole model, PIT = 1 inside the final pit
pit.to_csv(results / "newman1_pit_only.csv", only_pit=True)  # only the blocks of the final pit
(results / "newman1_pit_report.txt").write_text(pit.summary(), encoding="utf-8")

# 2. A full study: nested pits (factors 1 % to 100 % of the values) with top-down bench discounting, then the
#    ultimate pit OES. Block size: unit cubes (size unknown). 45 degrees and 1 level(s), discount rate and capacities as
#    MineLib documents them; benches per year: the benches of the final pit over its life.
pits = pymineflow.nested_pits(
    csv,
    block_size=(1, 1, 1),
    slopes=45,
    n_benches=1,
    value="VALUE",
    factors=np.arange(1, 101) / 100,
    discount_rate=0.08,
    benches_per_year=7,  # top-down discounting
    ore_per_year=1_100_000,
)  # the plant
oes = pits.oes(mining_rate=2_000_000)
print(oes.summary())
print(oes.check().summary())  # complete, ordered, slopes, value, capacities
print(oes.report())  # every phase as the final pit: best and worst case NPV, the OES
print(oes.report("incremental"))  # what every phase adds

# the reports (cumulative, incremental), the model with the answer, the final pit, npv.png (best vs worst
# case), oes.gif (the final pit mined in order) and oes_surface.gif (the ground dug), in plan view, and the
# details: the sequence block by block (oes.csv), the periods, the summary with the check and the NPV curve
oes.export(results, report="both", pit="both", details=True)

# 3-D windows with the slider along the sequence (close one to open the next)
oes.plot(remaining=True, view="iso")  # the whole final pit; the slider removes what is mined
oes.plot(surface=True, view="iso")  # the ground surface; the slider digs it
