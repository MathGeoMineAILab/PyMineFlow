# MineLib: marvin

Marvin test mine; blocks of 30 x 30 x 30 m, precedences: 45 degrees, 8 levels.

| Blocks | Precedence arcs | Published UPIT optimum | PyMineFlow | Blocks in the pit |
|---:|---:|---:|---:|---:|
| 53,271 | 650,631 | 1,415,655,436 | 1,415,655,436 (same pit, block for block) | 8,516 |

## Run

```bash
pip install pymineflow
python marvin.py
```

The first run downloads the instance from [MineLib](https://minelib.org) (CC BY-SA 3.0) into `../data/` and
writes it as `marvin.csv`: MineLib's `.blocks` file with the column names of its documentation (`ID, IX, IY, IZ,
...`) and `VALUE`, the block value of its `.upit` file.

## What the script does

1. **The ultimate pit as MineLib publishes it**, with its own precedences (`marvin.prec`):
   `pymineflow.solve(csv, precedence=prec, value="VALUE")`. It prints the report, checks the pit against
   MineLib's published solution and writes `results/marvin_pit.csv` (the whole model with `PIT` = 1 inside),
   `results/marvin_pit_only.csv` and `results/marvin_pit_report.txt`.
2. **A full study** with the block size (30 x 30 x 30 m), 45° and 8 level(s) that MineLib describes:
   100 nested pits (factors 1 % to 100 % of the values) with top-down bench discounting, then the ultimate
   pit OES, with the discount rate and capacities of MineLib's `.cpit` file:

   | Discount rate | Mining capacity | Processing capacity | Benches per year* |
   |---:|---:|---:|---:|
   | 10 % | 60,000,000 t a year | 20,000,000 t a year | 1 |

   \* MineLib does not give it: the benches of the final pit divided by its life. Ore: the blocks of positive
   value.

   It writes to `results/`: `report_cumulative.csv` and `report_incremental.csv` (every phase with its best and
   worst case NPV and the sequence), `model.csv` and `final_pit.csv`, `npv.png` (best against worst case),
   `oes.gif` (the final pit mined in the order of the sequence) and `oes_surface.gif` (the ground dug), both in
   plan view, and the sequence block by block (`oes.csv`, `oes_periods.csv`, `oes_report.txt`, `oes_npv.png`).
   Then it opens two 3-D windows with a slider along the sequence: the final pit mined away, and the ground
   surface dug.

   Result: 63 phases, a final pit of 7,424 blocks, OES NPV 804,377,504 over 13.39 periods (every
   block discounted when it is mined). `oes.check()` confirms that every block is mined once, after its
   predecessors, within the capacities.

