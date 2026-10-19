# MineLib: mclaughlin_limit

The limit pit of the McLaughlin gold mine (California); blocks of 25 x 25 x 20 ft, precedences: 45 degrees, 8 levels.

| Blocks | Precedence arcs | Published UPIT optimum | PyMineFlow | Blocks in the pit |
|---:|---:|---:|---:|---:|
| 112,687 | 3,035,483 | 1,495,726,474 | 1,495,726,474 (same pit, block for block) | 110,225 |

## Run

```bash
pip install pymineflow
python mclaughlin_limit.py
```

The first run downloads the instance from [MineLib](https://minelib.org) (CC BY-SA 3.0) into `../data/` and
writes it as `mclaughlin_limit.csv`: MineLib's `.blocks` file with the column names of its documentation (`ID, IX, IY, IZ,
...`) and `VALUE`, the block value of its `.upit` file.

## What the script does

1. **The ultimate pit as MineLib publishes it**, with its own precedences (`mclaughlin_limit.prec`):
   `pymineflow.solve(csv, precedence=prec, value="VALUE")`. It prints the report, checks the pit against
   MineLib's published solution and writes `results/mclaughlin_limit_pit.csv` (the whole model with `PIT` = 1 inside),
   `results/mclaughlin_limit_pit_only.csv` and `results/mclaughlin_limit_pit_report.txt`.
2. **A full study** with the block size (25 x 25 x 20 ft), 45° and 8 level(s) that MineLib describes:
   100 nested pits (factors 1 % to 100 % of the values) with top-down bench discounting, then the ultimate
   pit OES, with the discount rate and capacities of MineLib's `.cpit` file:

   | Discount rate | Mining capacity | Processing capacity | Benches per year* |
   |---:|---:|---:|---:|
   | 15 % | unlimited | 3,300,000 t a year | 5 |

   \* MineLib does not give it: the benches of the final pit divided by its life. Ore: the blocks of positive
   value.

   It writes to `results/`: `report_cumulative.csv` and `report_incremental.csv` (every phase with its best and
   worst case NPV and the sequence), `model.csv` and `final_pit.csv`, `npv.png` (best against worst case),
   `oes.gif` (the final pit mined in the order of the sequence) and `oes_surface.gif` (the ground dug), both in
   plan view, and the sequence block by block (`oes.csv`, `oes_periods.csv`, `oes_report.txt`, `oes_npv.png`).
   Then it opens two 3-D windows with a slider along the sequence: the final pit mined away, and the ground
   surface dug.

   Result: 98 phases, a final pit of 104,153 blocks, OES NPV 898,185,282 over 9.54 periods (every
   block discounted when it is mined). `oes.check()` confirms that every block is mined once, after its
   predecessors, within the capacities.

