# MineLib: w23

Partial mine (phases 2 and 3); blocks of 25 x 25 x 20 ft, precedences: 45 degrees, 8 levels.

| Blocks | Precedence arcs | Published UPIT optimum | PyMineFlow | Blocks in the pit |
|---:|---:|---:|---:|---:|
| 74,260 | 764,786 | 510,973,998 | 510,973,998 (same pit, block for block) | 49,061 |

## Run

```bash
pip install pymineflow
python w23.py
```

The first run downloads the instance from [MineLib](https://minelib.org) (CC BY-SA 3.0) into `../data/` and
writes it as `w23.csv`: MineLib's `.blocks` file with the column names of its documentation (`ID, IX, IY, IZ,
...`) and `VALUE`, the block value of its `.upit` file.

## What the script does

1. **The ultimate pit as MineLib publishes it**, with its own precedences (`w23.prec`):
   `pymineflow.solve(csv, precedence=prec, value="VALUE")`. It prints the report, checks the pit against
   MineLib's published solution and writes `results/w23_pit.csv` (the whole model with `PIT` = 1 inside),
   `results/w23_pit_only.csv` and `results/w23_pit_report.txt`.
2. **A full study** with the block size (25 x 25 x 20 ft; its two mills (3.61 and 1 Mt a year) as one plant), 45° and 8 level(s) that MineLib describes:
   100 nested pits (factors 1 % to 100 % of the values) with top-down bench discounting, then the ultimate
   pit OES, with the discount rate and capacities of MineLib's `.cpit` file:

   | Discount rate | Mining capacity | Processing capacity | Benches per year* |
   |---:|---:|---:|---:|
   | 10 % | 68,000,000 t a year | 4,610,000 t a year | 8 |

   \* MineLib does not give it: the benches of the final pit divided by its life. Ore: the blocks of positive
   value.

   It writes to `results/`: `report_cumulative.csv` and `report_incremental.csv` (every phase with its best and
   worst case NPV and the sequence), `model.csv` and `final_pit.csv`, `npv.png` (best against worst case),
   `oes.gif` (the final pit mined in the order of the sequence) and `oes_surface.gif` (the ground dug), both in
   plan view, and the sequence block by block (`oes.csv`, `oes_periods.csv`, `oes_report.txt`, `oes_npv.png`).
   Then it opens two 3-D windows with a slider along the sequence: the final pit mined away, and the ground
   surface dug.

   Result: 86 phases, a final pit of 60,283 blocks, OES NPV 303,572,841 over 7.54 periods (every
   block discounted when it is mined). `oes.check()` confirms that every block is mined once, after its
   predecessors, within the capacities.

   MineLib's graph of w23 is not a symmetric 45° cone (two levels up it lacks two diagonals), so the pit built from the geometry is smaller than MineLib's.

