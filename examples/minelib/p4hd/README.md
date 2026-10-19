# MineLib: p4hd

Gold-copper mine in Nevada; blocks of 50 x 50 x 20 ft, precedences: 45 degrees, 8 levels.

| Blocks | Precedence arcs | Published UPIT optimum | PyMineFlow | Blocks in the pit |
|---:|---:|---:|---:|---:|
| 40,947 | 738,609 | 293,373,256 | 293,373,256 (same pit, block for block) | 28,844 |

## Run

```bash
pip install pymineflow
python p4hd.py
```

The first run downloads the instance from [MineLib](https://minelib.org) (CC BY-SA 3.0) into `../data/` and
writes it as `p4hd.csv`: MineLib's `.blocks` file with the column names of its documentation (`ID, IX, IY, IZ,
...`) and `VALUE`, the block value of its `.upit` file.

## What the script does

1. **The ultimate pit as MineLib publishes it**, with its own precedences (`p4hd.prec`):
   `pymineflow.solve(csv, precedence=prec, value="VALUE")`. It prints the report, checks the pit against
   MineLib's published solution and writes `results/p4hd_pit.csv` (the whole model with `PIT` = 1 inside),
   `results/p4hd_pit_only.csv` and `results/p4hd_pit_report.txt`.
2. **A full study** with the block size (50 x 50 x 20 ft), 45° and 8 level(s) that MineLib describes:
   100 nested pits (factors 1 % to 100 % of the values) with top-down bench discounting, then the ultimate
   pit OES, with the discount rate and capacities of MineLib's `.cpit` file:

   | Discount rate | Mining capacity | Processing capacity | Benches per year* |
   |---:|---:|---:|---:|
   | 15 % | 52,500,000 t a year | 12,500,000 t a year | 14 |

   \* MineLib does not give it: the benches of the final pit divided by its life. Ore: the blocks of positive
   value.

   It writes to `results/`: `report_cumulative.csv` and `report_incremental.csv` (every phase with its best and
   worst case NPV and the sequence), `model.csv` and `final_pit.csv`, `npv.png` (best against worst case),
   `oes.gif` (the final pit mined in the order of the sequence) and `oes_surface.gif` (the ground dug), both in
   plan view, and the sequence block by block (`oes.csv`, `oes_periods.csv`, `oes_report.txt`, `oes_npv.png`).
   Then it opens two 3-D windows with a slider along the sequence: the final pit mined away, and the ground
   surface dug.

   Result: 98 phases, a final pit of 28,045 blocks, OES NPV 226,672,186 over 4.60 periods (every
   block discounted when it is mined). `oes.check()` confirms that every block is mined once, after its
   predecessors, within the capacities.

   With slopes built from the geometry the pit differs slightly from MineLib's: MineLib drops the arcs to blocks missing from the model, PyMineFlow keeps the cone through them.

