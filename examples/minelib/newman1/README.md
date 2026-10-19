# MineLib: newman1

Mine provided by Alexandra Newman; blocks of unknown size, precedences: the 5 blocks above.

| Blocks | Precedence arcs | Published UPIT optimum | PyMineFlow | Blocks in the pit |
|---:|---:|---:|---:|---:|
| 1,060 | 3,922 | 26,086,899 | 26,086,899 (same pit, block for block) | 1,059 |

## Run

```bash
pip install pymineflow
python newman1.py
```

The first run downloads the instance from [MineLib](https://minelib.org) (CC BY-SA 3.0) into `../data/` and
writes it as `newman1.csv`: MineLib's `.blocks` file with the column names of its documentation (`ID, IX, IY, IZ,
...`) and `VALUE`, the block value of its `.upit` file.

## What the script does

1. **The ultimate pit as MineLib publishes it**, with its own precedences (`newman1.prec`):
   `pymineflow.solve(csv, precedence=prec, value="VALUE")`. It prints the report, checks the pit against
   MineLib's published solution and writes `results/newman1_pit.csv` (the whole model with `PIT` = 1 inside),
   `results/newman1_pit_only.csv` and `results/newman1_pit_report.txt`.
2. **A full study** with the block size (unit cubes (size unknown), one level: the 5 blocks above), 45° and 1 level(s) that MineLib describes:
   100 nested pits (factors 1 % to 100 % of the values) with top-down bench discounting, then the ultimate
   pit OES, with the discount rate and capacities of MineLib's `.cpit` file:

   | Discount rate | Mining capacity | Processing capacity | Benches per year* |
   |---:|---:|---:|---:|
   | 8 % | 2,000,000 t a year | 1,100,000 t a year | 7 |

   \* MineLib does not give it: the benches of the final pit divided by its life. Ore: the blocks of positive
   value.

   It writes to `results/`: `report_cumulative.csv` and `report_incremental.csv` (every phase with its best and
   worst case NPV and the sequence), `model.csv` and `final_pit.csv`, `npv.png` (best against worst case),
   `oes.gif` (the final pit mined in the order of the sequence) and `oes_surface.gif` (the ground dug), both in
   plan view, and the sequence block by block (`oes.csv`, `oes_periods.csv`, `oes_report.txt`, `oes_npv.png`).
   Then it opens two 3-D windows with a slider along the sequence: the final pit mined away, and the ground
   surface dug.

   Result: 5 phases, a final pit of 1,060 blocks, OES NPV 22,177,220 over 3.24 periods (every
   block discounted when it is mined). `oes.check()` confirms that every block is mined once, after its
   predecessors, within the capacities.

   With slopes built from the geometry the pit differs slightly from MineLib's: MineLib drops the arcs to blocks missing from the model, PyMineFlow keeps the cone through them.

