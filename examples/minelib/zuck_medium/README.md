# MineLib: zuck_medium

Mine provided by Mark Zuckerberg; blocks of unknown size, precedences computed by the provider.

| Blocks | Precedence arcs | Published UPIT optimum | PyMineFlow | Blocks in the pit |
|---:|---:|---:|---:|---:|
| 29,277 | 1,271,207 | 1,075,124,490 | 1,075,124,490 (same pit, block for block) | 27,387 |

## Run

```bash
pip install pymineflow
python zuck_medium.py
```

The first run downloads the instance from [MineLib](https://minelib.org) (CC BY-SA 3.0) into `../data/` and
writes it as `zuck_medium.csv`: MineLib's `.blocks` file with the column names of its documentation (`ID, IX, IY, IZ,
...`) and `VALUE`, the block value of its `.upit` file.

## What the script does

1. **The ultimate pit as MineLib publishes it**, with its own precedences (`zuck_medium.prec`):
   `pymineflow.solve(csv, precedence=prec, value="VALUE")`. It prints the report, checks the pit against
   MineLib's published solution and writes `results/zuck_medium_pit.csv` (the whole model with `PIT` = 1 inside),
   `results/zuck_medium_pit_only.csv` and `results/zuck_medium_pit_report.txt`.

MineLib gives no block size or slope for this instance (its precedences were computed by the provider), so the
nested pits and the extraction sequence, which build the slopes from the geometry, do not apply. The script also
saves a plan view of the final pit (`results/zuck_medium_pit.png`) and opens a 3-D view.

