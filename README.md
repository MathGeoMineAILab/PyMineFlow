# PyMineFlow

Fast **ultimate pit** (maximum closure) for open-pit mining in Python, powered by the **MineFlow** pseudoflow solver (C++). PyMineFlow does not reimplement the algorithm. It provides pybind11 bindings to MineFlow plus a small, typed Python API.

It is meant for mining engineers, researchers and students who want ultimate pits from a block model in a few lines, with:

- regular block models given by **coordinates** or by **indices** (IX, IY, IZ) plus the block size;
- **constant**, **azimuth-sector**, **per-domain** or **per-block** slopes;
- precedences generated inside C++ (MineFlow's minimum search pattern), so no huge arc lists in Python;
- block values from your own profit column, or computed from **grades, prices and costs** (g/t, ppm, %; several metals and processes, payable, refining costs, contaminant penalties and limits), each of them a number, a column, an array or a value per domain;
- **hundreds of nested pits** by profit or price factor in seconds, with the same pits as solving each factor alone;
- **top-down bench discounting** of the pits (an annual rate over a sinking rate) and the **best and worst case NPV** of every pit, with mining and processing capacities per year;
- the **ultimate pit OES**: a block-by-block extraction sequence of the final pit, phase by phase, ordered by proximity or lookahead values with the slopes respected, and its NPV;
- a **phase report** and the files of a study in one call: the whole model with its pit numbers, the final pit alone, the report, a plan view and the NPV chart;
- detailed reports, CSV output, a **3-D view** (PyVista) with a slider through the pits, and a **command line** for those who do not write Python;
- a block-by-block and pit-by-pit **comparison** with nested pits exported by another program;
- models cut by topography (empty cells are zero-value blocks) and explicit precedences (`src`/`dst` arcs or CSR);
- block models from CSV or any delimited text (GSLIB included), Parquet, Feather or a DataFrame, and MineLib instances.

It is validated against the published optima of MineLib (see [Validation on MineLib](#validation-on-minelib)).

![100 nested pits of McLaughlin by price factor, in plan view, coloured by pit number](https://raw.githubusercontent.com/MathGeoMineAILab/PyMineFlow/main/assets/mclaughlin_price_pits.png)

*100 nested pits of MineLib's McLaughlin gold mine (2.1 million blocks) by price factor (gold from $270 to $1,170/oz), with the exact 45° cone over the full height, in plan view; solved in about 25 seconds. Code in [examples/mclaughlin_price_factors.py](https://github.com/MathGeoMineAILab/PyMineFlow/blob/main/examples/mclaughlin_price_factors.py).*

## Installation

```bash
pip install pymineflow     # Python 3.10-3.14; wheels for Windows, Linux x86_64 and macOS
```

That is all: it also installs what reading files (pandas, pyarrow) and the 3-D view (PyVista) need. `pip install "pymineflow[polars]"` adds polars as another, fast reader engine.

Building from source needs a C++17 compiler (Visual Studio Build Tools on Windows, g++ or clang elsewhere).

## Quick start

Four lines, from a block model file to a report, a CSV and a 3-D view:

```python
import pymineflow

pit = pymineflow.solve("model.csv", block_size=(10, 10, 5), slopes=45)
print(pit.summary())     # the model, the slopes and the pit: value, blocks, ore, waste, strip ratio
pit.to_csv("pit.csv")    # your model plus a PIT column (1 = inside the pit, 0 = outside); only_pit=True: the pit alone
pit.plot()               # 3-D view with a jet colour scale and its legend
```

`solve` finds the columns by their usual names, in any case:
- **Block indices:** `IX, IY, IZ` or `I, J, K`.
- **Block-centre coordinates:** `X, Y, Z`, `XC, YC, ZC`, `XCENTRE, ...` or `EAST, NORTH, RL`.
- **Value:** `VALUE`, `BLOCK_VALUE`, `BVALUE`, `VALOR` or `PROFIT`.
- **Tonnage, for the report:** `TON`, `TONNES`, `TONNAGE`.

`block_size=10` is enough for cubic blocks. Pass `value="NPV"`, `indices=("I", "J", "K")` or `coords=(...)` when the names differ, `n_benches=8` to limit the pattern, `z_up=False` when z is depth, and `slopes=[(0, 42), (90, 48), (180, 45), (270, 50)]` for slopes by azimuth. Real values are scaled to integers automatically; `pit.real_value` is in your units. The precedence pattern is chosen automatically: there is nothing else to set.

Part of the report on the McLaughlin model (2.1 million blocks, read and solved in 2.6 s):

```
ULTIMATE PIT
  value                 1,495,886,962
  blocks                110,624 (5.17% of the model)
  ore (value > 0)       30,831 blocks, worth 1,599,936,712
  waste (value <= 0)    79,793 blocks, costing 104,049,750
  tonnage (TON)         110,938,303.64
    ore                 31,884,233.05
    waste               79,054,070.59
  strip ratio           2.4794 (waste / ore tonnage)
  benches               45 (levels 63 down to 19; 0 = lowest)
```

`pit.benches()` gives the same, bench by bench (blocks, value and tonnage per level). `pit.plot(color_by="AU", view="below")` colours the pit by any numeric column (or `"value"`, `"ore"`, `"level"`) and looks at it from below, where its shape shows. `view="top"` is a plan view (`rotation=-90` turns it so that north points right), `outline=False` hides the model box, and `pit.plot(screenshot="pit.png")` saves the image.

**Without writing Python**, from a terminal:

```bash
pymineflow model.csv --block-size 10 10 5 --slope 45
pymineflow model.csv --block-size 10 10 5 --sectors 0:42 90:48 180:45 270:50 --benches 8 --plot
```

The command prints the report and writes `model_pit.csv` (`--only-pit`: only the blocks of the pit). `--bench-report benches.csv` also writes the bench table, `--screenshot pit.png` saves the 3-D view and `--bench-discount 0.0125` discounts the blocks by bench. With `--factors` it computes nested pits, `--discount-rate`, `--benches-per-year` and `--ore-per-year` add the discounting and the NPV, and `--export FOLDER` writes every file of the study (see [A whole study in a few lines](#a-whole-study-in-a-few-lines)). `pymineflow --help` lists every option.

### Valuing the block model

If the model has a value (profit) column, `solve` uses it as it is. If it only has grades, `value_blocks` computes the value of every block in one call:

```python
model = pymineflow.value_blocks(
    "model.csv",
    metals={"AU": {"price": 1800, "grade_unit": "g/t", "recovery": 0.90},     # g/t, ppm, ppb, % or oz/t
            "CU": {"price": 4.0, "grade_unit": "%", "recovery": 0.85}},      # % is priced in $/lb
    processing_cost=18.0,       # per tonne processed
    mining_cost=2.0,            # per tonne mined
    tonnage="TON",              # or density="DENSITY" (a column, a number, per block or by domain) and block_size
)
pit = pymineflow.solve(model, block_size=(10, 10, 5), slopes=45)   # uses the new VALUE column
```

- **Units:** each metal can also take `price_unit` (`"oz"`, `"g"`, `"kg"`, `"lb"`, `"t"`), `payable` (fraction paid) and `selling_cost` (refining or selling cost per price unit).
- **Contaminants:** `contaminants={"AS": {"penalty": 0.003, "threshold": 1000, "limit": 3000}}` adds a penalty per tonne and per grade unit above the threshold; above the limit the block is waste.
- **Destination:** a block goes to the process when that pays, otherwise to waste (the `DESTINATION` column).
- **Per block:** every number (price, recovery, payable, selling cost, processing and mining cost) can also be a column name (`mining_cost="MCOST"`), an array with one value per block, or a dict by domain with `domain="ROCK"` (`processing_cost={"oxide": 12, "sulphide": 20}`).
- **Several processes** (a mill and a leach pad, each with its recoveries, cost and contaminant limits): describe them with `pymineflow.Economics`, or with a dict or a JSON file of the same fields. `solve`, `nested_pits` and `value_blocks` take it as `economics=`:

```python
eco = pymineflow.Economics(
    metals=[pymineflow.Metal("AU", price=1800, grade_unit="g/t", price_unit="oz", payable=0.99, selling_cost=10)],
    processes=[pymineflow.Process("mill", cost=18, recovery={"AU": 0.92}, limits={"AS": 2000}),
               pymineflow.Process("leach", cost=4, recovery={"AU": 0.60})],
    mining_cost=2.0, tonnage="TON",
    contaminants=[pymineflow.Contaminant("AS", penalty=0.003, threshold=500)],
)
pit = pymineflow.solve("model.csv", block_size=(10, 10, 5), slopes=45, economics=eco)
print(pit.summary())     # ore by process, waste and strip ratio
```

Every number of `Economics` (price, selling cost, payable, recovery, processing and mining cost, tonnage, penalty, threshold, limit) can be a constant, a column name, an array with one value per block, or a dict by domain:

```python
eco = pymineflow.Economics(
    metals=[pymineflow.Metal("AU", price=1800, grade_unit="g/t", price_unit="oz", selling_cost="SELL_AU")],
    processes=[pymineflow.Process("mill", cost={"oxide": 14, "sulphide": 21},       # by domain
                                  recovery={"AU": "REC_AU"})],                     # a column
    mining_cost=mining_cost_array,                                                 # one value per block
    domain="ROCK", density="DENSITY", block_size=10,                               # tonnage = density x volume
)
```

A grade that is already recovered is declared with `Metal(..., recovered=True)`: no recovery is applied to it again (giving one is an error).

### Nested pits

```python
import numpy as np

pits = pymineflow.nested_pits("model.csv", block_size=(10, 10, 5), slopes=45,
                              factors=np.linspace(0.3, 1.3, 1000))              # profit factor
pits = pymineflow.nested_pits("model.csv", block_size=(10, 10, 5), slopes=45,
                              factors=np.linspace(0.3, 1.3, 1000), economics=eco, scale="price")
print(pits.summary())        # one line per pit: factor, blocks, new blocks, value, tonnage, strip ratio
print(pits.report())         # one line per phase (a factor whose pit grows): profit, rock, ore, strip ratio, waste
pits.to_csv("nested.csv")    # your model plus PIT_NUMBER (1 = smallest pit, 0 = outside)
pits.to_csv("final.csv", pit="final")   # only the blocks of the final pit (or pit=37, or pit="best" with the NPV)
pits.export("results")       # pits.csv, final_pit.csv, the reports, plan.png, GIFs (and npv.png)
pits.plot()                  # 3-D, coloured by pit number, with a slider from the first pit to the last
pits[10]                     # pit 11 as a full result (summary, benches, to_csv, plot)
```

```bash
pymineflow model.csv --block-size 10 10 5 --slope 45 --factors 0.3:1.3:0.001
pymineflow model.csv --block-size 10 10 5 --slope 45 --economics eco.json --scale price --factors 0.3:1.3:0.001
```

- **`scale="revenue"`** (the default): with a value column, each factor multiplies the positive block values (waste keeps its cost); with `economics`, it multiplies every price net of its selling cost and the destination of every block is chosen again (the usual price factors).
- **`scale="profit"`**: each factor multiplies the positive block values at factor 1, destinations fixed (profit factors), with `economics` or a value column.
- **`scale="price"`** (needs `economics`): each factor multiplies every metal price while selling costs stay fixed, and the destination is chosen again at each price.

No block loses value as the factor grows, so the pits are nested. PyMineFlow solves the largest one on the whole model and every other one only on the ring between two pits already solved, the inner one mined and the outer one as the limit. Each ring is solved in C++ straight from the precedence pattern, with the same slopes as the whole model. The pits are the same as solving each factor alone, block for block, and they are kept as a single array of pit numbers, so a thousand pits need little memory.

**1,000 nested pits on McLaughlin** (2.1 million blocks, 45°, full height, factors 0.30 to 1.30; reproduce with `python benchmarks/nested_pits.py --download --check 40`):

| Factor | Time | Solver | Distinct pits | Peak memory | Check |
|---|---:|---:|---:|---:|---|
| Profit | 37.5 s | 20.7 s | 228 | 154 MB | 40 pits solved again alone: identical |
| Price ($900/oz, 90 %, $19/t, $1.32/t) | 51.9 s | 25.5 s | 909 | 185 MB | 40 pits solved again alone: identical |

Solving the 1,000 pits one by one would take about 36 minutes (2.2 s each). `PitModel.nested_pits` and `PitModel.nested_pit_numbers` do the same for any list of value scenarios.

**Watching how the pits are solved**, with either factor:

```python
pits = pymineflow.nested_pits(..., verbose=True)  # one line per pit, in the order it is solved
pits.plot(view="top", rotation=-90, outline=False, screenshot="pits.png", show=True)  # image + window
pits.animate("nested_pits.gif")                   # optional: the pit growing, as a GIF (step=5: every fifth pit)
pits.animate("mined.gif", remaining=True)         # the whole final pit, the pits mined away one after another
pits.plot(remaining=True)                         # the same in the window, with the slider
pits.plot(surface=True)                           # the ground surface, dug pit after pit (also animate(surface=True))
```

`verbose=True` prints, for every pit, the ring it was solved on (between which two pits; 0 = no inner pit), the cells solved, its blocks, its value at its factor and the solver time. McLaughlin, 100 profit factors from 0.30 to 1.00, 8 levels:

```text
Solving 100 nested pits by revenue factor: the largest on the whole model, then by bisection,
each one only on the ring between two pits already solved (inner pit mined, outer pit as limit).
 order   pit revenue factor   ring between  cells solved     blocks              value       time
     1   100         1.0000    whole model     2,817,920    110,624      1,495,886,962   997.1 ms
     2    50         0.6465      0 and 100       331,647    102,205        932,358,016   957.7 ms
     3    25         0.4697       0 and 50       314,394     98,989        652,332,735   813.0 ms
   ...
     7     1         0.3000        0 and 3       291,720     89,476        385,878,586   712.1 ms
     8     2         0.3071        1 and 3           737     89,540        396,828,995     0.4 ms
     9     4         0.3212        3 and 6         6,442     93,039        418,760,449     6.1 ms
   ...
   100    99         0.9929     98 and 100            97    110,605      1,484,574,732     0.1 ms
```

The largest pit is solved on the whole model; the next ones, down to the smallest, only inside pit 50 (a tenth of the cells). From then on every pit lies between two pits already solved, and most of these rings have a few hundred or thousand cells and take under a millisecond.

`pits.plot()` is one view for profit and price factors: every block coloured by its pit number (jet, with its legend), the pit's factor, blocks and value on screen, and a slider that walks from the first pit to the last. The faces of all the pits are computed once (each one knows from which pit to which pit it is on the surface), so moving the slider only filters an array: about 15 ms per pit on McLaughlin, and a 100-frame GIF takes about 6 s. `view` (`"iso"`, `"below"`, `"top"`, `"side"`), `rotation`, `outline` and `title` work as in `PitResult.plot`; `screenshot=` saves the image, and with `show=True` the window opens too. With `remaining=True` the view and the GIF show the whole final pit, from which the slider removes the pits already mined: what is left stays on screen until the pit is mined out. From the command line, `--verbose` prints the solving lines and `--animate pits.gif` saves the animation.

### Top-down bench discounting

Blocks deeper in the deposit are mined later, so their value counts less. `bench_discount=` multiplies every block value, positive or negative, by `(1 + rate) ** -k`, where `k` is the number of benches between the top of the topography (the highest block) and the block. The rate per bench is the annual rate over the average sinking rate, which `discount_rate=` and `benches_per_year=` compute for you:

```python
pits = pymineflow.nested_pits("model.csv", block_size=10, slopes=45, n_benches=8, factors=np.arange(1, 101) / 100,
                              discount_rate=0.10, benches_per_year=8)   # 10 % / 8 = 1.25 % per bench
pit = pymineflow.solve("model.csv", block_size=10, slopes=45, n_benches=8, bench_discount=0.0125)
pymineflow.bench_discount_weights(iz, 0.0125)    # the weight of every block (iz: its bench index)
```

Only the block values change: the solver, the precedences and the speed are the same (100 nested pits of the full McLaughlin model take 12.8 s with or without discounting). The pit maximises the discounted value, so discounting can change it and make it smaller; it is not forced to. Because the weights are powers of one number, any other horizontal reference multiplies every value by the same constant and gives the same pits. In the reports, the profit stays undiscounted.

The two usual kinds of factors: **profit factors** multiply the positive block values at factor 1, destinations fixed (`scale="profit"`, with `economics=` or a value column); **price factors** multiply every price net of its selling cost and choose the destination again (`economics=` with `scale="revenue"`; `scale="price"` keeps the selling cost fixed).

### A whole study in a few lines

```python
for rate in (0.0, 0.10):                                   # without and with discounting
    pits = pymineflow.nested_pits("model.csv", block_size=10, slopes=45, n_benches=8, economics=eco,
                                  scale="profit", factors=np.arange(1, 101) / 100,
                                  discount_rate=rate, benches_per_year=8,   # optional: bench discounting
                                  ore_per_year=3_000_000)                    # optional: best and worst case NPV
    print(pits.report())                                   # one line per phase
    pits.export(f"results_{rate:.0%}")                     # every file of the study
pits.to_csv("final_pit.csv", pit="final")                  # only the blocks of the final pit
pits.plot(view="top", outline=False)                       # 3-D window, slider from the first pit to the last
```

```bash
pymineflow model.csv --block-size 10 --slope 45 --benches 8 --economics economics.json --scale profit \
    --factors 0.01:1:0.01 --discount-rate 0.10 --benches-per-year 8 --ore-per-year 3000000 --export results --plot
```

- `report()` prints one line per phase (a factor whose pit is larger than the previous one): factor, profit (undiscounted, at factor 1), profit per tonne, rock, ore, strip ratio and waste; with `discount_rate` and `ore_per_year`, also the best and worst case NPV of every phase as the final pit and the phase of highest NPV.
- `export(folder)` writes `pits.csv` (the whole model with the pit number of every block), `final_pit.csv` (only the blocks of the final pit), `report_cumulative.csv` and `report_incremental.csv` (`report=` chooses one), `plan.png` (plan view coloured by pit number), `nested_pits.gif` (the pit growing), `nested_pits_remaining.gif` (the final pit mined away) and, with the NPV, `npv.png` (best and worst case).
- `to_csv(path)` writes the whole model with its pit numbers; `to_csv(path, pit="final")` only the blocks of the final pit, `pit="best"` those of the pit of highest NPV, `pit=37` those of pit 37. With economics, the value and destination of every block are written too. A single pit from `solve` does the same with `pit.to_csv(path, only_pit=True)`; from the command line, `--only-pit`.
- Everything is optional and can be changed: `report_table()` returns the columns as arrays, `report(timing=...)` and `export(timing=...)` choose when the cash comes in each year, `export(files=[...], view=..., rotation=..., cmap=..., title=...)` chooses the files and the picture, `npv(...)` takes every option of the NPV (mining and processing capacities, fixed costs, pushbacks) and `plot(...)` every option of the view. The colour scale and the slider start at the first pit that has blocks, so low factors that give empty pits do not waste the legend.

[examples/mclaughlin_discounted_nested_pits.py](https://github.com/MathGeoMineAILab/PyMineFlow/blob/main/examples/mclaughlin_discounted_nested_pits.py) runs such a study on MineLib's McLaughlin model with the exact 45° cone over the full height: 100 price factors with and without discounting (15 % a year, a mill of 3.3 Mt a year, 8 benches a year), in about a minute. Discounting takes the final pit from 185,873 to 179,086 blocks; the highest best case NPV is 1,100 M at factor 1 and the highest worst case NPV 767 M at factor 0.29.

### Comparing with another program

`pymineflow.read_reference_pits("export.csv", pits.table)` reads nested pits exported by another program (block centres, model origin, block size and, per block, the factor in percent of its first pit) onto the blocks of the model, and `pits.compare(reference)` compares them block by block and pit by pit: the blocks of both pits, those only in each, the Jaccard index, tonnage, ore, waste, strip ratio, value, lowest bench and footprint.

Checked against an independent commercial pit optimiser on two private deposits (data not distributed; 45 degrees, 8 benches, factors from 1 % to 100 %):

- Deposit A (Au, Cu, Ag; 209,839 blocks; 1 % per bench): with and without discounting, every block enters the same nested pit (25,694 and 23,542 blocks in the final pit, 37 and 38 pits); discounting removes 2,152 blocks from the final pit.
- Deposit B (Au, Cu; 113,982 blocks; 1.25 % per bench): the same final pit (10,745 blocks); 98.2 % and 98.3 % of the blocks enter the same nested pit with price and profit factors. The other program also used a rock code that the block model file does not have; with it, 100 % and 99.7 %. The remaining 35 blocks enter one factor later in PyMineFlow, where their group has a negative discounted value (-1,986 and -433), so these PyMineFlow pits are worth slightly more.
- The NPV of the other program comes from its own extraction sequence inside each phase, which it does not export; PyMineFlow's best and worst case NPV (cash at mid-year) are within about 0.6 to 0.9 % of it, while the pits themselves are the same.

### Discounted value (NPV), pit by pit

Each nested pit is taken in turn as the final pit, mined in a fixed sequence at your capacities, and its cash flow is discounted: the pit-by-pit analysis (best and worst cases) of strategic mine-planning software. No pushbacks are designed and nothing is optimised beyond the nested pits.

```python
pits = pymineflow.nested_pits("model.csv", block_size=(10, 10, 5), slopes=45,
                              factors=np.geomspace(0.01, 1.2, 100))   # start small: see below
npv = pits.npv(discount_rate=0.10,              # per period (a year, if the capacities are per year)
               mining_rate=20e6,                # tonnage mined per period, ore and waste
               processing_rate=4e6)             # ore per period; or {"mill": 4e6, "leach": 2e6}
print(npv.summary())                            # one line per pit and the pit of highest NPV in each case
npv.plot("npv.png")                             # best and worst case against the factor, maxima marked
npv.schedule(npv.best_pit("best"))              # period by period: tonnage, ore, cash, discounted cash
npv.to_csv("npv.csv")
```

`nested_pits(..., discount_rate=0.10, ore_per_year=4e6)` sets the defaults of `npv()`, and then `report()` and `export()` add the best and worst case NPV of every phase, with the same convention as `npv()` (every block discounted when it is mined by default) and any of its options: `pits.report("end", mining_rate=6e6, fixed_cost=1e6)`, `pits.export(folder, timing="end", npv_options={"mining_rate": 6e6})`. One study, one NPV: the report, the exported tables, the chart and the pit-by-pit table agree.

```bash
pymineflow model.csv --block-size 10 10 5 --slope 45 --factors 0.01:1.2:0.01 \
    --discount-rate 0.10 --mining-rate 20e6 --processing-rate 4e6 --npv-report npv.csv --npv-chart npv.png
```

- **Best case:** the nested pits are the phases (pit 1, then the ring from pit 1 to pit 2, ...), each mined bench by bench from the top.
- **Worst case:** the final pit mined bench by bench from the top, each bench complete.
- **Specified case** (optional, `pushbacks=[...]`): your pushbacks, then the final pit, each bench by bench.

**Capacities hold per period**, as in strategic mine-planning software and in MineLib's scheduling problems: every period mines the sequence as far as it can without exceeding the mining capacity (all tonnage) or the processing capacity (ore tonnage, per process if you give one per process). Waste is still mined while the plant is full; a bench can be split between periods, its tonnage, ore and value in proportion; there is no stockpile. By default every block is discounted when it is mined, `exp(-rate × t)` with `t` the time elapsed in periods (`timing="continuous"`, the default of every function and of the command line). By period instead, cash of period *k* is discounted by `(1 + rate)^-k` (`"end"`), `(1 + rate)^-(k - 0.5)` (`"mid"`) or `(1 + rate)^-(k - 1)` (`"start"`, MineLib's convention). Fixed costs per period (`fixed_cost=`) are paid over the life of the mine, the last period in proportion to its use of the binding capacity.

Blocks are valued at factor 1 (the real prices and costs). With `economics`, each block goes to its best process. Without it, the processed blocks are those of positive value, or a 0/1 column such as MineLib's `DESTINATION` with `ore="DESTINATION"`, which also counts the marginal blocks that are processed at a loss smaller than the cost of dumping them. The tonnage comes from the economics, a column called TON, TONNES, TONNAGE, ..., or `tonnage=`.

**What "best" and "worst" mean.** Both sequences respect every precedence: a block's predecessors lie in its phase or an earlier one and, within a phase, on a higher bench. So both are feasible schedules, and both are *lower* bounds of the highest NPV a schedule can reach with the same capacities. The names say how the nested pits are used, not that they bound every schedule: the best case usually beats the worst case, but not always. On 12,800 random pit evaluations at 10 %, the worst case came out higher in 0 of 3,200 with a mining capacity only, 1 of 3,200 with a processing capacity only and 37 of 6,400 with both, where mining bench by bench can blend ore and waste better within a period.

**Start with small pits.** The best case follows the order of value only as finely as the nested pits split it. On McLaughlin with MineLib's scheduling data, the best case reaches 77.6 % of MineLib's best known schedule with factors from 0.30 (pit 1 already holds 93 % of the ore), 86.0 % from 0.05, and 93.3 % with 100 geometric factors from 0.01. Use geometric factors from about 0.01.

**Validation on MineLib's McLaughlin** (`mclaughlin_limit`: 112,687 blocks, 3,035,483 precedences, MineLib's values, tonnage, destinations and economics):

- All the nested pits (101 profit factors and 101 price factors) are closed under MineLib's precedences and have the same value as the pits solved on MineLib's own precedence graph; the pit at factor 1 is MineLib's published ultimate pit, 1,495,726,474.
- MineLib's processing economics ($900/oz, 90 % recovery, $19/ton processing, $1.32/ton mining) give the values and destinations of its `.pcpsp` file exactly.
- With MineLib's scheduling parameters (15 %, 3.3 Mst processed per period, cash at the start of each period, 15 periods), the best case of the pit at factor 1 is 1,000,915,482: 93.3 % of MineLib's best known CPIT solution (1,073,327,197, an optimised schedule) and below its LP bound (1,078,979,501), as a feasible schedule must be. Written block by block as a CPIT solution, it gives the same NPV with MineLib's objective, no precedence violation and at most 3.3 Mst processed in any period.
- Against MineLib's PCPSP (where the destination of each block is chosen period by period, raising the cut-off when the plant is the bottleneck), the best case reaches 80.8 %: the pit-by-pit analysis keeps each block's destination fixed.

The tests check every case against a period-by-period schedule written independently, unit by unit, on random models with one or two processes, contaminants, fixed costs, the three timings and random pushbacks. The table for 101 pits of the full McLaughlin model (2.1 million blocks) takes 0.06 s, the chart 0.8 s.

### Ultimate pit OES (optimal extraction sequence)

After the (top-down discounted) ultimate pit and before the pushbacks, strategic mine-planning software orders every block of the final pit into one extraction sequence and values it: the ultimate pit OES. PyMineFlow builds it on the nested pits, which already rank the blocks by value, for any block model:

```python
pits = pymineflow.nested_pits("model.csv", block_size=10, slopes=45, economics=eco, scale="price",
                              factors=np.arange(1, 101) / 100, discount_rate=0.10, benches_per_year=8,
                              ore_per_year=3e6)          # top-down discounting and the plant
print(pits.report())                                     # the phases, every block discounted when it is mined
oes = pits.oes()                                         # proximity values, bench by bench in every phase
oes = pits.oes(search="lookahead", lookahead_benches=6)  # lookahead values
print(oes.summary())        # the method, the NPV, the best place to stop, phase by phase and period by period
print(oes.check().summary())   # completeness, order, slopes, value and capacities, checked independently
print(oes.report())            # phase by phase: best and worst case NPV and the sequence (cumulative)
print(oes.report("incremental"))  # what every phase adds
oes.export("results")       # reports, model.csv, npv.png, oes.gif and oes_surface.gif (plan view)
oes.plot(remaining=True, view="iso")  # 3-D window: the final pit; the slider removes what is mined, in order
oes.plot(surface=True, view="iso")    # the ground surface (jet by elevation); the slider digs it along the sequence
```

**The sequence on McLaughlin** ([examples/mclaughlin_oes.py](https://github.com/MathGeoMineAILab/PyMineFlow/blob/main/examples/mclaughlin_oes.py): MineLib's model and economics, 100 price factors, 15 % a year, 8 benches a year and MineLib's mill of 3.3 Mt a year, proximity search): on the left, the final pit mined away in the order of the sequence, coloured by the position of every block in it;
on the right, the ground surface dug along the same sequence, coloured by elevation. `oes.export(folder)` writes both
GIFs (`oes.gif` and `oes_surface.gif`).

<p>
<img src="https://raw.githubusercontent.com/MathGeoMineAILab/PyMineFlow/main/assets/mclaughlin_oes.gif" alt="McLaughlin's final pit mined away in the order of the ultimate pit OES, plan view, coloured by the position in the sequence" width="49%">
<img src="https://raw.githubusercontent.com/MathGeoMineAILab/PyMineFlow/main/assets/mclaughlin_oes_surface.gif" alt="McLaughlin's ground surface dug along the ultimate pit OES, plan view, coloured by elevation" width="49%">
</p>

**The results, in one line.** `oes.export(folder)` writes, every part optional:

| Option | Files |
|---|---|
| `report="both"` (or `"cumulative"`, `"incremental"`, `None`) | `report_cumulative.csv`: every phase as the final pit (profit, profit per tonne, best and worst case NPV, the NPV of the sequence at the end of the phase, rock, ore, strip ratio, waste); `report_incremental.csv`: what every phase adds (the same columns as increments, and when the sequence starts and ends it) |
| `pit="whole"` (or `"final"`, `"both"`, `None`) | `model.csv`: the whole model with the answer of every block (`VALUE`, `DESTINATION`, `PIT_NUMBER`, `FINAL_PIT`, `OES_RANK`, `OES_PHASE`, `OES_PERIOD`); `final_pit.csv`: only the blocks of the final pit |
| `chart=True` | `npv.png`: best case against worst case NPV of every phase |
| `gif=True` | `oes.gif`: the whole final pit mined away in the order of the sequence (`view=`, `rotation=`), and `oes_surface.gif`: the ground surface dug along the sequence |
| `details=False` | with `True`, also the sequence block by block, the periods, the summary and check, and the NPV along the sequence |

Every piece is also a function: `oes.report_table(kind)` (arrays), `oes.to_csv(path, pit="whole")`, `oes.model_table()`, `oes.plot_cases()` (the chart), `oes.cases()` (every pit as the final pit, `pit_by_pit`), `oes.animate(path, remaining=True, color_by="rank")`, `oes.plot(...)`, `oes.plot_npv()`, `oes.table()`, `oes.periods`. The reports use the discount rate, capacities and timing of the sequence; `pits.report(kind="incremental")` and `pits.export(folder, report=...)` give the same reports for the nested pits alone.

```bash
pymineflow model.csv --block-size 10 --slope 45 --economics eco.json --scale price --factors 0.01:1:0.01 \
    --discount-rate 0.10 --benches-per-year 8 --ore-per-year 3e6 --oes oes --search lookahead --report both
```

- **Phases:** every nested pit with new blocks is a phase (`phases=[...]` merges them), mined in order; inside a phase the benches go from the top down.
- **Proximity values** (`search="proximity"`, the default): in every phase-bench the block of highest value first, then repeatedly the block that maximises *value − c × distance* to the last block. Low-value blocks next to rich ones move up and the sequence grows in contiguous groups. A step of one block width costs `proximity` times the mean absolute value of the pit blocks whose value is not zero (default 1; 0 orders by value alone).
- **Lookahead values** (`search="lookahead"`): the key of a block is its value plus the value of the pit blocks in its downward cone, `lookahead_benches` deep and discounted bench by bench at the top-down rate (`lookahead_scope="phase"` counts only the blocks of its phase). `bench_order=False` instead takes any *available* block of the phase (every slope predecessor mined), whatever its bench: it can add NPV with coarse phases, at the price of an irregular sequence.
- **Continuity:** a group of fewer than `min_component` blocks of one phase (27 by default, a 3 × 3 × 3 cube) that touches nothing mined before is not mined alone: it moves to the phase that first needs it and is mined with its surroundings. Such groups are single rich blocks near the surface that pay for themselves at the lowest factors. `min_component=0` turns it off.
- **Slopes:** the bench order respects them by construction (nested pits are closed and every precedence points upwards); `oes.check()` verifies every sequence independently against the precedence pattern, constraints through empty cells included.
- **NPV:** the sequence is mined at the capacities (mining, processing or both, per process if needed), blocks split when they do not fit. By default every piece is discounted when it is mined, `exp(-rate × t)` with `t` the time elapsed in periods (`timing="continuous"`, the convention of strategic mine-planning reports); `"end"`, `"mid"` and `"start"` discount by period instead. `pits.report()` and `pits.npv()` use the same convention by default. `oes.npv_curve` is the NPV had mining stopped after every block and `oes.best_stop()` the best place to stop.
- **Speed:** the ordering, the lookahead values, the schedule, the check and the animation's faces run in C++ or are computed once, in memory proportional to the pit. McLaughlin (2.1 million blocks, final pit of 179,086): 0.3 s, check 0.5 s; a synthetic model of 4.5 million blocks with a final pit of 1.6 million: 3.6 s, check 1.2 s (`python benchmarks/oes_scaling.py`).

**Checked against a commercial optimiser** on McLaughlin (MineLib's economics, exact 45° cone, 100 price factors, 15 % a year, 8 benches a year, a mill of 10 Mt a year): the nested pits are the same, the sequence follows the other program's ultimate pit sequence block by block (rank correlation 0.998), and with continuous discounting the NPV of the final pit is 1,824.6 M in the phase report (the other program: 1,825.6 M, −0.06 %) and 1,825.2 M for the OES; the worst case is within 0.3 %. Like any OES it is a sequence of high NPV, not a proven optimum, and it has no mining width or pushback design: it is the economic and spatial reference for the pushbacks.

### Example: 100 nested pits of McLaughlin

[examples/mclaughlin_nested_pits.py](https://github.com/MathGeoMineAILab/PyMineFlow/blob/main/examples/mclaughlin_nested_pits.py) downloads MineLib's McLaughlin model and solves it as MineLib does (45°, 8 levels):

```python
model = pymineflow.read_blocks("mclaughlin.blocks", names=["ID", "IX", "IY", "IZ", "BLOCK_VALUE", "TON", "DESTINATION", "AU"])
pits = pymineflow.nested_pits(model, block_size=(7.62, 7.62, 6.096), slopes=45, n_benches=8,
                              factors=np.linspace(0.30, 1.00, 100))
print(pits[-1].summary())   # ultimate pit: 110,624 blocks, value 1,495,886,962 (MineLib's optimum)
print(pits.summary())
pits.plot(view="top", screenshot="mclaughlin_100_pits.png")   # plan view
```

```
Nested pits: 100 revenue factors, 89 distinct pits
  pit   factor     blocks      new              value   value (factor 1)          tonnage  strip ratio
------------------------------------------------------------------------------------------------------
    1   0.3000     89,476   89,476        385,878,586      1,469,932,494       89,527,031       2.0107
    2   0.3071     89,540       64        396,828,995      1,470,083,707       89,585,927       2.0118
  ...
   99   0.9929    110,605       66      1,484,574,732      1,495,886,858      110,918,512       2.4790
  100   1.0000    110,624       19      1,495,886,962      1,495,886,962      110,938,304       2.4794
solver time 5.154 s for 100 pits
```

The whole script (reading 2.1 million blocks, 100 pits, an 85 MB CSV and the image) runs in about 11 s.

[examples/mclaughlin_price_factors.py](https://github.com/MathGeoMineAILab/PyMineFlow/blob/main/examples/mclaughlin_price_factors.py) makes the image at the top: the same model valued with MineLib's economics ($900/oz, 90 % recovery, $19/ton processing, $1.32/ton mining), 100 price factors and the exact cone (all 100 pits are different):

```python
economics = Economics(metals=[Metal("AU", price=900, grade_unit="oz/t", price_unit="oz")],
                      processes=[Process("mill", cost=19, recovery={"AU": 0.9})],
                      mining_cost=1.32, tonnage="TON", tonnage_unit="st")
pits = pymineflow.nested_pits(model, block_size=(7.62, 7.62, 6.096), slopes=45, economics=economics,
                              scale="price", factors=np.linspace(0.30, 1.30, 100))
pits.plot(view="top", rotation=-90, outline=False)   # plan view, north to the right, no model box
```

## Working with arrays

With **indices** and the block size:

```python
import numpy as np
from pymineflow import read_blocks, scale_values, ultimate_pit

d = read_blocks("model.csv", ["IX", "IY", "IZ", "VALUE"])   # or a pandas/polars DataFrame
values, factor = scale_values(d["VALUE"])                     # int64, times a power of ten when there are decimals
result = ultimate_pit(values, ix=d["IX"], iy=d["IY"], iz=d["IZ"], block_size=(10, 10, 5), slopes=45)

result.mask        # bool per block, in the order of your rows
result.value / factor, result.n_blocks
```

With block-centre **coordinates** (tolerates decimal noise and rows in any order):

```python
result = ultimate_pit(values, x=X, y=Y, z=Z, block_size=(10, 10, 5), slopes=45)
# z_up=False when Z (or IZ) grows downwards (depth)
```

## Reading block models

`read_blocks` returns `{column: numpy array}`. For text files it detects the separator (comma, semicolon, tab, pipe or whitespace), the header, a decimal comma and a GSLIB header, whatever the extension. When the layout is ambiguous it raises an error that asks for `sep`, `header` or `decimal` instead of guessing:

```python
read_blocks("model.csv")                                    # all columns
read_blocks("model.txt", ["IX", "IY", "IZ", "VALUE"], sep=";")
read_blocks("marvin.blocks", names=["id", "ix", "iy", "iz", "tonn", "au", "cu", "profit"])  # no header
read_blocks("model.gslib")
read_blocks("tons.csv", sep=";", decimal=".", header=True, encoding="latin-1")  # say it when in doubt
read_blocks("model.parquet", ["IX", "IY", "IZ", "VALUE"])   # Parquet/Feather read only these columns
read_blocks(df)                                             # pandas/polars DataFrame, pyarrow Table or dict
```

- `engine="auto"` uses the fastest library installed: pyarrow, then polars, then pandas. All three return the same numbers; missing values (empty, `NA`, `nan`, ...) become NaN.
- Without header or `names`, columns are named by position: `"0"`, `"1"`, ...
- Rows with more fields than the header raise an error. Text separated by irregular whitespace (aligned columns, runs of spaces) needs pandas. Compressed files must be decompressed first.

## Slopes

```python
# constant slope (degrees from the horizontal)
ultimate_pit(v, ix=ix, iy=iy, iz=iz, block_size=size, slopes=45)

# azimuth sectors: [(azimuth, slope), ...]; azimuth 0 = north (+y), 90 = east (+x), clockwise
ultimate_pit(v, ix=ix, iy=iy, iz=iz, block_size=size, slopes=[(0, 42), (90, 48), (180, 45), (270, 50)])

# per domain: one integer per block plus one slope definition per domain
ultimate_pit(v, ix=ix, iy=iy, iz=iz, block_size=size, domains=dom,
             domain_slopes=[[(0, 40)], [(0, 50), (180, 55)]])

# per block: continuous angles per row, (n,) or (n, k) with azimuths (k,), discretized every slope_step degrees
ultimate_pit(v, ix=ix, iy=iy, iz=iz, block_size=size, block_slopes=angles, slope_step=1.0)
```

- **`slopes=None`** uses the classic 1:5 block pattern.
- **Patterns with domains:** when domains have different slopes, `full_pattern=True` is used automatically, giving every block all the offsets of its own cone. The minimum pattern would lose constraints at domain boundaries. Forcing `full_pattern=False` then raises a warning.
- **`air_domain`** is the domain of the empty cells, which always have precedences:
  - `"envelope"` (default): the steepest slope of all domains;
  - `"column"`: the domain of the column's highest block; conservative, it may require extra blocks at domain boundaries;
  - an integer: that domain for every empty cell.
- **Height limit with domains:** the full pattern grows with the cone, so with different slopes per domain and the default full height it fits models of up to roughly 25 benches (at 45° with cubic blocks); taller models raise a clear error asking for a smaller `n_benches`.

## Choosing `n_benches`

`n_benches` is how many benches each block's precedence pattern covers. Above that height, constraints only arrive **in a chain**, and the chain does not always reproduce the cone.

- **`n_benches=None` (default):** uses the full model height (`nz − 1`). The slope holds over the whole height by construction.
- **The model is solved on its own grid**; no cells are added around it. With slopes that vary by azimuth, the minimum pattern reaches part of the cone through chains of offsets, and near the model's side edges a chain can need cells outside the model, so a few constraints may be lost there (in random tests with strongly contrasting sectors, about 1 % of the configurations). `full_pattern=True` gives every block its whole cone inside the model and avoids it, but it is only practical on models of moderate height.
- **An integer:** used as is, with no checks or warnings. Fewer benches is **faster and uses less memory**, but the pit **may violate the slope**. A value above the model height equals `None`.

Sweep on the McLaughlin model (2,140,342 blocks, 7.62 × 7.62 × 6.096 m blocks, 45°, 68 benches). The last column counts pit blocks that have an unmined block of their own cone anywhere above them (geometric audit over the full height):

| n_benches | Pit blocks | Value | Blocks violating the slope |
|---|---|---|---|
| 8 | 110,624 | 1,495,886,962 | 2,108 |
| 12 | 111,950 | 1,493,622,579 | 382 |
| 16 | 112,422 | 1,493,172,019 | 128 |
| 24 | 112,587 | 1,492,991,167 | 54 |
| 40 | 112,687 | 1,492,897,346 | 0 |
| 67 (= `None`) | 112,687 | 1,492,897,346 | 0 |

With 8 benches, a common choice, this pit overstates the value by 0.20 % and breaks the slope. If the pattern would be disproportionate (a flat slope with thin blocks and many benches, or `full_pattern` on a very tall model), PyMineFlow raises a clear error asking for a smaller `n_benches` before allocating memory.

## Conventions and limits

- **Slope rule ("origin"):** each block requires its own cone, built with the slope of its own domain (or block), measured centre to centre with the boundary included. For example, a wall whose lower block lies in a 40° domain and whose upper part crosses a 60° domain uses 40° along the whole line from the lower block. Rules that use the slope of every material the line crosses are planned, not implemented.
- **Sectors:** the slope is interpolated linearly in azimuth between them.
- **Grid:** regular and axis-aligned. **Rotated models and sub-blocks are not supported**; both raise an explicit error, as do duplicate rows and NaN coordinates.
- **Empty cells** are zero-value blocks inside the grid. A very sparse model wastes memory and raises a warning (density below 0.2).
- **Values** are int64. Use `scale_values` for real numbers: it multiplies them by the largest power of ten (up to 10⁶) that keeps the sums exact. A sum of absolute values beyond int64 raises `OverflowError`. Floats, booleans and values beyond int64 passed to the solvers raise an error instead of being truncated.
- **Axes:** z grows upwards. Linear cell index `(iz*ny + iy)*nx + ix`.

## More API

```python
import numpy as np
from pymineflow import (ExplicitPit, PitModel, bench_discount_weights, nested_pits, pit_by_pit,
                        read_minelib_precedence, read_minelib_values, read_reference_pits, scale_values, solve,
                        ultimate_pit_oes)

model = PitModel(ix=ix, iy=iy, iz=iz, block_size=size, slopes=45)    # grid and precedences, built once
scenario = lambda f: np.where(v > 0, np.rint(v * f), v).astype(np.int64)  # price factor
pits = model.nested_pits([scenario(f) for f in (0.6, 0.8, 1.0, 1.2)])    # nested: solved on rings

ExplicitPit(n, src, dst, values).solve()               # block src[k] requires block dst[k]
ExplicitPit.from_csr(indptr, indices, values).solve()  # e.g. a scipy.sparse.csr_matrix

nested = nested_pits("model.csv", block_size=10, slopes=45, factors=np.arange(1, 101) / 100)
weights = bench_discount_weights(iz, 0.0125)            # top-down bench discounting, to apply to your own values
npv = pit_by_pit(nested, discount_rate=0.1, processing_rate=4e6)  # the same as nested.npv(...)
table = nested.report_table()                            # the phase report as numpy arrays
reference = read_reference_pits("export.csv", nested.table)       # nested pits exported by another program
comparison = nested.compare(reference)                   # PitComparison: .summary(), .table, .to_csv()
oes = ultimate_pit_oes(nested, discount_rate=0.1, processing_rate=4e6)  # the same as nested.oes(...)

src, dst = read_minelib_precedence("marvin.prec")      # MineLib files
values, factor = scale_values(read_minelib_values("marvin.upit"))
mask, value, n_blocks, seconds = ExplicitPit(values.size, src, dst, values).solve()
pit = solve("marvin.csv", precedence="marvin.prec", value="VALUE")  # the same in one line (blocks by ID or row)
```

`Pit` (low level) solves a full regular grid with values in x → y → z order. Every solver releases the GIL while solving, so different objects can solve in parallel threads; a second `solve` or `solve_with_values` on the **same** object while one is running raises an error (use one object per thread). `Pit.solve_subset` builds its own solver at every call, so several threads may call it on one object: the nested pits solve the two halves of every bisection at the same time this way (`PitModel.nested_pit_numbers(..., threads=0)`, every core by default; the pits do not depend on it).

## Validation on MineLib

**Every MineLib instance as an example**, in English and Spanish: [examples/minelib](https://github.com/MathGeoMineAILab/PyMineFlow/tree/main/examples/minelib) has one folder per open-pit instance with a script that downloads the data, solves the ultimate pit with MineLib's precedences (the published solution, block for block) and, when MineLib documents the block size and slope, runs a full study: nested pits with top-down discounting, the OES, the reports and the GIFs.

[MineLib](https://minelib.org) publishes the optimal ultimate pit (UPIT) of each instance. Reproduce with `python benchmarks/minelib.py --download --compare --repeat 5` (the comparison needs `pip install PyMaxflow pseudoflow networkx`). Median of 5 runs (pseudoflow: one run); i7-9700K, Windows 11, Python 3.13. Each time includes building the graph from the arrays already in memory.

**MineLib's own precedence graph** (`ExplicitPit`, or `solve(table, precedence="name.prec")` and `pymineflow name.csv --precedence name.prec` from a table with an `ID` column). The 10 smaller open-pit instances give the published optimum and the same pit, block for block. The largest one, McLaughlin (2.1 million blocks, 73 million arcs), is not in the script; its published optimum, 1,495,886,962, is exactly the `n_benches=8` value of `ultimate_pit` in [Choosing `n_benches`](#choosing-n_benches).

| Instance | Blocks | Arcs | Optimum (published = PyMineFlow) | PyMineFlow | PyMaxflow | pseudoflow* |
|---|---:|---:|---:|---:|---:|---:|
| newman1 | 1,060 | 3,922 | 26,086,899 | 1 ms | < 1 ms | 3.1 s† |
| zuck_small | 9,400 | 145,640 | 1,422,726,898 | 12 ms | 16 ms | 0.4 s |
| kd | 14,153 | 219,778 | 652,195,037 | 13 ms | 21 ms | 0.6 s |
| zuck_medium | 29,277 | 1,271,207 | 1,075,124,490 | 56 ms | 420 ms | 3.4 s |
| p4hd | 40,947 | 738,609 | 293,373,256 | 43 ms | 146 ms | 2.1 s |
| marvin | 53,271 | 650,631 | 1,415,655,436 | 14 ms | 50 ms | 1.9 s |
| w23 | 74,260 | 764,786 | 510,973,998 | 76 ms | 154 ms | 2.4 s |
| zuck_large | 96,821 | 1,053,105 | 122,220,280 | 142 ms | 150 ms | 3.4 s |
| sm2 | 99,014 | 96,642 | 2,743,603,730 | 42 ms | 8 ms | 0.7 s |
| mclaughlin_limit | 112,687 | 3,035,483 | 1,495,726,474 | 188 ms | 512 ms | 8.3 s |

\* The `pseudoflow` package (Hochbaum's algorithm, the one MineLib used) builds its graph with networkx, which dominates its time. PyMaxflow is faster on sm2, whose graph only links each block to the one above. † Includes importing networkx on the first call.

**From block coordinates and a 45° slope** (`ultimate_pit`), for the instances whose block size is documented:

| Instance | Block size | `n_benches=8` (MineLib's choice) | `n_benches=None` (full height) |
|---|---|---|---|
| kd | 20 × 20 × 15 | published optimum | −0.017 % |
| marvin | 30 × 30 × 30 | published optimum | −0.119 % |
| mclaughlin_limit | 25 × 25 × 20 ft | published optimum | −0.189 % |
| p4hd | 50 × 50 × 20 ft | −0.015 % | −0.320 % |
| newman1 | unknown (1:5 pattern) | −0.013 % (`n_benches=1`) | −0.013 % |
| w23 | 25 × 25 × 20 ft | −11.5 % | −12.6 % |

- With 8 benches, PyMineFlow's pattern has exactly MineLib's arcs on kd, marvin, mclaughlin_limit, p4hd and newman1.
- **p4hd and newman1** differ slightly because MineLib drops the arcs that point to blocks missing from the model. PyMineFlow keeps the cone through those cells (zero-value blocks), so its pit respects the slope there too.
- **w23:** MineLib's graph is not a symmetric 45° cone. Two levels up, it lacks the (+1, +1) and (+1, −1) diagonals. Its published pit breaks a true 45° cone on 14,551 arcs.
- **Full height:** `n_benches=None` gives lower values than 8 benches, which miss part of the cone (see [Choosing `n_benches`](#choosing-n_benches)).

## Performance

These numbers come from single runs. Treat them as orders of magnitude.

**McLaughlin model** (2,140,342 blocks; i7-9700K, 16 GB, Windows 11, Python 3.13, MSVC 19.50; one run each, data already loaded):

| Configuration | `ultimate_pit` | Solver only | Peak RAM | Pit |
|---|---|---|---|---|
| `n_benches=8` | 1.3 s | 0.9 s | 0.9 GB | 110,624 blocks |
| `n_benches=None` (full height, 67 benches) | 2.8 s | 2.0 s | 2.5 GB | 112,687 blocks |

Peak RAM includes Python, NumPy, the reader and the data.

**Comparison with PyMaxflow:** measured in a 2-core Linux container, **one run per engine**, on the 8-bench graph (same pit), with a development build of the bindings:

| Engine | Total | Solve only | Peak RAM |
|---|---|---|---|
| PyMaxflow with Python loops | 64.4 s | 11.3 s | 5,114 MB |
| PyMaxflow with vectorized edges | 14.7 s | 5.5 s | 5,426 MB |
| MineFlow from Python (explicit precedence) | 8.9 s | 0.39 s | 2,708 MB |
| MineFlow C++ executable (`--minsearch`) | 1.3 s | 0.78 s | 476 MB |

MineFlow does **not** always win: on synthetic regular models, vectorized PyMaxflow ties or wins in solver time, and on MineLib's sm2 (each block linked only to the one above) PyMaxflow is faster. On the other MineLib instances PyMineFlow was 1.1–7.5× faster than PyMaxflow (table above), and it uses much less memory. Most of the gain over a typical script comes from building the graph without Python loops (vectorized, or better, inside C++), not from the solver.

## Differences from the original MineFlow

PyMineFlow fixes a slope-interpolation bug in MineFlow; see [PATCHES.md](https://github.com/MathGeoMineAILab/PyMineFlow/blob/main/PATCHES.md). With **non-constant slopes** (several sectors), results in the **north-west quadrant** (azimuth 270° to 360°) differ from the original MineFlow, including its `--minsearch` executable. With a constant slope they are identical.

## References

- Deutsch, M., Dağdelen, K., & Johnson, T. (2022). An open-source program for efficiently computing ultimate pit limits: MineFlow. *Natural Resources Research*. https://doi.org/10.1007/s11053-022-10035-w
- Hochbaum, D. S. (2008). The pseudoflow algorithm: a new algorithm for the maximum-flow problem. *Operations Research*, 56(4), 992–1009. https://doi.org/10.1287/opre.1070.0447
- Picard, J.-C. (1976). Maximal closure of a graph and applications to combinatorial problems. *Management Science*, 22(11), 1268–1272. https://doi.org/10.1287/mnsc.22.11.1268
- Lerchs, H., & Grossmann, I. F. (1965). Optimum design of open-pit mines. *Canadian Mining and Metallurgical Bulletin*, 58, 47–54.
- Gallo, G., Grigoriadis, M. D., & Tarjan, R. E. (1989). A fast parametric maximum flow algorithm and applications. *SIAM Journal on Computing*, 18(1), 30–55. https://doi.org/10.1137/0218003
- Espinoza, D., Goycoolea, M., Moreno, E., & Newman, A. (2013). MineLib: a library of open pit mining problems. *Annals of Operations Research*, 206(1), 93–114. https://doi.org/10.1007/s10479-012-1258-3
- Deutsch, C. V., & Journel, A. G. (1998). *GSLIB: Geostatistical Software Library and User's Guide* (2nd ed.). Oxford University Press.

Top-down bench discounting (each block discounted by the number of benches between it and the top of the model) is a common approximation in pit optimisation.

## Citation and license

MIT; see [LICENSE](https://github.com/MathGeoMineAILab/PyMineFlow/blob/main/LICENSE). Includes [MineFlow](https://github.com/MineFlowCSM/MineFlow) in `third_party/mineflow` (© 2022 Matthew Deutsch, MIT, original license kept), with the patch described in [PATCHES.md](https://github.com/MathGeoMineAILab/PyMineFlow/blob/main/PATCHES.md).

If you use this package, please cite PyMineFlow (see [CITATION.cff](https://github.com/MathGeoMineAILab/PyMineFlow/blob/main/CITATION.cff)) and MineFlow, the solver it wraps:

> Deutsch, M., Dağdelen, K., Johnson, T. (2022). *An Open-Source Program for Efficiently Computing Ultimate Pit Limits: MineFlow*. Natural Resources Research. https://doi.org/10.1007/s11053-022-10035-w

and the methods you use, from the references above.

## Contact

Paul Alejandro Espinoza Tenecora — mathgeomine@gmail.com — [MathGeoMineAILab](https://github.com/MathGeoMineAILab)
