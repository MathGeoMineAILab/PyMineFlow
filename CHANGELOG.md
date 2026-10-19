# Changelog

## 0.1.0 — first release

Ultimate pit (maximum closure) with MineFlow's pseudoflow solver (Deutsch et al., 2022), plus nested pits, block
economics, top-down discounting and NPV, the ultimate pit extraction sequence (OES), reports, files and views.

### Pits
- `solve`, `ultimate_pit` and `PitModel`: ultimate pits from coordinates or indices, constant, azimuth-sector,
  per-domain or per-block slopes, models cut by topography; precedences as given with `ExplicitPit`,
  `solve(..., precedence=...)` or `--precedence` (a MineLib `.prec` file or arcs, blocks by `ID` column or by row).
- `nested_pits`: hundreds of nested pits by revenue, price or profit factor (`scale="revenue"`, `"price"`,
  `"profit"`), solved on rings between pits already solved, the two halves of every bisection at the same time
  (`threads=0`: every core), in C++, with the same pits as each factor alone; `verbose=True` prints every pit.
- `block_size` can be one number for cubic blocks.

### Economics
- `Economics`, `Metal`, `Process`, `Contaminant` and `value_blocks`: block values from grades, prices and costs
  (g/t, ppm, ppb, %, oz/t; several metals and processes; payable, selling costs, contaminant penalties and limits);
  `read_economics` and `economics_from_dict` read the same from JSON or a dict.
- Every price, selling cost, payable fraction, recovery, processing and mining cost, tonnage, density, penalty,
  threshold and limit can be a number, a column, an array with one value per block or a dict by domain
  (`domain="ROCK"`), in `Economics` and in the simple way of `value_blocks`; `density=` with `block_size=` instead
  of a tonnage.
- `Metal(..., recovered=True)` for grades that are already recovered.
- `block_values(..., rows=, grades=)` and `terms()`: values for a subset of rows or for other grades.

### Discounting and NPV
- Top-down bench discounting: `bench_discount=`, or `discount_rate=` with `benches_per_year=`;
  `bench_discount_weights` gives the weight of every block.
- `NestedPits.npv` and `pit_by_pit`: the best, worst and specified case NPV of every pit as the final pit, with
  mining and processing capacities per period (per process if needed) and fixed costs. Every block is discounted
  when it is mined (`timing="continuous"`, the default everywhere), or the cash of each period at its end, middle
  or start.

### Ultimate pit OES
- `NestedPits.oes()` and `ultimate_pit_oes`: a block-by-block extraction sequence of the final pit built on the
  nested pits, phase by phase and bench by bench, ordered by proximity or lookahead values, continuous in plan
  (`min_component`), with the slopes respected and its NPV under per-period capacities. The ordering, the lookahead
  values, the schedule and the check run in C++.
- `OES.check()`: an independent validation (completeness, order, slopes, value, capacities).
- `OES.summary()`, `report(kind)` and `report_table(kind)` (cumulative and incremental, with the best and worst case
  NPV and the sequence), `table()`, `model_table()`, `to_csv(pit=)`, `cases()`, `plot_cases()`, `plot_npv()`.
- `OES.export(folder, report=, pit=, chart=, gif=, details=)`: the reports, the whole model with the answer of every
  block or only the final pit, the best against worst case chart, `oes.gif` (the final pit mined away in order) and
  `oes_surface.gif` (the ground surface dug), and the sequence block by block.

### Reports, files and views
- `NestedPits.report(kind=)` and `report_table(kind=)`: cumulative or incremental, one line per phase, with the NPV
  options of `pit_by_pit`. `NestedPits.export(folder)`: the model with its pit numbers, the final pit, the reports,
  a plan view, the NPV chart and two GIFs, all optional.
- Views with a slider (jet, with its legend): `NestedPits.plot()` and `OES.plot()`; `remaining=True` shows the
  whole final pit mined away, `surface=True` the ground surface dug; `animate()` writes the same as GIFs.
- Charts fitted to every case: from the first pit with blocks, between the lowest and highest values, zeros lined
  up, the legend below.
- `NestedPits.compare` and `read_reference_pits`: comparison with nested pits exported by another program.

### Files and command line
- `read_blocks` and `write_table`: CSV or any delimited text (GSLIB included), Parquet, Feather, pandas, polars or
  pyarrow tables; MineLib readers.
- `pymineflow` command: the same features without writing Python (`--factors`, `--scale`, `--discount-rate`,
  `--benches-per-year`, `--ore-per-year`, `--export`, `--report`, `--oes`, `--precedence`, `--plot`, ...).
- Clear errors instead of guesses: ambiguous columns, numbers given as text or booleans, options that do not apply.

### Examples and validation
- `examples/minelib/`: every open-pit MineLib instance, documented in English and Spanish.
- MineLib (Espinoza et al., 2013): the published optimum and solution on every open-pit UPIT instance; nested pits
  closed under MineLib's precedences; the best case NPV below MineLib's LP bound.
- An independent commercial pit optimiser on two private deposits and on McLaughlin: the same nested pits; the OES
  follows its ultimate pit sequence (rank correlation 0.998).
