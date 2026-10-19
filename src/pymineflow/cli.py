"""Command line: ``pymineflow model.csv --block-size 10 10 5 --slope 45``.

Reads the model, solves the ultimate pit, prints a detailed report, writes the model with a PIT
column (``model_pit.csv``) and, on request, a bench-by-bench table and a 3-D view.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .economics import read_economics
from .readers import write_table
from .simple import nested_pits, solve

EXAMPLES = """examples:
  pymineflow model.csv --block-size 10 10 5 --slope 45
  pymineflow model.csv --block-size 10 10 5 --slope 45 --benches 8 --value PROFIT
  pymineflow kd.csv --precedence kd.prec --value VALUE   (precedences as given, e.g. a MineLib .prec file)
  pymineflow model.txt --block-size 25 25 20 --sectors 0:42 90:48 180:45 270:50 --plot
  pymineflow model.csv --block-size 10 10 5 --slope 45 --coords XC YC ZC --z-down
  pymineflow model.csv --block-size 10 10 5 --slope 45 --factors 0.3:1.0:0.05   (nested pits)
  pymineflow model.csv --block-size 10 10 5 --slope 45 --economics eco.json --scale price --factors 0.5:1.5:0.01
  pymineflow model.csv --block-size 10 --slope 45 --factors 0.01:1:0.01 --discount-rate 0.1
      --benches-per-year 8 --ore-per-year 3e6 --oes oes --search lookahead   (ultimate pit OES, one line)
"""


def _sector(text: str) -> tuple[float, float]:
    try:
        azimuth, slope = text.split(":")
        return float(azimuth), float(slope)
    except ValueError:
        raise argparse.ArgumentTypeError(f"a sector is AZIMUTH:SLOPE (e.g. 90:48), got {text!r}") from None


def _factors(texts: Sequence[str]) -> list[float]:
    """'0.5 0.75 1' or a range 'START:STOP:STEP' (STOP included)."""
    out: list[float] = []
    for text in texts:
        if ":" in text:
            try:
                start, stop, step = (float(t) for t in text.split(":"))
            except ValueError:
                raise ValueError(f"a factor range is START:STOP:STEP (e.g. 0.3:1.0:0.05), got {text!r}") from None
            if step <= 0 or stop < start:
                raise ValueError(f"a factor range needs STEP > 0 and STOP >= START, got {text!r}")
            count = math.floor((stop - start) / step + 1e-9) + 1  # STOP included, never passed
            out += [round(start + k * step, 10) for k in range(count)]
        else:
            out.append(float(text))
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pymineflow",
        description="Ultimate pit of a block model (CSV or any delimited text, Parquet, Feather).",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", type=Path, help="block model file")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--block-size",
        nargs="+",
        type=float,
        metavar="SIZE",
        help="block size: DX DY DZ, or one number for cubic blocks (optional with --precedence)",
    )
    slope = parser.add_mutually_exclusive_group(required=True)
    slope.add_argument("--slope", type=float, help="slope in degrees from the horizontal")
    slope.add_argument("--sectors", nargs="+", type=_sector, metavar="AZ:SLOPE", help="slope by azimuth (0 = north)")
    slope.add_argument(
        "--precedence",
        type=Path,
        metavar="PREC",
        help="instead of a slope: the precedences as given, a MineLib .prec file (blocks by ID column or by row)",
    )
    parser.add_argument("--benches", type=int, default=None, help="benches of the pattern (default: full height)")
    parser.add_argument(
        "--factors",
        nargs="+",
        metavar="F",
        help="nested pits for these revenue factors (e.g. 0.5 0.75 1 or 0.3:1.0:0.05)",
    )
    parser.add_argument("--value", help="value column (found by name when omitted)")
    parser.add_argument("--economics", type=Path, metavar="JSON", help="block values from grades, prices and costs")
    parser.add_argument(
        "--price-factor", type=float, help="with --economics, one pit: multiplies every price (default 1)"
    )
    parser.add_argument(
        "--scale",
        choices=("revenue", "price", "profit"),
        help="with --factors: what they multiply (default revenue; price needs --economics)",
    )
    position = parser.add_mutually_exclusive_group()
    position.add_argument("--indices", nargs=3, metavar=("IX", "IY", "IZ"), help="index columns")
    position.add_argument("--coords", nargs=3, metavar=("X", "Y", "Z"), help="block-centre coordinate columns")
    parser.add_argument("--z-down", action="store_true", help="z grows downwards (depth)")
    parser.add_argument(
        "--bench-discount",
        type=float,
        default=0.0,
        metavar="RATE",
        help="top-down bench discounting, rate per bench (e.g. 0.0125 = 10%% a year / 8 benches a year)",
    )
    parser.add_argument("--sep", help="text separator (detected when omitted)")
    parser.add_argument("--output", type=Path, help="output CSV (default: MODEL_pit.csv)")
    parser.add_argument("--no-output", action="store_true", help="do not write the output CSV")
    parser.add_argument("--only-pit", action="store_true", help="the output CSV holds only the blocks of the final pit")
    parser.add_argument(
        "--bench-report",
        type=Path,
        metavar="CSV",
        help="also write the pit bench by bench (with --factors: pit by pit)",
    )
    parser.add_argument("--verbose", action="store_true", help="with --factors: print every pit as it is solved")
    parser.add_argument(
        "--animate", type=Path, metavar="GIF", help="with --factors: save a GIF of the pit growing (needs PyVista)"
    )
    npv = parser.add_argument_group("discounted value of the nested pits (with --factors)")
    npv.add_argument("--discount-rate", type=float, metavar="RATE", help="a year, e.g. 0.10: NPV of every pit")
    npv.add_argument(
        "--benches-per-year",
        type=float,
        metavar="N",
        help="with --discount-rate: top-down bench discounting of the pits (rate / N per bench)",
    )
    npv.add_argument("--ore-per-year", type=float, metavar="T", help="with --discount-rate: ore a year, for the NPV")
    parser.add_argument(
        "--export",
        type=Path,
        metavar="FOLDER",
        help="with --factors: write pits.csv, final_pit.csv, the reports, plan.png, the GIFs (and npv.png) to FOLDER",
    )
    parser.add_argument(
        "--report",
        choices=["both", "cumulative", "incremental"],
        help="with --factors: the phase reports printed and written by --export and --oes (default both)",
    )
    npv.add_argument("--mining-rate", type=float, metavar="T", help="tonnage mined per period (ore and waste)")
    npv.add_argument("--processing-rate", type=float, metavar="T", help="ore tonnage processed per period")
    npv.add_argument("--fixed-cost", type=float, default=0.0, metavar="COST", help="cost per period (default 0)")
    npv.add_argument("--pushbacks", type=int, nargs="+", metavar="PIT", help="pit numbers of the specified pushbacks")
    npv.add_argument(
        "--timing",
        choices=["end", "mid", "start", "continuous"],
        help="when the cash comes: continuous (default: every block discounted when it is mined), or at the end, "
        "mid or start of each period",
    )
    npv.add_argument("--mid-period", action="store_true", help="same as --timing mid")
    npv.add_argument(
        "--ore", metavar="COLUMN", help="without --economics: column of processed blocks (0/1), e.g. DESTINATION"
    )
    npv.add_argument("--npv-report", type=Path, metavar="CSV", help="also write the pit-by-pit table")
    npv.add_argument("--npv-chart", type=Path, metavar="PNG", help="also save the best/worst case chart")
    oes = parser.add_argument_group("ultimate pit OES (with --factors and --discount-rate)")
    oes.add_argument(
        "--oes",
        type=Path,
        metavar="FOLDER",
        help="block-by-block extraction sequence of the final pit and its NPV, written to FOLDER",
    )
    oes.add_argument(
        "--search",
        choices=("proximity", "lookahead"),
        default=None,
        help="with --oes: proximity (block value, default) or lookahead (value of the cone below)",
    )
    oes.add_argument(
        "--proximity", type=float, default=None, metavar="W", help="with --oes: distance weight (default 1)"
    )
    oes.add_argument(
        "--lookahead-benches",
        type=int,
        default=None,
        metavar="N",
        help="with --oes --search lookahead: depth of the lookahead cone (default 6)",
    )
    oes.add_argument(
        "--min-component",
        type=int,
        default=None,
        metavar="N",
        help="with --oes: isolated groups of a phase under N blocks are mined with their surroundings "
        "(default 27; 0 = off)",
    )
    parser.add_argument("--plot", action="store_true", help="open a 3-D view (needs PyVista)")
    parser.add_argument("--screenshot", type=Path, metavar="PNG", help="save the 3-D view to an image")
    parser.add_argument(
        "--color-by", help="one pit: value, ore, level or a column (default value; nested pits: by pit number)"
    )
    return parser


def _nested(args: argparse.Namespace, options: dict[str, Any]) -> int:
    pits = nested_pits(
        args.model,
        block_size=tuple(args.block_size) if len(args.block_size) == 3 else args.block_size[0],
        slopes=args.slope if args.slope is not None else args.sectors,
        factors=_factors(args.factors),
        scale=args.scale or "revenue",
        value=args.value,
        economics=None if args.economics is None else read_economics(args.economics),
        indices=args.indices,
        coords=args.coords,
        n_benches=args.benches,
        z_up=not args.z_down,
        verbose=args.verbose,
        bench_discount=args.bench_discount,
        discount_rate=args.discount_rate or 0.0,
        benches_per_year=args.benches_per_year,
        ore_per_year=args.ore_per_year,
        **options,
    )
    if args.verbose:
        print()
    timing = args.timing or ("mid" if args.mid_period else "continuous")
    npv_options = {k: v for k, v in (("mining_rate", args.mining_rate), ("processing_rate", args.processing_rate),
                                     ("ore", args.ore)) if v is not None}  # fmt: skip
    if args.fixed_cost:
        npv_options["fixed_cost"] = args.fixed_cost
    for kind in ("cumulative", "incremental") if (args.report or "both") == "both" else (args.report,):
        print(pits.report(timing, kind, **npv_options))
    if not args.no_output:
        output = args.output or args.model.with_name(f"{args.model.stem}_nested.csv")
        print()
        print(f"pit numbers written to {pits.to_csv(output, pit='final' if args.only_pit else None)}")
    if args.bench_report is not None:
        print(f"pit table written to {write_table(pits.table_by_pit(), args.bench_report)}")
    if args.export is not None:
        print(
            f"written to {args.export}: "
            + ", ".join(
                f.name
                for f in pits.export(args.export, timing=timing, report=args.report or "both", npv_options=npv_options)
            )
        )
    wanted = (args.mining_rate, args.processing_rate, args.pushbacks, args.npv_report, args.npv_chart)
    if any(o is not None for o in wanted) or args.fixed_cost:
        result = pits.npv(discount_rate=args.discount_rate, pushbacks=args.pushbacks, timing=timing, **npv_options)
        print()
        print(result.summary())
        if args.npv_report is not None:
            print(f"pit-by-pit table written to {result.to_csv(args.npv_report)}")
        if args.npv_chart is not None:
            label = {"revenue": "revenue factor", "price": "price factor", "profit": "profit factor"}[pits.scale]
            result.plot(args.npv_chart, factor_label=label)
            print(f"pit-by-pit chart saved to {args.npv_chart}")
    if args.oes is not None:
        options = {k: v for k, v in (("search", args.search), ("proximity", args.proximity),
                                     ("lookahead_benches", args.lookahead_benches),
                                     ("min_component", args.min_component)) if v is not None}  # fmt: skip
        oes = pits.oes(discount_rate=args.discount_rate, mining_rate=args.mining_rate,
                       processing_rate=args.processing_rate, fixed_cost=args.fixed_cost, timing=timing, ore=args.ore,
                       **options)  # fmt: skip
        print()
        print(oes.summary())
        print()
        print(oes.check().summary())
        print(
            f"ultimate pit OES written to {args.oes}: "
            + ", ".join(f.name for f in oes.export(args.oes, report=args.report or "both", pit="both", details=True))
        )
    if args.animate is not None:
        print(f"animation saved to {pits.animate(args.animate)}")
    if args.plot or args.screenshot is not None:
        pits.plot(screenshot=args.screenshot, show=args.plot)
        if args.screenshot is not None:
            print(f"3-D view saved to {args.screenshot}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit code (0 on success)."""
    args = _parser().parse_args(argv)
    try:
        options = {} if args.sep is None else {"sep": args.sep}
        npv_options = [args.mining_rate, args.processing_rate, args.pushbacks, args.npv_report, args.npv_chart,
                       args.timing, args.ore]  # fmt: skip
        if args.discount_rate is None and (
            any(o is not None for o in npv_options) or args.fixed_cost or args.mid_period
        ):
            raise ValueError("the pit-by-pit options need --discount-rate")
        if args.discount_rate is not None and args.factors is None:
            raise ValueError("--discount-rate needs nested pits: add --factors")
        if args.timing is not None and args.mid_period and args.timing != "mid":
            raise ValueError("--mid-period and --timing disagree")
        rates = [args.mining_rate, args.processing_rate, args.ore_per_year, args.benches_per_year]
        if args.discount_rate is not None and all(r is None for r in rates):
            raise ValueError(
                "--discount-rate needs --benches-per-year, --ore-per-year, --mining-rate or --processing-rate"
            )
        if args.discount_rate is None and (args.benches_per_year is not None or args.ore_per_year is not None):
            raise ValueError("--benches-per-year and --ore-per-year need --discount-rate")
        if args.export is not None and args.factors is None:
            raise ValueError("--export needs nested pits: add --factors")
        if args.oes is not None and (args.factors is None or args.discount_rate is None):
            raise ValueError("--oes needs nested pits and a discount rate: add --factors and --discount-rate")
        oes_options = (args.search, args.proximity, args.lookahead_benches, args.min_component)
        if args.oes is None and any(o is not None for o in oes_options):
            raise ValueError("--search, --proximity, --lookahead-benches and --min-component need --oes FOLDER")
        if args.factors is None and (args.scale is not None or args.verbose):
            raise ValueError("--scale and --verbose are for nested pits: add --factors")
        if args.price_factor is not None and (args.economics is None or args.factors is not None):
            raise ValueError("--price-factor is for one pit valued with --economics (nested pits: --factors)")
        if args.color_by is not None and args.factors is not None:
            raise ValueError("--color-by is for one pit: nested pits are coloured by pit number")
        if args.color_by is not None and not (args.plot or args.screenshot is not None):
            raise ValueError("--color-by colours the 3-D view: add --plot or --screenshot")
        if args.factors is None and (args.animate is not None or args.report is not None):
            raise ValueError("--animate and --report are for nested pits: add --factors")
        npv_only = (args.pushbacks, args.npv_report, args.npv_chart, args.ore, args.timing)
        no_capacity = args.ore_per_year is None and args.mining_rate is None and args.processing_rate is None
        if (
            args.discount_rate is not None
            and no_capacity
            and (any(o is not None for o in npv_only) or args.fixed_cost or args.mid_period)
        ):
            raise ValueError(
                "the pit-by-pit options need a capacity: --ore-per-year, --mining-rate or --processing-rate"
            )
        if args.block_size is None and args.precedence is None:
            raise ValueError("--block-size is required (except with --precedence)")
        if args.block_size is not None and len(args.block_size) not in (1, 3):
            raise ValueError("--block-size takes three numbers (DX DY DZ) or one (cubic blocks)")
        if args.precedence is not None and (args.factors is not None or args.benches is not None):
            raise ValueError("--precedence solves one ultimate pit: it takes no --factors or --benches")
        if args.factors is not None:
            return _nested(args, options)
        block_size = None if args.block_size is None else args.block_size
        pit = solve(
            args.model,
            block_size=None if block_size is None else (tuple(block_size) if len(block_size) == 3 else block_size[0]),
            slopes=None if args.precedence is not None else (args.slope if args.slope is not None else args.sectors),
            precedence=args.precedence,
            value=args.value,
            economics=None if args.economics is None else read_economics(args.economics),
            price_factor=1.0 if args.price_factor is None else args.price_factor,
            indices=args.indices,
            coords=args.coords,
            n_benches=args.benches,
            z_up=not args.z_down,
            bench_discount=args.bench_discount,
            **options,
        )
        print(pit.summary())
        if not args.no_output:
            output = args.output or args.model.with_name(f"{args.model.stem}_pit.csv")
            print(f"\npit written to {pit.to_csv(output, only_pit=args.only_pit)}")
        if args.bench_report is not None:
            print(f"bench report written to {write_table(pit.benches(), args.bench_report)}")
        if args.plot or args.screenshot is not None:
            pit.plot(color_by=args.color_by or "value", screenshot=args.screenshot, show=args.plot)
            if args.screenshot is not None:
                print(f"3-D view saved to {args.screenshot}")
    except (ValueError, TypeError, ImportError, OSError, OverflowError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
