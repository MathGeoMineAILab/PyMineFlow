"""Ultimate pit (UPIT) on the open-pit instances of MineLib, checked against the published optima.

MineLib (Espinoza, Goycoolea, Moreno and Newman, 2013): https://minelib.org

For every instance:

1. ``ExplicitPit`` on MineLib's own precedence graph (``.prec``) and values (``.upit``); the value
   must equal the published optimum and the pit must equal the published solution (``.sol``).
2. ``ultimate_pit`` from block coordinates and a 45-degree slope, for the instances whose block
   size and slope are documented (MineLib built them with 45 degrees and 8 levels).
3. Optionally (``--compare``), the same graph solved with PyMaxflow and with ``pseudoflow``
   (``pip install PyMaxflow pseudoflow networkx``); pseudoflow runs once, being much slower.

The script stops with an error if a value or a pit differs from the published one.

Usage::

    python benchmarks/minelib.py --download          # fetch the files (about 30 MB) into benchmarks/data
    python benchmarks/minelib.py --compare --repeat 5
"""

from __future__ import annotations

import argparse
import io
import platform
import statistics
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import pymineflow
from pymineflow import (
    ExplicitPit,
    read_blocks,
    read_minelib_precedence,
    read_minelib_values,
    scale_values,
    ultimate_pit,
)

URL = "https://minelib.org/v1"

# name: (published UPIT optimum, block size or None, MineLib's bench levels)
INSTANCES: dict[str, tuple[int, tuple[float, float, float] | None, int]] = {
    "newman1": (26_086_899, (1.0, 1.0, 1.0), 1),  # "5 blocks above (+)": one level with a cube and 45 degrees
    "zuck_small": (1_422_726_898, None, 0),
    "kd": (652_195_037, (20.0, 20.0, 15.0), 8),
    "zuck_medium": (1_075_124_490, None, 0),
    "p4hd": (293_373_256, (50.0, 50.0, 20.0), 8),
    "marvin": (1_415_655_436, (30.0, 30.0, 30.0), 8),
    "w23": (510_973_998, (25.0, 25.0, 20.0), 8),
    "zuck_large": (122_220_280, None, 0),
    "sm2": (2_743_603_730, None, 0),
    "mclaughlin_limit": (1_495_726_474, (25.0, 25.0, 20.0), 8),
}


def download(data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    for name in INSTANCES:
        for part in ("blocks", "prec", "files"):
            with (
                urllib.request.urlopen(f"{URL}/data/{name}.{part}.zip") as r,
                zipfile.ZipFile(io.BytesIO(r.read())) as z,
            ):
                for member in z.namelist():
                    if member.endswith((".blocks", ".prec", ".upit")):
                        (data / Path(member).name).write_bytes(z.read(member))
        with urllib.request.urlopen(f"{URL}/sols/{name}_upit.sol") as r:
            (data / f"{name}_upit.sol").write_bytes(r.read())
        print(f"downloaded {name}")


def timed(fn: Callable[[], Any], repeat: int) -> tuple[Any, float]:
    """Result of the last run and median wall time."""
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return out, statistics.median(times)


def solve_pymaxflow(values: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    import maxflow

    n = values.size
    g = maxflow.GraphFloat(n, src.size)
    nodes = g.add_nodes(n)
    g.add_grid_tedges(nodes, np.maximum(values, 0).astype(float), np.maximum(-values, 0).astype(float))
    infinite = float(np.abs(values).sum() + 1)
    g.add_edges(src, dst, np.full(src.size, infinite), np.zeros(src.size))
    g.maxflow()
    return ~g.get_grid_segments(nodes)  # source side = mined


def solve_pseudoflow(values: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    import networkx as nx
    import pseudoflow

    n = values.size
    infinite = float(np.abs(values).sum() + 1)
    g = nx.DiGraph()
    pos, neg = np.flatnonzero(values > 0), np.flatnonzero(values < 0)
    g.add_edges_from(("s", int(i), {"c": float(values[i])}) for i in pos)
    g.add_edges_from((int(i), "t", {"c": float(-values[i])}) for i in neg)
    g.add_edges_from((int(a), int(b), {"c": infinite}) for a, b in zip(src, dst, strict=True))
    _, cuts, _ = pseudoflow.hpf(g, "s", "t", const_cap="c")
    mask = np.zeros(n, bool)
    mask[[k for k, v in cuts.items() if v[0] and k not in ("s", "t")]] = True
    return mask


def run(name: str, data: Path, repeat: int, compare: bool) -> tuple[str, list[tuple[Any, ...]]]:
    """One instance: the table row and the slope-based results."""
    published, size, levels = INSTANCES[name]
    (src, dst), t_read = timed(lambda: read_minelib_precedence(data / f"{name}.prec"), 1)
    real = read_minelib_values(data / f"{name}.upit")
    values, _ = scale_values(real)
    (mask, _, _, _), t_solve = timed(lambda: ExplicitPit(values.size, src, dst, values).solve(), repeat)
    sol = np.zeros(values.size, bool)
    sol[np.loadtxt(data / f"{name}_upit.sol", dtype=np.int64, ndmin=1)] = True
    value = real[mask].sum()
    if abs(value - published) > 1 or not np.array_equal(mask, sol):  # published optima are rounded
        raise AssertionError(
            f"{name}: value {value:,.2f} vs published {published:,}, same pit: {np.array_equal(mask, sol)}"
        )
    row = (
        f"| {name} | {values.size:,} | {src.size:,} | {published:,} | {value:,.0f} | yes | "
        f"{t_read:.2f} s | {t_solve * 1e3:.0f} ms |"
    )
    if compare:
        for solver in (solve_pymaxflow, solve_pseudoflow):
            other, t = timed(lambda s=solver: s(values, src, dst), 1 if solver is solve_pseudoflow else repeat)
            assert values[other].sum() == values[mask].sum(), (name, solver.__name__)
            row += f" {t * 1e3:,.0f} ms |"
    geometric = []
    if size is not None:
        b = read_blocks(data / f"{name}.blocks", ["1", "2", "3"])
        for n_benches in (levels, None):
            r, t = timed(
                lambda k=n_benches: ultimate_pit(
                    values, ix=b["1"], iy=b["2"], iz=b["3"], block_size=size, slopes=45, n_benches=k, min_density=0
                ),
                repeat,
            )
            geometric.append((name, size, n_benches, real[r.mask].sum(), published, r.n_blocks, t))
    return row, geometric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "minelib")
    parser.add_argument("--download", action="store_true", help="download the MineLib files first")
    parser.add_argument("--repeat", type=int, default=3, help="runs per timing (median)")
    parser.add_argument("--compare", action="store_true", help="also solve with PyMaxflow and pseudoflow")
    args = parser.parse_args()
    if args.download:
        download(args.data)

    print(f"PyMineFlow {pymineflow.__version__}, Python {platform.python_version()}, {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.machine()}; median of {args.repeat} runs\n")
    print("| Instance | Blocks | Arcs | Published UPIT | PyMineFlow | Same pit | Read | Solve |", end="")
    print(" PyMaxflow | pseudoflow |" if args.compare else "")
    print("|---|---:|---:|---:|---:|:---:|---:|---:|" + ("---:|---:|" if args.compare else ""))
    geometric = []
    for name in INSTANCES:
        row, rows = run(name, args.data, args.repeat, args.compare)
        print(row)
        geometric += rows

    print("\nSlope-based model: ultimate_pit(ix, iy, iz, block_size, slopes=45, n_benches)\n")
    print("| Instance | Block size | n_benches | Pit value | vs published | Blocks | Time |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for name, size, n_benches, value, published, n_blocks, t in geometric:
        size_txt = "x".join(f"{s:g}" for s in size)
        gap = (value - published) / published * 100
        gap = 0.0 if abs(gap) < 5e-4 else gap
        print(
            f"| {name} | {size_txt} | {n_benches or 'all'} | {value:,.0f} | {gap:+.3f} % | {n_blocks:,} | {t:.2f} s |"
        )


if __name__ == "__main__":
    main()
