"""Ultimate pit OES on a large synthetic block model: time and peak memory of every stage.

    python benchmarks/oes_scaling.py --size 300 300 50
    python benchmarks/oes_scaling.py --size 200 200 60 --factors 50 --benches 8

The model: blocks of 10 x 10 x 10 m under a flat surface, a gold grade made of a few rich bodies at depth
plus noise, 2.7 t/m3, gold at 60 $/g with 90 % recovery, processing 20 $/t and mining 2 $/t. Nested pits by
price factor (geometric, from 0.01), top-down discounting at 10 % a year and 8 benches a year, then the OES
with both searches and the independent check of each one.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time

import numpy as np

import pymineflow
from pymineflow import Economics, Metal, Process


def peak_memory_mb() -> float:
    """Peak resident memory of this process, in MB (Windows and POSIX)."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
            ]  # fmt: skip

        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def synthetic(nx: int, ny: int, nz: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ix, iy, iz = (a.ravel().astype(np.int32) for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                                                  indexing="ij"))  # fmt: skip
    grade = np.zeros(ix.size)
    for _ in range(6):  # rich bodies at depth
        cx, cy, cz = rng.uniform(0.2, 0.8) * nx, rng.uniform(0.2, 0.8) * ny, rng.uniform(0.2, 0.7) * nz
        rx, ry, rz = rng.uniform(0.06, 0.15) * nx, rng.uniform(0.06, 0.15) * ny, rng.uniform(0.1, 0.25) * nz
        grade += rng.uniform(2, 6) * np.exp(-(((ix - cx) / rx) ** 2 + ((iy - cy) / ry) ** 2 + ((iz - cz) / rz) ** 2))
    grade = np.maximum(grade + rng.normal(0, 0.15, ix.size), 0.0)
    return {"IX": ix, "IY": iy, "IZ": iz, "AU": grade.astype(np.float32), "TON": np.full(ix.size, 2700.0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", nargs=3, type=int, default=[300, 300, 50], metavar=("NX", "NY", "NZ"))
    parser.add_argument("--factors", type=int, default=30, help="number of price factors (default 30)")
    parser.add_argument("--benches", type=int, default=None, help="n_benches of the pattern (default: full height)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    nx, ny, nz = args.size
    start = time.perf_counter()
    table = synthetic(nx, ny, nz, args.seed)
    print(f"model {nx} x {ny} x {nz} = {table['IX'].size:,} blocks, built in {time.perf_counter() - start:.1f} s")
    eco = Economics(metals=[Metal("AU", price=60, grade_unit="g/t", price_unit="g")],
                    processes=[Process("mill", cost=20, recovery={"AU": 0.9})],
                    mining_cost=2, tonnage="TON")  # fmt: skip
    start = time.perf_counter()
    pits = pymineflow.nested_pits(table, block_size=10, slopes=45, n_benches=args.benches, economics=eco,
                                  scale="price", factors=np.geomspace(0.01, 1, args.factors), discount_rate=0.10,
                                  benches_per_year=8, ore_per_year=10e6)  # fmt: skip
    final = int((pits.numbers > 0).sum())
    print(f"nested pits: {args.factors} factors, final pit {final:,} blocks, {time.perf_counter() - start:.1f} s, "
          f"peak memory {peak_memory_mb():,.0f} MB")  # fmt: skip
    for search in ("proximity", "lookahead"):
        start = time.perf_counter()
        oes = pits.oes(search=search)
        elapsed = time.perf_counter() - start
        start = time.perf_counter()
        check = oes.check()
        checked = time.perf_counter() - start
        steps = oes.spatial()
        stages = ", ".join(f"{k} {v:.2f} s" for k, v in oes.seconds.items())
        print(f"OES {search}: {elapsed:.2f} s ({stages}); NPV {oes.npv:,.0f}; {oes.phase_pits.size} phases, "
              f"{oes.time[-1]:.1f} periods; mean step {steps['mean_step']:.2f}; check {checked:.2f} s, "
              f"{'ok' if check.ok else 'FAILED'} ({check.violations} slope violations); "
              f"peak memory {peak_memory_mb():,.0f} MB")  # fmt: skip
    best = pits.npv().table["npv_best"][-1]
    print(f"pit-by-pit best case of the final pit (cash at the end of each year): {best:,.0f}")


if __name__ == "__main__":
    main()
