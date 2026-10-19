"""Independent references for the tests: they never use PyMineFlow's graph.

- Cone geometry: a pure-Python replica of MineFlow's SlopeDefinition::Within (normalized azimuth).
- Maximum closure: scipy's max-flow (Dinic) on an explicitly built graph.
- Brute force: enumeration of every closed subset of a tiny model.
"""

from __future__ import annotations

import itertools
import math
from bisect import bisect_left

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

TAU = 6.283185307179586476925286766559
SLOPE_EPS = 1e-9  # the same inclusive-boundary margin as the binding
MCL = (7.62, 7.62, 6.096)  # a common real-world block size


def as_sectors(spec):
    """Number or [(azimuth, slope), ...] -> [(azimuth, slope), ...]."""
    return [(0.0, float(spec))] if np.isscalar(spec) else [(float(a), float(s)) for a, s in spec]


def slope_pairs(spec):
    """Sorted (azimuth_rad, slope_rad) pairs, built like the binding does."""
    return sorted(((a % 360.0) * TAU / 360.0, s * TAU / 360.0 - SLOPE_EPS) for a, s in as_sectors(spec))


def slope_at(azimuth, pairs):
    """Circular linear interpolation between pairs (SlopeDefinition::Get with a normalized azimuth)."""
    if len(pairs) == 1:
        return pairs[0][1]
    azimuth %= TAU
    right = bisect_left([p[0] for p in pairs], azimuth) % len(pairs)
    left = right - 1 if right > 0 else len(pairs) - 1
    (la, ls), (ra, rs) = pairs[left], pairs[right]
    to_left = TAU - la + azimuth if la > azimuth else azimuth - la
    to_right = TAU - azimuth + ra if ra < azimuth else ra - azimuth
    return ls + (rs - ls) * (to_left / (to_left + to_right))


def within(dx, dy, dz, pairs):
    """True when the vector (dx, dy, dz) (metres, dz > 0 upwards) lies inside the slope cone."""
    if dx == 0 and dy == 0:
        return True
    theta = math.atan(abs(dz) / math.sqrt(dx * dx + dy * dy))
    return theta >= slope_at(math.pi / 2 - math.atan2(dy, dx), pairs)


def cone_offsets(nx, ny, nz, block_size, pairs):
    """Every offset of the exact cone that fits in the grid (no reduction), full height."""
    sx, sy, sz = block_size
    return [
        (ox, oy, oz)
        for oz in range(1, nz)
        for ox in range(-(nx - 1), nx)
        for oy in range(-(ny - 1), ny)
        if within(ox * sx, oy * sy, oz * sz, pairs)
    ]


def arcs(nx, ny, nz, offsets):
    """Offsets -> arcs (src requires dst) inside the grid, cell index (z*ny + y)*nx + x."""
    gz, gy, gx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    gx, gy, gz = gx.ravel(), gy.ravel(), gz.ravel()
    src, dst = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    for ox, oy, oz in offsets:
        tx, ty, tz = gx + ox, gy + oy, gz + oz
        ok = (tx >= 0) & (tx < nx) & (ty >= 0) & (ty < ny) & (tz < nz)
        src.append(gz[ok] * ny * nx + gy[ok] * nx + gx[ok])
        dst.append(tz[ok] * ny * nx + ty[ok] * nx + tx[ok])
    return np.concatenate(src).astype(np.int64), np.concatenate(dst).astype(np.int64)


def max_closure_value(values, src, dst):
    """Maximum closure value by max-flow: sum of positive values minus the maximum flow."""
    values = np.asarray(values, dtype=np.int64)
    n = len(values)
    source, sink = n, n + 1
    infinite = int(np.abs(values).sum()) + 1
    pos = np.flatnonzero(values > 0)
    neg = np.flatnonzero(values < 0)
    rows = np.concatenate([np.full(len(pos), source), neg, src])
    cols = np.concatenate([pos, np.full(len(neg), sink), dst])
    caps = np.concatenate([values[pos], -values[neg], np.full(len(src), infinite)]).astype(np.int32)
    graph = csr_matrix((caps, (rows, cols)), shape=(n + 2, n + 2))
    return int(values[pos].sum()) - int(maximum_flow(graph, source, sink).flow_value)


def dense(grid, values):
    cells = np.zeros(grid.n_cells, dtype=np.int64)
    cells[grid.linear] = values
    return cells


def grid_cone_value(grid, values, spec):
    """Reference B: empty cells are zero-value blocks with the same slope (the defined model)."""
    offsets = cone_offsets(grid.nx, grid.ny, grid.nz, grid.block_size, slope_pairs(spec))
    return max_closure_value(dense(grid, values), *arcs(grid.nx, grid.ny, grid.nz, offsets))


def present_cone_arcs(grid, domains, domain_specs):
    """Reference A arcs: only present blocks require the blocks of their own domain's cone."""
    cell_domain = np.full(grid.n_cells, -1)
    cell_domain[grid.linear] = domains
    src_all, dst_all = [], []
    for d, spec in enumerate(domain_specs):
        offsets = cone_offsets(grid.nx, grid.ny, grid.nz, grid.block_size, slope_pairs(spec))
        src, dst = arcs(grid.nx, grid.ny, grid.nz, offsets)
        keep = cell_domain[src] == d
        src_all.append(src[keep])
        dst_all.append(dst[keep])
    return np.concatenate(src_all), np.concatenate(dst_all)


def present_cone_value(grid, values, domains, domain_specs):
    """Reference A: physical semantics (each present block requires its own cone)."""
    return max_closure_value(dense(grid, values), *present_cone_arcs(grid, domains, domain_specs))


def brute_force(ix, iy, iz, block_size, values, domains, domain_specs):
    """Best closed subset: if block i is mined, every present block of its cone is mined."""
    pairs = [slope_pairs(s) for s in domain_specs]
    n = len(values)
    required = [
        [
            j
            for j in range(n)
            if iz[j] > iz[i]
            and within(
                (ix[j] - ix[i]) * block_size[0],
                (iy[j] - iy[i]) * block_size[1],
                (iz[j] - iz[i]) * block_size[2],
                pairs[domains[i]],
            )
        ]
        for i in range(n)
    ]
    best = 0
    for bits in itertools.product((0, 1), repeat=n):
        if all(not bits[i] or all(bits[j] for j in required[i]) for i in range(n)):
            best = max(best, sum(int(values[i]) for i in range(n) if bits[i]))
    return best


def random_values(n, rng):
    """Mostly small negative values (waste) and ~4 % rich blocks."""
    v = -rng.integers(1, 8, n)
    k = rng.choice(n, size=max(3, n // 25), replace=False)
    v[k] = rng.integers(40, 300, k.size)
    return v.astype(np.int64)


def topography_model(seed, nx=11, ny=9, nz=6, topography=True):
    """Rows of a block model cut below a random topography (no holes), in shuffled order."""
    r = np.random.default_rng(seed)
    surface = r.integers(1, nz, size=(nx, ny)) if topography else np.full((nx, ny), nz - 1)
    gx, gy, gz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    below = gz <= surface[:, :, None]
    order = r.permutation(int(below.sum()))
    ix, iy, iz = gx[below][order], gy[below][order], gz[below][order]
    v = -r.integers(1, 9, ix.size)
    rich = r.choice(ix.size, max(2, ix.size // 12), replace=False)
    v[rich] = r.integers(30, 400, rich.size)
    return ix, iy, iz, v.astype(np.int64)
