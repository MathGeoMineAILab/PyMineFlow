"""Ultimate pit OES, checked against independent references: the exact geometric cone (slopes), brute-force
searches (proximity walk, precedence order, lookahead values) and a period-by-period schedule written block by
block."""

import dataclasses
import math

import numpy as np
import pytest

import pymineflow
from pymineflow import Economics, Metal, Process
from pymineflow._mineflow import _oes_lookahead, _oes_order, _oes_schedule
from pymineflow.cli import main

from _oracle import slope_pairs, topography_model, within

SIZE = (10.0, 10.0, 8.0)
SHIFT = {"end": 0.0, "mid": 0.5, "start": 1.0}


def model(seed, nx=9, ny=8, nz=6, topography=True):
    ix, iy, iz, v = topography_model(seed, nx=nx, ny=ny, nz=nz, topography=topography)
    rng = np.random.default_rng(seed + 100)
    tons = rng.uniform(800.0, 1200.0, ix.size)
    return {"IX": ix, "IY": iy, "IZ": iz, "VALUE": v.astype(float), "TON": tons}


def nested(table, slopes=45, factors=None, **kw):
    factors = np.geomspace(0.002, 1.0, 15) if factors is None else factors
    return pymineflow.nested_pits(table, block_size=SIZE, slopes=slopes, factors=factors, **kw)


# ----------------------------------------------------------------- independent references


def exact_cone_violations(oes, slope):
    """Pairs of final-pit blocks (a, b) with b in a's exact cone (geometry only) and b mined after a."""
    g = oes.pits.model.grid
    rank = oes.rank
    rows = np.flatnonzero(rank >= 0)
    pairs = slope_pairs(slope)
    bad = 0
    for a in rows:
        for b in rows:
            dz = int(g.iz[b]) - int(g.iz[a])
            if dz > 0 and within((int(g.ix[b]) - int(g.ix[a])) * SIZE[0], (int(g.iy[b]) - int(g.iy[a])) * SIZE[1],
                                 dz * SIZE[2], pairs) and rank[b] > rank[a]:  # fmt: skip
                bad += 1
    return bad


def walk_reference(x, y, key, c):
    """Greedy proximity walk, O(n^2): highest key first, then max key - c * distance, ties to the lowest index."""
    left = list(range(len(key)))
    current = min(left, key=lambda i: (-key[i], i))
    out = []
    while True:
        out.append(current)
        left.remove(current)
        if not left:
            return out
        cx, cy = x[current], y[current]
        current = min(left, key=lambda i: (-(key[i] - c * math.hypot(x[i] - cx, y[i] - cy)), i))


def lookahead_reference(oes, benches, radius, rate, scope):
    """Value of every block plus the discounted value of the blocks of its downward cone (brute force)."""
    g = oes.pits.model.grid
    rows = np.flatnonzero(oes.rank >= 0)
    out = {}
    for b in rows:
        s = oes.values[b]
        for j in rows:
            k = int(g.iz[b]) - int(g.iz[j])
            if not 1 <= k <= benches or (scope == "phase" and oes.phase[j] != oes.phase[b]):
                continue
            d = math.hypot((int(g.ix[j]) - int(g.ix[b])) * SIZE[0], (int(g.iy[j]) - int(g.iy[b])) * SIZE[1])
            if d <= k * radius * (1 + 1e-12):
                s += (1 + rate) ** -k * oes.values[j]
        out[b] = s
    return out


def schedule_reference(units, rate, mining, plant, names, fixed, timing):
    """Mine the units (value, tonnage, ore per process) in order; each period takes what every capacity allows."""
    constraints = []
    if mining is not None:
        constraints.append((mining, lambda u: u[1]))
    if isinstance(plant, dict):
        for name, cap in plant.items():
            index = names.index(name)
            constraints.append((cap, lambda u, p=index: u[2][p]))
    elif plant is not None:
        constraints.append((plant, lambda u: sum(u[2])))
    period, left, cash = 1, [c for c, _ in constraints], {1: 0.0}
    for unit in units:
        rest = 1.0
        while rest > 0.0:
            take = rest
            for (cap, use), room in zip(constraints, left, strict=True):
                if use(unit) > 0 and use(unit) * rest > room + 1e-9 * cap:
                    take = min(take, room / use(unit))
            take = max(take, 0.0)
            for j, (_, use) in enumerate(constraints):
                left[j] -= take * use(unit)
            cash[period] = cash.get(period, 0.0) + take * unit[0]
            rest = 0.0 if take == rest else rest - take
            if rest > 0.0:
                period += 1
                left = [c for c, _ in constraints]
    share = min(max(max((cap - room) / cap for (cap, _), room in zip(constraints, left, strict=True)), 0.0), 1.0)

    def factor(k):
        return (1.0 + rate) ** -(k - SHIFT[timing])

    npv = sum(v * factor(k) for k, v in cash.items())
    npv -= fixed * (sum(factor(k) for k in range(1, period)) + share * factor(period))
    return npv, period - 1 + share


def units_of(oes, upto=None):
    o = oes.order if upto is None else oes.order[: upto + 1]
    n = len(oes.processes)
    return [(oes.values[i], oes.tonnage[i], [oes.tonnage[i] if oes.destination[i] == p else 0.0 for p in range(n)])
            for i in o]  # fmt: skip


# ----------------------------------------------------------------- order and slopes


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("slope", [40, 45, 55])
@pytest.mark.parametrize("search", ["proximity", "lookahead"])
def test_slopes_hold_against_the_exact_cone(seed, slope, search):
    pits = nested(model(seed), slopes=slope, discount_rate=0.1, ore_per_year=3000)
    oes = pits.oes(search=search)
    check = oes.check()
    assert check.ok, check.summary()
    assert exact_cone_violations(oes, slope) == 0


@pytest.mark.parametrize("seed", [4, 5])
def test_limited_benches_and_one_five_pattern(seed):
    table = model(seed)
    for options in ({"slopes": 45, "n_benches": 2}, {"slopes": None}):
        pits = pymineflow.nested_pits(table, block_size=SIZE, factors=np.linspace(0.2, 1, 5), discount_rate=0.1,
                                      ore_per_year=3000, **options)  # fmt: skip
        for search in ("proximity", "lookahead"):
            check = pits.oes(search=search).check()
            assert check.ok, (options, search, check.summary())


def test_check_finds_a_wrong_order():
    pits = nested(model(7), discount_rate=0.1, ore_per_year=3000)
    oes = pits.oes()
    bad = dataclasses.replace(oes, order=oes.order[::-1].copy())
    check = bad.check()
    assert check.violations > 0
    assert not check.ordered
    assert not check.ok
    missing = dataclasses.replace(oes, order=oes.order[1:].copy())
    assert not missing.check().complete


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("c", [0.0, 0.3, 2.0, 50.0])
def test_proximity_walk_against_brute_force(seed, c):
    rng = np.random.default_rng(seed)
    cells = rng.choice(15 * 12, size=60, replace=False)
    ix, iy = cells % 15, cells // 15
    key = rng.normal(0, 10, cells.size)
    zeros = np.zeros(cells.size, dtype=np.int64)
    got = _oes_order(ix, iy, zeros, zeros + 1, key, c, 10.0, 7.0)
    if c == 0.0:
        expected = sorted(range(key.size), key=lambda i: (-key[i], i))
    else:
        expected = walk_reference(ix * 10.0, iy * 7.0, key, c)
    assert got.tolist() == expected


def test_bench_order_proximity_zero_sorts_each_phase_bench():
    pits = nested(model(8), discount_rate=0.1, ore_per_year=3000)
    oes = pits.oes(proximity=0.0)
    g = pits.model.grid
    o = oes.order
    groups = np.stack([oes.phase[o], -np.asarray(g.iz)[o]], axis=1)
    for p, z in {tuple(r) for r in groups.tolist()}:
        sel = o[(oes.phase[o] == p) & (-np.asarray(g.iz)[o] == z)]
        assert np.all(np.diff(oes.key[sel]) <= 0)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("proximity", [0.0, 1.0])
def test_precedence_search_takes_the_best_available_block(seed, proximity):
    pits = nested(model(seed, nx=7, ny=6, nz=5), discount_rate=0.1, ore_per_year=3000)
    oes = pits.oes(search="lookahead", proximity=proximity, lookahead_benches=2, bench_order=False)
    g = pits.model.grid
    pit = pits.model._pit
    offsets = pit.offsets(0)
    block_of = np.full(g.n_cells, -1)
    block_of[np.asarray(g.linear)] = np.arange(g.linear.size)
    rank = oes.rank
    w = oes.proximity_weight
    xyz = np.stack([np.asarray(g.ix) * SIZE[0], np.asarray(g.iy) * SIZE[1], np.asarray(g.iz) * SIZE[2]], axis=1)

    def done_cells(mined):
        done = np.zeros(g.n_cells, dtype=bool)
        for z in range(g.nz - 1, -1, -1):
            for y in range(g.ny):
                for x in range(g.nx):
                    cell = (z * g.ny + y) * g.nx + x
                    b = block_of[cell]
                    if b >= 0:
                        done[cell] = bool(mined[b])
                        continue
                    done[cell] = all(
                        done[((z + oz) * g.ny + y + oy) * g.nx + x + ox]
                        for ox, oy, oz in offsets
                        if 0 <= x + ox < g.nx and 0 <= y + oy < g.ny and z + oz < g.nz
                    )
        return done

    mined = np.zeros(g.linear.size, dtype=bool)
    last = None
    for step, b in enumerate(oes.order):
        done = done_cells(mined)
        phase = oes.phase[b]
        if step and oes.phase[oes.order[step - 1]] != phase:
            last = None
        available = []
        for c in np.flatnonzero((oes.phase == phase) & ~mined):
            cell = int(g.linear[c])
            x, y, z = cell % g.nx, (cell // g.nx) % g.ny, cell // (g.nx * g.ny)
            if all(done[((z + oz) * g.ny + y + oy) * g.nx + x + ox] for ox, oy, oz in offsets
                   if 0 <= x + ox < g.nx and 0 <= y + oy < g.ny and z + oz < g.nz):  # fmt: skip
                available.append(c)
        assert b in available, step

        def score(c, last=last):
            return oes.key[c] if last is None else oes.key[c] - w * float(np.linalg.norm(xyz[c] - xyz[last]))

        best = max(available, key=lambda c: (score(c), -c))
        assert score(b) == pytest.approx(score(best), rel=1e-12, abs=1e-9), step
        mined[b] = True
        last = b
    assert rank[oes.order].tolist() == list(range(oes.order.size))


@pytest.mark.parametrize("scope", ["pit", "phase"])
@pytest.mark.parametrize("benches", [1, 3])
def test_lookahead_values_against_brute_force(scope, benches):
    pits = nested(model(11), slopes=40, discount_rate=0.12, benches_per_year=4, ore_per_year=3000)
    oes = pits.oes(search="lookahead", lookahead_benches=benches, lookahead_scope=scope)
    radius = SIZE[2] / math.tan(math.radians(40) - 1e-9)
    assert oes.lookahead_radius == pytest.approx(radius)
    expected = lookahead_reference(oes, benches, radius, 0.12 / 4, scope)
    for b, value in expected.items():
        assert oes.key[b] == pytest.approx(value, rel=1e-12, abs=1e-9)


def test_lookahead_kernel_by_hand():
    #   z=1:  . A .      A sees the three blocks below within one block of radius
    #   z=0:  B C D
    got = _oes_lookahead([1, 0, 1, 2], [0, 0, 0, 0], [1, 0, 0, 0], [1.0, 2.0, 3.0, 4.0], 1.0, 1.0, [1.0], [0.5])
    assert got.tolist() == [1.0 + 0.5 * 9.0, 2.0, 3.0, 4.0]


# ----------------------------------------------------------------- schedule and NPV


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    ("rate", "mining", "plant", "fixed", "timing"),
    [
        (0.10, 9000.0, None, 0.0, "end"),
        (0.08, None, 3500.0, 0.0, "end"),
        (0.12, 12000.0, 2500.0, 1500.0, "end"),
        (0.15, 7000.0, 4000.0, 0.0, "mid"),
        (0.15, 20000.0, 3000.0, 200.0, "start"),
        (0.0, 10000.0, 3000.0, 500.0, "mid"),
    ],
)
def test_schedule_against_a_block_by_block_reference(seed, rate, mining, plant, fixed, timing):
    pits = nested(model(seed))
    oes = pits.oes(discount_rate=rate, mining_rate=mining, processing_rate=plant, fixed_cost=fixed, timing=timing)
    npv, life = schedule_reference(units_of(oes), rate, mining, plant, ["ore"], fixed, timing)
    assert oes.npv == pytest.approx(npv, rel=1e-9, abs=1e-6)
    assert oes.time[-1] == pytest.approx(life, rel=1e-9, abs=1e-9)
    for k in (0, oes.order.size // 3, oes.order.size // 2):  # NPV if mining stopped after block k
        npv_k, life_k = schedule_reference(units_of(oes, k), rate, mining, plant, ["ore"], fixed, timing)
        assert oes.npv_curve[k] == pytest.approx(npv_k, rel=1e-9, abs=1e-6)
        assert oes.time[k] == pytest.approx(life_k, rel=1e-9, abs=1e-9)
    p = oes.periods
    assert p["discounted"].sum() == pytest.approx(oes.npv, rel=1e-9, abs=1e-6)
    assert p["value"].sum() == pytest.approx(oes.values[oes.order].sum(), rel=1e-9)
    assert p["tonnage"].sum() == pytest.approx(oes.tonnage[oes.order].sum(), rel=1e-9)
    if mining is not None:
        assert p["tonnage"].max() <= mining * (1 + 1e-9)
    if plant is not None:
        assert p["ore_tonnage"].max() <= plant * (1 + 1e-9)
    assert oes.check().ok


def continuous_reference(oes, rate, cap, use):
    """Every block discounted over the time it takes (one capacity: time = cumulative use / capacity)."""
    v = oes.values[oes.order]
    t1 = np.cumsum(use[oes.order]) / cap
    t0 = t1 - use[oes.order] / cap
    width = np.where(t1 > t0, t1 - t0, 1.0)
    mean = np.where(t1 > t0, (np.exp(-rate * t0) - np.exp(-rate * t1)) / (rate * width), np.exp(-rate * t0))
    return float((v * mean).sum()), float(t1[-1])


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(("rate", "mining", "plant"), [(0.10, None, 3000.0), (0.15, 9000.0, None)])
def test_continuous_discounting_against_a_reference(seed, rate, mining, plant):
    pits = nested(model(seed))
    oes = pits.oes(discount_rate=rate, mining_rate=mining, processing_rate=plant, timing="continuous")
    use = oes.tonnage if mining is not None else np.where(oes.destination >= 0, oes.tonnage, 0.0)
    npv, life = continuous_reference(oes, rate, mining or plant, use)
    assert oes.npv == pytest.approx(npv, rel=1e-9)
    assert oes.time[-1] == pytest.approx(life, rel=1e-12)
    assert oes.periods["discounted"].sum() == pytest.approx(oes.npv, rel=1e-9)
    fixed = pits.oes(discount_rate=rate, mining_rate=mining, processing_rate=plant, timing="continuous", fixed_cost=700)
    assert fixed.npv == pytest.approx(npv - 700 * (1 - np.exp(-rate * life)) / rate, rel=1e-9)
    assert fixed.periods["discounted"].sum() == pytest.approx(fixed.npv, rel=1e-9)
    assert pits.oes(discount_rate=rate, mining_rate=mining, processing_rate=plant).timing == "continuous"


def test_continuous_pit_by_pit_is_the_oes_of_a_column():
    nz = 12
    value = np.where(np.arange(nz) < 8, np.linspace(900, 100, nz)[::-1], -40.0)
    table = {"IX": np.zeros(nz, int), "IY": np.zeros(nz, int), "IZ": np.arange(nz), "VALUE": value,
             "TON": np.full(nz, 100.0)}  # fmt: skip
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=np.linspace(0.05, 1, 20),
                                  discount_rate=0.1, ore_per_year=150)  # fmt: skip
    for options in ({}, {"mining_rate": 250}, {"mining_rate": 250, "fixed_cost": 5}):
        oes = pits.oes(timing="continuous", min_component=0, **options)
        r = pits.npv(timing="continuous", **options)
        assert np.allclose(oes.phase_table()["npv"], r.table["npv_best"][oes.phase_pits - 1], rtol=1e-12, atol=1e-9)
        assert np.allclose(oes.phase_table()["end"], r.table["life_best"][oes.phase_pits - 1], rtol=1e-12)
        k = r.best_pit("worst")
        schedule = r.schedule(k, "worst")
        assert schedule["discounted"].sum() == pytest.approx(r.table["npv_worst"][k - 1], rel=1e-9)
    assert "the moment each block is mined" in pits.report(timing="continuous")


def test_two_processes_and_capacity_per_process():
    rng = np.random.default_rng(3)
    table = model(3)
    table["AU"] = np.where(
        table["VALUE"] > 0, rng.uniform(1, 6, table["VALUE"].size), rng.uniform(0, 0.4, table["VALUE"].size)
    )
    eco = Economics(metals=[Metal("AU", price=60, grade_unit="g/t", price_unit="g")],
                    processes=[Process("mill", cost=40, recovery={"AU": 0.9}),
                               Process("leach", cost=8, recovery={"AU": 0.5})],
                    mining_cost=2, tonnage="TON")  # fmt: skip
    pits = nested(table, economics=eco, scale="price")
    plant = {"mill": 3000.0, "leach": 5000.0}
    oes = pits.oes(discount_rate=0.1, processing_rate=plant, mining_rate=15000, timing="end")
    npv, life = schedule_reference(units_of(oes), 0.1, 15000, plant, ["mill", "leach"], 0.0, "end")
    assert oes.npv == pytest.approx(npv, rel=1e-9)
    assert oes.time[-1] == pytest.approx(life, rel=1e-9)
    assert oes.periods["ore_tonnage_mill"].max() <= 3000 * (1 + 1e-9)
    assert oes.periods["ore_tonnage_leach"].max() <= 5000 * (1 + 1e-9)
    assert oes.check().ok


@pytest.mark.parametrize("timing", ["end", "mid", "start"])
def test_one_block_per_phase_bench_is_the_pit_by_pit_best_case(timing):
    # a single column: every phase-bench holds one block, so the OES is the best case of pit_by_pit exactly
    nz = 12
    value = np.where(np.arange(nz) < 8, np.linspace(900, 100, nz)[::-1][np.arange(nz)], -40.0)
    table = {"IX": np.zeros(nz, int), "IY": np.zeros(nz, int), "IZ": np.arange(nz), "VALUE": value,
             "TON": np.full(nz, 100.0)}  # fmt: skip
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=np.linspace(0.05, 1, 20),
                                  discount_rate=0.1, ore_per_year=150)  # fmt: skip
    oes = pits.oes(timing=timing, mining_rate=250, fixed_cost=5, min_component=0)  # the same phases
    best = pits.npv(timing=timing, mining_rate=250, fixed_cost=5).table["npv_best"]
    assert np.allclose(oes.phase_table()["npv"], best[oes.phase_pits - 1], rtol=1e-12, atol=1e-9)
    assert np.allclose(oes.check().best_case, 0.0, atol=1e-12)


def test_value_column_with_an_ore_column():
    table = model(9)
    table["DEST"] = (table["VALUE"] > 0).astype(int)
    pits = nested(table, discount_rate=0.1)
    oes = pits.oes(processing_rate=4000, ore="DEST")
    assert oes.processes == ("ore",)
    assert oes.check().ok


def pod_model():
    """A deep ore body and a rich block alone at the surface above its future pit: the block is a pit of its own
    at the lowest factors, far from the first phases of the body."""
    nx, ny, nz = 24, 14, 7
    ix, iy, iz = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"))
    value = np.full(ix.size, -2.0)
    body = (ix >= 4) & (ix <= 9) & (iy >= 4) & (iy <= 9) & (iz <= 2)
    value[body] = 60.0
    second = (ix >= 15) & (ix <= 19) & (iy >= 4) & (iy <= 9) & (iz <= 2)
    value[second] = 25.0
    pod = (ix == 17) & (iy == 7) & (iz == nz - 1)
    value[pod] = 40.0
    table = {"IX": ix, "IY": iy, "IZ": iz, "VALUE": value, "TON": np.full(ix.size, 100.0)}
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=np.geomspace(0.02, 1, 25),
                                  discount_rate=0.1, ore_per_year=3000)  # fmt: skip
    return pits, int(np.flatnonzero(pod)[0])


def test_continuity_isolated_groups_wait_for_their_surroundings():
    pits, pod = pod_model()
    plain = pits.oes(min_component=0)
    assert plain.phase[pod] == 1  # alone, the rich surface block would be mined first
    oes = pits.oes()
    assert oes.min_component == 27
    assert oes.moved_groups >= 1
    assert oes.phase[pod] > 1
    g = pits.model.grid
    near = [np.abs(np.asarray(a, dtype=int) - int(a[pod])) <= 1 for a in (g.ix, g.iy, g.iz)]
    neighbours = np.flatnonzero(near[0] & near[1] & near[2] & (oes.phase > 0))
    assert oes.phase[pod] <= max(oes.phase[n] for n in neighbours if n != pod)  # mined with its surroundings
    for search in ("proximity", "lookahead"):
        check = pits.oes(search=search).check()
        assert check.ok, check.summary()
    assert "Continuity:" in oes.summary()


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("size", [5, 40, 10_000])
def test_continuity_keeps_the_slopes(seed, size):
    pits = nested(model(seed), discount_rate=0.1, ore_per_year=3000)
    for search in ("proximity", "lookahead"):
        oes = pits.oes(search=search, min_component=size)
        check = oes.check()
        assert check.ok, check.summary()
        assert exact_cone_violations(oes, 45) == 0
    with pytest.raises(ValueError, match="min_component"):
        pits.oes(min_component=-1)


def test_phases_merge_nested_pits():
    pits = nested(model(10), factors=np.linspace(0.1, 1, 19), discount_rate=0.1, ore_per_year=3000)
    oes = pits.oes(phases=[5, 12])
    with_blocks = np.flatnonzero(np.bincount(pits.numbers, minlength=len(pits) + 1)[1:]) + 1
    assert oes.phase_pits.tolist()[-1] == with_blocks[-1]  # the last pit that adds blocks, not an empty one
    assert set(oes.phase_pits.tolist()) <= set(with_blocks.tolist())
    assert oes.report_table()["oes_npv"][-1] == pytest.approx(oes.npv)  # the last row is the whole sequence
    assert oes.check().ok
    every = pits.oes()
    assert oes.order.size == every.order.size


def test_top_down_discounting_and_defaults():
    pits = nested(model(12), discount_rate=0.15, benches_per_year=4, ore_per_year=6000)
    oes = pits.oes(search="lookahead")
    assert oes.discount_rate == 0.15
    assert oes.processing_rate == 6000
    assert oes.lookahead_discount == pytest.approx(0.15 / 4)
    assert oes.check().ok
    assert oes.bench_order is True
    assert oes.timing == "continuous"
    assert pits.oes().bench_order is True
    free = pits.oes(search="lookahead", bench_order=False)
    assert free.bench_order is False
    assert free.check().ok


def test_deterministic():
    pits = nested(model(13), discount_rate=0.1, ore_per_year=3000)
    for search in ("proximity", "lookahead"):
        assert np.array_equal(pits.oes(search=search).order, pits.oes(search=search).order)


def test_reports_and_files(tmp_path, rendering):
    pits = nested(model(14), discount_rate=0.1, ore_per_year=3000)
    oes = pits.oes(search="lookahead")
    text = oes.summary()
    assert "Ultimate pit OES" in text
    assert "NPV of the whole sequence" in text
    t = oes.table()
    assert t["rank"].size == oes.order.size
    assert np.isclose(t["npv"][-1], oes.npv)
    written = oes.export(tmp_path / "oes", details=True)
    names = ["report_cumulative.csv", "report_incremental.csv", "model.csv", "npv.png", "oes.gif", "oes_surface.gif",
             "oes.csv", "oes_periods.csv", "oes_report.txt", "oes_npv.png"]  # fmt: skip
    assert [p.name for p in written] == names
    assert all(p.stat().st_size > 0 for p in written)
    whole = pymineflow.read_blocks(tmp_path / "oes" / "model.csv")
    assert np.array_equal(whole["OES_RANK"], oes.rank)
    assert "_uses" not in pymineflow.read_blocks(tmp_path / "oes" / "oes_periods.csv")
    # every part is optional
    only = oes.export(tmp_path / "a", report="incremental", pit="final", chart=False, gif=False)
    assert [p.name for p in only] == ["report_incremental.csv", "final_pit.csv"]
    both = oes.export(tmp_path / "b", report=None, pit="both", chart=False, gif=False)
    assert [p.name for p in both] == ["model.csv", "final_pit.csv"]
    assert oes.export(tmp_path / "c", report=None, pit=None, chart=False, gif=False) == []
    for wrong in ({"report": "all"}, {"pit": "inside"}):
        with pytest.raises(ValueError, match=next(iter(wrong))):
            oes.export(tmp_path / "d", **wrong)
    with pytest.raises(ValueError, match="pit"):
        oes.to_csv(tmp_path / "e.csv", pit="both")


def test_model_table_whole_or_final_pit(tmp_path):
    pits = nested(model(19), discount_rate=0.1, ore_per_year=3000)
    oes = pits.oes()
    whole = pymineflow.read_blocks(oes.to_csv(tmp_path / "whole.csv", pit="whole"))
    final = pymineflow.read_blocks(oes.to_csv(tmp_path / "final.csv", pit="final"))
    inside = oes.phase > 0
    assert len(whole["IX"]) == oes.phase.size
    assert np.array_equal(whole["FINAL_PIT"], inside.astype(int))
    assert np.array_equal(whole["PIT_NUMBER"], pits.numbers)
    assert np.array_equal(whole["OES_PHASE"], oes.phase)
    assert np.array_equal(whole["OES_PERIOD"], oes.period)
    assert len(final["IX"]) == int(inside.sum())
    assert (final["FINAL_PIT"] == 1).all()
    assert (final["OES_RANK"] >= 0).all()
    assert np.allclose(final["VALUE"], oes.values[inside])
    assert sorted(final["OES_RANK"]) == list(range(oes.order.size))
    sequence = pymineflow.read_blocks(oes.to_csv(tmp_path / "sequence.csv"))  # extraction order
    assert np.array_equal(sequence["rank"], np.arange(oes.order.size))


@pytest.mark.parametrize("timing", ["continuous", "mid"])
def test_cumulative_and_incremental_reports(timing):
    pits = nested(model(20), discount_rate=0.1, ore_per_year=3000)
    oes = pits.oes(timing=timing, min_component=0)  # every nested pit with new blocks is a phase
    c, i = oes.report_table(), oes.report_table("incremental")
    assert not np.isnan(c["oes_npv"]).any()
    ref = pits.report_table(timing=timing)  # the same rate, plant and timing: the same phases and NPV
    for key in ("phase", "factor", "profit", "rock", "ore", "waste", "npv_best", "npv_worst"):
        assert np.allclose(c[key], ref[key], equal_nan=True), key
    assert np.allclose(pits.report_table(timing=timing, kind="incremental")["npv"], i["npv"])
    for key in ("profit", "rock", "ore", "waste"):  # the increments add up to the cumulative report
        assert np.allclose(np.cumsum(i[key]), c[key]), key
    assert np.allclose(np.cumsum(i["npv"]), c["npv_best"])
    assert np.allclose(np.cumsum(i["oes_npv"]), c["oes_npv"])
    assert c["oes_npv"][-1] == pytest.approx(oes.npv)
    assert c["oes_end"][-1] == pytest.approx(oes.time[-1])
    assert np.allclose(i["oes_start"][1:], i["oes_end"][:-1])
    assert np.allclose(i["profit_per_t"], i["profit"] / i["rock"])
    assert np.allclose(i["strip_ratio"], i["waste"] / i["ore"])
    # the sequence is the best case mined block by block: close to it, phase by phase
    assert np.allclose(c["oes_npv"], c["npv_best"], rtol=0.05)
    assert list(c)[:8] == ["phase", "factor", "profit", "profit_per_t", "npv_best", "npv_worst", "oes_npv", "oes_end"]
    assert "Highest NPV" in oes.report()
    assert "incremental" in oes.report("incremental")
    assert "OES start" in oes.report("incremental")
    with pytest.raises(ValueError, match="kind"):
        oes.report_table("total")
    # merged phases: the rows of the nested pits that do not end a phase of the sequence have no OES columns
    third = np.flatnonzero(np.bincount(pits.numbers, minlength=len(pits) + 1)[1:])[2] + 1  # the third phase
    merged = pits.oes(timing=timing, phases=[third], min_component=0)
    m, mi = merged.report_table(), merged.report_table("incremental")
    assert np.isnan(m["oes_npv"]).sum() == len(m["phase"]) - 2
    assert m["oes_npv"][-1] == pytest.approx(merged.npv)
    assert np.nansum(mi["oes_npv"]) == pytest.approx(merged.npv)
    # continuity moves the small first pits to later phases: those rows have no OES columns either
    moved = pits.oes(timing=timing)
    mc, mi = moved.report_table(), moved.report_table("incremental")
    ends = ~np.isnan(mc["oes_npv"])
    assert ends[-1]
    assert np.array_equal(ends, ~np.isnan(mi["oes_npv"]))
    assert np.allclose(np.nancumsum(mi["oes_npv"])[ends], mc["oes_npv"][ends])


def test_cases_chart(tmp_path, rendering):
    pits = nested(model(21), discount_rate=0.1, ore_per_year=3000)
    oes = pits.oes()
    fig = oes.plot_cases(tmp_path / "cases.png")
    assert (tmp_path / "cases.png").stat().st_size > 0
    assert fig is not None
    assert np.allclose(oes.cases().table["npv_best"], pits.npv(timing="continuous").table["npv_best"])


def test_views_and_animations(tmp_path, rendering):
    """The same scene for the nested pits and the OES: what is mined growing, or the whole pit mined away."""
    pits = nested(model(18), discount_rate=0.1, ore_per_year=3000)
    oes = pits.oes()
    for remaining in (False, True):
        gif = pits.animate(tmp_path / f"nested_{remaining}.gif", remaining=remaining, view="top")
        assert gif.stat().st_size > 0
        for color_by in ("rank", "period", "phase"):
            path = oes.animate(tmp_path / f"oes_{remaining}_{color_by}.gif", frames=12, color_by=color_by,
                               remaining=remaining)  # fmt: skip
            assert path.stat().st_size > 0
        oes.plot(remaining=remaining, screenshot=tmp_path / f"oes_{remaining}.png", show=False).close()
        pits.plot(remaining=remaining, screenshot=tmp_path / f"pits_{remaining}.png", show=False).close()
    scene = oes._scene(frames=10, color_by="rank", remaining=True, view="top", rotation=0.0, title=None,
                       outline=False, axes=False, cmap="jet", window_size=(300, 200), off_screen=True)  # fmt: skip
    first, last = scene.steps()[0], scene.steps()[-1]
    assert first == 0  # nothing mined: the whole final pit
    whole = scene._shown(first).sum()
    scene.show(last)
    assert 0 < scene.mesh.n_cells < whole  # only the last part is left
    scene.plotter.close()
    with pytest.raises(ValueError, match="frames"):
        oes.animate(tmp_path / "x.gif", frames=0)
    with pytest.raises(ValueError, match="color_by"):
        oes.plot(color_by="value", show=False)


def test_schedule_blocks_larger_than_a_period():
    track = np.zeros((0, 2))
    caps = np.array([1.0])
    for continuous in (False, True):  # block 0 takes periods 1 to 3, block 1 the rest of period 3
        out = _oes_schedule(np.array([10.0, 20.0]), np.array([[2.5, 0.5]]), caps, track, 0.1, 0.0, 0.0, continuous)
        assert out["time_end"] == pytest.approx([2.5, 3.0])
        assert list(out["first_period"]) == [1, 3]
        assert out["period_value"] == pytest.approx([4.0, 4.0, 22.0])
    # continuous discounting is exact also for very short pieces: mean of exp(-r t) over [0, h]
    out = _oes_schedule(np.array([1.0]), np.array([[1e-11]]), caps, np.zeros((0, 1)), 0.5, 0.0, 0.0, True)
    assert out["discounted"][0] == pytest.approx(-math.expm1(-0.5e-11) / 0.5e-11, rel=1e-15)
    # capacities in other units than the tonnage: an error, not a schedule without end
    with pytest.raises(ValueError, match="1,000,000 periods"):
        _oes_schedule(np.array([1.0]), np.array([[1e13]]), caps, np.zeros((0, 1)), 0.1, 0.0, 0.0, True)


def test_errors():
    pits = nested(model(15))
    with pytest.raises(ValueError, match="discount_rate"):
        pits.oes(processing_rate=1000)
    with pytest.raises(ValueError, match="mining_rate, processing_rate"):
        pits.oes(discount_rate=0.1)
    with pytest.raises(ValueError, match="search"):
        pits.oes(discount_rate=0.1, processing_rate=1000, search="best")
    with pytest.raises(ValueError, match="phases"):
        pits.oes(discount_rate=0.1, processing_rate=1000, phases=[3, 2])
    for wrong in ([1.5, 3], [True, 3], ["a"]):  # never read in silence as other pit numbers
        with pytest.raises(ValueError, match="phases"):
            pits.oes(discount_rate=0.1, processing_rate=1000, phases=wrong)
    assert pits.oes(discount_rate=0.1, processing_rate=1000, phases=[2.0]).npv == pytest.approx(
        pits.oes(discount_rate=0.1, processing_rate=1000, phases=[2]).npv
    )
    with pytest.raises(ValueError, match="lookahead_benches"):
        pits.oes(discount_rate=0.1, processing_rate=1000, lookahead_benches=-1)
    with pytest.raises(ValueError, match="proximity"):
        pits.oes(discount_rate=0.1, processing_rate=1000, proximity=-1)
    empty = pymineflow.nested_pits({"IX": [0], "IY": [0], "IZ": [0], "VALUE": [-5.0], "TON": [1.0]},
                                   block_size=SIZE, slopes=45, factors=[1.0])  # fmt: skip
    with pytest.raises(ValueError, match="empty"):
        empty.oes(discount_rate=0.1, processing_rate=10)


def test_npv_keeps_ore_per_year_with_a_mining_rate():
    pits = nested(model(16), discount_rate=0.1, ore_per_year=3000)
    got = pits.npv(mining_rate=9000).table["npv_best"]
    expected = pymineflow.pit_by_pit(pits, discount_rate=0.1, mining_rate=9000, processing_rate=3000).table["npv_best"]
    assert np.allclose(got, expected)


def test_command_line(tmp_path, capsys, rendering):  # --oes writes the GIFs: it draws
    table = model(17)
    path = tmp_path / "model.csv"
    pymineflow.write_table(table, path)
    base = [str(path), "--block-size", "10", "10", "8", "--slope", "45", "--factors", "0.2:1:0.2", "--no-output"]
    folder = tmp_path / "oes"
    assert main([*base, "--discount-rate", "0.1", "--ore-per-year", "20000", "--oes", str(folder),
                 "--search", "lookahead", "--lookahead-benches", "3", "--min-component", "10"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "Ultimate pit OES" in out
    assert "ok   slopes: 0 blocks" in out
    assert (folder / "oes.csv").exists()
    assert (folder / "oes_report.txt").exists()
    assert {"report_cumulative.csv", "report_incremental.csv", "model.csv", "final_pit.csv", "npv.png",
            "oes.gif"} <= {f.name for f in folder.iterdir()}  # fmt: skip
    assert main([*base, "--oes", str(folder)]) == 1
    # a zero discount rate, and the same NPV as the API (continuous timing by default)
    pits = pymineflow.nested_pits(table, block_size=(10, 10, 8), slopes=45, factors=[0.2, 0.4, 0.6, 0.8, 1.0])
    for rate in ("0", "0.1"):
        assert main([*base, "--discount-rate", rate, "--ore-per-year", "20000", "--oes", str(folder)]) == 0
        npv = pits.oes(discount_rate=float(rate), processing_rate=20000).npv
        assert f"NPV of the whole sequence {npv:,.0f}" in (folder / "oes_report.txt").read_text(encoding="utf-8")
    capsys.readouterr()
    assert main([str(path), "--block-size", "10", "--slope", "45", "--search", "lookahead"]) == 1
    # --ore-per-year is the plant also with --mining-rate (the pit-by-pit table and the report agree)
    assert main([*base, "--discount-rate", "0.1", "--ore-per-year", "3000", "--mining-rate", "9000",
                 "--npv-report", str(tmp_path / "npv.csv")]) == 0  # fmt: skip
    npv = pymineflow.read_blocks(tmp_path / "npv.csv")["npv_best"]
    pits = pymineflow.nested_pits(table, block_size=(10, 10, 8), slopes=45, factors=[0.2, 0.4, 0.6, 0.8, 1.0])
    expected = pymineflow.pit_by_pit(pits, discount_rate=0.1, mining_rate=9000, processing_rate=3000).table["npv_best"]
    assert np.allclose(npv, expected)


def test_surface_view_digs_the_ground(tmp_path, rendering):
    """The surface of every column: the top of the model before mining, the bottom of the deepest block mined
    after, step after step, for the OES and for the nested pits."""
    from pymineflow.plot import _model_bounds, _SurfaceScene

    pits = nested(model(22, nx=10, ny=9, nz=7), discount_rate=0.1, ore_per_year=3000)
    oes = pits.oes()
    scene = oes._scene(frames=12, color_by="rank", remaining=True, surface=True, view="iso", rotation=0.0,
                       title=None, outline=False, axes=False, cmap="jet", window_size=(400, 300),
                       off_screen=True)  # fmt: skip
    assert isinstance(scene, _SurfaceScene)
    g = pits.model.grid
    z0, dz = scene.z0, scene.dz
    ix, iy, iz = (np.asarray(a) for a in (g.ix, g.iy, g.iz))
    column = iy * g.nx + ix
    life = float(oes.time[-1])
    step = np.zeros(oes.phase.size, dtype=np.int64)
    step[oes.order] = np.clip(np.ceil(oes.time / life * 12 - 1e-9), 1, 12)
    for k in (0, 5, 12):
        expected = np.full(g.nx * g.ny, np.nan)
        for c in np.unique(column):
            rows = column == c
            done = rows & (step >= 1) & (step <= k)
            expected[c] = z0 + (iz[done].min() if done.any() else iz[rows].max() + 1) * dz
        got = scene._heights(k)
        assert np.allclose(got[~np.isnan(expected)], expected[~np.isnan(expected)])
        # every point of the mesh stands on the centre of its column, at that column's height
        scene.show(k)
        points = np.asarray(scene.mesh.points)
        x0, _, y0, _, _, _ = (float(b) for b in _model_bounds(g))
        assert np.allclose(points[:, 0], x0 + (scene.ids % g.nx + 0.5) * g.block_size[0])
        assert np.allclose(points[:, 1], y0 + (scene.ids // g.nx + 0.5) * g.block_size[1])
        assert np.allclose(points[:, 2], expected[scene.ids])
    scene.plotter.close()
    assert oes.animate(tmp_path / "s.gif", surface=True, frames=8, view="iso").stat().st_size > 0
    assert pits.animate(tmp_path / "n.gif", surface=True).stat().st_size > 0
    oes.plot(surface=True, view="iso", screenshot=tmp_path / "s.png").close()
    assert (tmp_path / "s.png").stat().st_size > 0
