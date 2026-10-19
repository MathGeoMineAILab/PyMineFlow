"""Nested pits by bisection on rings: the same pits as solving each scenario alone, and the exact cone.

The rings use the very same precedence pattern (the same slopes) as the whole grid, so these tests
check the bookkeeping: nesting order, ring construction, inner pits taken as mined, results order.
"""

import itertools

import numpy as np
import pytest

from pymineflow import Pit, PitModel, ultimate_pit

from _oracle import grid_cone_value, present_cone_value, topography_model

FACTORS = (0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3)
SIZE = (10.0, 10.0, 8.0)


def scenarios(v, factors=FACTORS):
    """Revenue factors: positive values scaled, negative ones kept (no block loses value as f grows)."""
    return [np.where(v > 0, np.rint(v * f), v).astype(np.int64) for f in factors]


def alone(scenario_values, **kwargs):
    return [ultimate_pit(s, **kwargs) for s in scenario_values]


@pytest.mark.parametrize(
    "slopes",
    [45, 38.5, [(0, 40), (90, 48), (180, 44), (270, 50)], [(30, 52), (200, 46)]],
    ids=["45", "38.5", "four-sectors", "two-sectors"],
)
@pytest.mark.parametrize("seed", range(4))
def test_same_pits_as_alone_and_exact_cone(seed, slopes):
    ix, iy, iz, v = topography_model(seed)
    S = scenarios(v)
    kw = {"ix": ix, "iy": iy, "iz": iz, "block_size": SIZE, "slopes": slopes}
    nested = PitModel(**kw).nested_pits(S)
    for r, ref, s in zip(nested, alone(S, **kw), S, strict=True):
        assert r.value == ref.value
        assert np.array_equal(r.mask, ref.mask)
        assert r.value == int(s[r.mask].sum())
        assert r.n_blocks == int(r.mask.sum())
        if np.ndim(slopes) == 0:  # constant slope: also the exact cone (max-flow on every cone arc)
            assert r.value == grid_cone_value(r.grid, s, slopes)
    for small, large in itertools.pairwise(nested):
        assert (small.mask <= large.mask).all()  # nested


@pytest.mark.parametrize("seed", range(3))
def test_domains_with_different_slopes(seed):
    ix, iy, iz, v = topography_model(seed, nx=10, ny=8, nz=5)
    domains = (ix >= 5).astype(np.int64)
    specs = [[(0, 38)], [(0, 50), (180, 44)]]
    S = scenarios(v)
    kw = {"ix": ix, "iy": iy, "iz": iz, "block_size": SIZE, "domains": domains, "domain_slopes": specs}
    model = PitModel(**kw)
    assert model.pattern == "full"
    for r, s in zip(model.nested_pits(S), S, strict=True):
        assert r.value == present_cone_value(r.grid, s, domains, specs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"slopes": None},
        {"slopes": 45, "n_benches": 2},
        {"slopes": 45, "full_pattern": True},
        {"slopes": 45, "z_up": False},
    ],
    ids=["1:5", "n_benches=2", "full_pattern", "z-down"],
)
def test_other_patterns(kwargs):
    ix, iy, iz, v = topography_model(7)
    if kwargs.get("z_up") is False:
        iz = iz.max() - iz
    S = scenarios(v)
    kw = {"ix": ix, "iy": iy, "iz": iz, "block_size": SIZE, **kwargs}
    for r, ref in zip(PitModel(**kw).nested_pits(S), alone(S, **kw), strict=True):
        assert r.value == ref.value
        assert np.array_equal(r.mask, ref.mask)


def test_any_input_order_and_repeated_scenarios():
    ix, iy, iz, v = topography_model(3)
    S = scenarios(v)
    order = [5, 0, 9, 2, 2, 7, 1, 8, 3, 6, 4]  # shuffled, with a repeated scenario
    kw = {"ix": ix, "iy": iy, "iz": iz, "block_size": SIZE, "slopes": 45}
    got = PitModel(**kw).nested_pits([S[k] for k in order])
    ref = alone(S, **kw)
    for r, k in zip(got, order, strict=True):
        assert r.value == ref[k].value
        assert np.array_equal(r.mask, ref[k].mask)


def test_scenarios_that_are_not_nested_are_solved_alone():
    ix, iy, iz, v = topography_model(4)
    rng = np.random.default_rng(0)
    S = [v, rng.permutation(v), scenarios(v)[3], -np.abs(v)]  # no common order
    kw = {"ix": ix, "iy": iy, "iz": iz, "block_size": SIZE, "slopes": 45}
    for r, ref in zip(PitModel(**kw).nested_pits(S), alone(S, **kw), strict=True):
        assert r.value == ref.value
        assert np.array_equal(r.mask, ref.mask)


def test_large_number_of_scenarios():
    ix, iy, iz, v = topography_model(11, nx=14, ny=12, nz=7)
    factors = np.linspace(0.1, 2.0, 60)
    S = scenarios(v, factors)
    kw = {"ix": ix, "iy": iy, "iz": iz, "block_size": SIZE, "slopes": 45}
    got = PitModel(**kw).nested_pits(S)
    for k in range(0, 60, 7):
        assert got[k].value == grid_cone_value(got[k].grid, S[k], 45)
    assert all((a.mask <= b.mask).all() for a, b in itertools.pairwise(got))


@pytest.mark.parametrize("seed", range(3))
def test_parallel_bisection_gives_the_same_pits(seed):
    """The two halves of every bisection are solved at the same time: same numbers, values and order of calls."""
    ix, iy, iz, v = topography_model(20 + seed, nx=16, ny=13, nz=8)
    S = scenarios(v, np.linspace(0.05, 2.0, 45))
    model = PitModel(ix=ix, iy=iy, iz=iz, block_size=SIZE, slopes=45)

    def scenario(k, rows):
        return S[k] if rows is None else S[k][rows]

    serial = model.nested_pit_numbers(scenario, len(S), threads=1)
    seen = []
    parallel = model.nested_pit_numbers(scenario, len(S), seen.append, threads=4)
    assert np.array_equal(serial[0], parallel[0])
    assert serial[1] == parallel[1]
    assert sorted(d["pit"] for d in seen) == list(range(1, len(S) + 1))
    assert [d["order"] for d in seen] == list(range(1, len(S) + 1))
    for k in range(0, len(S), 9):
        mask = (parallel[0] >= 1) & (parallel[0] <= k + 1)
        assert np.array_equal(mask, ultimate_pit(S[k], ix=ix, iy=iy, iz=iz, block_size=SIZE, slopes=45).mask)
    for wrong in (-1, 1.5, True):
        with pytest.raises(ValueError, match="threads"):
            model.nested_pit_numbers(scenario, len(S), threads=wrong)


def test_solve_subset_from_several_threads():
    from concurrent.futures import ThreadPoolExecutor

    nx, ny, nz = 9, 8, 6
    v = np.random.default_rng(3).integers(-6, 6, nx * ny * nz).astype(np.int64)
    pit = Pit(nx, ny, nz, v, pattern="minsearch", slope_deg=40, n_benches=nz - 1, block_size=SIZE)
    whole = pit.solve()
    cells = np.arange(nx * ny * nz)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: pit.solve_subset(cells, v), range(32)))
    assert all(np.array_equal(r[0], whole[0]) and r[1] == whole[1] for r in results)


# The C++ subset solve


def test_solve_subset_equals_the_whole_grid():
    nx, ny, nz = 7, 6, 5
    v = np.random.default_rng(1).integers(-6, 6, nx * ny * nz).astype(np.int64)
    pit = Pit(nx, ny, nz, v, pattern="minsearch", slope_deg=40, n_benches=nz - 1, block_size=SIZE)
    whole = pit.solve()
    cells = np.arange(nx * ny * nz)
    mask, value, n_blocks, _ = pit.solve_subset(cells, v)
    assert value == whole[1]
    assert np.array_equal(mask, whole[0])
    assert n_blocks == whole[2]
    empty = pit.solve_subset(np.array([], np.int64), np.array([], np.int64))
    assert empty[0].size == 0
    assert empty[1] == 0


def test_solve_subset_errors():
    pit = Pit(3, 3, 3, np.zeros(27, np.int64), pattern="minsearch", slope_deg=45, n_benches=2)
    with pytest.raises(ValueError, match="repeated"):
        pit.solve_subset([1, 1], [1, 1])
    with pytest.raises(ValueError, match="out of the grid"):
        pit.solve_subset([0, 27], [1, 1])
    with pytest.raises(ValueError, match="expected 2 values"):
        pit.solve_subset([0, 1], [1])
    with pytest.raises(TypeError, match="must be integers"):
        pit.solve_subset([0, 1], [1.5, 2.0])
