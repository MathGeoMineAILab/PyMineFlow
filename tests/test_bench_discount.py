"""Top-down bench discounting: only the block weights change, checked against independent references."""

import numpy as np
import pytest

import pymineflow
from pymineflow import Economics, Metal, Process, bench_discount_weights, scale_values, ultimate_pit
from pymineflow.cli import main

from _oracle import brute_force, grid_cone_value, topography_model

SIZE = (10.0, 10.0, 10.0)


def test_weights():
    assert bench_discount_weights([7, 6, 5, 7], 0.0125).tolist() == pytest.approx([1.0, 1 / 1.0125, 1 / 1.0125**2, 1.0])
    assert bench_discount_weights([3, 1], 0.0).tolist() == [1.0, 1.0]
    assert bench_discount_weights([], 0.1).size == 0
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="bench_discount must be finite and >= 0"):
            bench_discount_weights([1, 2], bad)
    with pytest.raises(TypeError, match="bench_discount must be a number"):
        bench_discount_weights([1, 2], "1%")


def discounted(table, rate):
    """The reference: discount the values by hand, independently of the library."""
    iz = np.asarray(table["IZ"])
    return np.asarray(table["VALUE"], dtype=np.float64) * (1.0 + rate) ** -(iz.max() - iz).astype(np.float64)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("rate", [0.0125, 0.05, 0.3])
def test_solve_against_max_flow(seed, rate):
    """The discounted pit has the maximum discounted value (max-flow of scipy on the exact cone; values
    divided by 100 so that the scaled integers fit the int32 capacities of scipy)."""
    ix, iy, iz, values = topography_model(seed)
    table = {"IX": ix, "IY": iy, "IZ": iz, "VALUE": values / 100.0}
    pit = pymineflow.solve(table, block_size=SIZE, slopes=45, bench_discount=rate)
    ints, factor = scale_values(discounted(table, rate))
    assert pit.factor == factor
    assert pit.value == grid_cone_value(pit.grid, ints, 45)
    assert pit.value == int(ints[pit.mask].sum())
    assert "discounted" in pit.value_column


def test_solve_brute_force():
    rng = np.random.default_rng(3)
    ix, iy, iz = (a.ravel() for a in np.meshgrid(np.arange(4), [0], np.arange(3), indexing="ij"))
    for _ in range(25):
        values = rng.integers(-30, 60, ix.size).astype(np.float64)
        table = {"IX": ix, "IY": iy, "IZ": iz, "VALUE": values}
        rate = float(rng.choice([0.01, 0.2, 1.0]))
        pit = pymineflow.solve(table, block_size=SIZE, slopes=45, bench_discount=rate)
        ints, _ = scale_values(discounted(table, rate))
        best = brute_force(ix, iy, iz, SIZE, ints, np.zeros(ix.size, int), [45])
        assert pit.value == best


def test_discounting_can_shrink_the_pit():
    """A deep ore block that pays for its cone undiscounted, but not once discounted."""
    ix = [0, 1, 2, 1]
    iz = [1, 1, 1, 0]
    table = {"IX": ix, "IY": [0] * 4, "IZ": iz, "VALUE": [-10.0, -10.0, -10.0, 31.0]}
    plain = pymineflow.solve(table, block_size=SIZE, slopes=45)
    assert plain.mask.all()
    assert plain.real_value == pytest.approx(1.0)
    pit = pymineflow.solve(table, block_size=SIZE, slopes=45, bench_discount=0.05)  # 31 / 1.05 < 30
    assert not pit.mask.any()
    # depth indices (z_up=False): the same model upside down gives the same pits
    flipped = {**table, "IZ": [1 - z for z in iz]}
    assert (
        pymineflow.solve(flipped, block_size=SIZE, slopes=45, z_up=False, bench_discount=0.05).mask.tolist()
        == [False] * 4
    )
    assert pymineflow.solve(flipped, block_size=SIZE, slopes=45, z_up=False).mask.all()


def test_zero_rate_changes_nothing():
    ix, iy, iz, values = topography_model(1)
    table = {"IX": ix, "IY": iy, "IZ": iz, "VALUE": values.astype(np.float64) / 7}
    a = pymineflow.solve(table, block_size=SIZE, slopes=45)
    b = pymineflow.solve(table, block_size=SIZE, slopes=45, bench_discount=0.0)
    assert np.array_equal(a.mask, b.mask)
    assert a.value == b.value
    assert a.value_column == b.value_column
    p = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=[0.5, 1.0])
    q = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=[0.5, 1.0], bench_discount=0)
    assert np.array_equal(p.numbers, q.numbers)
    # the reports match line by line, except the solver time of the last line, which varies run to run
    assert p.summary().splitlines()[:-1] == q.summary().splitlines()[:-1]
    assert q.summary().splitlines()[-1].startswith("solver time")


@pytest.mark.parametrize("seed", range(4))
def test_nested_profit_factors(seed):
    """Profit factors (positive values times the factor): every nested pit is the discounted ultimate pit
    of its factor, solved alone."""
    ix, iy, iz, values = topography_model(seed)
    table = {"IX": ix, "IY": iy, "IZ": iz, "VALUE": values / 100.0}  # int32 capacities of scipy
    rate = 0.04
    factors = [0.2, 0.35, 0.5, 0.75, 1.0]
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=factors, bench_discount=rate)
    ints, factor = scale_values(discounted(table, rate))
    assert pits.factor == factor
    for k, f in enumerate(factors):
        scenario = np.where(ints > 0, np.rint(ints * f), ints).astype(np.int64)
        alone = ultimate_pit(scenario, ix=ix, iy=iy, iz=iz, block_size=SIZE, slopes=45)
        assert pits[k].value == alone.value
        assert pits[k].value == grid_cone_value(alone.grid, scenario, 45)
    # the reports: objective discounted, value at factor 1 undiscounted
    t = pits.table_by_pit()
    last = pits[-1].mask
    assert t["value_at_factor_1"][-1] == pytest.approx(table["VALUE"][last].sum())
    assert t["value"][-1] == pytest.approx(discounted(table, rate)[last].sum(), rel=1e-6, abs=1e-3)
    assert "top-down bench discounting 4.0000% per bench" in pits.summary()
    assert "discounted value" in pits.summary()


@pytest.mark.parametrize("scale", ["revenue", "price"])
def test_nested_with_economics(scale):
    """Grades, prices and costs, then discounted: every pit equals its factor solved alone."""
    rng = np.random.default_rng(5)
    ix, iy, iz = (a.ravel() for a in np.meshgrid(np.arange(9), np.arange(7), np.arange(6), indexing="ij"))
    tons = rng.uniform(900, 1100, ix.size)
    grade = np.maximum(rng.normal(0.5 + 0.12 * (5 - iz), 0.6), 0)
    table = {"IX": ix, "IY": iy, "IZ": iz, "TON": tons, "AU": grade}
    eco = Economics(
        metals=[Metal("AU", price=60, grade_unit="g/t", price_unit="g", selling_cost=12)],
        processes=[Process("mill", cost=25, recovery={"AU": 0.9})],
        mining_cost=2,
        tonnage="TON",
    )
    factors = [0.4, 0.6, 0.8, 1.0, 1.2]
    rate = 0.03
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=factors, economics=eco, scale=scale,
                                  bench_discount=rate)  # fmt: skip
    w = (1 + rate) ** -(iz.max() - iz).astype(float)
    keyword = "price_factor" if scale == "price" else "revenue_factor"
    for k, f in enumerate(factors):
        real = eco.block_values(table, **{keyword: f})[0] * w
        ints = np.rint(real * pits.factor).astype(np.int64)
        alone = ultimate_pit(ints, ix=ix, iy=iy, iz=iz, block_size=SIZE, slopes=45)
        assert pits[k].value == alone.value
    # the value at factor 1 of the reports stays undiscounted
    t = pits.table_by_pit()
    assert t["value_at_factor_1"][-1] == pytest.approx(eco.block_values(table)[0][pits[-1].mask].sum())


def test_errors_and_command_line(tmp_path, capsys):
    ix, iy, iz, values = topography_model(2)
    table = {"IX": ix, "IY": iy, "IZ": iz, "VALUE": values.astype(np.float64)}
    with pytest.raises(ValueError, match="bench_discount must be finite and >= 0"):
        pymineflow.solve(table, block_size=SIZE, slopes=45, bench_discount=-0.1)
    with pytest.raises(ValueError, match="bench_discount must be finite and >= 0"):
        pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=[1.0], bench_discount=float("nan"))
    path = tmp_path / "model.csv"
    pymineflow.write_table(table, path)
    args = [str(path), "--block-size", "10", "10", "10", "--slope", "45", "--no-output", "--bench-discount", "0.05"]
    assert main(args) == 0
    assert "discounted 5.0000% per bench" in capsys.readouterr().out
    assert main([*args, "--factors", "0.5", "1"]) == 0
    assert "Top-down bench discounting: 5.0000% per bench" in capsys.readouterr().out
    assert main([*args[:-1], "-1"]) == 1
    assert "bench_discount must be finite" in capsys.readouterr().err
