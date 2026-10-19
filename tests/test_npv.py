"""Pit-by-pit discounted value, checked against a period-by-period schedule written independently."""

import numpy as np
import pytest

import pymineflow
from pymineflow import Contaminant, Economics, Metal, Process, read_blocks
from pymineflow.cli import main

SIZE = (10.0, 10.0, 10.0)
SHIFT = {"end": 0.0, "mid": 0.5, "start": 1.0}


def random_model(seed, nx=9, ny=7, nz=5):
    rng = np.random.default_rng(seed)
    ix, iy, iz = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"))
    depth = (nz - 1 - iz).astype(float)
    grade = np.maximum(rng.normal(0.4 + 0.15 * depth, 0.5), 0.0)  # richer at depth
    tons = rng.uniform(800.0, 1200.0, ix.size)
    value = np.round((grade - 0.6) * tons * 2.5, 2)
    return {"IX": ix, "IY": iy, "IZ": iz, "VALUE": value, "TON": tons, "CU": grade, "AS": rng.uniform(0, 3, ix.size)}


# ----------------------------------------------------------------- the reference, written unit by unit


def units_of(pits, values, destination, tons, n_processes, phases, final):
    """(value, tonnage, ore per process) of every bench of every phase, from the top, block by block."""
    numbers = pits.numbers
    iz = pits.model.grid.iz
    limits = [p for p in phases if p < final] + [final]
    out, low = [], 0
    for high in limits:
        rows = [i for i in range(numbers.size) if low < numbers[i] <= high]
        for level in sorted({int(iz[i]) for i in rows}, reverse=True):
            bench = [i for i in rows if iz[i] == level]
            ore = [sum(tons[i] for i in bench if destination[i] == p) for p in range(n_processes)]
            out.append((sum(values[i] for i in bench), sum(tons[i] for i in bench), ore))
        low = high
    return out


def reference(units, rate, mining, plant, names, fixed, timing):
    """Mine the units in order; each period takes as much as every capacity left allows."""
    constraints = []  # (capacity, use of a unit)
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
    share = max((cap - room) / cap for (cap, _), room in zip(constraints, left, strict=True))
    share = min(max(share, 0.0), 1.0)
    life = period - 1 + share

    def factor(k):
        return (1.0 + rate) ** -(k - SHIFT[timing])

    npv = sum(v * factor(k) for k, v in cash.items())
    npv -= fixed * (sum(factor(k) for k in range(1, period)) + share * factor(period))
    return npv, life


def check(pits, values, destination, tons, rate, mining, plant, fixed, timing, pushbacks, names=("ore",), **kw):
    r = pits.npv(discount_rate=rate, mining_rate=mining, processing_rate=plant, fixed_cost=fixed, timing=timing,
                 pushbacks=pushbacks, **kw)  # fmt: skip
    n = len(pits)
    cases = {"best": list(range(1, n + 1)), "worst": [], "specified": list(pushbacks or [])}
    for case, phases in cases.items():
        if case == "specified" and pushbacks is None:
            assert "npv_specified" not in r.table
            continue
        for final in range(1, n + 1):
            units = units_of(pits, values, destination, tons, len(names), phases, final)
            npv, life = reference(units, rate, mining, plant, list(names), fixed, timing)
            assert r.table[f"npv_{case}"][final - 1] == pytest.approx(npv, rel=1e-9, abs=1e-6), (case, final)
            assert r.table[f"life_{case}"][final - 1] == pytest.approx(life, rel=1e-9, abs=1e-9), (case, final)
    return r


# ----------------------------------------------------------------- against the reference


@pytest.mark.parametrize("seed", [0, 1, 2])
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
def test_matches_period_by_period_schedule(seed, rate, mining, plant, fixed, timing):
    table = random_model(seed)
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=np.linspace(0.4, 1.0, 7))
    values = pits.base()
    destination = np.where(values > 0, 0, -1)
    pushbacks = [2, 5] if len(pits) >= 5 else None
    check(pits, values, destination, table["TON"], rate, mining, plant, fixed, timing, pushbacks)


def test_matches_with_an_ore_column():
    """MineLib-like data: processed blocks given by a column, some of them of negative value."""
    table = random_model(11)
    rng = np.random.default_rng(0)
    table["DESTINATION"] = ((table["VALUE"] > 0) | (rng.random(table["VALUE"].size) < 0.15)).astype(int)
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=np.linspace(0.3, 1.0, 6))
    values = pits.base()
    destination = np.where(table["DESTINATION"] == 1, 0, -1)
    assert ((values <= 0) & (destination == 0)).any()  # marginal blocks processed
    for mining, plant in ((None, 3000.0), (15000.0, 3000.0)):
        check(pits, values, destination, table["TON"], 0.15, mining, plant, 0.0, "start", [3], ore="DESTINATION")
    by_array = pits.npv(discount_rate=0.15, processing_rate=3000.0, ore=table["DESTINATION"] == 1).table
    by_column = pits.npv(discount_rate=0.15, processing_rate=3000.0, ore="DESTINATION").table
    assert np.array_equal(by_array["npv_best"], by_column["npv_best"])


def test_matches_with_economics_and_two_processes():
    table = random_model(7)
    eco = Economics(
        metals=[Metal("CU", price=600.0, grade_unit="%", price_unit="t")],  # mill above ~0.33 %, leach below
        processes=[
            Process("mill", cost=1.2, recovery={"CU": 0.9}),
            Process("leach", cost=0.4, recovery={"CU": 0.5}, limits={"AS": 2.0}),
        ],
        mining_cost=0.8,
        tonnage="TON",
        contaminants=[Contaminant("AS", penalty=0.3, threshold=1.0)],
    )
    for scale in ("revenue", "price"):
        pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=np.linspace(0.5, 1.4, 6),
                                      economics=eco, scale=scale)  # fmt: skip
        values, destination = eco.block_values(table)
        assert set(np.unique(destination)) >= {0, 1}  # both processes are used
        for plant in ({"mill": 2000.0, "leach": 3000.0}, {"leach": 1500.0}, 4000.0):
            check(pits, values, destination, table["TON"], 0.1, 15000.0, plant, 300.0, "end", [2, 4],
                  names=("mill", "leach"))  # fmt: skip
        with pytest.raises(ValueError, match="processing_rate keys must be among the processes"):
            pits.npv(discount_rate=0.1, processing_rate={"waste": 100.0})
        with pytest.raises(ValueError, match="ore= is for models without economics"):
            pits.npv(discount_rate=0.1, processing_rate=100.0, ore="CU")


# ----------------------------------------------------------------- properties


def test_without_discount_the_npv_is_the_value():
    table = random_model(3)
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=np.linspace(0.3, 1.0, 8))
    r = pits.npv(discount_rate=0.0, mining_rate=5000, processing_rate=2000, fixed_cost=100)
    t = r.table
    for case in ("best", "worst"):
        assert np.allclose(t[f"npv_{case}"], t["value"] - 100 * t[f"life_{case}"], rtol=1e-12, atol=1e-6)
    by_pit = pits.table_by_pit()
    assert np.allclose(t["value"], by_pit["value_at_factor_1"])
    assert np.allclose(t["tonnage"], by_pit["tonnage"])
    assert np.isclose(t["ore_tonnage"][-1], by_pit["ore_tonnage"][-1])  # the last factor is 1


def test_pushbacks_at_every_pit_or_none():
    table = random_model(4)
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=np.linspace(0.3, 1.0, 8))
    n = len(pits)
    every = pits.npv(discount_rate=0.1, mining_rate=6000, processing_rate=2500, pushbacks=range(1, n + 1)).table
    assert np.allclose(every["npv_specified"], every["npv_best"], rtol=1e-12)
    assert np.allclose(every["life_specified"], every["life_best"], rtol=1e-12)
    last = pits.npv(discount_rate=0.1, mining_rate=6000, processing_rate=2500, pushbacks=[n]).table
    assert np.allclose(last["npv_specified"], last["npv_worst"], rtol=1e-12)


def test_schedule_period_by_period():
    table = random_model(5)
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=np.linspace(0.3, 1.0, 8))
    r = pits.npv(discount_rate=0.12, mining_rate=6000, processing_rate=2000, fixed_cost=150, pushbacks=[3])
    for case in ("best", "worst", "specified"):
        for pit in (1, 4, len(pits)):
            s = r.schedule(pit, case)
            assert s["discounted"].sum() == pytest.approx(r.table[f"npv_{case}"][pit - 1], rel=1e-9)
            assert s["tonnage"].sum() == pytest.approx(r.table["tonnage"][pit - 1], rel=1e-9)
            assert s["ore_tonnage"].sum() == pytest.approx(r.table["ore_tonnage"][pit - 1], rel=1e-9)
            assert s["value"].sum() == pytest.approx(r.table["value"][pit - 1], rel=1e-9, abs=1e-6)
            assert (s["tonnage"] <= 6000 * (1 + 1e-9)).all()
            assert (s["ore_tonnage"] <= 2000 * (1 + 1e-9)).all()
            # every period but the last is full on the mine or on the plant
            full = np.isclose(s["tonnage"][:-1], 6000) | np.isclose(s["ore_tonnage"][:-1], 2000)
            assert full.all()
            assert len(s["period"]) == int(np.ceil(r.table[f"life_{case}"][pit - 1] - 1e-9))
    with pytest.raises(ValueError, match="pit must be a pit number"):
        r.schedule(0)
    with pytest.raises(ValueError, match="case must be one of"):
        pits.npv(discount_rate=0.1, mining_rate=1e4).schedule(1, "specified")


def test_capacities_are_per_period():
    """The plant processes the ore of one bench while the mine strips the waste of another, in the
    same period: a top bench of waste and an ore bench below fit in one period."""
    two = {"IX": [0, 0], "IY": [0, 0], "IZ": [0, 1], "VALUE": [100.0, -10.0], "TON": [4.0, 10.0]}
    pits = pymineflow.nested_pits(two, block_size=SIZE, slopes=45, factors=[1.0])
    t = pits.npv(discount_rate=0.1, mining_rate=20, processing_rate=4, timing="end").table
    assert t["npv_worst"][0] == pytest.approx(90 / 1.1)
    assert t["life_worst"][0] == pytest.approx(1.0)  # the plant is full, the mine is at 14 of 20
    # waste mined while the plant is full: 3 benches, ore - waste - ore
    three = {"IX": [0, 0, 0], "IY": [0, 0, 0], "IZ": [0, 1, 2], "VALUE": [50.0, -5.0, 60.0], "TON": [4.0, 6.0, 4.0]}
    pits = pymineflow.nested_pits(three, block_size=SIZE, slopes=45, factors=[1.0])
    s = pits.npv(discount_rate=0.1, mining_rate=100, processing_rate=4).schedule(1, "worst")
    assert s["tonnage"].tolist() == pytest.approx([10.0, 4.0])  # ore 4 + waste 6, then ore 4
    assert s["value"].tolist() == pytest.approx([55.0, 50.0])


def test_hand_examples():
    one = {"IX": [0], "IY": [0], "IZ": [0], "VALUE": [1000.0], "TON": [100.0]}
    pits = pymineflow.nested_pits(one, block_size=SIZE, slopes=45, factors=[1.0])
    end = pits.npv(discount_rate=0.1, mining_rate=50, timing="end").table
    assert end["npv_best"][0] == pytest.approx(500 / 1.1 + 500 / 1.1**2)
    mid = pits.npv(discount_rate=0.1, mining_rate=50, timing="mid").table
    assert mid["npv_worst"][0] == pytest.approx(500 / 1.1**0.5 + 500 / 1.1**1.5)
    start = pits.npv(discount_rate=0.1, mining_rate=50, timing="start").table
    assert start["npv_worst"][0] == pytest.approx(500 + 500 / 1.1)
    part = pits.npv(discount_rate=0.1, mining_rate=40, fixed_cost=10, timing="end").table  # 2.5 periods
    expected = 400 / 1.1 + 400 / 1.1**2 + 200 / 1.1**3 - (10 / 1.1 + 10 / 1.1**2 + 5 / 1.1**3)
    assert part["npv_worst"][0] == pytest.approx(expected)
    assert part["life_worst"][0] == pytest.approx(2.5)
    # two benches: the top one (waste) first, then the ore below it
    two = {"IX": [0, 0], "IY": [0, 0], "IZ": [0, 1], "VALUE": [1000.0, -100.0], "TON": [100.0, 100.0]}
    pits = pymineflow.nested_pits(two, block_size=SIZE, slopes=45, factors=[1.0])
    t = pits.npv(discount_rate=0.1, mining_rate=100, timing="end").table
    assert t["npv_worst"][0] == pytest.approx(-100 / 1.1 + 1000 / 1.1**2)
    t = pits.npv(discount_rate=0.1, processing_rate=100, timing="end").table  # without a mining capacity, waste is free
    assert t["npv_worst"][0] == pytest.approx(-100 / 1.1 + 1000 / 1.1)
    # the same with depth indices (z_up=False): IZ 0 is now the top block
    down = {"IX": [0, 0], "IY": [0, 0], "IZ": [1, 0], "VALUE": [1000.0, -100.0], "TON": [100.0, 100.0]}
    pits = pymineflow.nested_pits(down, block_size=SIZE, slopes=45, factors=[1.0], z_up=False)
    t = pits.npv(discount_rate=0.1, mining_rate=100, timing="end").table
    assert t["npv_worst"][0] == pytest.approx(-100 / 1.1 + 1000 / 1.1**2)


def test_tonnage_options_report_and_chart(tmp_path):
    table = random_model(6)
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=np.linspace(0.4, 1.0, 5))
    by_column = pits.npv(discount_rate=0.1, mining_rate=8000, tonnage="TON").table
    automatic = pits.npv(discount_rate=0.1, mining_rate=8000).table
    assert np.array_equal(by_column["npv_best"], automatic["npv_best"])
    constant = pits.npv(discount_rate=0.1, mining_rate=8000, tonnage=1000.0).table
    assert np.allclose(constant["tonnage"], pits.table_by_pit()["blocks"] * 1000.0)
    r = pits.npv(discount_rate=0.1, mining_rate=8000, processing_rate=3000, pushbacks=[2])
    text = r.summary()
    assert "Pit by pit: discount rate 10.00 % per period" in text
    assert "Specified pushbacks: pits 2 and the final pit" in text
    for case in ("best", "worst", "specified"):
        k = r.best_pit(case)
        assert k == int(np.argmax(r.table[f"npv_{case}"])) + 1
        assert f"Highest NPV, {case} case: pit {k} " in text
    back = read_blocks(r.to_csv(tmp_path / "npv.csv"))
    assert np.allclose(back["npv_worst"], r.table["npv_worst"])
    with pytest.raises(ValueError, match="case must be one of"):
        pits.npv(discount_rate=0.1, mining_rate=8000).best_pit("specified")
    pytest.importorskip("matplotlib")
    fig = r.plot(tmp_path / "npv.png", factor_label="revenue factor")
    assert (tmp_path / "npv.png").stat().st_size > 10_000
    assert len(fig.axes) == 2  # NPV and tonnage
    assert r.plot(tmp_path / "plain.svg", tonnage=False).axes[0].get_ylabel().startswith("discounted NPV")


def test_errors():
    table = random_model(8)
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=[0.5, 0.8, 1.0])
    ok = {"discount_rate": 0.1, "mining_rate": 8000.0}
    bad = [
        ({"discount_rate": -0.1}, "discount_rate must be finite and >= 0"),
        ({"discount_rate": float("nan")}, "discount_rate must be finite"),
        ({"mining_rate": 0.0}, "mining_rate must be finite and > 0"),
        ({"mining_rate": None}, "give mining_rate, processing_rate or both"),
        ({"processing_rate": -5}, "processing_rate must be finite and > 0"),
        ({"processing_rate": {"mill": 10}}, "processing_rate keys must be among the processes"),
        ({"processing_rate": {}}, "processing_rate keys must be among the processes"),
        ({"fixed_cost": -1}, "fixed_cost must be finite and >= 0"),
        ({"timing": "begin"}, "timing must be 'end', 'mid', 'start' or 'continuous'"),
        ({"pushbacks": [2, 1]}, "pushbacks must be increasing"),
        ({"pushbacks": [2, 2]}, "pushbacks must be increasing"),
        ({"pushbacks": [0]}, "pushbacks must be pit numbers from 1 to 3"),
        ({"pushbacks": [4]}, "pushbacks must be pit numbers from 1 to 3"),
        ({"pushbacks": [1.5]}, "pushbacks must be pit numbers"),
        ({"tonnage": "TONNES"}, "tonnage column 'TONNES' not found"),
        ({"tonnage": -1.0}, "tonnage must be finite and >= 0"),
        ({"tonnage": [1.0, 2.0]}, "one value per block"),
        ({"ore": "NOPE"}, "ore column 'NOPE' not found"),
        ({"ore": [1, 0]}, "ore must be a column or an array of 0/1"),
        ({"ore": np.full(table["TON"].size, np.nan)}, "ore has missing values"),
    ]
    for change, message in bad:
        with pytest.raises((ValueError, TypeError), match=message):
            pits.npv(**{**ok, **change})
    with pytest.raises(TypeError, match="discount_rate must be a number"):
        pits.npv(discount_rate="ten", mining_rate=1.0)
    without = {k: v for k, v in table.items() if k != "TON"}
    pits = pymineflow.nested_pits(without, block_size=SIZE, slopes=45, factors=[1.0])
    with pytest.raises(ValueError, match="needs the tonnage of the blocks"):
        pits.npv(discount_rate=0.1, mining_rate=1.0)
    assert pits.npv(discount_rate=0.1, mining_rate=1.0, tonnage=10.0).table["tonnage"][0] >= 0


def test_command_line(tmp_path, capsys):
    path = tmp_path / "model.csv"
    pymineflow.write_table(random_model(9), path)
    args = [str(path), "--block-size", "10", "10", "10", "--slope", "45", "--factors", "0.4:1.0:0.2", "--no-output"]
    report = tmp_path / "npv.csv"
    rates = ["--discount-rate", "0.1", "--mining-rate", "8000", "--processing-rate", "3000", "--pushbacks", "2"]
    assert main([*args, *rates, "--fixed-cost", "100", "--mid-period", "--npv-report", str(report)]) == 0
    out = capsys.readouterr().out
    assert "Pit by pit: discount rate 10.00 % per period, cash at the mid of each period" in out
    pits = pymineflow.nested_pits(str(path), block_size=SIZE, slopes=45, factors=[0.4, 0.6, 0.8, 1.0])
    same = pits.npv(discount_rate=0.1, mining_rate=8000, processing_rate=3000, pushbacks=[2], fixed_cost=100,
                    timing="mid")  # fmt: skip
    assert np.allclose(read_blocks(report)["npv_specified"], same.table["npv_specified"])
    for extra, message in [
        (["--mining-rate", "8000"], "need --discount-rate"),
        (["--discount-rate", "0.1"], "needs --benches-per-year, --ore-per-year, --mining-rate or --processing-rate"),
    ]:
        assert main([*args, *extra]) == 1
        assert message in capsys.readouterr().err
    assert main([*args[:-3], "--no-output", "--discount-rate", "0.1", "--mining-rate", "1"]) == 1
    assert "add --factors" in capsys.readouterr().err
    # MineLib style: cash at the start of each period, processed blocks from a column, the chart
    table = random_model(9)
    table["DESTINATION"] = (table["VALUE"] > -200).astype(int)
    pymineflow.write_table(table, path)
    chart = tmp_path / "npv.png"
    minelib = ["--discount-rate", "0.15", "--processing-rate", "3000", "--timing", "start", "--ore", "DESTINATION"]
    assert main([*args, *minelib, "--npv-chart", str(chart), "--npv-report", str(report)]) == 0
    assert "cash at the start of each period" in capsys.readouterr().out
    pits = pymineflow.nested_pits(str(path), block_size=SIZE, slopes=45, factors=[0.4, 0.6, 0.8, 1.0])
    same = pits.npv(discount_rate=0.15, processing_rate=3000, timing="start", ore="DESTINATION")
    assert np.allclose(read_blocks(report)["npv_best"], same.table["npv_best"])
    if chart.exists():  # matplotlib present
        assert chart.stat().st_size > 10_000
    assert main([*args, "--discount-rate", "0.1", "--mining-rate", "1", "--timing", "end", "--mid-period"]) == 1
    assert "--mid-period and --timing disagree" in capsys.readouterr().err
