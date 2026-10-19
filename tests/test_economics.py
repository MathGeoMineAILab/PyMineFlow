"""Economics: block values from grades, prices and costs, checked by hand; and pits from them."""

import json

import numpy as np
import pytest

import pymineflow
from pymineflow import Contaminant, Economics, Metal, Process, economics_from_dict, read_economics
from pymineflow.cli import main
from pymineflow.economics import _contained_per_grade

OZ = 31.1034768  # grams per troy ounce


@pytest.mark.parametrize(
    ("grade_unit", "price_unit", "tonnage_unit", "expected"),
    [
        ("oz/t", "oz", "st", 1.0),  # ounces per short ton, tonnage in short tons (McLaughlin)
        ("oz/t", "oz", "t", 1.0),
        ("g/t", "oz", "t", 1 / OZ),
        ("g/t", "g", "t", 1.0),
        ("ppm", "g", "t", 1.0),
        ("ppb", "g", "t", 1e-3),
        ("%", "t", "t", 0.01),
        ("%", "kg", "t", 10.0),
        ("%", "lb", "t", 1e4 / 453.59237),  # 22.046 lb of metal per tonne and per percent
        ("%", "lb", "st", 20.0),  # 2000 lb per short ton
        ("g/t", "g", "st", 0.90718474),
    ],
)
def test_units(grade_unit, price_unit, tonnage_unit, expected):
    assert _contained_per_grade(grade_unit, price_unit, tonnage_unit) == pytest.approx(expected, rel=1e-12)


def gold(**kwargs):
    metal = {"price": 1800.0, "grade_unit": "g/t", "price_unit": "oz"}
    metal.update(kwargs)
    return Metal("AU", **metal)


def test_one_metal_by_hand():
    eco = Economics([gold(payable=0.95, selling_cost=10)], [Process("mill", 20, {"AU": 0.9})], mining_cost=2)
    table = {"AU": np.array([0.0, 0.5, 2.0]), "TON": np.array([100.0, 100.0, 50.0])}
    values, dest = eco.block_values(table)
    oz = table["AU"] * table["TON"] / OZ
    process = oz * 0.9 * 0.95 * (1800 - 10) - table["TON"] * 20
    expected = np.maximum(process, 0) - table["TON"] * 2
    assert np.allclose(values, expected)
    assert dest.tolist() == [-1, (0 if process[1] > 0 else -1), 0]


def test_two_metals_two_processes_by_hand():
    metals = [gold(), Metal("CU", price=4.0, grade_unit="%", price_unit="lb", payable=0.965, selling_cost=0.3)]
    processes = [Process("flotation", 12.0, {"AU": 0.7, "CU": 0.88}), Process("leach", 4.0, {"AU": 0.6})]
    eco = Economics(metals, processes, mining_cost=1.5)
    table = {"AU": np.array([1.0, 0.2, 0.0]), "CU": np.array([0.1, 0.6, 0.0]), "TON": np.array([10.0, 10.0, 10.0])}
    values, dest = eco.block_values(table)
    au = table["AU"] / OZ  # oz per tonne
    cu = table["CU"] * 1e4 / 453.59237  # lb per tonne
    flot = 10 * (au * 0.7 * 1800 + cu * 0.88 * 0.965 * (4.0 - 0.3) - 12.0)
    leach = 10 * (au * 0.6 * 1800 - 4.0)
    best = np.maximum(np.maximum(flot, leach), 0)
    assert np.allclose(values, best - 15)
    expected = [0 if f >= max(lc, 0) and f > 0 else (1 if lc > 0 else -1) for f, lc in zip(flot, leach, strict=True)]
    assert dest.tolist() == expected
    assert eco.destination_names() == ["flotation", "leach", "waste"]


def test_contaminant_penalty_and_limits():
    arsenic = Contaminant("AS", penalty=0.01, threshold=500, limit=3000)
    processes = [Process("mill", 20, {"AU": 0.9}, limits={"AS": 1500}), Process("pad", 5, {"AU": 0.5})]
    eco = Economics([gold()], processes, mining_cost=2, contaminants=[arsenic])
    table = {"AU": np.full(4, 3.0), "AS": np.array([100.0, 1200.0, 2000.0, 3500.0]), "TON": np.full(4, 10.0)}
    values, dest = eco.block_values(table)
    oz = 3.0 / OZ
    penalty = 0.01 * np.maximum(table["AS"] - 500, 0)
    mill = 10 * (oz * 0.9 * 1800 - penalty - 20)
    pad = 10 * (oz * 0.5 * 1800 - penalty - 5)
    assert dest.tolist() == [0, 0, 1, -1]  # 2000 > the mill's limit; 3500 > every limit: waste
    assert np.allclose(values[:2], mill[:2] - 20)
    assert values[2] == pytest.approx(pad[2] - 20)
    assert values[3] == pytest.approx(-20)


def test_factors():
    eco = Economics([gold(selling_cost=100)], [Process("mill", 20, {"AU": 0.9})], mining_cost=2)
    table = {"AU": np.array([1.0]), "TON": np.array([10.0])}
    oz = 10 / OZ
    price = eco.block_values(table, price_factor=1.5)[0][0]
    revenue = eco.block_values(table, revenue_factor=1.5)[0][0]
    assert price == pytest.approx(oz * 0.9 * (1800 * 1.5 - 100) - 200 - 20)
    assert revenue == pytest.approx(1.5 * oz * 0.9 * (1800 - 100) - 200 - 20)


@pytest.mark.parametrize("seed", range(5))
def test_values_never_decrease_with_the_factors(seed):
    r = np.random.default_rng(seed)
    n = 2000
    table = {
        "AU": r.exponential(0.6, n),
        "CU": r.exponential(0.3, n),
        "AS": r.uniform(0, 3000, n),
        "TON": r.uniform(5, 50, n),
    }
    eco = Economics(
        [gold(payable=0.98, selling_cost=12), Metal("CU", price=4, grade_unit="%", price_unit="lb", selling_cost=0.4)],
        [Process("mill", 15, {"AU": 0.85, "CU": 0.8}, limits={"AS": 2000}), Process("leach", 4, {"AU": 0.55})],
        mining_cost=1.8,
        contaminants=[Contaminant("AS", penalty=0.004, threshold=400, limit=2800)],
    )
    for keyword in ("price_factor", "revenue_factor"):
        previous = eco.block_values(table, **{keyword: 0.0})[0]
        for f in np.linspace(0.05, 3, 25):
            current = eco.block_values(table, **{keyword: f})[0]
            assert (current >= previous - 1e-9).all()
            previous = current


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: Metal("AU", 1800, grade_unit="g"), "grade_unit"),
        (lambda: Metal("AU", 1800, price_unit="ton"), "price_unit"),
        (lambda: Metal("AU", -1), "price"),
        (lambda: Metal("AU", 1800, payable=1.2), "payable"),
        (lambda: Process("mill", 10, {"AU": 1.5}), "recovery"),
        (lambda: Economics([gold()], [], 1), "at least one process"),
        (lambda: Economics([gold()], [Process("mill", 10, {"CU": 0.9})], 1), "unknown metals"),
        (lambda: Economics([gold(), gold()], [Process("mill", 10, {"AU": 0.9})], 1), "repeated metal"),
        (lambda: Economics([gold()], [Process("mill", 10, {"AU": 0.9})], 1, tonnage_unit="kt"), "tonnage_unit"),
        (lambda: Economics([gold()], [Process("m", 1, {"AU": 1}, limits={"AS": 1})], 1), "limits on unknown"),
        (lambda: economics_from_dict({"mining_cost": 1, "price": 3}), "unknown economics keys"),
    ],
)
def test_validation(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_bad_columns():
    eco = Economics([gold()], [Process("mill", 20, {"AU": 0.9})], mining_cost=2)
    with pytest.raises(ValueError, match="no tonnage column"):
        eco.block_values({"AU": [1.0]})
    with pytest.raises(ValueError, match="no missing values"):
        eco.block_values({"AU": [np.nan], "TON": [1.0]})
    with pytest.raises(ValueError, match="tonnage"):
        eco.block_values({"AU": [1.0], "TON": [-1.0]})


ECO = {
    "metals": [{"grade": "AU", "price": 1800, "grade_unit": "g/t", "price_unit": "oz", "selling_cost": 15}],
    "processes": [
        {"name": "mill", "cost": 18, "recovery": {"AU": 0.9}},
        {"name": "pad", "cost": 4, "recovery": {"AU": 0.55}},
    ],
    "mining_cost": 2.0,
    "tonnage": "TON",
    "contaminants": [{"grade": "AS", "penalty": 0.002, "threshold": 300, "limit": 2500}],
}


def model_table(seed=5, nx=10, ny=9, nz=6):
    r = np.random.default_rng(seed)
    gx, gy, gz = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"))
    au = np.where(r.random(gx.size) < 0.15, r.exponential(2.5, gx.size), r.exponential(0.1, gx.size))
    return {"IX": gx, "IY": gy, "IZ": gz, "AU": au, "AS": r.uniform(0, 3000, gx.size), "TON": np.full(gx.size, 2000.0)}


def test_json_and_solve_with_economics(tmp_path):
    (tmp_path / "eco.json").write_text(json.dumps(ECO))
    eco = read_economics(tmp_path / "eco.json")
    assert eco == economics_from_dict(ECO)
    table = model_table()
    real, dest = eco.block_values(table)
    values, _factor = pymineflow.scale_values(real)
    ref = pymineflow.ultimate_pit(
        values, ix=table["IX"], iy=table["IY"], iz=table["IZ"], block_size=(10, 10, 10), slopes=45
    )
    pit = pymineflow.solve(table, block_size=(10, 10, 10), slopes=45, economics=eco)
    assert np.array_equal(pit.mask, ref.mask)
    assert pit.real_value == pytest.approx(real[pit.mask].sum())
    assert np.array_equal(pit.destination, dest)
    text = pit.summary()
    assert "ore (processed)" in text
    assert "    mill" in text
    assert "    pad" in text
    with pytest.raises(ValueError, match="not both"):
        pymineflow.solve(table, block_size=(10, 10, 10), slopes=45, economics=eco, value="AU")


@pytest.mark.parametrize("scale", ["price", "revenue"])
def test_nested_pits_with_economics_against_each_factor_alone(scale):
    eco = economics_from_dict(ECO)
    table = model_table(seed=9)
    factors = [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
    pits = pymineflow.nested_pits(
        table, block_size=(10, 10, 10), slopes=45, economics=eco, factors=factors, scale=scale
    )
    for f, pit in zip(factors, pits, strict=True):
        keyword = "price_factor" if scale == "price" else "revenue_factor"
        real, dest = eco.block_values(table, **{keyword: f})
        values = np.rint(real * pits.factor).astype(np.int64)
        ref = pymineflow.ultimate_pit(
            values, ix=table["IX"], iy=table["IY"], iz=table["IZ"], block_size=(10, 10, 10), slopes=45
        )
        assert pit.value == ref.value
        assert np.array_equal(pit.mask, ref.mask)
        assert np.array_equal(pit.destination, dest)
    t = pits.table_by_pit()
    assert (np.diff(t["blocks"]) >= 0).all()
    assert t["ore_tonnage"][-1] == pytest.approx(table["TON"][pits[-1].mask & (pits[-1].destination >= 0)].sum())


def test_price_scale_needs_economics():
    with pytest.raises(ValueError, match="needs economics"):
        pymineflow.nested_pits(
            model_table(), block_size=(10, 10, 10), slopes=45, factors=[1.0], scale="price", value="AU"
        )


def test_command_line_with_economics(tmp_path, capsys):
    pd = pytest.importorskip("pandas")
    pd.DataFrame(model_table()).to_csv(tmp_path / "model.csv", index=False)
    (tmp_path / "eco.json").write_text(json.dumps(ECO))
    base = [
        str(tmp_path / "model.csv"),
        "--block-size",
        "10",
        "10",
        "10",
        "--slope",
        "45",
        "--economics",
        str(tmp_path / "eco.json"),
    ]
    assert main(base) == 0
    assert "ore (processed)" in capsys.readouterr().out
    assert main([*base, "--scale", "price", "--factors", "0.5:1.5:0.25", "--no-output"]) == 0
    assert "Nested pits: 5 price factors" in capsys.readouterr().out
