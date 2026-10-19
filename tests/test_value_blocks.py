"""value_blocks: valuing a block model in one call (g/t, ppm, %, tonnage or density), checked by hand."""

import json

import numpy as np
import pytest

import pymineflow
from pymineflow import Economics, Metal, Process, value_blocks

OZ = 31.1034768
LB = 453.59237


def model():
    r = np.random.default_rng(2)
    gx, gy, gz = (a.ravel() for a in np.meshgrid(np.arange(8), np.arange(7), np.arange(5), indexing="ij"))
    n = gx.size
    au = np.where(r.random(n) < 0.2, r.exponential(2.0, n), r.exponential(0.05, n))
    return {"IX": gx, "IY": gy, "IZ": gz, "AU": au, "CU": r.exponential(0.2, n), "AG": r.exponential(5, n),
            "TON": np.full(n, 1500.0), "DENSITY": np.full(n, 2.5)}  # fmt: skip


def test_grams_per_tonne_by_hand():
    t = model()
    out = value_blocks(t, metals={"AU": {"price": 1800, "recovery": 0.9}}, processing_cost=20, mining_cost=2)
    process = t["TON"] * (t["AU"] / OZ * 0.9 * 1800 - 20)
    assert np.allclose(out["VALUE"], np.maximum(process, 0) - t["TON"] * 2)
    assert set(out["DESTINATION"]) <= {"process", "waste"}
    assert (out["DESTINATION"] == "process").tolist() == (process > 0).tolist()
    assert all(np.array_equal(out[c], t[c]) for c in t)  # the source columns are kept


def test_ppm_equals_grams_per_tonne_and_percent_in_pounds():
    t = model()
    a = value_blocks(t, metals={"AG": {"price": 25, "grade_unit": "g/t"}}, processing_cost=5, mining_cost=1)
    b = value_blocks(t, metals={"AG": {"price": 25, "grade_unit": "ppm"}}, processing_cost=5, mining_cost=1)
    assert np.array_equal(a["VALUE"], b["VALUE"])
    c = value_blocks(
        t, metals={"CU": {"price": 4.0, "grade_unit": "%", "recovery": 0.85}}, processing_cost=9, mining_cost=1
    )
    process = t["TON"] * (t["CU"] * 1e4 / LB * 0.85 * 4.0 - 9)  # % -> pounds by default
    assert np.allclose(c["VALUE"], np.maximum(process, 0) - t["TON"])
    d = value_blocks(t, metals={"CU": {"price": 9000, "grade_unit": "%", "price_unit": "t", "recovery": 0.85}},
                     processing_cost=9, mining_cost=1)  # fmt: skip
    assert np.allclose(d["VALUE"], np.maximum(t["TON"] * (t["CU"] * 0.01 * 0.85 * 9000 - 9), 0) - t["TON"])


def test_several_metals_payable_selling_and_contaminants():
    t = model()
    t["AS"] = np.linspace(0, 4000, t["AU"].size)
    out = value_blocks(
        t,
        metals={
            "AU": {"price": 1800, "recovery": 0.9, "payable": 0.97, "selling_cost": 8},
            "CU": {"price": 4.0, "grade_unit": "%", "recovery": 0.8},
        },
        processing_cost=15,
        mining_cost=2,
        contaminants={"AS": {"penalty": 0.003, "threshold": 1000, "limit": 3000}},
    )
    revenue = t["AU"] / OZ * 0.9 * 0.97 * (1800 - 8) + t["CU"] * 1e4 / LB * 0.8 * 4.0
    penalty = 0.003 * np.maximum(t["AS"] - 1000, 0)
    process = np.where(t["AS"] <= 3000, t["TON"] * (revenue - penalty - 15), -np.inf)
    assert np.allclose(out["VALUE"], np.maximum(process, 0) - t["TON"] * 2)
    assert (out["DESTINATION"][t["AS"] > 3000] == "waste").all()


def test_density_instead_of_tonnage():
    t = model()
    del t["TON"]
    out = value_blocks(t, metals={"AU": {"price": 1800}}, processing_cost=20, mining_cost=2, density="DENSITY",
                       block_size=(10, 10, 5))  # fmt: skip
    assert np.allclose(out["TONNAGE"], 2.5 * 500)
    same = value_blocks(t, metals={"AU": {"price": 1800}}, processing_cost=20, mining_cost=2, density=2.5,
                        block_size=(10, 10, 5))  # fmt: skip
    assert np.array_equal(out["VALUE"], same["VALUE"])


def test_full_economics_dict_and_json(tmp_path):
    t = model()
    eco = {
        "metals": [{"grade": "AU", "price": 1800, "grade_unit": "g/t", "price_unit": "oz"}],
        "processes": [
            {"name": "mill", "cost": 20, "recovery": {"AU": 0.9}},
            {"name": "pad", "cost": 5, "recovery": {"AU": 0.5}},
        ],
        "mining_cost": 2,
        "tonnage": "TON",
    }
    (tmp_path / "eco.json").write_text(json.dumps(eco))
    a = value_blocks(t, economics=eco)
    b = value_blocks(t, economics=tmp_path / "eco.json")
    c = value_blocks(t, economics=pymineflow.economics_from_dict(eco), price_factor=1.0)
    assert np.array_equal(a["VALUE"], b["VALUE"])
    assert np.array_equal(a["VALUE"], c["VALUE"])
    assert set(a["DESTINATION"]) <= {"mill", "pad", "waste"}


def test_values_go_straight_into_solve_and_nested_pits():
    t = model()
    valued = value_blocks(t, metals={"AU": {"price": 1800, "recovery": 0.9}}, processing_cost=20, mining_cost=2)
    pit = pymineflow.solve(valued, block_size=(10, 10, 5), slopes=45)  # finds the VALUE column
    eco = Economics([Metal("AU", 1800)], [Process("process", 20, {"AU": 0.9})], mining_cost=2)
    same = pymineflow.solve(t, block_size=(10, 10, 5), slopes=45, economics=eco)
    assert pit.value == same.value
    assert np.array_equal(pit.mask, same.mask)
    pits = pymineflow.nested_pits(valued, block_size=(10, 10, 5), slopes=45, factors=[0.5, 1.0])  # profit factor
    assert pits[1].value == pit.value


def test_an_existing_profit_column_is_used_as_it_is():
    t = model()
    t["PROFIT"] = (t["AU"] - 0.5) * 1000
    pit = pymineflow.solve(t, block_size=(10, 10, 5), slopes=45, value="PROFIT")
    assert pit.real_value == pytest.approx(t["PROFIT"][pit.mask].sum())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "needs metals="),
        ({"metals": {"AU": {"recovery": 0.9}}, "processing_cost": 1, "mining_cost": 1}, "a price is needed"),
        ({"metals": {"AU": {"price": 1, "grade": "x"}}, "processing_cost": 1, "mining_cost": 1}, "unknown keys"),
        ({"metals": {"AU": {"price": 1}}, "processing_cost": 1, "mining_cost": 1, "economics": {}}, "not both"),
        ({"metals": {"AU": {"price": 1}}, "processing_cost": 1, "mining_cost": 1, "density": 2.5}, "block_size"),
        ({"metals": {"AU": {"price": 1}}, "processing_cost": 1, "mining_cost": 1, "column": "TON"}, "already has"),
        ({"metals": {"ZN": {"price": 1}}, "processing_cost": 1, "mining_cost": 1}, r"columns \['ZN'\] not found"),
    ],
)
def test_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        value_blocks(model(), **kwargs)


def test_no_tonnage_column():
    t = model()
    del t["TON"]
    with pytest.raises(ValueError, match="no tonnage column found"):
        value_blocks(t, metals={"AU": {"price": 1}}, processing_cost=1, mining_cost=1)
