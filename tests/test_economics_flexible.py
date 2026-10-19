"""Economic parameters as constants, columns, arrays or dicts by domain: the same values every way."""

import json

import numpy as np
import pytest

import pymineflow
from pymineflow import Economics, Metal, Process, economics_from_dict, nested_pits, solve


def table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    ix, iy, iz = (a.ravel() for a in np.meshgrid(np.arange(10), np.arange(5), np.arange(4), indexing="ij"))
    t = {
        "IX": ix,
        "IY": iy,
        "IZ": iz,
        "AU": np.maximum(rng.normal(1.0, 1.0, ix.size), 0),
        "CU": np.maximum(rng.normal(0.4, 0.3, ix.size), 0),
        "TON": np.full(ix.size, 2500.0),
        "ROCK": np.where(iz > 1, "oxide", "sulphide"),
    }
    t["MCOST"] = np.where(t["ROCK"] == "oxide", 1.5, 2.5)
    return t


def reference(**changes):
    metals = [Metal("AU", price=60, grade_unit="g/t", price_unit="g", selling_cost=2),
              Metal("CU", price=9000, grade_unit="%", price_unit="t", selling_cost=500)]  # fmt: skip
    processes = [Process("mill", cost=25, recovery={"AU": 0.9, "CU": 0.85}),
                 Process("leach", cost=6, recovery={"AU": 0.6})]  # fmt: skip
    kwargs = {"metals": metals, "processes": processes, "mining_cost": 2.0}  # the tonnage: the TON column
    kwargs.update(changes)
    return Economics(**kwargs)


def test_constants_columns_and_arrays_give_the_same_values():
    t = table()
    base = reference().block_values(t)
    n = t["AU"].size
    t["PRICE_AU"] = np.full(n, 60.0)
    t["REC_AU"] = np.full(n, 0.9)
    variants = [
        reference(mining_cost=np.full(n, 2.0)),
        reference(mining_cost="MC"),
        reference(tonnage=2500.0),
        reference(tonnage=np.full(n, 2500.0)),
        reference(metals=[Metal("AU", price="PRICE_AU", grade_unit="g/t", price_unit="g", selling_cost=2),
                          Metal("CU", price=9000, grade_unit="%", price_unit="t", selling_cost=np.full(n, 500.0))]),
        reference(processes=[Process("mill", cost=np.full(n, 25.0), recovery={"AU": "REC_AU", "CU": 0.85}),
                             Process("leach", cost="LEACH", recovery={"AU": np.full(n, 0.6)})]),
    ]  # fmt: skip
    t["MC"] = np.full(n, 2.0)
    t["LEACH"] = np.full(n, 6.0)
    for eco in variants:
        values, destination = eco.block_values(t)
        assert np.allclose(values, base[0], rtol=0, atol=1e-6)
        assert np.array_equal(destination, base[1])


def test_dict_by_domain_equals_a_column():
    t = table()
    by_domain = reference(mining_cost={"oxide": 1.5, "sulphide": 2.5}, domain="ROCK").block_values(t)
    by_column = reference(mining_cost="MCOST").block_values(t)
    assert np.allclose(by_domain[0], by_column[0])
    by_array = reference(mining_cost={"oxide": 1.5, "sulphide": 2.5}, domain=t["ROCK"]).block_values(t)
    assert np.allclose(by_array[0], by_column[0])
    rec = {"AU": {"oxide": 0.9, "sulphide": 0.7}, "CU": 0.85}
    eco = reference(processes=[Process("mill", cost=25, recovery=rec)], domain="ROCK")
    values, _ = eco.block_values(t)
    oxide = t["ROCK"] == "oxide"
    for rock, r in (("oxide", 0.9), ("sulphide", 0.7)):
        one = reference(processes=[Process("mill", cost=25, recovery={"AU": r, "CU": 0.85})]).block_values(t)[0]
        keep = oxide if rock == "oxide" else ~oxide
        assert np.allclose(values[keep], one[keep])


def test_density_and_block_size_give_the_tonnage():
    t = table()
    t["DENSITY"] = np.full(t["AU"].size, 2.5)
    a = reference().block_values(t)[0]
    b = reference(density="DENSITY", block_size=10).block_values(t)[0]  # 2.5 * 1000 = 2500
    assert np.allclose(a, b)
    with pytest.raises(ValueError, match="not both"):  # a tonnage and a density: which one? never a guess
        reference(tonnage="TON", density="DENSITY", block_size=10)
    c = reference(density=2.5, block_size=(10, 10, 10)).block_values(t)[0]
    assert np.allclose(a, c)
    assert np.allclose(reference(density=2.5, block_size=10).tonnages(t), 2500.0)


def test_the_tonnage_column_is_found_by_name():
    t = table()
    t["TONNAGE"] = t.pop("TON")
    values = reference().block_values(t)[0]  # no tonnage given: TON, or the one tonnage column (TONNAGE)
    t2 = table()
    assert np.allclose(values, reference().block_values(t2)[0])
    with pytest.raises(ValueError, match="'TON' not found"):  # a column given is that column
        reference(tonnage="TON").block_values(t)
    t["TONNES"] = t["TONNAGE"]
    with pytest.raises(ValueError, match="several tonnage columns"):
        reference().block_values(t)


def test_recovered_grades_are_not_recovered_twice():
    t = table()
    t["AU_REC"] = t["AU"] * 0.9
    t["CU_REC"] = t["CU"] * 0.85
    metals = [Metal("AU_REC", price=60, grade_unit="g/t", price_unit="g", selling_cost=2, recovered=True),
              Metal("CU_REC", price=9000, grade_unit="%", price_unit="t", selling_cost=500,
                    recovered=True)]  # fmt: skip
    eco = Economics(metals, [Process("mill", cost=25, recovery={})], mining_cost=2.0)
    plain = reference(processes=[Process("mill", cost=25, recovery={"AU": 0.9, "CU": 0.85})]).block_values(t)[0]
    assert np.allclose(eco.block_values(t)[0], plain)
    with pytest.raises(ValueError, match="not applied twice"):
        Economics(metals, [Process("mill", cost=25, recovery={"AU_REC": 0.9})], mining_cost=2.0)
    Economics(metals, [Process("mill", cost=25, recovery={"AU_REC": 1})], mining_cost=2.0)  # 1 is harmless


def test_rows_and_grades_select_and_replace():
    t = table()
    eco = reference(mining_cost="MCOST")
    full = eco.block_values(t)
    rows = np.array([5, 0, 17, 199])
    sub = eco.block_values(t, rows=rows)
    assert np.allclose(sub[0], full[0][rows])
    assert np.array_equal(sub[1], full[1][rows])
    other = dict(t)
    other["AU"] = t["AU"] * 2
    replaced = eco.block_values(t, grades={"AU": other["AU"][rows]}, rows=rows)
    assert np.allclose(replaced[0], eco.block_values(other)[0][rows])


def test_terms_rebuild_the_values():
    t = table()
    eco = reference()
    for f in (0.5, 1.0, 1.7):
        terms = eco.terms(t, f)
        margins = np.array(terms.margins())
        best = np.where(np.array(terms.allowed), margins, -np.inf).max(axis=0)
        assert np.allclose(np.maximum(best, 0) - terms.mining, eco.block_values(t, price_factor=f)[0])
    # the revenue is linear in the price factor
    r0, r1, r2 = (np.array(eco.terms(t, f).revenue) for f in (0.0, 1.0, 2.0))
    assert np.allclose(r2 - r1, r1 - r0)


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        (lambda: Metal("AU", price=-1), ValueError, "price"),
        (lambda: Metal("AU", price=np.array([1.0, -1.0])), ValueError, "price"),
        (lambda: Metal("AU", price=True), TypeError, "number"),
        (lambda: Metal("AU", price=[[1.0]]), ValueError, "1-D"),
        (lambda: Process("m", cost={"a": "x"}, recovery={}), TypeError, "number"),
        (lambda: Process("m", cost=1, recovery={"AU": 1.5}), ValueError, "recovery"),
        (lambda: reference(density=2.5), ValueError, "go together"),
        (lambda: reference(tonnage="TONS", density=2.5, block_size=10), ValueError, "not both"),
        (lambda: reference(density=2.5, block_size=(1, 2)), ValueError, "block_size"),
    ],
)
def test_bad_parameters(build, error, match):
    with pytest.raises(error, match=match):
        build()


def test_bad_parameters_against_the_table():
    t = table()
    with pytest.raises(ValueError, match="pass Economics"):
        reference(mining_cost={"oxide": 1.0, "sulphide": 2.0}).block_values(t)
    with pytest.raises(ValueError, match="no value for the domains"):
        reference(mining_cost={"oxide": 1.0}, domain="ROCK").block_values(t)
    with pytest.raises(ValueError, match="one value per block"):
        reference(mining_cost=np.ones(3)).block_values(t)
    with pytest.raises(ValueError, match="not found"):
        reference(mining_cost="NOPE").block_values(t)
    t["NEG"] = -np.ones(t["AU"].size)
    with pytest.raises(ValueError, match="finite and in"):
        reference(mining_cost="NEG").block_values(t)


def test_from_dict_and_json(tmp_path):
    data = {"metals": [{"grade": "AU", "price": 60, "grade_unit": "g/t", "price_unit": "g", "recovered": False}],
            "processes": [{"name": "mill", "cost": {"oxide": 20, "sulphide": 30}, "recovery": {"AU": 0.9}}],
            "mining_cost": 2, "domain": "ROCK", "density": 2.5, "block_size": [10, 10, 10]}  # fmt: skip
    eco = economics_from_dict(data)
    path = tmp_path / "eco.json"
    path.write_text(json.dumps(data))
    t = table()
    assert np.allclose(pymineflow.read_economics(path).block_values(t)[0], eco.block_values(t)[0])
    assert eco.columns == ["AU", "ROCK"]


def test_solve_and_nested_pits_with_per_block_parameters():
    t = table()
    n = t["AU"].size
    eco_cols = reference(mining_cost="MCOST")
    eco_arr = reference(mining_cost=np.asarray(t["MCOST"]))
    eco_dom = reference(mining_cost={"oxide": 1.5, "sulphide": 2.5}, domain="ROCK")
    pits = [solve(t, block_size=10, slopes=45, economics=e) for e in (eco_cols, eco_arr, eco_dom)]
    assert all(np.array_equal(p.mask, pits[0].mask) for p in pits)
    factors = [0.6, 0.8, 1.0, 1.2]
    for scale in ("revenue", "price"):
        a = nested_pits(t, block_size=10, slopes=45, economics=eco_cols, factors=factors, scale=scale)
        b = nested_pits(t, block_size=10, slopes=45, economics=eco_arr, factors=factors, scale=scale)
        assert np.array_equal(a.pit_number(), b.pit_number())
    assert n == pits[0].mask.size


def test_value_blocks_simple_way_per_block():
    """The simple way takes columns, arrays and dicts by domain for every number, as Economics."""
    rng = np.random.default_rng(1)
    n = 40
    t = {"IX": np.arange(n) % 5, "IY": np.arange(n) // 5 % 4, "IZ": np.arange(n) // 20, "AU": rng.uniform(0, 3, n),
         "TON": np.full(n, 1000.0), "PRICE": rng.uniform(1700, 1900, n), "REC": rng.uniform(0.8, 0.9, n),
         "PCOST": rng.uniform(15, 20, n), "MCOST": rng.uniform(1.5, 2.5, n),
         "ROCK": rng.choice(["ox", "su"], n)}  # fmt: skip
    contained = t["AU"] * t["TON"] / 31.1034768  # g/t of a tonne, priced per troy ounce
    out = pymineflow.value_blocks(t, metals={"AU": {"price": "PRICE", "recovery": "REC"}}, processing_cost="PCOST",
                                  mining_cost="MCOST")  # fmt: skip
    expected = np.maximum(0, contained * t["REC"] * t["PRICE"] - t["TON"] * t["PCOST"]) - t["TON"] * t["MCOST"]
    assert np.allclose(out["VALUE"], expected)
    out = pymineflow.value_blocks(t, metals={"AU": {"price": 1800, "recovery": t["REC"]}}, processing_cost=t["PCOST"],
                                  mining_cost=np.full(n, 2.0))  # fmt: skip
    assert np.allclose(out["VALUE"], np.maximum(0, contained * t["REC"] * 1800 - t["TON"] * t["PCOST"]) - 2000.0)
    out = pymineflow.value_blocks(t, metals={"AU": {"price": 1800, "recovery": {"ox": 0.9, "su": 0.8}}},
                                  processing_cost={"ox": 12, "su": 20}, mining_cost=2, domain="ROCK")  # fmt: skip
    rec, cost = np.where(t["ROCK"] == "ox", 0.9, 0.8), np.where(t["ROCK"] == "ox", 12, 20)
    assert np.allclose(out["VALUE"], np.maximum(0, contained * rec * 1800 - t["TON"] * cost) - 2000.0)
    with pytest.raises(ValueError, match="domain"):
        pymineflow.value_blocks(t, metals={"AU": {"price": 1800, "recovery": {"ox": 0.9, "su": 0.8}}},
                                processing_cost=18, mining_cost=2)  # fmt: skip
