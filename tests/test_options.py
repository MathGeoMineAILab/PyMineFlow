"""Options for short scripts: one block size, profit factors with economics, first factors, comparisons
and exports of other programs."""

import numpy as np
import pytest

import pymineflow
from pymineflow import Economics, Metal, Process, read_reference_pits, value_blocks, write_table
from pymineflow.cli import main

SIZE = (10.0, 10.0, 10.0)
FACTORS = [0.3, 0.5, 0.7, 1.0]


def model(seed=4):
    rng = np.random.default_rng(seed)
    ix, iy, iz = (a.ravel() for a in np.meshgrid(np.arange(1, 10), np.arange(1, 8), np.arange(1, 7), indexing="ij"))
    grade = np.maximum(rng.normal(0.6 + 0.15 * (6 - iz), 0.6), 0)
    return {"IX": ix, "IY": iy, "IZ": iz, "TON": rng.uniform(900, 1100, ix.size), "AU": grade}


ECO = Economics(
    metals=[Metal("AU", price=60, grade_unit="g/t", price_unit="g", selling_cost=12)],
    processes=[Process("mill", cost=25, recovery={"AU": 0.9})],
    mining_cost=2,
    tonnage="TON",
)


def test_one_block_size():
    t = model()
    a = pymineflow.nested_pits(t, block_size=10, slopes=45, factors=FACTORS, economics=ECO)
    b = pymineflow.nested_pits(t, block_size=SIZE, slopes=45, factors=FACTORS, economics=ECO)
    assert np.array_equal(a.numbers, b.numbers)
    assert pymineflow.solve(t, block_size=10, slopes=45, economics=ECO).value == b[-1].value
    with pytest.raises(ValueError, match="block_size must be three positive numbers"):
        pymineflow.solve(t, block_size=(10, 10), slopes=45, economics=ECO)
    with pytest.raises(ValueError, match="block_size must be three positive numbers"):
        pymineflow.solve(t, block_size=-10, slopes=45, economics=ECO)


@pytest.mark.parametrize("discount", [0.0, 0.02])
def test_profit_factors_with_economics(discount):
    """scale='profit' with economics = the factor-1 values as a value column (destinations fixed)."""
    t = model()
    pits = pymineflow.nested_pits(t, block_size=SIZE, slopes=45, factors=FACTORS, economics=ECO, scale="profit",
                                  bench_discount=discount)  # fmt: skip
    same = pymineflow.nested_pits(value_blocks(t, economics=ECO), block_size=SIZE, slopes=45, factors=FACTORS,
                                  value="VALUE", scale="revenue", bench_discount=discount)  # fmt: skip
    assert np.array_equal(pits.numbers, same.numbers)
    assert pits.values == pytest.approx(same.values)
    fixed = ECO.block_values(t)[1]
    for k in range(len(FACTORS)):
        assert np.array_equal(pits[k].destination, fixed)  # the destination does not change with the factor
    assert "profit factors" in pits.summary()
    npv = pits.npv(discount_rate=0.1, processing_rate=50_000)
    assert npv.processes == ("mill",)
    with pytest.raises(ValueError, match="scale must be 'revenue', 'price' or 'profit'"):
        pymineflow.nested_pits(t, block_size=SIZE, slopes=45, factors=FACTORS, economics=ECO, scale="prices")


def test_first_factor_and_compare():
    t = model()
    pits = pymineflow.nested_pits(t, block_size=SIZE, slopes=45, factors=FACTORS, economics=ECO, scale="profit")
    first = pits.first_factor()
    numbers = pits.pit_number()
    assert np.array_equal(first > 0, numbers > 0)
    assert np.allclose(first[numbers > 0], np.asarray(FACTORS)[numbers[numbers > 0] - 1])
    c = pits.compare(first)  # against itself
    assert c.same_first == c.compared == int((numbers > 0).sum())
    assert c.identical == len(c.table["factor"])
    assert np.allclose(c.table["jaccard"], 1.0)
    assert np.allclose(c.table["tonnage"], c.table["tonnage_reference"])
    by_pit = pits.table_by_pit()
    k = np.searchsorted(FACTORS, c.table["factor"])
    assert np.allclose(c.table["tonnage"], by_pit["tonnage"][k])
    # a reference that puts one block of the first pit in a later pit
    ref = first.copy()
    moved = int(np.flatnonzero(first == first[first > 0].min())[0])
    ref[moved] = 1.0
    c = pits.compare(ref)
    assert c.same_first == c.compared - 1
    assert c.identical < len(c.table["factor"])
    assert c.table["only_pits"][0] == 1
    assert "Block by block:" in c.summary()
    with pytest.raises(ValueError, match="reference must be one finite factor"):
        pits.compare(first[:-1])


def pit_export(table, first, path, base=1):
    """A pit export of another program: only the blocks of some pit, with centres, origin and size."""
    inside = first > 0
    x0, y0, z0 = 1000.0, 2000.0, 300.0
    out = {
        "INDEX": np.arange(int(inside.sum())),
        "XC": x0 + (np.asarray(table["IX"])[inside] - base + 0.5) * 10,
        "YC": y0 + (np.asarray(table["IY"])[inside] - base + 0.5) * 10,
        "ZC": z0 + (np.asarray(table["IZ"])[inside] - base + 0.5) * 10,
        "XINC": 10.0, "YINC": 10.0, "ZINC": 10.0, "XMORIG": x0, "YMORIG": y0, "ZMORIG": z0,
        "DEST": np.where(np.arange(int(inside.sum())) % 2 == 0, "_WASTE_", "Café"),
        "UPT_PIT": np.rint(first[inside] * 100).astype(int),
    }  # fmt: skip
    out = {k: (np.full(int(inside.sum()), v) if np.isscalar(v) else v) for k, v in out.items()}
    write_table(out, path)


def test_read_reference_pits(tmp_path):
    t = model()
    pits = pymineflow.nested_pits(t, block_size=SIZE, slopes=45, factors=FACTORS, economics=ECO, scale="profit")
    first = pits.first_factor()
    path = tmp_path / "reference_pits.csv"
    pit_export(t, first, path)
    raw = path.read_bytes().decode("utf-8").encode("latin-1")  # such exports are often latin-1 (accented labels)
    path.write_bytes(raw)
    ref = read_reference_pits(path, t)
    assert np.allclose(ref, first)
    assert pits.compare(ref).identical == len(np.unique(first[first > 0]))
    with pytest.raises(ValueError, match="check index_base"):
        read_reference_pits(path, t, index_base=5)
    with pytest.raises(ValueError, match="it must list XC, YC, ZC"):
        read_reference_pits(path, t, column="NOPE")
    with pytest.raises(ValueError, match="index column 'I' not found"):
        read_reference_pits(path, t, indices=("I", "J", "K"))


def test_command_line_profit_scale(tmp_path, capsys):
    path = tmp_path / "model.csv"
    write_table(value_blocks(model(), economics=ECO), path)
    args = [str(path), "--block-size", "10", "10", "10", "--slope", "45", "--value", "VALUE", "--no-output"]
    assert main([*args, "--factors", "0.5", "1", "--scale", "profit"]) == 0
    assert "profit factors" in capsys.readouterr().out


def test_annual_discount_rate_report_and_export(tmp_path, rendering):
    t = model()
    kw = {"block_size": 10, "slopes": 45, "factors": np.arange(1, 101) / 100, "economics": ECO, "scale": "profit"}
    by_rate = pymineflow.nested_pits(t, discount_rate=0.10, benches_per_year=8, ore_per_year=60_000, **kw)
    by_bench = pymineflow.nested_pits(t, bench_discount=0.10 / 8, **kw)
    assert by_rate.bench_discount == pytest.approx(0.0125)
    assert np.array_equal(by_rate.numbers, by_bench.numbers)
    r = by_rate.report_table()
    phases = np.flatnonzero(np.bincount(by_rate.numbers, minlength=101)[1:] > 0)
    assert r["phase"].tolist() == list(range(1, phases.size + 1))
    assert np.allclose(r["factor"], np.asarray(by_rate.factors)[phases])
    npv = by_rate.npv().table  # defaults: the rate and the ore per year of nested_pits, continuous timing
    assert np.allclose(r["npv_best"], npv["npv_best"][phases])
    assert np.allclose(r["npv_worst"], npv["npv_worst"][phases])
    assert np.allclose(r["rock"], npv["tonnage"][phases])
    assert np.allclose(r["ore"], npv["ore_tonnage"][phases])
    assert np.allclose(r["profit"], npv["value"][phases])
    assert np.allclose(r["waste"], r["rock"] - r["ore"])
    text = by_rate.report()
    assert "Top-down bench discounting: 1.2500% per bench (10 % a year, 8 benches a year)" in text
    assert "NPV best case" in text
    assert "Highest NPV: best case phase" in text
    plain = pymineflow.nested_pits(t, **kw).report()
    assert "NPV" not in plain
    assert "discounting" not in plain
    files = by_rate.export(tmp_path / "out")
    assert [f.name for f in files] == ["pits.csv", "final_pit.csv", "report_cumulative.csv",
                                       "report_incremental.csv", "plan.png", "npv.png", "nested_pits.gif",
                                       "nested_pits_remaining.gif"]  # fmt: skip
    assert all(f.stat().st_size > 0 for f in files)
    assert np.allclose(pymineflow.read_blocks(files[2])["npv_best"], r["npv_best"])
    inc = pymineflow.read_blocks(files[3])
    assert np.allclose(np.cumsum(inc["npv"]), r["npv_best"])
    assert np.allclose(np.cumsum(inc["profit"]), r["profit"])
    assert np.allclose(np.cumsum(inc["rock"]), r["rock"])
    assert "incremental" in by_rate.report(kind="incremental")
    assert [f.name for f in pymineflow.nested_pits(t, **kw).export(tmp_path / "plain")] == [
        "pits.csv",
        "final_pit.csv",
        "report_cumulative.csv",
        "report_incremental.csv",
        "plan.png",
        "nested_pits.gif",
        "nested_pits_remaining.gif",
    ]
    with pytest.raises(ValueError, match="not both"):
        pymineflow.nested_pits(t, bench_discount=0.01, discount_rate=0.1, benches_per_year=8, **kw)
    for bad, message in (({"discount_rate": -0.1}, "discount_rate must be finite and >= 0"),
                         ({"benches_per_year": 0}, "benches_per_year must be finite and > 0"),
                         ({"ore_per_year": -1}, "ore_per_year must be finite and > 0")):  # fmt: skip
        with pytest.raises(ValueError, match=message):
            pymineflow.nested_pits(t, **bad, **kw)


def test_command_line_without_programming(tmp_path, capsys, rendering):
    """One command: nested pits, bench discounting from an annual rate, NPV and every file."""
    path = tmp_path / "model.csv"
    write_table(model(), path)
    economics = tmp_path / "economics.json"
    economics.write_text(
        '{"metals": [{"grade": "AU", "price": 60, "grade_unit": "g/t", "price_unit": "g", "selling_cost": 12}],'
        ' "processes": [{"name": "mill", "cost": 25, "recovery": {"AU": 0.9}}], "mining_cost": 2, "tonnage": "TON"}'
    )
    out = tmp_path / "results"
    args = [str(path), "--block-size", "10", "--slope", "45", "--economics", str(economics), "--scale", "profit",
            "--factors", "0.01:1:0.01", "--discount-rate", "0.10", "--benches-per-year", "8", "--ore-per-year", "60000",
            "--export", str(out), "--no-output"]  # fmt: skip
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "Top-down bench discounting: 1.2500% per bench" in printed
    assert "NPV best case" in printed
    assert "written to" in printed
    assert {f.name for f in out.iterdir()} == {"pits.csv", "final_pit.csv", "report_cumulative.csv",
                                               "report_incremental.csv", "plan.png", "npv.png", "nested_pits.gif",
                                               "nested_pits_remaining.gif"}  # fmt: skip
    for extra, message in ((["--benches-per-year", "8"], "need --discount-rate"),
                           (["--block-size", "10", "10"], "three numbers")):  # fmt: skip
        base = [a for a in args if a not in ("--discount-rate", "0.10")] if "need" in message else args
        assert main([*base, *extra]) == 1
        assert message in capsys.readouterr().err


LEGEND = """
import numpy as np, pymineflow
from pymineflow.plot import _NestedScene
table = {"IX": [0, 1, 1, 2], "IY": [0, 0, 0, 0], "IZ": [0, 0, 1, 0], "VALUE": [9.5, -1.0, -1.0, -1.0]}
pits = pymineflow.nested_pits(table, block_size=10, slopes=45, factors=np.arange(1, 101) / 100)
s = _NestedScene(pits, view="top", rotation=0, title=None, outline=False, axes=False, cmap="jet",
                 window_size=(400, 300), off_screen=True)
print(int((pits.numbers > 0).sum()), s.first_pit, s.last_pit, *s.plotter.mapper.GetScalarRange())
s.plotter.close()
"""


def test_legend_follows_the_pits_that_exist():
    """Low factors give empty pits: the colour scale and the slider start at the first pit with blocks."""
    import subprocess
    import sys

    pytest.importorskip("pyvista")
    done = subprocess.run([sys.executable, "-c", LEGEND], capture_output=True, text=True, timeout=300, check=False)
    if done.returncode != 0:
        pytest.skip(f"no rendering available here: {done.stderr[-300:]}")
    blocks, first, last, low, high = done.stdout.split()
    # the rich block (9.5) pays for its waste block (1.0) from factor 0.11: pit 11 is the first one
    assert (int(blocks), int(first), int(last)) == (2, 11, 12)
    assert (float(low), float(high)) == (11.0, 12.0)


def test_everything_can_be_changed(tmp_path, rendering):
    """For programmers: choose the files, the picture and the NPV convention; the tables are plain arrays."""
    t = model()
    kw = {"discount_rate": 0.10, "benches_per_year": 8, "ore_per_year": 60_000}
    pits = pymineflow.nested_pits(t, block_size=10, slopes=45, factors=np.arange(1, 101) / 100, economics=ECO,
                                  scale="profit", **kw)  # fmt: skip
    only = pits.export(tmp_path / "a", files=["report", "npv"], timing="end", report="cumulative")
    assert [f.name for f in only] == ["report_cumulative.csv", "npv.png"]
    assert [f.name for f in pits.export(tmp_path / "i", files=["report"], report="incremental")] == [
        "report_incremental.csv"
    ]
    with pytest.raises(ValueError, match="report"):
        pits.export(tmp_path / "x", report="all")
    with pytest.raises(ValueError, match="kind"):
        pits.report_table(kind="total")
    end = pits.report_table("end")
    assert np.allclose(pymineflow.read_blocks(only[0])["npv_best"], end["npv_best"])
    mid, start = pits.report_table("mid"), pits.report_table("start")
    assert (start["npv_best"] > mid["npv_best"]).all()
    assert (mid["npv_best"] > end["npv_best"]).all()
    assert "cash at the start of each year" in pits.report("start")
    iso = pits.export(tmp_path / "b", files=["plan"], view="iso", cmap="viridis", title="", window_size=(600, 400))
    assert iso[0].stat().st_size > 0
    with pytest.raises(ValueError, match="files must be among"):
        pits.export(tmp_path / "c", files=["plans"])
    with pytest.raises(ValueError, match="timing must be"):
        pits.report("noon")
    # the same pits, other NPV options, directly
    npv = pits.npv(mining_rate=200_000, fixed_cost=1000, pushbacks=[20, 50])
    assert "npv_specified" in npv.table
    assert npv.discount_rate == pytest.approx(0.10)  # the defaults come from nested_pits


def test_export_the_whole_model_or_only_a_pit(tmp_path, capsys):
    t = model()
    kw = {"discount_rate": 0.10, "benches_per_year": 8, "ore_per_year": 60_000}
    pits = pymineflow.nested_pits(t, block_size=10, slopes=45, factors=np.arange(1, 101) / 100, economics=ECO,
                                  scale="profit", **kw)  # fmt: skip
    whole = pymineflow.read_blocks(pits.to_csv(tmp_path / "whole.csv"))
    assert whole["PIT_NUMBER"].size == t["IX"].size
    assert np.allclose(whole["VALUE"], ECO.block_values(t)[0])
    assert set(whole["DESTINATION"].tolist()) <= {"mill", "waste"}
    final = pymineflow.read_blocks(pits.to_csv(tmp_path / "final.csv", pit="final"))
    assert final["PIT_NUMBER"].size == pits[-1].n_blocks
    assert (final["PIT_NUMBER"] >= 1).all()
    k = pits.npv().best_pit("best")
    best = pymineflow.read_blocks(pits.to_csv(tmp_path / "best.csv", pit="best"))
    assert best["PIT_NUMBER"].size == pits[k - 1].n_blocks
    some = int(np.flatnonzero(np.bincount(pits.numbers, minlength=101)[1:] > 0)[0]) + 1
    first = pymineflow.read_blocks(pits.to_csv(tmp_path / "first.csv", pit=some))
    assert first["PIT_NUMBER"].size == pits[some - 1].n_blocks
    for bad in (0, 101, "largest", True):
        with pytest.raises(ValueError, match="pit must be 'final', 'best' or a pit number"):
            pits.to_csv(tmp_path / "bad.csv", pit=bad)
    plain = pymineflow.nested_pits(t, block_size=10, slopes=45, factors=[0.5, 1.0], economics=ECO)
    with pytest.raises(ValueError, match="pit='best' needs discount_rate and ore_per_year"):
        plain.to_csv(tmp_path / "x.csv", pit="best")
    # one pit from solve: the whole model or only the pit
    pit = pymineflow.solve(value_blocks(t, economics=ECO), block_size=10, slopes=45, value="VALUE")
    assert pymineflow.read_blocks(pit.to_csv(tmp_path / "all.csv"))["PIT"].size == t["IX"].size
    only = pymineflow.read_blocks(pit.to_csv(tmp_path / "only.csv", only_pit=True))
    assert only["PIT"].size == pit.n_blocks
    assert (only["PIT"] == 1).all()
    # the command line
    path = tmp_path / "model.csv"
    write_table(value_blocks(t, economics=ECO), path)
    out = tmp_path / "pit_only.csv"
    args = [str(path), "--block-size", "10", "--slope", "45", "--value", "VALUE", "--only-pit", "--output", str(out)]
    assert main(args) == 0
    assert pymineflow.read_blocks(out)["PIT"].size == pit.n_blocks
    capsys.readouterr()
