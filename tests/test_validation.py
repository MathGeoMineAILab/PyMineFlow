"""Inputs are never read wrong in silence, and every report of one study uses the same NPV."""

import numpy as np
import pytest

import pymineflow
from pymineflow.cli import _factors, main

SIZE = (10.0, 10.0, 10.0)


def small():
    return {"IX": [1, 0, 1, 2], "IY": [0, 0, 0, 0], "IZ": [0, 1, 1, 1], "VALUE": [9, -1, -1, -1]}


def test_integer_values_with_fractional_factors():
    """Integer values times a fractional factor are fractional: they are not rounded to whole units."""
    pits = pymineflow.nested_pits(small(), block_size=SIZE, slopes=45, factors=[0.35, 0.39, 1.0])
    assert pits.values == pytest.approx([0.15, 0.51, 6.0])
    for k, f in enumerate(pits.factors):
        alone = pymineflow.solve({**small(), "VALUE": np.where(np.array(small()["VALUE"]) > 0, 9 * f, -1.0)},
                                 block_size=SIZE, slopes=45)  # fmt: skip
        assert pits.values[k] == pytest.approx(alone.real_value)


def test_large_factors_choose_their_own_scale():
    n = 1000
    k = np.arange(n)
    table = {"IX": k % 10, "IY": k // 10 % 10, "IZ": k // 100, "VALUE": np.full(n, 2.5e9 + 0.5)}
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=[1, 3])
    assert pits.values[-1] == pytest.approx(3 * n * (2.5e9 + 0.5))


def test_numbers_are_numbers():
    with pytest.raises(TypeError, match="factors"):
        pymineflow.nested_pits(small(), block_size=SIZE, slopes=45, factors=[True, 0.5])
    with pytest.raises(TypeError, match="bench_discount"):
        pymineflow.solve(small(), block_size=SIZE, slopes=45, bench_discount=True)
    with pytest.raises(TypeError, match="bench_discount"):
        pymineflow.solve(small(), block_size=SIZE, slopes=45, bench_discount="0.1")
    with pytest.raises(TypeError, match="discount_rate"):
        pymineflow.nested_pits(small(), block_size=SIZE, slopes=45, factors=[1.0], discount_rate=True)
    pits = pymineflow.nested_pits({**small(), "TON": [1.0] * 4}, block_size=SIZE, slopes=45, factors=[0.5, 1.0])
    with pytest.raises(TypeError, match="discount_rate"):
        pits.npv(discount_rate=True, processing_rate=1)
    with pytest.raises(ValueError, match="price_factor"):
        pymineflow.solve(small(), block_size=SIZE, slopes=45, price_factor=0.5)


def test_one_value_column_or_an_error():
    table = {**small(), "PROFIT": [1.0, 2.0, 3.0, 4.0]}
    with pytest.raises(ValueError, match="several value columns"):
        pymineflow.solve(table, block_size=SIZE, slopes=45)
    assert pymineflow.solve(table, block_size=SIZE, slopes=45, value="VALUE").real_value == 6


def test_reports_use_the_npv_options_and_check_the_timing():
    table = {**small(), "TON": [10.0] * 4}
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=[0.5, 1.0], discount_rate=0.1,
                                  ore_per_year=5)  # fmt: skip
    options = {"mining_rate": 15.0, "fixed_cost": 0.5}
    report = pits.report_table("end", **options)
    cases = pits.npv(timing="end", **options).table
    assert report["npv_best"][-1] == pytest.approx(cases["npv_best"][-1])
    assert report["npv_best"][-1] != pytest.approx(pits.report_table("end")["npv_best"][-1])
    assert "15 t mined a year" in pits.report("end", **options)
    with pytest.raises(ValueError, match="kind="):
        pits.report_table("incremental")  # the first argument is the timing
    plain = pymineflow.nested_pits(small(), block_size=SIZE, slopes=45, factors=[0.5, 1.0])
    with pytest.raises(ValueError, match="timing"):
        plain.report_table("incremental")
    assert pits.report_table()["npv_best"][-1] == pytest.approx(pits.npv().table["npv_best"][-1])  # continuous


def test_tonnage_of_the_economics_everywhere():
    n = 8
    k = np.arange(n)
    table = {"IX": k % 4, "IY": k // 4, "IZ": np.zeros(n, int), "AU": np.linspace(0, 5, n), "TON": np.full(n, 100.0),
             "TON_WET": np.full(n, 110.0)}  # fmt: skip
    eco = pymineflow.Economics(metals=[pymineflow.Metal("AU", price=600)],
                               processes=[pymineflow.Process("mill", cost=10, recovery={"AU": 0.9})], mining_cost=1,
                               tonnage="TON_WET")  # fmt: skip
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=[0.5, 1.0], economics=eco)
    assert pits.table_by_pit()["tonnage"][-1] == pytest.approx(pits.report_table()["rock"][-1])
    assert pits.table_by_pit()["tonnage"][-1] == pytest.approx(110.0 * pits[1].n_blocks)
    pit = pymineflow.solve(table, block_size=SIZE, slopes=45, economics=eco)
    assert pit._tonnage()[1][0] == 110.0


def test_bad_tonnage_is_an_error():
    table = {**small(), "TON": [1.0, np.nan, 1.0, 1.0]}
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=[0.5, 1.0])
    with pytest.raises(ValueError, match="tonnage must be finite"):
        pits.report_table()


def test_comma_file_with_spaces_in_text(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("IX,IY,IZ,VALUE,ROCK TYPE\n0,0,0,1.5,oxide ore\n1,0,0,-1,fresh rock\n")
    table = pymineflow.read_blocks(path)
    assert list(table) == ["IX", "IY", "IZ", "VALUE", "ROCK TYPE"]
    spaced = tmp_path / "s.txt"
    spaced.write_text("IX IY IZ VALUE\n0 0 0 1,5\n1 0 0 -2,25\n")
    assert pymineflow.read_blocks(spaced)["VALUE"].tolist() == [1.5, -2.25]


def test_value_blocks_options():
    table = {"IX": [0, 1], "IY": [0, 0], "IZ": [0, 0], "AU": [0.0, 2.0]}
    eco = {"metals": [{"grade": "AU", "price": 60}], "processes": [{"name": "mill", "cost": 10, "recovery": {"AU": 1}}],
           "mining_cost": 1}  # fmt: skip
    with pytest.raises(ValueError, match="not both"):
        pymineflow.value_blocks({**table, "TON": [1.0, 1.0]}, economics=eco, mining_cost=100)
    with pytest.raises(ValueError, match="tonnage unit"):
        pymineflow.value_blocks({**table, "TON": [1.0, 1.0]}, economics=eco, tonnage_unit="st")
    out = pymineflow.value_blocks(table, economics=eco, density=2.5, block_size=2)
    assert out["TONNAGE"].tolist() == [20.0, 20.0]


def test_economics_from_json_text_keys_and_missing_cost(tmp_path):
    table = {"IX": [0, 1], "IY": [0, 0], "IZ": [0, 0], "AU": [2.0, 2.0], "TON": [1.0, 1.0], "ROCK": [1, 2]}
    eco = {"metals": [{"grade": "AU", "price": 600}],
           "processes": [{"name": "mill", "cost": {"1": 10, "2": 30}, "recovery": {"AU": 1}}], "mining_cost": 1,
           "domain": "ROCK"}  # fmt: skip
    out = pymineflow.value_blocks(table, economics=eco)
    assert out["VALUE"][0] > out["VALUE"][1]
    with pytest.raises(ValueError, match="mining_cost"):
        pymineflow.economics_from_dict({"metals": [], "processes": []})
    path = tmp_path / "eco.json"
    path.write_text('{"metals": [], "processes": []}')
    model = tmp_path / "m.csv"
    pymineflow.write_table({**table, "VALUE": [1.0, 1.0]}, model)
    assert main([str(model), "--block-size", "10", "--slope", "45", "--economics", str(path), "--no-output"]) == 1


def test_factor_ranges_stop_at_stop():
    assert _factors(["0:1:0.35"]) == [0.0, 0.35, 0.7]
    assert _factors(["0.3:1:0.1"])[-1] == pytest.approx(1.0)
    assert len(_factors(["0.01:1:0.01"])) == 100


def test_command_line_options_are_used_or_refused(tmp_path, capsys):
    table = {**small(), "TON": [10.0] * 4}
    path = tmp_path / "m.csv"
    pymineflow.write_table(table, path)
    base = [str(path), "--block-size", "10", "--slope", "45", "--no-output"]
    nested = [*base, "--factors", "0.5:1:0.25", "--discount-rate", "0.1", "--ore-per-year", "5"]
    assert main([*nested, "--npv-report", str(tmp_path / "npv.csv"), "--fixed-cost", "1"]) == 0
    assert (tmp_path / "npv.csv").exists()
    out = capsys.readouterr().out
    assert "fixed cost 1 a year" in out  # the phase report uses the same NPV as the pit-by-pit table
    assert main([*base, "--scale", "price"]) == 1
    assert main([*base, "--verbose"]) == 1
    assert main([*base, "--price-factor", "0.9"]) == 1
    assert main([*nested, "--color-by", "level"]) == 1
    no_capacity = [*base, "--factors", "0.5:1:0.25", "--discount-rate", "0.1", "--benches-per-year", "8"]
    assert main([*no_capacity, "--npv-report", str(tmp_path / "x.csv")]) == 1


def test_precedence_with_coordinates_takes_the_spacing_as_block_size():
    """Without block_size, coordinates give the block size (their spacing), so the benches are right."""
    t = {"X": [500005.0, 500015.0, 500025.0, 500015.0], "Y": [9000005.0, 9000015.0, 9000005.0, 9000005.0],
         "Z": [3207.5, 3207.5, 3207.5, 3192.5], "VALUE": [-1.0, -1.0, -1.0, 10.0]}  # fmt: skip
    arcs = (np.array([3, 3, 3]), np.array([0, 1, 2]))
    flat = {**t, "Y": [9000005.0] * 4}  # one row: no spacing to read, the block size is asked for
    with pytest.raises(ValueError, match="block_size"):
        pymineflow.solve(flat, precedence=arcs, bench_discount=0.1)
    estimated = pymineflow.solve(t, precedence=arcs, bench_discount=0.1)
    given = pymineflow.solve(t, precedence=arcs, bench_discount=0.1, block_size=(10, 10, 15))
    assert estimated.real_value == pytest.approx(given.real_value)
    assert estimated.real_value == pytest.approx(10 / 1.1 - 3)
    assert pymineflow.solve(t, precedence=list(arcs)).n_blocks == 4  # a list of two arrays too


def test_npv_options_are_used_or_refused():
    table = {**small(), "TON": [10.0] * 4}
    plain = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=[0.5, 1.0])
    with pytest.raises(ValueError, match="need a capacity"):
        plain.report_table(discount_rate=0.1)
    with pytest.raises(ValueError, match="need a discount rate"):
        plain.report_table(mining_rate=50)
    with pytest.raises(TypeError, match="unknown NPV options"):
        plain.report_table(discount_rate=0.1, mining_rate=50, foo=1)
    assert "npv_best" in plain.report_table(discount_rate=0.1, mining_rate=50)
    with pytest.raises(ValueError, match="view"):
        plain.export("never_written", view="diagonal")


def test_value_blocks_density_in_every_form():
    t = {"IX": [0, 1, 2], "IY": [0, 0, 0], "IZ": [0, 0, 0], "AU": [0.0, 2.0, 3.0], "ROCK": ["a", "b", "a"],
         "RHO": [2.0, 2.5, 3.0]}  # fmt: skip
    eco = {"metals": [{"grade": "AU", "price": 600}],
           "processes": [{"name": "mill", "cost": 10, "recovery": {"AU": 1}}], "mining_cost": 1}  # fmt: skip
    expected = [16.0, 20.0, 24.0]  # density x 2 x 2 x 2
    for density in ("RHO", np.array([2.0, 2.5, 3.0])):
        out = pymineflow.value_blocks(t, economics=eco, density=density, block_size=2)
        assert out["TONNAGE"].tolist() == expected
    by_domain = pymineflow.value_blocks(t, metals={"AU": {"price": 600}}, processing_cost=10, mining_cost=1,
                                        density={"a": 2.0, "b": 2.5}, block_size=2, domain="ROCK")  # fmt: skip
    assert by_domain["TONNAGE"].tolist() == [16.0, 20.0, 16.0]


def test_section_models_export_without_a_surface(tmp_path, rendering):
    n = 20
    k = np.arange(n * 8)
    ore = (np.abs(k % n - 10) < 4) & (k // n < 4)
    table = {"IX": k % n, "IY": np.zeros(k.size, int), "IZ": k // n, "VALUE": np.where(ore, 50.0, -1.0),
             "TON": np.full(k.size, 10.0)}  # fmt: skip
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=[0.5, 1.0], discount_rate=0.1,
                                  ore_per_year=100)  # fmt: skip
    oes = pits.oes(min_component=0)
    names = [p.name for p in oes.export(tmp_path, details=True)]
    assert "oes.gif" in names
    assert "oes_surface.gif" not in names  # a vertical section has no ground surface
    assert "oes.csv" in names
    with pytest.raises(ValueError, match="seconds"):
        oes.animate(tmp_path / "x.gif", seconds_per_frame=-1)


def test_sequence_colours_are_those_of_every_block(rendering):
    from pymineflow.plot import _model_bounds

    n = 12
    k = np.arange(n * n * 5)
    rng = np.random.default_rng(3)
    table = {"IX": k % n, "IY": k // n % n, "IZ": k // (n * n), "VALUE": rng.normal(5, 20, k.size),
             "TON": np.full(k.size, 10.0)}  # fmt: skip
    pits = pymineflow.nested_pits(table, block_size=SIZE, slopes=45, factors=np.linspace(0.2, 1, 6),
                                  discount_rate=0.1, ore_per_year=500)  # fmt: skip
    oes = pits.oes(min_component=0)
    for color_by, own in (("period", oes.period), ("phase", oes.phase), ("rank", oes.rank)):
        scene = oes._scene(frames=10, color_by=color_by, remaining=True, view="top", rotation=0.0, title=None,
                           outline=False, axes=False, cmap="jet", window_size=(300, 200), off_screen=True)  # fmt: skip
        assert set(np.unique(scene.face_colors)) <= set(np.unique(own[own >= 0]).astype(float))
        scene.plotter.close()
    assert _model_bounds(pits.model.grid)[0] < _model_bounds(pits.model.grid)[1]


def test_command_line_refuses_options_that_do_not_apply(tmp_path):
    table = {**small(), "TON": [10.0] * 4}
    path = tmp_path / "m.csv"
    pymineflow.write_table(table, path)
    base = [str(path), "--block-size", "10", "--slope", "45", "--no-output"]
    assert main([*base, "--animate", str(tmp_path / "a.gif")]) == 1
    assert main([*base, "--report", "incremental"]) == 1
    assert main([*base, "--color-by", "level"]) == 1
    nested = [*base, "--factors", "0.5", "1", "--discount-rate", "0.1", "--benches-per-year", "8"]
    assert main([*nested, "--timing", "end"]) == 1
    assert main([*base, "--factors", "0.5", "1", "--report", "incremental"]) == 0
