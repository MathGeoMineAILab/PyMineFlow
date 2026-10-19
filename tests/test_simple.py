"""solve(), the reports of PitResult (summary, benches, to_csv), the command line and the 3-D view."""

import subprocess
import sys

import numpy as np
import pytest

import pymineflow
from pymineflow import read_blocks, scale_values, solve, ultimate_pit, write_table
from pymineflow.cli import main

# A small model cut by a topography, with real (decimal) values and a tonnage column
rng = np.random.default_rng(3)
NX, NY, NZ = 9, 8, 6
surface = rng.integers(3, NZ, size=(NX, NY))
gx, gy, gz = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")
below = gz <= surface[:, :, None]
IX, IY, IZ = gx[below], gy[below], gz[below]
N = IX.size
VALUE = np.round(rng.normal(-2.0, 6.0, N), 2)
TON = np.round(rng.uniform(900, 1100, N), 1)
SIZE = (10.0, 10.0, 5.0)
TABLE = {"IX": IX, "IY": IY, "IZ": IZ, "VALUE": VALUE, "TON": TON, "ROCK": np.where(VALUE > 0, "ore", "waste")}


def reference(slopes=45, **kwargs):
    values, factor = scale_values(VALUE)
    return ultimate_pit(values, ix=IX, iy=IY, iz=IZ, block_size=SIZE, slopes=slopes, **kwargs), factor


def test_solve_matches_ultimate_pit():
    pit = solve(TABLE, block_size=SIZE, slopes=45)
    ref, factor = reference()
    assert np.array_equal(pit.mask, ref.mask)
    assert pit.value == ref.value
    assert pit.factor == factor
    assert pit.real_value == pytest.approx(VALUE[pit.mask].sum(), abs=1e-6)
    assert pit.value_column == "VALUE"


@pytest.mark.parametrize("name", ["model.csv", "model.parquet", "model.txt"])
def test_solve_from_files(tmp_path, name):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(TABLE)
    if name.endswith(".parquet"):
        pytest.importorskip("pyarrow")
        df.to_parquet(tmp_path / name)
    else:
        df.to_csv(tmp_path / name, index=False, sep=";" if name.endswith(".txt") else ",")
    pit = solve(tmp_path / name, block_size=SIZE, slopes=45)
    assert np.array_equal(pit.mask, reference()[0].mask)


def test_column_names_are_found_in_any_case():
    table = {"i": IX, "j": IY, "k": IZ, "Block_Value": VALUE}
    assert np.array_equal(solve(table, block_size=SIZE, slopes=45).mask, reference()[0].mask)


def test_coordinates_and_depth():
    x, y = 500.0 + IX * SIZE[0], 9000.0 + IY * SIZE[1]
    depth = (NZ - 1 - IZ) * SIZE[2] + 2.5  # z grows downwards
    pit = solve({"XC": x, "YC": y, "ZC": depth, "PROFIT": VALUE}, block_size=SIZE, slopes=45, z_up=False)
    assert np.array_equal(pit.mask, reference()[0].mask)


def test_explicit_columns_and_options():
    table = {"a": IX, "b": IY, "c": IZ, "v": VALUE, "X": IX * 1.0, "Y": IY * 1.0, "Z": IZ * 1.0}
    pit = solve(table, block_size=SIZE, slopes=[(0, 45), (180, 45)], value="v", indices=("a", "b", "c"), n_benches=3)
    ref, _ = reference(slopes=[(0, 45), (180, 45)], n_benches=3)
    assert np.array_equal(pit.mask, ref.mask)
    assert pit.n_benches == 3


@pytest.mark.parametrize(
    ("table", "kwargs", "match"),
    [
        ({"IX": IX, "IY": IY, "IZ": IZ}, {}, "no value column found"),
        ({"IX": IX, "IY": IY, "IZ": IZ, "VALUE": VALUE}, {"value": "NPV"}, "value column 'NPV' not found"),
        ({"A": IX, "B": IY, "C": IZ, "VALUE": VALUE}, {}, "no block positions found"),
        ({"IX": IX, "IY": IY, "IZ": IZ, "VALUE": VALUE}, {"indices": ("IX", "IY", "Q")}, r"index columns \['Q'\]"),
        (
            {"IX": IX, "IY": IY, "IZ": IZ, "VALUE": VALUE},
            {"indices": ("IX", "IY", "IZ"), "coords": ("IX", "IY", "IZ")},
            "not both",
        ),
    ],
)
def test_solve_errors(table, kwargs, match):
    with pytest.raises(ValueError, match=match):
        solve(table, block_size=SIZE, slopes=45, **kwargs)


def test_summary_reports_the_model_and_the_pit():
    pit = solve(TABLE, block_size=SIZE, slopes=45)
    text = pit.summary()
    inside = pit.mask
    ore, waste = inside & (VALUE > 0), inside & (VALUE <= 0)
    for expected in (
        f"PyMineFlow {pymineflow.__version__}",
        f"blocks                {N:,}",
        f"grid                  {NX} x {NY} x {NZ} cells",
        "block size            10 x 10 x 5",
        "value column          VALUE (scaled x",
        "slope                 45 deg",
        f"pattern               minimum, {NZ - 1} benches (full height)",
        f"value                 {pit.real_value:,.2f}",
        f"blocks                {pit.n_blocks:,}",
        f"ore (value > 0)       {int(ore.sum()):,} blocks",
        f"waste (value <= 0)    {int(waste.sum()):,} blocks",
        f"tonnage (TON)         {TON[inside].sum():,.2f}",
        f"strip ratio           {TON[waste].sum() / TON[ore].sum():.4f}",
        "solver time",
    ):
        assert expected in text, expected


def test_summary_without_tonnage_and_for_ultimate_pit():
    ref, _ = reference()
    text = ref.summary()  # integer values, no source table
    assert "tonnage" not in text
    assert "value column" not in text
    assert f"blocks                {ref.n_blocks:,}" in text
    assert "1:5" in ultimate_pit(np.array([5, -1]), ix=[0, 0], iy=[0, 0], iz=[0, 1], block_size=SIZE).summary()


def test_benches_add_up_to_the_pit():
    pit = solve(TABLE, block_size=SIZE, slopes=45)
    b = pit.benches()
    assert list(b) == ["level", "blocks", "value", "tonnage"]
    assert list(b["level"]) == sorted(b["level"], reverse=True)  # from the top
    assert b["blocks"].sum() == pit.n_blocks
    assert b["value"].sum() == pytest.approx(pit.real_value)
    assert b["tonnage"].sum() == pytest.approx(TON[pit.mask].sum())
    for level, blocks in zip(b["level"], b["blocks"], strict=True):
        assert blocks == int((pit.mask & (level == IZ)).sum())


@pytest.mark.parametrize("engine", ["auto", "pyarrow", "polars", "pandas", "csv"])
def test_to_csv_round_trip(tmp_path, engine):
    if engine not in ("auto", "csv"):
        pytest.importorskip(engine)
    pit = solve(TABLE, block_size=SIZE, slopes=45)
    if engine == "auto":
        path = pit.to_csv(tmp_path / "pit.csv")
    else:
        table = dict(TABLE, PIT=pit.mask.astype(np.int8))
        path = write_table(table, tmp_path / "pit.csv", engine=engine) if engine != "csv" else None
        if path is None:  # Python's csv module (no table library)
            from pymineflow import readers

            original = readers.find_spec
            readers.find_spec = lambda name: None
            try:
                path = write_table(table, tmp_path / "pit.csv")
            finally:
                readers.find_spec = original
    back = read_blocks(path, engine="pandas") if engine != "auto" else read_blocks(path)
    assert list(back) == [*TABLE, "PIT"]
    assert np.array_equal(back["PIT"].astype(bool), pit.mask)
    assert np.allclose(back["VALUE"], VALUE)
    assert back["ROCK"].tolist() == TABLE["ROCK"].tolist()


def test_to_csv_without_table_and_errors(tmp_path):
    ref, _ = reference()
    back = read_blocks(ref.to_csv(tmp_path / "pit.csv"))
    assert list(back) == ["IX", "IY", "IZ", "PIT"]
    assert back["PIT"].sum() == ref.n_blocks
    pit = solve(TABLE, block_size=SIZE, slopes=45)
    with pytest.raises(ValueError, match="already has a column 'VALUE'"):
        pit.to_csv(tmp_path / "x.csv", column="VALUE")
    with pytest.raises(ValueError, match="same length"):
        write_table({"a": [1, 2], "b": [1]}, tmp_path / "bad.csv")


# Command line


def model_file(tmp_path):
    pd = pytest.importorskip("pandas")
    pd.DataFrame(TABLE).to_csv(tmp_path / "model.csv", index=False)
    return tmp_path / "model.csv"


def test_command_line(tmp_path, capsys):
    path = model_file(tmp_path)
    code = main(
        [str(path), "--block-size", "10", "10", "5", "--slope", "45", "--bench-report", str(tmp_path / "b.csv")]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "ULTIMATE PIT" in out
    assert "pit written to" in out
    pit = read_blocks(tmp_path / "model_pit.csv")
    assert np.array_equal(pit["PIT"].astype(bool), reference()[0].mask)
    assert read_blocks(tmp_path / "b.csv")["blocks"].sum() == pit["PIT"].sum()


def test_command_line_sectors_and_options(tmp_path, capsys):
    path = model_file(tmp_path)
    args = [str(path), "--block-size", "10", "10", "5", "--sectors", "0:45", "180:45", "--benches", "3"]
    assert main([*args, "--no-output", "--value", "VALUE", "--indices", "IX", "IY", "IZ"]) == 0
    out = capsys.readouterr().out
    assert "az 0: 45 deg, az 180: 45 deg" in out
    assert not (tmp_path / "model_pit.csv").exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--value", "NPV"], "value column 'NPV' not found"),
        (["--block-size", "10", "10", "0"], "block_size"),
    ],
)
def test_command_line_errors(tmp_path, capsys, args, message):
    path = model_file(tmp_path)
    base = [str(path), "--block-size", "10", "10", "5", "--slope", "45"]
    assert main([*base, *args]) == 1
    assert message in capsys.readouterr().err


def test_command_line_bad_sector_and_help():
    with pytest.raises(SystemExit):
        main(["m.csv", "--block-size", "1", "1", "1", "--sectors", "45"])
    done = subprocess.run([sys.executable, "-m", "pymineflow", "--help"], capture_output=True, text=True, check=False)
    assert done.returncode == 0
    assert "--block-size" in done.stdout


# 3-D view (PyVista is installed with PyMineFlow)

RENDER = """
import sys
import numpy as np
from pymineflow import nested_pits, solve
table = {"IX": [0, 1, 1, 2], "IY": [0, 0, 0, 0], "IZ": [0, 0, 1, 0], "VALUE": [9.5, -1.0, -1.0, -1.0]}
pit = solve(table, block_size=(10, 10, 10), slopes=45)
for view in ("iso", "below", "top", "side"):
    pit.plot(screenshot=sys.argv[1] + view + ".png", view=view).close()
pit.plot(screenshot=sys.argv[1] + "plan.png", view="top", rotation=-90, outline=False, axes=False, title="").close()
pits = nested_pits(table, block_size=(10, 10, 10), slopes=45, factors=[0.0, 0.6, 1.0])  # the first pit is empty
pits.animate(sys.argv[1] + "nested.gif", step=2)
pits.animate(sys.argv[1] + "plan.gif", view="top", rotation=-90, outline=False)
pits.plot(screenshot=sys.argv[1] + "nested.png").close()
pits.plot(screenshot=sys.argv[1] + "nested_plan.png", view="top", rotation=-90, outline=False, title="").close()
plotter = pits.plot(show=False)  # the scene with its slider, not shown
plotter.close()
"""


def test_pit_mesh(tmp_path):
    pytest.importorskip("pyvista")
    from pymineflow.plot import pit_mesh

    pit = solve(TABLE, block_size=SIZE, slopes=45)
    mesh = pit_mesh(pit, "value", surface=False)
    assert mesh.n_cells == pit.n_blocks
    assert np.allclose(np.sort(mesh.cell_data["VALUE"]), np.sort(VALUE[pit.mask]))
    assert pit_mesh(pit, "ore", surface=False).cell_data["ore (1) / waste (0)"].sum() == (VALUE[pit.mask] > 0).sum()
    assert pit_mesh(pit, "TON", surface=False).n_cells == pit.n_blocks
    assert pit_mesh(pit, "level").n_cells > 0  # outer faces only
    centres = mesh.cell_centers().points  # in model coordinates: block centres
    assert np.allclose(np.sort(centres[:, 2]), np.sort(IZ[pit.mask] * SIZE[2]))
    with pytest.raises(ValueError, match="color_by must be one of"):
        pit_mesh(pit, "nothing")
    with pytest.raises(ValueError, match="not numeric"):
        pit_mesh(pit, "ROCK")
    with pytest.raises(ValueError, match="view must be"):
        pit.plot(view="front", show=False)


def test_nested_faces():
    """The faces computed once for all the nested pits: for every pit, exactly its outer faces."""
    from pymineflow.plot import _nested_faces

    pits = pymineflow.nested_pits(TABLE, block_size=SIZE, slopes=45, factors=np.linspace(0.0, 1.0, 11))
    g = pits.model.grid
    final = pits[len(pits) - 1].mask
    sides = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    def outer_faces(mask):
        cells = {(int(x), int(y), int(z)) for x, y, z in zip(g.ix[mask], g.iy[mask], g.iz[mask], strict=True)}
        return sum((x + dx, y + dy, z + dz) not in cells for x, y, z in cells for dx, dy, dz in sides)

    for remaining in (False, True):
        points, quads, first, last, own, cell = _nested_faces(g, pits.numbers, len(pits), remaining)
        assert np.array_equal(
            own,
            np.zeros(g.n_cells, dtype=np.int64)[cell]
            + np.bincount(np.asarray(g.linear), weights=pits.numbers, minlength=g.n_cells)[cell],
        )  # the step of the face's block
        for k, pit in enumerate(pits, start=1):
            # mined: the faces of pit k; remaining: the faces of what is left after pit k
            mask = final & ~pit.mask if remaining else pit.mask
            shown = (first <= k) & (k < last)
            assert shown.sum() == outer_faces(mask), (remaining, k)
            assert np.all(own[shown] > k) if remaining else np.all(own[shown] <= k)
            if shown.any():  # every face lies on the box of a block
                corners = points[quads[shown]]  # (faces, 4, 3)
                assert np.allclose(np.ptp(corners, axis=1).min(axis=1), 0)  # flat
        start = (first <= 0) & (last > 0)  # nothing mined yet: the remaining part is the whole final pit
        assert not remaining or start.sum() == outer_faces(final)
    pytest.importorskip("pyvista")
    with pytest.raises(ValueError, match="every nested pit is empty"):
        pymineflow.nested_pits(TABLE, block_size=SIZE, slopes=45, factors=[0.0]).plot(show=False)


def test_plot_saves_images(tmp_path):
    """Rendered in a separate process: a machine without OpenGL may crash instead of raising."""
    pytest.importorskip("pyvista")
    done = subprocess.run(
        [sys.executable, "-c", RENDER, str(tmp_path / "pit_")], capture_output=True, text=True, timeout=300, check=False
    )
    if done.returncode != 0:
        pytest.skip(f"no rendering available here (exit {done.returncode}): {done.stderr[-300:]}")
    for view in ("iso", "below", "top", "side", "plan", "nested", "nested_plan"):
        assert (tmp_path / f"pit_{view}.png").stat().st_size > 0
    from PIL import Image

    with Image.open(tmp_path / "pit_nested.gif") as gif:
        assert gif.n_frames == 2  # pits 1 (empty) and 3
    with Image.open(tmp_path / "pit_plan.gif") as gif:
        assert gif.n_frames >= 2  # equal frames in a row are merged


def test_import_after_pyarrow():
    """Windows: an older msvcp140.dll loaded first (e.g. by pyarrow) must not crash the extension."""
    code = "import pyarrow, pymineflow; print(pymineflow.ExplicitPit(2, [0], [1], [5, -1]).solve()[1])"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "4"


# Nested pits, the easy way


FACTORS = [1.0, 0.4, 0.7, 0.55, 0.85]  # any order: the result is sorted


def test_nested_pits_easy():
    pits = pymineflow.nested_pits(TABLE, block_size=SIZE, slopes=45, factors=FACTORS)
    assert pits.factors == sorted(FACTORS)
    values, factor = scale_values(VALUE)
    for f, pit in zip(pits.factors, pits, strict=True):
        scenario = np.where(values > 0, np.rint(values * f), values).astype(np.int64)
        ref = ultimate_pit(scenario, ix=IX, iy=IY, iz=IZ, block_size=SIZE, slopes=45)
        assert pit.value == ref.value
        assert np.array_equal(pit.mask, ref.mask)
        assert pit.factor == factor
    assert np.array_equal(pits[-1].mask, solve(TABLE, block_size=SIZE, slopes=45).mask)  # factor 1
    number = pits.pit_number()
    for k, pit in enumerate(pits):
        assert np.array_equal((number >= 1) & (number <= k + 1), pit.mask)


def test_nested_pits_reports(tmp_path):
    pits = pymineflow.nested_pits(TABLE, block_size=SIZE, slopes=45, factors=FACTORS)
    t = pits.table_by_pit()
    assert list(t) == ["pit", "factor", "blocks", "new_blocks", "value", "value_at_factor_1", "tonnage",
                       "ore_tonnage", "waste_tonnage", "strip_ratio"]  # fmt: skip
    for k, pit in enumerate(pits):
        assert t["blocks"][k] == pit.n_blocks
        assert t["value"][k] == pytest.approx(pit.real_value)
        assert t["value_at_factor_1"][k] == pytest.approx(VALUE[pit.mask].sum())
        assert t["tonnage"][k] == pytest.approx(TON[pit.mask].sum())
    text = pits.summary()
    assert f"Nested pits: {len(FACTORS)} revenue factors" in text
    assert "strip ratio" in text
    back = read_blocks(pits.to_csv(tmp_path / "nested.csv"))
    assert np.array_equal(back["PIT_NUMBER"], pits.pit_number())
    with pytest.raises(ValueError, match="factors must"):
        pymineflow.nested_pits(TABLE, block_size=SIZE, slopes=45, factors=[])
    with pytest.raises(ValueError, match="finite and >= 0"):
        pymineflow.nested_pits(TABLE, block_size=SIZE, slopes=45, factors=[0.5, -1])


def test_nested_pits_command_line(tmp_path, capsys):
    path = model_file(tmp_path)
    args = [str(path), "--block-size", "10", "10", "5", "--slope", "45", "--factors", "0.4:1.0:0.15", "0.55"]
    assert main([*args, "--bench-report", str(tmp_path / "pits.csv"), "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "Nested pits: 6 revenue factors" in out  # 0.4, 0.55, 0.7, 0.85, 1.0 and 0.55 again
    assert "Solving 6 nested pits by revenue factor" in out
    assert "whole model" in out
    numbers = read_blocks(tmp_path / "model_nested.csv")["PIT_NUMBER"]
    same = pymineflow.nested_pits(TABLE, block_size=SIZE, slopes=45, factors=[0.4, 0.55, 0.7, 0.85, 1.0, 0.55])
    assert np.array_equal(numbers, same.pit_number())
    assert read_blocks(tmp_path / "pits.csv")["blocks"].size == 6
    assert main([*args[:-3], "--factors", "1:0.5:0.1"]) == 1
    assert "STOP >= START" in capsys.readouterr().err


def test_nested_pits_verbose(capsys):
    """Every pit is reported once, as it is solved: the largest first, then each one on a ring."""
    factors = np.linspace(0.3, 1.0, 9)
    pits = pymineflow.nested_pits(TABLE, block_size=SIZE, slopes=45, factors=factors, verbose=True)
    lines = capsys.readouterr().out.splitlines()
    assert "Solving 9 nested pits by revenue factor" in lines[0]
    rows = [line.split() for line in lines[3:]]
    assert len(rows) == 9
    assert [int(r[0]) for r in rows] == list(range(1, 10))
    assert sorted(int(r[1]) for r in rows) == list(range(1, 10))
    assert rows[0][1] == "9"
    assert " ".join(rows[0][3:5]) == "whole model"
    for r in rows:
        k = int(r[1])
        assert float(r[2]) == pytest.approx(pits.factors[k - 1], abs=1e-4)
        blocks = int(r[-4].replace(",", ""))  # ... blocks value time ms
        assert blocks == pits[k - 1].n_blocks
    with pytest.raises(ValueError, match="step must be"):
        pits.animate("never.gif", step=0)


def test_nested_pit_numbers_progress():
    """The progress callback: ring bounds, cells and blocks of every pit, in solving order."""
    seen = []
    pits = pymineflow.nested_pits(TABLE, block_size=SIZE, slopes=45, factors=np.linspace(0.3, 1.0, 9))
    model = pits.model
    numbers, values, _ = model.nested_pit_numbers(pits.scenario, len(pits), seen.append)
    assert np.array_equal(numbers, pits.numbers)
    assert [s["order"] for s in seen] == list(range(1, 10))
    assert seen[0]["pit"] == 9
    assert seen[0]["outer"] == 0
    solved = {9}
    for s in seen[1:]:
        assert s["inner"] < s["pit"] < s["outer"]  # solved between two pits already known
        assert s["inner"] in solved | {0}
        assert s["outer"] in solved
        solved.add(s["pit"])
        assert s["blocks"] == pits[s["pit"] - 1].n_blocks
        assert s["value"] == values[s["pit"] - 1]
        assert s["cells"] <= seen[0]["cells"]


def test_solve_with_explicit_precedence(tmp_path):
    """precedence= solves with the arcs as given (a MineLib .prec file or arrays), by ID or by row."""
    rng = np.random.default_rng(5)
    n = 60
    ids = rng.permutation(np.arange(100, 100 + n))  # ids in any order
    value = rng.normal(0, 10, n)
    k = np.arange(n)
    table = {"ID": ids, "IX": k % 5, "IY": (k // 5) % 4, "IZ": k // 20, "VALUE": value}
    src_rows = rng.integers(0, n, 150)
    dst_rows = rng.integers(0, n, 150)
    keep = src_rows != dst_rows
    src_rows, dst_rows = src_rows[keep], dst_rows[keep]
    values, _ = pymineflow.scale_values(value)
    expected, _, _, _ = pymineflow.ExplicitPit(n, src_rows, dst_rows, values).solve()
    by_id = pymineflow.solve(table, precedence=(ids[src_rows], ids[dst_rows]))
    assert np.array_equal(by_id.mask, expected)
    assert by_id.pattern == "explicit"
    assert "as given" in by_id.summary()
    lines = {}
    for a, b in zip(ids[src_rows], ids[dst_rows], strict=True):
        lines.setdefault(int(a), []).append(int(b))
    prec = tmp_path / "toy.prec"
    prec.write_text("".join(f"{i} {len(p)} {' '.join(map(str, p))}\n" for i, p in lines.items()))
    assert np.array_equal(pymineflow.solve(table, precedence=prec).mask, expected)
    by_row = pymineflow.solve({k: v for k, v in table.items() if k != "ID"}, precedence=(src_rows, dst_rows))
    assert np.array_equal(by_row.mask, expected)
    assert by_row.to_csv(tmp_path / "pit.csv").exists()
    with pytest.raises(ValueError, match="not in column"):
        pymineflow.solve(table, precedence=(np.array([1]), np.array([2])))
    with pytest.raises(ValueError, match="rows from 0"):
        pymineflow.solve({k: v for k, v in table.items() if k != "ID"}, precedence=(np.array([0]), np.array([n])))
    with pytest.raises(ValueError, match="one or the other"):
        pymineflow.solve(table, precedence=prec, slopes=45)
    with pytest.raises(ValueError, match="block_size and slopes"):
        pymineflow.solve(table)
    with pytest.raises(ValueError, match="integer arrays"):
        pymineflow.solve(table, precedence=(np.array([0.5]), np.array([1.0])))


def test_command_line_with_precedence(tmp_path, capsys):
    from pymineflow.cli import main

    table = {"ID": [0, 1, 2], "IX": [0, 1, 0], "IY": [0, 0, 0], "IZ": [0, 0, 1], "VALUE": [9.0, -2.0, -3.0]}
    pymineflow.write_table(table, tmp_path / "m.csv")
    (tmp_path / "m.prec").write_text("0 2 1 2\n1 0\n2 0\n")
    assert main([str(tmp_path / "m.csv"), "--precedence", str(tmp_path / "m.prec"), "--no-output"]) == 0
    out = capsys.readouterr().out
    assert "as given" in out
    assert "4" in out
    assert main([str(tmp_path / "m.csv"), "--slope", "45"]) == 1  # a slope needs a block size
    assert main([str(tmp_path / "m.csv"), "--precedence", str(tmp_path / "m.prec"), "--factors", "0.5", "1"]) == 1
