"""read_blocks: every engine returns the same arrays for every file layout; in-memory tables are accepted."""

import numpy as np
import pytest

from pymineflow import read_blocks

COLUMNS = ["IX", "IY", "IZ", "VALUE"]
DATA = {
    "IX": np.arange(50) % 5,
    "IY": np.arange(50) // 5 % 5,
    "IZ": np.arange(50) // 25,
    "VALUE": np.arange(50) - 25,
    "OTHER": np.zeros(50),
}
ENGINES = ["auto", "pandas", "pyarrow", "polars"]


def skip_missing(engine):
    if engine != "auto":
        pytest.importorskip(engine)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", ["model.csv", "model.parquet", "model.feather"])
def test_engines_return_the_same_arrays(tmp_path, engine, name):
    pd = pytest.importorskip("pandas")
    skip_missing(engine)
    df = pd.DataFrame(DATA)
    if name.endswith(".csv"):
        df.to_csv(tmp_path / name, index=False)
    else:
        pytest.importorskip("pyarrow")  # pandas writes Parquet/Feather through pyarrow
        getattr(df, "to_parquet" if name.endswith(".parquet") else "to_feather")(tmp_path / name)
    d = read_blocks(tmp_path / name, COLUMNS, engine=engine)
    assert list(d) == COLUMNS
    assert all(np.array_equal(d[c], DATA[c]) for c in COLUMNS)


def test_in_memory_tables():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(DATA)
    tables = [df, dict(DATA)]
    for name in ("pyarrow", "polars"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        tables.append(module.Table.from_pandas(df) if name == "pyarrow" else module.from_pandas(df))
    for table in tables:
        d = read_blocks(table, ["IX", "VALUE"])
        assert np.array_equal(d["IX"], DATA["IX"])
        assert np.array_equal(d["VALUE"], DATA["VALUE"])


def test_all_columns_by_default():
    assert list(read_blocks(dict(DATA))) == list(DATA)


def test_errors(tmp_path):
    (tmp_path / "model.xlsx").write_bytes(b"PK\x03\x04\x00\x00binary")
    with pytest.raises(ValueError, match="is not a text file"):
        read_blocks(tmp_path / "model.xlsx")
    with pytest.raises(ValueError, match="engine must be one of"):
        read_blocks(tmp_path / "model.csv", engine="other")
    with pytest.raises(TypeError, match="unsupported source int"):
        read_blocks(42)
    (tmp_path / "model.csv").write_text("IX,VALUE\n0,5\n1,-1\n")
    with pytest.raises(ValueError, match=r"columns \['IY'\] not found"):
        read_blocks(tmp_path / "model.csv", ["IX", "IY"])
    with pytest.raises(ValueError, match="names has 3 entries"):
        read_blocks(tmp_path / "model.csv", names=["A", "B", "C"])
    with pytest.raises(ValueError, match="sep must be one character"):
        read_blocks(tmp_path / "model.csv", sep="::")
    (tmp_path / "empty.txt").write_text("")
    with pytest.raises(ValueError, match="is empty"):
        read_blocks(tmp_path / "empty.txt")


# Delimited text: every layout gives the same arrays with every engine

TEXT = {
    "IX": np.arange(60) % 5,
    "IY": np.arange(60) // 5 % 4,
    "IZ": np.arange(60) // 20,
    "VALUE": np.arange(60) * 7 - 200,
    "GRADE": np.round(np.linspace(0.0, 3.0, 60), 4),
}


def write_text(path, sep=",", header=True, decimal=".", pad=0):
    def cell(v):
        s = repr(float(v)) if isinstance(v, np.floating) else str(int(v))
        return s.replace(".", decimal).rjust(pad)

    rows = [sep.join(cell(TEXT[c][i]) for c in TEXT) for i in range(60)]
    lines = ([sep.join(name.rjust(pad) for name in TEXT)] if header else []) + rows
    path.write_text("\n".join(lines) + "\n")
    return path


def check(d, names=tuple(TEXT)):
    assert list(d) == list(names)
    for name, key in zip(names, TEXT, strict=True):
        assert np.array_equal(d[name], TEXT[key]), name


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("name", "sep"),
    [("m.csv", ","), ("m.txt", ";"), ("m.tsv", "\t"), ("m.dat", "|"), ("m.blocks", " "), ("model", ",")],
)
def test_text_separators_with_header(tmp_path, engine, name, sep):
    skip_missing(engine)
    check(read_blocks(write_text(tmp_path / name, sep), engine=engine))


@pytest.mark.parametrize("engine", ENGINES)
def test_headerless_whitespace_like_minelib(tmp_path, engine):
    skip_missing(engine)
    path = write_text(tmp_path / "m.blocks", " ", header=False)
    check(read_blocks(path, engine=engine), names=[str(i) for i in range(len(TEXT))])
    names = ["ix", "iy", "iz", "value", "grade"]
    check(read_blocks(path, engine=engine, names=names), names=names)
    d = read_blocks(path, ["value", "ix"], engine=engine, names=names)
    assert list(d) == ["value", "ix"]
    assert np.array_equal(d["value"], TEXT["VALUE"])
    assert np.array_equal(d["ix"], TEXT["IX"])


@pytest.mark.parametrize("engine", ENGINES)
def test_decimal_comma(tmp_path, engine):
    skip_missing(engine)
    check(read_blocks(write_text(tmp_path / "m.csv", ";", decimal=","), engine=engine))


def test_irregular_whitespace_needs_pandas(tmp_path):
    pytest.importorskip("pandas")
    path = write_text(tmp_path / "m.txt", " ", pad=12)  # aligned columns
    check(read_blocks(path))
    for engine in ("pyarrow", "polars"):
        with pytest.raises(ValueError, match="irregular whitespace"):
            read_blocks(path, engine=engine)


def test_gslib(tmp_path):
    pytest.importorskip("pandas")
    rows = [" ".join(f"{TEXT[c][i]:>10}" for c in TEXT) for i in range(60)]
    (tmp_path / "m.gslib").write_text("Block model\n5\n" + "\n".join(TEXT) + "\n" + "\n".join(rows) + "\n")
    check(read_blocks(tmp_path / "m.gslib"))


@pytest.mark.parametrize("engine", ENGINES)
def test_text_columns_and_late_types(tmp_path, engine):
    """A text column, and a float column whose first 200 rows look like integers."""
    skip_missing(engine)
    n = 300
    lines = [f"{i} {i % 7} {0 if i < 200 else i / 8} {'ore' if i % 2 else 'waste'}" for i in range(n)]
    (tmp_path / "m.blocks").write_text("\n".join(lines) + "\n")
    d = read_blocks(tmp_path / "m.blocks", engine=engine, names=["id", "ix", "au", "rock"])
    assert np.array_equal(d["id"], np.arange(n))
    assert d["au"].dtype == np.float64
    assert d["au"][-1] == (n - 1) / 8
    assert d["rock"][:2].tolist() == ["waste", "ore"]


# Layouts that are easy to misread: the same numbers with every engine, or a clear error


def read_all(path, **kwargs):
    """read_blocks with every installed engine; all results must be equal."""
    results = []
    for engine in ("pyarrow", "polars", "pandas"):
        try:
            __import__(engine)
        except ImportError:
            continue
        try:
            results.append(read_blocks(path, engine=engine, **kwargs))
        except ValueError as e:  # an old polars refuses a quoted header instead of misreading it
            if "upgrade polars" not in str(e):
                raise
    for other in results[1:]:
        assert list(other) == list(results[0])
        for key in other:
            assert np.array_equal(other[key], results[0][key], equal_nan=other[key].dtype.kind == "f"), key
    return results[0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IX,V\n1,2\n\n3,4\n", {"IX": [1, 3], "V": [2, 4]}),  # a blank line
        ("1 2\n\n3 4\n", {"0": [1, 3], "1": [2, 4]}),
        ('ID,ROCK,V\n1,"a,b",5\n2,"c",6\n', {"ID": [1, 2], "ROCK": ["a,b", "c"], "V": [5, 6]}),  # quoted separator
        ("IX, IY, V\n1, 2, 3.5\n4, 5, 6\n", {"IX": [1, 4], "IY": [2, 5], "V": [3.5, 6.0]}),  # padded fields
        ("IX,V,\n1,2,\n3,4,\n", {"IX": [1, 3], "V": [2, 4]}),  # trailing separator
        ("0 0 0 1,5\n1 0 0 2,25\n", {"0": [0, 1], "1": [0, 0], "2": [0, 0], "3": [1.5, 2.25]}),  # decimal comma
        (",2\n1,3\n5,6\n", {"0": [np.nan, 1, 5], "1": [2, 3, 6]}),  # missing value, not a header
        ("NA,2\n1,3\n5,6\n", {"0": [np.nan, 1, 5], "1": [2, 3, 6]}),
        ("IX,V\nNA,2\n1,nan\n", {"IX": [np.nan, 1], "V": [2, np.nan]}),
        ("V\n1\n2\n3\n", {"V": [1, 2, 3]}),  # not GSLIB
        ("-5\n1\n2\n3\n4\n", {"0": [-5, 1, 2, 3, 4]}),
        ('Grid 10" spacing\n2\nIX\nAU "ppm\n0 0.5\n1 1.5\n', {"IX": [0, 1], 'AU "ppm': [0.5, 1.5]}),  # GSLIB
        ('IX,AU "ppm,V\n1,2,3\n4,5,6\n', {"IX": [1, 4], 'AU "ppm': [2, 5], "V": [3, 6]}),
    ],
)
def test_layouts_that_are_easy_to_misread(tmp_path, text, expected):
    (tmp_path / "m.txt").write_text(text)
    d = read_all(tmp_path / "m.txt")
    assert list(d) == list(expected)
    for key, values in expected.items():
        if isinstance(values[0], str):
            assert d[key].tolist() == values
        else:
            assert np.allclose(d[key].astype(float), values, equal_nan=True), key


def test_blank_lines_after_the_sample(tmp_path):
    rows = "".join(f"{i},{i / 8}\n" for i in range(6000))
    (tmp_path / "m.csv").write_text("IX,V\n" + rows + "\n6000,7.5\n")
    d = read_all(tmp_path / "m.csv")
    assert d["IX"].dtype == np.int64
    assert d["IX"].size == 6001


@pytest.mark.parametrize("bad_row", ["8000  1 2.5\n", " 8000 1 2.5\n", "8000 1 2.5 7\n"])
def test_irregular_rows_after_the_sample_raise(tmp_path, bad_row):
    """A shifted or longer row must raise, never move numbers to another column."""
    rows = "".join(f"{i} {i % 7} {i * 0.5}\n" for i in range(8000))
    (tmp_path / "m.blocks").write_text(rows + bad_row)
    for engine in ("pyarrow", "polars"):
        pytest.importorskip(engine)
        with pytest.raises(Exception):  # noqa: B017, PT011  (each engine raises its own error type)
            read_blocks(tmp_path / "m.blocks", engine=engine)


def test_extra_field_raises_with_pandas(tmp_path):
    pytest.importorskip("pandas")
    rows = "".join(f"{i},{i % 7},{i / 2}\n" for i in range(6000))
    (tmp_path / "m.csv").write_text("IX,IY,V\n" + rows + "6000,1,2.5,9\n")
    with pytest.raises(Exception, match="Expected 3 fields"):
        read_blocks(tmp_path / "m.csv", ["IX", "V"], engine="pandas")


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("IX,V\n1,2,3\n4,5,6\n", "could not detect the separator"),  # header shorter than the rows
        ("IX,V,W\n1,2\n4,5\n", "could not detect the separator"),
        ("IX;A;B\n1;0,5;1.5\n", "mixes decimal points and decimal commas"),
    ],
)
def test_ambiguous_layouts_raise(tmp_path, text, match):
    (tmp_path / "m.csv").write_text(text)
    with pytest.raises(ValueError, match=match):
        read_blocks(tmp_path / "m.csv")


def test_explicit_layout_options(tmp_path):
    (tmp_path / "m.csv").write_text("1,2,3\n10,20,30\n")
    d = read_all(tmp_path / "m.csv", header=True)
    assert d["1"].tolist() == [10]
    (tmp_path / "t.csv").write_text("IX;TON\n1;1,234\n2;2,500\n")  # commas as thousands separators
    assert read_all(tmp_path / "t.csv")["TON"].tolist() == [1.234, 2.5]
    assert read_all(tmp_path / "t.csv", decimal=".")["TON"].tolist() == ["1,234", "2,500"]


def test_encodings(tmp_path):
    (tmp_path / "m.csv").write_bytes("LEY_\xd1,V\n1,2\n".encode("latin-1"))
    with pytest.raises(ValueError, match="pass encoding="):
        read_blocks(tmp_path / "m.csv")
    pytest.importorskip("pandas")
    d = read_blocks(tmp_path / "m.csv", encoding="latin-1", engine="pandas")
    assert d["LEY_\xd1"].tolist() == [1]


def test_argument_errors(tmp_path):
    (tmp_path / "m.csv").write_text("IX,VALUE\n0,5\n")
    with pytest.raises(TypeError, match="list of names"):
        read_blocks(tmp_path / "m.csv", "VALUE")
    with pytest.raises(ValueError, match="engine must be one of"):
        read_blocks({"a": [1]}, engine="other")
    with pytest.raises(ValueError, match="decimal must be"):
        read_blocks(tmp_path / "m.csv", decimal=";")
    assert list(read_blocks(tmp_path / "m.csv", ["VALUE", "VALUE"])) == ["VALUE"]
    (tmp_path / "m.csv.gz").write_bytes(b"\x1f\x8b\x08\x00rest")
    with pytest.raises(ValueError, match="gzip"):
        read_blocks(tmp_path / "m.csv.gz")
