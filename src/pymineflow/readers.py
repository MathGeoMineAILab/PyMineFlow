"""Read block models from files or from tables already in memory, and MineLib instances.

Files: delimited text with any extension (comma, semicolon, tab, pipe or whitespace; with or
without header; decimal comma; GSLIB), Parquet and Feather/Arrow. The layout of a text file is
detected from its first lines; when it is ambiguous, ``read_blocks`` asks for it (``sep``,
``header``, ``decimal``) instead of guessing.

Engines (``engine="auto"`` picks the fastest one installed): ``pyarrow`` (multithreaded CSV,
column-selective Parquet/Feather), ``polars`` (optional: ``pip install "pymineflow[polars]"``) and
``pandas``. pyarrow and pandas are installed with PyMineFlow.

MineLib (Espinoza et al. 2013): ``read_minelib_precedence`` and ``read_minelib_values`` read the
``.prec`` and ``.upit`` files for ``ExplicitPit``; ``read_blocks`` reads the ``.blocks`` file.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["read_blocks", "read_minelib_precedence", "read_minelib_values", "read_reference_pits", "write_table"]

_ENGINES = ("auto", "pyarrow", "polars", "pandas")
_BINARY = {".parquet": "parquet", ".pq": "parquet", ".feather": "feather", ".arrow": "feather", ".ipc": "feather"}
_DELIMITERS = ("\t", ";", "|", ",")
_SAMPLE_BYTES = 1 << 16
_SAMPLE_LINES = 1000
_BOM = chr(0xFEFF)
# Missing-value tokens, the same for every engine (pandas' defaults)
_NA = ("", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA",
       "NULL", "NaN", "None", "n/a", "nan", "null")  # fmt: skip
_COMMA_NUMBER = re.compile(r"^[+-]?\d+,\d+(?:[eE][+-]?\d+)?$")
_DOT_NUMBER = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _check_engine(engine: str) -> None:
    if engine not in _ENGINES:
        raise ValueError(f"engine must be one of {_ENGINES}, got {engine!r}")


def _pick_engine(engine: str, exclude: tuple[str, ...] = ()) -> str:
    _check_engine(engine)
    if engine != "auto":
        if find_spec(engine) is None:
            raise ImportError(f"engine {engine!r} is not installed: pip install {engine}")
        return engine
    for candidate in ("pyarrow", "polars", "pandas"):
        if candidate not in exclude and find_spec(candidate) is not None:
            return candidate
    raise ImportError("read_blocks needs pandas, pyarrow or polars: pip install pandas pyarrow")


def _from_table(table: Any, columns: Sequence[str] | None) -> dict[str, NDArray[Any]]:
    if isinstance(table, dict):
        names = list(table) if columns is None else list(columns)
        return {c: np.asarray(table[c]) for c in names}
    library = type(table).__module__.split(".")[0]
    if library == "pyarrow":
        names = table.column_names if columns is None else list(columns)
        return {c: table.column(c).to_numpy() for c in names}
    if library in ("pandas", "polars"):
        names = list(table.columns) if columns is None else list(columns)
        return {str(c): table[c].to_numpy() for c in names}
    raise TypeError(
        f"unsupported source {type(table).__name__}: pass a path, a pandas/polars DataFrame, "
        f"a pyarrow Table or a dict of columns"
    )


# Delimited text: layout detection


@dataclass(frozen=True, slots=True)
class _Layout:
    sep: str | None  # None = whitespace (no quoting)
    names: list[str]
    output: list[str]  # names returned by default (without an empty trailing column)
    skip: int  # lines before the first data row (header included)
    decimal: str
    single_space: bool  # whitespace made of single spaces only: every engine can read it
    quoted_head: bool  # a skipped header line holds a quote (older polars honour it even without quoting)
    has_data: bool  # the sample holds data rows


def _is_number(token: str, decimal: str = ".") -> bool:
    try:
        float(token.replace(",", ".") if decimal == "," else token)
    except ValueError:
        return False
    return token.strip().lower() not in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity")


def _fields(line: str, sep: str | None) -> list[str]:
    if sep is None:
        return line.split()
    return [f.strip() for f in next(csv.reader([line], delimiter=sep))]


def _sample_lines(path: Path, encoding: str) -> list[str]:
    with path.open("rb") as f:
        sample = f.read(_SAMPLE_BYTES)
    if sample[:2] == b"\x1f\x8b":
        raise ValueError(f"{path.name} is gzip-compressed: decompress it first")
    if b"\0" in sample:
        raise ValueError(f"{path.name} is not a text file: read_blocks reads delimited text, Parquet and Feather/Arrow")
    if len(sample) == _SAMPLE_BYTES and b"\n" in sample:
        sample = sample[: sample.rindex(b"\n")]  # drop the possibly cut last line
    try:
        text = sample.decode(encoding)
    except UnicodeDecodeError:
        raise ValueError(f"{path.name} is not {encoding} text: pass encoding= (e.g. 'latin-1')") from None
    lines = text.removeprefix(_BOM).splitlines()[:_SAMPLE_LINES]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or not lines[0].strip():
        raise ValueError(f"{path.name} is empty or starts with a blank line")
    return lines


def _gslib_columns(lines: list[str]) -> int:
    """Number of variables if the sample is GSLIB (title, k [nx ny nz], k names, data rows), else 0."""
    if len(lines) < 4 or all(_is_number(t) for t in lines[0].split()):
        return 0
    head = lines[1].split()
    if not head or not all(t.isdigit() for t in head):
        return 0
    k = int(head[0])
    if k < 1 or len(lines) < k + 3:
        return 0
    if any(not name.strip() or _is_number(name.strip()) for name in lines[2 : k + 2]):
        return 0
    row = lines[k + 2].split()
    return k if len(row) == k and all(_is_number(t) for t in row) else 0


def _detect_sep(path: Path, lines: list[str]) -> str | None:
    """The delimiter that splits every sampled line into the same number (> 1) of fields (quotes respected)."""
    rows = [line for line in lines if line.strip()]
    widths = {len(line.split()) for line in rows}
    for d in _DELIMITERS:
        counts = {len(_fields(line, d)) for line in rows}
        if len(counts) == 1 and counts.pop() > 1:
            # "0 0 0 1,5": whitespace with a decimal comma, not comma-separated ("IX, IY" is comma-separated)
            spaced = any(" " in f for line in rows for f in _fields(line, d))
            if d == "," and spaced and len(widths) == 1 and widths != {1}:
                tokens = [t for line in (rows[1:] or rows) for t in line.split()]  # the data, not the header
                if all(_is_number(t.replace(",", ".")) for t in tokens) and any(_COMMA_NUMBER.match(t) for t in tokens):
                    return None
            return d
    if len(widths) == 1 and (widths != {1} or not any(d in line for line in rows for d in _DELIMITERS)):
        return None  # whitespace, or a single column
    raise ValueError(f"could not detect the separator of {path.name} (rows differ in length): pass sep=")


def _detect_decimal(path: Path, data: list[list[str]], sep: str | None) -> str:
    if sep == ",":
        return "."
    tokens = [t for row in data for t in row]
    comma = any(_COMMA_NUMBER.match(t) for t in tokens)
    if comma and any(_DOT_NUMBER.match(t) for t in tokens):
        raise ValueError(f"{path.name} mixes decimal points and decimal commas: pass decimal='.' or ','")
    return "," if comma else "."


def _layout(
    path: Path,
    sep: str | None,
    header: bool | None,
    names: Sequence[str] | None,
    decimal: str | None,
    encoding: str,
) -> _Layout:
    lines = _sample_lines(path, encoding)
    k = _gslib_columns(lines) if sep is None and header is None else 0
    if k:
        delimiter, skip = None, k + 2
        file_names = [line.strip() for line in lines[2 : k + 2]]
        data = [line.split() for line in lines[skip:] if line.strip()]
        dec = decimal or "."
    else:
        delimiter = _detect_sep(path, lines) if sep is None else (None if sep == " " else sep)
        first = _fields(lines[0], delimiter)
        if header is None:  # a header has names and no numbers
            header = any(first) and not any(_is_number(t) for t in first)
        skip = int(header)
        data = [_fields(line, delimiter) for line in lines[skip:] if line.strip()]
        dec = decimal or _detect_decimal(path, data, delimiter)
        width = len(first) if header or not data else len(data[0])
        file_names = [t or str(i) for i, t in enumerate(first)] if header else [str(i) for i in range(width)]
    if names is not None:
        if len(names) != len(file_names):
            raise ValueError(f"names has {len(names)} entries but {path.name} has {len(file_names)} columns")
        file_names = [str(n) for n in names]
    if len(set(file_names)) != len(file_names):
        raise ValueError(f"repeated column names in {path.name}: {file_names}; pass names=")
    trailing = delimiter is not None and bool(data) and all(row[-1] == "" for row in data)  # "1,2,\n"
    output = file_names[:-1] if trailing and names is None else file_names
    single = delimiter is None and all(
        line == line.strip() and "  " not in line and "\t" not in line for line in lines[skip:] if line.strip()
    )
    quoted_head = any('"' in line for line in lines[:skip])
    return _Layout(delimiter, file_names, output, skip, dec, single, quoted_head, bool(data))


# Engines


def _read_text_pyarrow(path: str, lay: _Layout, columns: list[str], encoding: str) -> Any:
    import pyarrow.csv as pc

    return pc.read_csv(
        path,
        read_options=pc.ReadOptions(use_threads=True, skip_rows=lay.skip, column_names=lay.names, encoding=encoding),
        parse_options=pc.ParseOptions(
            delimiter=lay.sep or " ", quote_char=False if lay.sep is None else '"', ignore_empty_lines=True
        ),
        convert_options=pc.ConvertOptions(
            include_columns=columns, decimal_point=lay.decimal, null_values=list(_NA), strings_can_be_null=True
        ),
    )


def _polars_skips_lines() -> bool:
    """polars >= 1.x has ``skip_lines``, which skips lines without parsing quotes."""
    import inspect

    import polars as pl

    return "skip_lines" in inspect.signature(pl.read_csv).parameters


def _read_text_polars(path: str, lay: _Layout, columns: list[str], encoding: str) -> Any:
    import polars as pl

    skip: dict[str, Any] = {"skip_lines" if _polars_skips_lines() else "skip_rows": lay.skip}

    def read(infer: int | None) -> Any:
        return pl.read_csv(
            path,
            has_header=False,
            separator=lay.sep or " ",
            quote_char=None if lay.sep is None else '"',
            **skip,
            decimal_comma=lay.decimal == ",",
            null_values=list(_NA),
            infer_schema_length=infer,
        )

    try:  # every column is read, so that rows with extra fields raise instead of being cut
        df = read(100)
    except pl.exceptions.NoDataError:
        if lay.has_data:
            raise
        return pl.DataFrame({name: pl.Series(name, [], dtype=pl.String) for name in columns})  # header only
    except pl.exceptions.ComputeError:  # a column whose type changes after the first rows: infer from all
        df = read(None)
    if df.width != len(lay.names):
        raise ValueError(f"expected {len(lay.names)} columns, polars found {df.width}")
    df.columns = lay.names
    df = df.filter(~pl.all_horizontal(pl.all().is_null()))  # polars keeps blank lines as empty rows
    fixed = []
    for name in columns:
        column = df[name]
        if column.dtype == pl.String:  # padded numbers (" 2.5") stay as text in polars
            stripped = column.str.strip_chars()
            for target in (pl.Int64, pl.Float64):
                text = stripped.str.replace(",", ".") if lay.decimal == "," and target == pl.Float64 else stripped
                try:
                    column = text.cast(target, strict=True)
                    break
                except pl.exceptions.InvalidOperationError:
                    continue
        fixed.append(column.alias(name))
    return pl.DataFrame(fixed)


def _read_text_pandas(path: str, lay: _Layout, columns: list[str], encoding: str) -> Any:
    import pandas as pd

    # every column is read: with usecols, pandas silently drops the extra fields of a longer row
    return pd.read_csv(
        path,
        sep=lay.sep if lay.sep is not None else (" " if lay.single_space else r"\s+"),
        header=None,
        names=lay.names,
        skiprows=lay.skip,
        decimal=lay.decimal,
        encoding=encoding,
        quoting=csv.QUOTE_NONE if lay.sep is None else csv.QUOTE_MINIMAL,
        skipinitialspace=lay.sep is not None,
        float_precision="round_trip",  # the same floats as pyarrow and polars
        low_memory=False,  # one type per column (no chunk-wise mixed types)
    )


def _read_binary(engine: str, path: str, kind: str, columns: list[str] | None) -> Any:
    if engine == "pyarrow":
        if kind == "parquet":
            import pyarrow.parquet as pq

            return pq.read_table(path, columns=columns, use_threads=True)
        import pyarrow.feather as pf

        return pf.read_table(path, columns=columns)
    if engine == "polars":
        import polars as pl

        return pl.read_parquet(path, columns=columns) if kind == "parquet" else pl.read_ipc(path, columns=columns)
    import pandas as pd

    return pd.read_parquet(path, columns=columns) if kind == "parquet" else pd.read_feather(path, columns=columns)


_TEXT_READERS: dict[str, Callable[[str, _Layout, list[str], str], Any]] = {
    "pyarrow": _read_text_pyarrow,
    "polars": _read_text_polars,
    "pandas": _read_text_pandas,
}


def read_blocks(
    source: str | Path | Any,
    columns: Sequence[str] | None = None,
    engine: str = "auto",
    *,
    sep: str | None = None,
    header: bool | None = None,
    names: Sequence[str] | None = None,
    decimal: str | None = None,
    encoding: str = "utf-8",
) -> dict[str, NDArray[Any]]:
    """Read a block model as ``{column: numpy array}``.

    Parameters
    ----------
    source : a path or a table already in memory (pandas or polars DataFrame, pyarrow Table, or
        dict of columns). ``.parquet``/``.pq`` and ``.feather``/``.arrow``/``.ipc`` are binary;
        any other extension (``.csv``, ``.txt``, ``.dat``, ``.blocks``, ``.gslib``, ...) is read as
        delimited text.
    columns : columns to read (``None`` = all). Parquet/Feather only read those from disk.
    engine : ``"auto"`` (pyarrow > polars > pandas, the fastest installed), ``"pyarrow"``,
        ``"polars"`` or ``"pandas"``. All of them return the same numbers.
    sep : text separator, detected when ``None`` (tab, ``;``, ``|``, ``,`` or whitespace);
        ``" "`` means any whitespace.
    header : whether the first line holds the column names; detected when ``None`` (a line with
        names and no numbers). GSLIB headers are detected too.
    names : column names, replacing the header if there is one. Without header or names, columns
        are named by position: ``"0"``, ``"1"``, ...
    decimal : ``"."`` or ``","``; detected when ``None`` (pass it when commas separate thousands).
    encoding : text encoding (default UTF-8).

    Ambiguous text layouts raise a ``ValueError`` that asks for ``sep``, ``header`` or ``decimal``.
    Text separated by irregular whitespace (runs of spaces, tabs, aligned columns) needs pandas.

    Examples
    --------
    >>> read_blocks({"IX": [0, 1], "VALUE": [5, -1]}, ["VALUE"])["VALUE"].tolist()
    [5, -1]
    """
    _check_engine(engine)
    if isinstance(columns, str):
        raise TypeError(f"columns must be a list of names, got the string {columns!r}")
    cols = list(dict.fromkeys(columns)) if columns is not None else None
    if not isinstance(source, (str, Path)):
        return _from_table(source, cols)
    path = Path(source)
    kind = _BINARY.get(path.suffix.lower())
    if kind is not None:
        return _from_table(_read_binary(_pick_engine(engine), str(path), kind, cols), cols)
    if sep is not None and len(sep) != 1:
        raise ValueError(f"sep must be one character (' ' = any whitespace), got {sep!r}")
    if decimal not in (None, ".", ","):
        raise ValueError(f"decimal must be '.' or ',', got {decimal!r}")
    lay = _layout(path, sep, header, names, decimal, encoding)
    wanted = cols if cols is not None else lay.output
    missing = [c for c in wanted if c not in lay.names]
    if missing:
        raise ValueError(f"columns {missing} not found in {path.name}; available: {lay.output}")
    utf8 = encoding.lower().replace("-", "").replace("_", "") in ("utf8", "utf8sig")
    if lay.sep is None and not lay.single_space:
        if engine in ("pyarrow", "polars"):
            raise ValueError(f"{path.name} is separated by irregular whitespace: use engine='pandas'")
        if find_spec("pandas") is None:
            raise ImportError(f"{path.name} is separated by irregular whitespace and needs pandas: pip install pandas")
        engine = "pandas"
    if engine == "polars" and not utf8:
        raise ValueError(f"polars only reads UTF-8 text: use engine='pyarrow' or 'pandas' for {encoding!r}")
    no_polars = not utf8 or (lay.quoted_head and find_spec("polars") is not None and not _polars_skips_lines())
    if engine == "polars" and no_polars:
        raise ValueError(f"the header of {path.name} has a quote that this polars version misreads: upgrade polars")
    chosen = _pick_engine(engine, exclude=("polars",) if no_polars else ())
    return _from_table(_TEXT_READERS[chosen](str(path), lay, wanted, encoding), wanted)


# MineLib


def _parse_numbers(tokens: Sequence[str | bytes], dtype: type, name: str) -> NDArray[Any]:
    try:
        return np.array(tokens, dtype=dtype)
    except (ValueError, OverflowError):
        raise ValueError(f"{name}: non-numeric or out-of-range data") from None


def read_reference_pits(
    path: str | Path,
    table: Any,
    *,
    indices: tuple[str, str, str] = ("IX", "IY", "IZ"),
    index_base: int = 1,
    column: str = "UPT_PIT",
    encoding: str = "latin-1",
) -> NDArray[np.float64]:
    """Nested pits exported by another program, per block of ``table``: the factor of the first pit that
    holds the block (``column``, in percent, divided by 100), 0 outside.

    The file has a layout common to block-model exports: it lists the blocks of the pits with their
    centres (``XC, YC, ZC``), the model origin
    (``XMORIG, YMORIG, ZMORIG``) and the block size (``XINC, YINC, ZINC``); they are turned into the
    indices of ``table`` (``index_base``: the index of the first cell, often 1). Every block of the file
    must be in ``table``. Use the result with ``NestedPits.compare``.
    """
    d = read_blocks(path, encoding=encoding)
    need = ["XC", "YC", "ZC", "XMORIG", "YMORIG", "ZMORIG", "XINC", "YINC", "ZINC", column]
    missing = [c for c in need if c not in d]
    if missing:
        raise ValueError(
            f"{path}: columns {missing} not found; it must list XC, YC, ZC, the origin, the block size and {column}"
        )
    cells = []
    for axis in "XYZ":
        c = (np.asarray(d[f"{axis}C"], float) - np.asarray(d[f"{axis}MORIG"], float)) / np.asarray(
            d[f"{axis}INC"], float
        )
        cells.append(np.rint(c - 0.5).astype(np.int64) + index_base)
    for name in indices:
        if name not in table:
            raise ValueError(f"index column {name!r} not found in the table; available: {list(table)}")
    mine = [np.asarray(table[n]).astype(np.int64) for n in indices]
    low = [min(int(np.min(a)), int(np.min(b))) for a, b in zip(mine, cells, strict=True)]
    size = [max(int(np.max(a)), int(np.max(b))) - lo + 1 for a, b, lo in zip(mine, cells, low, strict=True)]
    key_table = np.ravel_multi_index(tuple(a - lo for a, lo in zip(mine, low, strict=True)), size)
    key_file = np.ravel_multi_index(tuple(a - lo for a, lo in zip(cells, low, strict=True)), size)
    order = np.argsort(key_table)
    pos = np.searchsorted(key_table[order], key_file)
    found = (pos < key_table.size) & (key_table[order][np.minimum(pos, key_table.size - 1)] == key_file)
    if not found.all():
        raise ValueError(
            f"{int((~found).sum())} blocks of {path} are not in the table: check index_base "
            f"(the index of the first cell, often 1)"
        )
    factor = np.asarray(d[column], dtype=np.float64)
    if not np.isfinite(factor).all() or (factor <= 0).any():
        raise ValueError(f"{column} must be the pit factor in percent (> 0) for every block of {path}")
    out = np.zeros(key_table.size)
    out[order[pos]] = factor / 100.0
    return out


def read_minelib_precedence(path: str | Path) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """MineLib ``.prec`` file -> arcs ``(src, dst)`` for ``ExplicitPit``: block ``src[k]`` requires ``dst[k]``.

    Each line is ``id n p1 ... pn``: block ``id`` requires its ``n`` predecessors.

    Examples
    --------
    >>> import os, tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = os.path.join(d, "toy.prec")
    ...     _ = open(p, "w").write("0 2 1 2\\n1 0\\n2 0\\n")
    ...     src, dst = read_minelib_precedence(p)
    >>> src.tolist(), dst.tolist()
    ([0, 0], [1, 2])
    """
    path = Path(path)
    raw = path.read_bytes().removeprefix(_BOM.encode())
    if not raw.strip():
        return np.empty(0, np.int64), np.empty(0, np.int64)
    data = np.frombuffer(raw, dtype=np.uint8)
    space = (data == 32) | (data == 9) | (data == 10) | (data == 13) | (data == 11) | (data == 12)
    starts = np.flatnonzero(~space & np.concatenate(([True], space[:-1])))  # first byte of every token
    line = np.searchsorted(np.flatnonzero(data == 10), starts)  # line of every token
    del data, space
    tokens = _parse_numbers(raw.split(), np.int64, path.name)
    if tokens.size != starts.size:
        raise ValueError(f"{path.name}: non-numeric data")
    first = np.flatnonzero(np.diff(line, prepend=-1))  # first token of every non-empty line
    count = np.diff(np.append(first, tokens.size))
    ok = (count >= 2) & (tokens[np.minimum(first + 1, tokens.size - 1)] == count - 2)
    if not ok.all():
        raise ValueError(f"{path.name}, line {int(line[first[np.argmin(ok)]]) + 1}: expected 'id n p1 ... pn'")
    position = np.arange(tokens.size) - np.repeat(first, count)
    src = np.repeat(tokens[first], count - 2)
    dst = tokens[position >= 2]
    if src.size and min(int(src.min()), int(dst.min())) < 0:
        raise ValueError(f"{path.name}: negative block id")
    return src, dst


_SECTION = re.compile(r"^[ \t]*OBJECTIVE_FUNCTION[ \t]*:?", re.IGNORECASE | re.MULTILINE)


def read_minelib_values(path: str | Path) -> NDArray[np.float64]:
    """MineLib ``.upit`` file -> block values (float64, by block id); scale them with ``scale_values``.

    Examples
    --------
    >>> import os, tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = os.path.join(d, "toy.upit")
    ...     _ = open(p, "w").write("NAME: toy\\nTYPE: UPIT\\nNBLOCKS: 2\\nOBJECTIVE_FUNCTION:\\n1 -2.5\\n0 4\\nEOF\\n")
    ...     values = read_minelib_values(p)
    >>> values.tolist()
    [4.0, -2.5]
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace").removeprefix(_BOM)
    section = _SECTION.search(text)
    if section is None:
        raise ValueError(f"{path.name}: no OBJECTIVE_FUNCTION section (is it a MineLib .upit file?)")
    head, body = text[: section.start()], text[section.end() :]
    kind = re.search(r"^\s*TYPE\s*:\s*(\S+)", head, re.IGNORECASE | re.MULTILINE)
    if kind and kind.group(1).upper() != "UPIT":
        raise ValueError(f"{path.name}: TYPE is {kind.group(1)}, expected UPIT")
    body = re.split(r"^\s*EOF\b", body, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)[0]
    pairs = _parse_numbers(body.split(), np.float64, path.name)
    if pairs.size % 2:
        raise ValueError(f"{path.name}: expected 'id value' lines")
    ids, values = pairs[0::2], pairs[1::2]
    match = re.search(r"^\s*NBLOCKS\s*:\s*(\d+)", head, re.IGNORECASE | re.MULTILINE)
    n = int(match.group(1)) if match else ids.size
    if ids.size != n or not np.array_equal(np.sort(ids), np.arange(n)):
        raise ValueError(f"{path.name}: block ids must be the integers 0..{n - 1}, each once")
    out = np.empty(n)
    out[ids.astype(np.int64)] = values
    return out


# Writing


def write_table(table: Mapping[str, ArrayLike], path: str | Path, engine: str = "auto") -> Path:
    """Write ``{column: array}`` as a CSV file (comma, header); returns the path.

    Uses the fastest library installed (pyarrow > polars > pandas), or Python's ``csv`` module.
    """
    _check_engine(engine)
    path = Path(path)
    columns = {str(k): np.asarray(v) for k, v in table.items()}
    if len({v.shape for v in columns.values()}) > 1 or any(v.ndim != 1 for v in columns.values()):
        raise ValueError("every column must be a 1-D array of the same length")
    chosen = engine
    if engine == "auto":
        chosen = next((e for e in ("pyarrow", "polars", "pandas") if find_spec(e) is not None), "csv")
    elif find_spec(engine) is None:
        raise ImportError(f"engine {engine!r} is not installed: pip install {engine}")
    if chosen == "pyarrow":
        import pyarrow as pa
        import pyarrow.csv as pc

        pc.write_csv(pa.table(columns), str(path))
    elif chosen == "polars":
        import polars as pl

        pl.DataFrame(columns).write_csv(path)
    elif chosen == "pandas":
        import pandas as pd

        pd.DataFrame(columns).to_csv(path, index=False)
    else:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(zip(*(v.tolist() for v in columns.values()), strict=True))
    return path
