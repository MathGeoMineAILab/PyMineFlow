"""MineLib files (.blocks, .prec, .upit): readers, ExplicitPit and the slope-based model agree."""

import numpy as np
import pytest

from pymineflow import (
    ExplicitPit,
    Pit,
    read_blocks,
    read_minelib_precedence,
    read_minelib_values,
    scale_values,
    ultimate_pit,
)

from _oracle import max_closure_value


def write_prec(path, n, src, dst):
    preds = [[] for _ in range(n)]
    for s, d in zip(src.tolist(), dst.tolist(), strict=True):
        preds[s].append(d)
    path.write_text("".join(f"{i} {len(p)}{''.join(f' {j}' for j in p)}\n" for i, p in enumerate(preds)))


def write_upit(path, values, order):
    rows = "".join(f"{i} {float(values[i])!r}\n" for i in order)
    path.write_text(f"NAME: toy\nTYPE: UPIT\nNBLOCKS: {len(values)}\nOBJECTIVE_FUNCTION:\n{rows}EOF\n")


@pytest.mark.parametrize("seed", range(3))
def test_random_instance_against_max_flow(tmp_path, seed):
    r = np.random.default_rng(seed)
    n = 300
    src = r.integers(0, n - 1, 1500)
    dst = np.minimum(n - 1, src + r.integers(1, 20, src.size))  # acyclic
    values = np.round(r.normal(0, 50, n), 3)
    write_prec(tmp_path / "toy.prec", n, src, dst)
    write_upit(tmp_path / "toy.upit", values, r.permutation(n))  # ids in any order
    s, d = read_minelib_precedence(tmp_path / "toy.prec")
    v = read_minelib_values(tmp_path / "toy.upit")
    assert np.array_equal(v, values)
    assert sorted(zip(s.tolist(), d.tolist(), strict=True)) == sorted(zip(src.tolist(), dst.tolist(), strict=True))
    iv = np.rint(v * 1000).astype(np.int64)  # 3 decimals; the max-flow reference needs int32 capacities
    mask, value, _, _ = ExplicitPit(n, s, d, iv).solve()
    assert value == max_closure_value(iv, s, d)
    assert np.all(mask[d][mask[s]])


def test_slope_model_matches_its_minelib_export(tmp_path):
    """A full grid exported as MineLib arcs (pattern offsets) gives the same pit as ultimate_pit."""
    nx, ny, nz = 9, 8, 6
    r = np.random.default_rng(7)
    values = r.integers(-30, 25, nx * ny * nz).astype(np.int64)
    size = (10.0, 10.0, 8.0)
    pit = Pit(nx, ny, nz, values, pattern="minsearch", slope_deg=45, n_benches=nz - 1, block_size=size)
    iz, iy, ix = np.unravel_index(np.arange(values.size), (nz, ny, nx))
    src, dst = [], []
    for ox, oy, oz in pit.offsets():
        jx, jy, jz = ix + ox, iy + oy, iz + oz
        ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny) & (jz < nz)
        src.append(np.flatnonzero(ok))
        dst.append(((jz * ny + jy) * nx + jx)[ok])
    write_prec(tmp_path / "grid.prec", values.size, np.concatenate(src), np.concatenate(dst))
    write_upit(tmp_path / "grid.upit", values.astype(float), range(values.size))
    blocks = "".join(f"{i} {ix[i]} {iy[i]} {iz[i]} 1.0\n" for i in range(values.size))
    (tmp_path / "grid.blocks").write_text(blocks)

    b = read_blocks(tmp_path / "grid.blocks", names=["id", "x", "y", "z", "ton"])
    v, _ = scale_values(read_minelib_values(tmp_path / "grid.upit"))
    s, d = read_minelib_precedence(tmp_path / "grid.prec")
    explicit = ExplicitPit(v.size, s, d, v).solve()
    geometric = ultimate_pit(v, ix=b["x"], iy=b["y"], iz=b["z"], block_size=size, slopes=45)
    assert explicit[1] == geometric.value == pit.solve()[1]
    assert np.array_equal(explicit[0], geometric.mask)


def test_errors(tmp_path):
    (tmp_path / "bad.prec").write_text("0 2 1\n1 0\n")
    with pytest.raises(ValueError, match=r"line 1: expected 'id n p1 ... pn'"):
        read_minelib_precedence(tmp_path / "bad.prec")
    (tmp_path / "text.prec").write_text("0 1 x\n")
    with pytest.raises(ValueError, match="non-numeric"):
        read_minelib_precedence(tmp_path / "text.prec")
    (tmp_path / "empty.prec").write_text("\n")
    assert read_minelib_precedence(tmp_path / "empty.prec")[0].size == 0
    (tmp_path / "bad.upit").write_text("NAME: x\nNBLOCKS: 2\n0 1\n1 2\n")
    with pytest.raises(ValueError, match="no OBJECTIVE_FUNCTION"):
        read_minelib_values(tmp_path / "bad.upit")
    (tmp_path / "dup.upit").write_text("NBLOCKS: 2\nOBJECTIVE_FUNCTION:\n0 1\n0 2\nEOF\n")
    with pytest.raises(ValueError, match="each once"):
        read_minelib_values(tmp_path / "dup.upit")
    (tmp_path / "cpit.upit").write_text("TYPE: CPIT\nNBLOCKS: 1\nOBJECTIVE_FUNCTION:\n0 1 2\nEOF\n")
    with pytest.raises(ValueError, match="TYPE is CPIT"):
        read_minelib_values(tmp_path / "cpit.upit")
    (tmp_path / "frac.upit").write_text("NBLOCKS: 2\nOBJECTIVE_FUNCTION:\n0 1\n0.5 2\nEOF\n")
    with pytest.raises(ValueError, match="integers"):
        read_minelib_values(tmp_path / "frac.upit")


def test_upit_and_prec_variants(tmp_path):
    text = "\ufeffNAME: OBJECTIVE_FUNCTION_test\ntype:upit\nnblocks: 2\nobjective_function\n1 -2.5\n0 4e1\neof\n"
    (tmp_path / "v.upit").write_text(text, encoding="utf-8")
    assert read_minelib_values(tmp_path / "v.upit").tolist() == [40.0, -2.5]
    (tmp_path / "v.prec").write_bytes(b"\xef\xbb\xbf0 1 1\r\n\r\n1\t0  \r\n")
    src, dst = read_minelib_precedence(tmp_path / "v.prec")
    assert (src.tolist(), dst.tolist()) == ([0], [1])
    (tmp_path / "big.prec").write_text("0 1 99999999999999999999\n")
    with pytest.raises(ValueError, match="out-of-range"):
        read_minelib_precedence(tmp_path / "big.prec")
