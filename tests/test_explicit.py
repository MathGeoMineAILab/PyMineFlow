"""ExplicitPit: (src, dst) arcs and CSR, against scipy and MineFlow's original ExplicitPrecedence."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pymineflow import ExplicitPit
from pymineflow._mineflow import _solve_explicit_precedence

from _oracle import max_closure_value


def random_dag(n, m, seed):
    r = np.random.default_rng(seed)
    src = r.integers(0, n - 1, m)
    dst = np.minimum(n - 1, src + r.integers(1, 25, m))  # i requires j > i: no cycles
    return src.astype(np.int64), dst.astype(np.int64), r.integers(-50, 60, n).astype(np.int64)


def as_csr(src, dst, n):
    g = csr_matrix((np.ones(src.size), (src, dst)), shape=(n, n))  # duplicates are summed: harmless
    return g.indptr.astype(np.int64), g.indices.astype(np.int64)


@pytest.mark.parametrize("seed", range(5))
def test_arcs_and_csr_against_scipy_and_explicit_precedence(seed):
    n = 400
    src, dst, v = random_dag(n, 2000, seed)
    m0, v0 = _solve_explicit_precedence(n, src, dst, v)
    m1, v1, _, _ = ExplicitPit(n, src, dst, v).solve()
    m2, v2, _, _ = ExplicitPit.from_csr(*as_csr(src, dst, n), v).solve()
    assert v0 == v1 == v2 == max_closure_value(v, src, dst)
    assert (m0 == m1).all()
    assert (m0 == m2).all()
    assert v[m1].sum() == v1
    assert np.all(m1[dst][m1[src]])


def test_no_arcs_and_duplicate_arcs():
    v = np.array([5, -3, 2, -1], dtype=np.int64)
    assert ExplicitPit(4, np.array([], np.int64), np.array([], np.int64), v).solve()[1] == 7
    assert ExplicitPit(4, np.array([0, 0, 0]), np.array([1, 1, 3]), v).solve()[1] == 5 - 3 - 1 + 2
    assert ExplicitPit.from_csr(np.array([0, 2, 2, 2, 2]), np.array([1, 3]), v).solve()[1] == 3


@pytest.mark.parametrize("csr", [False, True])
def test_solve_with_values_equals_solving_from_scratch(csr):
    n = 300
    src, dst, v = random_dag(n, 1200, 9)
    pit = ExplicitPit.from_csr(*as_csr(src, dst, n), v) if csr else ExplicitPit(n, src, dst, v)
    pit.solve()
    for v2 in (v * 3 - 7, -np.abs(v), v + 40):
        mask, value, _, _ = pit.solve_with_values(v2)
        fresh_mask, fresh_value, _, _ = ExplicitPit(n, src, dst, v2).solve()
        assert value == fresh_value == max_closure_value(v2, src, dst)
        assert (mask == fresh_mask).all()


@pytest.mark.parametrize(
    ("indptr", "indices", "match"),
    [
        ([1, 2, 2], [0, 1], "must start at 0"),
        ([0, 2, 1], [0, 1], "end at len"),
        ([0, 2, 1, 2], [0, 1], "non-decreasing"),
        ([0, 1, 2], [0, 5], r"indices\[1\] = 5 is out of range"),
        ([0], [], "len\\(indptr\\) = n \\+ 1"),
    ],
)
def test_invalid_csr(indptr, indices, match):
    n = max(len(indptr) - 1, 1)
    with pytest.raises(ValueError, match=match):
        ExplicitPit.from_csr(np.array(indptr, np.int64), np.array(indices, np.int64), np.zeros(n, np.int64))


def test_invalid_arcs():
    with pytest.raises(ValueError, match=r"arc 0 \(0 -> 3\) is out of range \[0, 3\)"):
        ExplicitPit(3, np.array([0]), np.array([3]), np.zeros(3, np.int64))
    with pytest.raises(ValueError, match=r"same length, got shapes \(2,\) and \(1,\)"):
        ExplicitPit(3, np.array([0, 1]), np.array([1]), np.zeros(3, np.int64))
    with pytest.raises(ValueError, match="n must be positive, got 0"):
        ExplicitPit(0, np.array([], np.int64), np.array([], np.int64), np.zeros(0, np.int64))


def test_float_values_raise_type_error():
    with pytest.raises(TypeError):
        ExplicitPit(2, np.array([0]), np.array([1]), np.array([1.5, -1.0]))
