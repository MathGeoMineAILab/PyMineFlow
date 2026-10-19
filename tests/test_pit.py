"""Low-level Pit: block sizes, sectors, slope domains and the full pattern, against independent references."""

import numpy as np
import pytest

from pymineflow import ExplicitPit, Pit

from _oracle import (
    MCL,
    arcs,
    cone_offsets,
    max_closure_value,
    random_values,
    slope_pairs,
    within,
)

CASES = {
    "constant_45": (MCL, {"slope_deg": 45.0}, 45.0),
    "anisotropic_40": ((10.0, 12.0, 8.0), {"slope_deg": 40.0}, 40.0),
    "sectors": (MCL, {"sectors": [(0, 40), (90, 50), (180, 45), (270, 55)]}, [(0, 40), (90, 50), (180, 45), (270, 55)]),
}


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("case", list(CASES))
def test_against_exact_cone_max_flow(case, seed):
    size, kwargs, spec = CASES[case]
    nx, ny, nz = 12, 10, 6
    v = random_values(nx * ny * nz, np.random.default_rng(seed))
    # n_benches >= nz - 1: the pattern covers the whole height and its closure must equal the cone
    pit = Pit(nx, ny, nz, v, pattern="minsearch", n_benches=nz, block_size=size, **kwargs)
    mask, value, _, _ = pit.solve()
    src, dst = arcs(nx, ny, nz, cone_offsets(nx, ny, nz, size, slope_pairs(spec)))
    assert value == max_closure_value(v, src, dst)
    assert v[mask].sum() == value
    assert np.all(mask[dst][mask[src]])  # a valid closure of the exact cone


@pytest.mark.parametrize("case", list(CASES))
def test_same_graph_as_explicit_pit_with_its_offsets(case):
    """The same graph built two ways checks the binding's x -> y -> z indexing."""
    size, kwargs, _ = CASES[case]
    nx, ny, nz = 15, 11, 7
    v = random_values(nx * ny * nz, np.random.default_rng(7))
    pit = Pit(nx, ny, nz, v, pattern="minsearch", n_benches=4, block_size=size, **kwargs)
    src, dst = arcs(nx, ny, nz, [tuple(o) for o in pit.offsets()])
    m1, v1, _, _ = pit.solve()
    m2, v2, _, _ = ExplicitPit(nx * ny * nz, src, dst, v).solve()
    assert v1 == v2
    assert (m1 == m2).all()


def test_equal_sectors_give_the_constant_pattern():
    v = np.zeros(10 * 10 * 6, dtype=np.int64)
    a = Pit(10, 10, 6, v, pattern="minsearch", block_size=MCL, slope_deg=45.0)
    b = Pit(10, 10, 6, v, pattern="minsearch", block_size=MCL, sectors=[(0, 45), (90, 45), (180, 45), (270, 45)])
    assert np.array_equal(a.offsets(), b.offsets())


def test_azimuth_patch_regression_west_axis():
    """Regression for the [PyMineFlow] patch in SlopeDefinition::Get (see PATCHES.md).

    With n_benches=1 the pattern is exactly {(x, y, 1): Within}. Due west (dx<0, dy=0) Within passes
    azimuth -90; the original MineFlow interpolated with that non-normalized azimuth and applied 43
    degrees instead of 55 (reach 1.0 m instead of 0.7 m with 0.1 m blocks).
    """
    sectors = [(0, 45), (90, 50), (180, 40), (270, 55)]
    s = 0.1
    pit = Pit(3, 3, 2, np.zeros(18, np.int64), pattern="minsearch", n_benches=1, block_size=(s, s, 1.0),
              sectors=sectors)  # fmt: skip
    pattern = {tuple(map(int, o)) for o in pit.offsets() if o[2] == 1}
    assert max(-o[0] for o in pattern if o[1] == 0) == 7  # floor(1 / tan(55) / 0.1); 10 with the bug
    pairs = slope_pairs(sectors)
    r = 15
    expected = {(x, y, 1) for x in range(-r, r + 1) for y in range(-r, r + 1) if within(x * s, y * s, 1.0, pairs)}
    assert pattern == expected


@pytest.mark.parametrize(("flat_azimuth", "axis", "sign"), [(0, "y", 1), (90, "x", 1), (180, "y", -1), (270, "x", -1)])
def test_azimuth_convention(flat_azimuth, axis, sign):
    """Azimuth 0 = +y (north), 90 = +x (east): the pit opens wider towards the flat sector."""
    n, nz = 21, 8
    c = n // 2
    v = -np.ones(n * n * nz, dtype=np.int64)
    v[c * n + c] = 10**6  # rich block at the bottom centre
    sectors = [(a, 30 if a == flat_azimuth else 70) for a in (0, 90, 180, 270)]
    pit = Pit(n, n, nz, v, pattern="minsearch", n_benches=nz, block_size=(10, 10, 10), sectors=sectors)
    top = pit.solve()[0].reshape(nz, n, n)[-1]  # [y, x] of the top bench
    ys, xs = np.nonzero(top)
    d = (xs if axis == "x" else ys) - c
    assert (d.max() > -d.min()) if sign > 0 else (-d.min() > d.max())


def test_one_five_default_pattern():
    v = random_values(8 * 8 * 5, np.random.default_rng(3))
    assert Pit(8, 8, 5, v).offsets().shape == (5, 3)


# ---------- slope domains ----------
NX, NY, NZ = 12, 10, 6
N = NX * NY * NZ
DOMAIN_SLOPES = [[(0, 40)], [(0, 50), (90, 35), (180, 55), (270, 45)]]
DOMAINS = (np.arange(N) % NX >= NX // 2).astype(np.int64)  # x = index % nx


def domain_arcs(domains, offsets_per_domain):
    src_all, dst_all = [], []
    for d, offsets in enumerate(offsets_per_domain):
        src, dst = arcs(NX, NY, NZ, offsets)
        keep = domains[src] == d
        src_all.append(src[keep])
        dst_all.append(dst[keep])
    return np.concatenate(src_all), np.concatenate(dst_all)


def exact_domain_arcs(domains, domain_slopes):
    return domain_arcs(domains, [cone_offsets(NX, NY, NZ, MCL, slope_pairs(s)) for s in domain_slopes])


def domain_pit(v, full, domains=DOMAINS, domain_slopes=DOMAIN_SLOPES, n_benches=NZ):
    return Pit(NX, NY, NZ, v, pattern="minsearch", n_benches=n_benches, block_size=MCL, domains=domains,
               domain_slopes=domain_slopes, full_pattern=full)  # fmt: skip


@pytest.mark.parametrize("seed", range(3))
def test_two_domains_full_pattern_against_exact_cone(seed):
    v = random_values(N, np.random.default_rng(seed))
    pit = domain_pit(v, full=True)
    assert pit.n_domains == 2
    mask, value, _, _ = pit.solve()
    src, dst = exact_domain_arcs(DOMAINS, DOMAIN_SLOPES)
    assert value == max_closure_value(v, src, dst)
    assert np.all(mask[dst][mask[src]])


@pytest.mark.parametrize("full", [False, True])
def test_domains_same_graph_as_explicit_pit(full):
    v = random_values(N, np.random.default_rng(4))
    pit = domain_pit(v, full=full, n_benches=3)
    src, dst = domain_arcs(DOMAINS, [[tuple(o) for o in pit.offsets(d)] for d in range(pit.n_domains)])
    m1, v1, _, _ = pit.solve()
    m2, v2, _, _ = ExplicitPit(N, src, dst, v).solve()
    assert v1 == v2
    assert (m1 == m2).all()


def test_minimum_pattern_loses_constraints_between_domains_and_full_does_not():
    """Alternating 30/70 stripes: the closure of the MINIMUM pattern passes through blocks of the other
    domain and loses constraints (a relaxation, value >= exact); the full pattern is exact."""
    domains = ((np.arange(N) // NX) % NY % 2).astype(np.int64)
    slopes = [[(0, 30)], [(0, 70)]]
    src, dst = exact_domain_arcs(domains, slopes)
    strictly_larger = 0
    for seed in range(5):
        v = random_values(N, np.random.default_rng(seed))
        exact = max_closure_value(v, src, dst)
        assert domain_pit(v, True, domains, slopes).solve()[1] == exact
        minimum = domain_pit(v, False, domains, slopes).solve()[1]
        assert minimum >= exact
        strictly_larger += minimum > exact
    assert strictly_larger > 0


def test_full_and_minimum_agree_for_a_single_constant_slope():
    v = random_values(N, np.random.default_rng(8))
    kwargs = {"pattern": "minsearch", "n_benches": NZ, "block_size": MCL, "slope_deg": 45.0}
    a, b = Pit(NX, NY, NZ, v, **kwargs), Pit(NX, NY, NZ, v, full_pattern=True, **kwargs)
    assert a.solve()[1] == b.solve()[1]
    assert len(b.offsets()) > len(a.offsets())


# ---------- errors ----------
V = np.zeros(N, dtype=np.int64)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pattern": "1:5", "block_size": (1, 1, 1)}, "take no slope_deg, block_size"),
        ({"pattern": "1:9", "sectors": [(0, 45)]}, "take no slope_deg, block_size"),
        ({"pattern": "1:5", "full_pattern": True}, "take no slope_deg, block_size"),
        ({"pattern": "minsearch", "slope_deg": 45, "sectors": [(0, 45)]}, "not both"),
        ({"pattern": "minsearch", "slope_deg": 90}, r"\[0.001, 90\) degrees, got 90"),
        ({"pattern": "minsearch", "slope_deg": 0}, r"\[0.001, 90\) degrees, got 0"),
        ({"pattern": "minsearch", "block_size": (1, 0, 1)}, "three positive numbers"),
        ({"pattern": "minsearch", "sectors": [(0, 45), (360, 50)]}, "repeated azimuth"),
        ({"pattern": "minsearch", "sectors": [(np.inf, 45)]}, "must be finite"),
        ({"pattern": "minsearch", "sectors": []}, "must not be empty"),
        ({"pattern": "minsearch", "n_benches": 0}, "n_benches must be >= 1, got 0"),
        ({"pattern": "other"}, "pattern must be '1:5', '1:9' or 'minsearch', got 'other'"),
        ({"pattern": "minsearch", "domains": DOMAINS}, "go together"),
        ({"pattern": "minsearch", "domain_slopes": DOMAIN_SLOPES}, "go together"),
        (
            {"pattern": "minsearch", "domains": DOMAINS, "domain_slopes": DOMAIN_SLOPES, "slope_deg": 45},
            "only domain_slopes",
        ),
        ({"pattern": "minsearch", "domains": DOMAINS, "domain_slopes": []}, "must not be empty"),
        ({"pattern": "minsearch", "domains": DOMAINS, "domain_slopes": [[(0, 45)]]}, r"domain 1 out of range \[0, 1\)"),
        ({"pattern": "minsearch", "domains": DOMAINS[:-1], "domain_slopes": DOMAIN_SLOPES}, "one domain per cell"),
        ({"pattern": "minsearch", "domains": DOMAINS, "domain_slopes": [[(0, 45)], [(0, 95)]]}, "got 95"),
    ],
)
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Pit(NX, NY, NZ, V, **kwargs)


def test_invalid_dimensions():
    with pytest.raises(ValueError, match=r"must be positive, got \(0, 1, 1\)"):
        Pit(0, 1, 1, np.zeros(0, np.int64))


def test_offsets_domain_out_of_range():
    with pytest.raises(IndexError, match=r"domain 2 out of range \[0, 2\)"):
        domain_pit(V, full=False).offsets(2)


def test_float_values_are_not_truncated():
    with pytest.raises(TypeError):
        Pit(2, 1, 1, np.array([1.5, -1.0]))


def test_values_must_be_1d_with_one_per_cell():
    with pytest.raises(ValueError, match="1-D array"):
        Pit(2, 1, 1, np.zeros((2, 1), dtype=np.int64))
    with pytest.raises(ValueError, match="expected 2 values"):
        Pit(2, 1, 1, np.zeros(3, dtype=np.int64))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([2**62, 2**62], "positive values overflows"),
        ([-(2**62), -(2**62)], "negative values overflows"),
        ([np.iinfo(np.int64).min, 0], "negative values overflows"),
        ([2**62, -(2**62)], r"sum of \|values\| exceeds int64"),
    ],
)
def test_int64_overflow(values, match):
    v = np.array(values, dtype=np.int64)
    with pytest.raises(OverflowError, match=match):
        Pit(2, 1, 1, v)
    with pytest.raises(OverflowError, match=match):
        ExplicitPit(2, np.array([0]), np.array([1]), v)
