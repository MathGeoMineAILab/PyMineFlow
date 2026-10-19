"""High-level API: ultimate_pit / PitModel / scale_values."""

import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from pymineflow import Pit, PitModel, domains_to_cells, indices_to_grid, scale_values, ultimate_pit

from _oracle import MCL, grid_cone_value, present_cone_arcs, present_cone_value

# model cut below a topography (no holes)
rng = np.random.default_rng(11)
NX, NY, NZ = 14, 12, 7
surface = rng.integers(3, NZ, size=(NX, NY))
gx, gy, gz = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")
below = gz <= surface[:, :, None]
IX, IY, IZ = gx[below], gy[below], gz[below]
N = IX.size
V = -rng.integers(1, 8, N)
rich = rng.choice(N, size=N // 20, replace=False)
V[rich] = rng.integers(60, 400, rich.size)
V = V.astype(np.int64)
SECTORS = [(0, 40), (90, 50), (180, 45), (270, 55)]
SLOPES = {"1:5": None, "45": 45.0, "sectors": SECTORS}


def coords(noise, decimals, x0=1234567.891, y0=-98765.4321, z0=-250.5):
    r = np.random.default_rng(5)
    x = x0 + IX * MCL[0] + r.normal(0, noise, N)
    y = y0 + IY * MCL[1] + r.normal(0, noise, N)
    z = z0 + IZ * MCL[2] + r.normal(0, noise, N)
    return np.round(x, decimals), np.round(y, decimals), np.round(z, decimals)


@pytest.mark.parametrize("name", list(SLOPES))
def test_indices_and_coordinates_give_the_same_pit(name):
    slopes = SLOPES[name]
    a = ultimate_pit(V, ix=IX + 100, iy=IY - 5, iz=IZ, block_size=MCL, slopes=slopes)
    x, y, z = coords(noise=1e-3, decimals=3)
    p = np.random.default_rng(9).permutation(N)
    b = ultimate_pit(V[p], x=x[p], y=y[p], z=z[p], block_size=MCL, slopes=slopes)
    assert a.value == b.value
    assert a.n_blocks == b.n_blocks
    assert (b.mask == a.mask[p]).all()  # mask in the order of the input rows
    assert a.mask.dtype == bool
    assert a.mask.size == N
    assert a.pattern == ("1:5" if slopes is None else "minsearch")
    assert 0 < a.n_blocks < N


def test_depth_coordinates():
    a = ultimate_pit(V, ix=IX, iy=IY, iz=IZ, block_size=MCL, slopes=45.0)
    b = ultimate_pit(V, ix=IX, iy=IY, iz=-IZ, block_size=MCL, slopes=45.0, z_up=False)
    x, y, z = coords(noise=0, decimals=6)
    c = ultimate_pit(V, x=x, y=y, z=-z, block_size=MCL, slopes=45.0, z_up=False)
    assert (a.mask == b.mask).all()
    assert (a.mask == c.mask).all()


@pytest.mark.parametrize("name", ["45", "sectors"])
def test_against_exact_cone(name):
    r = ultimate_pit(V, ix=IX, iy=IY, iz=IZ, block_size=MCL, slopes=SLOPES[name])
    assert r.value == grid_cone_value(r.grid, V, SLOPES[name])


def test_result_is_immutable():
    r = ultimate_pit(V, ix=IX, iy=IY, iz=IZ, block_size=MCL, slopes=45.0)
    with pytest.raises(AttributeError):
        r.value = 0


# ---------- n_benches ----------
def test_n_benches_defaults_to_the_full_height():
    kwargs = {"ix": IX, "iy": IY, "iz": IZ, "block_size": MCL, "slopes": 45.0}
    model = PitModel(**kwargs)
    assert model.n_benches == model.grid.nz - 1
    assert PitModel(n_benches=2, **kwargs).n_benches == 2
    assert PitModel(n_benches=10_000, **kwargs).n_benches == model.grid.nz - 1
    assert ultimate_pit(V, n_benches=10_000, **kwargs).value == ultimate_pit(V, **kwargs).value


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "8"])
def test_invalid_n_benches(bad):
    with pytest.raises(ValueError, match=r"n_benches must be None or an integer >= 1, got"):
        PitModel(ix=IX, iy=IY, iz=IZ, block_size=MCL, slopes=45.0, n_benches=bad)


def test_oversized_full_pattern_raises():
    gx, gy, gz = (a.ravel() for a in np.meshgrid(np.arange(60), np.arange(60), np.arange(80), indexing="ij"))
    with pytest.raises(ValueError, match=r"full_pattern with n_benches=79 .* use a smaller n_benches"):
        ultimate_pit(np.zeros(gx.size, np.int64), ix=gx, iy=gy, iz=gz, block_size=(10, 10, 10), slopes=45.0,
                     full_pattern=True)  # fmt: skip


def test_oversized_minsearch_window_raises():
    # blocks thin in plan and tall (1 x 1 x 50), flat slope, 30 benches: the MinSearch search window
    # does not depend on the grid and would need tens of GB
    gx, gy, gz = (a.ravel() for a in np.meshgrid(np.arange(2), np.arange(2), np.arange(31), indexing="ij"))
    zeros = np.zeros(gx.size, np.int64)
    for n_benches in (None, 4):
        with pytest.raises(ValueError, match="use a smaller n_benches"):
            ultimate_pit(zeros, ix=gx, iy=gy, iz=gz, block_size=(1, 1, 50), slopes=20.0, n_benches=n_benches)
    assert ultimate_pit(zeros, ix=gx, iy=gy, iz=gz, block_size=(10, 10, 10), slopes=45.0).value == 0


# ---------- validation ----------
def test_exactly_one_input_mode():
    x, y, z = coords(noise=0, decimals=6)
    with pytest.raises(ValueError, match="exactly one input mode"):
        ultimate_pit(V, x=x, y=y, z=z, ix=IX, iy=IY, iz=IZ, block_size=MCL)
    with pytest.raises(ValueError, match="exactly one input mode"):
        ultimate_pit(V, block_size=MCL)
    with pytest.raises(ValueError, match="missing components"):
        ultimate_pit(V, x=x, y=y, block_size=MCL)
    with pytest.raises(ValueError, match=r"block_size=.* is required"):
        ultimate_pit(V, ix=IX, iy=IY, iz=IZ, block_size=None)
    with pytest.raises(TypeError):
        ultimate_pit(V, ix=IX, iy=IY, iz=IZ)  # block_size is a required keyword


def test_float_values_raise_type_error():
    with pytest.raises(TypeError, match="scale_values"):
        ultimate_pit(V.astype(float), ix=IX, iy=IY, iz=IZ, block_size=MCL)


def test_values_length():
    with pytest.raises(ValueError, match=rf"one entry per block \({N}\)"):
        ultimate_pit(V[:-1], ix=IX, iy=IY, iz=IZ, block_size=MCL)


def test_sparse_grid_warns():
    ix, iy, iz = np.array([0, 40, 0]), np.array([0, 40, 40]), np.array([0, 9, 9])
    with pytest.warns(RuntimeWarning, match="ExplicitPit"):
        ultimate_pit(np.array([5, -1, -1]), ix=ix, iy=iy, iz=iz, block_size=(1, 1, 1))


def test_full_pattern_needs_slopes():
    with pytest.raises(ValueError, match="full_pattern needs slopes"):
        PitModel(ix=IX, iy=IY, iz=IZ, block_size=MCL, full_pattern=True)


# ---------- scale_values and int64 overflow ----------
def test_scale_integer_valued_floats():
    scaled, factor = scale_values(np.array([1.0, -2.0, 3.0000000001]))
    assert factor == 1
    assert scaled.dtype == np.int64
    assert scaled.tolist() == [1, -2, 3]


def test_scale_decimals_and_value_in_that_scale():
    real = V / 100.0  # cents
    scaled, factor = scale_values(real)
    assert factor == 10**6  # the largest factor that fits
    assert (scaled == V * 10**4).all()
    r = ultimate_pit(scaled, ix=IX, iy=IY, iz=IZ, block_size=MCL, slopes=45.0)
    assert r.value == ultimate_pit(V, ix=IX, iy=IY, iz=IZ, block_size=MCL, slopes=45.0).value * 10**4
    assert abs(r.value / factor - real[r.mask].sum()) < 1e-6 * r.n_blocks


def test_scale_keeps_small_values():
    scaled, factor = scale_values([0.0123, -0.0049, 0.0151])
    assert (scaled.tolist(), factor) == ([12300, -4900, 15100], 10**6)
    scaled, factor = scale_values(np.array([4e12, -1.5]))  # a smaller factor keeps the sums in int64
    assert factor == 10**3
    assert scaled.tolist() == [4 * 10**15, -1500]


def test_scale_keeps_integers():
    scaled, factor = scale_values(V.astype(np.int32))
    assert factor == 1
    assert scaled.dtype == np.int64
    assert (scaled == V).all()


@pytest.mark.parametrize(
    ("values", "error", "match"),
    [
        (np.full(2000, 2.0**52), OverflowError, r"sum of \|values\| exceeds int64"),
        (np.array([2**61, 2**61], dtype=np.int64), OverflowError, r"sum of \|values\| exceeds int64"),
        (np.array([2**62, -(2**62)], dtype=np.int64), OverflowError, r"sum of \|values\| exceeds int64"),
        (np.array([1e17, 1.5]), ValueError, "too large to scale exactly"),
        (np.array([1.5, np.nan]), ValueError, "NaN"),
        (np.zeros((2, 2)), ValueError, "1-D"),
    ],
)
def test_scale_errors(values, error, match):
    with pytest.raises(error, match=match):
        scale_values(values)


def test_sum_of_absolute_values_above_int64_raises():
    v = np.zeros(N, dtype=np.int64)
    v[0], v[1] = 2**62, -(2**62)
    with pytest.raises(OverflowError, match=r"sum of \|values\| exceeds int64"):
        ultimate_pit(v, ix=IX, iy=IY, iz=IZ, block_size=MCL, slopes=45.0)
    v[0], v[1] = 2**61, -(2**61)  # close to the limit but valid
    assert ultimate_pit(v, ix=IX, iy=IY, iz=IZ, block_size=MCL, slopes=45.0).value in (0, 2**61)


# ---------- domains per row ----------
DOMAINS = (IX >= NX // 2).astype(np.int64)
DOMAIN_SLOPES = [[(0, 35)], SECTORS]


@pytest.mark.parametrize("seed", range(3))
def test_topography_two_domains_against_exact_cone(seed):
    v = np.random.default_rng(seed).permutation(V)
    r = ultimate_pit(v, ix=IX, iy=IY, iz=IZ, block_size=MCL, domains=DOMAINS, domain_slopes=DOMAIN_SLOPES)
    assert r.pattern == "full"  # automatic: domains with different slopes
    assert r.value == present_cone_value(r.grid, v, DOMAINS, DOMAIN_SLOPES)


def test_column_air_is_conservative_and_envelope_is_exact():
    """Alternating 30/70 stripes (boundaries everywhere): 'envelope' gives exactly each block's cone;
    'column' may require extra blocks at boundaries (valid pit, possibly smaller)."""
    domains = (IY % 2).astype(np.int64)
    slopes = [[(0, 30)], [(0, 70)]]
    kwargs = {"ix": IX, "iy": IY, "iz": IZ, "block_size": MCL, "domains": domains, "domain_slopes": slopes}
    grid = indices_to_grid(IX, IY, IZ, MCL)
    for seed in range(3):
        v = np.random.default_rng(seed).permutation(V)
        exact = present_cone_value(grid, v, domains, slopes)
        assert ultimate_pit(v, **kwargs).value == exact
        assert ultimate_pit(v, air_domain="column", **kwargs).value <= exact
    # targeted case: a block whose 'column' closure contains a block outside its exact cone
    i, extra = _block_with_extra_requirement(grid, domains, slopes)
    v = -np.ones(N, dtype=np.int64)
    v[i], v[extra] = 10_000, -1_000_000
    exact = present_cone_value(grid, v, domains, slopes)
    assert ultimate_pit(v, **kwargs).value == exact > 0
    assert ultimate_pit(v, air_domain="column", **kwargs).value < exact


def _block_with_extra_requirement(grid, domains, slopes):
    """(row i, row j): j is in the closure of i with air_domain='column' but not in i's exact cone."""
    from _oracle import arcs

    cells = domains_to_cells(grid, domains, "column")
    pit = Pit(grid.nx, grid.ny, grid.nz, np.zeros(grid.n_cells, np.int64), pattern="minsearch",
              n_benches=grid.nz, block_size=grid.block_size, domains=cells, domain_slopes=slopes,
              full_pattern=True)  # fmt: skip
    src, dst = [], []
    for d in range(2):
        s, t = arcs(grid.nx, grid.ny, grid.nz, [tuple(o) for o in pit.offsets(d)])
        src.append(s[cells[s] == d])
        dst.append(t[cells[s] == d])
    column = csr_matrix((np.ones(sum(x.size for x in src)), (np.concatenate(src), np.concatenate(dst))),
                        shape=(grid.n_cells,) * 2)  # fmt: skip
    exact_src, exact_dst = present_cone_arcs(grid, domains, slopes)
    exact = csr_matrix((np.ones(exact_src.size), (exact_src, exact_dst)), shape=(grid.n_cells,) * 2)
    row_of = {c: k for k, c in enumerate(grid.linear.tolist())}
    for cell in grid.linear:
        extra = set(breadth_first_order(column, cell, return_predecessors=False)) - set(
            breadth_first_order(exact, cell, return_predecessors=False)
        )
        extra = [c for c in extra if c in row_of]
        if extra:
            return row_of[int(cell)], row_of[extra[0]]
    raise AssertionError("expected a block with an extra requirement under air_domain='column'")


def test_constraint_chain_through_the_air_with_limited_n_benches():
    """A rich block R in a valley and a costly block H on the hillside, exactly on R's 45-degree cone
    but 3 benches higher. With n_benches=1 the constraint R -> H only arrives in a chain, and the chain
    passes through air cells: if the air had no precedences R would be mined alone."""
    ix, iy, iz = np.array([4, 1]), np.array([0, 0]), np.array([0, 3])
    kwargs = {"ix": ix, "iy": iy, "iz": iz, "block_size": (1, 1, 1), "slopes": 45.0, "min_density": 0}
    v = np.array([100, -1000])
    assert ultimate_pit(v, n_benches=1, **kwargs).value == 0
    assert ultimate_pit(v, n_benches=3, **kwargs).value == 0
    assert ultimate_pit(np.array([2000, -1000]), n_benches=1, **kwargs).value == 1000


def test_automatic_pattern_and_warning():
    kwargs = {"ix": IX, "iy": IY, "iz": IZ, "block_size": MCL, "domains": DOMAINS, "n_benches": 4}
    assert PitModel(domain_slopes=[[(0, 45)], [(0, 45), (90, 45)]], **kwargs).pattern == "minsearch"
    with pytest.warns(RuntimeWarning, match="larger than valid"):
        model = PitModel(domain_slopes=DOMAIN_SLOPES, full_pattern=False, **kwargs)
    assert model.pattern == "minsearch"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert PitModel(domain_slopes=DOMAIN_SLOPES, **kwargs).solve(V).pattern == "full"


def test_continuous_block_slopes():
    continuous = np.where(DOMAINS == 0, 34.8, 50.3)  # -> 35 and 50 degrees
    a = ultimate_pit(V, ix=IX, iy=IY, iz=IZ, block_size=MCL, block_slopes=continuous)
    b = ultimate_pit(V, ix=IX, iy=IY, iz=IZ, block_size=MCL, domains=DOMAINS, domain_slopes=[[(0, 35)], [(0, 50)]])
    assert a.value == b.value
    assert (a.mask == b.mask).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"slopes": 45, "domains": DOMAINS, "domain_slopes": DOMAIN_SLOPES}, "pass only one of"),
        ({"domain_slopes": DOMAIN_SLOPES}, "pass only one of"),
        ({"domains": DOMAINS}, "domains needs domain_slopes"),
        ({"domains": DOMAINS + 5, "domain_slopes": DOMAIN_SLOPES}, r"domains must be in \[0, 2\)"),
        ({"domains": DOMAINS.astype(float), "domain_slopes": DOMAIN_SLOPES}, "one integer per block"),
        ({"domains": DOMAINS, "domain_slopes": DOMAIN_SLOPES, "air_domain": 7}, r"air_domain 7 out of range"),
        ({"domains": DOMAINS, "domain_slopes": DOMAIN_SLOPES, "air_domain": "other"}, "air_domain must be"),
        ({"block_slopes": np.full(3, 45.0)}, rf"one row per block \({N}\)"),
    ],
)
def test_domain_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PitModel(ix=IX, iy=IY, iz=IZ, block_size=MCL, **kwargs)


# ---------- nested pits (warm start) ----------
FACTORS = [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5]


def scenario(f):
    """Price factor: scales only the positive values (revenue)."""
    return np.where(V > 0, np.rint(V * f), V).astype(np.int64)


@pytest.mark.parametrize("slopes", [45.0, SECTORS])
def test_nested_pits_equal_fresh_solves_and_are_nested(slopes):
    kwargs = {"ix": IX, "iy": IY, "iz": IZ, "block_size": MCL, "slopes": slopes}
    results = PitModel(**kwargs).nested_pits([scenario(f) for f in FACTORS])
    previous = np.zeros(N, dtype=bool)
    for f, r in zip(FACTORS, results, strict=True):
        fresh = ultimate_pit(scenario(f), **kwargs)
        assert r.value == fresh.value
        assert (r.mask == fresh.mask).all()
        assert not (previous & ~r.mask).any()  # each pit contains the previous one
        previous = r.mask
    assert results[0].n_blocks < results[-1].n_blocks


def test_solve_in_any_order():
    model = PitModel(ix=IX, iy=IY, iz=IZ, block_size=MCL, slopes=45.0)
    for f in (1.2, 0.3, 0.85, 1.5, 0.5):  # non-monotone order
        fresh = ultimate_pit(scenario(f), ix=IX, iy=IY, iz=IZ, block_size=MCL, slopes=45.0)
        assert model.solve(scenario(f)).value == fresh.value
