"""End-to-end correctness against references that never use PyMineFlow's graph (see _oracle.py).

A: present blocks require the present blocks of their own domain's cone (physical semantics).
B: empty cells are zero-value blocks with the same slope (the model PyMineFlow defines).
Brute force: every closed subset of a tiny model.
"""

import warnings

import numpy as np
import pytest

from pymineflow import PitModel, scale_values, ultimate_pit

from _oracle import brute_force, grid_cone_value, present_cone_value, topography_model

SIZE = (5.0, 8.0, 3.0)  # different block sizes in x, y and z
SECTORS = [(0, 40), (90, 55), (180, 45), (270, 60)]


def as_sectors(spec):
    return [(0.0, float(spec))] if np.isscalar(spec) else spec


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("topography", [False, True])
def test_one_domain_constant_slope(seed, topography):
    ix, iy, iz, v = topography_model(seed, topography=topography)
    r = ultimate_pit(v, ix=ix, iy=iy, iz=iz, block_size=SIZE, slopes=45.0)
    assert r.value == present_cone_value(r.grid, v, np.zeros(v.size, int), [45.0])


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("topography", [False, True])
def test_one_domain_sectors(seed, topography):
    ix, iy, iz, v = topography_model(seed, topography=topography)
    r = ultimate_pit(v, ix=ix, iy=iy, iz=iz, block_size=SIZE, slopes=SECTORS)
    assert r.value == grid_cone_value(r.grid, v, SECTORS)
    assert r.value <= present_cone_value(r.grid, v, np.zeros(v.size, int), [SECTORS])


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("specs", [[40.0, 65.0], [SECTORS, 35.0], [30.0, SECTORS, 70.0]])
def test_several_domains(seed, specs):
    ix, iy, iz, v = topography_model(seed)
    domains = np.random.default_rng(100 + seed).integers(0, len(specs), v.size)
    model = PitModel(ix=ix, iy=iy, iz=iz, block_size=SIZE, domains=domains,
                     domain_slopes=[as_sectors(s) for s in specs])  # fmt: skip
    assert model.pattern == "full"
    assert model.solve(v).value == present_cone_value(model.grid, v, domains, specs)


def test_continuous_block_slopes():
    ix, iy, iz, v = topography_model(7)
    angles = 35.0 + 30.0 * (ix / ix.max()) + 0.3  # continuous west-east gradient
    model = PitModel(ix=ix, iy=iy, iz=iz, block_size=SIZE, block_slopes=angles, slope_step=5.0)
    assert model.solve(v).value == present_cone_value(model.grid, v, model.domains, model.domain_slopes)


@pytest.mark.parametrize("seed", range(3))
def test_with_and_without_full_pattern(seed):
    ix, iy, iz, v = topography_model(seed)
    kwargs = {"ix": ix, "iy": iy, "iz": iz, "block_size": SIZE}
    a = ultimate_pit(v, slopes=45.0, full_pattern=False, **kwargs)
    b = ultimate_pit(v, slopes=45.0, full_pattern=True, **kwargs)
    assert a.value == b.value == present_cone_value(a.grid, v, np.zeros(v.size, int), [45.0])
    domains = (ix % 2).astype(int)
    with pytest.warns(RuntimeWarning, match="larger than valid"):
        c = ultimate_pit(v, domains=domains, domain_slopes=[[(0, 30)], [(0, 70)]], full_pattern=False, **kwargs)
    assert c.value >= present_cone_value(c.grid, v, domains, [30.0, 70.0])  # documented relaxation


@pytest.mark.parametrize("seed", range(3))
def test_noisy_coordinates_match_indices(seed):
    ix, iy, iz, v = topography_model(seed)
    r = np.random.default_rng(seed)
    x = np.round(351234.567 + ix * SIZE[0] + r.uniform(-1e-3, 1e-3, v.size), 3)
    y = np.round(-8765.4321 + iy * SIZE[1] + r.uniform(-1e-3, 1e-3, v.size), 3)
    z = np.round(1200.25 + iz * SIZE[2] + r.uniform(-1e-3, 1e-3, v.size), 3)
    p = r.permutation(v.size)
    a = ultimate_pit(v, ix=ix + 40, iy=iy - 7, iz=iz, block_size=SIZE, slopes=SECTORS)
    b = ultimate_pit(v[p], x=x[p], y=y[p], z=z[p], block_size=SIZE, slopes=SECTORS)
    assert a.value == b.value
    assert (b.mask == a.mask[p]).all()


# ---------- edge cases ----------
@pytest.mark.parametrize("value", [7, -7, 0])
def test_single_block(value):
    r = ultimate_pit(np.array([value]), ix=[3], iy=[4], iz=[5], block_size=SIZE, slopes=45.0)
    assert r.value == max(value, 0)
    assert r.n_blocks == (1 if value > 0 else 0)


def test_nx_equal_to_one():
    ix, iy, iz, v = topography_model(3, nx=1, ny=12, nz=6)
    r = ultimate_pit(v, ix=ix, iy=iy, iz=iz, block_size=SIZE, slopes=SECTORS)
    assert r.value == grid_cone_value(r.grid, v, SECTORS)


def test_all_negative_all_positive_and_zeros():
    ix, iy, iz, v = topography_model(5)
    kwargs = {"ix": ix, "iy": iy, "iz": iz, "block_size": SIZE, "slopes": 45.0}
    assert ultimate_pit(-np.abs(v) - 1, **kwargs).n_blocks == 0
    positive = np.abs(v) + 1
    r = ultimate_pit(positive, **kwargs)
    assert r.n_blocks == v.size
    assert r.value == int(positive.sum())
    assert ultimate_pit(np.zeros(v.size, np.int64), **kwargs).value == 0


def test_values_near_the_int64_limit():
    kwargs = {"ix": [0, 0, 1], "iy": [0, 0, 0], "iz": [0, 1, 1], "block_size": (1, 1, 1), "slopes": 45.0}
    assert ultimate_pit(np.array([2**61, -(2**61) + 5, 3]), **kwargs).value == 8
    with pytest.raises(OverflowError, match=r"sum of \|values\| exceeds int64"):
        ultimate_pit(np.array([2**62, -(2**62), 1]), **kwargs)


def test_explicit_errors():
    ix, iy, iz, v = topography_model(1)
    with pytest.raises(ValueError, match="duplicate blocks"):
        ultimate_pit(np.append(v, 1), ix=np.append(ix, ix[0]), iy=np.append(iy, iy[0]), iz=np.append(iz, iz[0]),
                     block_size=SIZE, slopes=45.0)  # fmt: skip
    x, y, z = ix * SIZE[0] + 0.0, iy * SIZE[1] + 0.0, iz * SIZE[2] + 0.0
    x_nan = x.copy()
    x_nan[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ultimate_pit(v, x=x_nan, y=y, z=z, block_size=SIZE, slopes=45.0)
    with pytest.raises(ValueError, match="NaN"):
        scale_values(np.array([1.0, np.nan]))
    a = np.radians(12)
    with pytest.raises(ValueError, match="regular grid"):  # rotated model
        ultimate_pit(v, x=x * np.cos(a) - y * np.sin(a), y=x * np.sin(a) + y * np.cos(a), z=z, block_size=SIZE)
    shifted = x.copy()
    shifted[:5] += SIZE[0] / 2  # sub-blocks
    with pytest.raises(ValueError, match="regular grid"):
        ultimate_pit(v, x=shifted, y=y, z=z, block_size=SIZE)


# ---------- n_benches ----------
def test_n_benches_none_small_and_larger_than_height():
    ix, iy, iz, v = topography_model(2, nz=8)
    kwargs = {"ix": ix, "iy": iy, "iz": iz, "block_size": (10, 10, 10), "slopes": 45.0}
    full = PitModel(**kwargs)
    r_full = full.solve(v)
    assert full.n_benches == full.grid.nz - 1
    assert r_full.value == present_cone_value(full.grid, v, np.zeros(v.size, int), [45.0])
    assert ultimate_pit(v, n_benches=1, **kwargs).value >= r_full.value  # fewer constraints
    assert ultimate_pit(v, n_benches=500, **kwargs).value == r_full.value


# ---------- warm start ----------
def test_solve_with_values_equals_solving_from_scratch():
    ix, iy, iz, v = topography_model(4)
    domains = (iy % 3).astype(int)
    kwargs = {"ix": ix, "iy": iy, "iz": iz, "block_size": SIZE, "domains": domains,
              "domain_slopes": [[(0, 40)], SECTORS, [(0, 60)]]}  # fmt: skip
    model = PitModel(**kwargs)
    for f in (1.0, 0.4, 2.5, 0.0, 1.3):
        vf = np.where(v > 0, np.rint(v * f), v).astype(np.int64)
        a, b = model.solve(vf), PitModel(**kwargs).solve(vf)
        assert a.value == b.value
        assert (a.mask == b.mask).all()


# ---------- brute force on tiny models ----------
@pytest.mark.parametrize("seed", range(12))
def test_brute_force(seed):
    r = np.random.default_rng(seed)
    gx, gy, gz = (a.ravel() for a in np.meshgrid(np.arange(3), np.arange(2), np.arange(3), indexing="ij"))
    keep = r.random(gx.size) < 0.6  # random empty cells
    keep[0] = True
    ix, iy, iz = gx[keep], gy[keep], gz[keep]
    v = r.integers(-20, 25, ix.size).astype(np.int64)
    size = (1.0, 1.5, 1.2)
    a = ultimate_pit(v, ix=ix, iy=iy, iz=iz, block_size=size, slopes=45.0, min_density=0)
    assert a.value == brute_force(ix, iy, iz, size, v, np.zeros(v.size, int), [45.0])
    domains = r.integers(0, 2, v.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = ultimate_pit(v, ix=ix, iy=iy, iz=iz, block_size=size, domains=domains,
                         domain_slopes=[[(0, 35)], [(0, 65)]], min_density=0)  # fmt: skip
    assert b.value == brute_force(ix, iy, iz, size, v, domains, [35.0, 65.0])
