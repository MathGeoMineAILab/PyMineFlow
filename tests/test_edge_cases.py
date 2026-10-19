"""Edge cases: inputs and situations that could give wrong results, wrong data or a crash."""

import threading
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

import pymineflow
from pymineflow import ExplicitPit, Pit, PitModel, coords_to_grid, discretize_slopes, scale_values, ultimate_pit
from pymineflow.slopes import normalize_slopes, slopes_differ

from _oracle import cone_offsets, grid_cone_value, slope_pairs


def box(nx, ny, nz):
    """Every cell of an nx * ny * nz grid, as (ix, iy, iz) rows."""
    return [a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")]


# Binding inputs


@pytest.mark.parametrize(
    ("values", "error"),
    [([5.7, -1.9, -1.9, -1.0], TypeError), (np.array([1, 0, 0, 1], bool), TypeError)],
)
def test_non_integer_values_are_not_truncated(values, error):
    with pytest.raises(error, match="must be integers"):
        ExplicitPit(4, [0, 0], [1, 2], values)
    with pytest.raises(error, match="must be integers"):
        Pit(2, 1, 2, values)


def test_non_integer_arcs_and_huge_unsigned_values():
    with pytest.raises(TypeError, match="src must be integers"):
        ExplicitPit(4, [0.9, 0.2], [1.5, 2.5], [5, -1, -1, -1])
    with pytest.raises(OverflowError, match="exceed the int64 range"):
        ExplicitPit(2, [0], [1], np.array([2**64 - 1, 1], np.uint64))


def test_integer_lists_and_other_integer_dtypes():
    assert ExplicitPit(4, [0, 0], [1, 2], [5, -1, -1, -1]).solve()[1] == 3
    assert ExplicitPit(2, [], [], [3, -1]).solve()[1] == 3
    src, dst = np.array([0], np.uint32), np.array([1], np.int32)
    assert ExplicitPit(3, src, dst, np.asfortranarray(np.array([4, -1, 2]))).solve()[1] == 5


def test_one_solve_at_a_time_per_object():
    """Two threads on the same object: the second one gets an error, never a crash."""
    nx, ny, nz = 60, 60, 30
    v = np.random.default_rng(0).integers(-10, 10, nx * ny * nz).astype(np.int64)
    pit = Pit(nx, ny, nz, v, pattern="minsearch", slope_deg=45, n_benches=8)
    reference = pit.solve()[1]
    results, errors = [], []

    def run(fn):
        try:
            results.append(fn()[1])
        except RuntimeError as e:
            errors.append(str(e))

    threads = [threading.Thread(target=run, args=(pit.solve,)) for _ in range(2)]
    threads.append(threading.Thread(target=run, args=(lambda: pit.solve_with_values(v),)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results
    assert all(r == reference for r in results)
    assert all("one object per thread" in e for e in errors)
    assert pit.solve()[1] == reference


def test_flat_slopes_are_rejected_before_allocating():
    v = np.zeros(18, np.int64)
    with pytest.raises(ValueError, match=r"\[0.001, 90\) degrees"):
        Pit(3, 3, 2, v, pattern="minsearch", slope_deg=1e-7, n_benches=1)
    with pytest.raises(ValueError, match="window over 2 GB"):
        Pit(3, 3, 2, v, pattern="minsearch", slope_deg=0.01, n_benches=60, block_size=(1, 1, 10))
    with pytest.raises(ValueError, match="take no slope_deg"):
        Pit(3, 3, 2, v, pattern="1:5", slope_deg=10)


def test_azimuth_just_below_zero_is_north():
    z = np.zeros(7 * 7 * 4, np.int64)
    with pytest.raises(ValueError, match="repeated azimuth"):
        Pit(7, 7, 4, z, pattern="minsearch", sectors=[(0, 45), (-1e-14, 45)])
    with pytest.raises(ValueError, match="repeated azimuth"):
        normalize_slopes([(0, 45), (-1e-14, 45)])
    a = Pit(7, 7, 4, z, pattern="minsearch", sectors=[(-1e-14, 45), (90, 45)], n_benches=3).offsets()
    b = Pit(7, 7, 4, z, pattern="minsearch", slope_deg=45, n_benches=3).offsets()
    assert sorted(map(tuple, a)) == sorted(map(tuple, b))


# Slopes that vary by azimuth near the model edges


SPEC = [(15, 67.6), (60, 53.9), (135, 72.5), (255, 39.5), (270, 69.5), (300, 46.1), (330, 25.5)]


def test_full_pattern_keeps_the_cone_at_the_model_edges():
    """With varying slopes the minimum pattern may need cells outside the model; the full pattern does not."""
    ix, iy, iz = box(11, 4, 7)
    v = np.zeros(ix.size, np.int64)
    rich = (ix == 4) & (iy == 0) & (iz == 0)
    cost = (ix == 0) & (iy == 3) & (iz == 5)  # inside the rich block's cone (azimuth 307, slope ~41 deg)
    v[rich], v[cost] = 100, -1000
    r = ultimate_pit(v, ix=ix, iy=iy, iz=iz, block_size=(10, 10, 10), slopes=SPEC, full_pattern=True)
    assert r.pattern == "full"
    assert r.grid.n_cells == 11 * 4 * 7  # the model's own grid
    assert r.value == 0 == grid_cone_value(r.grid, v, SPEC)


@pytest.mark.parametrize("seed", range(3))
def test_full_pattern_cone_of_every_cell(seed):
    """With one rich cell, the pit holds that cell's whole cone inside the model (full pattern)."""
    r = np.random.default_rng(seed)
    for _ in range(40):
        size = [(10, 10, 10), (5, 10, 15), (25, 25, 20), (7.62, 7.62, 6.096)][r.integers(4)]
        az = np.sort(r.choice(np.arange(0, 360, 15), int(r.integers(2, 9)), replace=False))
        spec = [(float(a), float(r.uniform(25, 75))) for a in az]
        nx, ny, nz = (int(r.integers(3, 12)), int(r.integers(3, 12)), int(r.integers(3, 8)))
        ix, iy, iz = box(nx, ny, nz)
        model = PitModel(ix=ix, iy=iy, iz=iz, block_size=size, slopes=spec, full_pattern=True)
        cone = np.array(cone_offsets(nx, ny, nz, size, slope_pairs(spec))).reshape(-1, 3)
        for c in r.choice(ix.size, 6, replace=False):
            v = -np.ones(ix.size, np.int64)
            v[c] = 10**9
            mask = model.solve(v).mask
            tx, ty, tz = ix[c] + cone[:, 0], iy[c] + cone[:, 1], iz[c] + cone[:, 2]
            inside = (tx >= 0) & (tx < nx) & (ty >= 0) & (ty < ny) & (tz < nz)
            assert mask[(tx[inside] * ny + ty[inside]) * nz + tz[inside]].all(), (size, spec, (nx, ny, nz))


# High-level inputs


def test_unsigned_inputs_do_not_wrap():
    with pytest.raises(OverflowError, match="unsigned values exceed int64"):
        ultimate_pit(np.array([5, 2**64 - 1], np.uint64), ix=[0, 1], iy=[0, 0], iz=[0, 0], block_size=(1, 1, 1))
    iz = np.array([1, 1, 0], np.uint32)  # depth
    r = ultimate_pit([5, -1, -1], ix=[0, 1, 1], iy=[0, 0, 0], iz=iz, block_size=(10, 10, 10), slopes=45, z_up=False)
    assert (
        r.value
        == ultimate_pit([5, -1, -1], ix=[0, 1, 1], iy=[0, 0, 0], iz=[0, 0, 1], block_size=(10, 10, 10), slopes=45).value
    )


def test_missing_data_sentinels_in_coordinates():
    with pytest.raises(ValueError, match="missing-data sentinels"):
        coords_to_grid([5.0, 15.0, 1e30], [0, 0, 0], [0, 0, 0], block_size=(10, 10, 10))
    with pytest.raises(ValueError, match="missing-data sentinels"):
        ultimate_pit([1, 1], ix=[0, 10**15], iy=[0, 0], iz=[0, 0], block_size=(1, 1, 1))
    with pytest.raises(ValueError, match="tol must be in"):
        coords_to_grid([5.0], [0.0], [0.0], block_size=(10, 10, 10), tol=0.7)


def test_scale_values_keeps_precision():
    values, factor = scale_values([0.0123, -0.0049, 0.0151])
    assert values.tolist() == [12300, -4900, 15100]
    assert factor == 10**6


def test_narrow_slope_differences_are_detected():
    assert slopes_differ([[(0, 45)], [(0, 45), (10, 45), (10.2, 60), (10.4, 45)]])
    assert not slopes_differ([[(0, 45)], [(0, 45), (180, 45)]])


def test_discretize_rounding():
    assert discretize_slopes([47.3], step=0.1, rounding="down")[1] == [[(0.0, 47.3)]]
    assert [s[0][1] for s in discretize_slopes([44.5, 45.5, 46.5])[1]] == [45.0, 46.0, 47.0]


@pytest.mark.parametrize("bad", [True, [45], (0, 45), [(0, 45, 1)]])
def test_bad_slope_definitions(bad):
    with pytest.raises(TypeError, match="slopes must be a number or a list"):
        normalize_slopes(bad)


def test_version_is_the_same_everywhere():
    assert pymineflow.__version__ == version("pymineflow")
    citation = Path(__file__).resolve().parents[1] / "CITATION.cff"
    if citation.exists():  # in the repository and the sdist
        assert f"version: {pymineflow.__version__}" in citation.read_text(encoding="utf-8").splitlines()
