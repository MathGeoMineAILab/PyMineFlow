"""Coordinates / indices -> regular grid."""

import numpy as np
import pytest

from pymineflow import coords_to_grid, domains_to_cells, empty_cells_below_blocks, indices_to_grid
from pymineflow.grid import cells_to_rows, values_to_cells

rng = np.random.default_rng(0)
# synthetic model cut below a topography, rows in shuffled order
NX, NY, NZ = 60, 70, 30
surface = rng.integers(10, NZ, size=(NX, NY))
gx, gy, gz = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")
below = gz <= surface[:, :, None]
IX, IY, IZ = gx[below], gy[below], gz[below]
order = rng.permutation(IX.size)
IX, IY, IZ = IX[order], IY[order], IZ[order]
SHAPE = (int(IX.max()) + 1, int(IY.max()) + 1, int(IZ.max()) + 1)
SIZE = (7.62, 7.62, 6.096)


def coords(noise=0.0, decimals=None, x0=1234567.891, y0=-98765.4321, z0=-250.5):
    x, y, z = x0 + IX * SIZE[0], y0 + IY * SIZE[1], z0 + IZ * SIZE[2]
    if noise:
        x, y, z = (c + rng.normal(0, noise, c.size) for c in (x, y, z))
    if decimals is not None:
        x, y, z = (np.round(c, decimals) for c in (x, y, z))
    return x, y, z


def check(grid):
    assert (grid.ix == IX).all()
    assert (grid.iy == IY).all()
    assert (grid.iz == IZ).all()
    assert (grid.nx, grid.ny, grid.nz) == SHAPE


def test_estimated_block_size():
    grid = coords_to_grid(*coords())
    check(grid)
    assert np.allclose(grid.block_size, SIZE)


def test_given_block_size():
    check(coords_to_grid(*coords(), block_size=SIZE))


@pytest.mark.parametrize("decimals", [3, 2])
def test_rounded_coordinates(decimals):
    check(coords_to_grid(*coords(decimals=decimals), block_size=SIZE))


def test_noisy_coordinates():
    check(coords_to_grid(*coords(noise=1e-3), block_size=SIZE))


def test_z_down():
    x, y, z = coords()
    check(coords_to_grid(x, y, -z, block_size=SIZE, z_up=False))


def test_values_and_mask_round_trip():
    grid = coords_to_grid(*coords(), block_size=SIZE)
    v = rng.integers(-50, 50, IX.size).astype(np.int64)
    cells = values_to_cells(grid, v)
    assert cells.sum() == v.sum()
    assert (cells[grid.linear] == v).all()
    cell_mask = rng.random(grid.n_cells) < 0.3
    assert (cells_to_rows(grid, cell_mask) == cell_mask[grid.linear]).all()


def test_linear_index_is_int32_when_it_fits():
    assert indices_to_grid(IX, IY, IZ, SIZE).linear.dtype == np.int32


def test_sub_blocks_raise():
    x, y, z = coords()
    x = x.copy()
    x[:10] += SIZE[0] / 2
    with pytest.raises(ValueError, match="regular grid"):
        coords_to_grid(x, y, z, block_size=SIZE)


def test_rotated_model_raises():
    x, y, z = coords()
    a = np.radians(10)
    with pytest.raises(ValueError, match="regular grid"):
        coords_to_grid(x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a), z, block_size=SIZE)


def test_duplicate_coordinates_raise():
    x, y, z = coords()
    x[1], y[1], z[1] = x[0], y[0], z[0]
    with pytest.raises(ValueError, match="duplicate blocks: 1 cells"):
        coords_to_grid(x, y, z, block_size=SIZE)


def test_nan_coordinates_raise():
    x, y, z = coords()
    x[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        coords_to_grid(x, y, z)


def test_coordinate_lengths_must_match():
    with pytest.raises(ValueError, match="same non-zero length"):
        coords_to_grid([0.0, 1.0], [0.0], [0.0])


def test_single_coordinate_axis_needs_block_size():
    with pytest.raises(ValueError, match="single distinct coordinate"):
        coords_to_grid([0.0, 10.0], [5.0, 5.0], [1.0, 1.0])


# ---------- indices + block size ----------
def test_indices():
    grid = indices_to_grid(IX, IY, IZ, SIZE)
    check(grid)
    assert grid.base == (0, 0, 0)
    assert grid.origin is None
    assert grid.block_size == SIZE


def test_indices_and_coordinates_give_the_same_grid():
    a = indices_to_grid(IX, IY, IZ, SIZE)
    b = coords_to_grid(*coords(), block_size=SIZE)
    assert (a.linear == b.linear).all()
    assert a.n_cells == b.n_cells


def test_indices_with_offset_and_negative_values():
    grid = indices_to_grid(IX + 1000, IY - 50, IZ - 7, SIZE)
    check(grid)
    assert grid.base == (1000, -50, -7)


def test_integer_valued_float_indices():
    check(indices_to_grid(IX.astype(float), IY.astype(float), IZ.astype(float), SIZE))


def test_non_integer_indices_raise():
    with pytest.raises(ValueError, match="non-integer"):
        indices_to_grid(IX + 0.5, IY, IZ, SIZE)


def test_non_numeric_indices_raise():
    with pytest.raises(ValueError, match="must be numeric"):
        indices_to_grid(IX.astype(str), IY, IZ, SIZE)


def test_nan_indices_raise():
    a = IX.astype(float)
    a[0] = np.nan
    with pytest.raises(ValueError, match="ix contains NaN"):
        indices_to_grid(a, IY, IZ, SIZE)


def test_index_lengths_must_match():
    with pytest.raises(ValueError, match="same length"):
        indices_to_grid(IX, IY[:-1], IZ, SIZE)


@pytest.mark.parametrize("size", [None, (1, 1), (1, 0, 1), (1, -1, 1), (1, np.nan, 1)])
def test_indices_need_a_valid_block_size(size):
    with pytest.raises(ValueError, match="block_size must be three positive numbers"):
        indices_to_grid(IX, IY, IZ, size)


def test_duplicate_indices_raise():
    a, b, c = IX.copy(), IY.copy(), IZ.copy()
    a[1], b[1], c[1] = a[0], b[0], c[0]
    with pytest.raises(ValueError, match="duplicate blocks"):
        indices_to_grid(a, b, c, SIZE)


def test_outlier_indices_raise():
    a, b, c = IX.astype(np.int64), IY.astype(np.int64), IZ.astype(np.int64)  # int32 by default on NumPy 1.x/Windows
    a[0] = b[0] = c[0] = 10**12
    with pytest.raises(ValueError, match="too large for int64"):
        indices_to_grid(a, b, c, SIZE)


# ---------- empty cells ----------
def test_air_domain_by_column():
    # column (0, 0): blocks at z=0 (domain 0) and z=2 (domain 1) -> hole z=1 and air take 1
    # column (1, 0): block at z=0 (domain 0) -> its air takes 0; columns 2-3 have no blocks
    grid = indices_to_grid([0, 0, 1, 4], [0, 0, 0, 0], [0, 2, 0, 0], (1, 1, 1))
    d = domains_to_cells(grid, np.array([0, 1, 0, 1]), "column").reshape(grid.nz, grid.ny, grid.nx)
    assert d[:, 0, 0].tolist() == [0, 1, 1]
    assert d[:, 0, 1].tolist() == [0, 0, 0]
    assert d[:, 0, 4].tolist() == [1, 1, 1]
    assert d[:, 0, 2].tolist() == [0, 0, 0]
    assert d[:, 0, 3].tolist() == [1, 1, 1]
    fixed = domains_to_cells(grid, np.array([0, 1, 0, 1]), 1).reshape(grid.nz, grid.ny, grid.nx)
    assert (fixed[:, 0, 2] == 1).all()
    assert (d >= 0).all()


def test_air_domain_errors():
    grid = indices_to_grid([0], [0], [0], (1, 1, 1))
    with pytest.raises(ValueError, match="air_domain must be >= 0"):
        domains_to_cells(grid, [0], -1)
    with pytest.raises(ValueError, match="air_domain must be 'column' or an integer"):
        domains_to_cells(grid, [0], "envelope")


def test_empty_cells_below_blocks():
    assert empty_cells_below_blocks(indices_to_grid(IX, IY, IZ, SIZE)) == 0  # only air above
    grid = indices_to_grid([0, 0, 1], [0, 0, 0], [0, 3, 0], (1, 1, 1))
    assert empty_cells_below_blocks(grid) == 2
