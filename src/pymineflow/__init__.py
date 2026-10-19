"""PyMineFlow: ultimate pit (maximum closure) with MineFlow's pseudoflow solver.

Conventions (MineFlow's):

- Linear cell index ``(iz*ny + iy)*nx + ix`` (x fastest, then y, then z); z grows upwards.
- Azimuth in degrees: 0 = +y (north), 90 = +x (east), clockwise.
- Slope in degrees from the horizontal, in (0, 90), measured centre to centre, boundary included.
- Values are int64; scale real values with ``scale_values``.

The shortest path: ``pit = solve("model.csv", block_size=(10, 10, 5), slopes=45)``, then
``print(pit.summary())``, ``pit.to_csv("pit.csv")`` and ``pit.plot()``. From a terminal:
``pymineflow model.csv --block-size 10 10 5 --slope 45``.
"""

from ._mineflow import ExplicitPit, Pit
from .compare import PitComparison
from .economics import Contaminant, Economics, Metal, Process, economics_from_dict, read_economics
from .grid import Grid, coords_to_grid, domains_to_cells, empty_cells_below_blocks, indices_to_grid
from .model import PitModel, PitResult, bench_discount_weights, scale_values, ultimate_pit
from .npv import PitByPit, pit_by_pit
from .oes import OES, OESCheck, ultimate_pit_oes
from .plot import plot_pit
from .readers import read_blocks, read_minelib_precedence, read_minelib_values, read_reference_pits, write_table
from .simple import NestedPits, nested_pits, solve, value_blocks
from .slopes import discretize_slopes

__version__ = "0.1.0"

__all__ = [
    "OES",
    "Contaminant",
    "Economics",
    "ExplicitPit",
    "Grid",
    "Metal",
    "NestedPits",
    "OESCheck",
    "Pit",
    "PitByPit",
    "PitComparison",
    "PitModel",
    "PitResult",
    "Process",
    "__version__",
    "bench_discount_weights",
    "coords_to_grid",
    "discretize_slopes",
    "domains_to_cells",
    "economics_from_dict",
    "empty_cells_below_blocks",
    "indices_to_grid",
    "nested_pits",
    "pit_by_pit",
    "plot_pit",
    "read_blocks",
    "read_economics",
    "read_minelib_precedence",
    "read_minelib_values",
    "read_reference_pits",
    "scale_values",
    "solve",
    "ultimate_pit",
    "ultimate_pit_oes",
    "value_blocks",
    "write_table",
]
