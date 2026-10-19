"""Type stubs for the compiled extension (see _mineflow.cpp)."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

compiler: str

_Solution = tuple[NDArray[np.bool_], int, int, float]

class Pit:
    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        values: ArrayLike,
        *,
        pattern: str = "1:5",
        slope_deg: float | None = None,
        n_benches: int = 8,
        block_size: tuple[float, float, float] | None = None,
        sectors: Sequence[tuple[float, float]] | None = None,
        domains: ArrayLike | None = None,
        domain_slopes: Sequence[Sequence[tuple[float, float]]] | None = None,
        full_pattern: bool = False,
    ) -> None: ...
    def solve(self) -> _Solution: ...
    def solve_with_values(self, values: ArrayLike) -> _Solution: ...
    def offsets(self, domain: int = 0) -> NDArray[np.int64]: ...
    @property
    def n_domains(self) -> int: ...
    def solve_subset(self, cells: ArrayLike, values: ArrayLike) -> _Solution: ...

class ExplicitPit:
    def __init__(self, n: int, src: ArrayLike, dst: ArrayLike, values: ArrayLike) -> None: ...
    @staticmethod
    def from_csr(indptr: ArrayLike, indices: ArrayLike, values: ArrayLike) -> ExplicitPit: ...
    def solve(self) -> _Solution: ...
    def solve_with_values(self, values: ArrayLike) -> _Solution: ...

def _solve_explicit_precedence(
    n: int, src: ArrayLike, dst: ArrayLike, values: ArrayLike
) -> tuple[NDArray[np.bool_], int]: ...

# Ultimate pit OES kernels (_oes.cpp); called by pymineflow.oes, which validates the inputs.
def _oes_lookahead(
    ix: ArrayLike,
    iy: ArrayLike,
    iz: ArrayLike,
    value: ArrayLike,
    dx: float,
    dy: float,
    radii: Sequence[float],
    weights: Sequence[float],
    threads: int = 0,
) -> NDArray[np.float64]: ...
def _oes_order(
    ix: ArrayLike, iy: ArrayLike, iz: ArrayLike, phase: ArrayLike, key: ArrayLike, c: float, dx: float, dy: float
) -> NDArray[np.int64]: ...
def _oes_order_free(
    nx: int,
    ny: int,
    nz: int,
    cell: ArrayLike,
    ix: ArrayLike,
    iy: ArrayLike,
    iz: ArrayLike,
    phase: ArrayLike,
    key: ArrayLike,
    c: float,
    dx: float,
    dy: float,
    dz: float,
    offsets: Sequence[ArrayLike],
    domain: ArrayLike | None,
    others: ArrayLike,
) -> NDArray[np.int64]: ...
def _oes_merge_small(
    nx: int,
    ny: int,
    nz: int,
    cell: ArrayLike,
    phase: ArrayLike,
    offsets: Sequence[ArrayLike],
    domain: ArrayLike | None,
    others: ArrayLike,
    min_blocks: int,
    max_rounds: int = 50,
) -> tuple[NDArray[np.int64], int, int]: ...
def _oes_schedule(
    value: ArrayLike,
    uses: ArrayLike,
    caps: ArrayLike,
    track: ArrayLike,
    rate: float,
    shift: float,
    fixed: float,
    continuous: bool = False,
) -> dict[str, Any]: ...
def _oes_check(
    nx: int,
    ny: int,
    nz: int,
    rank: ArrayLike,
    offsets: Sequence[ArrayLike],
    domain: ArrayLike | None = None,
    threads: int = 0,
) -> tuple[int, int]: ...
