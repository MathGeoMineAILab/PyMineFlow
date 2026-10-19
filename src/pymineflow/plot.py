"""3-D view of an ultimate pit with PyVista (installed with PyMineFlow; versions 0.44 and later).

The grid is a VTK ``ImageData`` (a regular grid: no per-block geometry is built in Python) and only
the cells of the pit are extracted, so large pits draw quickly.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .model import PitResult

__all__ = ["animate_nested", "pit_mesh", "plot_nested", "plot_pit"]


def _pyvista() -> Any:
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("plotting needs PyVista: pip install pyvista") from None
    return pv


def _has_display() -> bool:
    """Whether a native window can open (on Linux it needs an X11 or Wayland display)."""
    if not sys.platform.startswith("linux"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _plotter(off_screen: bool, window_size: tuple[int, int]) -> Any:
    """Every PyVista plotter of PyMineFlow: a native VTK window, or an off-screen render for images and GIFs.

    Also in Jupyter and IPython: never a notebook backend (Trame, images embedded in the notebook), so every
    view opens in its own window, as in a script, with the same calls.
    """
    pv = _pyvista()
    if not off_screen and not _has_display():
        raise RuntimeError(
            "there is no display to open a 3-D window (no DISPLAY or WAYLAND_DISPLAY): save the view instead, "
            "with screenshot='view.png', or a GIF with animate() or export()"
        )
    plotter = pv.Plotter(off_screen=off_screen, notebook=False, window_size=list(window_size))
    plotter._repr_png_ = lambda: None  # returned to a notebook cell, it is not shown as an image
    return plotter


def _block_scalars(result: PitResult, color_by: str) -> tuple[NDArray[np.float64], str]:
    """One number per block (input rows) and its legend title."""
    if color_by == "value":
        if result.values is None:
            raise ValueError("color_by='value' needs the block values")
        return result.values / result.factor, result.value_column or "value"
    if color_by == "ore":
        if result.values is None:
            raise ValueError("color_by='ore' needs the block values")
        return (result.values > 0).astype(np.float64), "ore (1) / waste (0)"
    if color_by == "level":
        return np.asarray(result.grid.iz, dtype=np.float64), "level"
    if result.table is not None and color_by in result.table:
        column = np.asarray(result.table[color_by])
        if column.dtype.kind not in "iufb":
            raise ValueError(f"column {color_by!r} is not numeric")
        return column.astype(np.float64), color_by
    available = ["value", "ore", "level", *(result.table or {})]
    raise ValueError(f"color_by must be one of {available}, got {color_by!r}")


def _image(result: PitResult) -> Any:
    """The model's grid as a VTK ImageData (cell (ix, iy, iz) follows the linear index)."""
    pv = _pyvista()
    g = result.grid
    dx, dy, dz = g.block_size
    centre = g.origin if g.origin is not None else tuple(b * s for b, s in zip(g.base, g.block_size, strict=True))
    corner = (centre[0] - dx / 2, centre[1] - dy / 2, centre[2] - dz / 2)
    return pv.ImageData(dimensions=(g.nx + 1, g.ny + 1, g.nz + 1), spacing=(dx, dy, dz), origin=corner)


def pit_mesh(result: PitResult, color_by: str = "value", surface: bool = True) -> Any:
    """The pit blocks as a PyVista mesh with the scalar ``color_by`` (see ``plot_pit``).

    ``surface=True`` keeps only the outer faces (the inner ones are hidden anyway), which draws
    much faster on large pits.
    """
    scalars, title = _block_scalars(result, color_by)
    image = _image(result)
    cells = np.full(result.grid.n_cells, np.nan)
    linear = np.asarray(result.grid.linear)
    cells[linear[result.mask]] = scalars[result.mask]
    image.cell_data[title] = cells
    mesh = image.extract_cells(linear[result.mask])
    return _surface(mesh) if surface else mesh


def _surface(mesh: Any) -> Any:
    """Outer faces only (``algorithm`` exists from PyVista 0.47 on; older versions take no argument)."""
    import inspect

    if "algorithm" in inspect.signature(mesh.extract_surface).parameters:
        return mesh.extract_surface(algorithm="dataset_surface")
    return mesh.extract_surface()


def plot_pit(
    result: PitResult,
    color_by: str = "value",
    *,
    cmap: str = "jet",
    clim: tuple[float, float] | None = None,
    view: str = "iso",
    rotation: float = 0.0,
    title: str | None = None,
    outline: bool = True,
    axes: bool = True,
    show_edges: bool = False,
    opacity: float = 1.0,
    show_model: bool = False,
    surface: bool = True,
    screenshot: str | Path | None = None,
    window_size: tuple[int, int] = (1280, 800),
    show: bool = True,
) -> Any:
    """Draw the ultimate pit in 3-D with a colour legend (jet by default).

    Parameters
    ----------
    result : the result of ``solve`` or ``ultimate_pit``.
    color_by : ``"value"`` (block value), ``"ore"`` (value > 0), ``"level"``, or any numeric
        column of the source table (e.g. a grade).
    cmap : colour map of the legend (default ``"jet"``).
    clim : colour range; by default the 2nd to 98th percentile of the pit blocks, so that a few
        extreme blocks do not wash out the colours.
    view : ``"iso"`` (from above), ``"below"`` (from below: the shape of the pit shell), ``"top"``
        (plan view, north up) or ``"side"``; the last two without perspective.
    rotation : plan view only: turn the view by this many degrees, counter-clockwise (e.g. 90 to
        lay a north-south pit horizontally).
    title : text at the top left (default: the pit's blocks and value; ``""`` for none).
    outline : draw the box of the model.
    axes : draw the small x, y, z axes.
    show_edges : draw the block edges.
    opacity : of the pit blocks (lower it with ``surface=False`` to see inside).
    show_model : also draw the blocks outside the pit, faint (slower on large models).
    surface : draw only the outer faces of the pit (faster).
    screenshot : save the image to this file (PNG) instead of opening a window.
    show : open the interactive window (ignored with ``screenshot``).

    Returns the ``pyvista.Plotter``, to add more elements or save the view.
    """
    if result.n_blocks == 0:
        raise ValueError("the pit is empty: nothing to draw")
    scalars, legend = _block_scalars(result, color_by)
    if clim is None:
        low, high = np.nanpercentile(scalars[result.mask], [2, 98])
        clim = (float(low), float(high)) if high > low else (float(low) - 0.5, float(high) + 0.5)
    if view not in ("iso", "below", "top", "side"):
        raise ValueError(f"view must be 'iso', 'below', 'top' or 'side', got {view!r}")
    plotter = _plotter(screenshot is not None or not show, window_size)
    mesh = pit_mesh(result, color_by, surface)
    plotter.add_mesh(
        mesh,
        scalars=legend,
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        opacity=opacity,
        show_scalar_bar=True,
        scalar_bar_args={"title": legend, "vertical": True, "fmt": "%.4g"},
    )
    image = _image(result)
    if show_model:
        outside = ~result.mask
        rest = image.extract_cells(np.asarray(result.grid.linear)[outside])
        plotter.add_mesh(_surface(rest), color="lightgray", opacity=0.08)
    if outline:
        plotter.add_mesh(image.outline(), color="gray")
    if title is None:
        title = f"Ultimate pit: {result.n_blocks:,} blocks, value {result.real_value:,.0f}"
    if title:
        plotter.add_text(title, font_size=10, color="black")
    plotter.set_background("white")
    if axes:
        plotter.add_axes()
    if view == "top":
        plotter.view_xy()  # plan view: north up, east to the right
    elif view == "side":
        plotter.view_xz()
    else:
        plotter.view_isometric()
        if view == "below":
            plotter.camera.elevation = -60.0
    if view in ("top", "side"):
        plotter.enable_parallel_projection()  # a map, without perspective
    plotter.reset_camera(bounds=mesh.bounds)  # frame the pit, not the whole model
    if view in ("top", "side"):
        plotter.camera.parallel_scale *= 1.08  # a small margin around the pit
    if view == "top" and rotation:
        plotter.camera.roll += rotation
        x0, x1, y0, y1 = mesh.bounds[:4]
        turn = np.radians(rotation)
        across = abs((x1 - x0) * np.cos(turn)) + abs((y1 - y0) * np.sin(turn))  # extent across the screen
        up = abs((x1 - x0) * np.sin(turn)) + abs((y1 - y0) * np.cos(turn))
        aspect = 0.75 * window_size[0] / window_size[1]  # the legend takes the right of the window
        plotter.camera.parallel_scale = 0.54 * max(up, across / aspect)
    plotter.camera_set = True  # keep this framing when the window opens
    if screenshot is not None:
        plotter.show(screenshot=str(screenshot), auto_close=False)
    elif show:
        plotter.show()
    return plotter


def _nested_faces(
    grid: Any, numbers: NDArray[np.integer], count: int, remaining: bool = False
) -> tuple[
    NDArray[np.float64], NDArray[np.int64], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int64]
]:
    """Every block face that some step of a cumulative sequence shows, computed once.

    ``numbers`` gives, per block, the step (1 to ``count``) in which it is mined (0 = never). After ``k`` steps
    the mined part is the blocks with numbers 1 to ``k``. Its surface (``remaining=False``) shows the face of a
    block of step ``a`` whose neighbour is of step ``b`` (0 = no block or never mined) for ``a <= k < b`` (``k >=
    a`` when ``b`` is 0); the part still to mine (``remaining=True``) shows it for ``b <= k < a`` (``k < a`` when
    ``b`` is 0). Returns the points, the quads (point indices), ``first`` and ``last`` (exclusive) of every face,
    so that the surface after any step is a plain filter, ``own``: the step of the block of every face, and
    ``cell``: the grid cell of that block.
    """
    numbers = np.asarray(numbers)
    cells = np.zeros(grid.n_cells, dtype=np.int32)
    inside = numbers > 0
    cells[np.asarray(grid.linear)[inside]] = numbers[inside]
    volume = cells.reshape(grid.nz, grid.ny, grid.nx)  # [iz, iy, ix]
    never = np.int32(count + 1)
    # the 4 corners of a face normal to each axis, as offsets in [iz, iy, ix]
    corners = {
        0: [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        1: [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)],
        2: [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    }
    quads, first, last, own, cell = [], [], [], [], []
    for axis in range(3):
        for side in (-1, 1):
            neighbour = np.zeros_like(volume)
            src: list[slice] = [slice(None)] * 3
            dst: list[slice] = [slice(None)] * 3
            if side == 1:
                src[axis], dst[axis] = slice(1, None), slice(None, -1)
            else:
                src[axis], dst[axis] = slice(None, -1), slice(1, None)
            neighbour[tuple(dst)] = volume[tuple(src)]
            if remaining:
                shown = (volume > 0) & (neighbour < volume)
                start, end = neighbour[shown], volume[shown]
            else:
                neighbour = np.where(neighbour == 0, never, neighbour)
                shown = (volume > 0) & (neighbour > volume)
                start, end = volume[shown], neighbour[shown]
            at = np.stack(np.nonzero(shown), axis=1)
            cell.append((at[:, 0] * grid.ny + at[:, 1]) * grid.nx + at[:, 2])  # [iz, iy, ix] -> linear cell
            if side == 1:
                at[:, axis] += 1  # the face on the far side of the block
            quads.append(np.stack([at + np.array(c) for c in corners[axis]], axis=1))  # (m, 4, 3)
            first.append(start)
            last.append(end)
            own.append(volume[shown])
    corner = np.concatenate(quads).astype(np.int64)
    # point ids on the lattice of block corners; only the points that are used are kept
    ids = (corner[..., 0] * (grid.ny + 1) + corner[..., 1]) * (grid.nx + 1) + corner[..., 2]
    used, quad = np.unique(ids.ravel(), return_inverse=True)
    pz, rest = np.divmod(used, (grid.ny + 1) * (grid.nx + 1))
    py, px = np.divmod(rest, grid.nx + 1)
    x0, _, y0, _, z0, _ = _model_bounds(grid)
    dx, dy, dz = grid.block_size
    points = np.column_stack([x0 + px * dx, y0 + py * dy, z0 + pz * dz])
    return (points, quad.reshape(-1, 4).astype(np.int64), np.concatenate(first), np.concatenate(last),
            np.concatenate(own), np.concatenate(cell).astype(np.int64))  # fmt: skip


def _model_bounds(g: Any) -> tuple[float, float, float, float, float, float]:
    """The box of the model's grid (block corners)."""
    dx, dy, dz = g.block_size
    centre = g.origin if g.origin is not None else tuple(b * s for b, s in zip(g.base, g.block_size, strict=True))
    x0, y0, z0 = centre[0] - dx / 2, centre[1] - dy / 2, centre[2] - dz / 2
    return (x0, x0 + g.nx * dx, y0, y0 + g.ny * dy, z0, z0 + g.nz * dz)


class _NestedScene:
    """A plotter with a cumulative sequence of nested pits; ``show(k)`` shows the state after ``k`` steps in
    milliseconds: it only filters the faces computed once and updates the same mesh.

    ``remaining=False`` draws what is mined (the pit growing, steps ``first..n``); ``remaining=True`` draws what
    is still to mine (the whole pit first, then less and less, steps ``first - 1..last - 1``). ``legend``,
    ``colors`` (the value drawn for every block, one per row; by default its step) and ``label`` (the text after
    ``k`` steps) let other sequences use the same scene (e.g. the frames of an extraction sequence).
    """

    def __init__(
        self,
        pits: Any,
        *,
        view: str,
        rotation: float,
        title: str | None,
        outline: bool,
        axes: bool,
        cmap: str,
        window_size: tuple[int, int],
        off_screen: bool,
        remaining: bool = False,
        legend: str = "pit number",
        colors: NDArray[np.float64] | None = None,
        label: Callable[[int], str] | None = None,
    ) -> None:
        pv = _pyvista()
        if view not in ("iso", "below", "top", "side"):
            raise ValueError(f"view must be 'iso', 'below', 'top' or 'side', got {view!r}")
        self.n = n = len(pits)
        numbers = np.asarray(pits.numbers)
        self.blocks = np.cumsum(np.bincount(numbers, minlength=n + 1)[1:])  # pit k has blocks[k - 1]
        if self.blocks[-1] == 0:
            raise ValueError("every nested pit is empty: there is nothing to show")
        # the legend and the steps follow the pits that exist: from the first pit with blocks (low factors may
        # give empty pits) to the last pit that adds blocks
        self.first_pit = int(np.flatnonzero(self.blocks > 0)[0]) + 1
        self.last_pit = max(int(numbers.max()), self.first_pit + 1)  # the top of the colour scale
        self.remaining = remaining
        # remaining: from nothing mined to the last state that still has blocks to mine
        self.low, self.high = (self.first_pit - 1, int(numbers.max()) - 1) if remaining else (self.first_pit, n)
        self.legend = legend
        self.colors = None if colors is None else np.asarray(colors, dtype=np.float64)
        self.label = label
        if label is None:
            self.values = list(pits.values)
            self.factors = list(pits.factors)
            self.kind = {"revenue": "revenue factor", "price": "price factor", "profit": "profit factor"}[pits.scale]
            if title is None:
                title = f"Nested pits: {n} {self.kind}s, {self.factors[0]:g} to {self.factors[-1]:g}"
        self.title = title
        grid = pits.model.grid
        self.points, self.quads, self.first, self.last, self.own, cell = _nested_faces(grid, numbers, n, remaining)
        if self.colors is not None:  # the colour of every face: that of its block
            row_of_cell = np.full(grid.n_cells, -1, dtype=np.int64)
            row_of_cell[np.asarray(grid.linear)] = np.arange(numbers.size)
            self.face_colors = self.colors[row_of_cell[cell]]
        start = self.low if remaining else self.high  # the whole pit: to mine, or mined
        self.mesh = pv.PolyData(self.points, faces=self._faces(start))
        self.mesh.cell_data[legend] = self._scalars(start)
        if self.colors is None:
            clim = (float(self.first_pit), float(self.last_pit))
        else:
            used = self.colors[numbers > 0]
            low, high = float(np.min(used)), float(np.max(used))
            clim = (low, max(high, low + 1.0))
        self.plotter = plotter = _plotter(off_screen, window_size)
        plotter.set_background("white")
        plotter.add_mesh(
            self.mesh, scalars=legend, cmap=cmap, clim=clim,
            scalar_bar_args={"title": legend, "vertical": True, "fmt": "%.0f"},
        )  # fmt: skip
        if outline:
            plotter.add_mesh(pv.Box(bounds=_model_bounds(grid)).outline(), color="gray")
        if axes:
            plotter.add_axes()
        # a text actor (not a corner annotation, which VTK may lay out with matplotlib on every frame)
        self.text = plotter.add_text(self._label(start), position=(0.01, 0.985), viewport=True, font_size=10,
                                     color="black")  # fmt: skip
        self.text.GetTextProperty().SetVerticalJustificationToTop()
        self._frame(view, rotation, window_size)

    def steps(self, step: int = 1) -> list[int]:
        """The states to show, every ``step`` steps, the first and the last always included."""
        ks = list(range(self.low, self.high + 1, step))
        if ks[-1] != self.high:
            ks.append(self.high)
        return ks

    def _shown(self, k: int) -> NDArray[np.bool_]:
        return (self.first <= k) & (k < self.last)

    def _scalars(self, k: int) -> NDArray[np.float64]:
        own = self.own[self._shown(k)]
        if self.colors is not None:
            return self.face_colors[self._shown(k)]
        return own.astype(np.float64)

    def _faces(self, k: int) -> NDArray[np.int64]:
        quads = self.quads[self._shown(k)]
        return np.column_stack([np.full(len(quads), 4, dtype=np.int64), quads]).ravel()

    def _label(self, k: int) -> str:
        if self.label is not None:
            info = self.label(k)
        elif not self.remaining:
            info = (
                f"Pit {k} of {self.n}: {self.kind} {self.factors[k - 1]:.4f}, {int(self.blocks[k - 1]):,} blocks,"
                f" value {self.values[k - 1]:,.0f}"
            )
        else:
            total = int(self.blocks[-1])
            mined = int(self.blocks[k - 1]) if k >= 1 else 0
            done = f"pits 1 to {k} mined ({self.kind} {self.factors[k - 1]:.4f})" if mined else "nothing mined"
            info = f"{done}: {total - mined:,} of {total:,} blocks still to mine"
        return f"{self.title}\n{info}" if self.title else info

    def show(self, position: float) -> None:
        """Show the state after ``position`` steps (rounded: the slider gives a float)."""
        k = min(max(round(position), self.low), self.high)
        self.mesh.faces = self._faces(k)
        self.mesh.cell_data[self.legend] = self._scalars(k)
        self.mesh.cell_data.active_scalars_name = self.legend
        self.text.SetInput(self._label(k))

    def _frame(self, view: str, rotation: float, window_size: tuple[int, int]) -> None:
        plotter = self.plotter
        if view == "top":
            plotter.view_xy()  # plan view: north up, east to the right
        elif view == "side":
            plotter.view_xz()
        else:
            plotter.view_isometric()
            if view == "below":
                plotter.camera.elevation = -60.0
        if view in ("top", "side"):
            plotter.enable_parallel_projection()  # a map, without perspective
        bounds = self.mesh.bounds  # the largest pit, not the whole model
        plotter.reset_camera(bounds=bounds)
        if view in ("top", "side"):
            plotter.camera.parallel_scale *= 1.08
        if view == "top" and rotation:
            plotter.camera.roll += rotation
            x0, x1, y0, y1 = bounds[:4]
            turn = np.radians(rotation)
            across = abs((x1 - x0) * np.cos(turn)) + abs((y1 - y0) * np.sin(turn))
            up = abs((x1 - x0) * np.sin(turn)) + abs((y1 - y0) * np.cos(turn))
            plotter.camera.parallel_scale = 0.54 * max(up, across / (0.75 * window_size[0] / window_size[1]))
        plotter.camera_set = True


def _surface_cells(grid: Any) -> NDArray[np.bool_]:
    """The cells of the ground surface: one per 2 x 2 neighbouring columns that all hold blocks (index j * (nx - 1)
    + i, x fastest). None on a vertical section (one row of columns)."""
    filled = np.zeros(grid.nx * grid.ny, dtype=bool)
    filled[np.asarray(grid.iy, dtype=np.int64) * grid.nx + np.asarray(grid.ix, dtype=np.int64)] = True
    f = filled.reshape(grid.ny, grid.nx)
    return (f[:-1, :-1] & f[1:, :-1] & f[:-1, 1:] & f[1:, 1:]).ravel()


class _SurfaceScene(_NestedScene):
    """The ground surface while a cumulative sequence is mined: in every column of the model, the top of what is
    left (the top of the model before mining, the bottom of the deepest block mined after), drawn through the
    column centres and coloured by elevation (jet). ``show(k)`` shows the surface after ``k`` steps; the steps,
    the text and the slider are those of ``_NestedScene`` with ``remaining=True``. The surface of every step is
    looked up in arrays computed once, so moving the slider only replaces the heights of one mesh."""

    def __init__(
        self,
        pits: Any,
        *,
        view: str,
        rotation: float,
        title: str | None,
        outline: bool,
        axes: bool,
        cmap: str,
        window_size: tuple[int, int],
        off_screen: bool,
        remaining: bool = True,
        legend: str = "elevation",
        colors: NDArray[np.float64] | None = None,
        label: Callable[[int], str] | None = None,
    ) -> None:
        pv = _pyvista()
        if view not in ("iso", "below", "top", "side"):
            raise ValueError(f"view must be 'iso', 'below', 'top' or 'side', got {view!r}")
        self.n = n = len(pits)
        numbers = np.asarray(pits.numbers, dtype=np.int64)
        self.blocks = np.cumsum(np.bincount(numbers, minlength=n + 1)[1:])
        if self.blocks[-1] == 0:
            raise ValueError("every nested pit is empty: there is nothing to show")
        self.first_pit = int(np.flatnonzero(self.blocks > 0)[0]) + 1
        self.remaining = True
        self.low, self.high = self.first_pit - 1, int(numbers.max())
        self.legend = "elevation"
        self.colors = None
        self.label = label
        if label is None:
            self.values = list(pits.values)
            self.factors = list(pits.factors)
            self.kind = {"revenue": "revenue factor", "price": "price factor", "profit": "profit factor"}[pits.scale]
            if title is None:
                title = f"Nested pits: {n} {self.kind}s, {self.factors[0]:g} to {self.factors[-1]:g}"
        self.title = title
        g = pits.model.grid
        x0, _, y0, _, z0, _ = _model_bounds(g)
        dx, dy, dz = g.block_size
        ix, iy, iz = (np.asarray(a, dtype=np.int64) for a in (g.ix, g.iy, g.iz))
        column = iy * g.nx + ix
        columns = g.nx * g.ny
        top = np.full(columns, -1, dtype=np.int64)
        np.maximum.at(top, column, iz)
        self.filled = top >= 0  # columns with blocks
        self.top = np.where(self.filled, z0 + (top + 1) * dz, np.nan)  # elevation before mining
        # mined blocks by column, then by step: the lowest bench mined so far in every column
        mined = np.flatnonzero(numbers > 0)
        order = mined[np.lexsort((numbers[mined], column[mined]))]
        c, step, level = column[order], numbers[order], iz[order]
        span = int(g.nz) + 1
        self.lowest = np.minimum.accumulate(level - c * span) + c * span  # running minimum inside every column
        self.keys = c * (n + 2) + step  # sorted: the search below finds the last block mined by step k
        self.column_keys = np.arange(columns, dtype=np.int64) * (n + 2)
        self.column_of = c
        self.z0, self.dz = z0, dz
        # the surface: one point per column centre, the cells whose four columns hold blocks
        xs = x0 + (np.arange(g.nx) + 0.5) * dx
        ys = y0 + (np.arange(g.ny) + 0.5) * dy
        gx, gy = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny), x fastest: point index = iy * nx + ix = column
        full = pv.StructuredGrid(gx[..., None], gy[..., None], np.zeros_like(gx)[..., None])
        cells = _surface_cells(g)
        if not cells.any():
            raise ValueError("the model needs at least 2 x 2 neighbouring columns of blocks to draw a surface")
        self.mesh = full.extract_cells(np.flatnonzero(cells))
        self.ids = np.asarray(self.mesh.point_data["vtkOriginalPointIds"], dtype=np.int64)
        self.base_points = np.asarray(self.mesh.points, dtype=np.float64).copy()
        start = self.low
        heights = self._heights(start)
        self._set(heights)
        everything = self._heights(self.high)
        shown = np.concatenate([self.top[self.ids], everything[self.ids]])
        clim = (float(np.nanmin(shown)), float(np.nanmax(shown)))
        if clim[1] <= clim[0]:
            clim = (clim[0], clim[0] + dz)
        self.plotter = plotter = _plotter(off_screen, window_size)
        plotter.set_background("white")
        plotter.add_mesh(
            self.mesh, scalars="elevation", cmap=cmap, clim=clim, smooth_shading=True,
            scalar_bar_args={"title": "elevation", "vertical": True, "fmt": "%.0f"},
        )  # fmt: skip
        if outline:
            plotter.add_mesh(pv.Box(bounds=_model_bounds(g)).outline(), color="gray")
        if axes:
            plotter.add_axes()
        self.text = plotter.add_text(self._label(start), position=(0.01, 0.985), viewport=True, font_size=10,
                                     color="black")  # fmt: skip
        self.text.GetTextProperty().SetVerticalJustificationToTop()
        self._set(everything)  # the camera sees the whole depth of the final pit, not only the ground before mining
        self._frame(view, rotation, window_size)
        self._set(heights)
        plotter.reset_camera_clipping_range()

    def _heights(self, k: int) -> NDArray[np.float64]:
        """The elevation of the surface of every column after ``k`` steps."""
        last = np.searchsorted(self.keys, self.column_keys + k, side="right") - 1
        valid = (last >= 0) & (self.column_of[np.maximum(last, 0)] == np.arange(self.column_keys.size))
        bottom = self.z0 + self.lowest[np.maximum(last, 0)] * self.dz
        return np.where(valid, bottom, self.top)

    def _set(self, heights: NDArray[np.float64]) -> None:
        points = self.base_points.copy()
        points[:, 2] = heights[self.ids]
        self.mesh.points = points
        self.mesh.point_data["elevation"] = points[:, 2]

    def show(self, position: float) -> None:
        """Show the surface after ``position`` steps (rounded: the slider gives a float)."""
        k = min(max(round(position), self.low), self.high)
        self._set(self._heights(k))
        self.mesh.point_data.active_scalars_name = "elevation"
        self.text.SetInput(self._label(k))
        self.plotter.reset_camera_clipping_range()  # the pit gets deeper: keep its bottom in view


def plot_nested(
    pits: Any,
    *,
    view: str = "iso",
    rotation: float = 0.0,
    title: str | None = None,
    outline: bool = True,
    axes: bool = True,
    cmap: str = "jet",
    slider: bool = True,
    remaining: bool = False,
    surface: bool = False,
    screenshot: str | Path | None = None,
    window_size: tuple[int, int] = (1280, 800),
    show: bool | None = None,
) -> Any:
    """Nested pits in 3-D, every block coloured by its pit number (jet), with a slider that walks
    from the first pit to the last: the same view for profit and price factors.

    The faces of all the pits are computed once, so moving the slider only filters them.

    Parameters
    ----------
    view : ``"iso"``, ``"below"``, ``"top"`` (plan view) or ``"side"``, as in ``plot_pit``.
    rotation : plan view only: turn the view by this many degrees, counter-clockwise (``-90``:
        north to the right).
    title : first line of the text at the top left (``""`` for none); the second line is the pit
        shown, its factor, blocks and value.
    outline, axes, cmap : the model box, the small axes and the colour map.
    slider : add the slider to the window.
    remaining : show what is still to mine instead of what is mined: the whole final pit, from which the slider
        removes the pits in order.
    surface : show the ground surface instead of the blocks: the top of the model, which the slider digs pit
        after pit (coloured by elevation; ``remaining`` does not apply).
    screenshot : save the image of all the pits to this file (PNG).
    show : open the interactive window; by default only when no ``screenshot`` is given
        (``show=True`` with a ``screenshot`` does both).

    Returns the ``pyvista.Plotter`` of the window (or of the image).
    """
    if show is None:
        show = screenshot is None
    options: dict[str, Any] = {"view": view, "rotation": rotation, "title": title, "outline": outline,
                               "axes": axes, "cmap": cmap, "window_size": window_size,
                               "remaining": remaining}  # fmt: skip
    kind = _SurfaceScene if surface else _NestedScene
    if screenshot is not None:
        scene = kind(pits, off_screen=True, **options)
        if surface:
            scene.show(scene.high)  # the image: the final pit dug
        scene.plotter.show(screenshot=str(screenshot), auto_close=False)
        if not show:
            return scene.plotter
        scene.plotter.close()
    scene = kind(pits, off_screen=not show, **options)
    plotter = scene.plotter
    if slider and scene.high > scene.low:
        plotter.add_slider_widget(
            scene.show, [scene.low, scene.high], value=scene.low if scene.remaining else scene.high,
            title="pits mined" if scene.remaining else "pit", fmt="%.0f", pointa=(0.25, 0.06), pointb=(0.70, 0.06),
            style="modern", interaction_event="always",
        )  # fmt: skip
    if show:
        plotter.show()
    return plotter


def animate_nested(
    pits: Any,
    path: str | Path = "nested_pits.gif",
    *,
    view: str = "iso",
    rotation: float = 0.0,
    title: str | None = None,
    outline: bool = True,
    axes: bool = True,
    cmap: str = "jet",
    step: int = 1,
    remaining: bool = False,
    surface: bool = False,
    seconds_per_pit: float = 0.1,
    hold: float = 2.0,
    window_size: tuple[int, int] = (960, 600),
) -> Path:
    """Save an animated GIF of the pit growing from the first nested pit to the last or, with
    ``remaining=True``, of the whole final pit mined pit after pit, what is left staying on screen, or, with
    ``surface=True``, of the ground surface dug pit after pit.

    ``step`` skips pits (e.g. 5 = every fifth pit), ``seconds_per_pit`` sets the speed and ``hold``
    keeps the last frame on screen. The other options are those of ``plot_nested``. Returns the path.
    """
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError(f"step must be an integer >= 1, got {step!r}")
    kind = _SurfaceScene if surface else _NestedScene
    scene = kind(pits, view=view, rotation=rotation, title=title, outline=outline, axes=axes, cmap=cmap,
                 window_size=window_size, off_screen=True, remaining=remaining)  # fmt: skip
    return _write_gif(scene, scene.steps(int(step)), Path(path), seconds_per_pit, hold)


def _write_gif(scene: _NestedScene, ks: list[int], path: Path, seconds: float, hold: float) -> Path:
    """Render the frames ``ks`` of a scene and save them as an animated GIF."""
    from PIL import Image  # installed with PyVista

    for name, x in (("the seconds per frame", seconds), ("hold", hold)):
        if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)) or not 0 < x < 3600:
            raise ValueError(f"{name} must be a number of seconds > 0, got {x!r}")

    plotter = scene.plotter
    plotter.show(auto_close=False)
    shots = []
    for k in ks:
        scene.show(k)
        plotter.render()
        shots.append(plotter.screenshot(return_img=True))
    plotter.close()
    # the first and the last frames hold every colour (the whole pit to mine and mined, or the surface before
    # and after): the palette of the GIF comes from both
    palette = Image.fromarray(np.concatenate([shots[0], shots[-1]])).convert("P", palette=Image.Palette.ADAPTIVE)
    frames = [Image.fromarray(shot).quantize(palette=palette, dither=Image.Dither.NONE) for shot in shots]
    durations = [round(seconds * 1000)] * (len(frames) - 1) + [round(hold * 1000)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return path
