"""3-D views open in their own window, or render off screen for images and GIFs, also in Jupyter and IPython:
never a notebook backend (Trame) and nothing embedded in the notebook."""

import sys
import textwrap

import numpy as np
import pytest

import pymineflow
from pymineflow import plot

NOTEBOOK_MODULES = ("trame", "pyvista.trame", "pyvista.jupyter.notebook")


def pits():
    nx, ny, nz = 8, 7, 5
    ix, iy, iz = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"))
    grade = np.exp(-(((ix - 4) / 2.0) ** 2 + ((iy - 3) / 2.0) ** 2 + ((iz - 1) / 1.5) ** 2))
    table = {"IX": ix, "IY": iy, "IZ": iz, "VALUE": np.round(30 * grade - 4, 2), "TON": np.full(ix.size, 100.0)}
    return pymineflow.nested_pits(table, block_size=(10, 10, 10), slopes=45, factors=[0.4, 0.7, 1.0],
                                  discount_rate=0.1, ore_per_year=2000)  # fmt: skip


def notebook_modules():
    return {m for m in sys.modules if m.startswith(NOTEBOOK_MODULES)}


@pytest.fixture
def jupyter_theme(monkeypatch):
    """PyVista as inside Jupyter: its theme asks every plotter for the notebook backend."""
    pv = pytest.importorskip("pyvista")
    monkeypatch.setattr(pv.global_theme, "notebook", True)
    return pv


def test_display_detection(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(plot.sys, "platform", "linux")
    assert not plot._has_display()
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert plot._has_display()
    monkeypatch.delenv("WAYLAND_DISPLAY")
    monkeypatch.setenv("DISPLAY", ":0")
    assert plot._has_display()
    for platform in ("win32", "darwin"):
        monkeypatch.setattr(plot.sys, "platform", platform)
        assert plot._has_display()


def test_images_and_gifs_never_use_the_notebook(tmp_path, rendering, jupyter_theme):
    nested = pits()
    before = notebook_modules()
    shown = [
        nested[len(nested) - 1].plot(screenshot=tmp_path / "pit.png"),
        nested.plot(screenshot=tmp_path / "nested.png"),
        nested.plot(surface=True, screenshot=tmp_path / "surface.png"),
    ]
    nested.animate(tmp_path / "nested.gif")
    oes = nested.oes()
    shown.append(oes.plot(screenshot=tmp_path / "oes.png"))
    oes.animate(tmp_path / "oes.gif", frames=4)
    oes.export(tmp_path / "oes", details=False)
    for plotter in shown:
        assert plotter.notebook is False
        assert plotter._repr_png_() is None  # the value of a notebook cell: not shown as an image
        plotter.close()
    for name in ("pit.png", "nested.png", "surface.png", "oes.png", "nested.gif", "oes.gif", "oes/oes.gif",
                 "oes/oes_surface.gif"):  # fmt: skip
        assert (tmp_path / name).stat().st_size > 0, name
    assert notebook_modules() == before  # no Trame, no notebook display


def test_windows_open_natively(monkeypatch, jupyter_theme):
    """Every view asks for a native window (the window itself is not opened here)."""
    if not plot._has_display():
        pytest.skip("no display to create a window")
    pv = jupyter_theme
    seen = []
    monkeypatch.setattr(pv.Plotter, "show", lambda self, *a, **k: seen.append((self.notebook, self.off_screen)))
    nested = pits()
    oes = nested.oes()
    plotters = [nested[len(nested) - 1].plot(), nested.plot(), nested.plot(remaining=True), nested.plot(surface=True),
                oes.plot(), oes.plot(surface=True)]  # fmt: skip
    for plotter in plotters:
        plotter.close()
    assert seen == [(False, False)] * len(plotters)


def test_no_display_is_a_clear_error(monkeypatch, tmp_path, rendering):
    monkeypatch.setattr(plot, "_has_display", lambda: False)
    nested = pits()
    for draw in (nested[len(nested) - 1].plot, nested.plot, nested.oes().plot):
        with pytest.raises(RuntimeError, match="no display to open a 3-D window"):
            draw()
    nested[len(nested) - 1].plot(screenshot=tmp_path / "pit.png").close()  # images need no display
    nested.animate(tmp_path / "nested.gif")
    assert (tmp_path / "pit.png").exists()
    assert (tmp_path / "nested.gif").exists()


KERNEL = """
import sys
import scooby
import pyvista as pv
import pymineflow
from pymineflow import plot

assert scooby.in_ipykernel()
before = {{m for m in sys.modules if m.startswith({modules!r})}}
seen = []
pv.Plotter.show_original = pv.Plotter.show
if {window}:  # the window is not opened in the test: only how it would be opened
    pv.Plotter.show = lambda self, *a, **k: seen.append((self.notebook, self.off_screen))
nested = pymineflow.nested_pits({table!r}, block_size=(10, 10, 10), slopes=45, factors=[0.5, 1.0])
if {window}:
    nested.plot().close()
    nested[1].plot().close()
    pv.Plotter.show = pv.Plotter.show_original
image = nested[1].plot(screenshot={png!r})
nested.animate({gif!r})
after = {{m for m in sys.modules if m.startswith({modules!r})}}
print("SEEN", seen, "NEW", sorted(after - before))
image
"""


def test_in_a_jupyter_kernel(tmp_path, rendering):
    """The same calls in a real Jupyter kernel: native windows, off-screen images and GIFs, nothing embedded."""
    pytest.importorskip("ipykernel")
    manager = pytest.importorskip("jupyter_client.manager")
    n = 4
    table = {"IX": list(range(n)) * 2, "IY": [0] * (2 * n), "IZ": [1] * n + [0] * n,
             "VALUE": [-1.0] * n + [1.0, 9.0, 9.0, 1.0]}  # fmt: skip
    window = plot._has_display()
    code = KERNEL.format(modules=NOTEBOOK_MODULES, window=window, table=table, png=str(tmp_path / "pit.png"),
                         gif=str(tmp_path / "nested.gif"))  # fmt: skip
    km, kc = manager.start_new_kernel()
    messages = []
    try:
        reply = kc.execute_interactive(textwrap.dedent(code), timeout=300, output_hook=messages.append)
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    errors = [m["content"] for m in messages if m["msg_type"] == "error"]
    assert reply["content"]["status"] == "ok", errors
    text = "".join(m["content"].get("text", "") for m in messages if m["msg_type"] == "stream")
    expected = "[(False, False), (False, False)]" if window else "[]"
    assert f"SEEN {expected} NEW []" in text, text
    displayed = [m["content"]["data"] for m in messages if m["msg_type"] in ("execute_result", "display_data")]
    assert displayed, "the plotter is the value of the cell"
    assert all(set(d) == {"text/plain"} for d in displayed), displayed  # no image, no widget, no HTML
    assert (tmp_path / "pit.png").stat().st_size > 0
    assert (tmp_path / "nested.gif").stat().st_size > 0
