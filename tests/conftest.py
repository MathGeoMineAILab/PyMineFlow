"""Shared fixtures."""

import subprocess
import sys

import pytest

PROBE = """
import pyvista as pv
plotter = pv.Plotter(off_screen=True, window_size=[64, 64])
plotter.add_mesh(pv.Cube())
plotter.screenshot()
plotter.close()
"""


@pytest.fixture(scope="session")
def rendering():
    """Skip a test that draws with PyVista in-process when this machine cannot render off screen.

    The probe runs in a separate process: a machine without OpenGL may crash instead of raising.
    """
    pytest.importorskip("pyvista")
    done = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=300, check=False)
    if done.returncode != 0:
        pytest.skip(f"no off-screen rendering here: {done.stderr[-300:]}")
