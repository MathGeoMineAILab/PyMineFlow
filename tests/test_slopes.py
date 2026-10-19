"""Slope definitions: normalization, steep envelope and per-block discretization."""

import numpy as np
import pytest

from pymineflow import discretize_slopes
from pymineflow.slopes import normalize_slopes, slope_at_azimuth, slopes_differ, steep_envelope


def test_normalize():
    assert normalize_slopes(45) == [(0.0, 45.0)]
    assert normalize_slopes([(0, 40), (370, 50)]) == [(0.0, 40.0), (370.0, 50.0)]


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (90, r"\(0, 90\)"),
        (0, r"\(0, 90\)"),
        ([(0, 45), (360, 50)], "repeated azimuth"),
        ([], "at least one"),
        ([(np.nan, 45)], "finite azimuths"),
    ],
)
def test_normalize_errors(spec, match):
    with pytest.raises(ValueError, match=match):
        normalize_slopes(spec)


def test_interpolation_is_linear_and_circular():
    sectors = [(0, 40), (90, 50), (180, 40), (270, 30)]
    deg = np.degrees(slope_at_azimuth(np.radians([0, 45, 90, 315, 360, -45]), sectors))
    assert np.allclose(deg, [40, 45, 50, 35, 40, 35])


def test_steep_envelope_is_above_every_domain():
    domains = [[(0, 40), (180, 60)], [(90, 50), (270, 45)], [(0, 35)]]
    envelope = steep_envelope(domains)
    az = np.radians(np.arange(0, 360, 0.25))
    top = slope_at_azimuth(az, envelope)
    for sectors in domains:
        assert (top >= slope_at_azimuth(az, sectors) - 1e-12).all()


def test_slopes_differ():
    assert not slopes_differ([45])
    assert not slopes_differ([[(0, 45)], [(0, 45), (90, 45)]])
    assert slopes_differ([[(0, 45)], [(0, 46)]])


def test_discretize_one_slope_per_block():
    domains, domain_slopes = discretize_slopes([44.6, 45.4, 40.2, 39.9, 45.0])
    assert domain_slopes == [[(0.0, 40.0)], [(0.0, 45.0)]]
    assert domains.dtype == np.int32
    assert domains.tolist() == [1, 1, 0, 0, 1]


@pytest.mark.parametrize(("rounding", "expected"), [("nearest", 46.0), ("down", 45.0), ("up", 46.0)])
def test_discretize_rounding(rounding, expected):
    _, domain_slopes = discretize_slopes([45.6], rounding=rounding)
    assert domain_slopes[0][0][1] == expected


def test_discretize_step():
    domains, domain_slopes = discretize_slopes([41.2, 42.4, 47.4], step=5.0)
    assert [s[0][1] for s in domain_slopes] == [40.0, 45.0]
    assert domains.tolist() == [0, 0, 1]


def test_discretize_by_azimuth():
    slopes = np.array([[40.2, 50.1, 45.0, 44.8], [40.4, 49.9, 45.2, 45.1], [30.0, 30.0, 30.0, 30.0]])
    domains, domain_slopes = discretize_slopes(slopes, azimuths=[0, 90, 180, 270])
    assert len(domain_slopes) == 2
    assert domains[0] == domains[1] != domains[2]
    assert domain_slopes[domains[0]] == [(0.0, 40.0), (90.0, 50.0), (180.0, 45.0), (270.0, 45.0)]


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        (([45, 0.3],), {}, "rounds outside"),
        (([45, 89.8],), {}, "rounds outside"),
        (([45],), {"step": 0}, "step must be positive"),
        (([45],), {"rounding": "other"}, "rounding must be one of"),
        (([45],), {"azimuths": [0]}, "only used with"),
        (([[45, 45]],), {}, "need azimuths"),
        (([[45, 45]],), {"azimuths": [0]}, "expected 2 azimuths"),
        (([[45, 45]],), {"azimuths": [0, 360]}, "repeated azimuth"),
        (([np.nan],), {}, "finite"),
        ((np.zeros((1, 1, 1)),), {}, "shape"),
    ],
)
def test_discretize_errors(args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        discretize_slopes(*args, **kwargs)
