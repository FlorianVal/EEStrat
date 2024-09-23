import pytest
from src.utils import discretize


def test_discretize_basic():
    assert discretize(0.6, 3) == 0.5  # 3 step is 0, 0.5, 1
    assert discretize(0.7, 5) == 0.75  # 5 steps is 0, 0.25, 0.5, 0.75, 1
    assert round(discretize(0.3, 4).item(), 2) == 0.33


def test_discretize_edge_cases():
    assert discretize(0, 2) == 0
    assert discretize(1, 2) == 1
    assert discretize(0.49, 2) == 0
    assert discretize(0.51, 2) == 1


def test_discretize_many_steps():
    assert discretize(0.1001, 10001) == 0.1001


def test_discretize_value_out_of_range():
    with pytest.raises(ValueError, match="All values must be between 0 and 1, but found values in range"):
        discretize(-0.1, 3)
    with pytest.raises(ValueError, match="All values must be between 0 and 1, but found values in range"):
        discretize(1.1, 3)


def test_discretize_invalid_steps():
    with pytest.raises(ValueError, match="Number of steps must be at least 2"):
        discretize(0.5, 1)


def test_discretize_float_precision():
    assert abs(discretize(1 / 3, 3) - 0.5) < 1e-10