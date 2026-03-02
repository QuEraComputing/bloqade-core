"""Tests that run the Kirin interpreter by creating kernels with the @geometry decorator."""

import pytest
from kirin.dialects import ilist

from bloqade.geometry import grid
from bloqade.geometry.prelude import geometry


def test_kernel_sub_grid_negative_one_last_index():
    """Kernel: sub_grid(zone, [-1], [3]) uses -1 as last x-index (geometry 0..4)."""
    zone = grid.Grid.from_positions(
        x_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
        y_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
    )

    @geometry
    def kernel():
        z = grid.from_positions(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )
        return grid.sub_grid(z, [-1], [3])

    result = kernel()
    assert result.x_positions == (4.0,)
    assert result.y_positions == (3.0,)
    assert result.x_init == 4.0
    assert result.y_init == 3.0
    assert result.is_equal(zone.get_view(ilist.IList([-1]), ilist.IList([3])))


def test_kernel_sub_grid_out_of_bounds_raises():
    """Kernel: sub_grid(zone, [15], [7]) raises IndexError (geometry 0..4)."""

    @geometry
    def kernel():
        z = grid.from_positions(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )
        return grid.sub_grid(z, [15], [7])

    with pytest.raises(IndexError, match="Index out of range"):
        kernel()


def test_kernel_sub_grid_valid_returns_same_as_direct():
    """Kernel sub_grid result matches direct Grid.get_view for valid indices."""
    zone = grid.Grid.from_positions(
        x_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
        y_positions=[0.0, 1.0, 2.0, 3.0, 4.0],
    )

    @geometry
    def kernel():
        z = grid.from_positions(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )
        return grid.sub_grid(z, [0, 2], [1, 3])

    result = kernel()
    expected = zone.get_view(ilist.IList([0, 2]), ilist.IList([1, 3]))
    assert result.is_equal(expected)
    assert result.x_positions == (0.0, 2.0)
    assert result.y_positions == (1.0, 3.0)
