from functools import cached_property

import pytest
from kirin.dialects import ilist

from bloqade.geometry.dialects.grid.types import Grid


class TestGrid:
    def _get_grid(self):
        return Grid(
            x_spacing=(1, 2, 3),
            y_spacing=(4, 5),
            x_init=1,
            y_init=2,
        )

    @cached_property
    def grid_obj(self):
        return self._get_grid()

    def test_is_equal(self):
        other_grid_obj = self._get_grid()

        assert self.grid_obj.is_equal(other_grid_obj)
        other_grid_obj.x_init = 2

        assert not self.grid_obj.is_equal(other_grid_obj)
        assert not self.grid_obj.is_equal(None)

    def test_grid_from_positions(self):
        x_positions = [1, 2, 4, 7]
        y_positions = [2, 6, 11]
        grid_obj = Grid.from_positions(x_positions, y_positions)
        assert grid_obj.is_equal(self.grid_obj)

    def test_grid_width(self):
        assert self.grid_obj.width == 6

    def test_grid_height(self):
        assert self.grid_obj.height == 9

    def test_grid_x_bounds(self):
        assert self.grid_obj.x_bounds() == (1, 7)

    def test_grid_y_bounds(self):
        assert self.grid_obj.y_bounds() == (2, 11)

    def test_grid_x_positions(self):
        assert self.grid_obj.x_positions == (1, 2, 4, 7)

    def test_grid_y_positions(self):
        assert self.grid_obj.y_positions == (2, 6, 11)

    @pytest.mark.parametrize(
        "ix, iy, expected",
        [
            (0, 0, (1, 2)),
            (1, 0, (2, 2)),
            (2, 0, (4, 2)),
            (0, 1, (1, 6)),
            (1, 1, (2, 6)),
            (2, 1, (4, 6)),
        ],
    )
    def test_grid_get(self, ix: int, iy: int, expected: tuple[float, float]):
        assert self.grid_obj.get((ix, iy)) == expected

    def test_unwrap(self):
        assert self.grid_obj.unwrap() == self.grid_obj

    def test_get_subgrid(self):
        subgrid = self.grid_obj.get_view(ilist.IList([0, 2]), ilist.IList([0, 2]))

        assert subgrid.x_spacing == (3,)
        assert subgrid.y_spacing == (9,)
        assert subgrid.x_init == 1
        assert subgrid.y_init == 2

    def test_get_view_out_of_bounds_raises(self):
        """Out-of-bounds indices (e.g. y=7 when grid has 3 rows) raise IndexError."""
        with pytest.raises(IndexError, match="Index out of range"):
            self.grid_obj.get_view(ilist.IList([0]), ilist.IList([7]))
        with pytest.raises(IndexError, match="Index out of range"):
            self.grid_obj.get_view(ilist.IList([15]), ilist.IList([0]))
        with pytest.raises(IndexError, match="Index out of range"):
            self.grid_obj.get_view(ilist.IList([0, 1]), ilist.IList([0, 3]))

    def test_get_view_negative_index_last(self):
        """-1 means last index (Python convention), not second-to-last."""
        # Grid has 4 x positions (0..3) and 3 y positions (0..2). -1 should be last.
        subgrid = self.grid_obj.get_view(ilist.IList([-1]), ilist.IList([-1]))
        # Last x position is 7, last y is 11; single-point subgrid has no spacing
        assert subgrid.x_init == 7
        assert subgrid.y_init == 11
        assert subgrid.x_positions == (7,)
        assert subgrid.y_positions == (11,)

    def test_get_view_indices_must_be_ascending(self):
        """Subgrid indices must be in strictly ascending order after normalization."""
        with pytest.raises(ValueError, match="strictly ascending order"):
            self.grid_obj.get_view(ilist.IList([2, 0]), ilist.IList([0]))
        with pytest.raises(ValueError, match="strictly ascending order"):
            self.grid_obj.get_view(ilist.IList([0, 1, 1]), ilist.IList([0]))
        with pytest.raises(ValueError, match="strictly ascending order"):
            self.grid_obj.get_view(ilist.IList([0]), ilist.IList([2, 1]))

    def test_getitem_with_plain_lists(self):
        """Grid __getitem__ accepts plain list[int] indices (same as get_view)."""
        subgrid = self.grid_obj[[0, 2], [0, 2]]
        expected = self.grid_obj.get_view(ilist.IList([0, 2]), ilist.IList([0, 2]))
        assert subgrid.x_spacing == expected.x_spacing
        assert subgrid.y_spacing == expected.y_spacing
        assert subgrid.x_init == expected.x_init
        assert subgrid.y_init == expected.y_init

    def test_shift(self):
        shifted_grid = self.grid_obj.shift(1, 2)
        expected_grid = Grid(
            x_spacing=(1, 2, 3),
            y_spacing=(4, 5),
            x_init=2,
            y_init=4,
        )
        assert shifted_grid.is_equal(expected_grid)

    @pytest.mark.parametrize(
        "x_indices, x_shift, expected_grid",
        [
            (
                ilist.IList([]),
                0,
                Grid(
                    x_spacing=(1, 2, 3),
                    y_spacing=(4, 5),
                    x_init=1,
                    y_init=2,
                ),
            ),
            (
                ilist.IList([0, 1]),
                1,
                Grid(
                    x_spacing=(1, 1, 3),
                    y_spacing=(4, 5),
                    x_init=2,
                    y_init=2,
                ),
            ),
            (
                ilist.IList([1]),
                1,
                Grid(
                    x_spacing=(2, 1, 3),
                    y_spacing=(4, 5),
                    x_init=1,
                    y_init=2,
                ),
            ),
            (
                ilist.IList([1, 2, 3]),
                1,
                Grid(
                    x_spacing=(2, 2, 3),
                    y_spacing=(4, 5),
                    x_init=1,
                    y_init=2,
                ),
            ),
            (
                slice(1, 4, 1),
                1,
                Grid(
                    x_spacing=(2, 2, 3),
                    y_spacing=(4, 5),
                    x_init=1,
                    y_init=2,
                ),
            ),
            (ilist.IList([1]), 3, None),
        ],
    )
    def test_shift_subgrid_x(self, x_indices, x_shift, expected_grid):
        if expected_grid is None:
            with pytest.raises(AssertionError):
                shifted_grid = self.grid_obj.shift_subgrid_x(x_indices, x_shift)
            return

        shifted_grid = self.grid_obj.shift_subgrid_x(x_indices, x_shift)
        assert shifted_grid.is_equal(expected_grid)

    @pytest.mark.parametrize(
        "y_indices, y_shift, expected_grid",
        [
            (
                ilist.IList([]),
                0,
                Grid(
                    x_spacing=(1, 2, 3),
                    y_spacing=(4, 5),
                    x_init=1,
                    y_init=2,
                ),
            ),
            (
                ilist.IList([0]),
                -1,
                Grid(
                    x_spacing=(1, 2, 3),
                    y_spacing=(5, 5),
                    x_init=1,
                    y_init=1,
                ),
            ),
            (
                ilist.IList([1]),
                1,
                Grid(
                    x_spacing=(1, 2, 3),
                    y_spacing=(5, 4),
                    x_init=1,
                    y_init=2,
                ),
            ),
            (
                ilist.IList([0, 2]),
                1,
                Grid(
                    x_spacing=(1, 2, 3),
                    y_spacing=(3, 6),
                    x_init=1,
                    y_init=3,
                ),
            ),
            (
                slice(0, 1, 1),
                -1,
                Grid(
                    x_spacing=(1, 2, 3),
                    y_spacing=(5, 5),
                    x_init=1,
                    y_init=1,
                ),
            ),
            (ilist.IList([0]), 5, None),
        ],
    )
    def test_shift_subgrid_y(self, y_indices, y_shift, expected_grid):

        if expected_grid is None:
            with pytest.raises(AssertionError):
                shifted_grid = self.grid_obj.shift_subgrid_y(y_indices, y_shift)
            return

        shifted_grid = self.grid_obj.shift_subgrid_y(y_indices, y_shift)
        assert shifted_grid.is_equal(expected_grid)

    def test_invalid_slice(self):
        with pytest.raises(TypeError):
            self.grid_obj[[1.5], ilist.IList([0, 1])]  # type: ignore

    def test_scale(self):
        scaled_grid = self.grid_obj.scale(2, 3)
        expected_grid = Grid(
            x_spacing=(2, 4, 6),
            y_spacing=(12, 15),
            x_init=1,
            y_init=2,
        )
        assert scaled_grid.is_equal(expected_grid)

    def test_repeat(self):
        repeated_grid = self.grid_obj.repeat(2, 3, 0.5, 2.1)
        expected_grid = Grid(
            x_spacing=(1, 2, 3, 0.5, 1, 2, 3),
            y_spacing=(4, 5, 2.1, 4, 5, 2.1, 4, 5),
            x_init=1,
            y_init=2,
        )
        assert repeated_grid.is_equal(expected_grid)

    @pytest.mark.parametrize(
        "x_init, y_init",
        [(None, None), (1, 2), (3, 4), (5, None), (None, 6)],
    )
    def test_set_init(self, x_init: float | None, y_init: float | None):
        new_grid = self.grid_obj.set_init(x_init, y_init)
        expected_grid = Grid(
            x_spacing=(1, 2, 3),
            y_spacing=(4, 5),
            x_init=x_init,
            y_init=y_init,
        )

        assert new_grid.is_equal(expected_grid)

    def test_empty_positions_x(self):
        grid_obj = Grid.from_positions([], [1])
        assert grid_obj.x_positions == ()
        assert grid_obj.y_positions == (1,)
        with pytest.raises(ValueError):
            grid_obj.x_bounds()
        assert grid_obj.y_bounds() == (1, 1)
        assert grid_obj.width == 0
        assert grid_obj.height == 0

    def test_empty_positions_y(self):
        grid_obj = Grid.from_positions([1], [])
        assert grid_obj.x_positions == (1,)
        assert grid_obj.y_positions == ()
        with pytest.raises(ValueError):
            grid_obj.y_bounds()
        assert grid_obj.x_bounds() == (1, 1)
        assert grid_obj.width == 0
        assert grid_obj.height == 0

    def test_row_x_pos(self):
        grid_obj = Grid.from_positions([1, 2, 3], [4, 5, 6])
        assert grid_obj.row_x_pos(0) == ilist.IList([1, 2, 3])

    def test_col_y_pos(self):
        grid_obj = Grid.from_positions([1, 2, 3], [4, 5, 6])
        assert grid_obj.col_y_pos(1) == ilist.IList([4, 5, 6])


class TestSubGrid(TestGrid):
    def _get_grid(self):
        grid_obj = super()._get_grid()
        return grid_obj.get_view(ilist.IList([0, 1, 2, 3]), ilist.IList([0, 1, 2]))
