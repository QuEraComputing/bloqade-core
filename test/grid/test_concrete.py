from functools import cached_property

import pytest
from kirin import interp, ir
from kirin.dialects import ilist

from bloqade.geometry.dialects import grid


class TestGridInterpreter:
    def _get_grid(self):
        return grid.Grid.from_positions([1, 2], [3, 4])

    @cached_property
    def grid_obj(self):
        return self._get_grid()

    def init_interpreter(self):
        interpreter = interp.Interpreter(ir.DialectGroup([grid]))
        interpreter.initialize()
        return interpreter

    def run_stmt(self, stmt_type: type[ir.Statement], *values):
        interpreter = self.init_interpreter()
        ssa_values = tuple(ir.TestValue() for _ in values)
        new_stmt = stmt_type(*ssa_values)
        with interpreter.new_frame(new_stmt) as frame:
            frame.set_values(ssa_values, values)
            return interpreter.frame_eval(frame, new_stmt)

    def test_from_positions(self):
        expected_grid_obj = grid.Grid.from_positions(
            x_positions := [1, 2], y_positions := [3, 4]
        )
        result = self.run_stmt(grid.FromPositions, x_positions, y_positions)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert expected_grid_obj.is_equal(result[0])

    def test_new(self):
        x_spacing = (1, 2)
        y_spacing = (3, 4)
        x_init = 5
        y_init = 6
        expected_grid_obj = grid.Grid(x_spacing, y_spacing, x_init, y_init)

        result = self.run_stmt(grid.New, x_spacing, y_spacing, x_init, y_init)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert expected_grid_obj.is_equal(result[0])

    def test_from_ranges(self):
        x_start = 1
        x_stop = 5
        x_step = 1
        y_start = 2
        y_stop = 6
        y_step = 1

        expected_grid_obj = grid.Grid.from_positions(
            [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]
        )

        result = self.run_stmt(
            grid.FromRanges, x_start, x_stop, x_step, y_start, y_stop, y_step
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert expected_grid_obj.is_equal(result[0])

    @pytest.mark.parametrize(
        ("stmt_type", "method_name", "args"),
        [
            (grid.GetXBounds, "x_bounds", ()),
            (grid.GetYBounds, "y_bounds", ()),
            (grid.GetXPos, "x_positions", ()),
            (grid.GetYPos, "y_positions", ()),
            (grid.Get, "get", ((1, 0),)),
            (grid.Shift, "shift", (1.0, 2.0)),
            (grid.ShiftSubgridX, "shift_subgrid_x", (ilist.IList([0]), -1)),
            (grid.ShiftSubgridY, "shift_subgrid_y", (ilist.IList([0]), -1)),
            (grid.Scale, "scale", (1.0, 2.0)),
            (grid.Repeat, "repeat", (1, 2, 0.5, 1.0)),
            (grid.GetSubGrid, "get_view", (ilist.IList((0,)), ilist.IList((1,)))),
            (grid.Shape, "shape", ()),
        ],
    )
    def test_template(self, stmt_type, method_name, args):
        prop = getattr(self.grid_obj, method_name)
        if callable(prop):
            expected = prop(*args)
        else:
            assert len(args) == 0
            expected = prop

        result = self.run_stmt(stmt_type, self.grid_obj, *args)

        assert result == (expected,)
