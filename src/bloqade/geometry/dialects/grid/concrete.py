from kirin.dialects import ilist
from kirin.interp import Frame, Interpreter, MethodTable, impl

from . import stmts
from ._dialect import dialect
from .types import Grid


@dialect.register
class GridInterpreter(MethodTable):
    @impl(stmts.FromPositions)
    def from_positions(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.FromPositions,
    ):
        return (
            Grid.from_positions(
                x_positions=frame.get(stmt.x_positions),
                y_positions=frame.get(stmt.y_positions),
            ),
        )

    @impl(stmts.FromRanges)
    def from_ranges(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.FromRanges,
    ):
        x_positions = list(
            map(
                float,
                range(
                    frame.get(stmt.x_start),
                    frame.get(stmt.x_stop),
                    frame.get(stmt.x_step),
                ),
            )
        )
        y_positions = list(
            map(
                float,
                range(
                    frame.get(stmt.y_start),
                    frame.get(stmt.y_stop),
                    frame.get(stmt.y_step),
                ),
            )
        )
        return (
            Grid.from_positions(
                x_positions=x_positions,
                y_positions=y_positions,
            ),
        )

    @impl(stmts.New)
    def new(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.New,
    ):
        return (
            Grid(
                x_spacing=frame.get(stmt.x_spacing),
                y_spacing=frame.get(stmt.y_spacing),
                x_init=frame.get(stmt.x_init),
                y_init=frame.get(stmt.y_init),
            ),
        )

    @impl(stmts.Shape)
    def shape(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.Shape,
    ):
        return (frame.get_casted(stmt.zone, Grid).shape,)

    @impl(stmts.Get)
    def get(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.Get,
    ):
        idx = frame.get_casted(stmt.idx, tuple)

        return (frame.get_casted(stmt.zone, Grid).get(idx),)

    @impl(stmts.GetXPos)
    def get_x_pos(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.GetXPos,
    ):
        return (frame.get_casted(stmt.zone, Grid).x_positions,)

    @impl(stmts.GetYPos)
    def get_y_pos(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.GetYPos,
    ):
        return (frame.get_casted(stmt.zone, Grid).y_positions,)

    @impl(stmts.GetSubGrid)
    def get_view(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.GetSubGrid,
    ):
        x_indices = frame.get_casted(stmt.x_indices, ilist.IList)
        y_indices = frame.get_casted(stmt.y_indices, ilist.IList)

        return (frame.get_casted(stmt.zone, Grid).get_view(x_indices, y_indices),)

    @impl(stmts.GetXBounds)
    def get_x_bounds(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.GetXBounds,
    ):
        return (frame.get_casted(stmt.zone, Grid).x_bounds(),)

    @impl(stmts.GetYBounds)
    def get_y_bounds(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.GetYBounds,
    ):
        return (frame.get_casted(stmt.zone, Grid).y_bounds(),)

    @impl(stmts.Shift)
    def shift(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.Shift,
    ):
        grid = frame.get_casted(stmt.zone, Grid)
        x_shift = frame.get_casted(stmt.x_shift, float)
        y_shift = frame.get_casted(stmt.y_shift, float)

        return (grid.shift(x_shift, y_shift),)

    @impl(stmts.ShiftSubgridX)
    def shift_subgrid_x(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.ShiftSubgridX,
    ):
        grid = frame.get_casted(stmt.zone, Grid)
        x_indices = frame.get_casted(stmt.x_indices, ilist.IList)
        x_shift = frame.get_casted(stmt.x_shift, float)

        return (grid.shift_subgrid_x(x_indices, x_shift),)

    @impl(stmts.ShiftSubgridY)
    def shift_subgrid_y(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.ShiftSubgridY,
    ):
        grid = frame.get_casted(stmt.zone, Grid)
        y_indices = frame.get_casted(stmt.y_indices, ilist.IList)
        y_shift = frame.get_casted(stmt.y_shift, float)

        return (grid.shift_subgrid_y(y_indices, y_shift),)

    @impl(stmts.Scale)
    def scale(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.Scale,
    ):
        grid = frame.get_casted(stmt.zone, Grid)
        x_scale = frame.get_casted(stmt.x_scale, float)
        y_scale = frame.get_casted(stmt.y_scale, float)

        return (grid.scale(x_scale, y_scale),)

    @impl(stmts.Repeat)
    def repeat(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.Repeat,
    ):
        grid = frame.get_casted(stmt.zone, Grid)
        x_times = frame.get_casted(stmt.x_times, int)
        y_times = frame.get_casted(stmt.y_times, int)
        x_gap = frame.get_casted(stmt.x_gap, float)
        y_gap = frame.get_casted(stmt.y_gap, float)

        return (grid.repeat(x_times, y_times, x_gap, y_gap),)

    @impl(stmts.Positions)
    def positions(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.Positions,
    ):
        grid = frame.get_casted(stmt.zone, Grid)
        return (grid.positions,)

    @impl(stmts.RowXPos)
    def row_x_pos(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.RowXPos,
    ):
        grid = frame.get_casted(stmt.zone, Grid)
        row_index = frame.get_casted(stmt.row_index, int)
        return (grid.row_x_pos(row_index),)

    @impl(stmts.ColYPos)
    def col_y_pos(
        self,
        interp: Interpreter,
        frame: Frame,
        stmt: stmts.ColYPos,
    ):
        grid = frame.get_casted(stmt.zone, Grid)
        column_index = frame.get_casted(stmt.column_index, int)
        return (grid.col_y_pos(column_index),)
