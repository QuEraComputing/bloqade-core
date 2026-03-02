from kirin.dialects import ilist
from kirin.interp import (
    Frame,
    Interpreter,
    MethodTable,
    impl,
)

from bloqade.geometry.dialects.grid.types import Grid

from . import stmts
from ._dialect import dialect
from .types import FilledGrid


@dialect.register
class FilledGridMethods(MethodTable):

    @impl(stmts.Vacate)
    def vacate(self, interp: Interpreter, frame: Frame, stmt: stmts.Vacate):
        zone = frame.get_casted(stmt.zone, Grid)
        vacancies_val = frame.get(stmt.vacancies)
        vacancies = (
            vacancies_val
            if isinstance(vacancies_val, ilist.IList)
            else ilist.IList(vacancies_val)
        )
        return (FilledGrid.vacate(zone, vacancies),)

    @impl(stmts.Fill)
    def fill(self, interp: Interpreter, frame: Frame, stmt: stmts.Fill):
        zone = frame.get_casted(stmt.zone, Grid)
        filled_val = frame.get(stmt.filled)
        filled = (
            filled_val
            if isinstance(filled_val, ilist.IList)
            else ilist.IList(filled_val)
        )
        return (FilledGrid.fill(zone, filled),)

    @impl(stmts.GetParent)
    def get_parent(self, interp: Interpreter, frame: Frame, stmt: stmts.GetParent):
        filled_grid = frame.get_casted(stmt.filled_grid, FilledGrid)
        return (filled_grid.parent,)
