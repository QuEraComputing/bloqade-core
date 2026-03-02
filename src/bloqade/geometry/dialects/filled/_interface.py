from typing import Any, TypeVar

from kirin.dialects import ilist
from kirin.lowering import wraps as _wraps

from bloqade.geometry.dialects import grid

from .stmts import Fill, GetParent, Vacate
from .types import FilledGrid

Nx = TypeVar("Nx")
Ny = TypeVar("Ny")


@_wraps(Vacate)
def vacate(
    zone: grid.Grid[Nx, Ny],
    vacancies: ilist.IList[tuple[int, int], Any] | list[tuple[int, int]],
) -> FilledGrid[Nx, Ny]:
    """Create a FilledGrid by vacating specified positions from a grid.

    Args:
        zone: The original grid from which positions will be vacated.
        vacancies: An IList or list of (x_index, y_index) tuples indicating positions to vacate

    Returns:
        A FilledGrid with the specified vacancies.

    """
    ...


@_wraps(Fill)
def fill(
    zone: grid.Grid[Nx, Ny],
    filled: ilist.IList[tuple[int, int], Any] | list[tuple[int, int]],
) -> FilledGrid[Nx, Ny]:
    """Create a FilledGrid by filling specified positions in a grid.

    Args:
        zone: The original grid in which positions will be filled.
        filled: An IList or list of (x_index, y_index) tuples indicating positions to fill

    Returns:
        A FilledGrid with the specified positions filled.

    """
    ...


@_wraps(GetParent)
def get_parent(filled_grid: FilledGrid[Nx, Ny]) -> grid.Grid[Nx, Ny]:
    """Retrieve the parent grid of a FilledGrid.

    Args:
        filled_grid: The FilledGrid whose parent grid is to be retrieved.

    Returns:
        The parent grid of the provided FilledGrid.

    """
    ...
