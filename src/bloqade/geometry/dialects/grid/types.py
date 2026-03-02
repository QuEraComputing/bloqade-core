import dataclasses
from functools import cached_property
from itertools import chain, product
from typing import Any, Generic, Literal, Sequence, TypeVar, overload

from kirin import ir, types
from kirin.dialects import ilist
from kirin.print.printer import Printer
from kirin.serialization.base.deserializer import Deserializer
from kirin.serialization.base.serializer import Serializer
from kirin.serialization.core.serializationunit import SerializationUnit

from ._dialect import dialect

NumX = TypeVar("NumX")
NumY = TypeVar("NumY")


def get_indices(size: int, index: Any) -> ilist.IList[int, Any]:
    if isinstance(index, slice):
        return ilist.IList(range(size)[index])
    elif isinstance(index, int):
        if index < 0:
            index += size

        if index < 0 or index >= size:
            raise IndexError("Index out of range")

        return ilist.IList([index])

    index = ilist.IList(list(index))
    if any(not isinstance(i, int) for i in index.data):
        raise TypeError("Index must be an int, slice, or Sequence of ints")

    return index


@dataclasses.dataclass
@dialect.register
class Grid(ir.Data["Grid"], Generic[NumX, NumY]):
    name = "grid"

    x_spacing: tuple[float, ...]
    """A tuple of x spacings between grid points."""
    y_spacing: tuple[float, ...]
    """A tuple of y spacings between grid points."""
    x_init: float | None
    """The initial x position of the grid, or None if not set."""
    y_init: float | None
    """The initial y position of the grid, or None if not set."""

    def __post_init__(self):
        assert all(ele >= 0 for ele in self.x_spacing)
        assert all(ele >= 0 for ele in self.y_spacing)
        self.type = types.Generic(
            Grid,
            types.Literal(len(self.x_spacing) + 1),
            types.Literal(len(self.y_spacing) + 1),
        )

    def __repr__(self):
        return (
            f"Grid({self.x_spacing!r}, "
            f"{self.y_spacing!r}, "
            f"{self.x_init!r}, {self.y_init!r})"
        )

    def is_equal(self, other: Any) -> bool:
        """Check if two grid geometry are equal."""
        if not isinstance(other, Grid):
            return False
        return (
            self.x_spacing == other.x_spacing
            and self.y_spacing == other.y_spacing
            and self.x_init == other.x_init
            and self.y_init == other.y_init
        )

    @classmethod
    def from_positions(
        cls,
        x_positions: Sequence[float],
        y_positions: Sequence[float],
    ):
        """Create a grid from sequence of x and y positions.

        Args:

            x_positions (Sequence[float]): The x positions.
            y_positions (Sequence[float]): The y positions.

        Returns:
            Grid: A grid object with the specified x and y positions.
        """
        x_init = x_positions[0] if len(x_positions) > 0 else None
        y_init = y_positions[0] if len(y_positions) > 0 else None

        if len(x_positions) > 1:
            x_spacing = tuple(
                x_positions[i + 1] - x_positions[i] for i in range(len(x_positions) - 1)
            )
        else:
            x_spacing = ()

        if len(y_positions) > 1:
            y_spacing = tuple(
                y_positions[i + 1] - y_positions[i] for i in range(len(y_positions) - 1)
            )
        else:
            y_spacing = ()

        return cls(x_spacing, y_spacing, x_init, y_init)

    @cached_property
    def shape(self) -> tuple[int, int]:
        """Shape of the grid, which is (num_x, num_y).

        Note:
            if x_init or y_init is None, num_x or num_y will be 0 respectively.

        """
        num_x = 0 if self.x_init is None else len(self.x_spacing) + 1
        num_y = 0 if self.y_init is None else len(self.y_spacing) + 1
        return (num_x, num_y)

    @cached_property
    def width(self):
        """Width of the grid, which is the sum of `x_spacing`."""
        return sum(self.x_spacing)

    @cached_property
    def height(self):
        """Height of the grid, which is the sum of `y_spacing`."""
        return sum(self.y_spacing)

    def x_bounds(self):
        """X bounds of the grid, which is `(x_init, x_init + width)`.

        Raises:
            ValueError: If x_init is None, cannot compute bounds.

        """
        if self.x_init is None:
            raise ValueError("x_init is None, cannot compute bounds")

        return (self.x_init, self.x_init + self.width)

    def y_bounds(self):
        """Y bounds of the grid, which is `(y_init, y_init + height)`.

        Raises:
            ValueError: If y_init is None, cannot compute bounds.

        """
        if self.y_init is None:
            raise ValueError("y_init is None, cannot compute bounds")

        return (self.y_init, self.y_init + self.height)

    @cached_property
    def x_positions(self) -> tuple[float, ...]:
        """X positions of the grid.

        Note:
            If `x_init` is None, returns an empty tuple.

        """
        if self.x_init is None:
            return ()
        return tuple(
            chain(
                [pos := self.x_init],
                (pos := pos + spacing for spacing in self.x_spacing),
            )
        )

    @cached_property
    def y_positions(self) -> tuple[float, ...]:
        """Y positions of the grid.

        Note:
            If `y_init` is None, returns an empty tuple.

        """
        if self.y_init is None:
            return ()

        return tuple(
            chain(
                [pos := self.y_init],
                (pos := pos + spacing for spacing in self.y_spacing),
            )
        )

    @cached_property
    def positions(self) -> ilist.IList[tuple[float, float], Any]:
        """All positions in the grid as a list of tuples (x, y) in lexicographic order."""
        return ilist.IList(tuple(product(self.x_positions, self.y_positions)))

    def get(self, idx: tuple[int, int]) -> tuple[float, float]:
        """Get the (x, y) position at the specified grid index.

        Args:
            idx (tuple[int, int]): The (x, y) index in the grid.

        Returns:
            tuple[float, float]: The (x, y) position in the grid.
        """
        return (self.x_positions[idx[0]], self.y_positions[idx[1]])

    Nx = TypeVar("Nx")
    Ny = TypeVar("Ny")

    @overload
    def get_view(
        self, x_indices: ilist.IList[int, Nx], y_indices: ilist.IList[int, Ny]
    ) -> "Grid[Nx, Ny]": ...

    @overload
    def get_view(
        self, x_indices: Sequence[int], y_indices: ilist.IList[int, Ny]
    ) -> "Grid[Any, Ny]": ...

    @overload
    def get_view(
        self, x_indices: ilist.IList[int, Nx], y_indices: Sequence[int]
    ) -> "Grid[Nx, Any]": ...

    @overload
    def get_view(
        self, x_indices: Sequence[int], y_indices: Sequence[int]
    ) -> "Grid[Any, Any]": ...

    def get_view(self, x_indices, y_indices) -> "Grid":
        """Get a sub-grid view based on the specified x and y indices.

        Args:
            x_indices (Sequence[int]): The x indices to include in the sub-grid.
            y_indices (Sequence[int]): The y indices to include in the sub-grid.

        Returns:
            Grid: The sub-grid view.
        """
        if isinstance(x_indices, ilist.IList):
            x_indices = x_indices.data

        if isinstance(y_indices, ilist.IList):
            y_indices = y_indices.data

        return SubGrid(
            parent=self,
            x_indices=ilist.IList(x_indices),
            y_indices=ilist.IList(y_indices),
        )

    @overload
    def __getitem__(
        self, indices: tuple[int, int]
    ) -> "Grid[Literal[1], Literal[1]]": ...
    @overload
    def __getitem__(
        self, indices: tuple[int, slice | list[int]]
    ) -> "Grid[Literal[1], Any]": ...

    @overload
    def __getitem__(
        self, indices: tuple[int, ilist.IList[int, Ny]]
    ) -> "Grid[Literal[1], Ny]": ...
    @overload
    def __getitem__(
        self, indices: tuple[slice | list[int], int]
    ) -> "Grid[Any, Literal[1]]": ...
    @overload
    def __getitem__(
        self, indices: tuple[slice | list[int], slice]
    ) -> "Grid[Any, Any]": ...

    @overload
    def __getitem__(
        self, indices: tuple[slice | list[int], ilist.IList[int, Ny]]
    ) -> "Grid[Any, Ny]": ...
    @overload
    def __getitem__(
        self, indices: tuple[ilist.IList[int, Nx], int]
    ) -> "Grid[Nx, Literal[1]]": ...

    @overload
    def __getitem__(
        self, indices: tuple[ilist.IList[int, Nx], slice | list[int]]
    ) -> "Grid[Nx, Any]": ...

    @overload
    def __getitem__(
        self, indices: tuple[ilist.IList[int, Nx], ilist.IList[int, Ny]]
    ) -> "Grid[Nx, Ny]": ...

    @overload
    def __getitem__(self, indices: tuple[list[int], list[int]]) -> "Grid[Any, Any]": ...

    def __getitem__(self, indices):
        if len(indices) != 2:
            raise IndexError("Grid indexing requires two indices (x, y)")

        x_index, y_index = indices
        x_indices = get_indices(len(self.x_spacing) + 1, x_index)
        y_indices = get_indices(len(self.y_spacing) + 1, y_index)

        return self.get_view(x_indices=x_indices, y_indices=y_indices)

    def __hash__(self) -> int:
        return hash((self.x_spacing, self.y_spacing, self.x_init, self.y_init))

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Grid)
            and self.x_spacing == other.x_spacing
            and self.y_spacing == other.y_spacing
            and self.x_init == other.x_init
            and self.y_init == other.y_init
        )

    def print_impl(self, printer: Printer) -> None:
        printer.plain_print("Grid(")
        printer.print(self.x_spacing)
        printer.plain_print(", ")
        printer.print(self.y_spacing)
        printer.plain_print(", ")
        printer.print(self.x_init)
        printer.plain_print(", ")
        printer.print(self.y_init)
        printer.plain_print(")")

    def unwrap(self):
        return self

    def scale(self, x_scale: float, y_scale: float) -> "Grid[NumX, NumY]":
        """Scale the grid spacings by the specified x and y factors with fixed x and y initial positions.

        Args:
            x_scale (float): The scaling factor for the x spacings.
            y_scale (float): The scaling factor for the y spacings.

        Returns:
            Grid[NumX, NumY]: A new grid with scaled x and y spacings

        """
        return Grid(
            x_spacing=tuple(spacing * x_scale for spacing in self.x_spacing),
            y_spacing=tuple(spacing * y_scale for spacing in self.y_spacing),
            x_init=self.x_init,
            y_init=self.y_init,
        )

    def set_init(
        self, x_init: float | None, y_init: float | None
    ) -> "Grid[NumX, NumY]":
        """Set the initial positions of the grid.

        Args:
            x_init (float | None): The new initial x position. If None, the grid
                will not have an initial x position.
            y_init (float | None): The new initial y position. If None, the grid
                will not have an initial y position.

        Returns:
            Grid[NumX, NumY]: A new grid with the specified initial positions.

        """
        return Grid(self.x_spacing, self.y_spacing, x_init, y_init)

    def shift(self, x_shift: float, y_shift: float) -> "Grid[NumX, NumY]":
        """Shift the grid by the specified x and y amounts.

        Args:
            x_shift (float): The amount to shift the grid in the x direction.
            y_shift (float): The amount to shift the grid in the y direction.

        Returns:
            Grid[NumX, NumY]: A new grid with the specified shifts applied to the initial positions.

        """
        return Grid(
            x_spacing=self.x_spacing,
            y_spacing=self.y_spacing,
            x_init=self.x_init + x_shift if self.x_init is not None else None,
            y_init=self.y_init + y_shift if self.y_init is not None else None,
        )

    def shift_subgrid_x(
        self, x_indices: ilist.IList[int, Nx] | slice, x_shift: float
    ) -> "Grid[NumX, NumY]":
        """Shift a sub grid of grid in the x directions.

        Args:
            grid (Grid): a grid object
            x_indices (float): a list/ilist of x indices to shift
            x_shift (float): shift in the x direction
        Returns:
            Grid: a new grid object that has been shifted
        """
        indices = get_indices(len(self.x_spacing) + 1, x_indices)

        def shift_x(index):
            new_spacing = self.x_spacing[index]
            if index in indices and (index + 1) not in indices:
                new_spacing -= x_shift
            elif index not in indices and (index + 1) in indices:
                new_spacing += x_shift
            return new_spacing

        new_spacing = tuple(shift_x(i) for i in range(len(self.x_spacing)))

        assert all(
            x >= 0 for x in new_spacing
        ), "Invalid shift: column order changes after shift."

        x_init = self.x_init
        if x_init is not None and 0 in indices:
            x_init += x_shift

        return Grid(
            x_spacing=new_spacing,
            y_spacing=self.y_spacing,
            x_init=x_init,
            y_init=self.y_init,
        )

    def shift_subgrid_y(
        self, y_indices: ilist.IList[int, Ny] | slice, y_shift: float
    ) -> "Grid[NumX, NumY]":
        """Shift a sub grid of grid in the y directions.

        Args:
            grid (Grid): a grid object
            y_indices (float): a list/ilist of y indices to shift
            y_shift (float): shift in the y direction
        Returns:
            Grid: a new grid object that has been shifted
        """
        indices = get_indices(len(self.y_spacing) + 1, y_indices)

        def shift_y(index):
            new_spacing = self.y_spacing[index]
            if index in indices and (index + 1) not in indices:
                new_spacing -= y_shift
            elif index not in indices and (index + 1) in indices:
                new_spacing += y_shift
            return new_spacing

        new_spacing = tuple(shift_y(i) for i in range(len(self.y_spacing)))

        assert all(
            y >= 0 for y in new_spacing
        ), "Invalid shift: row order changes after shift."

        y_init = self.y_init
        if y_init is not None and 0 in indices:
            y_init += y_shift

        return Grid(
            x_spacing=self.x_spacing,
            y_spacing=new_spacing,
            x_init=self.x_init,
            y_init=y_init,
        )

    def repeat(
        self, x_times: int, y_times: int, x_gap: float, y_gap: float
    ) -> "Grid[NumX, NumY]":
        """Repeat the grid in both x and y directions with specified gaps.

        Args:
            x_times (int): The number of times to repeat the grid in the x direction.
            y_times (int): The number of times to repeat the grid in the y direction.
            x_gap (float): The gap between repeated grids in the x direction.
            y_gap (float): The gap between repeated grids in the y direction.

        Returns:
            Grid[NumX, NumY]: A new grid with the specified repetitions and gaps.

        """

        if x_times < 1 or y_times < 1:
            raise ValueError("x_times and y_times must be non-negative")

        return Grid(
            x_spacing=sum((self.x_spacing + (x_gap,) for _ in range(x_times - 1)), ())
            + self.x_spacing,
            y_spacing=sum((self.y_spacing + (y_gap,) for _ in range(y_times - 1)), ())
            + self.y_spacing,
            x_init=self.x_init,
            y_init=self.y_init,
        )

    def row_x_pos(self, row_index: int) -> ilist.IList[float, NumX]:
        """Get the x positions of a specific row in the grid.

        Args:
            row_index (int): The index of the row.

        Returns:
            IList[float, NumX]: The x positions of the specified row.
        """
        return ilist.IList(list(self.x_positions))

    def col_y_pos(self, column_index: int) -> ilist.IList[float, NumY]:
        """Get the y positions of a specific column in the grid.

        Args:
            column_index (int): The index of the column.

        Returns:
            IList[float, NumY]: The y positions of the specified column.
        """
        return ilist.IList(list(self.y_positions))

    def serialize(self, serializer: "Serializer") -> "SerializationUnit":

        return SerializationUnit(
            kind="grid",
            module_name=dialect.name,
            class_name=Grid.__name__,
            data={
                "x_init": serializer.serialize(self.x_init),
                "y_init": serializer.serialize(self.y_init),
                "x_spacing": serializer.serialize(self.x_spacing),
                "y_spacing": serializer.serialize(self.y_spacing),
            },
        )

    @classmethod
    def deserialize(
        cls, serUnit: "SerializationUnit", deserializer: "Deserializer"
    ) -> "Grid":
        x_init = deserializer.deserialize(serUnit.data["x_init"])
        y_init = deserializer.deserialize(serUnit.data["y_init"])
        x_spacing = deserializer.deserialize(serUnit.data["x_spacing"])
        y_spacing = deserializer.deserialize(serUnit.data["y_spacing"])
        return Grid(
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            x_init=x_init,
            y_init=y_init,
        )


@dataclasses.dataclass
@dialect.register
class SubGrid(Grid[NumX, NumY]):
    """A sub-grid view of a parent grid with specified x and y indices.

    For API documentation see the `Grid` class.

    """

    name = "sub_grid"

    x_spacing: tuple[float, ...] = dataclasses.field(init=False)
    y_spacing: tuple[float, ...] = dataclasses.field(init=False)
    x_init: float | None = dataclasses.field(init=False)
    y_init: float | None = dataclasses.field(init=False)

    parent: Grid[Any, Any]
    x_indices: ilist.IList[int, NumX]
    y_indices: ilist.IList[int, NumY]

    def __post_init__(self):
        if len(self.x_indices) == 0 or len(self.y_indices) == 0:
            raise ValueError("Indices cannot be empty")

        self.x_spacing = tuple(
            sum(self.parent.x_spacing[start:end])
            for start, end in zip(self.x_indices[:-1], self.x_indices[1:])
        )

        self.y_spacing = tuple(
            sum(self.parent.y_spacing[start:end])
            for start, end in zip(self.y_indices[:-1], self.y_indices[1:])
        )
        if self.parent.x_init is not None:
            self.x_init = self.parent.x_init + sum(
                self.parent.x_spacing[: self.x_indices[0]]
            )
        else:
            self.x_init = None

        if self.parent.y_init is not None:
            self.y_init = self.parent.y_init + sum(
                self.parent.y_spacing[: self.y_indices[0]]
            )
        else:
            self.y_init = None

        self.type = types.Generic(
            SubGrid,
            types.Literal(len(self.x_indices)),
            types.Literal(len(self.y_indices)),
        )

    def get_view(self, x_indices, y_indices):
        return self.parent.get_view(
            x_indices=ilist.IList([self.x_indices[x_index] for x_index in x_indices]),
            y_indices=ilist.IList([self.y_indices[y_index] for y_index in y_indices]),
        )

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other: Any) -> bool:
        return super().__eq__(other)

    def __repr__(self):
        return super().__repr__()

    def serialize(self, serializer: "Serializer") -> "SerializationUnit":
        return SerializationUnit(
            kind="sub_grid",
            module_name=dialect.name,
            class_name=SubGrid.__name__,
            data={
                "parent": serializer.serialize_attribute(self.parent),
                "x_indices": serializer.serialize(self.x_indices),
                "y_indices": serializer.serialize(self.y_indices),
            },
        )

    @classmethod
    def deserialize(
        cls, serUnit: "SerializationUnit", deserializer: "Deserializer"
    ) -> "SubGrid":
        parent = deserializer.deserialize(serUnit.data["parent"])
        x_indices = deserializer.deserialize(serUnit.data["x_indices"])
        y_indices = deserializer.deserialize(serUnit.data["y_indices"])
        return SubGrid(parent=parent, x_indices=x_indices, y_indices=y_indices)


GridType = types.Generic(Grid, types.TypeVar("Nx"), types.TypeVar("Ny"))
