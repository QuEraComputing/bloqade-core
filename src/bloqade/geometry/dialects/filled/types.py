from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from typing import Any, TypeVar

from kirin import types
from kirin.dialects import ilist
from kirin.serialization.base.deserializer import Deserializer
from kirin.serialization.base.serializer import Serializer
from kirin.serialization.core.serializationunit import SerializationUnit

from bloqade.geometry.dialects import grid

from ._dialect import dialect

NumX = TypeVar("NumX")
NumY = TypeVar("NumY")


@dataclass(eq=False)
@dialect.register
class FilledGrid(grid.Grid[NumX, NumY]):
    name = "filled_grid"

    x_spacing: tuple[float, ...] = field(init=False)
    y_spacing: tuple[float, ...] = field(init=False)
    x_init: float | None = field(init=False)
    y_init: float | None = field(init=False)

    parent: grid.Grid[NumX, NumY]
    vacancies: frozenset[tuple[int, int]]

    def __post_init__(self):
        self.x_spacing = self.parent.x_spacing
        self.y_spacing = self.parent.y_spacing
        self.x_init = self.parent.x_init
        self.y_init = self.parent.y_init

        self.type = types.Generic(
            FilledGrid,
            types.Literal(len(self.x_spacing) + 1),
            types.Literal(len(self.y_spacing) + 1),
        )

    def __hash__(self):
        return hash((self.parent, self.vacancies))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FilledGrid)
            and self.parent == other.parent
            and self.vacancies == other.vacancies
        )

    def is_equal(self, other: Any) -> bool:
        return self == other

    @cached_property
    def positions(self) -> ilist.IList[tuple[float, float], Any]:
        positions = tuple(
            (x, y)
            for (ix, x), (iy, y) in product(
                enumerate(self.x_positions), enumerate(self.y_positions)
            )
            if (ix, iy) not in self.vacancies
        )

        return ilist.IList(positions)

    @classmethod
    def fill(
        cls, grid_obj: grid.Grid[NumX, NumY], filled: Sequence[tuple[int, int]]
    ) -> "FilledGrid[NumX, NumY]":
        num_x, num_y = grid_obj.shape

        if isinstance(grid_obj, FilledGrid):
            vacancies = grid_obj.vacancies
            parent = grid_obj.parent
        else:
            vacancies = frozenset(product(range(num_x), range(num_y)))
            parent = grid_obj

        vacancies = vacancies - frozenset(filled)

        return cls(parent=parent, vacancies=vacancies)

    @classmethod
    def vacate(
        cls, grid_obj: grid.Grid[NumX, NumY], vacancies: Iterable[tuple[int, int]]
    ) -> "FilledGrid[NumX, NumY]":

        if isinstance(grid_obj, FilledGrid):
            input_vacancies = grid_obj.vacancies
            parent = grid_obj.parent
        else:
            input_vacancies = frozenset()
            parent = grid_obj

        input_vacancies = input_vacancies.union(vacancies)

        return cls(parent=parent, vacancies=input_vacancies)

    def get_view(  # type: ignore
        self, x_indices: ilist.IList[int, Any], y_indices: ilist.IList[int, Any]
    ):
        remapping_x = {ix: i for i, ix in enumerate(x_indices)}
        remapping_y = {iy: i for i, iy in enumerate(y_indices)}
        return FilledGrid(
            parent=self.parent.get_view(x_indices, y_indices),
            vacancies=frozenset(
                (remapping_x[x], remapping_y[y])
                for x, y in self.vacancies
                if x in remapping_x and y in remapping_y
            ),
        )

    def shift(self, x_shift: float, y_shift: float):
        return FilledGrid(
            parent=self.parent.shift(x_shift, y_shift),
            vacancies=self.vacancies,
        )

    def scale(self, x_scale: float, y_scale: float):
        return FilledGrid(
            parent=self.parent.scale(x_scale, y_scale),
            vacancies=self.vacancies,
        )

    def repeat(self, x_times: int, y_times: int, x_gap: float, y_gap: float):
        new_parent = self.parent.repeat(x_times, y_times, x_gap, y_gap)
        x_dim, y_dim = self.shape
        vacancies = frozenset(
            (x + x_dim * i, y + y_dim * j)
            for i, j, (x, y) in product(range(x_times), range(y_times), self.vacancies)
        )
        return FilledGrid.vacate(new_parent, vacancies)

    def row_x_pos(self, row_index: int):
        x_vacancies = {x for x, y in self.vacancies if y == row_index}
        return ilist.IList(
            [x for i, x in enumerate(self.parent.x_positions) if i not in x_vacancies]
        )

    def col_y_pos(self, column_index: int):
        y_vacancies = {y for x, y in self.vacancies if x == column_index}
        return ilist.IList(
            [y for i, y in enumerate(self.y_positions) if i not in y_vacancies]
        )

    def serialize(self, serializer: "Serializer") -> "SerializationUnit":
        return SerializationUnit(
            kind="filled_grid",
            module_name=dialect.name,
            class_name=FilledGrid.__name__,
            data={
                "parent": serializer.serialize_attribute(self.parent),
                "vacancies": serializer.serialize(self.vacancies),
            },
        )

    @classmethod
    def deserialize(
        cls, serUnit: "SerializationUnit", deserializer: "Deserializer"
    ) -> "FilledGrid":
        parent = deserializer.deserialize(serUnit.data["parent"])
        vacancies = deserializer.deserialize(serUnit.data["vacancies"])
        return FilledGrid(parent=parent, vacancies=vacancies)


FilledGridType = types.Generic(FilledGrid, types.TypeVar("NumX"), types.TypeVar("NumY"))
