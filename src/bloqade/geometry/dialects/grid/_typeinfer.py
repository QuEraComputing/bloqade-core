from typing import cast

from kirin import types
from kirin.analysis import TypeInference
from kirin.dialects import ilist, py
from kirin.interp import Frame, MethodTable, impl

from ._dialect import dialect
from .stmts import New
from .types import Grid, GridType


@dialect.register(key="typeinfer")
class TypeInferMethods(MethodTable):
    @impl(New)
    def inter_new(self, _: TypeInference, frame: Frame[types.TypeAttribute], node: New):
        def get_len(typ: types.TypeAttribute):
            if (typ := cast(types.Generic, typ)).is_subseteq(
                ilist.IListType
            ) and isinstance(typ.vars[1], types.Literal):
                # assume typ is Generic since it must be if it passes the first check
                # and the second check is to ensure that the length is a literal
                return types.Literal(typ.vars[1].data + 1)

            return types.Any

        x_len = get_len(frame.get(node.x_spacing))
        y_len = get_len(frame.get(node.y_spacing))

        return (GridType[x_len, y_len],)

    @classmethod
    def infer_new_grid_size(cls, index_type: types.TypeAttribute):
        if index_type.is_subseteq(types.Int):
            return types.Literal(1)
        elif index_type.is_subseteq(types.Slice):
            return types.Any
        elif (index_type := cast(types.Generic, index_type)).is_subseteq(
            ilist.IListType
        ):
            return index_type.vars[1]
        else:
            return types.Bottom

    @impl(py.indexing.GetItem, types.PyClass(Grid), types.PyClass(tuple))
    def infer_getitem(
        self,
        interp: TypeInference,
        frame: Frame[types.TypeAttribute],
        stmt: py.indexing.GetItem,
    ):
        index = frame.get(stmt.index)

        if not (index := cast(types.Generic, index)).is_subseteq(
            types.Tuple[types.Any, types.Any]
        ):
            return (types.Any,)
        x_index, y_index = index.vars

        x_len = self.infer_new_grid_size(x_index)
        y_len = self.infer_new_grid_size(y_index)

        if x_len.is_subseteq(types.Bottom) or y_len.is_subseteq(types.Bottom):
            return (types.Any,)

        return (GridType[x_len, y_len],)
