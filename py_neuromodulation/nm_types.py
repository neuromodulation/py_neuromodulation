from os import PathLike
from typing import NamedTuple, Type, Any
from importlib import import_module

_PathLike = str | PathLike


class ImportDetails(NamedTuple):
    module_name: str
    class_name: str


def get_class(module_details: ImportDetails) -> Type[Any]:
    return getattr(import_module(module_details.module_name), module_details.class_name)
