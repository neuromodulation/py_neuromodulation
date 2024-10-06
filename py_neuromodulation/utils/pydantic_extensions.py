from typing import Any, get_type_hints
from pydantic.fields import FieldInfo, _FieldInfoInputs, _FromFieldInfoInputs
from pydantic import BaseModel, ConfigDict, model_serializer
from pydantic_core import PydanticUndefined
from typing_extensions import Unpack, TypedDict
from pprint import pformat


class _NMExtraFieldInputs(TypedDict, total=False):
    """Additional fields to add on top of the pydantic FieldInfo"""

    meta: dict[str, Any]


class _NMFieldInfoInputs(_FieldInfoInputs, _NMExtraFieldInputs, total=False):
    """Combine pydantic FieldInfo inputs with PyNM additional inputs"""

    pass


class _NMFromFieldInfoInputs(_FromFieldInfoInputs, _NMExtraFieldInputs, total=False):
    """Combine pydantic FieldInfo.from_field inputs with PyNM additional inputs"""

    pass


class NMFieldInfo(FieldInfo):
    # Add default values for any other custom fields here
    _default_values = {}

    def __init__(self, **kwargs: Unpack[_NMFieldInfoInputs]) -> None:
        extra_fields = get_type_hints(_NMExtraFieldInputs)
        for field, field_type in extra_fields.items():
            # If no default value, try to instantiate the field type with no arguments, if it fails, set to None
            try:
                value = (
                    kwargs.pop(field, self._default_values.get(field, field_type())),  # type: ignore
                )
            except Exception:
                value = None

            setattr(self, field, value)
        super().__init__(**kwargs)

    @staticmethod
    def from_field(
        default: Any = PydanticUndefined,
        **kwargs: Unpack[_NMFromFieldInfoInputs],
    ) -> "NMFieldInfo":
        if "annotation" in kwargs:
            raise TypeError('"annotation" is not permitted as a Field keyword argument')
        return NMFieldInfo(default=default, **kwargs)

    def __repr_args__(self):
        yield from super().__repr_args__()
        extra_fields = get_type_hints(_NMExtraFieldInputs)
        for field in extra_fields:
            value = getattr(self, field)
            yield field, value


def NMField(
    default: Any = PydanticUndefined,
    **kwargs: Unpack[_NMFromFieldInfoInputs],
) -> Any:
    return NMFieldInfo.from_field(default=default, **kwargs)


class NMBaseModel(BaseModel):
    model_config = ConfigDict(validate_assignment=False, extra="allow")

    def __init__(self, *args, **kwargs) -> None:
        if kwargs:
            super().__init__(**kwargs)
        else:
            field_names = list(self.model_fields.keys())
            kwargs = {}
            for i in range(len(args)):
                kwargs[field_names[i]] = args[i]
            super().__init__(**kwargs)

    def __str__(self):
        return pformat(self.model_dump())

    def __repr__(self):
        return pformat(self.model_dump())

    def validate(self) -> Any:  # type: ignore
        return self.model_validate(self.model_dump())

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        setattr(self, key, value)

    @classmethod
    def get_fields(cls) -> dict[str, NMFieldInfo]:
        return cls.model_fields  # type: ignore

    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        result = {"__field_type__": self.__class__.__name__}
        for field_name, field_info in self.model_fields.items():
            value = getattr(self, field_name)
            if isinstance(value, NMBaseModel):
                result[field_name] = value.serialize_model()
            elif isinstance(value, list):
                result[field_name] = [
                    item.serialize_model() if isinstance(item, NMBaseModel) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                result[field_name] = {
                    k: v.serialize_model() if isinstance(v, NMBaseModel) else v
                    for k, v in value.items()
                }
            else:
                result[field_name] = value

            # Extract unit information from Annotated type
            if get_origin(field_info.annotation) is Annotated:
                metadata = get_args(field_info.annotation)[1:]
                for item in metadata:
                    if isinstance(item, dict) and "unit" in item:
                        result[f"{field_name}_metadata"] = {"unit": item["unit"]}
                        break

        return result
