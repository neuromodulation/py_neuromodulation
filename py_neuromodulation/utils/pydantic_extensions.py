from typing import Any, get_type_hints, TypeVar, Generic, Literal, overload
from typing_extensions import Unpack, TypedDict
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_core import PydanticUndefined, ValidationError, InitErrorDetails
from pydantic.fields import FieldInfo, _FieldInfoInputs, _FromFieldInfoInputs
from pprint import pformat


def create_validation_error(
    error_message: str,
    loc: list[str | int] | None = None,
    title: str = "Validation Error",
    input_type: Literal["python", "json"] = "python",
    hide_input: bool = False,
) -> ValidationError:
    """
    Factory function to create a Pydantic v2 ValidationError instance from a single error message.

    Args:
    error_message (str): The error message for the ValidationError.
    loc (List[str | int], optional): The location of the error. Defaults to None.
    title (str, optional): The title of the error. Defaults to "Validation Error".
    input_type (Literal["python", "json"], optional): Whether the error is for a Python object or JSON. Defaults to "python".
    hide_input (bool, optional): Whether to hide the input value in the error message. Defaults to False.

    Returns:
    ValidationError: A Pydantic ValidationError instance.
    """
    if loc is None:
        loc = []

    line_errors = [
        InitErrorDetails(
            type="value_error", loc=tuple(loc), input=None, ctx={"error": error_message}
        )
    ]

    return ValidationError.from_exception_data(
        title=title,
        line_errors=line_errors,
        input_type=input_type,
        hide_input=hide_input,
    )


class _NMExtraFieldInputs(TypedDict, total=False):
    """Additional fields to add on top of the pydantic FieldInfo"""

    custom_metadata: dict[str, Any]


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
        self.custom_metadata = kwargs.pop("custom_metadata", {})
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
        """Pydantic does not support positional arguments by default.
        This is a workaround to support positional arguments for models like FrequencyRange.
        It converts positional arguments to kwargs and then calls the base class __init__.
        """
        if not args:
            # Simple case - just use kwargs
            super().__init__(**kwargs)
            return

        field_names = list(self.model_fields.keys())
        # If we have more positional args than fields, that's an error
        if len(args) > len(field_names):
            raise ValueError(
                f"Too many positional arguments. Expected at most {len(field_names)}, got {len(args)}"
            )

        # Convert positional args to kwargs, checking for duplicates        if args:
        complete_kwargs = {}
        for i, arg in enumerate(args):
            field_name = field_names[i]
            if field_name in kwargs:
                raise ValueError(
                    f"Got multiple values for field '{field_name}': "
                    f"positional argument and keyword argument"
                )
            complete_kwargs[field_name] = arg

        # Add remaining kwargs
        complete_kwargs.update(kwargs)
        super().__init__(**complete_kwargs)

    def __str__(self):
        return pformat(self.model_dump())

    def __repr__(self):
        return pformat(self.model_dump())

    def validate(self) -> Any:  # type: ignore
        return self.model_validate_strings(self.model_dump())

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        setattr(self, key, value)

    @property
    def fields(self) -> dict[str, FieldInfo | NMFieldInfo]:
        return self.model_fields  # type: ignore

    def serialize_with_metadata(self):
        result: dict[str, Any] = {"__field_type__": self.__class__.__name__}

        for field_name, field_info in self.fields.items():
            value = getattr(self, field_name)
            if isinstance(value, NMBaseModel):
                result[field_name] = value.serialize_with_metadata()
            elif isinstance(value, list):
                result[field_name] = [
                    item.serialize_with_metadata()
                    if isinstance(item, NMBaseModel)
                    else item
                    for item in value
                ]
            elif isinstance(value, dict):
                result[field_name] = {
                    k: v.serialize_with_metadata() if isinstance(v, NMBaseModel) else v
                    for k, v in value.items()
                }
            else:
                result[field_name] = value

            # Extract unit information from Annotated type
            if isinstance(field_info, NMFieldInfo):
                for tag, value in field_info.custom_metadata.items():
                    result[f"__{tag}__"] = value
        return result


#################################
#### Generic Pydantic models ####
#################################


def create_alias_property(index: int, alias: str, classname: str):
    """Creates a property that accesses the root sequence at the given index"""

    def getter(self):
        return self.root[index]

    def setter(self, value):
        if isinstance(self.root, tuple):
            new_values = list(self.root)
            new_values[index] = value
            self.root = tuple(new_values)
        else:
            self.root[index] = value

    return property(
        fget=getter,
        fset=setter,
        doc=f"Alias '{alias}' for position [{index}] of class '{classname}'.",
    )


T = TypeVar("T")
C = TypeVar("C", list, tuple)


class NMSequenceModel(NMBaseModel, Generic[C]):
    """Base class for sequence models with a single root value"""

    root: C = NMField(default_factory=list)

    # Class variable for aliases - override in subclasses
    __aliases__: dict[int, list[str]] = {}

    def __init__(self, *args, **kwargs) -> None:
        # Generate properties programatically (not used currently)
        # for index, aliases in self.__aliases__.items():
        #     for alias in aliases:
        #         if not hasattr(self.__class__, alias):
        #             setattr(
        #                 self.__class__,
        #                 alias,
        #                 create_alias_property(index, alias, self.__class__.__name__),
        #             )

        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            kwargs["root"] = args[0]
        elif len(args) == 1:
            kwargs["root"] = [args[0]]
        elif len(args) > 1:  # Add this case
            kwargs["root"] = tuple(args)
        super().__init__(**kwargs)

    def __iter__(self):  # type: ignore[reportIncompatibleMethodOverride]
        return iter(self.root)

    def __getitem__(self, idx):
        return self.root[idx]

    def __len__(self):
        return len(self.root)

    def model_dump(self):  # type: ignore[reportIncompatibleMethodOverride]
        return self.root

    def model_dump_json(self, **kwargs):
        import json

        return json.dumps(self.root, **kwargs)

    def serialize_with_metadata(self) -> dict[str, Any]:
        result = {"__field_type__": self.__class__.__name__, "value": self.root}

        # Add any field metadata from the root field
        field_info = self.model_fields.get("root")
        if isinstance(field_info, NMFieldInfo):
            for tag, value in field_info.custom_metadata.items():
                result[f"__{tag}__"] = value

        return result

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, value: Any) -> dict[str, Any]:
        # If it's a dict, just return it
        if isinstance(value, dict):
            if "root" in value:
                return value

            # Check for aliased fields if class has aliases defined
            if hasattr(cls, "__aliases__"):
                # Collect all possible alias names for each position
                alias_values = []
                max_index = max(cls.__aliases__.keys()) if cls.__aliases__ else -1

                # Try to find a value for each position using its aliases
                for i in range(max_index + 1):
                    aliases = cls.__aliases__.get(i, [])
                    value_found = None

                    # Try each alias for this position
                    for alias in aliases:
                        if alias in value:
                            value_found = value[alias]
                            break

                    if value_found is not None:
                        alias_values.append(value_found)
                    else:
                        # If we're missing any position's value, don't use aliases
                        break

                # If we found all values through aliases, use them
                if len(alias_values) == max_index + 1:
                    return {"root": alias_values}

        # if it's a sequence, return the value as the root
        if isinstance(value, (list, tuple)):
            return {"root": value}

        # Else, make it a list
        return {"root": [value]}


class NMValueModel(NMBaseModel, Generic[T]):
    """Base class for single-value models that behave like their contained type"""

    root: T

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            if "root" in value:
                return value
            # If it's a dict without root, assume the first value is our value
            if len(value) > 0:
                return {"root": next(iter(value.values()))}
            return {"root": None}
        return {"root": value}

    def __str__(self) -> str:
        return str(self.root)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.root)})"

    def model_dump(self):  # type: ignore[reportIncompatibleMethodOverride]
        return self.root

    def model_dump_json(self, **kwargs):
        import json

        return json.dumps(self.root, **kwargs)

    def serialize_with_metadata(self) -> dict[str, Any]:
        result = {"__field_type__": self.__class__.__name__, "value": self.root}

        # Add any field metadata from the root field
        field_info = self.model_fields.get("root")
        if isinstance(field_info, NMFieldInfo):
            for tag, value in field_info.custom_metadata.items():
                result[f"__{tag}__"] = value

        return result
