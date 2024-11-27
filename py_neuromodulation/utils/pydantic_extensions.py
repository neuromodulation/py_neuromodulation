import copy
from typing import (
    Any,
    get_origin,
    get_args,
    get_type_hints,
    TypeVar,
    Generic,
    Literal,
    cast,
    Sequence,
)
from typing_extensions import Unpack, TypedDict
from pydantic import BaseModel, model_validator, model_serializer

from pydantic_core import (
    ErrorDetails,
    PydanticUndefined,
    InitErrorDetails,
    ValidationError,
)
from pydantic.fields import FieldInfo, _FieldInfoInputs, _FromFieldInfoInputs
from pprint import pformat


def create_validation_error(
    error_message: str,
    location: list[str | int] = [],
    title: str = "Validation Error",
    error_type="value_error",
) -> ValidationError:
    """
    Factory function to create a Pydantic v2 ValidationError.

    Args:
    error_message (str): The error message for the ValidationError.
    loc (List[str | int], optional): The location of the error. Defaults to None.
    title (str, optional): The title of the error. Defaults to "Validation Error".

    Returns:
    ValidationError: A Pydantic ValidationError instance.
    """

    return ValidationError.from_exception_data(
        title=title,
        line_errors=[
            InitErrorDetails(
                type=error_type,
                loc=tuple(location),
                input=None,
                ctx={"error": error_message},
            )
        ],
        input_type="python",
        hide_input=False,
    )


class NMErrorList:
    """Class to handle data about Pydantic errors.
    Stores data in a list of InitErrorDetails. Errors can be accessed but not modified.

    :return: _description_
    :rtype: _type_
    """

    def __init__(
        self, errors: Sequence[InitErrorDetails | ErrorDetails] | None = None
    ) -> None:
        self.__errors: list[InitErrorDetails | ErrorDetails] = [e for e in errors or []]

    def add_error(
        self,
        error_message: str,
        location: list[str | int] = [],
        error_type="value_error",
    ) -> None:
        self.__errors.append(
            InitErrorDetails(
                type=error_type,
                loc=tuple(location),
                input=None,
                ctx={"error": error_message},
            )
        )

    def create_error(self, title: str = "Validation Error") -> ValidationError:
        return ValidationError.from_exception_data(
            title=title, line_errors=cast(list[InitErrorDetails], self.__errors)
        )

    def extend(self, errors: "NMErrorList"):
        self.__errors.extend(errors.__errors)

    def __iter__(self):
        return iter(self.__errors)

    def __len__(self):
        return len(self.__errors)

    def __getitem__(self, idx):
        # Return a copy of the error to prevent modification
        return copy.deepcopy(self.__errors[idx])

    def __repr__(self):
        return repr(self.__errors)

    def __str__(self):
        return str(self.__errors)


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
        self.custom_metadata: dict[str, Any] = kwargs.pop("custom_metadata", {})
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
    # model_config = ConfigDict(validate_assignment=False, extra="allow")

    def __init__(self, *args, **kwargs) -> None:
        """Pydantic does not support positional arguments by default.
        This is a workaround to support positional arguments for models like FrequencyRange.
        It converts positional arguments to kwargs and then calls the base class __init__.
        """

        if not args:
            # Simple case - just use kwargs
            super().__init__(*args, **kwargs)
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

    __init__.__pydantic_base_init__ = True  # type: ignore

    def __str__(self):
        return pformat(self.model_dump())

    # def __repr__(self):
    #     return pformat(self.model_dump())

    def validate(self, context: Any | None = None) -> Any:  # type: ignore
        return self.model_validate(self.model_dump(), context=context)

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
                # Convert scalar value to dict with metadata
                field_dict = {
                    "value": value,
                    # __field_type__ will be overwritte if set in custom_metadata
                    "__field_type__": type(value).__name__,
                    **{
                        f"__{tag}__": value
                        for tag, value in field_info.custom_metadata.items()
                    },
                }
                # Add possible values for Literal types
                if get_origin(field_info.annotation) is Literal:
                    field_dict["__valid_values__"] = list(
                        get_args(field_info.annotation)
                    )

                result[field_name] = field_dict
        return result

    @classmethod
    def unvalidated(cls, **data: Any) -> Any:
        def process_value(value: Any, field_type: Any) -> Any:
            if isinstance(value, dict) and hasattr(
                field_type, "__pydantic_core_schema__"
            ):
                # Recursively handle nested Pydantic models
                return field_type.unvalidated(**value)
            elif isinstance(value, list):
                # Handle lists of Pydantic models
                if hasattr(field_type, "__args__") and hasattr(
                    field_type.__args__[0], "__pydantic_core_schema__"
                ):
                    return [
                        field_type.__args__[0].unvalidated(**item)
                        if isinstance(item, dict)
                        else item
                        for item in value
                    ]
            return value

        processed_data = {}
        for name, field in cls.model_fields.items():
            try:
                value = data[name]
                processed_data[name] = process_value(value, field.annotation)
            except KeyError:
                if not field.is_required():
                    processed_data[name] = copy.deepcopy(field.default)
                else:
                    raise TypeError(f"Missing required keyword argument {name!r}")

        self = cls.__new__(cls)
        object.__setattr__(self, "__dict__", processed_data)
        object.__setattr__(self, "__pydantic_private__", {"extra": None})
        object.__setattr__(self, "__pydantic_fields_set__", set(processed_data.keys()))
        return self


#################################
#### Generic Pydantic models ####
#################################


T = TypeVar("T")
C = TypeVar("C", list, tuple)


class NMSequenceModel(NMBaseModel, Generic[C]):
    """Base class for sequence models with a single root value"""

    root: C = NMField(default_factory=list)

    # Class variable for aliases - override in subclasses
    __aliases__: dict[int, list[str]] = {}

    def __init__(self, *args, **kwargs) -> None:
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

    @model_serializer
    def ser_model(self):
        return self.root

    # Custom validator to skip the 'root' field in validation errors
    @model_validator(mode="wrap")  # type: ignore[reportIncompatibleMethodOverride]
    def rewrite_error_locations(self, handler):
        try:
            return handler(self)
        except ValidationError as e:
            errors = []
            for err in e.errors():
                loc = list(err["loc"])
                # Find and remove 'root' from the location path
                if "root" in loc:
                    root_idx = loc.index("root")
                    if root_idx < len(loc) - 1:
                        loc = loc[:root_idx] + loc[root_idx + 1 :]
                err["loc"] = tuple(loc)
                errors.append(err)
            raise ValidationError.from_exception_data(
                title="ValidationError", line_errors=errors
            )
