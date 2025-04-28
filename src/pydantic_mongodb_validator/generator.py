import sys
import typing
from datetime import datetime
from enum import Enum
from decimal import Decimal  # Add Decimal import
from uuid import UUID       # Add UUID import
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from bson import ObjectId
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


def _map_pydantic_type_to_bson_type(field_info: FieldInfo) -> Union[str, List[str]]:
    """Maps a Pydantic field's type annotation to MongoDB BSON type(s).

    Handles basic types and determines if a field is optional (Union[T, None]).
    Returns the BSON type string or a list `[bson_type, "null"]` for optional fields.
    More complex types like List, Dict, nested Models, and Unions are primarily
    handled by `_get_schema_for_type`. This function may raise NotImplementedError
    if it encounters such types directly without them being handled as Optional.

    Args:
        field_info (FieldInfo): Pydantic field information object containing
                                annotation and other details.

    Returns:
        Union[str, List[str]]: The corresponding BSON type string (e.g., "string",
                               "int", "objectId") or a list `[bson_type, "null"]`
                               if the field is optional.

    Raises:
        NotImplementedError: If the field type or a complex structure (like
                             non-optional Union, List, Dict, Model) is not
                             supported by this specific mapping function.
    """
    annotation = field_info.annotation
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Optional[T] or Union[T, None]
    is_optional = False
    if origin is Union:
        # Check if NoneType is one of the arguments
        if type(None) in args:
            is_optional = True
            # Filter out NoneType and get the actual type T
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                annotation = non_none_args[0]
                # Re-evaluate origin for the inner type
                origin = get_origin(annotation)
                # Re-evaluate args for the inner type
                args = get_args(annotation)
            else:
                # Handle Union[T1, T2, None] - currently not supported beyond basic optional
                raise NotImplementedError(
                    f"Complex Union types with None are not yet supported: {field_info.annotation}")
        # Non-optional Union handling is now primarily in _get_schema_for_type
        # This function might be simplified or removed later if not needed elsewhere.
        # For now, just pass through if it's a non-optional Union.
        # We still need to handle the case where Optional wraps a complex type like List or Model.
        pass  # Let _get_schema_for_type handle non-optional Unions

    bson_type: Optional[str] = None

    # Basic Types (only if annotation is not a complex type like Union, List, Dict, Model)
    if annotation is str:
        bson_type = "string"
    elif annotation is int:
        # Pydantic v2 doesn't distinguish between int sizes by default
        # We map to "long" if constraints suggest it, otherwise "int" might be okay,
        # but "long" is safer for general MongoDB use unless specific 32-bit is needed.
        # Let's default to "int" for now, constraints can refine later if needed.
        bson_type = "int"  # Could also be "long"
    elif annotation is float:
        bson_type = "double"  # MongoDB uses "double" for floating-point
    elif annotation is bool:
        bson_type = "bool"
    elif annotation is bytes:
        bson_type = "binData"
    elif annotation is datetime:
        bson_type = "date"
    elif annotation is ObjectId:
        bson_type = "objectId"
    # Check for Enums (must come before complex types)
    elif isinstance(annotation, type) and issubclass(annotation, Enum):
        # Determine the type of the enum values (e.g., str, int)
        enum_value_types = {type(item.value) for item in annotation}
        if len(enum_value_types) == 1:
            enum_value_type = list(enum_value_types)[0]
            if enum_value_type is str:
                bson_type = "string"
            elif enum_value_type is int:
                bson_type = "int"  # Or "long"
            # Add other enum value types if needed (float -> double, etc.)
            else:
                raise NotImplementedError(
                    f"Enum with value type {enum_value_type} not supported yet.")
        else:
            # Enum with mixed value types - less common, maybe raise error or handle specifically
            raise NotImplementedError(
                f"Enum with mixed value types not supported: {annotation}")

    # Placeholder for Complex Types (to be implemented later)
    elif origin is list or annotation is list:
        raise NotImplementedError(
            f"List types not yet supported: {field_info.annotation}")
    elif origin is dict or annotation is dict:
        raise NotImplementedError(
            f"Dict types not yet supported: {field_info.annotation}")
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        raise NotImplementedError(
            f"Nested Pydantic models not yet supported: {field_info.annotation}")

    # If no specific type matched
    if bson_type is None:
        raise NotImplementedError(
            f"Type {field_info.annotation} is not yet supported.")

    # Return type or list including "null" for optional fields
    if is_optional:
        return [bson_type, "null"]
    else:
        return bson_type


# --- Helper function to get schema for a type (recursive) ---
# This needs to be defined before generate_validator_schema uses it.

# Forward reference for recursive type hints
_GenerateValidatorSchemaFunc = typing.Callable[[
    Type[BaseModel]], Dict[str, Any]]

# Store the main generator function globally to handle recursion
_generator_func_ref: Optional[_GenerateValidatorSchemaFunc] = None


def _get_schema_for_type(annotation: Type[Any]) -> Dict[str, Any]:
    """Determines the MongoDB JSON schema fragment for a given Python type annotation.

    Handles basic types, Optional[T], Union[T1, T2, ...], List[T], Dict[str, V],
    nested Pydantic BaseModel, and Enum types recursively. Uses the global
    `_generator_func_ref` to enable recursive calls for nested models within
    the main `generate_validator_schema` execution context.

    Args:
        annotation (Type[Any]): The Python type annotation to generate a schema
                                for (e.g., `str`, `Optional[int]`, `List[MyModel]`,
                                `Union[str, int]`).

    Returns:
        Dict[str, Any]: A dictionary representing the JSON schema fragment for the
                        type (e.g., `{"bsonType": "string"}`,
                        `{"bsonType": "array", "items": {...}}`,
                        `{"anyOf": [...]}`). Includes "null" in bsonType list or
                        anyOf if the original type was Optional or included None.

    Raises:
        RuntimeError: If the global `_generator_func_ref` is not set when
                      attempting recursion for a nested model.
        NotImplementedError: If an unsupported type (e.g., Enum with non-str/int
                             values, or other unhandled types) is encountered.
        ValueError: For invalid type definitions like `Union[None]`.
    """
    global _generator_func_ref
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Optional[T] or Union[T1, ..., None] by extracting T or Union[T1, ...]
    is_optional = False
    if origin is Union and type(None) in args:
        is_optional = True
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            # It was Optional[T], extract T
            annotation = non_none_args[0]
            origin = get_origin(annotation)  # Re-evaluate origin/args for T
            args = get_args(annotation)
        elif len(non_none_args) > 1:
            # It was Union[T1, T2, ..., None]. Create a new Union[T1, T2, ...]
            # The is_optional flag is True, and the Union logic below will handle anyOf.
            annotation = Union[tuple(non_none_args)]
            origin = get_origin(annotation)  # Should be Union
            args = get_args(annotation)  # Should be (T1, T2, ...)
        else:  # pragma: no cover
            # This case (Union[None]) should ideally not happen with valid Pydantic models
            raise ValueError("Invalid Union type found: Union[None]")

    field_schema: Dict[str, Any] = {}

    # Handle Union[T1, T2, ...] (Must come before specific types like List, Dict, Model)
    # Note: The Optional[Union[...]] case is handled by is_optional flag + this block
    if origin is Union:
        any_of_schemas = []
        for arg_type in args:
            # Recursively get schema for each type in the Union
            # Important: Pass the arg_type itself, not field_info
            any_of_schemas.append(_get_schema_for_type(arg_type))
        field_schema = {"anyOf": any_of_schemas}

    # 1. Nested Pydantic Model (Check *after* Union)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if not _generator_func_ref:
            raise RuntimeError(
                "Generator function reference not set for recursion.")
        # Recursive call for nested model
        nested_schema = _generator_func_ref(annotation)
        # Return the inner $jsonSchema part
        field_schema = nested_schema.get(
            "$jsonSchema", {"bsonType": "object"})  # Fallback just in case

    # 2. List Type
    elif origin is list or annotation is list:
        field_schema["bsonType"] = "array"
        if args:
            item_type = args[0]
            # Recursively get schema for the item type
            field_schema["items"] = _get_schema_for_type(item_type)
        else:
            # List without type annotation (e.g., list), treat as any type
            # MongoDB doesn't have a direct "any" type, allowing anything is default
            pass  # No specific "items" schema means any type is allowed

    # 3. Dict Type
    elif origin is dict or annotation is dict:
        field_schema["bsonType"] = "object"
        if args and len(args) == 2:
            key_type, value_type = args
            if key_type is not str:
                print(
                    f"Warning: Dictionary keys are assumed to be strings in MongoDB schemas. Found key type: {key_type}", file=sys.stderr)
            # Recursively get schema for the value type
            field_schema["additionalProperties"] = _get_schema_for_type(
                value_type)
        else:
            # Dict without type annotation (e.g., dict), allow any properties
            # Allow any extra fields/values
            field_schema["additionalProperties"] = True

    # 4. Basic Types (Simplified mapping, no constraints here)
    elif annotation is str:
        field_schema["bsonType"] = "string"
    elif annotation is int:
        field_schema["bsonType"] = "int"  # Or "long"
    elif annotation is float:
        field_schema["bsonType"] = "double"
    elif annotation is bool:
        field_schema["bsonType"] = "bool"
    elif annotation is bytes:
        field_schema["bsonType"] = "binData"
    elif annotation is datetime:
        field_schema["bsonType"] = "date"
    elif annotation is ObjectId:
        field_schema["bsonType"] = "objectId"
    elif annotation is Decimal:  # Handle Decimal
        field_schema["bsonType"] = "decimal"
    elif annotation is UUID:    # Handle UUID
        field_schema["bsonType"] = "binData"
    elif isinstance(annotation, type) and issubclass(annotation, Enum):
        # Determine enum value type for bsonType
        enum_value_types = {type(item.value) for item in annotation}
        if len(enum_value_types) == 1:
            enum_value_type = list(enum_value_types)[0]
            if enum_value_type is str:
                field_schema["bsonType"] = "string"
            elif enum_value_type is int:
                field_schema["bsonType"] = "int"  # Or "long"
            else:
                raise NotImplementedError(
                    f"Enum with value type {enum_value_type} not supported.")
        else:
            raise NotImplementedError(
                f"Enum with mixed value types not supported: {annotation}")
        # Add enum constraint
        field_schema["enum"] = [item.value for item in annotation]

    # 5. Fallback for unsupported types
    else:
        # Only raise error if the type wasn't handled by the Union logic above
        if not field_schema:
            raise NotImplementedError(
                f"Type {annotation} is not supported for schema generation.")

    # Handle optionality AFTER the main schema (including anyOf) is generated
    if is_optional:
        if "anyOf" in field_schema:
            # If schema is anyOf, add null as one of the allowed types
            # Check if null is already somehow present (e.g., Union[str, None, int])
            has_null = any({"bsonType": "null"} ==
                           s for s in field_schema["anyOf"])
            if not has_null:
                field_schema["anyOf"].append({"bsonType": "null"})
        elif "bsonType" in field_schema:
            # Standard handling for basic types, lists, dicts
            current_bson_type = field_schema["bsonType"]
            if isinstance(current_bson_type, list):
                if "null" not in current_bson_type:
                    field_schema["bsonType"].append("null")
            else:
                field_schema["bsonType"] = [current_bson_type, "null"]
        # If bsonType wasn't set (e.g., for a nested model which uses bsonType: object implicitly),
        # optionality is handled by the 'required' array in the parent schema,
        # so no modification is needed here in the child schema fragment.

    return field_schema


# --- Constraint Mapping (Remains mostly the same) ---
def _map_pydantic_constraints(field_info: FieldInfo, field_schema: Dict[str, Any]) -> None:
    """Adds Pydantic constraints to the MongoDB field schema dictionary in-place.

    Modifies the provided `field_schema` dictionary to include validation rules
    derived from the `field_info` metadata. This includes constraints like
    string length (`minLength`, `maxLength`), regex patterns (`pattern`), and
    numeric bounds (`minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`).

    Enum constraints (`enum`) are handled within `_get_schema_for_type` as they
    are directly tied to the type definition.

    Args:
        field_info (FieldInfo): The Pydantic field information object containing
                                constraints in its attributes or metadata.
        field_schema (Dict[str, Any]): The dictionary representing the field\'s
                                       schema, which will be modified in-place.

    Returns:
        None: The `field_schema` dictionary is modified directly.
    """
    # This function primarily adds constraints like minLength, maximum, etc.
    # It should NOT overwrite bsonType or structural properties like 'items'.

    # Iterate through metadata to find constraints (Pydantic v2 style)
    if field_info.metadata:
        for constraint in field_info.metadata:
            # String Constraints
            if hasattr(constraint, 'min_length') and constraint.min_length is not None:
                field_schema["minLength"] = constraint.min_length
            if hasattr(constraint, 'max_length') and constraint.max_length is not None:
                field_schema["maxLength"] = constraint.max_length
            if hasattr(constraint, 'pattern') and isinstance(constraint.pattern, str):
                field_schema["pattern"] = constraint.pattern

            # Numeric Constraints
            if hasattr(constraint, 'ge') and constraint.ge is not None:
                field_schema["minimum"] = constraint.ge
            if hasattr(constraint, 'gt') and constraint.gt is not None:
                field_schema["minimum"] = constraint.gt
                # MongoDB uses boolean flag
                field_schema["exclusiveMinimum"] = True
            if hasattr(constraint, 'le') and constraint.le is not None:
                field_schema["maximum"] = constraint.le
            if hasattr(constraint, 'lt') and constraint.lt is not None:
                field_schema["maximum"] = constraint.lt
                # MongoDB uses boolean flag
                field_schema["exclusiveMaximum"] = True

    # Add field description if present and not already added by _get_schema_for_type
    # (e.g., for basic types where description wasn't part of the type schema itself)
    if field_info.description and "description" not in field_schema:
        field_schema["description"] = field_info.description

    # Enum constraints are handled entirely within _get_schema_for_type


# --- Main Generator Function ---


def generate_validator_schema(
    model: Type[BaseModel],
    *,  # Make options keyword-only
    include_title: bool = True,
    include_description: bool = True
) -> Dict[str, Any]:
    """Generates the complete MongoDB $jsonSchema validator for a Pydantic model.

    Iterates through the fields of the given Pydantic `BaseModel`, generating
    the corresponding MongoDB JSON schema for each field. It utilizes helper
    functions (`_get_schema_for_type`, `_map_pydantic_constraints`) to handle
    type mapping, nested structures (models, lists, dicts), optionality, unions,
    enums, and constraints.

    The final schema includes the `bsonType: object`, `properties`, `required`
    fields, and `additionalProperties` settings based on the model's configuration.
    It can optionally include the model's title and description at the root level.

    Handles recursion for nested models by temporarily setting a global reference
    to itself (`_generator_func_ref`).

    Args:
        model (Type[BaseModel]): The Pydantic model class to generate the schema from.
        include_title (bool, optional): Keyword-only argument. If True, includes
            the model's title (from `model_config['title']`) in the schema root.
            Defaults to True.
        include_description (bool, optional): Keyword-only argument. If True,
            includes the model's description (from the model's docstring) in the
            schema root. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary representing the complete MongoDB collection
                        validator, structured as `{"$jsonSchema": {...}}`.
    """
    global _generator_func_ref
    # Set the reference for recursive calls, passing along the config options
    original_ref_is_none = _generator_func_ref is None
    if original_ref_is_none:
        def _generator_func_ref(m): return generate_validator_schema(
            m, include_title=include_title, include_description=include_description
        )

    properties: Dict[str, Any] = {}
    required_fields: List[str] = []

    for field_name, field_info in model.model_fields.items():
        # Determine the MongoDB field name, prioritizing serialization_alias > alias > field_name
        if field_info.serialization_alias is not None:
            mongo_field_name = field_info.serialization_alias
        elif field_info.alias is not None:
            mongo_field_name = field_info.alias
        else:
            mongo_field_name = field_name

        # 1. Get Schema for Field Type (Handles Recursion and Structure)
        annotation = field_info.annotation  # Get the original annotation
        try:
            # Use the new helper to get the base schema structure
            # _get_schema_for_type uses the global _generator_func_ref for recursion
            field_schema = _get_schema_for_type(annotation)

            # If the type was Optional[T], _get_schema_for_type handles the ["type", "null"]
            # but we still need the underlying type T for constraint mapping etc.
            origin = get_origin(annotation)
            args = get_args(annotation)
            is_optional = False
            actual_annotation = annotation
            if origin is Union and type(None) in args:
                is_optional = True
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    actual_annotation = non_none_args[0]

            # Special handling for top-level nested models: merge properties etc.
            # _get_schema_for_type already called generate_validator_schema if needed,
            # and returned the inner $jsonSchema part. We just use it.
            # Note: Constraints like description might need to be applied to the outer object schema.

            # Add field description if present (this is field-level, not model-level)
            if field_info.description:
                field_schema["description"] = field_info.description

            # 2. Map Constraints (apply to the generated schema)
            # Pass the potentially modified actual_annotation if it was Optional
            # Create a temporary FieldInfo-like object if needed, or adjust _map_pydantic_constraints
            # For now, assume _map_pydantic_constraints works with the original field_info
            _map_pydantic_constraints(field_info, field_schema)

            # 3. Apply json_schema_extra
            json_schema_extra = field_info.json_schema_extra
            if json_schema_extra:
                extra_schema = json_schema_extra() if callable(
                    json_schema_extra) else json_schema_extra
                if isinstance(extra_schema, dict):
                    field_schema.update(extra_schema)

            # Ensure Enum constraint is handled correctly (might be duplicated from _get_schema_for_type)
            # Let's refine this - _get_schema_for_type should handle enum fully.
            # We remove the enum part from _map_pydantic_constraints later if needed.

        except NotImplementedError as e:
            print(
                f"Warning: Skipping field '{mongo_field_name}' due to unsupported type logic: {e}", file=sys.stderr)
            continue  # Skip field if type logic fails

        # 3. Add to properties dictionary
        properties[mongo_field_name] = field_schema

        # 4. Check if required (Handles Optional/Union[T, None] correctly)
        # Pydantic v2: field_info.is_required() is the most reliable way
        # It correctly handles defaults, Optional, Union[T, None] etc.
        if field_info.is_required():
            # Check if default is PydanticUndefined (meaning no default was set)
            # This check might be redundant if is_required() is fully reliable,
            # but adds an extra layer of clarity.
            # if field_info.default is PydanticUndefined:
            required_fields.append(mongo_field_name)
            # Handle cases where default is None but field is not Optional explicitly
            # elif field_info.default is None:
            #    # Need to check if None is part of the allowed types (Optional/Union)
            #    origin = get_origin(field_info.annotation)
            #    args = get_args(field_info.annotation)
            #    is_explicitly_optional = origin is Union and type(None) in args
            #    if not is_explicitly_optional:
            #        required_fields.append(mongo_field_name)

    # --- Construct Final Schema ---
    json_schema_props: Dict[str, Any] = {
        "bsonType": "object",
        "properties": properties,
        # required and additionalProperties added conditionally below
    }

    # Add required fields if any
    if required_fields:
        json_schema_props["required"] = sorted(
            required_fields)  # Sort for consistent output

    # Handle additionalProperties based on model_config
    # Use model_extra in Pydantic v2
    extra_config = model.model_config.get(
        'extra', 'ignore')  # Default to 'ignore'

    if extra_config == 'forbid':
        json_schema_props["additionalProperties"] = False
    elif extra_config == 'allow':
        # True allows any additional fields, {} might imply validation against a schema (not applicable here)
        json_schema_props["additionalProperties"] = True
    # else: 'ignore' is the default MongoDB behavior (allows extra fields without validation)

    # Add model title if requested and available
    if include_title:
        # Use model_config.get('title') for Pydantic v2 compatibility
        model_title = model.model_config.get('title')
        if model_title:
            json_schema_props["title"] = model_title

    # Add model description if requested and available
    if include_description:
        model_description = model.__doc__
        if model_description:
            # Clean up potential indentation from docstring
            cleaned_description = "\n".join(
                line.strip() for line in model_description.strip().splitlines())
            if cleaned_description:  # Ensure it's not empty after stripping
                json_schema_props["description"] = cleaned_description

    schema: Dict[str, Any] = {"$jsonSchema": json_schema_props}

    # Reset the global reference only if it was set by this top-level call
    if original_ref_is_none:
        _generator_func_ref = None

    return schema
