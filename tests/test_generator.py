# tests/test_generator.py
import pytest
from pydantic import BaseModel, Field, model_validator, ValidationError, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
from enum import Enum
from decimal import Decimal
from uuid import UUID

# Assuming the function is in src/pydantic_mongodb_validator/generator.py
from src.pydantic_mongodb_validator.generator import generate_validator_schema

# --- Test Models ---


class BasicTypesModel(BaseModel):
    name: str
    age: int
    price: float
    is_active: bool
    data: bytes
    created_at: datetime
    mongo_id: ObjectId


class RequiredOptionalModel(BaseModel):
    required_field: str
    optional_field: Optional[int] = None
    default_field: bool = True


class AliasModel(BaseModel):
    field_with_alias: str = Field(alias="aliasedName")


class StringConstraintsModel(BaseModel):
    short_str: str = Field(min_length=3)
    long_str: str = Field(max_length=10)
    pattern_str: str = Field(pattern=r"^\d{3}-\d{2}-\d{4}$")  # SSN example


class NumericConstraintsModel(BaseModel):
    min_num: int = Field(ge=0)  # Greater than or equal to 0
    max_num: float = Field(le=100.5)  # Less than or equal to 100.5
    exclusive_min: int = Field(gt=0)  # Greater than 0
    exclusive_max: float = Field(lt=100.5)  # Less than 100.5
    multiple_of_num: int = Field(multiple_of=5)


class StatusEnum(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"


class EnumModel(BaseModel):
    status: StatusEnum


class ExtraAllowModel(BaseModel):
    known_field: str
    model_config = ConfigDict(extra="allow")


class ExtraForbidModel(BaseModel):
    known_field: str
    model_config = ConfigDict(extra="forbid")


class NestedModel(BaseModel):
    nested_id: int
    nested_name: str


class SimpleNestedOuterModel(BaseModel):
    outer_id: str
    nested_data: NestedModel


class ListBasicModel(BaseModel):
    tags: List[str]


class ListNestedModel(BaseModel):
    items: List[NestedModel]


class SimpleDictModel(BaseModel):
    scores: Dict[str, int]

# --- Test Functions ---


def test_basic_types():
    generated_schema = generate_validator_schema(BasicTypesModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["name", "age", "price", "is_active", "data", "created_at", "mongo_id"],
            "properties": {
                "name": {"bsonType": "string", "description": "name"},
                "age": {"bsonType": "int", "description": "age"},
                "price": {"bsonType": "double", "description": "price"},
                "is_active": {"bsonType": "bool", "description": "is_active"},
                "data": {"bsonType": "binData", "description": "data"},
                "created_at": {"bsonType": "date", "description": "created_at"},
                "mongo_id": {"bsonType": "objectId", "description": "mongo_id"},
            },
            "additionalProperties": False,
        }
    }
    assert generated_schema == expected_schema


def test_required_optional():
    generated_schema = generate_validator_schema(RequiredOptionalModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            # optional_field is not required, default_field has a default
            "required": ["required_field"],
            "properties": {
                "required_field": {"bsonType": "string", "description": "required_field"},
                "optional_field": {"bsonType": ["int", "null"], "description": "optional_field"},
                "default_field": {"bsonType": "bool", "description": "default_field"},
            },
            "additionalProperties": False,
        }
    }
    assert generated_schema == expected_schema
    # Check required fields explicitly
    assert "required_field" in expected_schema["$jsonSchema"]["required"]
    assert "optional_field" not in expected_schema["$jsonSchema"]["required"]
    assert "default_field" not in expected_schema["$jsonSchema"]["required"]


def test_alias():
    generated_schema = generate_validator_schema(AliasModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [],  # Field has default value ""
            "properties": {
                "aliasedName": {"bsonType": "string", "description": "field_with_alias"},
            },
            "additionalProperties": False,
        }
    }
    # Field has a default value, so it's not required
    assert generated_schema == expected_schema

# --- Model for Alias Prioritization Test ---


class AliasPriorityModel(BaseModel):
    field_no_alias: Optional[str] = None
    field_only_alias: Optional[str] = Field(None, alias="alias_name")
    field_only_serialization: Optional[str] = Field(
        None, serialization_alias="serialization_name")
    field_both_aliases: Optional[str] = Field(
        None, alias="alias_should_be_ignored", serialization_alias="serialization_should_be_used")
    # Validation alias should not affect serialization name unless serialization_alias is also set
    field_validation_alias: Optional[str] = Field(
        None, validation_alias="validation_name")
    field_validation_and_serialization: Optional[str] = Field(
        None, validation_alias="validation_should_be_ignored", serialization_alias="serialization_name_2")

# --- Test Function for Alias Prioritization ---


def test_alias_prioritization():
    """
    Tests that serialization_alias is prioritized over alias, which is prioritized
    over the field name when generating the MongoDB schema property name.
    Validation alias should not be used for the property name unless it's the only one
    or serialization_alias is also present (in which case serialization_alias wins).
    """
    generated_schema = generate_validator_schema(AliasPriorityModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [],  # All fields are optional
            "properties": {
                "field_no_alias": {"bsonType": ["string", "null"], "description": "field_no_alias"},
                "alias_name": {"bsonType": ["string", "null"], "description": "field_only_alias"},
                "serialization_name": {"bsonType": ["string", "null"], "description": "field_only_serialization"},
                "serialization_should_be_used": {"bsonType": ["string", "null"], "description": "field_both_aliases"},
                # Uses field name
                "field_validation_alias": {"bsonType": ["string", "null"], "description": "field_validation_alias"},
                # Uses serialization_alias
                "serialization_name_2": {"bsonType": ["string", "null"], "description": "field_validation_and_serialization"},
            },
            "additionalProperties": False,
        }
    }
    assert generated_schema == expected_schema


def test_string_constraints():
    generated_schema = generate_validator_schema(StringConstraintsModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [],  # Fields have default values
            "properties": {
                "short_str": {"bsonType": "string", "minLength": 3, "description": "short_str"},
                "long_str": {"bsonType": "string", "maxLength": 10, "description": "long_str"},
                "pattern_str": {"bsonType": "string", "pattern": r"^\d{3}-\d{2}-\d{4}$", "description": "pattern_str"},
            },
            "additionalProperties": False,
        }
    }
    assert generated_schema == expected_schema


def test_numeric_constraints():
    generated_schema = generate_validator_schema(NumericConstraintsModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [],  # Fields have default values
            "properties": {
                "min_num": {"bsonType": "int", "minimum": 0, "description": "min_num"},
                "max_num": {"bsonType": "double", "maximum": 100.5, "description": "max_num"},
                "exclusive_min": {"bsonType": "int", "exclusiveMinimum": 0, "description": "exclusive_min"},
                "exclusive_max": {"bsonType": "double", "exclusiveMaximum": 100.5, "description": "exclusive_max"},
                "multiple_of_num": {"bsonType": "int", "multipleOf": 5, "description": "multiple_of_num"},
            },
            "additionalProperties": False,
        }
    }
    assert generated_schema == expected_schema


def test_enum():
    generated_schema = generate_validator_schema(EnumModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["status"],
            "properties": {
                "status": {"enum": ["pending", "processing", "completed"], "description": "status"},
            },
            "additionalProperties": False,
        }
    }
    assert generated_schema == expected_schema


def test_extra_allow():
    generated_schema = generate_validator_schema(ExtraAllowModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["known_field"],
            "properties": {
                "known_field": {"bsonType": "string", "description": "known_field"},
            },
            "additionalProperties": True,  # Key difference
        }
    }
    assert generated_schema == expected_schema


def test_extra_forbid():
    generated_schema = generate_validator_schema(ExtraForbidModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["known_field"],
            "properties": {
                "known_field": {"bsonType": "string", "description": "known_field"},
            },
            "additionalProperties": False,  # Key difference
        }
    }
    assert generated_schema == expected_schema


def test_simple_nested_model():
    generated_schema = generate_validator_schema(SimpleNestedOuterModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["outer_id", "nested_data"],
            "properties": {
                "outer_id": {"bsonType": "string", "description": "outer_id"},
                "nested_data": {
                    "bsonType": "object",
                    "required": ["nested_id", "nested_name"],
                    "properties": {
                        "nested_id": {"bsonType": "int", "description": "nested_id"},
                        "nested_name": {"bsonType": "string", "description": "nested_name"},
                    },
                    "additionalProperties": False,
                    "description": "nested_data"
                },
            },
            "additionalProperties": False,
        }
    }
    assert generated_schema == expected_schema


def test_list_basic_type():
    generated_schema = generate_validator_schema(ListBasicModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["tags"],
            "properties": {
                "tags": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"},
                    "description": "tags"
                },
            },
            "additionalProperties": False,
        }
    }
    assert generated_schema == expected_schema


def test_list_nested_model():
    generated_schema = generate_validator_schema(ListNestedModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["items"],
            "properties": {
                "items": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "object",
                        "required": ["nested_id", "nested_name"],
                        "properties": {
                            "nested_id": {"bsonType": "int", "description": "nested_id"},
                            "nested_name": {"bsonType": "string", "description": "nested_name"},
                        },
                        "additionalProperties": False,
                    },
                    "description": "items"
                },
            },
            "additionalProperties": False,
        }
    }
    assert generated_schema == expected_schema


def test_simple_dictionary():
    generated_schema = generate_validator_schema(SimpleDictModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["scores"],
            "properties": {
                "scores": {
                    "bsonType": "object",
                    "additionalProperties": {"bsonType": "int"},
                    "description": "scores"
                    # MongoDB $jsonSchema doesn't have a direct equivalent for Dict keys being strings,
                    # but `bsonType: "object"` with `additionalProperties` implies string keys.
                },
            },
            "additionalProperties": False,
        }
    }
    assert generated_schema == expected_schema


def test_union_types():
    generated_schema = generate_validator_schema(UnionTestModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            # optional_union and required_optional_union are not required
            "required": ["simple_union", "complex_union"],
            "properties": {
                "simple_union": {
                    "anyOf": [
                        {"bsonType": "string"},
                        {"bsonType": "int"}
                    ],
                    "description": "simple_union"
                },
                "complex_union": {
                    "anyOf": [
                        {"bsonType": "double"},
                        {  # NestedModel schema
                            "bsonType": "object",
                            "required": ["nested_id", "nested_name"],
                            "properties": {
                                "nested_id": {"bsonType": "int", "description": "nested_id"},
                                "nested_name": {"bsonType": "string", "description": "nested_name"},
                            },
                            "additionalProperties": False,
                            # Description for nested model itself is not typically added here by generator
                        }
                    ],
                    "description": "complex_union"
                },
                "optional_union": {
                    "anyOf": [
                        {"bsonType": "bool"},
                        {"bsonType": "string"},
                        {"bsonType": "null"}  # Added null because it's Optional
                    ],
                    "description": "optional_union"
                },
                "required_optional_union": {  # Handled like Optional[datetime]
                    "bsonType": ["date", "null"],
                    "description": "required_optional_union"
                }
            },
            "additionalProperties": False,
        }
    }
    assert generated_schema == expected_schema

# --- Model for Decimal/UUID ---


class DecimalUUIDModel(BaseModel):
    transaction_id: UUID
    amount: Decimal
    optional_uuid: Optional[UUID] = None
    optional_decimal: Optional[Decimal] = None


# --- Test for Decimal/UUID ---

def test_decimal_uuid_types():
    generated_schema = generate_validator_schema(DecimalUUIDModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["transaction_id", "amount"],
            "properties": {
                "transaction_id": {"bsonType": "binData", "description": "transaction_id"},
                "amount": {"bsonType": "decimal", "description": "amount"},
                "optional_uuid": {"bsonType": ["binData", "null"], "description": "optional_uuid"},
                "optional_decimal": {"bsonType": ["decimal", "null"], "description": "optional_decimal"},
            },
            "additionalProperties": False,
        }
    }
    assert generated_schema == expected_schema


def test_json_schema_extra():
    """Tests that json_schema_extra (dict and callable) is merged into the field schema."""
    generated_schema = generate_validator_schema(JsonSchemaExtraModel)
    expected_schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": sorted(["field_with_extra", "field_with_callable_extra", "field_without_extra"]),
            "properties": {
                "field_with_extra": {
                    "bsonType": "string",
                    "description": "field_with_extra",  # Default description
                    "customMongoKeyword": True,       # From json_schema_extra
                    "anotherKey": 123                 # From json_schema_extra
                },
                "field_with_callable_extra": {
                    "bsonType": "int",
                    "description": "Overridden Description",  # Overridden by json_schema_extra
                    # From json_schema_extra (callable)
                    "dynamicKey": "value"
                },
                "field_without_extra": {
                    "bsonType": "bool",
                    "description": "field_without_extra"  # Default description
                },
            },
            "additionalProperties": False,  # Assuming default config
        }
    }
    assert generated_schema == expected_schema

# --- Test Model for Configuration Options ---


class ConfigTestModel(BaseModel):
    """
    This is a test model description.
    It spans multiple lines.
    """
    model_config = ConfigDict(title="Test Model Title")

    field1: str
    field2: Optional[int] = None


# --- Test Function for Configuration Options ---

def test_generator_configuration_options():
    """
    Tests the include_title and include_description flags.
    """
    base_properties = {
        "field1": {"bsonType": "string", "description": "field1"},
        "field2": {"bsonType": ["int", "null"], "description": "field2"},
    }
    base_required = ["field1"]

    # 1. Default (both True)
    generated_schema_default = generate_validator_schema(ConfigTestModel)
    expected_schema_default = {
        "$jsonSchema": {
            "bsonType": "object",
            "title": "Test Model Title",
            "description": "This is a test model description.\nIt spans multiple lines.",
            "required": base_required,
            "properties": base_properties,
            "additionalProperties": False,  # Assuming default extra='ignore'
        }
    }
    assert generated_schema_default == expected_schema_default

    # 2. include_title=False, include_description=True
    generated_schema_no_title = generate_validator_schema(
        ConfigTestModel, include_title=False
    )
    expected_schema_no_title = {
        "$jsonSchema": {
            "bsonType": "object",
            # No title
            "description": "This is a test model description.\nIt spans multiple lines.",
            "required": base_required,
            "properties": base_properties,
            "additionalProperties": False,
        }
    }
    assert generated_schema_no_title == expected_schema_no_title
    assert "title" not in generated_schema_no_title["$jsonSchema"]

    # 3. include_title=True, include_description=False
    generated_schema_no_desc = generate_validator_schema(
        ConfigTestModel, include_description=False
    )
    expected_schema_no_desc = {
        "$jsonSchema": {
            "bsonType": "object",
            "title": "Test Model Title",
            # No description
            "required": base_required,
            "properties": base_properties,
            "additionalProperties": False,
        }
    }
    assert generated_schema_no_desc == expected_schema_no_desc
    assert "description" not in generated_schema_no_desc["$jsonSchema"]

    # 4. include_title=False, include_description=False
    generated_schema_none = generate_validator_schema(
        ConfigTestModel, include_title=False, include_description=False
    )
    expected_schema_none = {
        "$jsonSchema": {
            "bsonType": "object",
            # No title
            # No description
            "required": base_required,
            "properties": base_properties,
            "additionalProperties": False,
        }
    }
    assert generated_schema_none == expected_schema_none
    assert "title" not in generated_schema_none["$jsonSchema"]
    assert "description" not in generated_schema_none["$jsonSchema"]
