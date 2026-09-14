"""Contains the model for prompt, return types and parameters."""

from typing import Literal
from pydantic import BaseModel, ConfigDict
from pydantic import Field, field_validator, model_validator


VALID_TYPES = {"number", "string", "boolean", "integer"}


class Parameter(BaseModel):
    """Definition of a function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["number", "string", "boolean", "integer"]


class ReturnType(BaseModel):
    """Definition of a function return type."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["number", "string", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    """A single callable function."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, Parameter] = Field(min_length=1)
    returns: ReturnType

    @model_validator(mode="before")
    @classmethod
    def fix_return_type(cls, data: object) -> object:
        """Force an invalid return type to a valid type."""
        if not isinstance(data, dict):
            return data

        returns = data.get("returns")

        if not isinstance(returns, dict):
            return data

        return_type = returns.get("type")

        if return_type not in VALID_TYPES:
            returns["type"] = "string"

        return data


class PromptEntry(BaseModel):
    """Define a valid prompt."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)

    @field_validator("prompt")
    @classmethod
    def not_blank(cls, v: str) -> str:
        """Guard for blank strings."""
        if not v.strip():
            raise ValueError("prompt must not be blank or whitespace-only")
        return v


class FunctionCallresult(BaseModel):
    """A single entry in the output file."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, object]
