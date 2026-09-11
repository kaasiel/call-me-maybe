"""Contains the model for prompt, return types and parametres."""

from typing import Literal
from pydantic import field_validator
from pydantic import BaseModel, ConfigDict, Field


class Parameter(BaseModel):
    """Definiton of function parameter."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["number", "string", "boolean", "integer"]


class ReturnType(BaseModel):
    """Definition of a function return type."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["number", "string", "boolean", "integer"]


class FunctionDefinition(BaseModel):
    """A single callable function like in the functon_definitions."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, Parameter] = Field(min_length=1)
    returns: ReturnType


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
