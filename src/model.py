"""Contains the model for prompt, return types and parametres."""

from pydantic import BaseModel, ConfigDict
from typing import Literal


class Parameter(BaseModel):
    """Definiton of function parameter."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["number", "string", "boolean"]


class ReturnType(BaseModel):
    """Definition of a function return type."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["number", "string", "boolean"]


class FunctionDefinition(BaseModel):
    """A single callable function like in the functon_definitions."""
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType


class PromptEntry(BaseModel):
    """A single entry from function-calling_test."""
    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCallresult(BaseModel):
    """A single entry in the output file."""
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, object]
