"""Contains the model for prompst, return types and parametres."""

from pydantic import BaseModel
from typing import Literal


class Parameter(BaseModel):
    """Definiton of function parameter."""

    type: Literal["number", "string", "boolean"]


class ReturnType(BaseModel):
    """Definition of a function return type."""

    type: Literal["number", "string", "boolean"]


class FunctionDefinition(BaseModel):
    """A single callable function like in the functon_definitions."""

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType


class PromptEntry(BaseModel):
    """A single entry from function-calling_test."""

    prompt: str


class FunctionCallresult(BaseModel):
    """A single entry in the output file."""

    prompt: str
    name: str
    parameters: dict[str, object]
