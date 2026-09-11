"""This module loads and validates JSON function/prompt definition files."""

from src.model import FunctionDefinition, PromptEntry
from pydantic import TypeAdapter
import json
import pydantic
import sys


def function_loader(filepath: str) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    result: list[FunctionDefinition] = []
    try:
        with open(filepath, "r") as f:
            raw = json.load(f)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
        print(f"function error: {e}")
        sys.exit(1)

    adapter = TypeAdapter(FunctionDefinition)
    for i, element in enumerate(raw):
        try:
            result.append(adapter.validate_python(element))
        except pydantic.ValidationError as error:
            msg = "; ".join(e["msg"] for e in error.errors())
            print(f"function {i} rejected: {msg}\n")

    if not result:
        print("function error: no valid function definitions found")
        sys.exit(1)

    print(f"{len(result)}/{len(raw)} functions loaded successfully")
    return result


def prompt_loader(filepath: str) -> list[PromptEntry]:
    """Load and validate prompts from a JSON file."""
    result: list[PromptEntry] = []
    try:
        with open(filepath, "r") as f:
            raw = json.load(f)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
        print(f"prompt error: {e}")
        sys.exit(1)

    adapter = TypeAdapter(PromptEntry)
    for i, element in enumerate(raw):
        try:
            result.append(adapter.validate_python(element))
        except pydantic.ValidationError as error:
            msg = "; ".join(e["msg"] for e in error.errors())
            print(f"prompt {i}: {msg}")

    if not result:
        print("prompt warning: no valid prompts found, writing empty results")

    print(f"{len(result)}/{len(raw)} prompts loaded successfully")
    return result


def build_prompt(
    functions: list[FunctionDefinition],
    prompts: str,
) -> str:
    """Build a formatted prompt for a function-calling assistant.

    Constructs a prompt string that includes available function definitions
    and the user request, formatted for JSON function-call response generation.
    """
    function_lines = [
        "- {}({}): {}".format(
            function.name,
            ", ".join(
                f"{name}: {parameter.type}"
                for name, parameter in function.parameters.items()
            ),
            function.description,
        )
        for function in functions
    ]
    function_tab = "\n".join(function_lines)

    return (
        "respond ONLY in the format "
        '{"name": "<function_name>", "parameters": {...}}\n\n'
        f"Available functions:\n{function_tab}\n\n"
        f"User request: {prompts}\n"
    )