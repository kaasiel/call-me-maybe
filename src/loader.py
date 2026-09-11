"""This file contains thef ucntion needed to load the jsonfiles."""

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
            res = json.load(f)
            adapter = TypeAdapter(list[FunctionDefinition])
            result = adapter.validate_python(res)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
        print(f"function error: {e}")
        sys.exit(1)
    except pydantic.ValidationError as error:
        print(f"Functions validation error: {error}")
        return result
    return result


def prompt_loader(filepath: str) -> list[PromptEntry]:
    """Load and validate prompts from a JSON file."""
    result: list[PromptEntry] = []
    try:
        with open(filepath, "r") as f:
            res = json.load(f)
            adapter = TypeAdapter(PromptEntry)
            for i, elements in enumerate(res):
                try:
                    ele_temps = adapter.validate_python(elements)
                    result.append(ele_temps)
                except pydantic.ValidationError as error:
                    msg = "; ".join(e['msg'] for e in error.errors())
                    print(f"prompt: {i}: {msg}")
    except (FileNotFoundError, PermissionError,
            json.JSONDecodeError, pydantic.ValidationError) as e:
        print(f"function error: {e}")
        sys.exit(1)

    print(f"{len(result)}/{len(res)} prompts loaded successfully")
    return result


def build_prompt(
    functions: list[FunctionDefinition],
    prompts: str,
) -> str:
    """Build a formatted prompt for a function-calling assistant.

    Constructs a prompt string that includes available function definitions
    and the user request, formatted for JSON function-call response generation.
    """
    function_lines = []

    for function in functions:
        parameters = ", ".join(
            f"{name}: {parameter.type}"
            for name, parameter in function.parameters.items()
        )
        function_lines.append(
            f"- {function.name}({parameters}): {function.description}"
        )

    function_tab = "\n".join(function_lines)

    return (
        "respond ONLY in the format "
        '{"name": "<function_name>", "parameters": {...}}\n\n'
        f"Available functions:\n{function_tab}\n\n"
        f"User request: {prompts}\n"
    )
