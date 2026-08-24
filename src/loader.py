from src.model import FunctionDefinition, PromptEntry
from pydantic import TypeAdapter
import json
import pydantic
import sys


def function_loader(filepath: str) -> list[FunctionDefinition]:
    try:
        with open(filepath, "r") as f:
            res = json.load(f)
            adapter = TypeAdapter(list[FunctionDefinition])
            result = adapter.validate_python(res)
    except (FileNotFoundError, PermissionError,
            json.JSONDecodeError, pydantic.ValidationError) as e:
        print(f"function error: {e}")
        sys.exit(1)
    return result


def prompt_loader(filepath: str) -> list[PromptEntry]:
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
                    print(f"prompt: {i}: {error.errors()[0]['msg']}")
    except (FileNotFoundError, PermissionError,
            json.JSONDecodeError, pydantic.ValidationError) as e:
        print(f"function error: {e}")
        sys.exit(1)
        # return [] a mediter

    print(f"{len(result)}/{len(res)} prompts loaded successfully")
    return result


print(function_loader("/home/belaindr/goinfre/call-me-maybe/data/input/functions_definition.json"))
print("\n" * 3)
print(prompt_loader("/home/belaindr/goinfre/call-me-maybe/data/input/function_calling_tests.json"))
