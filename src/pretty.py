"""This module makes the output pretiier and fix the parameter."""
import json
from typing import Any
from src import FunctionDefinition
RESET = "\033[0m"
PROMPT = "\033[38;5;51m"
FUNC_NAME = "\033[38;5;208m"
PARAM = "\033[38;5;46m"


def normalize_params(
        data: dict[str, Any],
        functions: list[FunctionDefinition]) -> dict[str, Any]:
    """Correct the output type."""
    params = data["parameters"]
    match = next(
        (
            func for func in
            functions if func.name == data["name"]
            ), None)

    schema = match.parameters if match else {}

    for key, value in params.items():
        param_def = schema.get(key)
        if param_def is not None and param_def.type == "number":
            params[key] = float(value)
    return data


def print_pretty(to_use: dict[str, Any]) -> None:
    """Make the printing prettier by using ANSI colors."""
    print("\n{")
    print(f'  "prompt": {PROMPT}"{to_use["prompt"]}"{RESET},')
    print(f'  "name": {FUNC_NAME}"{to_use["name"]}"{RESET},')

    params_str = json.dumps(to_use["parameters"], indent=2)
    indented = "\n".join("  " + line for line in params_str.splitlines())
    print(f'  "parameters": {PARAM}{indented}{RESET}')

    print("}")


def header_printer() -> None:
    """Print the header."""
    lines = [
        "▗▄▄▖ ▗▄▖ ▗▖   ▗▖       ▗▖  ▗▖▗▄▄▄▖    ▗▖  ▗▖ ▗▄▖▗▖  ▗▖▗▄▄▖ ▗▄▄▄▖",
        "▐▌   ▐▌ ▐▌▐▌   ▐▌       ▐▛▚▞▜▌▐▌       ▐▛▚▞▜▌▐▌ ▐▌▝▚▞▘ ▐▌ ▐▌▐▌   ",
        "▐▌   ▐▛▀▜▌▐▌   ▐▌       ▐▌  ▐▌▐▛▀▀▘    ▐▌  ▐▌▐▛▀▜▌ ▐▌  ▐▛▀▚▖▐▛▀▀▘",
        "▝▚▄▄▖▐▌ ▐▌▐▙▄▄▖▐▙▄▄▖    ▐▌  ▐▌▐▙▄▄▖    ▐▌  ▐▌▐▌ ▐▌ ▐▌  ▐▙▄▞▘▐▙▄▄▖",
    ]

    colors = [PROMPT, PROMPT, PARAM, PARAM]

    for line, color in zip(lines, colors):
        print(f"{color}{line.center(100)}{RESET}")
