"""This module makes the output pretiier and fix the parameter."""
import json

RESET = "\033[0m"
PROMPT = "\033[38;5;51m"
FUNC_NAME = "\033[38;5;208m"
PARAM = "\033[38;5;46m"


def normalize_params(data: dict) -> dict:
    """Convert int parameter values to float."""
    params = data["parameters"]
    for key, value in params.items():
        if isinstance(value, int) and not isinstance(value, bool):
            params[key] = float(value)
    return data


def print_pretty(to_use: dict) -> None:
    """Make the printing prettier by using ANSI colors."""
    print("{")
    print(f'  "prompt": {PROMPT}"{to_use["prompt"]}"{RESET},')
    print(f'  "name": {FUNC_NAME}"{to_use["name"]}"{RESET},')

    params_str = json.dumps(to_use["parameters"], indent=2)
    indented = "\n".join("  " + line for line in params_str.splitlines())
    print(f'  "parameters": {PARAM}{indented}{RESET}')

    print("}")
