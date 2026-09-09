# import json

# RESET = "\033[0m"

# PROMPT = "\033[38;5;51m"
# FUNC_NAME = "\033[38;5;208m"
# PARAM = "\033[38;5;46m"
# KEY = "\033[38;5;226m"


# def print_pretty(text: str) -> None:
#     to_use = json.loads(text)

#     end = "}"
#     start = "{"
#     re = f"{PARAM}{start}{RESET}\n"
#     re += f'  {PROMPT}"prompt{RESET}": {to_use["prompt"]},\n'
#     re += f'  {FUNC_NAME}"name{RESET}": {to_use["name"]},\n'
#     re += f'  {KEY}"parameters{RESET}": {json.dumps(to_use["parameters"])}\n'
#     re += f"{PARAM}{end}{RESET}"

#     print(re)

import json

RESET = "\033[0m"
PROMPT = "\033[38;5;51m"
FUNC_NAME = "\033[38;5;208m"
PARAM = "\033[38;5;46m"


def print_pretty(text: str) -> None:
    to_use = json.loads(text)

    print("{")
    print(f'  "prompt": {PROMPT}"{to_use["prompt"]}"{RESET},')
    print(f'  "name": {FUNC_NAME}"{to_use["name"]}"{RESET},')

    params_str = json.dumps(to_use["parameters"], indent=2)
    indented = "\n".join("  " + line for line in params_str.splitlines())
    print(f'  "parameters": {PARAM}{indented}{RESET}')

    print("}")
