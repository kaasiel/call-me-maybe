"""This lauches all th functons that runs the program."""
import os
import json
import argparse
from typing import Any
from functools import wraps
from time import perf_counter
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from collections.abc import Callable
from src import JSONenforce, header_printer
from src import function_loader, prompt_loader, FunctionCallresult
from src import FunctionDefinition, print_pretty, normalize_params


def args_parser() -> argparse.Namespace:
    """Parse the arguments and set default values."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definitions",
        default="data/input/functions_definition.json",
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )

    parser.add_argument(
        "--output",
        default="data/output/functions_call_result.json",
    )

    return parser.parse_args()


def spell_timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """Timer warper used to time the function time."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"Casting {func.__name__}...")
        start = perf_counter()
        res = func(*args, **kwargs)
        end = perf_counter()
        return (res, end - start)
    return wrapper


def jsonencode(model: Small_LLM_Model, prompt_text: str,
               functions: list[FunctionDefinition]) -> FunctionCallresult:
    """Call the FSM to model the output as valid JSON."""
    fsm = JSONenforce(model, prompt_text, functions)
    return fsm.output_modelisation()


@spell_timer
def main() -> None:
    """Launch all the codes."""
    args = args_parser()
    prompts = prompt_loader(args.input)
    functions = function_loader(args.functions_definitions)

    total_time = 0.0
    result = []
    for prompt in prompts:
        try:
            model = Small_LLM_Model()
            start = perf_counter()
            result_obj = jsonencode(model, prompt.prompt, functions)
            elapsed = perf_counter() - start

            data = normalize_params(result_obj.model_dump(), functions)
            print_pretty(data)

            total_time += elapsed
            result.append(data)

        except (Exception, UnboundLocalError) as e:
            print(f"Error for '{prompt.prompt}': {e}")
            print("\n" + '*' * 10)

    print(f"\nTotal time for {len(prompts)} prompts: {total_time:.3f}s")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w') as file:
        json.dump(result, file, indent=2)

    print(f"Result saved to {args.output}")


if __name__ == "__main__":
    try:
        header_printer()
        main()
    except KeyboardInterrupt:
        print("\n Operation aborted")
