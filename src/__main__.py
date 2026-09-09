from functools import wraps
from llm_sdk import Small_LLM_Model
from src import function_loader, prompt_loader, FunctionCallresult
from src.test import JSONenforce
from time import perf_counter
from typing import Any
from collections.abc import Callable

model = Small_LLM_Model()

prompts = prompt_loader("/goinfre/belaindr/call-me-maybe/data/input/function_calling_tests.json")
functions = function_loader("/goinfre/belaindr/call-me-maybe/data/input/functions_definition.json")


def spell_timer(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"Casting {func.__name__}...")
        start = perf_counter()
        res = func(*args, **kwargs)
        end = perf_counter()
        print(f"Spell completed in {end - start:.3f} seconds")
        return res
    return wrapper


def jsonencode(prompt_text: str) -> FunctionCallresult:
    fsm = JSONenforce(model, prompt_text, functions)
    return fsm.output_modelisation()


@spell_timer
def main() -> float:
    total_time = 0.0

    for prompt in prompts:
        try:
            start = perf_counter()
            result_obj = jsonencode(prompt.prompt)
            elapsed = perf_counter() - start

            print(result_obj.model_dump_json())
            print(f"time: {elapsed:.3f}s")
            print('*' * 10)

            total_time += elapsed

        except Exception as e:
            print(f"Erreur pour '{prompt.prompt}': {e}")
            print('*' * 10)

    print(f"\nTotal time for {len(prompts)} prompts: {total_time:.3f}s")
    return total_time


if __name__ == "__main__":
    main()
