"""This file contains thef ucntion needed to load the jsonfiles."""

from src.model import FunctionDefinition, PromptEntry
from pydantic import TypeAdapter
import json
import pydantic
import sys


def function_loader(filepath: str) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
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


# from llm_sdk import Small_LLM_Model
# import torch


# model = Small_LLM_Model()

# input_ids = model.encode("Quel temps fait-il à Paris ?")
# generated_ids = input_ids[0].tolist()

# max_new_tokens = 50

# for _ in range(max_new_tokens):
#     logits = model.get_logits_from_input_ids(generated_ids)
#     next_token_id = torch.tensor(logits).argmax().item()
#     generated_ids.append(next_token_id)
#     if next_token_id == model._tokenizer.eos_token_id:
#         break

# result_text = model.decode(generated_ids)
# print(result_text)
