from llm_sdk import Small_LLM_Model
from src import function_loader, prompt_loader, build_prompt
from typing import Generator
from src.test import FiniteState, State
model = Small_LLM_Model()

prompts = prompt_loader("/goinfre/belaindr/call-me-maybe/data/input/function_calling_tests.json")
functions = function_loader("/goinfre/belaindr/call-me-maybe/data/input/functions_definition.json")


def loading() -> Generator[None, None, None]:
    """Générateur qui affiche l'indicateur de chargement, en boucle infinie."""
    count = 1
    while True:
        print("\033[K" + "Loading" + "." * count, end="\r")
        count += 1
        if count > 3:
            count = 1
        yield


def filter_logits(logits, allowed_tokens):
    """Filter the tokens to get only the right answers."""
    filtered = [float("-inf")] * len(logits)

    for token_id in allowed_tokens:
        filtered[token_id] = logits[token_id]

    return filtered

# def filter_logits(logits, allowed_logits):
#     return [
#         l if i in allowed_logits
#         else float('-inf') for i, l in enumerate(logits)]


loading_gen = loading()

func: list[str] = []
for function in functions:
    parameters = ", ".join(
        f"{name}: {parameter.type}"
        for name, parameter in function.parameters.items()
        )
    func.append(
        f"- {function.name}({parameters}): {function.description}"
        )
function_tab = "\n".join(func)


def get_generated_ids(prompt) -> list[int]:
    """Return a list of id."""
    to_send = build_prompt(functions, prompt.prompt)
    input_ids = model.encode(to_send)
    return input_ids[0].tolist()


for prompt in prompts:
    fsm = FiniteState(model, prompt.prompt, functions)

    fsm.text_modelisation()

    result = model.decode(fsm.generated_ids)

    print(result)
    print()
    # generated_ids = get_generated_ids(prompt)
    # allowed_text = (
    #     '{"prompt": "'
    #     + prompt.prompt
    #     + '", "name": "'
    #     + function_tab
    #     + '", "parameters": {}}')
    # limiter = set(model.encode(allowed_text).tolist()[0])
    # count = 0
    # li = 0
    # while True and li < 100:
    #     logits = model.get_logits_from_input_ids(generated_ids)
    #     masked_logits = filter_logits(logits, limiter)
    #     next_token_id = masked_logits.index(max(masked_logits))
    #     generated_ids.append(next_token_id)
    #     res.append(next_token_id)
    #     next(loading_gen)
    #     decoded_so_far = model.decode(res)
    #     brace_depth = decoded_so_far.count("{") - decoded_so_far.count("}")
    #     if brace_depth == 0 and "{" in decoded_so_far:
    #         break
    #     li += 1
    # returned_text = model.decode(res)

    # print(returned_text)
