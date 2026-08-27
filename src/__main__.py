from src.lms import State
from src.lms import ConstrainedJSONFSM
from llm_sdk import Small_LLM_Model
from src import FunctionCallresult
from src import function_loader, prompt_loader, build_prompt
from typing import Generator

model = Small_LLM_Model()


prompts = prompt_loader("/goinfre/belaindr/call-me-maybe/data/input/function_calling_tests.json")
functions = function_loader("/goinfre/belaindr/call-me-maybe/data/input/functions_definition.json")

max_new_tokens = 5


def loading() -> Generator[None, None, None]:
    """Générateur qui affiche l'indicateur de chargement, en boucle infinie."""
    count = 1
    while True:
        print("\033[K" + "Loading" + "." * count, end="\r")
        count += 1
        if count > 3:
            count = 1
        yield


loading_gen = loading()
fms = ConstrainedJSONFSM(functions)
for prompt in prompts:
    to_send = build_prompt(functions, prompt.prompt)
    input_ids = model.encode(to_send)
    generated_ids = input_ids[0].tolist()

    for i in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(generated_ids)
        next_token_id = logits.index(max(logits))
        generated_ids.append(next_token_id)
        next(loading_gen)
        if next_token_id == model._tokenizer.eos_token_id:
            break

returned_text = model.decode(generated_ids)
print(returned_text)
