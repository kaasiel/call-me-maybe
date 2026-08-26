from llm_sdk import Small_LLM_Model
from src import FunctionCallresult
from src import function_loader, prompt_loader, build_prompt

model = Small_LLM_Model()


functions = function_loader("data/input/functions_definitions.json")
prompts = prompt_loader("data/input/functions_calling_test.json")

max_new_tokens = 100

for prompt in prompts:
    to_send = build_prompt(functions, prompt.prompt)
    input_ids = model.encode(to_send)
    generated_ids = input_ids[0].tolist()
    logits = model.get_logits_from_input_ids(generated_ids)
    next_token_id = logits.index(max(logits))

# for _ in range(max_new_tokens):
#     logits = model.get_logits_from_input_ids(generated_ids)
#     next_token_id = torch.tensor(logits).argmax().item()
#     generated_ids.append(next_token_id)
#     if next_token_id == model._tokenizer.eos_token_id:
#         break

# result_text = model.decode(generated_ids)
# print(result_text)
