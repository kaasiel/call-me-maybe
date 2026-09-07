from llm_sdk import Small_LLM_Model
from src import function_loader, prompt_loader, FunctionCallresult
from src.test import JSONenforce

# Initialize model
model = Small_LLM_Model()

# Load prompts and functions
prompts = prompt_loader("/goinfre/belaindr/call-me-maybe/data/input/function_calling_tests.json")
functions = function_loader("/goinfre/belaindr/call-me-maybe/data/input/functions_definition.json")


def jsonencode(prompt_text: str) -> FunctionCallresult:
    fsm = JSONenforce(model, prompt_text, functions)
    return fsm.output_modelisation()

def main():
    for prompt in prompts:
        try:
            result_obj = jsonencode(prompt.prompt)
            print(result_obj.json())
            print()
        except Exception as e:
            print(f"Erreur pour '{prompt.prompt}': {e}")
            print()



if __name__ == "__main__":
    main()
