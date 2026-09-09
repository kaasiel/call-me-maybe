from enum import Enum, auto
from src import build_prompt, FunctionDefinition

def filter_logits(logits, allowed_tokens):
    """Filter the tokens to get only the right answers."""
    filtered = [float("-inf")] * len(logits)

    for token_id in allowed_tokens:
        filtered[token_id] = logits[token_id]

    return filtered


class State(Enum):
    START = auto()
    PROMPT = auto()
    NAME = auto()
    PARAMETERS = auto()
    END = auto()


class FiniteState():
    def __init__(self,
                 model,
                 prompt: str,
                 functions: list[FunctionDefinition]):
        self.model = model
        self.state = State.START
        self.prompt = prompt
        self.functions = functions
        self.generated_ids = self.get_generated_ids(prompt, functions)
        self.res: list[int] = []

    def get_generated_ids(
            self, prompt: str,
            functions: list[FunctionDefinition]) -> list[int]:
        """Return a list of id."""
        to_send = build_prompt(functions, prompt)
        input_ids = self.model.encode(to_send)
        return input_ids[0].tolist()

    def text_modelisation(self, prompt: list[int]):

        if self.state == State.START:
            self.generated_ids.extend(self.model.encode("{").tolist()[0])
            self.res.append(self.model.encode("{").tolist()[0])
            self.state = State.PROMPT
            return self.generated_ids

        if self.state == State.PROMPT:
            text = f'"prompt": {self.prompt}"'
            self.generated_ids.extend(self.model.encode(text).tolist()[0])
            self.res.append(self.model.encode(text).tolist()[0])
            self.state = State.NAME
            return self.generated_ids

        # if self.state == State.NAME:
        #     name_res = [function.name for function in self.functions]
        #     allowed = self.model.encode(name_res).tolist()[0]
        #     new_logits = filter_logits(prompt, allowed)
        #     next_token_id = new_logits.index(max(new_logits))
        #     self.generated_ids.append(next_token_id)
        #     self.state = State.PARAMETERS
        if self.state == State.NAME:
            function_names = [f.name for f in self.functions]
            name_ids_options = [
                self.model.encode(n).tolist()[0] for n in function_names]

            chosen: list[int] = []
            while True:
                candidates = [
                    ids for ids in name_ids_options
                    if ids[:len(chosen)] == chosen
                ]

                if len(candidates) == 1:
                    remaining = candidates[0][len(chosen):]
                    chosen.extend(remaining)
                    self.generated_ids.extend(remaining)
                    chosen_index = name_ids_options.index(candidates[0])
                    self.chosen_function = self.functions[chosen_index]
                    break

                allowed = {ids[len(chosen)] for ids in candidates}
                logits = self.model.get_logits_from_input_ids(
                    self.generated_ids)
                masked_logits = filter_logits(logits, allowed)
                next_token_id = masked_logits.index(max(masked_logits))
                chosen.append(next_token_id)
                self.generated_ids.append(next_token_id)

            self.state = State.PARAMETERS
            return self.generated_ids

        if self.state == State.PARAMETERS:
            allowed = self.model.encode('", "parameters": {}}').tolist()[0]
            new_logits = filter_logits(prompt, allowed)
            next_token_id = new_logits.index(max(new_logits))
            self.generated_ids.append(next_token_id)
            self.state = State.END

        if self.state == State.END:
            self.generated_ids.extend(self.model.encode("}").tolist()[0])
            self.res.append(self.model.encode("{").tolist()[0])
            self.state = State.PROMPT
            return self.generated_ids

        return self.generated_ids
