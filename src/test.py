import json
from enum import Enum, auto
from src import FunctionDefinition, build_prompt, FunctionCallresult


def filter_logits(logits, allowed_tokens):
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


class JSONenforce:
    def __init__(self, model, prompt: str, functions: list[FunctionDefinition]):
        self.model = model
        self.state = State.START
        self.prompt = prompt
        self.functions = functions
        self.res = []
        self.input_ids = []
        self.chosen_func = None

        to_send = build_prompt(functions, prompt)
        self.input_ids = self.model.encode(to_send).tolist()[0]

    def tokeniser(self, text: str) -> None:
        ids = self.model.encode(text).tolist()[0]
        self.input_ids.extend(ids)
        self.res.extend(ids)

    def output_modelisation(self):
        # START → PROMPT
        if self.state == State.START:
            self.tokeniser('{"prompt": "')
            self.state = State.PROMPT

        # PROMPT → NAME
        if self.state == State.PROMPT:
            self.tokeniser(self.prompt)
            self.tokeniser('", "name": "')
            self.state = State.NAME

        # NAME → PARAMETERS
        if self.state == State.NAME:
            function_names = [fn.name for fn in self.functions]
            name_ids = [self.model.encode(name).tolist()[0] for name in function_names]

            chosen = []
            while True:
                candidates = [ids for ids in name_ids if ids[:len(chosen)] == chosen]

                if not candidates:
                    raise ValueError("No valid function name")

                if len(candidates) == 1:
                    rest = candidates[0][len(chosen):]
                    self.input_ids.extend(rest)
                    self.res.extend(rest)

                    chosen_index = name_ids.index(candidates[0])
                    self.chosen_func = self.functions[chosen_index]

                    self.tokeniser('", "parameters": {')
                    self.state = State.PARAMETERS
                    break

                allowed = {ids[len(chosen)] for ids in candidates}
                logits = self.model.get_logits_from_input_ids(self.input_ids)
                masked_logits = filter_logits(logits, allowed)
                next_token_id = masked_logits.index(max(masked_logits))

                chosen.append(next_token_id)
                self.input_ids.append(next_token_id)
                self.res.append(next_token_id)

        # PARAMETERS → END
        if self.state == State.PARAMETERS:
            param_items = list(self.chosen_func.parameters.items())

            for i, (param_name, param) in enumerate(param_items):
                key_text = f'"{param_name}": '
                self.tokeniser(key_text)

                if param.type == "string":
                    self.tokeniser('"')
                    terminator = ['"']
                    allowed_char = None
                elif param.type == "number":
                    allowed_char = "0123456789.-"
                    terminator = [",", "}"]
                else:
                    allowed_char = "true, false, TRUE, FALSE"
                    terminator = [",", "}"]

                while True:
                    logits = self.model.get_logits_from_input_ids(self.input_ids)

                    if allowed_char is not None:
                        allowed_ids = set()
                        for char in allowed_char:
                            ids = self.model.encode(char).tolist()[0]
                            allowed_ids.update(ids)
                        for char in terminator:
                            ids = self.model.encode(char).tolist()[0]
                            allowed_ids.update(ids)
                    else:
                        allowed_ids = set(range(len(logits)))

                    masked_logits = filter_logits(logits, allowed_ids)
                    next_token_id = masked_logits.index(max(masked_logits))
                    decoded = self.model.decode([next_token_id])

                    if decoded in terminator:
                        # Terminator only signals "stop" — the real
                        # punctuation is added by the surrounding code,
                        # so we don't commit this token to the output.
                        if param.type == "string":
                            self.tokeniser('"')
                        break

                    self.input_ids.append(next_token_id)
                    self.res.append(next_token_id)

                if i < len(param_items) - 1:
                    self.tokeniser(", ")

            self.tokeniser("}")
            self.state = State.END

        # PARAMETERS → END (close the outer object)
        if self.state == State.END:
            self.tokeniser("}")

        decoded = self.model.decode(self.res)

        try:
            data = json.loads(decoded)
        except json.JSONDecodeError as e:
            raise ValueError(f"Json invalide: {decoded}") from e

        return FunctionCallresult(**data)