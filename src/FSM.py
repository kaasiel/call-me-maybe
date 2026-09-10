"""Finite-state JSON generation helpers for constrained function calls."""

import json
from enum import Enum, auto
from src import FunctionDefinition, build_prompt, FunctionCallresult


def filter_logits(logits, allowed_tokens):
    """Retain only the logits for the allowed token IDs."""
    filtered = [float("-inf")] * len(logits)
    for token_id in allowed_tokens:
        filtered[token_id] = logits[token_id]
    return filtered


def quote_counter(text: str) -> bool:
    """Return whether an odd number of quote marks has been seen."""
    return text.count('"') % 2 == 1


class State(Enum):
    """Manage the FSM states."""

    START = auto()
    PROMPT = auto()
    NAME = auto()
    PARAMETERS = auto()
    END = auto()


class JSONenforce:
    """Enforce JSON parsing for model output."""

    NUMBERS_CHAR = ".-0123456789"
    A_TERMINATOR = [",", "}"]
    MAX_STRING_TOKENS = 200
    MAX_NUMBER_TOKENS = 40

    def __init__(self,
                 model,
                 prompt: str,
                 functions: list[FunctionDefinition]):
        self.model = model
        self.state = State.START
        self.prompt = prompt
        self.functions = functions
        self.res: list[int] = []
        self.input_ids: list[int] = []
        self.chosen_func = None

        to_send = build_prompt(functions, prompt)
        self.input_ids = self.model.encode(to_send).tolist()[0]
        self._number_allowed_ids = set()
        for char in self.NUMBERS_CHAR + "".join(self.A_TERMINATOR):
            self._number_allowed_ids.update(
                self.model.encode(char).tolist()[0]
            )

    def tokeniser(self, text: str) -> None:
        ids = self.model.encode(text).tolist()[0]
        self.input_ids.extend(ids)
        self.res.extend(ids)

    def _walk_fixed_choices(self, choices: list[str]) -> str:
        choice_ids = [self.model.encode(c).tolist()[0] for c in choices]
        chosen: list[int] = []
        while True:
            candidates = [
                ids for ids in choice_ids
                if ids[:len(chosen)] == chosen
                ]

            if not candidates:
                raise ValueError(f"No valid choice among {choices}")
            if len(candidates) == 1:
                rest = candidates[0][len(chosen):]
                self.input_ids.extend(rest)
                self.res.extend(rest)
                return choices[choice_ids.index(candidates[0])]
            allowed = {ids[len(chosen)] for ids in candidates}
            logits = self.model.get_logits_from_input_ids(self.input_ids)
            masked_logits = filter_logits(logits, allowed)
            next_token_id = masked_logits.index(max(masked_logits))
            chosen.append(next_token_id)
            self.input_ids.append(next_token_id)
            self.res.append(next_token_id)

    def _generate_string(self, param_name: str) -> None:
        self.tokeniser('"')
        temporary = []
        ended = False

        for _ in range(self.MAX_STRING_TOKENS):
            logits = self.model.get_logits_from_input_ids(self.input_ids)
            next_token_id = max(range(len(logits)), key=lambda i: logits[i])

            temporary.append(next_token_id)
            decoded_so_far = self.model.decode(temporary)

            print(f"\r  [{param_name}] generating: \"{decoded_so_far}\"",
                  end="", flush=True)
            if quote_counter(decoded_so_far):
                temporary.pop()
                ended = True
                break

            self.input_ids.append(next_token_id)
            self.res.append(next_token_id)

        print()

        if not ended:
            print(
                f"[warn] string param '{param_name}' "
                f"truncated at {self.MAX_STRING_TOKENS} tokens"
            )

        self.tokeniser('"')

    def _generate_number(self, param_name: str) -> None:
        iters = 0

        while True:
            iters += 1
            if iters > self.MAX_NUMBER_TOKENS:
                raise ValueError(
                    f"Number param '{param_name}' did not terminate")

            logits = self.model.get_logits_from_input_ids(self.input_ids)

            masked_logits = filter_logits(
                logits, self._number_allowed_ids
            )
            next_token_id = masked_logits.index(max(masked_logits))
            decoded = self.model.decode([next_token_id])

            print(f"\r  [{param_name}] generating: \"{decoded}\"",
                  end="", flush=True)
            if decoded in self.A_TERMINATOR:
                break

            print()

            self.input_ids.append(next_token_id)
            self.res.append(next_token_id)

    def output_modelisation(self):
        if self.state == State.START:
            self.tokeniser('{"prompt": "')
            self.state = State.PROMPT

        if self.state == State.PROMPT:
            escaped_prompt = json.dumps(self.prompt)[1:-1]
            self.tokeniser(escaped_prompt)
            self.tokeniser('", "name": "')
            self.state = State.NAME

        if self.state == State.NAME:
            function_names = [fn.name for fn in self.functions]
            chosen_name = self._walk_fixed_choices(function_names)
            self.chosen_func = self.functions[
                function_names.index(chosen_name)]
            self.tokeniser('", "parameters": {')
            self.state = State.PARAMETERS

        if self.state == State.PARAMETERS:
            param_items = list(self.chosen_func.parameters.items())

            for i, (param_name, param) in enumerate(param_items):
                key_text = f'"{param_name}":'
                self.tokeniser(key_text)

                if param.type == "string":
                    self._generate_string(param_name)
                elif param.type == "number":
                    self._generate_number(param_name)
                elif param.type == "boolean":
                    self._walk_fixed_choices(["true", "false"])
                else:
                    raise NotImplementedError(
                        f"Unsupported param type: {param.type}")

                if i < len(param_items) - 1:
                    self.tokeniser(", ")

            self.tokeniser("}")
            self.state = State.END

        if self.state == State.END:
            self.tokeniser("}")

        decoded = self.model.decode(self.res)

        try:
            data = json.loads(decoded)
        except json.JSONDecodeError as e:
            raise ValueError(f"Json invalide: {decoded}") from e

        return FunctionCallresult(**data)
