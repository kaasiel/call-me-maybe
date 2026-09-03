from enum import Enum, auto

from src import build_prompt, FunctionDefinition


def filter_logits(logits, allowed_tokens):
    """Filter the tokens to keep only allowed tokens."""
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


class FiniteState:
    def __init__(
        self,
        model,
        prompt: str,
        functions: list[FunctionDefinition],
    ):
        self.model = model
        self.state = State.START
        self.prompt = prompt
        self.functions = functions

        self.generated_ids = self.get_generated_ids(
            prompt,
            functions,
        )

        self.res: list[int] = []
        self.chosen_function = None

    def get_generated_ids(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> list[int]:
        """Return the input token IDs."""
        to_send = build_prompt(functions, prompt)
        input_ids = self.model.encode(to_send)

        return input_ids[0].tolist()

    def _generate_fixed_text(self, text: str):
        """Add a fixed piece of text to the generated result."""
        ids = self.model.encode(text).tolist()[0]

        self.generated_ids.extend(ids)
        self.res.extend(ids)

    def _generate_value(self, param_type: str):
        """Generate a parameter value according to its type."""

        if param_type == "number":
            allowed_chars = "0123456789.-"
        else:
            allowed_chars = None

        while True:
            logits = self.model.get_logits_from_input_ids(
                self.generated_ids
            )

            if allowed_chars is not None:
                allowed_ids = set()

                for char in allowed_chars:
                    ids = self.model.encode(char).tolist()[0]
                    allowed_ids.update(ids)

                # Allow the end of the value
                for char in [",", "}"]:
                    ids = self.model.encode(char).tolist()[0]
                    allowed_ids.update(ids)

            else:
                # For strings, don't restrict the content yet.
                allowed_ids = set(range(len(logits)))

            masked_logits = filter_logits(
                logits,
                allowed_ids,
            )

            next_token_id = masked_logits.index(
                max(masked_logits)
            )

            self.generated_ids.append(next_token_id)
            self.res.append(next_token_id)

            decoded = self.model.decode([next_token_id])

            if param_type == "number":
                if decoded in (",", "}"):
                    break

            else:
                if decoded == '"':
                    break

    def text_modelisation(self):
        """Generate the JSON according to the FSM."""

        # ----------------
        # START
        # ----------------
        if self.state == State.START:
            self._generate_fixed_text('{"prompt": "')
            self.state = State.PROMPT

        # ----------------
        # PROMPT
        # ----------------
        if self.state == State.PROMPT:
            self._generate_fixed_text(self.prompt)
            self._generate_fixed_text('", "name": "')

            self.state = State.NAME

        # ----------------
        # NAME
        # ----------------
        if self.state == State.NAME:

            function_names = [
                function.name
                for function in self.functions
            ]

            name_ids_options = [
                self.model.encode(name).tolist()[0]
                for name in function_names
            ]

            chosen = []

            while True:
                candidates = [
                    ids
                    for ids in name_ids_options
                    if ids[:len(chosen)] == chosen
                ]

                if not candidates:
                    raise ValueError(
                        "No valid function name found."
                    )

                # One function remains
                if len(candidates) == 1:
                    remaining = candidates[0][len(chosen):]

                    self.generated_ids.extend(remaining)
                    self.res.extend(remaining)

                    chosen_index = name_ids_options.index(
                        candidates[0]
                    )

                    self.chosen_function = (
                        self.functions[chosen_index]
                    )

                    break

                allowed = {
                    ids[len(chosen)]
                    for ids in candidates
                }

                logits = self.model.get_logits_from_input_ids(
                    self.generated_ids
                )

                masked_logits = filter_logits(
                    logits,
                    allowed,
                )

                next_token_id = masked_logits.index(
                    max(masked_logits)
                )

                chosen.append(next_token_id)

                self.generated_ids.append(
                    next_token_id
                )

                self.res.append(
                    next_token_id
                )

            self._generate_fixed_text(
                '", "parameters": {'
            )

            self.state = State.PARAMETERS

        # ----------------
        # PARAMETERS
        # ----------------
        if self.state == State.PARAMETERS:

            param_items = list(
                self.chosen_function.parameters.items()
            )

            for i, (param_name, param) in enumerate(
                param_items
            ):
                key_text = f'"{param_name}": '

                # String parameter needs opening quote
                if param.type == "string":
                    key_text += '"'

                self._generate_fixed_text(key_text)

                self._generate_value(param.type)

                if i < len(param_items) - 1:
                    self._generate_fixed_text(", ")

            self._generate_fixed_text("}")

            self.state = State.END

        # ----------------
        # END
        # ----------------
        if self.state == State.END:
            self._generate_fixed_text("}")

        return self.res