"""Finite-state JSON generation helpers for constrained function calls."""

import functools
import json
from enum import Enum, auto
from collections.abc import Iterable
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src import FunctionDefinition, build_prompt, FunctionCallresult


def filter_logits(
        logits: list[int],
        allowed_tokens: Iterable[int]
        ) -> list[float]:
    """Retain only the logits for the allowed token IDs."""
    filtered = [float("-inf")] * len(logits)
    for token_id in allowed_tokens:
        filtered[token_id] = logits[token_id]
    return filtered


@functools.lru_cache(maxsize=4)
def _load_control_char_ids(
        vocab_path: str, control_chars: str) -> frozenset[int]:
    """Return vocab token ids whose text contains a raw control character.

    Built once per vocab file -- and cached across every JSONenforce
    instance / prompt -- straight from the vocab file's token strings.
    Never decodes the ~150k vocabulary entries one at a time.
    """
    with open(vocab_path, "r", encoding="utf-8") as vocab_file:
        vocab: dict[str, int] = json.load(vocab_file)

    blocked: set[int] = set()
    for token_str, token_id in vocab.items():
        cleaned = token_str.replace("\u0120", " ").replace("\u010a", "\n")
        if any(char in control_chars for char in cleaned):
            blocked.add(token_id)
    return frozenset(blocked)


class State(Enum):
    """Manage the FSM states."""

    START = auto()
    PROMPT = auto()
    NAME = auto()
    PARAMETERS = auto()
    END = auto()


class JSONenforce:
    """Enforce JSON parsing for model output."""

    DIGIT_CHAR = "0123456789"
    CONTROL_CHARS = "\n\r\t"
    SIGN_CHAR = "-"
    DECIMAL_CHAR = "."
    A_TERMINATOR = [",", "}"]
    MAX_STRING_TOKENS = 200
    MAX_NUMBER_TOKENS = 40

    _TYPE_ALIASES = {
        "string": "string", "str": "string", "text": "string",
        "integer": "integer", "int": "integer", "long": "integer",
        "number": "number", "float": "number", "double": "number",
        "boolean": "boolean", "bool": "boolean",
    }

    def __init__(self,
                 model: Small_LLM_Model,
                 prompt: str,
                 functions: list[FunctionDefinition]) -> None:
        """Init the variable that will be used."""
        self.model = model
        self.state = State.START
        self.prompt = prompt
        self.functions: list[FunctionDefinition] = functions
        self.res: list[int] = []
        self.input_ids: list[int] = []
        self.chosen_func: FunctionDefinition | None = None

        to_send = build_prompt(functions, prompt)
        self.input_ids = self.model.encode(to_send).tolist()[0]

        terminator_chars = "".join(self.A_TERMINATOR)
        self._integer_allowed_ids = self._char_ids(
            self.DIGIT_CHAR + self.SIGN_CHAR + terminator_chars
        )
        self._number_allowed_ids = self._char_ids(
            self.DIGIT_CHAR + self.SIGN_CHAR + self.DECIMAL_CHAR
            + terminator_chars
        )
        self._sign_ids = self._char_ids(self.SIGN_CHAR)
        self._decimal_ids = self._char_ids(self.DECIMAL_CHAR)
        self._control_char_ids = _load_control_char_ids(
            self.model.get_path_to_vocab_file(), self.CONTROL_CHARS
        )

    def _char_ids(self, chars: str) -> set[int]:
        ids: set[int] = set()
        for char in chars:
            ids.update(self.model.encode(char).tolist()[0])
        return ids

    def tokeniser(self, text: str) -> None:
        """Transform a peace of text int logits."""
        ids = self.model.encode(text).tolist()[0]
        self.input_ids.extend(ids)
        self.res.extend(ids)

    def _walk_fixed_choices(self, choices: list[str]) -> str:
        """Generate and return one of the provided fixed choices."""
        choice_ids = [self.model.encode(c + '"').tolist()[0] for c in choices]
        chosen: list[int] = []
        while True:
            candidates = [
                ids for ids in choice_ids
                if ids[:len(chosen)] == chosen
                ]

            if not candidates:
                raise ValueError(f"No valid choice among {choices}")
            if len(candidates) == 1:
                rest = candidates[0][len(chosen):-1]
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

    def _generate_number(self, param_name: str, allow_decimal: bool) -> None:
        """Generate a numeric literal.

        Refuses a second sign or decimal point once one has been seen.
        """
        base_allowed = (
            self._number_allowed_ids if allow_decimal
            else self._integer_allowed_ids
        )
        seen_decimal = False
        seen_digit = False
        iters = 0

        while True:
            iters += 1
            if iters > self.MAX_NUMBER_TOKENS:
                raise ValueError(
                    f"Number param '{param_name}' did not terminate")

            allowed_ids = set(base_allowed)
            if seen_digit:
                allowed_ids -= self._sign_ids
            if seen_decimal or not allow_decimal:
                allowed_ids -= self._decimal_ids

            logits = self.model.get_logits_from_input_ids(self.input_ids)
            masked_logits = filter_logits(logits, allowed_ids)
            next_token_id = masked_logits.index(max(masked_logits))
            decoded = self.model.decode([next_token_id])

            print(f"\r  [{param_name}] generating: \"{decoded}\"",
                  end="", flush=True)
            if decoded in self.A_TERMINATOR:
                break

            print()

            if next_token_id in self._decimal_ids:
                seen_decimal = True
            elif next_token_id not in self._sign_ids:
                seen_digit = True

            self.input_ids.append(next_token_id)
            self.res.append(next_token_id)

    def _generate_string(self, param_name: str) -> None:
        """Generate a JSON string value.

        Blocks raw control characters and escapes any backslash or
        quote the model produces as content.
        """
        self.tokeniser('"')

        ended = False
        for _ in range(self.MAX_STRING_TOKENS):
            logits = self.model.get_logits_from_input_ids(self.input_ids)
            masked_logits = filter_logits(
                logits,
                set(range(len(logits))) - self._control_char_ids,
            )
            next_token_id = masked_logits.index(max(masked_logits))
            token_text = self.model.decode([next_token_id])

            print(f"\r  [{param_name}] generating: \"{token_text}\"\x1b[K",
                  end="", flush=True)

            if '"' in token_text:
                # a bare quote always ends the value; flush whatever came
                # before it in this token first
                content, _, _ = token_text.partition('"')
                if content:
                    self._emit_escaped(content)
                ended = True
                break

            self._emit_escaped(token_text)

        print()

        if not ended:
            print(
                f"[warn] string param '{param_name}' "
                f"truncated at {self.MAX_STRING_TOKENS} tokens"
            )

        self.tokeniser('"')

    def _emit_escaped(self, text: str) -> None:
        """Write text to the output, escaping backslashes and quotes.

        Ensures they can never be misread as JSON control sequences.
        """
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        self.tokeniser(escaped)

    def output_modelisation(self) -> FunctionCallresult:
        """Force the output to be a valid JSON."""
        if self.state == State.START:
            self.tokeniser('{"prompt": "')
            self.state = State.PROMPT

        if self.state == State.PROMPT:
            escaped_prompt = json.dumps(self.prompt, ensure_ascii=False)[1:-1]
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
            if self.chosen_func is None:
                raise ValueError("No function has been chosen yet")
            param_items = list(self.chosen_func.parameters.items())

            for i, (param_name, param) in enumerate(param_items):
                key_text = f'"{param_name}":'
                self.tokeniser(key_text)

                resolved_type = param.type

                if resolved_type == "string":
                    self._generate_string(param_name)
                elif resolved_type == "integer":
                    self._generate_number(param_name, allow_decimal=False)
                elif resolved_type == "number":
                    self._generate_number(param_name, allow_decimal=True)
                elif resolved_type == "boolean":
                    self._walk_fixed_choices(["true", "false"])

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
