*This project has been created as part of the 42 curriculum by belaindr.*

# Call Me Maybe

## Description

`call me maybe` translates natural language prompts into structured, schema-compliant
function calls, using a small local LLM (Qwen/Qwen3-0.6B by default). Instead of hoping
the model spontaneously emits valid JSON, the program drives generation with **constrained
decoding**: at every generation step, the raw logits returned by the model are masked so
that only tokens compatible with valid JSON *and* the target function schema can be
selected. This guarantees 100% parseable, schema-correct output, even from a 500M
parameter model that would otherwise produce malformed JSON a large fraction of the time.

Given a set of function definitions and a list of natural language prompts, the program
outputs, for each prompt, the name of the function to call and its arguments with the
correct types — never a direct natural-language answer.

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk` package, placed alongside `src/` (see Submission requirements)

### Installation

```bash
make install
```

This resolves and installs dependencies (numpy, pydantic, etc.) via `uv sync`.

### Running

```bash
uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
```

By default, input is read from `data/input/` and output is written to
`data/output/function_calling_results.json`. Example:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Makefile targets

| Target | Description |
|---|---|
| `install` | Install dependencies via `uv` |
| `run` | Run the main program |
| `debug` | Run the program under `pdb` |
| `lint` | Run `flake8` and `mypy` (project-mandated flags) |
| `lint-strict` | Run `flake8` and `mypy --strict` |
| `clean` | Remove `__pycache__`, `.mypy_cache`, etc. |

## Algorithm Explanation

The core problem: given a prompt like *"What is the sum of 40 and 2?"*, produce
`{"name": "fn_add_numbers", "parameters": {"a": 40, "b": 2}}` — and guarantee that the
output is *always* valid JSON that matches the function's schema exactly.

1. **Prompt assembly** — for each natural-language prompt, a text prompt is built that
   includes the available function definitions (name, description, parameter types) and
   the user's request.
2. **Token-by-token generation** — the prompt is tokenized via `llm_sdk`'s `encode`, and
   generation proceeds one token at a time using `get_logits_from_input_ids`.
3. **Constrained decoding via a finite state machine** — a JSON/schema-aware FSM tracks
   what characters are syntactically and semantically valid at the current position
   (e.g. "we're inside a string value for parameter `a`, which must be a number").
   At each step:
   - the FSM reports the set of allowed next characters,
   - that set is mapped to the subset of vocabulary token IDs that would keep the
     output consistent with it (using the vocabulary file returned by
     `get_path_to_vocab_file`),
   - logits for every other token are set to `-inf`,
   - the highest-scoring token among the remaining candidates is selected.
4. **Termination** — generation stops once the FSM reaches its `DONE` state (a complete,
   valid function-call object), and the result is decoded and validated with pydantic
   before being appended to the output list.

This means the model is *only* ever allowed to pick tokens that keep the output valid —
it cannot produce malformed JSON or violate the schema, regardless of how confident it is
in an invalid token.

## Design Decisions

- **Pydantic for error handlings**: function definitions, prompt entries, and function-call
  results are all pydantic models, so structural validation is enforced at load time and
  at output time, not just during generation.
- **Per-prompt fault isolation**: if an individual entry in `function_calling_tests.json`
  is invalid, it is skipped rather than aborting the whole run. If every prompt turns out
  invalid, the program still completes and writes an empty results array instead of
  crashing.
- **No private `llm_sdk` access**: the program only uses the public methods
  (`encode`, `decode`, `get_logits_from_input_ids`) — no reliance on internal model or tokenizer internals.


## Performance Analysis

- **Validity**: 100% of outputs are valid schema-compliant JSON by construction — the
  FSM makes invalid tokens unreachable, so there is nothing to "get wrong" structurally.
- **Accuracy**: function selection and argument extraction depend on the underlying
  model's understanding of the prompt; correctness here is measured separately from
  JSON validity (see Testing Strategy).
- **Speed**: generation is one forward pass per output token, dominated by the model's
  per-token latency; typical runs over the provided test prompts complete well within
  the 5-minute budget on standard hardware.

## Challenges Faced

- Mapping FSM-allowed *characters* to a set of allowed *tokens* in order to enhance the accuracy of the result yet, it became more complicated to identify **strings** values and **boleans** in the FunctionCallResults.
- Numeric literals required extra care in the FSM (distinguishing integers, floats,
  leading/trailing digit rules) to avoid producing syntactically valid but
  semantically wrong values (e.g. `"a": 4.` with a trailing dot).

## Testing Strategy

- Unit tests (via `pytest`) cover the JSON/schema FSM in isolation: valid transitions,
  rejected characters, and full accept/reject traces for representative function
  signatures (numbers, strings, nested objects).
- End-to-end runs against the provided `function_calling_tests.json` /
  `functions_definition.json`, followed by manual inspection of
  `function_calling_results.json` for both JSON validity and semantic correctness
  (right function, right arguments, right types).
- Edge cases specifically exercised: empty/malformed input files, missing files,
  ambiguous prompts, large numbers, and functions with multiple parameters.

## Example Usage

```bash
$ uv run python -m src
Loaded 5 function definitions.
Loaded 11 prompts (0 skipped as invalid).
Processing prompts... done in 12.4s.
Wrote 11 results to data/output/function_calling_results.json
```

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Resources

- [OpenAI: Function calling](https://platform.openai.com/docs/guides/function-calling) —
  general overview of the function-calling paradigm.
- [Guidance / constrained decoding literature] — background on logit masking and
  grammar-constrained generation for structured LLM output.
- Hugging Face tokenizers documentation — BPE tokenization internals, useful for
  understanding vocabulary/token-to-character mapping.
- **AI usage**: an AI assistant was used to help think through the token-to-character
  mapping strategy for the FSM. the FSM and generation loop implementation itself was written and
  understood by the author, per the subject's AI usage guidelines.