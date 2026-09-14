"""Crash tests for the loader / CLI layer -- no LLM required.

Drop this file in tests/ at the repo root and run:
    uv run pytest tests/test_crash_loader.py -v

Every case here asserts the program degrades gracefully (clean exit(1)
with a message, or a skip-and-continue) instead of raising an
uncaught traceback. None of these need Small_LLM_Model, so they run
in well under a second and can be part of your normal dev loop.
"""
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from src.loader import function_loader, prompt_loader


VALID_FN: dict[str, Any] = {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"},
}


def _write(
    tmp_path: Path,
    name: str,
    content: str | dict[str, Any] | list[Any],
) -> str:
    path = tmp_path / name
    if isinstance(content, (dict, list)):
        path.write_text(json.dumps(content))
    else:
        path.write_text(content)
    return str(path)


# ---------------------------------------------------------------- files ---

def test_missing_functions_file_exits_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit cleanly when the functions file is missing."""
    with pytest.raises(SystemExit) as exc:
        function_loader(str(tmp_path / "nope.json"))
    assert exc.value.code == 1
    assert "function error" in capsys.readouterr().out


def test_missing_prompts_file_exits_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit cleanly when the prompts file is missing."""
    with pytest.raises(SystemExit) as exc:
        prompt_loader(str(tmp_path / "nope.json"))
    assert exc.value.code == 1
    assert "prompt error" in capsys.readouterr().out


def test_malformed_json_functions(tmp_path: Path) -> None:
    """Exit cleanly when the functions file contains malformed JSON."""
    path = _write(tmp_path, "fd.json", '[{"name": "x",}]')
    with pytest.raises(SystemExit):
        function_loader(path)


def test_malformed_json_prompts(tmp_path: Path) -> None:
    """Exit cleanly when the prompts file contains malformed JSON."""
    path = _write(tmp_path, "p.json", '[{"prompt": "hi"},]')
    with pytest.raises(SystemExit):
        prompt_loader(path)


def test_zero_byte_file(tmp_path: Path) -> None:
    """Exit cleanly when the functions file is empty."""
    path = _write(tmp_path, "fd.json", "")
    with pytest.raises(SystemExit):
        function_loader(path)


def test_whitespace_only_file(tmp_path: Path) -> None:
    """Exit cleanly when the prompts file contains only whitespace."""
    path = _write(tmp_path, "p.json", "   \n\t  ")
    with pytest.raises(SystemExit):
        prompt_loader(path)


# ------------------------------------------------------ wrong top-level ---

def test_functions_file_is_object_not_array(tmp_path: Path) -> None:
    """Exit cleanly when the functions file contains an object."""
    path = _write(tmp_path, "fd.json", VALID_FN)
    with pytest.raises(SystemExit):
        function_loader(path)


def test_prompts_file_is_object_not_array_does_not_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dict at the top level makes `enumerate(raw)` walk its keys, not.

    its values -- confirm this degrades to zero prompts instead of a
    KeyError/AttributeError.
    """
    path = _write(tmp_path, "p.json", {"prompt": "hi"})
    result = prompt_loader(path)
    assert result == []
    assert "no valid prompts found" in capsys.readouterr().out


# ------------------------------------------------------------- content ---

def test_empty_function_list_exits(tmp_path: Path) -> None:
    """Exit cleanly when the functions file contains an empty list."""
    path = _write(tmp_path, "fd.json", [])
    with pytest.raises(SystemExit):
        function_loader(path)


def test_empty_prompt_list_does_not_exit(tmp_path: Path) -> None:
    """Return no prompts when the prompts file contains an empty list."""
    path = _write(tmp_path, "p.json", [])
    result = prompt_loader(path)
    assert result == []


def test_zero_parameter_function_is_rejected(tmp_path: Path) -> None:
    """Reject functions with no parameters.

    FunctionDefinition.parameters has Field(min_length=1), so a
    genuinely no-arg function is silently dropped rather than crashing.
    Documents a real limitation -- if a hidden test set includes a
    zero-arg function, it will never be selectable.
    """
    fn = {**VALID_FN, "parameters": {}}
    path = _write(tmp_path, "fd.json", [fn])
    with pytest.raises(SystemExit):
        function_loader(path)


def test_function_missing_required_key_is_skipped(tmp_path: Path) -> None:
    """Skip functions that are missing a required key."""
    fn = {k: v for k, v in VALID_FN.items() if k != "returns"}
    other = VALID_FN
    path = _write(tmp_path, "fd.json", [fn, other])
    result = function_loader(path)
    assert len(result) == 1
    assert result[0].name == "fn_add_numbers"


def test_function_unknown_extra_key_is_skipped(tmp_path: Path) -> None:
    """Skip functions that contain unknown extra keys."""
    fn = {**VALID_FN, "deprecated": True}
    path = _write(tmp_path, "fd.json", [fn, VALID_FN])
    result = function_loader(path)
    assert len(result) == 1


def test_function_unsupported_param_type_is_skipped(tmp_path: Path) -> None:
    """Skip functions with unsupported parameter types."""
    fn = {**VALID_FN, "parameters": {"items": {"type": "array"}}}
    path = _write(tmp_path, "fd.json", [fn, VALID_FN])
    result = function_loader(path)
    assert len(result) == 1
    assert result[0].name == "fn_add_numbers"


def test_duplicate_function_names_both_load_but_first_wins(
    tmp_path: Path,
) -> None:
    """Load duplicate function names, with the first definition winning.

    This is not a crash, but a real correctness gap: two definitions with the
    same name both load successfully, and FSM.output_modelisation()
    resolves the name via `function_names.index(chosen_name)`, which
    always returns the FIRST match. The second definition becomes
    permanently unreachable.
    """
    two_param = VALID_FN
    three_param = {
        **VALID_FN,
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"},
            "c": {"type": "number"},
        },
    }
    path = _write(tmp_path, "fd.json", [two_param, three_param])
    result = function_loader(path)
    assert len(result) == 2  # both accepted -- no dedup / rejection today
    names = [f.name for f in result]
    first_match = result[names.index("fn_add_numbers")]
    assert set(first_match.parameters) == {"a", "b"}


def test_blank_prompt_is_skipped(tmp_path: Path) -> None:
    """Skip prompts containing only whitespace."""
    path = _write(tmp_path, "p.json", [{"prompt": "   "}])
    assert prompt_loader(path) == []


def test_empty_string_prompt_is_skipped(tmp_path: Path) -> None:
    """Skip empty prompts."""
    path = _write(tmp_path, "p.json", [{"prompt": ""}])
    assert prompt_loader(path) == []


def test_non_string_prompt_is_skipped(tmp_path: Path) -> None:
    """Skip prompts whose values are not strings."""
    path = _write(tmp_path, "p.json", [{"prompt": 42}])
    assert prompt_loader(path) == []


def test_null_prompt_is_skipped(tmp_path: Path) -> None:
    """Skip prompts with null values."""
    path = _write(tmp_path, "p.json", [{"prompt": None}])
    assert prompt_loader(path) == []


def test_prompt_unknown_extra_key_is_skipped(tmp_path: Path) -> None:
    """Skip prompts containing unknown keys."""
    path = _write(tmp_path, "p.json", [{"prompt": "hi", "priority": "high"}])
    assert prompt_loader(path) == []


def test_mixed_valid_and_invalid_prompts_keeps_the_valid_ones(
    tmp_path: Path,
) -> None:
    """Keep valid prompts while skipping invalid entries."""
    entries = [
        {"prompt": "Greet shrek"},
        {"prompt": "   "},
        {"prompt": 42},
        {"prompt": "Reverse the string 'hello'"},
        {"not_prompt": "oops"},
    ]
    path = _write(tmp_path, "p.json", entries)
    result = prompt_loader(path)
    assert [p.prompt for p in result] == [
        "Greet shrek",
        "Reverse the string 'hello'",
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
