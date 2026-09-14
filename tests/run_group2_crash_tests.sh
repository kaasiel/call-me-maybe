#!/usr/bin/env bash
# Runs every Group 2 (LLM-dependent) crash-test pair through the real
# program and reports pass/fail based on exit code + valid JSON output.
# Needs the real Qwen3-0.6B model, so run this yourself:
#
#   chmod +x run_group2_crash_tests.sh
#   ./run_group2_crash_tests.sh
#
set -u
CRASH_DIR="tests/data/input/crash"
OUT_DIR="tests/data/output/crash_results"
mkdir -p "$OUT_DIR"

pass=0
fail=0

for fd in "$CRASH_DIR"/fd_*.json; do
    n=$(basename "$fd" | sed -E 's/^fd_([0-9]+)_.*/\1/')
    tests=$(ls "$CRASH_DIR"/tests_${n}_*.json 2>/dev/null | head -n1)
    label=$(basename "$fd" .json | sed -E 's/^fd_[0-9]+_//')
    out="$OUT_DIR/${n}_${label}.json"

    echo "=== [$n] $label ==="
    python3 -m src \
      --functions_definition "$fd" \
      --input "$tests" \
      --output "$out"
    code=$?

    if [ $code -ne 0 ]; then
        echo "  -> FAIL: process exited with code $code (should always be 0)"
        fail=$((fail+1))
    elif ! python3 -c "import json,sys; json.load(open('$out'))" 2>/dev/null; then
        echo "  -> FAIL: $out is not valid JSON"
        fail=$((fail+1))
    else
        echo "  -> OK, wrote $(python3 -c "import json; print(len(json.load(open('$out'))))" 2>/dev/null) result(s) to $out"
        pass=$((pass+1))
    fi
    echo
done

echo "================================================================"
echo "Group 2: $pass passed, $fail failed"
[ $fail -eq 0 ]
