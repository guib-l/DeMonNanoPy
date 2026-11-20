#!/usr/bin/env bash

here="$(cd "$(dirname "$0")" && pwd)"

failed=()
for script in "$here"/example_*.py; do
    echo "=== ${script#"$PWD"/}"
    python "$script" || failed+=("${script##*/}")
done

echo
echo "${#failed[@]} failure(s)"
[ "${#failed[@]}" -eq 0 ] || { printf '  %s\n' "${failed[@]}"; exit 1; }
