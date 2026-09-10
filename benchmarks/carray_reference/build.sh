#!/usr/bin/env bash
# Build the reference C-API array extension next to this script.
# It exists only as a lower bound for binding overhead (see benchmarks/bench_overhead.py).
set -euo pipefail
PY="$("${PYTHON:-python}" -c "import sys; print(sys.executable)")"
cd "$(dirname "$0")"
INC=$("$PY" -c "import sysconfig; print(sysconfig.get_path('include'))")
EXT=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
gcc -O3 -shared -fPIC -I"$INC" carray.c -o "carray$EXT"
echo "built carray$EXT"
