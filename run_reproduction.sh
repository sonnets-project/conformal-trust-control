#!/usr/bin/env bash
set -euo pipefail
CTC_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$CTC_ROOT"
python3 scripts/reproduce_paper.py "$@"
