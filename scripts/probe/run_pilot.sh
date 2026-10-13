#!/usr/bin/env bash
set -euo pipefail
probe_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROBOTWIN_PYTHON:?set ROBOTWIN_PYTHON}" "${probe_script_dir}/pilot.py" "$@"
