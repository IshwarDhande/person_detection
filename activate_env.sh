#!/usr/bin/env bash
# Activation script for Vision-Based First Contact

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

if [ -d "${SCRIPT_DIR}/.venv" ]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
elif [ -d "${SCRIPT_DIR}/venv" ]; then
    source "${SCRIPT_DIR}/venv/bin/activate"
elif [ -f "/home/eshwar/code/interrupt/venv/bin/activate" ]; then
    source "/home/eshwar/code/interrupt/venv/bin/activate"
fi

echo "Activated environment. Python: $(which python3) ($(python3 --version))"
echo "PYTHONPATH set to: ${PYTHONPATH}"
