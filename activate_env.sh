#!/usr/bin/env bash
# activation script for detection project using interrupt venv

source /home/eshwar/code/interrupt/venv/bin/activate
export PYTHONPATH="/home/eshwar/code/detection:${PYTHONPATH:-}"

echo "activated virtual environment: /home/eshwar/code/interrupt/venv"
echo "python interpreter: $(which python) ($(python --version))"

