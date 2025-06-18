#!/bin/bash
set -e

python3 -m venv .venv
echo "Virtual environment created at .venv"

source .venv/bin/activate
pip install --upgrade pip

if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

echo "Python virtual environment is ready. Activate it with 'source .venv/bin/activate'."
