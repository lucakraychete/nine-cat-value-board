#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, and a local .env from the template.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

if [ ! -d .venv ]; then
    echo "Creating virtualenv in .venv"
    "$PYTHON" -m venv .venv
fi

# venv layout differs: bin/ on macOS/Linux, Scripts/ on Windows (Git Bash)
if [ -f .venv/bin/python ]; then
    VENV_PY=.venv/bin/python
    ACTIVATE=.venv/bin/activate
else
    VENV_PY=.venv/Scripts/python
    ACTIVATE=.venv/Scripts/activate
fi

"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example. Fill in your league details."
else
    echo ".env already exists, leaving it alone."
fi

echo
echo "Done. Next:"
echo "  source $ACTIVATE"
echo "  python -m ninecat.ingest     # pull and cache league data"
echo "  streamlit run app.py"
