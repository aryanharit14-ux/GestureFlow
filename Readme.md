# GestureFlow

Hands-free control of media, presentations and desktop tasks on macOS and Windows,
using hand gestures and local voice dictation. Everything runs on your machine.

## Status
Early development. See `docs/decisions.md` for design choices.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mac]"
pytest
```
