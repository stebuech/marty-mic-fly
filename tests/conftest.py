"""Pytest config for martymicfly tests."""

import sys
from pathlib import Path

# Ensure src/ layout is importable when running pytest directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
