"""Make the repo-root modules (topology, config, cell_library, ...) importable
from tests/ regardless of where pytest is launched."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tests", "golden")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
