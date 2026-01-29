from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_path() -> None:
    # Make the D-FINE root importable for `src.*` imports.
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_project_path()

# Normalise TTNN runtime behaviour globally: disable fallback and fast runtime mode.
import ttnn

ttnn.CONFIG.throw_exception_on_fallback = True
ttnn.CONFIG.enable_fast_runtime_mode = False
ttnn.CONFIG.enable_model_cache = True
