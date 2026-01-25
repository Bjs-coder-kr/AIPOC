from __future__ import annotations

import os
import runpy
from pathlib import Path


def render() -> None:
    os.environ["DOCUMIND_UNIFIED_APP"] = "1"
    script_path = Path(__file__).resolve().parents[1] / "views" / "analy_app.py"
    runpy.run_path(str(script_path), run_name="__main__")
