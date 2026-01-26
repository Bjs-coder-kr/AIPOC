from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

runpy.run_path(str(Path(__file__).resolve().parent / "views" / "analy_app.py"), run_name="__main__")
