from __future__ import annotations

import importlib
import os

MODULES = {"alfworld": "alf_env", "webshop": "ws_env", "sciworld": "sw_env"}
NAME = os.environ.get("OALF_ENV", "alfworld")
if NAME not in MODULES:
    raise SystemExit(f"OALF_ENV={NAME!r}: expected one of {sorted(MODULES)}")
E = importlib.import_module(MODULES[NAME])
