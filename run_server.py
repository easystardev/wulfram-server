#!/usr/bin/env python3
"""
Run the Wulfram2 server emulator.
"""

import os
import sys
import faulthandler
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

_FAULTHANDLER_LOG = None

if os.environ.get("WULFRAM_FAULTHANDLER", "1") == "1":
    try:
        log_path = os.environ.get("WULFRAM_FAULTHANDLER_LOG", "").strip()
        if log_path:
            _FAULTHANDLER_LOG = open(log_path, "a", encoding="utf-8", buffering=1)
            faulthandler.enable(file=_FAULTHANDLER_LOG)
        else:
            faulthandler.enable()

        timeout_raw = os.environ.get("WULFRAM_FAULTHANDLER_TIMEOUT", "").strip()
        if timeout_raw:
            timeout = float(timeout_raw)
            faulthandler.dump_traceback_later(timeout, repeat=True, file=_FAULTHANDLER_LOG)
    except Exception as exc:
        print(f"[WARN] Failed to enable faulthandler: {exc}")

from wulfram.server import main

if __name__ == "__main__":
    main()
