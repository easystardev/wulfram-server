"""Quick test: verify FEATURES import and auto_join_team state"""
import os
import sys
from pathlib import Path

# Match the other test_*.py: make wulfram (server/) and wulfram2_protocol
# (../shared) importable when run standalone (uv run python test_features.py).
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

os.environ["WULFRAM_AUTO_JOIN_TEAM"] = "1"

from wulfram.session import FEATURES
print("before:", FEATURES.auto_join_team)

FEATURES.auto_join_team = True
print("after set:", FEATURES.auto_join_team)

from wulfram.handlers import FEATURES as HF
print("handlers:", HF.auto_join_team, "same:", FEATURES is HF)
print("ids:", id(FEATURES), id(HF))
