"""Quick test: verify FEATURES import, current fields, env overrides, singleton."""
import os
import sys
from pathlib import Path

# Match the other test_*.py: make wulfram (server/) and wulfram2_protocol
# (../shared) importable when run standalone (uv run python test_features.py).
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

# Env override must be set BEFORE import (applied in Features.__post_init__).
os.environ["WULFRAM_AUTO_LOGIN"] = "1"

from wulfram.session import FEATURES, Features

# 1. Current Features fields all present with expected types.
EXPECTED_FIELDS = [
    "send_load_status",
    "send_behavior_packet",
    "send_translation_packet",
    "send_spawn_points",
    "send_player_on_login",
    "send_world_stats_on_login",
    "auto_login",
    "tick_loop_enabled",
    "send_update_array_empty",
    "wulfforge_compat",
]
for name in EXPECTED_FIELDS:
    assert hasattr(FEATURES, name), f"Features missing field: {name}"
    assert isinstance(getattr(FEATURES, name), bool), f"Features.{name} is not bool"
print(f"fields ok: {len(EXPECTED_FIELDS)}")

# auto_join_team was removed (auto-spawn could crash the client) -- stay removed.
assert not hasattr(FEATURES, "auto_join_team"), "auto_join_team should stay removed"
print("auto_join_team removed: ok")

# 2. Env override applied at construction (WULFRAM_AUTO_LOGIN=1 set above).
assert FEATURES.auto_login is True, "WULFRAM_AUTO_LOGIN=1 not applied"
os.environ["WULFRAM_AUTO_LOGIN"] = "0"
assert Features().auto_login is False, "WULFRAM_AUTO_LOGIN=0 not applied"
print("env overrides: ok")

# 3. Handlers share the same singleton instance.
from wulfram.handlers import FEATURES as HF
assert FEATURES is HF, "handlers FEATURES is not the session singleton"
print("singleton shared with handlers: ok")

print("OK test_features")
