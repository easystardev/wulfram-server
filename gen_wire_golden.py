#!/usr/bin/env python3
"""Generate the wire/session golden — the byte-exact freeze of the whole
replication-encode contract (codec + quantizers + replication echo + the
TRANSIENT_ARRAY fidelity-debt packet), mirroring gen_physics_golden /
gen_collision_golden (GOAL CH4, docs/precise-clone-goal-loop.md).

Deterministic by construction: every packet builder takes its `tick` and state
as explicit arguments (no wall-clock), so the frozen bytes are reproducible. The
inputs are seeded from the CH2 OG corpus (`ch2-og-corpus.pose.ndjson`) so the
golden replays the REAL captured session's poses/FX, recorded under the
deterministic client-tick regime (WULFRAM_USE_CLIENT_TICKS) the live replay uses.

Each case freezes the exact packet bytes (hex). test_wire_parity.py rebuilds and
compares byte-for-byte. A change to the codec, a quantizer width, or a
replication field shifts the bytes and fails — determinism is the contract.

Usage:  uv run python gen_wire_golden.py        (re)freeze
        uv run python test_wire_parity.py       exact replay
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from wulfram.packets import (  # noqa: E402
    build_update_array_player_update,
    build_update_array_heartbeat,
    build_transient_array,
    decode_transient_array,
    FX_PULSE_FIRE, FX_IMPACT_VEHICLE, FX_CHAIN_GUN_FIRE,
)

GOLDEN = Path(__file__).parent / "testdata" / "wire_parity_golden.json"
POSE_CORPUS = Path(__file__).parent / "ch2-og-corpus.pose.ndjson"

# Deterministic base tick (client-tick regime: ticks are explicit inputs).
BASE_TICK = 711_000
TICK_STEP = 84  # WARP frame cadence in ms (matches the OG corpus)
ENTITY_ID = 1337


def _load_poses(limit: int = 24):
    """Seed inputs from the real CH2 OG poses (pos + euler). Deterministic."""
    poses = []
    if POSE_CORPUS.exists():
        for line in POSE_CORPUS.open(encoding="ascii"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("read_ok") and r.get("pos") and r.get("euler"):
                poses.append((r["regime"], r["phase"], r["pos"], r["euler"]))
            if len(poses) >= limit:
                break
    if not poses:
        # Fallback synthetic poses so the golden is generatable without the corpus.
        poses = [("synthetic", "fixed",
                  [5050.0 + i, 4950.0 - i, 3.5], [0.0, 0.0, i * 0.1])
                 for i in range(8)]
    return poses


def build_cases() -> list:
    cases = []
    poses = _load_poses()

    # 1) UPDATE_ARRAY player-update replication of each captured OG pose (the
    #    canonical gameplay replication stream: pos + rot + local_state HUD).
    for i, (regime, phase, pos, euler) in enumerate(poses):
        tick = BASE_TICK + i * TICK_STEP
        pkt = build_update_array_player_update(
            tick, ENTITY_ID,
            pos=(pos[0], pos[1], pos[2]),
            vel=(0.0, 0.0, 0.0),
            rot=(euler[0], euler[1], euler[2]),
            include_vel=True, include_rot=True, include_local_state=True,
            health=1.0, fuel=1.0,
        )
        cases.append({"kind": "update_array_player_update",
                      "label": f"{regime}/{phase}#{i}", "tick": tick,
                      "hex": pkt.hex()})

    # 2) UPDATE_ARRAY heartbeat (HUD/health, zero-entity GOAL-7 form).
    for i in range(4):
        tick = BASE_TICK + i * TICK_STEP
        pkt = build_update_array_heartbeat(tick, ENTITY_ID, include_health=True)
        cases.append({"kind": "update_array_heartbeat", "label": f"beat#{i}",
                      "tick": tick, "hex": pkt.hex()})

    # 3) TRANSIENT_ARRAY FX (the CH4 fidelity-debt packet) — table-sourced widths.
    fx_sets = [
        [{"type": FX_PULSE_FIRE, "pos": (5050.0, 4950.0, 3.5), "entity_id": ENTITY_ID}],
        [{"type": FX_CHAIN_GUN_FIRE, "pos": None, "entity_id": ENTITY_ID}],
        [{"type": FX_IMPACT_VEHICLE, "pos": (5223.16, 4916.65, 6.75), "entity_id": 2002},
         {"type": FX_PULSE_FIRE, "pos": (5201.4, 4911.4, 3.5), "entity_id": ENTITY_ID}],
    ]
    for i, events in enumerate(fx_sets):
        pkt = build_transient_array(events)
        # round-trip: decode must recover the same field structure (CH4 exit 2)
        dec = decode_transient_array(pkt)
        rt = [{"type": e["type"], "has_pos": e["pos"] is not None,
               "entity_id": e["entity_id"]} for e in dec]
        cases.append({"kind": "transient_array", "label": f"fx#{i}",
                      "hex": pkt.hex(), "roundtrip": rt})

    return cases


def main() -> int:
    cases = build_cases()
    payload = {
        "contract": "codec + quantizers + replication echo + TRANSIENT_ARRAY",
        "encoding": "packet bytes as hex (exact)",
        "tick_regime": "WULFRAM_USE_CLIENT_TICKS (ticks are explicit builder args)",
        "seeded_from": POSE_CORPUS.name,
        "transient_legacy": os.environ.get("WULFRAM_TRANSIENT_LEGACY", "0"),
        "case_count": len(cases),
        "cases": cases,
    }
    GOLDEN.write_text(json.dumps(payload, indent=1))
    print(f"wrote {GOLDEN} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
