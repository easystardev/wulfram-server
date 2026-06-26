#!/usr/bin/env python3
"""Assert the wire-encode contract still reproduces the frozen wire golden EXACTLY.

The replication-encode analog of test_physics_parity / test_collision_parity
(GOAL CH4). Rebuilds every packet from wire_parity_golden.json and compares the
bytes byte-for-byte; also re-checks the TRANSIENT_ARRAY round-trip (decode of a
built packet recovers the same field structure through the quantizer table).

Comparison is EXACT (hex): determinism is the contract. Any codec / quantizer
width / replication-field change shifts the bytes and fails here.

Usage:  uv run python test_wire_parity.py   (gen_wire_golden.py to (re)freeze)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from wulfram.packets import (  # noqa: E402
    build_update_array_player_update,
    build_update_array_heartbeat,
    build_transient_array,
    decode_transient_array,
)

GOLDEN = Path(__file__).parent / "testdata" / "wire_parity_golden.json"
POSE_CORPUS = Path(__file__).parent / "ch2-og-corpus.pose.ndjson"
BASE_TICK = 711_000
TICK_STEP = 84
ENTITY_ID = 1337


def _load_poses(limit: int = 24):
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
        poses = [("synthetic", "fixed",
                  [5050.0 + i, 4950.0 - i, 3.5], [0.0, 0.0, i * 0.1])
                 for i in range(8)]
    return poses


def _rebuild(case, pose_iter):
    kind = case["kind"]
    if kind == "update_array_player_update":
        regime, phase, pos, euler = next(pose_iter)
        return build_update_array_player_update(
            case["tick"], ENTITY_ID,
            pos=(pos[0], pos[1], pos[2]), vel=(0.0, 0.0, 0.0),
            rot=(euler[0], euler[1], euler[2]),
            include_vel=True, include_rot=True, include_local_state=True,
            health=1.0, fuel=1.0,
        ).hex()
    if kind == "update_array_heartbeat":
        return build_update_array_heartbeat(case["tick"], ENTITY_ID,
                                            include_health=True).hex()
    if kind == "transient_array":
        return build_transient_array_from_label(case["label"]).hex()
    raise ValueError(f"unknown case kind {kind}")


# FX sets must match gen_wire_golden.build_cases exactly.
from wulfram.packets import (  # noqa: E402
    FX_PULSE_FIRE, FX_IMPACT_VEHICLE, FX_CHAIN_GUN_FIRE,
)
_FX_SETS = {
    "fx#0": [{"type": FX_PULSE_FIRE, "pos": (5050.0, 4950.0, 3.5), "entity_id": ENTITY_ID}],
    "fx#1": [{"type": FX_CHAIN_GUN_FIRE, "pos": None, "entity_id": ENTITY_ID}],
    "fx#2": [{"type": FX_IMPACT_VEHICLE, "pos": (5223.16, 4916.65, 6.75), "entity_id": 2002},
             {"type": FX_PULSE_FIRE, "pos": (5201.4, 4911.4, 3.5), "entity_id": ENTITY_ID}],
}


def build_transient_array_from_label(label):
    return build_transient_array(_FX_SETS[label])


def main() -> int:
    if not GOLDEN.exists():
        print(f"MISSING {GOLDEN}; run: uv run python gen_wire_golden.py")
        return 1
    golden = json.loads(GOLDEN.read_text())
    poses = iter(_load_poses())
    passed = failed = 0
    rt_passed = rt_failed = 0
    first_fail = None
    for case in golden["cases"]:
        got = _rebuild(case, poses)
        if got == case["hex"]:
            passed += 1
        else:
            failed += 1
            if first_fail is None:
                first_fail = {"kind": case["kind"], "label": case.get("label"),
                              "want": case["hex"][:48], "got": got[:48]}
        # TRANSIENT round-trip re-check (CH4 exit 2)
        if case["kind"] == "transient_array":
            dec = decode_transient_array(bytes.fromhex(got))
            rt = [{"type": e["type"], "has_pos": e["pos"] is not None,
                   "entity_id": e["entity_id"]} for e in dec]
            if rt == case["roundtrip"]:
                rt_passed += 1
            else:
                rt_failed += 1

    print("=" * 60)
    print(f"Wire parity: {passed} passed, {failed} failed (of {golden['case_count']})")
    print(f"TRANSIENT round-trip: {rt_passed} passed, {rt_failed} failed")
    print("=" * 60)
    if failed or rt_failed:
        if first_fail:
            print("First byte mismatch:")
            print(f"  {first_fail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
