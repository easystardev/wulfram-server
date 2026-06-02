#!/usr/bin/env python3
"""Generate the golden physics-kernel fixture from the CURRENT reference rotation
pipeline in wulfram/physics.py (VehiclePhysics). Run BEFORE any kernel refactor /
shared-kernel convergence; the committed golden is then the exact determinism
reference for test_physics_parity.py.

This is the analog of gen_collision_golden.py for the rotation/attitude kernel that
is currently DUPLICATED in server/physics.py and client/.../physics.py. A shared
kernel (Track 2 in docs/architecture/shared-core-design.md) must reproduce every
record here bit-for-bit.

The fixture is a deterministic cartesian grid (no randomness) over the physics
regimes; state is recorded as IEEE-754 double hex for EXACT comparison (determinism
is the contract, so no tolerance).

Usage:  uv run python gen_physics_golden.py
"""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from wulfram.physics import VehiclePhysics  # noqa: E402

GOLDEN = Path(__file__).parent / "physics_parity_golden.json"

# Tank damping from decompile (entity->0xbc->+4->+0x7c).
DAMP_COEFF = 2.0

# Representative regimes. Kept explicit + bounded for reproducibility, mirroring
# gen_collision_golden.py's grid style.
INIT_HEADINGS = [0.0, 0.5235987755982988, 0.7853981633974483, 1.5707963267948966,
                 3.141592653589793, 4.71238898038469, 6.0]          # incl. near-2pi
INIT_PITCH_ROLL = [(0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (1.4, 0.2)]  # last = steep pose
INIT_ANG_VEL = [0.0, 0.5, -0.5, 2.0]
# torque = turn_adjust(4.5) * quantized input. Right arrow=0.6409, left=0.580,
# thrust=0.855; plus zero and a large value.
TORQUES = [0.0, 4.5 * 0.6409, -4.5 * 0.580, 4.5 * 0.855, 30.0]
DTS = [0.033, 0.040, 0.084, 0.110]   # tick, inner substep, WARP frame, outer-max
STEP_COUNTS = [1, 5, 30]
MODES = ["step_f32", "substeps"]


def _hex(v: float) -> str:
    """IEEE-754 double bit pattern, big-endian hex — exact, JSON-safe."""
    return struct.pack(">d", float(v)).hex()


def _capture(p: "VehiclePhysics") -> dict:
    return {
        "euler": [_hex(x) for x in p._euler],
        "ang_vel": _hex(p._angular_velocity),
        "matrix": [_hex(x) for x in p._matrix],
    }


def _run_case(case: dict) -> dict:
    p = VehiclePhysics(damp_coeff=DAMP_COEFF)
    p._euler = list(case["init_euler"])
    p._angular_velocity = case["init_ang_vel"]
    torque, dt, steps, mode = case["torque"], case["dt"], case["steps"], case["mode"]
    for _ in range(steps):
        if mode == "step_f32":
            p.step_f32(torque, dt)
        else:
            p.step_client_substeps(torque, dt, use_f32=True)
    return _capture(p)


def fixture_cases() -> list:
    cases = []
    for mode in MODES:
        for hz in INIT_HEADINGS:
            for (px, ry) in INIT_PITCH_ROLL:
                for av in INIT_ANG_VEL:
                    for tq in TORQUES:
                        for dt in DTS:
                            for n in STEP_COUNTS:
                                cases.append({
                                    "mode": mode,
                                    "init_euler": [px, ry, hz],
                                    "init_ang_vel": av,
                                    "torque": tq,
                                    "dt": dt,
                                    "steps": n,
                                })
    return cases


def main() -> int:
    cases = fixture_cases()
    records = []
    for c in cases:
        records.append({**c, "out": _run_case(c)})
    payload = {
        "kernel": "VehiclePhysics rotation pipeline (server/wulfram/physics.py)",
        "damp_coeff": DAMP_COEFF,
        "encoding": "IEEE-754 double, big-endian hex (exact)",
        "case_count": len(records),
        "cases": records,
    }
    GOLDEN.write_text(json.dumps(payload, indent=1))
    print(f"wrote {GOLDEN} ({len(records)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
