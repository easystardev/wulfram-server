#!/usr/bin/env python3
"""Assert the rotation/attitude kernel still reproduces the frozen golden EXACTLY.

This is the determinism gate for the server.py decomposition (Track 1) and the
shared-kernel convergence (Track 2) in docs/architecture/shared-core-design.md. Any
change that perturbs VehiclePhysics' per-step result — including pointing the server
at a shared sim_kernel — must reproduce physics_parity_golden.json bit-for-bit.

Comparison is EXACT IEEE-754 hex (no tolerance): determinism is the contract.

Usage:  uv run python test_physics_parity.py   (or gen_physics_golden.py to (re)freeze)
"""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from wulfram.physics import VehiclePhysics  # noqa: E402

GOLDEN = Path(__file__).parent / "physics_parity_golden.json"


def _hex(v: float) -> str:
    return struct.pack(">d", float(v)).hex()


def _capture(p: "VehiclePhysics") -> dict:
    return {
        "euler": [_hex(x) for x in p._euler],
        "ang_vel": _hex(p._angular_velocity),
        "matrix": [_hex(x) for x in p._matrix],
    }


def _run_case(case: dict, damp: float) -> dict:
    p = VehiclePhysics(damp_coeff=damp)
    p._euler = list(case["init_euler"])
    p._angular_velocity = case["init_ang_vel"]
    torque, dt, steps, mode = case["torque"], case["dt"], case["steps"], case["mode"]
    for _ in range(steps):
        if mode == "step_f32":
            p.step_f32(torque, dt)
        else:
            p.step_client_substeps(torque, dt)
    return _capture(p)


def main() -> int:
    if not GOLDEN.exists():
        print(f"MISSING golden fixture {GOLDEN}; run: uv run python gen_physics_golden.py")
        return 1
    golden = json.loads(GOLDEN.read_text())
    damp = golden["damp_coeff"]
    passed = failed = 0
    first_failures = []
    for case in golden["cases"]:
        got = _run_case(case, damp)
        if got == case["out"]:
            passed += 1
        else:
            failed += 1
            if len(first_failures) < 5:
                first_failures.append({k: case[k] for k in
                                       ("mode", "init_euler", "init_ang_vel",
                                        "torque", "dt", "steps")})
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed (of {golden['case_count']})")
    print("=" * 60)
    if failed:
        print("First failing cases (kernel drifted from golden):")
        for f in first_failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
