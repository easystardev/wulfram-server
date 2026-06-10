"""Golden-trace capture: harvest a real OG-client input corpus for the physics
kernel fixture, WITHOUT editing the kernel or server.py.

Enable by setting WULFRAM_GOLDEN_CAPTURE=<path.ndjson>. `install()` (called from
run_server.py when that env is set) monkeypatches VehiclePhysics' step methods to
append one IEEE-hex record per kernel step. Disabled = zero footprint (install()
is never called, and the kernel is untouched).

The captured NDJSON is the real (torque, dt, pre, post) corpus a representative
fixture is generated from — see docs/architecture/shared-core-design.md section 7.
The kernel GOLDEN itself (physics_parity_golden.json) is synthetic and needs none
of this; this only makes the fixture representative of authentic OG input cadence.
"""
import json
import os
import struct
import threading

_PATH = os.environ.get("WULFRAM_GOLDEN_CAPTURE", "").strip()
_lock = threading.Lock()
_fh = None
_installed = False


def enabled() -> bool:
    return bool(_PATH)


def _h(v: float) -> str:
    return struct.pack(">d", float(v)).hex()


def _emit(rec: dict) -> None:
    global _fh
    if not _PATH:
        return
    with _lock:
        if _fh is None:
            _fh = open(_PATH, "a", buffering=1, encoding="ascii")
        _fh.write(json.dumps(rec) + "\n")


def install() -> bool:
    """Wrap VehiclePhysics.step_f32 / step_client_substeps to capture kernel I/O.

    Idempotent; no-op unless WULFRAM_GOLDEN_CAPTURE is set. Returns True if it
    installed the capture wrappers.
    """
    global _installed
    if _installed or not _PATH:
        return False
    from wulfram.physics import VehiclePhysics

    _orig_substeps = VehiclePhysics.step_client_substeps
    _orig_step_f32 = VehiclePhysics.step_f32

    def _wrapped_substeps(self, torque, frame_dt):
        pre_euler = list(self._euler)
        pre_av = self._angular_velocity
        r = _orig_substeps(self, torque, frame_dt)
        _emit({
            "k": "rot", "mode": "substeps",
            "torque": _h(torque), "dt": _h(frame_dt), "use_f32": True,
            "pre_euler": [_h(x) for x in pre_euler], "pre_ang_vel": _h(pre_av),
            "post_euler": [_h(x) for x in self._euler],
            "post_ang_vel": _h(self._angular_velocity),
            "post_matrix": [_h(x) for x in self._matrix],
        })
        return r

    def _wrapped_step_f32(self, torque, dt):
        pre_euler = list(self._euler)
        pre_av = self._angular_velocity
        r = _orig_step_f32(self, torque, dt)
        _emit({
            "k": "rot", "mode": "step_f32",
            "torque": _h(torque), "dt": _h(dt),
            "pre_euler": [_h(x) for x in pre_euler], "pre_ang_vel": _h(pre_av),
            "post_euler": [_h(x) for x in self._euler],
            "post_ang_vel": _h(self._angular_velocity),
            "post_matrix": [_h(x) for x in self._matrix],
        })
        return r

    VehiclePhysics.step_client_substeps = _wrapped_substeps
    VehiclePhysics.step_f32 = _wrapped_step_f32
    _installed = True
    print(f"[GOLDEN] capture installed -> {_PATH}")
    return True
