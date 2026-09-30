"""Explicitly selected, isolated native tank world for parity evaluation.

The x86 worker owns matrices, controller history and ordered contact response.
This API does not change live simulation or infer a client clock from receipt time.
"""
import json
import math
from pathlib import Path
import queue
import subprocess
import tempfile
import threading


class NativePhysicsError(RuntimeError):
    pass


class NativeTankWorld:
    def __init__(self, executable, data, map_name, behavior_payload, *, timeout=10):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive timeout required")
        self.timeout = timeout
        self._temporary = tempfile.TemporaryDirectory(prefix="wulfram-native-")
        self._process = None
        self._reader = None
        self._responses = queue.Queue()
        self._lock = threading.Lock()
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        payload = Path(self._temporary.name) / "behavior.bin"
        payload.write_bytes(behavior_payload)
        try:
            self._process = subprocess.Popen(
                [str(Path(executable).resolve()), str(Path(data).resolve()), map_name, str(payload)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                text=True, encoding="utf-8", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._reader = threading.Thread(target=self._read, daemon=True)
            self._reader.start()
            self.ready = self._receive()
            if self.ready.get("protocol") != 1 or self.ready.get("ready") != 1:
                raise NativePhysicsError("unsupported native worker protocol")
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            while True:
                line = self._process.stdout.readline(131073)
                if not line:
                    self._responses.put(None)
                    return
                if len(line) > 131072 or not line.endswith("\n"):
                    raise NativePhysicsError("native response exceeds frame budget")
                self._responses.put(json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))))
        except Exception as exc:
            self._responses.put(exc)

    def _receive(self):
        try:
            response = self._responses.get(timeout=self.timeout)
        except queue.Empty:
            raise NativePhysicsError("native worker response timeout") from None
        if response is None:
            self._process.wait(timeout=self.timeout)
            self._stderr.seek(0)
            message = self._stderr.read(8192).decode("utf-8", errors="replace").strip()
            raise NativePhysicsError(message or "native worker exited")
        if isinstance(response, Exception):
            raise NativePhysicsError(str(response)) from response
        if not isinstance(response, dict):
            raise NativePhysicsError("native response must be an object")
        return response

    def _command(self, *parts):
        line = " ".join(str(part) for part in parts)
        if len(line) > 4096 or "\n" in line or "\r" in line:
            raise ValueError("invalid native command frame")
        with self._lock:
            if self._process is None:
                raise NativePhysicsError("native world is closed")
            try:
                self._process.stdin.write(line + "\n")
                self._process.stdin.flush()
                return self._receive()
            except Exception:
                # A timeout cannot leave a late response queued for a new command.
                self.close()
                raise

    @staticmethod
    def _vector(values):
        values = tuple(float(value) for value in values)
        if len(values) != 3 or not all(math.isfinite(value) and abs(value) <= 3.402823466e38 for value in values):
            raise ValueError("finite float32 vector required")
        return values

    @staticmethod
    def _uint(value, *, minimum=0, maximum=0xffffffff):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("integer outside native range")
        return value

    def add(self, entity_id, position, *, team=0, velocity=(0, 0, 0), rotation=(0, 0, 0), angular_velocity=(0, 0, 0)):
        return self._command("add", self._uint(entity_id, minimum=1), self._uint(team), *self._vector(position), *self._vector(velocity), *self._vector(rotation), *self._vector(angular_velocity))

    def remove(self, entity_id):
        return self._command("remove", self._uint(entity_id, minimum=1))

    def input(self, entity_id, slot, value, tick):
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("input must be in [-1, 1]")
        return self._command("input", self._uint(entity_id, minimum=1), self._uint(slot, maximum=21), float(value), self._uint(tick))

    def resources(self, entity_id, *, health, fuel):
        if not math.isfinite(health) or not 0 <= health <= 1 or not math.isfinite(fuel) or not 0 <= fuel <= 3.402823466e38:
            raise ValueError("health fraction and nonnegative raw fuel required")
        return self._command("resources", self._uint(entity_id, minimum=1), float(health), float(fuel))

    def controls(self, entity_id, values, *, health, fuel):
        values = tuple(float(v) for v in values)
        if len(values) != 22 or not all(math.isfinite(v) and -1 <= v <= 1 for v in values):
            raise ValueError("22 normalized controls required")
        if not math.isfinite(health) or not 0 <= health <= 1 or not math.isfinite(fuel) or not 0 <= fuel <= 3.402823466e38:
            raise ValueError("health fraction and raw fuel required")
        return self._command("control", self._uint(entity_id, minimum=1), health, fuel, *values)

    def add_static(self, entity_id, entity_type, model, position, *, team=0, rotation=(0, 0, 0)):
        if not model or not all(c.isalnum() or c == "_" for c in model):
            raise ValueError("invalid model name")
        return self._command("static", self._uint(entity_id, minimum=1), self._uint(entity_type), self._uint(team), model, *self._vector(position), *self._vector(rotation))

    def advance(self, milliseconds):
        return self._command("advance", self._uint(milliseconds, minimum=1, maximum=550))

    def snapshot(self):
        return self._command("snapshot")

    def close(self):
        process = self._process
        if process is not None:
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            if self._reader is not None:
                self._reader.join(timeout=2)
            process.stdout.close()
            self._process = None
        self._stderr.close()
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
