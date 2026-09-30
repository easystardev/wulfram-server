"""Replay explicit simulation-clock commands in an isolated native world.

Run with uv; requires a worker built by rebuild/analysis/server-physics-worker/build.py.
No connection is made to a game or server. Receipt timestamps need alignment
before being used as simulation timestamps in a fidelity comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "server"), str(ROOT / "shared")]
from wulfram.native_physics import NativeTankWorld
from wulfram.packets import build_behavior_packet


def replay(executable, data, source, output, behavior_file=None):
    document = json.loads(Path(source).read_text(encoding="utf-8"))
    if document.get("clock") != "simulation_ms":
        raise ValueError("replay requires clock=simulation_ms; receipt time is not equivalent")
    commands = document["commands"]
    if not isinstance(commands, list) or len(commands) > 100000:
        raise ValueError("replay command budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    behavior = Path(behavior_file).read_bytes() if behavior_file else build_behavior_packet()[1:]
    (output / "behavior.bin").write_bytes(behavior)
    (output / "input.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assets = [Path(data) / "shapes.zip", Path(data) / "maps" / document["map"] / "land", *sorted((Path(data) / "tables").glob("tank_*"))]
    manifest = {"tier": "T0", "clock": "simulation_ms", "live_fidelity": False,
                "worker_sha256": sha(executable), "input_sha256": sha(source),
                "behavior_sha256": sha(output / "behavior.bin"),
                "assets": {str(path): sha(path) for path in assets}, "complete": False}
    try:
        with NativeTankWorld(executable, data, document["map"], behavior) as world, (output / "trajectory.ndjson").open("w", encoding="utf-8") as stream:
            manifest["ready"] = world.ready
            methods = {name: getattr(world, name) for name in ("add", "remove", "input", "resources", "advance", "snapshot")}
            for index, command in enumerate(commands):
                operation = command["op"]
                if operation not in methods:
                    raise ValueError("unknown replay operation")
                response = methods[operation](**{k: v for k, v in command.items() if k != "op"})
                stream.write(json.dumps({"index": index, "op": operation, **response}, allow_nan=False) + "\n")
        manifest.update(complete=True, commands=len(commands), trajectory_sha256=sha(output / "trajectory.ndjson"))
    except Exception as exc:
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "slurpysoft-wulfram/data")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--behavior-payload", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.worker, args.data, args.input, args.out, args.behavior_payload)))
