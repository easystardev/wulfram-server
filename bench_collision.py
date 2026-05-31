#!/usr/bin/env python3
"""Deterministic benchmark + profiler for the rough-cell terrain collision hot path.

Builds the real server (loads terrain + tank collision model), then calls
test_model_collision / _resolve_entity_world_collision at the rough-H180 deep-contact
cell with the live thrust pose, timing and (optionally) profiling. No physics loop, so
no divergence — the same inputs every call.

Usage:
  uv run python bench_collision.py            # time + profile resolve at rough cell
  uv run python bench_collision.py scan       # scan z to find the most expensive center
"""
from __future__ import annotations

import cProfile
import io
import math
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))


def _build():
    import contextlib

    class Mute(io.StringIO):
        def write(self, s):
            return len(s)

    with contextlib.redirect_stdout(Mute()):
        from wulfram.server import WulframServer
        from wulfram.client import ClientContext
        from wulfram.session import Session
        from wulfram.weapons import WeaponSystem

        srv = WulframServer(host="127.0.0.1", port=0)
        ctx = ClientContext(
            client_id=1, client_addr=("127.0.0.1", 5000), session=Session(), entity_id=0x1337
        )
        ctx.session.in_game = True
        ctx.session.team_id = 2
        ctx.entity_type = 0  # TANK
        ctx.weapon_system = WeaponSystem()
        ctx.weapon_system.behavior_slots[5] = 0.824  # thrust
    return srv, ctx


def _model(srv, ctx):
    m = srv._get_entity_world_collision_model(ctx)
    if m is None:
        raise SystemExit("no collision model")
    vertices, cbsp_tree, bounding_radius, z_lift = m
    return vertices, cbsp_tree, bounding_radius, z_lift


def time_query(srv, vertices, cbsp_tree, bounding_radius, center, heading, n=400):
    tgc = srv._terrain_grid_collision
    tgc._query_deadline = None
    fn = tgc.test_model_collision
    for _ in range(20):
        fn(center, heading, vertices, cbsp_tree, bounding_radius)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn(center, heading, vertices, cbsp_tree, bounding_radius)
        ts.append((time.perf_counter() - t) * 1000.0)
    ts.sort()
    return ts


def _instrument_query_count(srv):
    tgc = srv._terrain_grid_collision
    orig = tgc.test_model_collision
    state = {"n": 0}

    def counted(*a, **k):
        state["n"] += 1
        return orig(*a, **k)

    tgc.test_model_collision = counted
    return state


def time_resolve(srv, ctx, pre_pos, vel, dt, n=120):
    """Time the full _resolve_entity_world_collision at a deep-contact thrust pose."""
    tgc = srv._terrain_grid_collision
    tgc._query_deadline = None
    px = pre_pos[0] + vel[0] * dt
    py = pre_pos[1] + vel[1] * dt
    pz = pre_pos[2] + vel[2] * dt
    state = _instrument_query_count(srv)
    for _ in range(5):
        srv._resolve_entity_world_collision(ctx, px, py, pz, vel[0], vel[1], vel[2], pre_pos=pre_pos, pre_vel=vel, dt=dt)
    q_per_step = state["n"] / 5.0
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        srv._resolve_entity_world_collision(ctx, px, py, pz, vel[0], vel[1], vel[2], pre_pos=pre_pos, pre_vel=vel, dt=dt)
        ts.append((time.perf_counter() - t) * 1000.0)
    ts.sort()
    return ts, q_per_step


def main() -> int:
    srv, ctx = _build()
    vertices, cbsp_tree, bounding_radius, z_lift = _model(srv, ctx)
    print(f"model: {len(vertices)} verts, bounding_radius={bounding_radius:.2f}, z_lift={z_lift:.2f}")
    cx, cy = 3160.43, 2463.24
    terrain_h = srv.terrain.get_height(cx, cy)
    heading = math.pi  # 180 deg

    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        import json as _json
        dump = _json.load(open("collision_dump.json"))
        ctx.player_pos = tuple(dump["player_pos"])
        ctx.player_vel = tuple(dump["player_vel"])
        ctx.player_heading = float(dump["heading"]) if not isinstance(dump["heading"], str) else 0.0
        ctx.player_yaw = ctx.player_heading
        ctx.vehicle_physics.heading = ctx.player_heading
        ctx.entity_type = int(dump["entity_type"])
        ctx.session.team_id = dump.get("team_id") or 2
        if dump.get("ground_level_override") not in (None, "None"):
            try:
                ctx.ground_level_override = float(dump["ground_level_override"])
            except (TypeError, ValueError):
                pass
        pre = tuple(dump["pre_pos"]); vel = tuple(dump["pre_vel"]); dt = float(dump["dt"])
        px, py, pz = dump["px"], dump["py"], dump["pz"]
        vx, vy, vz = dump["vx"], dump["vy"], dump["vz"]
        srv._terrain_grid_collision._query_deadline = None
        st = _instrument_query_count(srv)
        for _ in range(3):
            st["n"] = 0
            srv._resolve_entity_world_collision(ctx, px, py, pz, vx, vy, vz, pre_pos=pre, pre_vel=vel, dt=dt)
        qps = st["n"]
        ts = []
        for _ in range(80):
            t = time.perf_counter()
            srv._resolve_entity_world_collision(ctx, px, py, pz, vx, vy, vz, pre_pos=pre, pre_vel=vel, dt=dt)
            ts.append((time.perf_counter() - t) * 1000.0)
        ts.sort()
        print(f"replay resolve (dump collision_ms={dump.get('collision_ms')}, queries={dump.get('query_count')}): median={ts[len(ts)//2]:.2f}ms mean={sum(ts)/len(ts):.2f}ms max={ts[-1]:.2f}ms queries/call={qps}")
        pr = cProfile.Profile(); pr.enable()
        for _ in range(120):
            srv._resolve_entity_world_collision(ctx, px, py, pz, vx, vy, vz, pre_pos=pre, pre_vel=vel, dt=dt)
        pr.disable()
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(22)
        print("\n=== cProfile (replay, by tottime) ===")
        for line in s.getvalue().splitlines():
            if ".py:" in line and ("wulfram" in line or "world_collision" in line or "entities" in line):
                print(line.strip()[:118])
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "resolve":
        ctx.player_heading = heading
        ctx.player_yaw = heading
        ctx.vehicle_physics.heading = heading
        dt = 1.0 / 30.0
        print(f"terrain height at ({cx},{cy}) = {terrain_h:.2f}")
        for dz in (4, 2, 0, -1, -2):
            pre = (cx, cy, terrain_h + dz)
            vel = (-30.0, 0.0, 0.0)  # forward at heading 180
            ts, qps = time_resolve(srv, ctx, pre, vel, dt, n=60)
            print(f"  pre z={pre[2]:7.2f} (terrain{dz:+.0f}): median={ts[len(ts)//2]:8.3f}ms mean={sum(ts)/len(ts):8.3f}ms max={ts[-1]:8.3f}ms  queries/step={qps:.0f}")
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        print(f"terrain height at ({cx},{cy}) = {terrain_h:.2f}")
        for dz in (6, 4, 2, 0, -1, -2, -3, -4, -6):
            cz = terrain_h + dz
            ts = time_query(srv, vertices, cbsp_tree, bounding_radius, (cx, cy, cz), heading, n=120)
            print(f"  center z={cz:7.2f} (terrain{dz:+.0f}): median={ts[len(ts)//2]:7.3f}ms mean={sum(ts)/len(ts):7.3f}ms max={ts[-1]:7.3f}ms")
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "gridscan":
        # Sweep x,y around the rough cell (tank climbs through steeper regions) to find
        # the most expensive query center (reproduce the live ~18ms/query offline).
        best = (0.0, None)
        for ix in range(-7, 8):
            for iy in range(-7, 8):
                x = cx + ix * 6.0
                y = cy + iy * 6.0
                th = srv.terrain.get_height(x, y)
                for dz in (2.0, 0.0, -2.0):
                    center = (x, y, th + dz)
                    ts = time_query(srv, vertices, cbsp_tree, bounding_radius, center, heading, n=25)
                    med = ts[len(ts) // 2]
                    if med > best[0]:
                        best = (med, (center, dz))
        med, info = best
        print(f"max query median={med:.3f}ms at center={info[0]} (terrain{info[1]:+.0f})")
        return 0

    # Default: profile the single query at the expensive steep center found by gridscan.
    center = (3184.43, 2433.24, srv.terrain.get_height(3184.43, 2433.24) + 2.0)
    ts = time_query(srv, vertices, cbsp_tree, bounding_radius, center, heading, n=400)
    print(f"single test_model_collision @ {center}: median={ts[len(ts)//2]:.3f}ms mean={sum(ts)/len(ts):.3f}ms max={ts[-1]:.3f}ms")
    contact = srv._terrain_grid_collision.test_model_collision(center, heading, vertices, cbsp_tree, bounding_radius)
    print(f"contact: {contact is not None}", end="")
    if contact is not None:
        print(f"  pos={tuple(round(v,4) for v in contact.position)} normal={tuple(round(v,4) for v in contact.normal)} pen={contact.penetration:.4f}")
    else:
        print()

    pr = cProfile.Profile()
    pr.enable()
    fn = srv._terrain_grid_collision.test_model_collision
    for _ in range(600):
        fn(center, heading, vertices, cbsp_tree, bounding_radius)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(16)
    print("\n=== cProfile (600 deep-contact queries, by tottime) ===")
    for line in s.getvalue().splitlines():
        if ".py:" in line and ("wulfram" in line or "world_collision" in line):
            print(line.strip()[:120])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
