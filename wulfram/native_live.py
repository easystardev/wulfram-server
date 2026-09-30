"""One native world owner; per-client networking only publishes snapshots."""
from .observer_lifecycle import serialized
import json
import math
import os
from pathlib import Path
import threading
import time

from .native_physics import NativeTankWorld, NativePhysicsError
from .packets import build_behavior_packet
from .building_collision import BuildingCollisionAssets

ROOT = Path(__file__).resolve().parents[2]


def configured_service(server):
    path = ROOT / "server/native-physics.json"
    config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    backend = os.environ.get("WULFRAM_PHYSICS_BACKEND", config.get("backend", "legacy"))
    if backend == "legacy":
        return None
    if backend != "native" or server.up_axis != "z":
        raise ValueError("native backend requires Z-up tanks")
    executable = os.environ.get("WULFRAM_NATIVE_PHYSICS_WORKER", config.get("worker", ""))
    if not executable:
        raise ValueError("native worker executable must be configured")
    return NativeLiveWorld(server, ROOT / executable, int(config.get("step_ms", 20)))


class NativeLiveWorld:
    def __init__(self, server, executable, step_ms=20, *, factory=NativeTankWorld):
        if not 8 <= step_ms <= 110:
            raise ValueError("native step_ms must be 8..110")
        self.server, self.executable, self.step_ms = server, executable, step_ms
        self.factory = factory
        self.world = None
        self.map_name = None
        self.members = {}
        self.statics = {}
        self.snapshots = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.frames = 0
        self.last_tick = 0
        self.dropped_wall_ms = 0
        self._open()

    def _open(self):
        if self.world:
            self.world.close()
        data = ROOT / "slurpysoft-wulfram/data"
        name = self.server.map_name
        matches = [p.name for p in (data / "maps").iterdir() if p.name.lower() == name.lower()]
        if not matches:
            raise ValueError(f"native map unavailable: {name}")
        override = os.environ.get("WULFRAM_TERRAIN_PATH")
        if override and Path(override).resolve() != (data / "maps" / matches[0] / "land").resolve():
            raise ValueError("native backend cannot use a different Python terrain override")
        self.world = self.factory(self.executable, data, matches[0], build_behavior_packet()[1:], timeout=2)
        self.map_name = name
        self.members.clear()
        self.statics.clear()
        with self.lock:
            self.snapshots = {}
        print(f"[NATIVE] world ready map={name} step_ms={self.step_ms} worker={self.executable}")

    @staticmethod
    def key(ctx):
        return int(getattr(ctx.session, "player_id", 0) or ctx.entity_id)

    @staticmethod
    def generation(ctx):
        return (id(ctx), float(getattr(ctx.session, "last_spawn_time", 0) or 0))

    @staticmethod
    def manages(ctx):
        return int(ctx.entity_type) == 0

    def start(self):
        self.thread = threading.Thread(target=self._run, name="native-physics-world", daemon=True)
        self.thread.start()

    def _run(self):
        period = self.step_ms / 1000
        deadline = time.perf_counter() + period
        try:
            while not self.stop_event.wait(max(0, deadline - time.perf_counter())):
                self.step()
                deadline += period
                # Never replay seconds of stale held controls after a stall.
                now = time.perf_counter()
                if now - deadline > .55:
                    self.dropped_wall_ms += int((now - deadline) * 1000)
                    deadline = now + period
                    print(f"[NATIVE] wall backlog clamped dropped_ms={self.dropped_wall_ms}")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[NATIVE] FAILED: {self.error}; native clients stop, no legacy fallback")
        finally:
            self.world.close()

    def controls(self, ctx):
        values = [self.server._normalize_behavior_axis_value(ctx, v) for v in ctx.weapon_system.behavior_slots[:22]]
        values += [0.] * (22 - len(values))
        # The native controller owns the OG negation of slots 1 and 3.
        values[1] = -self.server._get_raw_turn_input(ctx)
        injected = getattr(ctx, "injected_input", None)
        if injected is not None:
            values[2], values[3] = float(injected[0]), -float(injected[1])
        timeout = float(getattr(self.server, "input_stale_timeout_s", 0) or 0)
        last = float(getattr(ctx, "last_action_packet_time", 0) or 0)
        if injected is None and timeout > 0 and last > 0 and time.monotonic() - last > timeout:
            values[1:4] = [0., 0., 0.]
        return values

    def _advance_jump_jet(self, ctx):
        """Bridge the shared slot-4 jump extension into the native world owner.

        The native worker deliberately owns only original tank physics.  Jump
        jets are a clone extension, so keep their edge/cooldown policy in the
        server's shared implementation and turn a successful edge into the
        external velocity replacement already supported by this adapter.
        """
        required = (
            "_get_jumpjet_input",
            "_jump_jet_direction_vector",
            "_apply_jump_jets_fixed_step",
        )
        if not all(hasattr(self.server, name) for name in required):
            return False
        ground = self.server._terrain_physics_ground_z_at(
            ctx.player_pos[0], ctx.player_pos[1]
        ) if hasattr(self.server, "_terrain_physics_ground_z_at") else None
        ground = 0.0 if ground is None else float(ground)
        direction = self.server._jump_jet_direction_vector(ctx, vertical_idx=2)
        fired, impulse, delta = self.server._apply_jump_jets_fixed_step(
            ctx,
            dt=self.step_ms / 1000.0,
            jumpjet_input=self.server._get_jumpjet_input(ctx),
            current_altitude=float(ctx.player_pos[2]) - ground,
            current_vel_up=float(ctx.player_vel[2]),
            direction=direction,
            vertical_idx=2,
        )
        ctx.native_jump_jet_step = {
            "fired": bool(fired),
            "impulse": float(impulse),
            "velocity_delta": tuple(float(value) for value in delta),
            "cooldown_remaining": float(ctx.jump_cooldown_remaining),
            "spawn_lockout": float(ctx.jump_spawn_lockout),
        }
        if fired:
            ctx.player_vel = tuple(
                float(ctx.player_vel[index]) + float(delta[index]) for index in range(3)
            )
        return fired

    def _sync_statics(self):
        present = {}
        for oid, building in list(getattr(self.server, "_building_entities", {}).items()):
            if getattr(self.server, "_building_health", {}).get(oid, 1) <= 0:
                continue
            model = BuildingCollisionAssets.get_model_name(building.entity_type, building.team_id)
            if model:
                present[oid] = (int(building.entity_type), model, tuple(building.pos), int(building.team_id), float(building.heading))
        for oid in list(self.statics):
            if present.get(oid) != self.statics[oid]:
                self.world.remove(oid)
                del self.statics[oid]
        for oid, state in present.items():
            if oid not in self.statics:
                kind, model, pos, team, yaw = state
                self.world.add_static(oid, kind, model, pos, team=team, rotation=(0, 0, yaw))
                self.statics[oid] = state

    @serialized
    def step(self):
        if self.server.map_name != self.map_name:
            self._open()
        clients = [c for c in self.server._snapshot_in_game_clients()
                   if self.manages(c) and c.running and c.session.translation_ack_received
                   and c.player_health > 0 and self.key(c)]
        current = {self.key(c): c for c in clients}
        for oid in list(self.members):
            if oid not in current or self.members[oid] != self.generation(current[oid]):
                self.world.remove(oid)
                del self.members[oid]
        self._sync_statics()
        for oid, ctx in current.items():
            self._advance_jump_jet(ctx)
            # Explicit server teleports/impulses must replace the native state.
            published = getattr(ctx, "native_published_state", None)
            external = published is not None and (tuple(ctx.player_pos), tuple(ctx.player_vel)) != published
            if oid in self.members and external:
                self.world.remove(oid)
                del self.members[oid]
                ctx.native_published_state = None
            if oid not in self.members:
                pose = ctx.player_pose
                self.world.add(oid, ctx.player_pos, team=int(getattr(ctx.session, "team_id", 0) or 0),
                               velocity=ctx.player_vel, rotation=(pose.get("roll", 0), pose.get("pitch", 0), ctx.player_heading),
                               angular_velocity=(*ctx.spring_body_ang_vel, ctx.angular_vel_yaw))
                self.members[oid] = self.generation(ctx)
                print(f"[NATIVE] register client={ctx.client_id} entity={oid} static={len(self.statics)}")
            self.world.controls(oid, self.controls(ctx), health=ctx.player_health, fuel=ctx.player_fuel)
        result = self.world.advance(self.step_ms)
        with self.lock:
            self.snapshots = {b["id"]: (self.members[b["id"]], b, result["tick"]) for b in result["bodies"]}
        self.frames += 1
        self.last_tick = result["tick"]

    @serialized
    def publish(self, ctx):
        if self.error:
            ctx.running = False
            raise NativePhysicsError(self.error)
        with self.lock:
            item = self.snapshots.get(self.key(ctx))
        if item is None or item[0] != self.generation(ctx):
            return
        _, body, tick = item
        # Don't overwrite a teleport/impulse before the owner has consumed it.
        prior = getattr(ctx, "native_published_state", None)
        if prior is not None and (tuple(ctx.player_pos), tuple(ctx.player_vel)) != prior:
            return
        ctx.player_pos, ctx.player_vel = tuple(body["position"]), tuple(body["velocity"])
        roll, pitch, yaw = body["rotation"]
        ctx.player_heading, ctx.player_yaw = yaw, -yaw
        ctx.angular_vel_yaw = ctx.player_angular_vel = body["angular_velocity"][2]
        ctx.spring_body_ang_vel = tuple(body["angular_velocity"][:2])
        ctx.spring_body_matrix = tuple(body["matrix"])
        ctx.player_speed = math.hypot(*ctx.player_vel[:2])
        ctx.player_pose.update(pos=ctx.player_pos, vel=ctx.player_vel, roll=roll, pitch=pitch, yaw=-yaw)
        ctx.rigid_body_sleeping = body["sleeping"]
        ctx.rigid_body_target_pos = ctx.player_pos
        ctx.rigid_body_target_rot = tuple(body["rotation"])
        ctx.vehicle_physics.heading = yaw
        ctx.vehicle_physics.angular_velocity = ctx.angular_vel_yaw
        ctx.native_published_state = (ctx.player_pos, ctx.player_vel)
        ctx.debug_last_controller_step = {"backend": "native", "tick": tick, "pos": ctx.player_pos,
                                          "vel": ctx.player_vel, "body_rotation": body["rotation"],
                                          "angular_velocity": body["angular_velocity"], "clock": "server_sampled",
                                          "step_ms": self.step_ms,
                                          "jump_jet": dict(getattr(ctx, "native_jump_jet_step", {}) or {})}

    def status(self):
        return {"backend": "native", "map": self.map_name, "frames": self.frames,
                "tick": self.last_tick, "tanks": len(self.members), "static_bodies": len(self.statics),
                "step_ms": self.step_ms, "error": self.error, "dropped_wall_ms": self.dropped_wall_ms,
                "thread_alive": self.thread is not None and self.thread.is_alive()}

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        else:
            self.world.close()
