# Hot Reload Analysis Report

## Current Mechanism

The `_cmd_reload` method in `control.py` (line 1024) uses `importlib.reload()` to
hot-reload all server modules without restarting. It:

1. Imports all module references before reload.
2. Saves `_SERVER_START` and `FEATURES` as "precious" state.
3. Reloads all modules in dependency order (codec -> session -> physics -> ... -> server).
4. Restores `_SERVER_START` and `FEATURES` onto the newly-reloaded module objects.
5. Swaps `__class__` on the live `WulframServer` and `ControlServer` instances.
6. Swaps `__class__` on all live `VehiclePhysics` instances.

This is fundamentally sound but has gaps. This report catalogs every module-level
global, cross-module binding pattern, and class-swap issue that could cause problems.

---

## 1. Module-Level Globals That Need Preservation

### CRITICAL: State lost on reload

| Module | Global | Type | Risk | Currently Preserved? |
|--------|--------|------|------|---------------------|
| `packets.py:15` | `_SERVER_START` | `float` (monotonic time) | Tick clock resets to now; all `get_ticks()` calls return near-zero, breaking packet sequencing for connected clients. | **YES** |
| `session.py:236` | `FEATURES` | `Features()` instance | All runtime feature flag state (auto_login, tick_loop_enabled, etc.) lost; replaced by defaults + env vars. | **YES** |

### LOW RISK: Config globals re-read from env vars

These are re-initialized from `os.environ` on reload. They get *fresh* values, which
is actually the desired behavior (pick up env var changes). However, if env vars were
NOT set and defaults were overridden at runtime via control commands, those overrides
would be lost.

| Module | Global(s) | Notes |
|--------|-----------|-------|
| `packets.py:18-19` | `BEHAVIOR_SPRING_STATES`, `BEHAVIOR_ACTIVE_EXTRAS` | Boolean flags from env |
| `packets.py:20-30` | `_RAW_HEALTH_MODE`, `_ALLOW_LINEAR_HEALTH`, `HEALTH_RAW_MODE` | Health encoding mode; conditional logic runs at module scope |
| `packets.py:39-44` | `HEALTH_MAX`, `HEALTH_RANGE`, `ENERGY_MAX`, `ENERGY_RANGE`, `HEALTH_NORMALIZED`, `ENTITY_VITALS_MODE` | Quantizer scaling from env |
| `packets.py:48-52` | `BEHAVIOR_GROUND_FRICTION`, `BEHAVIOR_TURN_RATE`, `BEHAVIOR_SUSPENSION_DAMPENING`, `BEHAVIOR_MAX_ALTITUDE`, `BEHAVIOR_GRAVITY_PCT` | Behavior packet physics |
| `packets.py:54-61` | `VEC_POS_MAX/RANGE`, `VEC_VEL_MAX/RANGE`, `VEC_ROT_MAX/RANGE`, `VEC_SPIN_MAX/RANGE` | Vector quantizer config |
| `packets.py:64-65` | `LOCAL_STATE_TURRET_HEADER_BITS`, `LOCAL_STATE_TURRET_PRIORITY` | Turret angle encoding |

### NO RISK: Immutable constants and class definitions

These are safe because they define fixed values or class/enum types that don't hold
mutable state:

| Module | Global(s) | Notes |
|--------|-----------|-------|
| `packets.py:73-138` | `PacketType`, `PACKET_NAMES` | Protocol constants (dict of int->str) |
| `packets.py:100-103` | `BEHAVIOR_HEADER_SIZE`, `BEHAVIOR_WEAPON_UNITS`, etc. | Integer constants |
| `weapons.py:24-108` | `BehaviorSlot`, `EntityType`, `WeaponType`, `WEAPON_NAMES`, `TANK_WEAPON_SLOTS` | Enum/constant defs |
| `jump_jets.py:25-30` | `JUMP_JET_CONFIGS` | Dict of `JumpJetConfig` dataclasses (config, not runtime state) |
| `server.py:46-47` | `LOCAL_STATE_PRIMARY_TURRET_TYPES`, `LOCAL_STATE_SECONDARY_TURRET_TYPES` | Frozen sets |
| `__init__.py:11` | `__version__` | String constant |

---

## 2. Stale Name Binding Problem (FEATURES)

This is the most dangerous pattern in the codebase. Three modules import `FEATURES`
at module scope:

```python
# handlers.py:11
from .session import Phase, FEATURES

# server.py:22
from .session import Session, Phase, FEATURES

# control.py:27
from .session import Session, Phase, FEATURES
```

After `importlib.reload(session_mod)`, the `session` module object gets a **new**
`FEATURES` instance. The reload code then does:

```python
session_mod.FEATURES = old_features  # Restores the saved instance
```

This correctly patches `session_mod.FEATURES`. However, when `handlers_mod`,
`control_mod`, and `server_mod` are subsequently reloaded, their `from .session
import FEATURES` statement executes again, which binds their module-level `FEATURES`
name to whatever `session.FEATURES` is at that moment.

**Current reload order** (from `_cmd_reload`):
```
codec -> session -> physics -> jump_jets -> weapons -> client -> transport
-> packets -> handlers -> control -> server
```

Since `session` is reloaded early and `FEATURES` is restored immediately after all
reloads, the following happens:

1. `session` reloaded -> `session.FEATURES` = new `Features()` (wrong).
2. `handlers` reloaded -> `handlers.FEATURES` = `session.FEATURES` = new (wrong).
3. `control` reloaded -> `control.FEATURES` = `session.FEATURES` = new (wrong).
4. `server` reloaded -> `server.FEATURES` = `session.FEATURES` = new (wrong).
5. **After loop**: `session_mod.FEATURES = old_features` (patches session only).

**Result**: `handlers.FEATURES`, `control.FEATURES`, and `server.FEATURES` all
point to the **discarded** new `Features()` instance, not the preserved one. Any
feature flag checks in those modules use stale defaults.

**Severity**: HIGH. This silently breaks feature flags in handlers and server after
every full reload. The FEATURES object accessed from handlers.py (the most critical
consumer, used for spawn/BPS/WANT_UPDATES logic) will have default values instead of
the live runtime state.

---

## 3. Cross-Module Import Binding Issues

Beyond FEATURES, several other names are imported at module scope and could become
stale after reload:

### handlers.py imports from packets.py

```python
from .packets import (
    PacketType, build_hello_version, build_hello_session_key,
    build_login_status, build_player, build_team_info,
    build_world_stats, build_bps_response, build_chat_message,
    ...
)
```

When `handlers` is reloaded **after** `packets`, it re-executes these imports and
gets the new function objects from the reloaded `packets` module. This is **correct**
because `handlers` is reloaded after `packets` in the reload order.

However, if any **running threads** (tick loop, ping loop, client handler) hold
references to old function objects imported before the reload, those threads will
continue calling the old code. This is inherent to Python's reload mechanism and
cannot be fully fixed without thread restart.

### server.py module-level imports

`server.py` imports many names from `packets`, `weapons`, `session`, `handlers`,
etc. at module scope. Since `server` is the **last** module reloaded, all its
imports correctly pick up the new module contents. The class swap
(`self.server.__class__ = server_mod.WulframServer`) ensures the live instance uses
new methods.

### Closures in running threads

The tick loop (`_tick_loop`), ping loop (`_ping_loop`), and UDP loop (`_udp_loop`)
are started as daemon threads with closures over `self` (the server instance) and
`ctx` (client contexts). After reload:

- `self.__class__` is swapped, so new method implementations are used when called
  via `self.method_name()`.
- But any local variables captured in the closure (including imported names at the
  top of the method) retain old references.
- Thread functions defined as top-level methods on the class are correctly updated
  by the `__class__` swap.

**Risk**: MEDIUM. Running threads will pick up new method code through the class
swap, but any thread that captured a local reference to a function or constant
before the reload will use the old version until the thread restarts.

---

## 4. Class Swap Gaps

### Classes that ARE swapped:
- `WulframServer` (via `self.server.__class__ = server_mod.WulframServer`)
- `ControlServer` (via `self.__class__ = control_mod.ControlServer`)
- `VehiclePhysics` (via `ctx.vehicle_physics.__class__ = physics_mod.VehiclePhysics`)

### Classes that are NOT swapped:
- **`Session`** instances (`ctx.session`) -- not swapped. If Session gains new
  methods or fields, existing sessions will not have them.
- **`ClientContext`** instances -- not swapped. Same issue.
- **`TCPHandler`** / `UDPHandler` instances -- not swapped.
- **`WeaponSystem`** instances (`ctx.weapon_system`) -- not swapped.
- **`JumpJetSystem`** instances (`ctx.jump_jet_system`) -- not swapped.
- **`PacketLogger`** instance (`self.logger`) -- not swapped.

**Risk**: MEDIUM. If any of these classes have methods added or modified, live
instances will not pick up the changes. Adding a `__class__` swap for
`WeaponSystem` and `JumpJetSystem` would cover the most likely cases.

### isinstance checks

There are 3 `isinstance` calls in `server.py` (lines 319, 2367, 2371). All three
check against built-in types (`dict`, `str`), not against reloaded classes, so they
are safe.

No `isinstance` checks are made against `Session`, `ClientContext`, or other
custom classes, so the "old class vs new class" identity problem does not apply here.

---

## 5. Recommendations (Priority Ordered)

### P0 -- Fix FEATURES stale binding (HIGH impact, easy fix)

The FEATURES restore must happen **before** the modules that import it are reloaded,
or the importing modules must access FEATURES through the module object rather than
a bare name.

**Option A** (minimal change): Restore FEATURES onto `session_mod` immediately after
reloading session, before reloading handlers/control/server:

```python
importlib.reload(session_mod)
session_mod.FEATURES = old_features  # Restore BEFORE dependent modules reload
reloaded.append("session")
# ... continue reloading physics, weapons, etc. ...
importlib.reload(handlers_mod)  # Now picks up restored FEATURES
```

**Option B** (more robust): Change `handlers.py`, `server.py`, and `control.py` to
access FEATURES through the module:

```python
# Instead of: from .session import FEATURES
# Use:        from . import session
# Then:       session.FEATURES.auto_join_team
```

This way the name `session` always refers to the current module object, and
attribute access goes through the live module's namespace.

**Option C** (best long-term): Move FEATURES into the `WulframServer` instance as
`self.features`. This eliminates the global entirely and makes it naturally
preserved across reloads (since the server instance is preserved).

### P1 -- Add class swaps for WeaponSystem and JumpJetSystem (MEDIUM impact)

```python
with self.server.clients_lock:
    for ctx in self.server.clients.values():
        if ctx.weapon_system:
            ctx.weapon_system.__class__ = weapons_mod.WeaponSystem
        if ctx.jump_jet_system:
            ctx.jump_jet_system.__class__ = jump_jets_mod.JumpJetSystem
        if ctx.session:
            ctx.session.__class__ = session_mod.Session
```

### P2 -- Add class swap for Session (MEDIUM impact)

Session instances survive reload but their class is never swapped. If `Session`
gains new methods (e.g., a new `transition_to` rule), live sessions would not pick
them up.

### P3 -- Document which env-var configs are re-read on reload (LOW impact)

The `packets.py` module-level globals are re-read from `os.environ` on every reload.
This is actually useful -- it means changing an env var and running `reload` picks up
the new value. But it should be documented so operators know that:

- Env var changes take effect on reload (for `packets.py` globals).
- Runtime overrides via control commands to FEATURES are preserved (if P0 is fixed).
- Runtime overrides to `packets.py` globals (if any existed) would be lost.

### P4 -- Move packets.py config globals into a Config class (LOW impact, long-term)

Replace the ~30 module-level config globals in `packets.py` with a single
`PacketConfig` dataclass instance. This makes it trivial to save/restore across
reloads and provides a single point of truth for all quantizer/encoding settings.

```python
@dataclass
class PacketConfig:
    health_max: float = 1.0
    health_range: float = 1.0
    vec_pos_max: float = 8192.0
    ...

PACKET_CONFIG = PacketConfig()  # Would need preservation like FEATURES
```

### P5 -- Lazy imports in thread entry points (LOW impact, defensive)

Functions that run in long-lived threads (tick loop, ping loop) could use lazy
imports at the top of their main loop iteration rather than relying on module-level
imports. This ensures they pick up reloaded code on the next iteration:

```python
def _tick_loop(self, ctx):
    while ctx.running:
        from .packets import get_ticks, build_update_array_heartbeat
        # ... use fresh references each iteration ...
```

Several methods in `control.py` already use this pattern (local `from .packets
import ...` inside method bodies). Extending it to thread loops would make reload
more robust.

### P6 -- Consider thread restart on full reload (LOW priority, complex)

The most complete solution would be to gracefully stop and restart tick/ping threads
after a full reload. This ensures threads use 100% new code. However, this is complex
and risks dropping packets during the restart window.

---

## 6. Summary of Current State

| Item | Status | Notes |
|------|--------|-------|
| `_SERVER_START` preservation | OK | Saved and restored |
| `FEATURES` preservation | BROKEN | Restored to session_mod but stale in handlers/server/control |
| VehiclePhysics class swap | OK | Done in reload loop |
| WulframServer class swap | OK | Done in reload loop |
| ControlServer class swap | OK | Done in reload loop |
| Session class swap | MISSING | Live sessions keep old class |
| ClientContext class swap | MISSING | Live contexts keep old class |
| WeaponSystem class swap | MISSING | Live weapon systems keep old class |
| JumpJetSystem class swap | MISSING | Live jump jet systems keep old class |
| TCPHandler/UDPHandler class swap | MISSING | Low risk (rarely changed) |
| packets.py env-var globals | RE-READ | Fresh values from env on reload (by design) |
| Running thread code refresh | PARTIAL | Class swap helps; closures keep old refs |

---

## 7. Dependency Graph

```
codec.py          (leaf - no internal imports)
  ^
  |
packets.py        (imports codec)
  ^
  |
session.py        (leaf - no internal imports; defines FEATURES)
  ^    ^
  |    |
  |  transport.py  (imports codec, packets)
  |    ^
  |    |
weapons.py        (imports codec, packets)
  |
physics.py        (leaf - no internal imports)
  |
jump_jets.py      (leaf - no internal imports)
  |
client.py         (imports physics; TYPE_CHECKING: session, transport, weapons, jump_jets)
  |
handlers.py       (imports session, packets; TYPE_CHECKING: server, client)
  |
control.py        (imports session, packets; many lazy imports in methods)
  |
server.py         (imports everything: session, transport, codec, control, weapons,
                    jump_jets, client, packets, handlers)
```

The current reload order (codec -> session -> physics -> jump_jets -> weapons ->
client -> transport -> packets -> handlers -> control -> server) mostly follows this
dependency graph bottom-up, which is correct. The one issue is that `session` is
reloaded very early but `FEATURES` is not restored until after all modules are done.
