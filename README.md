# Wulfram Server

Python server emulator for Wulfram II / Wulfram 2.

This repo contains the live server runtime, deterministic gameplay simulation, packet builders, and test scripts. Shared wire-format definitions and quantizer math live in the separate public repo `easystardev/wulfram2-core`.

## Related Repo

- Shared protocol layer: `https://github.com/easystardev/wulfram2-core`

## Layout

```text
server/
  manage_server.py   Process manager for start/stop/restart/log
  run_server.py      Server entry point
  wulfram/           Main server package
  test_*.py          Regression tests
```

## Prerequisite: shared protocol checkout

The server scripts look for the shared protocol repo at `../shared` by default.

Recommended standalone layout:

```text
work/
  server/   <- this repo
  shared/   <- https://github.com/easystardev/wulfram2-core
```

Example:

```powershell
mkdir C:\dev\wulfram-runtime
cd C:\dev\wulfram-runtime
git clone https://github.com/easystardev/wulfram-server.git server
git clone https://github.com/easystardev/wulfram2-core.git shared
cd server
```

If you install `wulfram2-protocol` into your environment some imports will still work, but the checked-out sibling `shared/` repo is the configuration this codebase is actively developed and tested against.

## Quick Start

From the `server/` repo root:

```powershell
uv run python manage_server.py status
uv run python manage_server.py start
uv run python manage_server.py log -n 80
```

Useful commands:

```powershell
uv run python manage_server.py restart
uv run python manage_server.py stop
uv run python manage_server.py fg
uv run python manage_server.py clean
```

Default runtime behavior:

- TCP and UDP both bind on port `2627`
- The lightweight control server binds on `2628`
- `manage_server.py` writes logs to `../server.log`
- `run_server.py` automatically adds `../shared` to `sys.path`

## Current Scope

The current public server path is centered on:

- login, team-select, spawn, and session lifecycle
- deterministic server-side tank physics
- canonical `UPDATE_ARRAY` gameplay replication
- targeted `STATE_REQUEST` / `VIEW_UPDATE` sync replies
- multi-client UDP binding with explicit session keys
- projectile spawning and world-hit handling

This is an active reverse-engineering project, so behavior is still being tightened against `azurefishy-src` and live OG client retests.

## Status Tracking

The canonical parity/fidelity snapshot for the wider workspace lives in:

- `../web-ui/static/clone-status.json`

Append-only structured history lives in:

- `../web-ui/static/status-history.ndjson`

If server behavior changes materially, update those status files first and then sync the supporting markdown docs in the parent workspace.

## Testing

From the `server/` repo root:

```powershell
uv run python test_handlers.py
uv run python test_packets.py
uv run python test_session.py
uv run python test_udp_parser.py
```

## Notes

- Use `manage_server.py`, not direct ad-hoc background launches, for normal start/stop/restart flow.
- The server repo intentionally stays focused on runtime/server code. Shared packet definitions, quantizers, and entity/vehicle enums are maintained in `wulfram2-core`.

## License

Educational and preservation-oriented reverse-engineering work.
