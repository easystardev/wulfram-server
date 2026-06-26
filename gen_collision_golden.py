#!/usr/bin/env python3
"""Generate the golden collision-parity fixture from the CURRENT (reference) scalar
test_model_collision path. Run BEFORE optimizing; the committed golden is then the
determinism reference for test_collision_parity.py.
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

GOLDEN = Path(__file__).parent / "testdata" / "collision_parity_golden.json"


def build():
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
        ctx = ClientContext(client_id=1, client_addr=("127.0.0.1", 5000), session=Session(), entity_id=0x1337)
        ctx.session.team_id = 2
        ctx.entity_type = 0
        ctx.weapon_system = WeaponSystem()
        m = srv._get_entity_world_collision_model(ctx)
    return srv, m


def fixture_inputs(srv):
    """A spread of centers/headings/selections over the rough-H180 cell, flat ground, and
    a steep climb region, at several contact depths — covering the contact + clear paths."""
    cases = []
    cells = [(3160.43, 2463.24), (3184.43, 2433.24), (3136.43, 2487.24), (5050.0, 5050.0), (5183.16, 3073.09)]
    headings = [0.0, math.pi, math.pi / 2.0, -math.pi / 2.0, 0.7853981633974483]
    selections = ["first", "upward_min_depth"]
    for (cx, cy) in cells:
        th = srv.terrain.get_height(cx, cy)
        for dz in (6.0, 2.0, 0.0, -2.0, -4.0):
            for h in headings:
                for sel in selections:
                    cases.append({"center": [cx, cy, th + dz], "heading": h, "selection": sel})
    return cases


def contact_record(contact):
    if contact is None:
        return None
    return {
        "position": [round(float(v), 9) for v in contact.position],
        "normal": [round(float(v), 9) for v in contact.normal],
        "penetration": round(float(contact.penetration), 9),
    }


def main() -> int:
    srv, m = build()
    if m is None:
        raise SystemExit("no collision model")
    vertices, cbsp_tree, bounding_radius, z_lift = m
    tgc = srv._terrain_grid_collision
    tgc._query_deadline = None
    cases = fixture_inputs(srv)
    results = []
    for c in cases:
        contact = tgc.test_model_collision(
            tuple(c["center"]), c["heading"], vertices, cbsp_tree, bounding_radius,
            contact_selection=c["selection"],
        )
        results.append({**c, "contact": contact_record(contact)})
    GOLDEN.write_text(json.dumps({"bounding_radius": bounding_radius, "cases": results}, indent=2) + "\n")
    hits = sum(1 for r in results if r["contact"] is not None)
    print(f"wrote {GOLDEN} : {len(results)} cases, {hits} with contact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
