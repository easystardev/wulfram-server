#!/usr/bin/env python3
"""Determinism / parity test for the optimized terrain-collision CBSP path.

Proves the inlined/cached collision path in world_collision.py returns the SAME
contact results (hit / point / normal / penetration) as the reference scalar path
captured in collision_parity_golden.json, across a fixture of rough-cell, flat, and
steep centers at several depths/headings/selections.

The inlining (Moeller-Trumbore + point-in-triangle) is bit-identical arithmetic, so the
tolerance is tight. This is the determinism guarantee that lets the per-step collision
wall-clock budget be disabled.

Run standalone:  uv run python test_collision_parity.py
Also invoked by test_handlers.py.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

GOLDEN = Path(__file__).parent / "collision_parity_golden.json"

# Tolerances. The optimization is inlined bit-identical scalar math, so results match to
# near machine epsilon; we allow a small slack for float associativity on point/normal.
POS_TOL = 1e-6
NORMAL_TOL = 1e-6
PEN_TOL = 1e-6


def _build():
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


def _record(contact):
    if contact is None:
        return None
    return {
        "position": [float(v) for v in contact.position],
        "normal": [float(v) for v in contact.normal],
        "penetration": float(contact.penetration),
    }


def _vec_close(a, b, tol):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_collision_parity_matches_golden():
    """Optimized test_model_collision must reproduce the golden scalar-reference
    contacts within tolerance for every fixture case (the determinism guarantee)."""
    if not GOLDEN.exists():
        raise SystemExit(f"missing golden fixture {GOLDEN}; run gen_collision_golden.py")
    golden = json.loads(GOLDEN.read_text())
    srv, m = _build()
    assert m is not None, "no collision model"
    vertices, cbsp_tree, bounding_radius, z_lift = m
    tgc = srv._terrain_grid_collision
    tgc._query_deadline = None  # parity path must run unbudgeted

    mismatches = []
    n_contact = 0
    for case in golden["cases"]:
        got = _record(
            tgc.test_model_collision(
                tuple(case["center"]), case["heading"], vertices, cbsp_tree, bounding_radius,
                contact_selection=case["selection"],
            )
        )
        exp = case["contact"]
        if exp is None and got is None:
            continue
        if (exp is None) != (got is None):
            mismatches.append((case, exp, got, "hit/miss differs"))
            continue
        n_contact += 1
        if not _vec_close(got["position"], exp["position"], POS_TOL):
            mismatches.append((case, exp, got, "position"))
        elif not _vec_close(got["normal"], exp["normal"], NORMAL_TOL):
            mismatches.append((case, exp, got, "normal"))
        elif abs(got["penetration"] - exp["penetration"]) > PEN_TOL:
            mismatches.append((case, exp, got, "penetration"))

    if mismatches:
        c, e, g, why = mismatches[0]
        raise AssertionError(
            f"{len(mismatches)}/{len(golden['cases'])} collision parity mismatches; "
            f"first ({why}): center={c['center']} heading={c['heading']} sel={c['selection']} "
            f"expected={e} got={g}"
        )
    print(
        f"test_collision_parity_matches_golden: PASSED "
        f"({len(golden['cases'])} cases, {n_contact} contacts, pos_tol={POS_TOL})"
    )
    return True


def _ref_triangles_intersection(tgc, tri_a, tri_b):
    """Reference: the ORIGINAL _triangles_intersection_point WITHOUT the separating-plane
    early-reject (6 segment tests always, then coplanar). Uses the same (inlined) segment
    test, so this isolates the early-reject structural change."""
    from wulfram.world_collision import _cross3, _normalize3, _sub3
    for idx in range(3):
        hit = tgc._segment_triangle_intersection(tri_a[idx], tri_a[(idx + 1) % 3], tri_b)
        if hit is not None:
            return hit
    for idx in range(3):
        hit = tgc._segment_triangle_intersection(tri_b[idx], tri_b[(idx + 1) % 3], tri_a)
        if hit is not None:
            return hit
    normal_b = _normalize3(_cross3(_sub3(tri_b[1], tri_b[0]), _sub3(tri_b[2], tri_b[0])))
    if normal_b is not None and abs(tgc._point_plane_distance(tri_a[0], tri_b, normal_b)) <= 1e-5:
        if tgc._point_in_triangle(tri_a[0], tri_b, normal_b):
            return tri_a[0]
    normal_a = _normalize3(_cross3(_sub3(tri_a[1], tri_a[0]), _sub3(tri_a[2], tri_a[0])))
    if normal_a is not None and abs(tgc._point_plane_distance(tri_b[0], tri_a, normal_a)) <= 1e-5:
        if tgc._point_in_triangle(tri_b[0], tri_a, normal_a):
            return tri_b[0]
    return None


def test_triangles_intersection_early_reject_parity():
    """The separating-plane early-reject in _triangles_intersection_point must return the
    same hit/point as the no-early-reject reference across separated, intersecting, and
    near-coplanar triangle pairs (covers the structural change directly, not just via the
    end-to-end golden)."""
    import random

    srv, m = _build()
    tgc = srv._terrain_grid_collision
    rng = random.Random(1234567)

    def rand_tri(scale, off):
        return tuple(
            (off[0] + rng.uniform(-scale, scale), off[1] + rng.uniform(-scale, scale),
             off[2] + rng.uniform(-scale, scale))
            for _ in range(3)
        )

    pairs = []
    # Random pairs at varied separations (many separated, some intersecting).
    for _ in range(4000):
        a = rand_tri(5.0, (0.0, 0.0, 0.0))
        b = rand_tri(5.0, (rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(-8, 8)))
        pairs.append((a, b))
    # Crafted near-coplanar pairs (the parity-sensitive case for the early-reject).
    for _ in range(2000):
        a = rand_tri(5.0, (0.0, 0.0, 0.0))
        dz = rng.choice([0.0, 1e-7, -1e-7, 5e-6, -5e-6, 2e-5, -2e-5])
        b = tuple((p[0] + rng.uniform(-3, 3), p[1] + rng.uniform(-3, 3), p[2] + dz) for p in a)
        pairs.append((a, b))

    mism = 0
    first = None
    for a, b in pairs:
        got = tgc._triangles_intersection_point(a, b)
        exp = _ref_triangles_intersection(tgc, a, b)
        if (got is None) != (exp is None):
            mism += 1
            if first is None:
                first = (a, b, exp, got, "hit/miss")
        elif got is not None and not _vec_close(got, exp, 1e-9):
            mism += 1
            if first is None:
                first = (a, b, exp, got, "point")
    if mism:
        a, b, exp, got, why = first
        raise AssertionError(f"{mism}/{len(pairs)} early-reject mismatches; first ({why}): a={a} b={b} exp={exp} got={got}")
    print(f"test_triangles_intersection_early_reject_parity: PASSED ({len(pairs)} triangle pairs)")
    return True


if __name__ == "__main__":
    ok = test_collision_parity_matches_golden()
    ok = test_triangles_intersection_early_reject_parity() and ok
    raise SystemExit(0 if ok else 1)
