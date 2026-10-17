#!/usr/bin/env python3
"""Server wire changes from the upstream wulfram3.com 2026-10-16 follow-up lane.

Fixture ``test_data/upstream_wire_1016.json`` holds REAL server->client port-2627
game messages from the 10-16 capture (VIEW_UPDATE, UPDATE_ARRAY, DOCKING,
SHIP_STATUS, SUPPLY_SHIP_INFO), upstream's TRANSLATION/BEHAVIOR, and two OG
client WEAPON_DEMAND packets from 09-18. No login/HELLO/relay/token bytes.

Covers: TANK 0x18 local-state encoder fix, WEAPON_DEMAND 14-byte framing + ACK,
the opt-in community knobs (local vitals VU, UA byte-budget split, DOCKING
stream, ship batch repeat) and the dock pin wiring (should_pin monkeypatched).
Run: uv run python test_upstream_1016_wire.py
"""
import json
import math
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))
sys.path.insert(0, str(HERE.parent / "client"))

from wulfram import dock_pin, handlers  # noqa: E402
from wulfram import packets as P  # noqa: E402
from wulfram.server import WulframServer  # noqa: E402
from wulfram.server_net import NetMixin  # noqa: E402
from wulfram.session import Phase, Session  # noqa: E402
from wulfram2_protocol.codec import BitReader  # noqa: E402

from wulfram_client.network import decoder as D  # noqa: E402
from wulfram_client.network.behavior import parse_behavior  # noqa: E402
from wulfram_client.network.quantizer import parse_translation  # noqa: E402

FX = json.loads((HERE / "test_data" / "upstream_wire_1016.json").read_text())
H = bytes.fromhex
UP_Q = parse_translation(H(FX["translation_hex"]))
UP_B = parse_behavior(H(FX["behavior_hex"]))
OUR_Q = parse_translation(P.build_translation_packet())
OUR_B = parse_behavior(P.build_behavior_packet())

# Captured upstream TANK (0x18) packets (also in client/test_messages.py).
UP_TANKS = [
    "1847743fa58001004010000000000000833021090d2800ff523600130e4600000000000000000000000000",
    "184775fc9d800100401000000000000084c021090d2800ff523600130e4600000000000000000000000000",
    "18477c0ed8800100401000000000000094b0214e7f66010bbb7e0006021cb0000000000000000000000000",
    "18b735aa078001004010000000000020e45021090d2800ff52360012d17fc0000000000000000000000000",
]


def _fixed(raw):
    return (raw - (1 << 32) if raw & 0x80000000 else raw) / 65536.0


def raw_record(payload: bytes, q, b):
    """(local_state, [(entity_id, raw 10-bit mask)], EntityUpdates) for a UA/VU payload."""
    off = 9 if payload[0] == 0x0F else 5
    br = BitReader(payload[off:])
    ls = D._read_local_player_state(br, q, b)
    n = br.read_bits(8)
    ent_id = br.read_bits(32)
    br.read_bits(1)
    mask = br.read_bits(10)
    if payload[0] == 0x0F:
        _ts, _tick, _ls, ents = D.decode_view_update(payload, qtable=q, behavior_config=b)
    else:
        _tick, _ls, ents = D.decode_update_array(payload, qtable=q, behavior_config=b)
    return ls, n, ent_id, mask, ents


def clean_decode(payload, q, b):
    errors = []
    if payload[0] == 0x0F:
        out = D.decode_view_update(payload, qtable=q, behavior_config=b, errors=errors)
    else:
        out = D.decode_update_array(payload, qtable=q, behavior_config=b, errors=errors)
    assert not errors, errors
    return out


class FakeUDP:
    def __init__(self):
        self.sent = []

    def send_to(self, data, addr, log=True):
        self.sent.append((bytes(data), addr))


class FakeTCP:
    def __init__(self):
        self.sent = []

    def send(self, payload, log=True):
        self.sent.append(bytes(payload))


# =========================================================== 1. TANK 0x18


def _tank_fields(hexs):
    """Decode one captured TANK with the fixed client decoder + upstream tables."""
    pkt = H(hexs)
    br = BitReader(pkt[1:])
    tick = br.read_bits(32)
    ls = D._read_local_player_state(br, UP_Q, UP_B)
    unit, net_id, team = br.read_bits(32), br.read_bits(32), br.read_bits(8)
    pos = tuple(_fixed(br.read_bits(32)) for _ in range(3))
    rot = tuple(_fixed(br.read_bits(32)) for _ in range(3))
    return tick, ls, unit, net_id, team, pos, rot


def test_tank_builder_reproduces_captured_upstream_tank_bytes():
    """With upstream's weapon-0 widths (2 ammo bits + 8-bit HUD) our builder emits the
    captured bytes exactly; the pre-fix builder (no ammo/HUD bits) could not."""
    for hexs in UP_TANKS:
        tick, ls, unit, net_id, team, pos, rot = _tank_fields(hexs)
        assert ls.weapon_id == 0 and unit == 0 and team == 2
        built = P.build_udp_tank_packet_wf(
            net_id, unit, team, pos, rot, tick=tick, include_vitals=True, weapon_id=0,
            health=ls.health, energy=ls.energy,
            ammo_count_bits=2, ammo_count=ls.ammo_mask,
            primary_turret_bits=8, primary_turret_angle=ls.primary_turret,
            secondary_turret_bits=0,
        )
        assert built == H(hexs), (built.hex(), hexs)
        r = D.decode_tank_packet(built, qtable=UP_Q, behavior_config=UP_B)
        assert r[0] == net_id


def test_tank_builder_default_widths_follow_our_behavior():
    """Widths default to the BEHAVIOR this server sends: the OG reader (= fixed client
    decoder fed our tables) recovers id/type/team/pos for every weapon type."""
    pos = (1234.5, 2345.25, 101.0)
    for weapon in (0, 1, 2, 3):
        for builder in ("wf", "tcp"):
            if builder == "wf":
                pkt = P.build_udp_tank_packet_wf(4321, 0, 1, pos, (0.0, 0.0, 0.5), tick=777,
                                                 include_vitals=True, weapon_id=weapon)
            else:
                pkt = P.build_tank_packet(4321, 0, pos, (0.0, 0.0, 0.5), flags=1, include_vitals=True,
                                          weapon_id=weapon, tick=777)
            r = D.decode_tank_packet(pkt, qtable=OUR_Q, behavior_config=OUR_B)
            assert (r[0], r[1], r[2]) == (4321, 0, 1), (weapon, builder, r)
            assert r[3] == pos
            br = BitReader(pkt[1:])
            br.read_bits(32)
            D._read_local_player_state(br, OUR_Q, OUR_B)
            assert br.read_bits(32) == 0 and br.read_bits(32) == 4321
    # Negative control: weapon 0 really needs the extra 9 ammo + 8 HUD bits.
    assert P.tank_local_state_widths(0) == (9, 8, 0)
    assert P.tank_local_state_widths(1) == (0, 0, 8)
    assert P.tank_local_state_widths(2) == (0, 0, 0)


def test_tank_default_call_shapes_are_byte_identical_to_before():
    """Goldens recorded from the pre-fix encoder: presence-0 and weapon-2 TANKs (every
    default call site) are unchanged."""
    pos = (4240.822265625, 4085.13818359375, 304.89208984375)
    rot = (0.0, 0.0, 1.25)
    assert P.build_udp_tank_packet_wf(2099, 0, 2, pos, rot, tick=0x47743FA5).hex() == (
        "1847743fa50000000000000419810848694007fa91b00098723000000000000000000000a00000")
    assert P.build_udp_tank_packet_wf(2099, 0, 2, pos, rot, tick=0x47743FA5, include_vitals=True,
                                      weapon_id=2, health=0.5, energy=0.75).hex() == (
        "1847743fa58a0040000000000000020cc0842434a003fd48d8004c391800000000000000000000500000")


# ===================================================== 2. WEAPON_DEMAND 0x2E


OG_SHAKE = handlers._build_server_d_handshake(None)  # OG private table carries (0x2E, 3)
NATIVE_SHAKE = bytes.fromhex(
    "03" "00000110" "00000000" "00000001"
    "0010" "506c61796572207265717565737473" "00"
    "0000000a" "00000020" "00000025" "00000026" "0000002b" "0000002e"
    "00000033" "0000003b" "00000042" "00000047" "00000049"
    "0000000a"
    "0000002000000003" "0000002500000003" "0000002600000003" "0000002b00000003"
    "0000002e00000003" "0000003300000003" "0000003b00000003" "0000004200000003"
    "0000004700000003" "0000004900000003"
)


def _wd_server():
    srv = WulframServer.__new__(WulframServer)
    srv.clients = {}
    srv.clients_lock = threading.Lock()
    srv.udp_addr_to_client = {}
    srv.session_key_to_client = {}
    srv.udp_handler = FakeUDP()
    srv.debug_viewpoint = False
    srv.debug_udp_raw = False
    return srv


def _wd_ctx(shake):
    ctx = SimpleNamespace(client_id=1, running=True, entity_id=7, tcp_handler=None,
                          session=Session(phase=Phase.IN_GAME),
                          weapon_system=SimpleNamespace(current_weapon=4))
    if shake is not None:
        parsed = handlers._parse_empirical_client_d_handshake(shake)
        ctx.session.client_d_handshake = {k: parsed[k] for k in ("receive_window", "streams", "private_modes")}
    return ctx


def test_weapon_demand_framed_parse_and_split():
    for hexs in FX["weapon_demand_0918"]:
        wd = H(hexs)
        parsed = handlers.parse_framed_weapon_demand(wd)
        assert parsed["mode"] == 0 and parsed["amount"] == 1 and parsed["slot"] in (5, 6)
        srv = _wd_server()
        ping = bytes.fromhex("0b000007d000000000")
        # Whole 14-byte packet, then the next packet in the datagram parses.
        assert list(srv._parse_udp_datagram(wd + ping, None)) == [wd, ping]
    legacy = b"\x2E" + bytes((0,)) + struct.pack(">Ii", 4, 0)
    assert handlers.parse_framed_weapon_demand(legacy) is None
    assert list(_wd_server()._parse_udp_datagram(legacy, None)) == [legacy]


def test_weapon_demand_acked_when_declared_reliable_once_applied():
    wd = H(FX["weapon_demand_0918"][0])  # seq 1
    for shake, ack in ((OG_SHAKE, "02012e0001"), (NATIVE_SHAKE, "0200010009012e0001")):
        srv = _wd_server()
        ctx = _wd_ctx(shake)
        srv._handle_single_udp_packet(ctx, wd, ("127.0.0.1", 9))
        srv._handle_single_udp_packet(ctx, wd, ("127.0.0.1", 9))  # stream retransmit
        acks = [d for d, _ in srv.udp_handler.sent if d[0] == 0x02]
        assert len(acks) == 2 and acks[0].hex().startswith(ack[:4]) and acks[0].hex().endswith("2e0001"), acks
        assert ctx.last_weapon_demand == (0, 5, 1)
        assert ctx.weapon_system.current_weapon == 4  # buy/sell is not a weapon selection
    srv = _wd_server()
    ctx = _wd_ctx(None)  # no declared handshake: nothing to ACK
    srv._handle_single_udp_packet(ctx, wd, ("127.0.0.1", 9))
    assert not srv.udp_handler.sent


def test_weapon_demand_unframed_python_client_form_still_selects():
    srv = _wd_server()
    srv._debug_comm_allowed_for_client = lambda ctx: False
    ctx = _wd_ctx(OG_SHAKE)
    from wulfram.weapons import TANK_WEAPON_SLOTS
    slot = sorted(TANK_WEAPON_SLOTS)[-1]
    legacy = b"\x2E" + bytes((0,)) + struct.pack(">Ii", slot, 0)
    srv._handle_single_udp_packet(ctx, legacy, ("127.0.0.1", 9))
    assert ctx.weapon_system.current_weapon == slot
    assert not srv.udp_handler.sent  # unframed form carries no sequence to ACK


# ================================================ 3. community knobs (opt-in)


def _knob_server(**over):
    srv = WulframServer.__new__(WulframServer)
    srv.udp_handler = FakeUDP()
    srv.up_axis = "z"
    srv.update_local_state_mode = "wf"
    srv.behavior_weapon_caps = P.get_behavior_weapon_capability_counts()
    srv.local_state_active_interval_s = 0.05
    srv.local_state_idle_interval_s = 0.35
    srv.ship_status_interval_s = 0.0
    srv.ship_status_repeat_s = 0.0
    srv.ship_status_transport = "udp"
    srv.docking_stream = False
    srv.docking_interval_s = 0.05
    srv.dock_pin_enabled = False
    srv.dock_pin_clients = frozenset()
    srv.dock_pin_interval_s = 0.05
    srv.dock_grounded_vz = 1.0
    srv.local_vu_entity = False
    srv.ua_split_bytes = 0
    vit = {"health": 1.0, "fuel": 1.0, "fire": False}
    srv._vit = vit
    srv._get_health_value = lambda ctx: vit["health"]
    srv._get_energy_value = lambda ctx: vit["fuel"]
    srv._tank_primary_replication_active = lambda ctx: vit["fire"]
    srv._fresh_remote_view_update_timestamp = lambda ctx, tick: tick + 1000
    srv._community_local_state = lambda ctx: (True, dict(
        weapon_id=2, health=vit["health"], fuel=vit["fuel"], ammo_count_bits=0, ammo_count=0,
        primary_turret_bits=0, primary_turret_angle=0.0, secondary_turret_bits=0,
        secondary_turret_angle=0.0, turret_max=1.0, turret_range=1.0))
    srv._to_client_pos = lambda pos: tuple(pos)
    srv._local_player_sync_rotation = lambda ctx: (0.0, 0.0, ctx.player_heading)
    for k, v in over.items():
        setattr(srv, k, v)
    return srv


def _knob_ctx():
    s = Session(phase=Phase.IN_GAME)
    s.in_game = True
    s.entity_id = 2124
    s.udp_addr = ("10.0.0.5", 5000)
    return SimpleNamespace(client_id=1, entity_id=2124, entity_type=0, session=s, tcp_handler=FakeTCP(),
                           player_pos=(100.0, 200.0, 50.0), player_vel=(0.0, 0.0, 0.0), player_heading=0.5,
                           last_decoded_input={}, supply_pad_oid=None, last_ship_status_send=0.0)


def _mid(v, mx, rg, bits):
    """Centre of the quantisation bucket a decoded value came from, so re-encoding
    reproduces the same raw (a decoded bucket edge can round to the neighbour)."""
    if v is None or v == 0.0:
        return v
    denom = (1 << bits) - 2
    r = round((mx - v) * denom / rg) + 1
    return mx - (r - 0.5) * rg / denom


def _mid3(vec, mx, rg):
    return tuple(_mid(c, mx, rg, 16) for c in vec) if vec else (0.0, 0.0, 0.0)


def _local_record_kwargs(ls, mask, ent):
    rec = dict(entity_id=ent.entity_id, is_manned=True,
               pos=_mid3(ent.position, P.VEC_POS_MAX, P.VEC_POS_RANGE),
               vel=_mid3(ent.velocity, P.VEC_VEL_MAX, P.VEC_VEL_RANGE),
               rot=_mid3(ent.rotation, P.VEC_ROT_MAX, P.VEC_ROT_RANGE),
               include_pos=bool(mask & 2), include_vel=bool(mask & 4), include_rot=bool(mask & 8),
               include_spin=bool(mask & 16),
               spin=_mid3(ent.angular_velocity, P.VEC_SPIN_MAX, P.VEC_SPIN_RANGE),
               include_health=bool(mask & 32), include_fuel=bool(mask & 128),
               speed_scale=_mid(ent.speed, 1.0, 1.0, 10) if ent.speed is not None else 1.0,
               fuel=_mid(ent.fuel, 1.0, 1.0, 10) if ent.fuel is not None else 1.0,
               include_reset=bool(mask & 512))
    return rec


def _rebuild_upstream_vu(hexs):
    from wulfram2_protocol.hud_state import HUD_FRACTION_MAX, HUD_FRACTION_RANGE
    pkt = H(hexs)
    ls, n, eid, mask, ents = raw_record(pkt, UP_Q, UP_B)
    assert n == 1 and eid == 2124 and not (mask & (64 | 256))
    ts, tick = struct.unpack(">II", pkt[1:9])
    return pkt, mask, P.build_view_update_multi(
        tick, include_local_state=True, weapon_id=ls.weapon_id,
        health=_mid(ls.health, 1.0, 1.0, 10), fuel=_mid(ls.energy, 1.0, 1.0, 10),
        ammo_count_bits=2, ammo_count=ls.ammo_mask, primary_turret_bits=8,
        primary_turret_angle=_mid(ls.primary_turret, HUD_FRACTION_MAX, HUD_FRACTION_RANGE, 8),
        timestamp=ts, entities=[_local_record_kwargs(ls, mask, ents[0])])


def test_local_vitals_vu_golden_against_captured_records():
    """Our VU builder reproduces upstream's dirty-bits local records byte for byte
    (masks without the ammo block, whose layout depends on upstream's BEHAVIOR)."""
    done = set()
    for rec in FX["vu_dirty"]:
        pkt = H(rec["hex"])
        _ls, _n, _eid, mask, _e = raw_record(pkt, UP_Q, UP_B)
        if mask & 64:
            continue
        cap, mask, built = _rebuild_upstream_vu(rec["hex"])
        assert built == cap, (mask, built.hex(), rec["hex"])
        done.add(mask)
    assert done == {128, 160, 32}, done


def test_dock_pin_record_golden_against_captured_670():
    """Pin record (bits 1-4 + 7 + 9, vel 0, spin 0) rebuilt byte-exact from capture."""
    rec = [r for r in FX["vu_pin"] if raw_record(H(r["hex"]), UP_Q, UP_B)[3] == 670][0]
    cap, mask, built = _rebuild_upstream_vu(rec["hex"])
    assert mask == 670 and built == cap, built.hex()


def _mask_of(payload):
    return raw_record(payload, OUR_Q, OUR_B)[3]


def test_local_vu_entity_dirty_bits_and_cadence():
    srv = _knob_server(local_vu_entity=True)
    ctx = _knob_ctx()
    vit = srv._vit
    first = srv._build_community_local_vu(ctx, 1000, now=10.0)
    clean_decode(first, OUR_Q, OUR_B)
    assert _mask_of(first) == 32 | 64 | 128  # first record: everything "changed"
    assert srv._build_community_local_vu(ctx, 1001, now=10.02) is None  # < active interval
    assert srv._build_community_local_vu(ctx, 1002, now=10.06) is None  # nothing dirty, < idle
    vit["fuel"] = 0.5
    rec = srv._build_community_local_vu(ctx, 1003, now=10.12)
    assert _mask_of(rec) == 128
    vit["health"] = 0.25
    vit["fire"] = True
    rec = srv._build_community_local_vu(ctx, 1004, now=10.18)
    assert _mask_of(rec) == 32 | 64
    floor = srv._build_community_local_vu(ctx, 1005, now=10.6)  # idle floor, nothing dirty
    assert _mask_of(floor) == 32 | 128
    for payload in (first, rec, floor):
        m = _mask_of(payload)
        assert not (m & 0b11110) and not (m & 512), m  # never motion bits, never bit 9
        _ts, _tick, ls, ents = clean_decode(payload, OUR_Q, OUR_B)
        assert ls is not None and len(ents) == 1 and ents[0].entity_id == 2124
    # The interval hook: while the knob is on the heartbeat is evaluated every active interval.
    srv.update_heartbeat_interval = 0.1
    srv.local_state_cadence = "fixed"
    assert srv._effective_heartbeat_interval(ctx) == 0.05
    srv.local_vu_entity = False
    assert srv._effective_heartbeat_interval(ctx) == 0.1


def test_ua_split_matches_upstream_shape():
    a, b = (H(r["hex"]) for r in FX["ua_split"])
    ta = clean_decode(a, UP_Q, UP_B)
    tb = clean_decode(b, UP_Q, UP_B)
    assert a[1:5] == b[1:5] and len(a) <= 1200 and ta[1] is not None and tb[1] is not None
    # Our splitter on 50 tank records at the upstream budget.
    ents = []
    for i in range(50):
        ents.append(dict(entity_id=3000 + i, is_manned=True, pos=(100.0 + i, 200.0, 50.0),
                         vel=(1.0, -2.0, 0.0), rot=(0.0, 0.1, 0.01 * i), include_pos=True,
                         include_vel=True, include_rot=True, include_spin=True, spin=(0.0, 0.0, 0.2),
                         include_entity_vitals=True, speed_scale=1.0, fuel=1.0,
                         ammo_unit=0, ammo_active_bits=9, ammo_active_mask=i & 1))
    ls = dict(weapon_id=0, health=0.75, fuel=0.5, ammo_count_bits=9, ammo_count=1,
              primary_turret_bits=8, primary_turret_angle=0.0)
    parts = P.build_update_array_split(4242, 1200, include_local_state=True, entities=ents, **ls)
    assert len(parts) >= 2
    seen = []
    for part in parts:
        assert len(part) <= 1200 and part[1:5] == struct.pack(">I", 4242)
        tick, lstate, decoded = clean_decode(part, OUR_Q, OUR_B)
        assert lstate is not None and lstate.ammo_mask == 1
        seen += [e.entity_id for e in decoded]
    assert seen == [e["entity_id"] for e in ents]
    # Greedy: the first datagram could not take the next record.
    n0 = len(clean_decode(parts[0], OUR_Q, OUR_B)[2])
    assert len(P.build_update_array_multi(4242, include_local_state=True, entities=ents[:n0 + 1], **ls)) > 1200
    assert P.build_update_array_split(1, 1200, include_local_state=False, entities=[]) == []


def test_ua_split_wiring_in_remote_updates():
    sent = []
    srv = _knob_server(ua_split_bytes=300, combine_update_arrays=False, send_remote_updates=True,
                       remote_update_mode="pos_vel_rot", remote_yaw_negate=False, remote_yaw_offset=0.0,
                       remote_entity_vitals=True)
    srv._world_viewer_ready = lambda ctx: True
    srv._og_viewer_replication_enabled = lambda ctx, what: True
    srv._browser_demo_local_reconcile_enabled = lambda ctx: False
    srv._get_update_array_local_state_for_viewer = lambda ctx: (False, {})
    srv._send_packet_to_client = lambda ctx, payload, prefer_tcp=True: sent.append(payload) or True
    viewer = _knob_ctx()
    others = []
    for i in range(12):
        o = _knob_ctx()
        o.session.entity_id = 5000 + i
        o.player_pose = {"roll": 0.0, "pitch": 0.0}
        o.angular_vel_yaw = 0.0
        others.append(o)
    viewer.known_entity_ids = {o.session.entity_id for o in others}
    srv._snapshot_in_game_clients = lambda: [viewer] + others
    srv._send_remote_player_updates(viewer, 777, prefer_tcp=False)
    assert len(sent) >= 2 and all(len(p) <= 300 for p in sent)
    assert {p[1:5] for p in sent} == {struct.pack(">I", 777)}
    srv.ua_split_bytes = 0
    sent.clear()
    srv._send_remote_player_updates(viewer, 778, prefer_tcp=False)
    assert len(sent) == 12  # default: one UPDATE_ARRAY per remote entity, as before


def test_docking_encoder_golden_and_stream():
    for rec in FX["docking"]:
        pkt = H(rec["hex"])
        stamp, target, flag = struct.unpack(">IIB", pkt[1:])
        assert P.build_docking(stamp, target, bool(flag)) == pkt
    assert P.build_docking(1, 134403, False).hex() == "380000000100020d0300"
    stamps = [struct.unpack(">I", H(r["hex"])[1:5])[0] for r in FX["docking"]]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert all(45 <= g <= 60 for g in gaps), gaps  # upstream 20 Hz, ms clock
    srv = _knob_server()
    ctx = _knob_ctx()
    ctx.supply_pad_oid = 777
    srv._community_docking_step(ctx, 1.0, 100)
    assert not srv.udp_handler.sent  # knob off: nothing
    srv.docking_stream = True
    for i in range(4):
        srv._community_docking_step(ctx, 1.0 + 0.025 * i, 100 + 25 * i)
    docks = [struct.unpack(">IIB", d[1:]) for d, _ in srv.udp_handler.sent]
    assert docks == [(100, 777, 1), (150, 777, 1)]
    ctx.player_vel = (0.0, 0.0, -8.0)  # airborne: not docked
    srv._community_docking_step(ctx, 1.2, 200)
    assert struct.unpack(">IIB", srv.udp_handler.sent[-1][0][1:]) == (200, 777, 0)
    n = len(srv.udp_handler.sent)
    srv._community_docking_step(ctx, 1.3, 210)
    assert len(srv.udp_handler.sent) == n


def test_ship_status_batch_with_repeat_matches_upstream_cadence():
    times = sorted({r["t"] for r in FX["ship_batch"]})
    assert 0.030 <= times[1] - times[0] <= 0.050 and 0.9 <= times[2] - times[0] <= 1.1
    first = [r["hex"][:2] for r in FX["ship_batch"] if r["t"] == times[0]]
    assert first == ["27", "2d"] * (len(first) // 2)  # SS then SSI, per ship
    srv = _knob_server(ship_status_interval_s=1.0)
    srv._uplink_ships = {1: {"oid": 0x2D, "name": "Crimson Supply Ship"}, 2: {"oid": 0x2E, "name": "Azure Supply Ship"}}
    srv._build_uplink_ship_info_packet = lambda ship: P.build_supply_ship_info(int(ship["oid"]))
    sent = []
    srv._send_packet_to_client = lambda ctx, payload, prefer_tcp=True: sent.append((payload[0], prefer_tcp)) or True
    ctx = _knob_ctx()

    def run(now):
        before = len(sent)
        srv._community_ship_status_step(ctx, now)
        return [op for op, _ in sent[before:]]

    assert run(10.0) == [0x27, 0x2D, 0x27, 0x2D]
    assert run(10.04) == []  # default: no repeat
    srv.ship_status_repeat_s = 0.035
    srv.ship_status_transport = "tcp"
    assert run(11.0) == [0x27, 0x2D, 0x27, 0x2D]
    assert run(11.02) == []
    assert run(11.04) == [0x27, 0x2D, 0x27, 0x2D]
    assert run(11.5) == []
    assert all(tcp for _, tcp in sent[4:])


def test_all_knobs_default_off_is_silent():
    srv = _knob_server(ship_status_interval_s=0.0)
    called = []
    orig = dock_pin.should_pin
    dock_pin.should_pin = lambda inputs: called.append(inputs) or True
    try:
        ctx = _knob_ctx()
        ctx.supply_pad_oid = 9
        for i in range(10):
            srv._community_wire_step(ctx, 1.0 + 0.05 * i, 100 + i)
    finally:
        dock_pin.should_pin = orig
    assert not srv.udp_handler.sent and not called


# ============================================================ 4. dock pin


def test_dock_pin_off_never_calls_policy():
    srv = _knob_server()
    ctx = _knob_ctx()
    for i in range(5):
        assert srv._dock_pin_step(ctx, 1.0 + 0.05 * i, 10 + i) is False  # real should_pin would raise
    srv.dock_pin_enabled = True
    srv.dock_pin_clients = frozenset({"10.0.0.5"})
    try:
        srv._dock_pin_step(ctx, 2.0, 50)
    except NotImplementedError:
        pass  # knob on: the user's (unwritten) policy is consulted
    else:
        raise AssertionError("should_pin was not called with WULFRAM_DOCK_PIN=1")


def test_dock_pin_wiring_latch_emit_release():
    srv = _knob_server(dock_pin_enabled=True, dock_pin_clients=frozenset({"10.0.0.5"}))
    ctx = _knob_ctx()
    seen = []
    decision = {"pin": True}
    orig = dock_pin.should_pin
    dock_pin.should_pin = lambda inputs: seen.append(inputs) or decision["pin"]
    try:
        assert srv._dock_pin_step(ctx, 1.00, 1000) is True
        ctx.player_pos = (100.4, 200.2, 50.0)  # server pose creeps; the pin stays latched
        assert srv._dock_pin_step(ctx, 1.02, 1020) is False  # < interval: one call, no record
        assert srv._dock_pin_step(ctx, 1.05, 1050) is True
        ctx.last_decoded_input = {"fwd": 1.0}
        decision["pin"] = False
        assert srv._dock_pin_step(ctx, 1.10, 1100) is True  # release
        assert srv._dock_pin_step(ctx, 1.15, 1150) is False
    finally:
        dock_pin.should_pin = orig
    assert len(seen) == 5  # once per tick
    assert [s.pinned for s in seen] == [False, True, True, True, False]
    assert seen[0].enabled_for_client and seen[0].grounded and seen[0].divergence_pos == 0.0
    assert math.isclose(seen[2].divergence_pos, math.dist((100.0, 200.0, 50.0), (100.4, 200.2, 50.0)))
    assert seen[3].move_input_active and seen[3].seconds_since_thrust == 0.0
    assert seen[0].seconds_since_thrust == math.inf and seen[0].dock_target == 0
    vus = [d for d, _ in srv.udp_handler.sent if d[0] == 0x0F]
    docks = [struct.unpack(">IIB", d[1:]) for d, _ in srv.udp_handler.sent if d[0] == 0x38]
    assert docks == [(1000, 0, 1), (1050, 0, 1), (1100, 0, 0)]
    assert len(vus) == 3
    for pin in vus[:2]:
        _ts, _tick, ls, ents = clean_decode(pin, OUR_Q, OUR_B)
        m = _mask_of(pin)
        assert m & 0b11110 == 0b11110 and m & 512, m
        e = ents[0]
        assert e.entity_id == 2124 and e.velocity == (0.0, 0.0, 0.0) and e.angular_velocity == (0.0, 0.0, 0.0)
        assert all(abs(a - b) < 0.3 for a, b in zip(e.position, (100.0, 200.0, 50.0)))
    rel = _mask_of(vus[2])
    assert rel & 0b11110 == 0b10100 and not rel & 512, rel  # bits 2+4 only (+dirty), no bit 9
    assert getattr(ctx, "_dock_pin_state", None) is None
    # the vitals VU stays quiet on pin ticks (one local VU per tick)
    srv.local_vu_entity = True
    assert srv._build_community_local_vu(ctx, 1100, now=5.0) is None
    assert srv._build_community_local_vu(ctx, 1150, now=5.0) is not None  # floor resumes


def test_dock_pin_not_on_allowlist_sends_nothing():
    srv = _knob_server(dock_pin_enabled=True, dock_pin_clients=frozenset({"10.9.9.9"}))
    ctx = _knob_ctx()
    seen = []
    orig = dock_pin.should_pin
    dock_pin.should_pin = lambda inputs: seen.append(inputs) or True
    try:
        assert srv._dock_pin_step(ctx, 1.0, 10) is False
    finally:
        dock_pin.should_pin = orig
    assert len(seen) == 1 and seen[0].enabled_for_client is False
    assert not srv.udp_handler.sent


def test_knob_parsing_defaults_off():
    import os
    from wulfram import server_config
    keys = ["WULFRAM_LOCAL_VU_ENTITY", "WULFRAM_UA_SPLIT_BYTES", "WULFRAM_DOCKING_STREAM",
            "WULFRAM_SHIP_STATUS_REPEAT_MS", "WULFRAM_SHIP_STATUS_TRANSPORT", "WULFRAM_DOCK_PIN",
            "WULFRAM_DOCK_PIN_CLIENTS"]
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        s = SimpleNamespace()
        server_config.init_upstream_1016_profile(s)
        assert (s.local_vu_entity, s.ua_split_bytes, s.docking_stream, s.ship_status_repeat_s,
                s.ship_status_transport, s.dock_pin_enabled, s.dock_pin_clients) == (
            False, 0, False, 0.0, "udp", False, frozenset())
        os.environ.update({"WULFRAM_LOCAL_VU_ENTITY": "1", "WULFRAM_UA_SPLIT_BYTES": "1200",
                           "WULFRAM_DOCKING_STREAM": "1", "WULFRAM_SHIP_STATUS_REPEAT_MS": "35",
                           "WULFRAM_SHIP_STATUS_TRANSPORT": "tcp", "WULFRAM_DOCK_PIN": "1",
                           "WULFRAM_DOCK_PIN_CLIENTS": "10.0.0.5, 10.0.0.6"})
        server_config.init_upstream_1016_profile(s)
        assert (s.local_vu_entity, s.ua_split_bytes, s.docking_stream, s.ship_status_transport,
                s.dock_pin_enabled, s.dock_pin_clients) == (
            True, 1200, True, "tcp", True, frozenset({"10.0.0.5", "10.0.0.6"}))
        assert math.isclose(s.ship_status_repeat_s, 0.035)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_packet_names_motd_and_squad():
    assert P.get_packet_name(0x23) == "MOTD"
    assert P.get_packet_name(0x44) == "SQUAD_DEFINITION" and P.get_packet_name(0x45) == "SQUAD_MEMBER"
    assert P.PacketType.REQUEST_START == P.PacketType.MOTD == 0x23


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"{t.__name__}: PASSED")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{t.__name__}: FAILED - {exc!r}")
            failed += 1
    print(f"Results: {len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
