#!/usr/bin/env python3
"""
Tests for handler functions extracted from server.py.
"""

import json
import os
import math
import struct
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from wulfram.handlers import decode_lp_string
from wulfram.handlers import (
    send_initial_game_data,
    handle_spawn_at_point,
    handle_want_updates,
    _send_spawn_points_for_client,
)
from wulfram.client import ClientContext
from wulfram.control import ControlServer, build_input_sync_diagnosis
from wulfram.session import Session, Phase, FEATURES
from wulfram.server import WulframServer, _StaticWorldRayNode
from wulfram.building_collision import BuildingCollisionAssets
from wulfram.world_collision import TerrainContact, TerrainGridCollision, TerrainRaycastHit
from wulfram.weapons import (
    WeaponSystem,
    WeaponType,
    EntityType,
    Projectile,
    build_projectile_spawn_packet,
    build_projectile_update_packet,
)
from wulfram.packets import (
    build_chat_message,
    build_behavior_packet,
    build_player_info,
    build_translation_packet,
    build_update_array_create_tank,
    build_update_array_heartbeat,
    build_world_stats,
)
from wulfram2_protocol.codec import BitReader
from client.wulfram_client.network.behavior import parse_behavior
from client.wulfram_client.network.decoder import decode_update_array, decode_view_update, decode_tank_packet
from client.wulfram_client.network.quantizer import parse_translation
from client.wulfram_client.data.models import CBSPTree, CBSPTreeNode, Vec3
from client.wulfram_client.simulation.collision import (
    segment_hits_cbsp_tree,
    segment_raycast_cbsp_tree,
)


def test_decode_lp_string_basic():
    """Test decoding a length-prefixed string."""
    # Length = 6, string = "Hello\x00"
    data = b'\x00\x06Hello\x00'
    text, offset = decode_lp_string(data, 0)
    assert text == "Hello"
    assert offset == 8
    print("test_decode_lp_string_basic: PASSED")
    return True


def test_decode_lp_string_offset():
    """Test decoding with non-zero offset."""
    data = b'PREFIX\x00\x05Test\x00'
    text, offset = decode_lp_string(data, 6)
    assert text == "Test"
    assert offset == 13
    print("test_decode_lp_string_offset: PASSED")
    return True


def test_decode_lp_string_empty():
    """Test decoding empty string."""
    data = b'\x00\x01\x00'  # Length 1, just null terminator
    text, offset = decode_lp_string(data, 0)
    assert text == ""
    assert offset == 3
    print("test_decode_lp_string_empty: PASSED")
    return True


def test_decode_lp_string_truncated():
    """Test handling truncated data."""
    data = b'\x00'  # Only 1 byte, need 2 for length
    text, offset = decode_lp_string(data, 0)
    assert text == ""
    assert offset == 0
    print("test_decode_lp_string_truncated: PASSED")
    return True


def test_handlers_import():
    """Test that all handler functions can be imported."""
    from wulfram.handlers import (
        handle_hello,
        handle_login_request,
        send_initial_game_data,
        handle_bps,
        handle_want_updates,
        handle_reincarnate_tcp,
        handle_udp_d_handshake,
        handle_udp_chat,
        handle_udp_reincarnate,
        handle_team_switch,
        handle_spawn_at_point,
        send_udp_ack,
    )
    print("test_handlers_import: PASSED")
    return True


def test_input_sync_diagnosis_distinguishes_idle_snapback_from_correction_failure():
    """Live telemetry should call out idle controls while corrections are active."""
    diagnosis = build_input_sync_diagnosis(
        phase="IN_GAME",
        last_input={"fwd": 0.0, "strafe": 0.0, "turn": 0.0, "thrust": 0.824},
        last_action_age_s=0.1,
        last_nonzero_move_input_age_s=90.0,
        last_position_change_age_s=80.0,
        last_state_request_age_s=0.2,
        last_state_sync_reply_age_s=0.3,
        state_requests=100,
        state_sync_replies=99,
        state_sync_view_replies=99,
    )

    assert diagnosis["status"] == "idle_input_authoritative_snapback"
    assert diagnosis["corrections_active"] is True
    assert diagnosis["drive_idle"] is True
    assert diagnosis["movement_input_recent"] is False
    print("test_input_sync_diagnosis_distinguishes_idle_snapback_from_correction_failure: PASSED")
    return True


def test_input_sync_diagnosis_reports_movement_without_targeted_corrections():
    """Movement packets without reply traffic should stay visible as sync debt."""
    diagnosis = build_input_sync_diagnosis(
        phase="IN_GAME",
        last_input={"fwd": 1.0, "strafe": 0.0, "turn": 0.0},
        last_action_age_s=0.1,
        last_nonzero_move_input_age_s=0.1,
        last_position_change_age_s=0.2,
        last_state_request_age_s=30.0,
        last_state_sync_reply_age_s=None,
        state_requests=0,
        state_sync_replies=0,
        state_sync_view_replies=0,
    )

    assert diagnosis["status"] == "movement_without_targeted_corrections"
    assert diagnosis["corrections_active"] is False
    assert diagnosis["movement_input_recent"] is True
    print("test_input_sync_diagnosis_reports_movement_without_targeted_corrections: PASSED")
    return True


def test_spawn_override_wins_over_map_spawn_points():
    """Configured flat default spawn should win over map spawn pads."""
    old_spawn_pos = os.environ.get("WULFRAM_SPAWN_POS")
    try:
        os.environ["WULFRAM_SPAWN_POS"] = "4950,5100,5"
        server = WulframServer.__new__(WulframServer)
        server.map_name = "crossroads"
        server.up_axis = "z"
        server.spawn_height = 5.0
        server.use_map_spawn_points = True
        server.force_default_spawn_pos = True
        server.default_flat_spawn_pos = (4950.0, 5100.0, 5.0)
        server.get_spawn_points = lambda: [
            {"oid": 7001, "team": 2, "x": 6000.0, "y": 6000.0, "z": 90.0},
        ]

        pos = server._resolve_spawn_pos(2)
        assert pos == (4950.0, 5100.0, 5.0), pos
        print("test_spawn_override_wins_over_map_spawn_points: PASSED")
        return True
    finally:
        if old_spawn_pos is None:
            os.environ.pop("WULFRAM_SPAWN_POS", None)
        else:
            os.environ["WULFRAM_SPAWN_POS"] = old_spawn_pos


def test_spawn_at_point_honors_clicked_pad_when_default_configured():
    """Explicit spawn-point packets should honor the clicked pad."""
    old_spawn_pos = os.environ.get("WULFRAM_SPAWN_POS")
    try:
        os.environ["WULFRAM_SPAWN_POS"] = "4950,5100,5"
        server = WulframServer.__new__(WulframServer)
        server.map_name = "crossroads"
        server.up_axis = "z"
        server.spawn_height = 5.0
        server.use_map_spawn_points = True
        server.force_default_spawn_pos = True
        server.default_flat_spawn_pos = (4950.0, 5100.0, 5.0)
        server.spawn_allow_point_override = True
        server.spawn_point_override_min_interval = 0.0
        server.get_spawn_points = lambda: [
            {"oid": 7001, "team": 2, "x": 6000.0, "y": 6000.0, "z": 90.0},
        ]
        captured = {}
        server._spawn_wf_style = lambda ctx, team_id, pos=None, **kwargs: captured.update(
            team_id=team_id,
            pos=pos,
        )

        session = Session()
        ctx = ClientContext(
            client_id=1,
            client_addr=("127.0.0.1", 50000),
            session=session,
            entity_id=0x14EA,
        )
        handle_spawn_at_point(server, ctx, 7001, 0, ("127.0.0.1", 50000))

        assert captured["team_id"] == 2, captured
        assert captured["pos"] == (6000.0, 6000.0, 90.0), captured
        print("test_spawn_at_point_honors_clicked_pad_when_default_configured: PASSED")
        return True
    finally:
        if old_spawn_pos is None:
            os.environ.pop("WULFRAM_SPAWN_POS", None)
        else:
            os.environ["WULFRAM_SPAWN_POS"] = old_spawn_pos


def test_map_entity_z_aligns_buried_entities_to_terrain():
    """Map-state buildings/spawn points below terrain should be lifted to ground."""
    server = WulframServer.__new__(WulframServer)
    server.up_axis = "z"
    server.terrain_height_offset = 5.0
    server.terrain = SimpleNamespace(get_height=lambda x, y: 60.0)

    z, ground_z, aligned = server._align_map_entity_z_to_terrain(2578.7, 3040.0, 63.72)

    assert aligned is True
    assert ground_z == 65.0
    assert z == 65.0
    print("test_map_entity_z_aligns_buried_entities_to_terrain: PASSED")
    return True


def test_map_entity_z_preserves_elevated_entities():
    """Terrain alignment should not pull already-elevated map entries down."""
    server = WulframServer.__new__(WulframServer)
    server.up_axis = "z"
    server.terrain_height_offset = 5.0
    server.terrain = SimpleNamespace(get_height=lambda x, y: 0.0)

    z, ground_z, aligned = server._align_map_entity_z_to_terrain(5150.1, 5241.3, 7.7)

    assert aligned is False
    assert ground_z == 5.0
    assert z == 7.7
    print("test_map_entity_z_preserves_elevated_entities: PASSED")
    return True


def test_control_pose_reset_updates_ground_override():
    """Control-plane exact pose resets should move the local ground clamp too."""
    control = ControlServer.__new__(ControlServer)
    server = SimpleNamespace(
        spawn_sets_ground_level=True,
        up_axis="z",
        _get_network_tick=lambda ctx: 123,
    )
    control.server = server
    ctx = ClientContext(
        client_id=1,
        client_addr=("127.0.0.1", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.ground_level_override = 65.0

    control._apply_exact_client_pose(ctx, (2578.7, 3040.0, 63.7244))

    assert ctx.player_pos == (2578.7, 3040.0, 63.7244)
    assert ctx.ground_level_override == 63.7244
    assert ctx.player_vel == (0.0, 0.0, 0.0)
    print("test_control_pose_reset_updates_ground_override: PASSED")
    return True


def test_pulse_shell_default_spawn_uses_recovered_muzzle_offset():
    """Default pulse origin should stay near the tank, not raw shape-hardpoint scale."""
    keys = [
        "WULFRAM_PROJECTILE_SPAWN_MODE",
        "WULFRAM_PROJECTILE_SPAWN_OFFSET",
        "WULFRAM_PROJECTILE_BARREL_RIGHT",
        "WULFRAM_PROJECTILE_BARREL_UP",
    ]
    old_env = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)

        ws = WeaponSystem()
        ws.player_pos = (100.0, 200.0, 10.0)
        ws.player_rot = (0.0, 0.0, 0.0)
        ws.player_team = 2
        ws.player_id = 1337

        proj = ws._fire_pulse_cannon()

        assert proj is not None
        assert proj.pos == (105.5, 199.75, 11.25), proj.pos
        assert proj.vel == (75.0, 0.0, -0.0), proj.vel
        print("test_pulse_shell_default_spawn_uses_recovered_muzzle_offset: PASSED")
        return True
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_remote_spawn_points_use_udp_not_tcp():
    """Remote spawn-point UPDATE_ARRAY should avoid the TCP stream."""
    old_transport = os.environ.get("WULFRAM_SPAWN_POINTS_TRANSPORT")

    class DummyTCP:
        def __init__(self):
            self.sent = []
            self.sock = SimpleNamespace(fileno=lambda: 1)

        def send(self, payload):
            self.sent.append(payload)

    class DummyUDP:
        def __init__(self):
            self.sent = []

        def send_to(self, payload, addr):
            self.sent.append((payload, addr))

    try:
        os.environ["WULFRAM_SPAWN_POINTS_TRANSPORT"] = "tcp"
        session = Session()
        session.translation_ack_received = True
        session.udp_addr = ("10.10.10.2", 62588)
        ctx = ClientContext(
            client_id=7,
            client_addr=("10.10.10.2", 50000),
            session=session,
            entity_id=1337,
        )
        ctx.tcp_handler = DummyTCP()
        udp = DummyUDP()
        server = SimpleNamespace(
            get_spawn_points=lambda: [{"oid": 7001, "team": 2, "x": 4950.0, "y": 5100.0, "z": 5.0}],
            _to_client_pos=lambda pos: pos,
            udp_handler=udp,
        )

        _send_spawn_points_for_client(server, ctx)

        assert ctx.tcp_handler.sent == []
        assert len(udp.sent) == 1
        assert udp.sent[0][1] == ("10.10.10.2", 62588)
        print("test_remote_spawn_points_use_udp_not_tcp: PASSED")
        return True
    finally:
        if old_transport is None:
            os.environ.pop("WULFRAM_SPAWN_POINTS_TRANSPORT", None)
        else:
            os.environ["WULFRAM_SPAWN_POINTS_TRANSPORT"] = old_transport


def test_remote_want_updates_suppresses_empty_tcp_update_array():
    """Remote WANT_UPDATES bootstrap should not inject an empty TCP UPDATE_ARRAY."""

    class DummyTCP:
        def __init__(self):
            self.sent = []
            self.sock = SimpleNamespace(fileno=lambda: 1)

        def send(self, payload):
            self.sent.append(payload)

    session = Session()
    session.connected_at = 0.0
    ctx = ClientContext(
        client_id=7,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=1337,
    )
    ctx.tcp_handler = DummyTCP()
    server = SimpleNamespace(
        _start_ping_loop=lambda ctx: None,
        udp_handler=None,
    )

    handle_want_updates(server, ctx, b"\x39\x00\x00\x00\x01")

    empty_update = b"\x00\x09\x0e\x00\x00\x00\x00\x00\x00"
    assert empty_update not in ctx.tcp_handler.sent, ctx.tcp_handler.sent
    assert len(ctx.tcp_handler.sent) >= 2, len(ctx.tcp_handler.sent)
    print("test_remote_want_updates_suppresses_empty_tcp_update_array: PASSED")
    return True


def test_remote_spawn_create_update_array_avoids_tcp():
    """Remote spawn pre-creation UPDATE_ARRAY should stay off TCP when UDP is ready."""

    class DummyTCP:
        def __init__(self):
            self.sent = []

        def send(self, payload, log=True):
            self.sent.append(payload)

    class DummyUDP:
        def __init__(self):
            self.sent = []

        def send_to(self, payload, addr):
            self.sent.append((payload, addr))

    session = Session()
    session.udp_addr = ("10.10.10.2", 62588)
    ctx = ClientContext(
        client_id=7,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=1337,
    )
    ctx.tcp_handler = DummyTCP()

    server = WulframServer.__new__(WulframServer)
    server.udp_handler = DummyUDP()

    payload = build_update_array_create_tank(
        tick=0x12345678,
        entity_id=1337,
        entity_type=0,
        team=2,
        pos=(4950.0, 5100.0, 5.0),
        behavior_type=0,
        include_health=False,
        include_entity_vitals=False,
        is_manned=True,
        weapon_id=2,
    )

    assert server._send_spawn_create_update_array(ctx, payload) is True
    assert ctx.tcp_handler.sent == []
    assert len(server.udp_handler.sent) == 1
    assert server.udp_handler.sent[0][1] == ("10.10.10.2", 62588)
    print("test_remote_spawn_create_update_array_avoids_tcp: PASSED")
    return True


def test_send_initial_game_data_og_bootstrap_order():
    """Remote OG bootstrap should stop at the team-select-safe packet set."""
    old_mode = os.environ.get("WULFRAM_LOGIN_BOOTSTRAP")
    os.environ["WULFRAM_LOGIN_BOOTSTRAP"] = "og"

    class DummyTCP:
        def __init__(self):
            self.sent = []

        def send(self, payload):
            self.sent.append(payload)

    try:
        session = Session()
        session.username = "probe"
        ctx = SimpleNamespace(
            client_id=7,
            client_addr=("10.10.10.2", 50000),
            entity_id=1337,
            session=session,
            tcp_handler=DummyTCP(),
        )
        server = SimpleNamespace(
            build_world_stats_packet=lambda: build_world_stats("crossroads", 1, 1, 1.0),
        )

        send_initial_game_data(server, ctx)

        opcodes = [payload[0] for payload in ctx.tcp_handler.sent]
        assert opcodes == [0x28, 0x22, 0x17], opcodes
        assert session.player_id == 1337
        assert ctx.tcp_handler.sent[-1][-1] == 0x01  # spectator
        assert session.behavior_sent is False
        assert session.translation_sent is False
        assert session.roster_sent is False
        assert session.world_stats_sent is False
        print("test_send_initial_game_data_og_bootstrap_order: PASSED")
        return True
    finally:
        if old_mode is None:
            os.environ.pop("WULFRAM_LOGIN_BOOTSTRAP", None)
        else:
            os.environ["WULFRAM_LOGIN_BOOTSTRAP"] = old_mode


def test_build_chat_message_comm_layout():
    """COMM_MESSAGE should match the decompile-backed sender/target order."""
    payload = build_chat_message("test", source_id=0x11223344, target_id=0x55667788)
    assert payload[0] == 0x1F
    assert payload[1:3] == b"\x00\x00"  # sender_mode
    assert payload[3:7] == b"\x11\x22\x33\x44"
    assert payload[7:9] == b"\x00\x00"  # target_mode
    assert payload[9:13] == b"\x55\x66\x77\x88"
    assert payload[13:15] == b"\x00\x05"
    assert payload[15:] == b"test\x00"
    print("test_build_chat_message_comm_layout: PASSED")
    return True


def test_build_update_array_remote_heartbeat_shape():
    """Low-level heartbeat builder still supports the legacy single-entity stub shape."""
    payload = build_update_array_heartbeat(
        tick=0x12345678,
        entity_id=0x14EA,
        include_health=True,
        weapon_id=0,
        health=1.0,
        fuel=1.0,
        ammo_count_bits=9,
        ammo_count=0,
        primary_turret_bits=16,
        primary_turret_angle=1.234,
        include_entities=True,
        use_local_entity_when_no_transform=True,
    )
    assert payload[0] == 0x0E
    tick, local_state, entities = decode_update_array(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None
    assert local_state.weapon_id == 0
    assert len(entities) == 1
    assert entities[0].entity_id == 0x14EA
    print("test_build_update_array_remote_heartbeat_shape: PASSED")
    return True


def test_server_remote_heartbeat_helper_keeps_full_local_state():
    """Promoted remote OG heartbeats should keep local-state but avoid forced position snaps."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.heartbeat_view_update = False
    server.heartbeat_include_rot = True
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server.up_axis = "z"
    server.pos_offset = 0.0

    session = Session()
    session.translation_ack_received = True
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = True
    ctx.player_aim_yaw = 1.234
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (1.0, 2.0, 3.0)
    ctx.player_pose = {"roll": 0.25}
    ctx.player_heading = 1.234
    server._to_client_pos = lambda pos: pos

    payload = server._build_local_state_heartbeat(
        ctx,
        tick=0x12345678,
        entity_id=0x14EA,
        include_health=True,
        health=1.0,
        fuel=1.0,
    )

    tick, local_state, entities = decode_update_array(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert len(entities) == 1
    assert entities[0].entity_id == 0x14EA
    assert entities[0].position is None
    assert entities[0].velocity is None
    assert entities[0].rotation is not None
    assert entities[0].angular_velocity is not None
    print("test_server_remote_heartbeat_helper_keeps_full_local_state: PASSED")
    return True


def test_server_remote_heartbeat_helper_pre_state_request_is_spawn_safe():
    """Remote OG pre-sync heartbeats must stay on the 10-byte spawn-safe form."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.heartbeat_view_update = False
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server.up_axis = "z"
    server.pos_offset = 0.0

    session = Session()
    session.translation_ack_received = True
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = False

    payload = server._build_local_state_heartbeat(
        ctx,
        tick=0x12345678,
        entity_id=0x14EA,
        include_health=True,
        health=1.0,
        fuel=1.0,
    )

    assert len(payload) == 10
    tick, local_state, entities = decode_update_array(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert len(entities) == 0
    print("test_server_remote_heartbeat_helper_pre_state_request_is_spawn_safe: PASSED")
    return True


def test_remote_state_sync_reply_uses_safe_local_player_shape_when_ready():
    """Stable remote STATE_REQUEST replies must keep the short-form-safe local-state shape."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
    server.remote_full_local_state_delay = 2.0
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678
    server.debug_viewpoint = False
    server.debug_udp_raw = False
    server.pktlog = SimpleNamespace(enabled=False)
    captured = []
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: captured.append((payload, addr)))

    session = Session()
    session.translation_ack_received = True
    session.in_game = True
    session.entity_id = 0x14EA
    session.udp_addr = ("10.10.10.2", 50000)
    session.last_spawn_time = time.monotonic() - 5.0
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = True
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_pose = {"roll": 0.0}
    ctx.player_heading = 0.0
    ctx.player_yaw = 0.0
    ctx.last_state_sync_send = 0.0

    server._send_state_sync_snapshot(ctx, include_view_update=False, reason="state_request")

    assert len(captured) == 1, captured
    payload, addr = captured[0]
    assert addr == ("10.10.10.2", 50000)
    assert len(payload) == 43, len(payload)
    tick, local_state, entities = decode_update_array(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert local_state.ammo_mask == 0
    assert len(entities) == 1, entities
    assert entities[0].entity_id == 0x14EA
    assert entities[0].position is not None
    assert entities[0].velocity is not None
    assert entities[0].rotation is not None
    assert entities[0].angular_velocity is not None
    print("test_remote_state_sync_reply_uses_safe_local_player_shape_when_ready: PASSED")
    return True


def test_remote_state_sync_reply_stays_spawn_safe_immediately_after_spawn():
    """Fresh post-spawn remote STATE_REQUEST replies must stay on the safe local-state shape."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
    server.remote_full_local_state_delay = 2.0
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678
    server.debug_viewpoint = False
    server.debug_udp_raw = False
    server.pktlog = SimpleNamespace(enabled=False)
    captured = []
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: captured.append((payload, addr)))

    session = Session()
    session.translation_ack_received = True
    session.in_game = True
    session.entity_id = 0x14EA
    session.udp_addr = ("10.10.10.2", 50000)
    session.last_spawn_time = time.monotonic()
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = True
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_pose = {"roll": 0.0}
    ctx.player_heading = 0.0
    ctx.player_yaw = 0.0
    ctx.last_state_sync_send = 0.0

    server._send_state_sync_snapshot(ctx, include_view_update=True, replay_timestamp=0x89ABCDEF, reason="test")

    assert len(captured) == 2, captured
    update_payload = captured[0][0]
    view_payload = captured[1][0]
    assert len(update_payload) >= 37, len(update_payload)
    assert len(view_payload) == 41, len(view_payload)
    _, local_state, entities = decode_update_array(
        update_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    timestamp, view_tick, view_local_state, view_entities = decode_view_update(
        view_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert local_state is not None and local_state.weapon_id == 2
    assert view_local_state is not None and view_local_state.weapon_id == 2
    assert entities[0].position is not None
    assert view_entities[0].position is not None
    assert timestamp == 0x89ABCDEF
    assert view_tick == 0x12345678
    print("test_remote_state_sync_reply_stays_spawn_safe_immediately_after_spawn: PASSED")
    return True


def test_remote_state_sync_reply_stays_spawn_safe_after_spawn_delay():
    """Delayed remote STATE_REQUEST replies still keep the spawn-safe local-state shape."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
    server.remote_full_local_state_delay = 2.0
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678
    server.debug_viewpoint = False
    server.debug_udp_raw = False
    server.pktlog = SimpleNamespace(enabled=False)
    captured = []
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: captured.append((payload, addr)))

    session = Session()
    session.translation_ack_received = True
    session.in_game = True
    session.entity_id = 0x14EA
    session.udp_addr = ("10.10.10.2", 50000)
    session.last_spawn_time = time.monotonic() - 5.0
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = False
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_pose = {"roll": 0.0, "pitch": 0.0}
    ctx.player_heading = 0.0
    ctx.player_yaw = 0.0
    ctx.last_action_dump_time = session.last_spawn_time + 1.0
    ctx.last_state_sync_send = 0.0

    server._send_state_sync_snapshot(
        ctx,
        include_view_update=True,
        replay_timestamp=0x89ABCDEF,
        reason="state_request",
    )

    assert len(captured) == 2, captured
    update_payload = captured[0][0]
    view_payload = captured[1][0]
    assert len(update_payload) >= 37, len(update_payload)
    assert len(view_payload) == 41, len(view_payload)

    _, update_local_state, update_entities = decode_update_array(
        update_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    timestamp, view_tick, view_local_state, view_entities = decode_view_update(
        view_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert update_local_state is not None and update_local_state.weapon_id == 2
    assert view_local_state is not None and view_local_state.weapon_id == 2
    assert update_entities[0].position is not None
    assert update_entities[0].velocity is not None
    assert update_entities[0].rotation is not None
    assert view_entities[0].position is not None
    assert view_entities[0].velocity is not None
    assert view_entities[0].rotation is not None
    assert timestamp == 0x89ABCDEF
    assert view_tick == 0x12345678
    print("test_remote_state_sync_reply_stays_spawn_safe_after_spawn_delay: PASSED")
    return True


def test_remote_state_sync_reply_stays_safe_without_post_spawn_input_after_delay():
    """Delayed remote STATE_REQUEST replies should stay safe until real gameplay input arrives."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
    server.remote_full_local_state_delay = 2.0
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678
    server.debug_viewpoint = False
    server.debug_udp_raw = False
    server.pktlog = SimpleNamespace(enabled=False)
    captured = []
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: captured.append((payload, addr)))

    session = Session()
    session.translation_ack_received = True
    session.in_game = True
    session.entity_id = 0x14EA
    session.udp_addr = ("10.10.10.2", 50000)
    session.last_spawn_time = time.monotonic() - 5.0
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = False
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_pose = {"roll": 0.0, "pitch": 0.0}
    ctx.player_heading = 0.0
    ctx.player_yaw = 0.0
    ctx.last_action_dump_time = session.last_spawn_time
    ctx.last_state_sync_send = 0.0

    server._send_state_sync_snapshot(
        ctx,
        include_view_update=True,
        replay_timestamp=0x89ABCDEF,
        reason="state_request",
    )

    assert len(captured) == 2, captured
    update_payload = captured[0][0]
    view_payload = captured[1][0]
    assert len(update_payload) == 37, len(update_payload)
    assert len(view_payload) == 41, len(view_payload)

    _, update_local_state, _ = decode_update_array(
        update_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    _, _, view_local_state, _ = decode_view_update(
        view_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert update_local_state is not None and update_local_state.weapon_id == 2
    assert view_local_state is not None and view_local_state.weapon_id == 2
    print("test_remote_state_sync_reply_stays_safe_without_post_spawn_input_after_delay: PASSED")
    return True


def test_remote_state_sync_reply_emits_view_update_with_request_timestamp():
    """Remote OG STATE_REQUEST replies should preserve the replay/request timestamp."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
    server.remote_full_local_state_delay = 2.0
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678
    server.debug_viewpoint = False
    server.debug_udp_raw = False
    server.pktlog = SimpleNamespace(enabled=False)
    captured = []
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: captured.append((payload, addr)))

    session = Session()
    session.translation_ack_received = True
    session.in_game = True
    session.entity_id = 0x14EA
    session.udp_addr = ("10.10.10.2", 50000)
    session.last_spawn_time = time.monotonic() - 5.0
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = True
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_pose = {"roll": 0.0}
    ctx.player_heading = 0.0
    ctx.player_yaw = 0.125
    ctx.last_state_sync_send = 0.0

    server._send_state_sync_snapshot(
        ctx,
        include_view_update=True,
        replay_timestamp=0x89ABCDEF,
        reason="test",
    )

    assert len(captured) == 2, captured
    update_payload, update_addr = captured[0]
    view_payload, view_addr = captured[1]
    assert update_addr == ("10.10.10.2", 50000)
    assert view_addr == ("10.10.10.2", 50000)
    assert update_payload[0] == 0x0E
    assert view_payload[0] == 0x0F

    tick, local_state, entities = decode_update_array(
        update_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert len(entities) == 1

    timestamp, view_tick, view_local_state, view_entities = decode_view_update(
        view_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert timestamp == 0x89ABCDEF
    assert view_tick == 0x12345678
    assert view_local_state is not None
    assert view_local_state.weapon_id == 2
    assert len(view_entities) == 1
    assert view_entities[0].entity_id == 0x14EA
    assert view_entities[0].position is not None
    assert view_entities[0].velocity is not None
    assert view_entities[0].rotation is not None
    assert view_entities[0].rotation == entities[0].rotation
    assert ctx.last_state_sync_update_len == len(update_payload)
    assert ctx.last_state_sync_view_len == len(view_payload)
    assert ctx.last_state_sync_update_has_local_state is True
    assert ctx.last_state_sync_view_has_local_state is True
    assert ctx.last_state_sync_view_timestamp == 0x89ABCDEF
    assert ctx.last_state_sync_reason == "test"
    assert ctx.last_state_sync_update_hex == update_payload[:32].hex()
    assert ctx.last_state_sync_view_hex == view_payload[:32].hex()
    print("test_remote_state_sync_reply_emits_view_update_with_request_timestamp: PASSED")
    return True


def test_loopback_state_sync_reply_keeps_request_timestamp():
    """Loopback/Python STATE_REQUEST replies keep request-id timestamps for latency correlation."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
    server.remote_full_local_state_delay = 2.0
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678
    server.debug_viewpoint = False
    server.debug_udp_raw = False
    server.pktlog = SimpleNamespace(enabled=False)
    captured = []
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: captured.append((payload, addr)))

    session = Session()
    session.translation_ack_received = True
    session.in_game = True
    session.entity_id = 0x14EA
    session.udp_addr = ("127.0.0.1", 50000)
    session.last_spawn_time = time.monotonic() - 5.0
    ctx = ClientContext(
        client_id=1,
        client_addr=("127.0.0.1", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_pose = {"roll": 0.0, "pitch": 0.0}
    ctx.player_heading = 0.0
    ctx.player_yaw = 0.125
    ctx.last_state_sync_send = 0.0

    server._send_state_sync_snapshot(
        ctx,
        include_view_update=True,
        replay_timestamp=0x89ABCDEF,
        reason="test",
    )

    assert len(captured) == 2, captured
    view_payload, view_addr = captured[1]
    assert view_addr == ("127.0.0.1", 50000)
    assert view_payload[0] == 0x0F
    timestamp, view_tick, _view_local_state, view_entities = decode_view_update(
        view_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert timestamp == 0x89ABCDEF
    assert view_tick == 0x12345678
    assert len(view_entities) == 1
    assert view_entities[0].entity_id == 0x14EA
    assert view_entities[0].position is not None
    print("test_loopback_state_sync_reply_keeps_request_timestamp: PASSED")
    return True


def test_remote_state_sync_reply_uses_request_aligned_authoritative_pose():
    """STATE_REQUEST replies should use the cached authoritative pose nearest the replay tick."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
    server.remote_full_local_state_delay = 2.0
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678
    server.debug_viewpoint = False
    server.debug_udp_raw = False
    server.pktlog = SimpleNamespace(enabled=False)
    captured = []
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: captured.append((payload, addr)))

    session = Session()
    session.translation_ack_received = True
    session.in_game = True
    session.entity_id = 0x14EA
    session.udp_addr = ("10.10.10.2", 50000)
    session.last_spawn_time = time.monotonic() - 5.0
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = False
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (9.0, 9.0, 0.0)
    ctx.player_pose = {"roll": 0.0, "pitch": 0.0}
    ctx.player_heading = 1.25
    ctx.player_yaw = -1.25
    ctx.last_state_sync_send = 0.0
    ctx.authoritative_state_history.append({
        "tick": 0x89ABCDE0,
        "time": time.monotonic(),
        "pos": (4900.0, 5000.0, 5.0),
        "vel": (3.0, 0.0, 0.0),
        "rot": (0.0, 0.0, 0.5),
    })

    server._send_state_sync_snapshot(
        ctx,
        include_view_update=True,
        replay_timestamp=0x89ABCDEF,
        reason="state_request",
    )

    assert len(captured) == 2, captured
    update_payload = captured[0][0]
    view_payload = captured[1][0]

    tick, update_local_state, update_entities = decode_update_array(
        update_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    timestamp, view_tick, view_local_state, view_entities = decode_view_update(
        view_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )

    assert tick == 0x12345678
    assert timestamp == 0x89ABCDEF
    assert view_tick == 0x12345678
    assert update_local_state is not None and update_local_state.weapon_id == 2
    assert view_local_state is not None and view_local_state.weapon_id == 2
    assert all(abs(a - b) < 0.25 for a, b in zip(update_entities[0].position, (4900.0, 5000.0, 5.0)))
    assert all(abs(a - b) < 0.05 for a, b in zip(update_entities[0].velocity, (3.0, 0.0, 0.0)))
    assert all(abs(a - b) < 0.001 for a, b in zip(update_entities[0].rotation, (0.0, 0.0, 0.5)))
    assert all(abs(a - b) < 0.25 for a, b in zip(view_entities[0].position, (4900.0, 5000.0, 5.0)))
    assert all(abs(a - b) < 0.05 for a, b in zip(view_entities[0].velocity, (3.0, 0.0, 0.0)))
    assert all(abs(a - b) < 0.001 for a, b in zip(view_entities[0].rotation, (0.0, 0.0, 0.5)))
    assert update_entities[0].position == view_entities[0].position
    assert update_entities[0].velocity == view_entities[0].velocity
    assert update_entities[0].rotation == view_entities[0].rotation
    print("test_remote_state_sync_reply_uses_request_aligned_authoritative_pose: PASSED")
    return True


def test_remote_state_sync_reply_remaps_client_tick_to_server_history():
    """STATE_REQUEST replay alignment should use ctx.tick_offset when history is server-domain."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
    server.remote_full_local_state_delay = 2.0
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server.use_client_ticks = False
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678
    server.debug_viewpoint = False
    server.debug_udp_raw = False
    server.pktlog = SimpleNamespace(enabled=False)
    captured = []
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: captured.append((payload, addr)))

    session = Session()
    session.translation_ack_received = True
    session.in_game = True
    session.entity_id = 0x14EA
    session.udp_addr = ("10.10.10.2", 50000)
    session.last_spawn_time = time.monotonic() - 5.0
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = False
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (9.0, 9.0, 0.0)
    ctx.player_pose = {"roll": 0.0, "pitch": 0.0}
    ctx.player_heading = 1.25
    ctx.player_yaw = -1.25
    ctx.last_state_sync_send = 0.0
    ctx.tick_offset = 0x002B61E0 - 0x0002D18E
    ctx.last_client_tick = 0x002B61E0
    ctx.authoritative_state_history.append({
        "tick": 0x0002D180,
        "time": time.monotonic(),
        "pos": (4900.0, 5000.0, 5.0),
        "vel": (3.0, 0.0, 0.0),
        "rot": (0.0, 0.0, 0.5),
    })

    server._send_state_sync_snapshot(
        ctx,
        include_view_update=True,
        replay_timestamp=0x002B61E0,
        reason="state_request",
    )

    assert len(captured) == 2, captured
    update_payload = captured[0][0]
    view_payload = captured[1][0]

    tick, update_local_state, update_entities = decode_update_array(
        update_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    timestamp, view_tick, view_local_state, view_entities = decode_view_update(
        view_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )

    assert tick == 0x12345678
    assert timestamp == 0x002B61E0
    assert view_tick == 0x12345678
    assert update_local_state is not None and update_local_state.weapon_id == 2
    assert view_local_state is not None and view_local_state.weapon_id == 2
    assert all(abs(a - b) < 0.25 for a, b in zip(update_entities[0].position, (4900.0, 5000.0, 5.0)))
    assert all(abs(a - b) < 0.05 for a, b in zip(update_entities[0].velocity, (3.0, 0.0, 0.0)))
    assert all(abs(a - b) < 0.001 for a, b in zip(update_entities[0].rotation, (0.0, 0.0, 0.5)))
    assert all(abs(a - b) < 0.25 for a, b in zip(view_entities[0].position, (4900.0, 5000.0, 5.0)))
    assert all(abs(a - b) < 0.05 for a, b in zip(view_entities[0].velocity, (3.0, 0.0, 0.0)))
    assert all(abs(a - b) < 0.001 for a, b in zip(view_entities[0].rotation, (0.0, 0.0, 0.5)))
    assert update_entities[0].position == view_entities[0].position
    assert update_entities[0].velocity == view_entities[0].velocity
    assert update_entities[0].rotation == view_entities[0].rotation
    print("test_remote_state_sync_reply_remaps_client_tick_to_server_history: PASSED")
    return True


def test_remote_state_sync_reuses_cached_sample_when_replay_window_misses():
    """Remote OG replies should not pair an old replay id with the live/current pose."""
    server = WulframServer.__new__(WulframServer)
    server.use_client_ticks = False

    session = Session()
    session.in_game = True
    session.entity_id = 0x14EA
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    old_sample = {
        "tick": 0x00001000,
        "time": time.monotonic() - 1.0,
        "pos": (100.0, 200.0, 5.0),
        "vel": (1.0, 0.0, 0.0),
        "rot": (0.0, 0.0, 0.0),
    }
    newer_sample = {
        "tick": 0x00002000,
        "time": time.monotonic(),
        "pos": (300.0, 400.0, 5.0),
        "vel": (2.0, 0.0, 0.0),
        "rot": (0.0, 0.0, 0.1),
    }
    ctx.authoritative_state_history.append(old_sample)
    ctx.authoritative_state_history.append(newer_sample)

    selected = server._select_authoritative_state_snapshot(ctx, 0x00003000)

    assert selected is newer_sample
    print("test_remote_state_sync_reuses_cached_sample_when_replay_window_misses: PASSED")
    return True


def test_remote_promoted_heartbeat_stays_short_form_safe():
    """Ordinary promoted remote heartbeats should stay on the short-form-safe local-state."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server.update_entity_vitals = False
    server.heartbeat_view_update = False
    server.heartbeat_include_rot = True
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678

    session = Session()
    session.translation_ack_received = True
    session.in_game = True
    session.entity_id = 0x14EA
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = True
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_pose = {"roll": 0.0, "pitch": 0.0}
    ctx.player_heading = 0.0

    payload = server._build_local_state_heartbeat(
        ctx,
        tick=0x12345678,
        entity_id=0x14EA,
        include_health=True,
        health=1.0,
        fuel=1.0,
    )
    tick, local_state, entities = decode_update_array(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert len(entities) == 1
    assert entities[0].position is None
    assert entities[0].velocity is None
    assert entities[0].rotation is not None
    print("test_remote_promoted_heartbeat_stays_short_form_safe: PASSED")
    return True


def test_remote_state_sync_reply_keeps_full_motion_when_stable():
    """Repeated targeted sync replies should keep full motion vectors for correction verify."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
    server.remote_full_local_state_delay = 2.0
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server.update_epsilon = 0.001
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678
    server.debug_viewpoint = False
    server.debug_udp_raw = False
    server.pktlog = SimpleNamespace(enabled=False)
    captured = []
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: captured.append((payload, addr)))

    session = Session()
    session.translation_ack_received = True
    session.in_game = True
    session.entity_id = 0x14EA
    session.udp_addr = ("10.10.10.2", 50000)
    session.last_spawn_time = time.monotonic() - 5.0
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = True
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_pose = {"roll": 0.0}
    ctx.player_heading = 0.25
    ctx.player_yaw = -0.25
    ctx.last_action_dump_time = session.last_spawn_time + 1.0
    ctx.last_state_sync_send = 0.0

    server._send_state_sync_snapshot(
        ctx,
        include_view_update=True,
        replay_timestamp=0x89ABCDEF,
        reason="state_request",
    )
    assert len(captured) == 2
    first_update = captured[0][0]
    first_view = captured[1][0]
    _, first_local_state, first_entities = decode_update_array(
        first_update,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    _, _, first_view_local_state, first_view_entities = decode_view_update(
        first_view,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert first_local_state is not None and first_local_state.weapon_id == 2
    assert first_view_local_state is not None and first_view_local_state.weapon_id == 2
    assert first_entities[0].rotation is not None
    assert first_entities[0].velocity is not None
    assert first_view_entities[0].rotation is not None
    assert first_view_entities[0].velocity is not None

    captured.clear()
    ctx.last_state_sync_send = 0.0
    server._send_state_sync_snapshot(
        ctx,
        include_view_update=True,
        replay_timestamp=0x89ABCDF0,
        reason="state_request",
    )

    assert len(captured) == 2
    second_update = captured[0][0]
    second_view = captured[1][0]
    _, second_local_state, second_entities = decode_update_array(
        second_update,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    _, _, second_view_local_state, second_view_entities = decode_view_update(
        second_view,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert second_local_state is not None and second_local_state.weapon_id == 2
    assert second_view_local_state is not None and second_view_local_state.weapon_id == 2
    assert second_entities[0].rotation is not None
    assert second_entities[0].velocity is not None
    assert second_view_entities[0].rotation is not None
    assert second_view_entities[0].velocity is not None
    print("test_remote_state_sync_reply_keeps_full_motion_when_stable: PASSED")
    return True


def test_local_player_sync_rotation_uses_heading_not_player_yaw():
    """Local-player replication packets must use body heading, not camera-yaw sign."""
    server = WulframServer.__new__(WulframServer)
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_pose = {"roll": 0.125, "pitch": -0.375}
    ctx.player_heading = 0.25
    ctx.player_yaw = -0.25

    rot = server._local_player_sync_rotation(ctx)
    assert rot == (0.125, -0.375, 0.25)
    print("test_local_player_sync_rotation_uses_heading_not_player_yaw: PASSED")
    return True


def test_send_tank_uses_local_player_sync_rotation():
    """TankPacket resend/reset must use the same body-space rotation tuple."""
    class DummyTCP:
        def __init__(self):
            self.sent = []

        def send(self, payload):
            self.sent.append(payload)

    server = WulframServer.__new__(WulframServer)
    server._to_client_pos = lambda pos: pos
    server.tank_vitals = True
    server.weapon_id = 0
    server.spawn_height = 5.0
    server.up_axis = "z"
    server._get_energy_value = lambda ctx: 1.0
    server._log_vitals = lambda *args, **kwargs: None

    session = Session()
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.tcp_handler = DummyTCP()
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_pose = {"roll": 0.125, "pitch": -0.375}
    ctx.player_heading = 0.25
    ctx.player_yaw = -0.25

    server._send_tank(ctx)

    assert len(ctx.tcp_handler.sent) == 1
    payload = ctx.tcp_handler.sent[0]
    assert payload[0] == 0x18
    reader = BitReader(payload[1:])
    reader.read_bits(32)  # ticks
    assert reader.read_bits(1) == 1  # include_vitals
    reader.read_bits(5)   # weapon
    reader.read_bits(10)  # health
    reader.read_bits(10)  # energy
    reader.read_bits(32)  # unit_type
    assert reader.read_bits(32) == 0x14EA
    reader.read_bits(8)   # flags
    for _ in range(3):
        reader.read_bits(32)  # pos
    rx = struct.unpack(">i", struct.pack(">I", reader.read_bits(32)))[0] / 65536.0
    ry = struct.unpack(">i", struct.pack(">I", reader.read_bits(32)))[0] / 65536.0
    rz = struct.unpack(">i", struct.pack(">I", reader.read_bits(32)))[0] / 65536.0
    assert abs(rx - 0.125) < 1e-4
    assert abs(ry - (-0.375)) < 1e-4
    assert abs(rz - 0.25) < 1e-4
    print("test_send_tank_uses_local_player_sync_rotation: PASSED")
    return True


def test_spawn_wf_minimal_uses_local_player_sync_rotation():
    """Minimal UDP TankPacket spawn must preserve body-space pitch/heading."""
    captured = []

    server = WulframServer.__new__(WulframServer)
    server._pick_spawn_point = lambda team_id: None
    server.up_axis = "z"
    server.spawn_height = 5.0
    server.spawn_sets_ground_level = True
    server.clients_lock = threading.Lock()
    server.clients = {}
    server.multi_spawn_offset = 0.0
    server._to_client_pos = lambda pos: pos
    server.player_energy_max = 100.0
    server.tank_vitals = True
    server.weapon_id = 0
    server._log_vitals = lambda *args, **kwargs: None
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: captured.append((payload, addr)))

    session = Session()
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.player_pose = {"roll": 0.125, "pitch": -0.375}
    ctx.player_heading = 0.25
    ctx.player_yaw = -0.25

    server._spawn_wf_minimal(ctx, team_id=1, net_id=0x14EA, addr=("10.10.10.2", 51126))

    assert len(captured) == 1
    payload, addr = captured[0]
    assert addr == ("10.10.10.2", 51126)
    decoded = decode_tank_packet(payload)
    assert decoded is not None
    net_id, unit_type, team_id, pos, rot, health, energy = decoded
    assert net_id == 0x14EA
    assert unit_type == 0
    assert team_id == 1
    assert abs(rot[0] - 0.125) < 1e-4
    assert abs(rot[1] - (-0.375)) < 1e-4
    assert abs(rot[2] - 0.25) < 1e-4
    print("test_spawn_wf_minimal_uses_local_player_sync_rotation: PASSED")
    return True


def test_player_body_rotation_preserves_pitch_for_remote_entities():
    """Remote replicated body rotation should keep terrain-aligned pitch."""
    server = WulframServer.__new__(WulframServer)
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_pose = {"roll": 0.125, "pitch": -0.375}
    ctx.player_heading = 0.25

    rot = server._player_body_rotation(ctx, negate_yaw=True, yaw_offset=0.5)
    assert rot == (0.125, -0.375, 0.25)
    print("test_player_body_rotation_preserves_pitch_for_remote_entities: PASSED")
    return True


def test_remote_sync_heartbeat_helper_uses_heading_not_player_yaw():
    """Promoted remote heartbeat packets should decode with body heading."""
    server = WulframServer.__new__(WulframServer)
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server._to_client_pos = lambda pos: pos
    server._get_spawn_tank_weapon_type = lambda ctx: 2

    session = Session()
    session.entity_id = 0x14EA
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_pose = {"roll": 0.125, "pitch": -0.375}
    ctx.player_heading = 0.25
    ctx.player_yaw = -0.25

    payload = server._build_remote_sync_heartbeat_update(
        ctx,
        tick=0x12345678,
        rot=(0.125, -0.375, 0.25),
        include_vel=True,
        include_rot=True,
        include_local_state=True,
        health=1.0,
        fuel=1.0,
        weapon_type=2,
        ammo_bits=0,
        ammo_mask=0,
        pt_bits=0,
        pt_angle=0.0,
        st_bits=0,
        st_angle=0.0,
    )

    tick, local_state, entities = decode_update_array(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None
    assert len(entities) == 1
    assert entities[0].position is None
    assert entities[0].velocity is None
    assert entities[0].rotation is not None
    assert abs(entities[0].rotation[0] - 0.125) < 1e-3
    assert abs(entities[0].rotation[1] + 0.375) < 1e-3
    assert abs(entities[0].rotation[2] - 0.25) < 1e-3
    print("test_remote_sync_heartbeat_helper_uses_heading_not_player_yaw: PASSED")
    return True


def test_remote_spawn_bootstrap_heartbeat_uses_safe_rot_only_shape():
    """Fresh remote OG spawn bootstrap should get a rot-only safe heartbeat."""
    server = WulframServer.__new__(WulframServer)
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server._to_client_pos = lambda pos: pos
    server._get_spawn_tank_weapon_type = lambda ctx: 2

    session = Session()
    session.entity_id = 0x14EA
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (10.0, 20.0, 30.0)
    ctx.player_pose = {"roll": 0.125, "pitch": -0.375}
    ctx.player_heading = 0.25

    payload = server._build_remote_spawn_bootstrap_heartbeat(
        ctx,
        tick=0x12345678,
        entity_id=0x14EA,
        health=1.0,
        fuel=1.0,
    )

    tick, local_state, entities = decode_update_array(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert len(entities) == 1
    assert entities[0].entity_id == 0x14EA
    assert entities[0].position is None
    assert entities[0].velocity is None
    assert entities[0].rotation is not None
    assert entities[0].angular_velocity is not None
    assert abs(entities[0].rotation[0] - 0.125) < 1e-3
    assert abs(entities[0].rotation[1] + 0.375) < 1e-3
    assert abs(entities[0].rotation[2] - 0.25) < 1e-3
    print("test_remote_spawn_bootstrap_heartbeat_uses_safe_rot_only_shape: PASSED")
    return True


def test_state_request_does_not_overwrite_client_tick_offset():
    """STATE_REQUEST request_id must not replace input timing or sticky promotion state."""
    server = WulframServer.__new__(WulframServer)
    server._state_sync_reply_allowed_for_client = lambda ctx: False

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.tick_offset = 0x002B61E0 - 0x0002D18E
    ctx.last_client_tick = 0x002B61E0
    ctx.remote_full_local_state_ready = False

    payload = struct.pack(">BII", 0x0C, 0x0002D18E, 0)
    server._handle_state_request(ctx, payload, ctx.client_addr)

    assert ctx.tick_offset == (0x002B61E0 - 0x0002D18E)
    assert ctx.last_client_tick == 0x002B61E0
    assert ctx.remote_full_local_state_ready is False
    print("test_state_request_does_not_overwrite_client_tick_offset: PASSED")
    return True


def test_remote_udp_ping_request_gets_og_safe_reply():
    """Remote OG-style UDP 0x0B should receive the 5-byte 0x0C reply."""
    sent = []
    server = WulframServer.__new__(WulframServer)
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: sent.append((payload, addr)))
    server.udp_addr_to_client = {}
    server._recover_udp_client = lambda addr: None
    server.debug_udp_raw = False
    server.debug_sync = False
    server.udp_ping_reply_allow_all = True
    server.udp_ping_reply_hosts = set()
    server._udp_ping_reply_blocked_clients = set()

    ctx = ClientContext(
        client_id=7,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.session.udp_addr = ("10.10.10.2", 51126)

    payload = struct.pack(">BII", 0x0B, 0x12345678, 3)
    server._handle_single_udp_packet(ctx, payload, ctx.session.udp_addr)

    assert sent == [(b"\x0C\x12\x34\x56\x78", ("10.10.10.2", 51126))]
    print("test_remote_udp_ping_request_gets_og_safe_reply: PASSED")
    return True


def test_udp_ping_reply_default_policy_is_loopback_only():
    """Remote OG ping replies should be opt-in; local tools still get replies."""
    previous = os.environ.pop("WULFRAM_UDP_PING_REPLY_HOSTS", None)
    try:
        server = WulframServer(host="127.0.0.1", port=0)
    finally:
        if previous is not None:
            os.environ["WULFRAM_UDP_PING_REPLY_HOSTS"] = previous

    assert server.udp_ping_reply_allow_all is False
    assert {"127.0.0.1", "::1", "loopback"}.issubset(server.udp_ping_reply_hosts)

    remote = ClientContext(
        client_id=8,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    remote.session.udp_addr = ("10.10.10.2", 51126)
    local = ClientContext(
        client_id=9,
        client_addr=("127.0.0.1", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    local.session.udp_addr = ("127.0.0.1", 51126)

    assert server._udp_ping_reply_allowed_for_client(remote) is False
    assert server._udp_ping_reply_allowed_for_client(local) is True
    print("test_udp_ping_reply_default_policy_is_loopback_only: PASSED")
    return True


def test_jump_velocity_update_packet_uses_spawn_safe_local_state_for_remote_og():
    """Jump velocity updates must keep the OG-safe local-state shape."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 0.9
    server._get_network_tick = lambda ctx: 0x12345678

    session = Session()
    session.in_game = True
    session.entity_id = 0x14EA
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = True
    ctx.player_vel = (0.0, 0.0, 15.0)

    packet = server._build_velocity_update_packet(ctx)
    tick, local_state, entities = decode_update_array(
        packet,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None and local_state.weapon_id == 2
    assert entities[0].position is None
    assert entities[0].velocity is not None
    print("test_jump_velocity_update_packet_uses_spawn_safe_local_state_for_remote_og: PASSED")
    return True


def test_translation_velocity_quantizer_matches_decompile_defaults():
    """TRANSLATION velocity quantizer must stay on the OG 200/400 range."""
    table = parse_translation(build_translation_packet())
    q_vel = table.vec_quantizer(1)
    assert q_vel.bits == 4
    assert q_vel.max_total_bits == 16
    assert q_vel.max_val == 200.0
    assert q_vel.range_val == 400.0
    print("test_translation_velocity_quantizer_matches_decompile_defaults: PASSED")
    return True


def test_server_remote_local_state_kwargs_use_full_tank_shape():
    """Remote OG full updates should use the real tank local-state shape."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)]
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0

    session = Session()
    session.translation_ack_received = True
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = True
    ctx.player_aim_yaw = 1.234

    kwargs = server._get_local_state_kwargs(ctx)

    assert kwargs["weapon_id"] == 0
    assert kwargs["ammo_count_bits"] == 9
    assert kwargs["ammo_count"] == 0
    assert kwargs["primary_turret_bits"] == 16
    assert kwargs["secondary_turret_bits"] == 0
    print("test_server_remote_local_state_kwargs_use_full_tank_shape: PASSED")
    return True


def test_server_remote_projectile_spawn_uses_viewer_local_state():
    """Remote OG projectile spawns must carry the viewer's safe local-state prefix."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server.debug_health_value = 1.0
    server.debug_health_pattern = False
    server.player_energy_max = 100.0

    session = Session()
    session.translation_ack_received = True
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = True

    include_local_state, kwargs = server._get_projectile_local_state_for_viewer(ctx)
    proj = Projectile(
        entity_id=21000,
        entity_type=EntityType.PULSE_SHELL,
        owner_id=0x14EA,
        team=1,
        pos=(4984.25, 5117.75, 6.5),
        vel=(60.0, 45.0, 0.0),
        spawn_time=0.0,
        lifetime=5.0,
    )
    payload = build_projectile_spawn_packet(
        proj,
        0x12345678,
        include_local_state=include_local_state,
        **kwargs,
    )

    tick, local_state, entities = decode_update_array(payload)
    assert tick == 0x12345678
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert len(entities) == 1
    assert entities[0].entity_id == 21000
    assert entities[0].entity_type == EntityType.PULSE_SHELL
    assert entities[0].velocity is not None
    assert entities[0].rotation is not None
    assert math.isclose(entities[0].velocity[0], proj.vel[0], abs_tol=0.1)
    assert math.isclose(entities[0].velocity[1], proj.vel[1], abs_tol=0.1)
    assert math.isclose(entities[0].velocity[2], proj.vel[2], abs_tol=0.1)
    assert math.isclose(entities[0].rotation[1], 0.0, abs_tol=0.02)
    assert math.isclose(entities[0].rotation[2], math.atan2(proj.vel[1], proj.vel[0]), abs_tol=0.02)
    print("test_server_remote_projectile_spawn_uses_viewer_local_state: PASSED")
    return True


def test_server_remote_entity_packets_use_safe_local_state_after_promotion():
    """Promoted remote OG entity-only UPDATE_ARRAY packets must stay short-form-safe."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server.debug_health_value = 1.0
    server.debug_health_pattern = False
    server.player_energy_max = 100.0

    session = Session()
    session.translation_ack_received = True
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = True

    include_local_state, kwargs = server._get_update_array_local_state_for_viewer(ctx)

    assert include_local_state is True
    assert kwargs["weapon_id"] == 2
    assert kwargs["ammo_count_bits"] == 0
    assert kwargs["primary_turret_bits"] == 0
    assert kwargs["secondary_turret_bits"] == 0
    print("test_server_remote_entity_packets_use_safe_local_state_after_promotion: PASSED")
    return True


def test_server_remote_projectile_update_uses_safe_local_state_after_promotion():
    """Remote OG projectile updates must stay parse-safe after full-sync promotion."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server.debug_health_value = 1.0
    server.debug_health_pattern = False
    server.player_energy_max = 100.0

    session = Session()
    session.translation_ack_received = True
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = True

    include_local_state, kwargs = server._get_projectile_local_state_for_viewer(ctx)
    proj = Projectile(
        entity_id=21000,
        entity_type=EntityType.PULSE_SHELL,
        owner_id=0x14EA,
        team=1,
        pos=(4981.4, 5091.6, 6.5),
        vel=(-10.4, -74.3, 0.0),
        spawn_time=0.0,
        lifetime=5.0,
    )
    payload = build_projectile_update_packet(
        proj,
        0x12345679,
        0.0,
        include_local_state=include_local_state,
        **kwargs,
    )

    tick, local_state, entities = decode_update_array(payload)
    assert tick == 0x12345679
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert len(entities) == 1
    assert entities[0].entity_id == 21000
    print("test_server_remote_projectile_update_uses_safe_local_state_after_promotion: PASSED")
    return True


def test_loopback_projectile_update_stays_entity_only():
    """Loopback/Python projectile updates must remain entity-only."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32

    session = Session()
    session.translation_ack_received = True
    ctx = ClientContext(
        client_id=2,
        client_addr=("127.0.0.1", 50001),
        session=session,
        entity_id=0x14EE,
    )
    ctx.remote_full_local_state_ready = True

    include_local_state, kwargs = server._get_update_array_local_state_for_viewer(ctx)
    proj = Projectile(
        entity_id=21001,
        entity_type=EntityType.PULSE_SHELL,
        owner_id=0x14EE,
        team=1,
        pos=(4984.25, 5117.75, 6.5),
        vel=(60.0, 45.0, 0.0),
        spawn_time=0.0,
        lifetime=5.0,
    )
    payload = build_projectile_update_packet(
        proj,
        0x12345679,
        0.0,
        include_local_state=include_local_state,
        **kwargs,
    )

    tick, local_state, entities = decode_update_array(payload)
    assert tick == 0x12345679
    assert local_state is None
    assert len(entities) == 1
    assert entities[0].entity_id == 21001
    print("test_loopback_projectile_update_stays_entity_only: PASSED")
    return True


def test_server_remote_player_info_uses_spawn_safe_local_state():
    """Remote OG PLAYER_INFO should stay short-form-safe before gameplay ticks start."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.player_info_local_state_mode = "auto-remote"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]

    session = Session()
    session.translation_ack_received = True
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )

    include_local_state, kwargs = server._get_player_info_local_state_kwargs(ctx)

    assert include_local_state is True
    assert kwargs["weapon_id"] == 2
    assert kwargs["ammo_count_bits"] == 0
    assert kwargs["ammo_count"] == 0
    assert kwargs["primary_turret_bits"] == 0
    assert kwargs["secondary_turret_bits"] == 0
    print("test_server_remote_player_info_uses_spawn_safe_local_state: PASSED")
    return True


def test_remote_player_info_packet_short_local_state_layout():
    """Remote OG PLAYER_INFO should carry the short-form-safe local-state before vehicle fields."""
    payload = build_player_info(
        entity_oid=0x14EA,
        vehicle_type=0,
        pos=(4950.0, 5100.0, 5.0),
        rot=(0.0, 0.0, 0.0),
        include_local_state=True,
        weapon_id=2,
        health=1.0,
        fuel=1.0,
        properties=2,
        ammo_count_bits=0,
        ammo_count=0,
        primary_turret_bits=0,
        secondary_turret_bits=0,
        turret_max=6.3,
        turret_range=12.6,
    )

    assert payload[0] == 0x18
    br = BitReader(payload[1:])
    assert br.read_bits(32) == 0x14EA
    assert br.read_bits(1) == 1
    assert br.read_bits(5) == 2
    assert br.read_bits(10) == 1
    assert br.read_bits(10) == 1
    assert br.read_bits(32) == 0
    assert br.read_bits(32) == 0x14EA
    assert br.read_bits(8) == 2
    print("test_remote_player_info_packet_short_local_state_layout: PASSED")
    return True


def test_remote_spawn_entry_transition_sends_canonical_packets():
    """Remote OG auto-spawn should reassert the team-entry transition before tank spawn."""
    class DummyTCP:
        def __init__(self):
            self.sent = []

        def send(self, payload):
            self.sent.append(payload)

    class DummyUDP:
        def __init__(self):
            self.sent = []

        def send_to(self, payload, addr):
            self.sent.append((payload, addr))

    server = WulframServer.__new__(WulframServer)
    server.spawn_entry_transition = "1"
    server.udp_handler = DummyUDP()
    server.build_world_stats_packet = lambda: build_world_stats()

    session = Session()
    session.username = "RemoteOG"
    session.udp_addr = ("10.10.10.2", 55839)
    ctx = ClientContext(
        client_id=3,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x053B,
    )
    ctx.tcp_handler = DummyTCP()

    server._send_spawn_entry_transition(ctx, team_id=1, net_id=0x053B)

    assert len(server.udp_handler.sent) == 1
    assert server.udp_handler.sent[0][0][0] == 0x25
    assert server.udp_handler.sent[0][1] == ("10.10.10.2", 55839)
    assert [payload[0] for payload in ctx.tcp_handler.sent] == [0x17, 0x2F, 0x1A, 0x16]
    assert session.player_id == 0x053B
    assert session.team_id == 1
    assert session.roster_sent is True
    assert session.world_stats_sent is True
    print("test_remote_spawn_entry_transition_sends_canonical_packets: PASSED")
    return True


def test_udp_team_switch_sends_update_stats_before_reincarnate():
    """OG team switch should mirror captured UPDATE_STATS -> REINCARNATE order."""
    from wulfram.handlers import handle_team_switch

    class DummyTCP:
        def __init__(self):
            self.sent = []
            self.sock = SimpleNamespace(fileno=lambda: 1)

        def send(self, payload):
            self.sent.append(payload)

    class DummyUDP:
        def __init__(self):
            self.sent = []

        def send_to(self, payload, addr):
            self.sent.append((payload, addr))

    server = WulframServer.__new__(WulframServer)
    server.udp_handler = DummyUDP()
    server.spawn_on_team_select = False
    server.spawn_force_after = 0.0
    server.team_switch_send_update_stats = True
    server.team_switch_update_stats_transport = "udp"
    server.team_switch_update_stats_variant = "canonical"
    server.team_switch_send_reincarnate = True
    server.team_switch_send_roster = False
    server.team_switch_send_entry_packets = True
    server.build_world_stats_packet = lambda: build_world_stats()

    session = Session()
    session.player_id = 0x0539
    session.roster_sent = True
    session.world_stats_sent = True
    ctx = ClientContext(
        client_id=4,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x0539,
    )
    ctx.tcp_handler = DummyTCP()

    handle_team_switch(server, ctx, 1, ("10.10.10.2", 59507))

    assert [payload[0] for payload, _ in server.udp_handler.sent] == [0x1C, 0x25]
    assert [payload[0] for payload in ctx.tcp_handler.sent] == [0x17, 0x2F]
    assert session.team_id == 1
    print("test_udp_team_switch_sends_update_stats_before_reincarnate: PASSED")
    return True


def test_udp_team_switch_can_suppress_duplicate_entry_packets():
    """Live OG isolation can disable duplicate PLAYER/GAME_CLOCK after team click."""
    from wulfram.handlers import handle_team_switch

    class DummyTCP:
        def __init__(self):
            self.sent = []
            self.sock = SimpleNamespace(fileno=lambda: 1)

        def send(self, payload):
            self.sent.append(payload)

    server = WulframServer.__new__(WulframServer)
    server.udp_handler = None
    server.spawn_on_team_select = False
    server.spawn_force_after = 0.0
    server.team_switch_send_update_stats = False
    server.team_switch_update_stats_transport = "udp"
    server.team_switch_update_stats_variant = "canonical"
    server.team_switch_send_reincarnate = False
    server.team_switch_send_roster = False
    server.team_switch_send_entry_packets = False
    server.build_world_stats_packet = lambda: build_world_stats()

    session = Session()
    session.player_id = 0x0539
    session.roster_sent = True
    session.world_stats_sent = True
    ctx = ClientContext(
        client_id=4,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x0539,
    )
    ctx.tcp_handler = DummyTCP()

    handle_team_switch(server, ctx, 1, ("10.10.10.2", 59507))

    assert ctx.tcp_handler.sent == []
    assert session.team_id == 1
    print("test_udp_team_switch_can_suppress_duplicate_entry_packets: PASSED")
    return True


def test_udp_team_switch_can_reassert_roster_without_entry_packets():
    """Team-click isolation can send only ADD_TO_ROSTER as the visible team mutation."""
    from wulfram.handlers import handle_team_switch

    class DummyTCP:
        def __init__(self):
            self.sent = []
            self.sock = SimpleNamespace(fileno=lambda: 1)

        def send(self, payload):
            self.sent.append(payload)

    server = WulframServer.__new__(WulframServer)
    server.udp_handler = None
    server.spawn_on_team_select = False
    server.spawn_force_after = 0.0
    server.team_switch_send_update_stats = False
    server.team_switch_update_stats_transport = "udp"
    server.team_switch_update_stats_variant = "canonical"
    server.team_switch_send_reincarnate = False
    server.team_switch_send_roster = True
    server.team_switch_send_entry_packets = False
    server.build_world_stats_packet = lambda: build_world_stats()

    session = Session()
    session.username = "RosterProbe"
    session.player_id = 0x0539
    session.roster_sent = True
    session.world_stats_sent = True
    ctx = ClientContext(
        client_id=4,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x0539,
    )
    ctx.tcp_handler = DummyTCP()

    handle_team_switch(server, ctx, 2, ("10.10.10.2", 59507))

    assert [payload[0] for payload in ctx.tcp_handler.sent] == [0x1A]
    assert ctx.tcp_handler.sent[0][1:5] == b"\x00\x00\x05\x39"
    assert ctx.tcp_handler.sent[0][5:9] == b"\x00\x00\x00\x02"
    assert session.team_id == 2
    print("test_udp_team_switch_can_reassert_roster_without_entry_packets: PASSED")
    return True


def test_udp_team_switch_can_send_update_stats_over_tcp():
    """Team switch UPDATE_STATS can be isolated onto TCP for OG live probing."""
    from wulfram.handlers import handle_team_switch

    class DummyTCP:
        def __init__(self):
            self.sent = []
            self.sock = SimpleNamespace(fileno=lambda: 1)

        def send(self, payload):
            self.sent.append(payload)

    class DummyUDP:
        def __init__(self):
            self.sent = []

        def send_to(self, payload, addr):
            self.sent.append((payload, addr))

    server = WulframServer.__new__(WulframServer)
    server.udp_handler = DummyUDP()
    server.spawn_on_team_select = False
    server.spawn_force_after = 0.0
    server.team_switch_send_update_stats = True
    server.team_switch_update_stats_transport = "tcp"
    server.team_switch_update_stats_variant = "canonical"
    server.team_switch_send_reincarnate = False
    server.team_switch_send_roster = False
    server.team_switch_send_entry_packets = False
    server.build_world_stats_packet = lambda: build_world_stats()

    session = Session()
    session.player_id = 0x0539
    session.roster_sent = True
    session.world_stats_sent = True
    ctx = ClientContext(
        client_id=4,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x0539,
    )
    ctx.tcp_handler = DummyTCP()

    handle_team_switch(server, ctx, 2, ("10.10.10.2", 59507))

    assert server.udp_handler.sent == []
    assert [payload[0] for payload in ctx.tcp_handler.sent] == [0x1C]
    assert ctx.tcp_handler.sent[0][17:19] == b"\x00\x02"
    print("test_udp_team_switch_can_send_update_stats_over_tcp: PASSED")
    return True


def test_udp_team_switch_can_use_team_first_update_stats_variant():
    """OG team switch can use archived team-first UPDATE_STATS layout."""
    from wulfram.handlers import handle_team_switch

    class DummyTCP:
        def __init__(self):
            self.sent = []
            self.sock = SimpleNamespace(fileno=lambda: 1)

        def send(self, payload):
            self.sent.append(payload)

    class DummyUDP:
        def __init__(self):
            self.sent = []

        def send_to(self, payload, addr):
            self.sent.append((payload, addr))

    server = WulframServer.__new__(WulframServer)
    server.udp_handler = DummyUDP()
    server.spawn_on_team_select = False
    server.spawn_force_after = 0.0
    server.team_switch_send_update_stats = True
    server.team_switch_update_stats_transport = "udp"
    server.team_switch_update_stats_variant = "team_first"
    server.team_switch_send_reincarnate = False
    server.team_switch_send_roster = False
    server.team_switch_send_entry_packets = False
    server.build_world_stats_packet = lambda: build_world_stats()

    session = Session()
    session.player_id = 0x0539
    session.roster_sent = True
    session.world_stats_sent = True
    ctx = ClientContext(
        client_id=4,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x0539,
    )
    ctx.tcp_handler = DummyTCP()

    handle_team_switch(server, ctx, 2, ("10.10.10.2", 59507))

    assert len(server.udp_handler.sent) == 1
    packet = server.udp_handler.sent[0][0]
    assert packet[9:11] == b"\x00\x02"
    assert packet[17:19] == b"\x00\x02"
    print("test_udp_team_switch_can_use_team_first_update_stats_variant: PASSED")
    return True


def test_spawn_entry_transition_stays_off_by_default():
    """Entry transition injection should be opt-in outside real REINCARNATE handling."""
    class DummyTCP:
        def __init__(self):
            self.sent = []

        def send(self, payload):
            self.sent.append(payload)

    server = WulframServer.__new__(WulframServer)
    server.spawn_entry_transition = "off"
    server.udp_handler = None

    session = Session()
    ctx = ClientContext(
        client_id=1,
        client_addr=("127.0.0.1", 50000),
        session=session,
        entity_id=0x0539,
    )
    ctx.tcp_handler = DummyTCP()

    server._send_spawn_entry_transition(ctx, team_id=1, net_id=0x0539)

    assert ctx.tcp_handler.sent == []
    print("test_spawn_entry_transition_stays_off_by_default: PASSED")
    return True


def test_control_game_clock_builder_matches_packet_signature():
    """Control-plane GAME_CLOCK injection should use the current packet builder signature."""
    control = ControlServer(port=0)
    payload = control._build_game_clock("123", "true", "30000", "0")
    assert payload[0] == 0x2F
    assert len(payload) == 14
    print("test_control_game_clock_builder_matches_packet_signature: PASSED")
    return True


def test_weapon_system_og_direct_trigger_slot_fires_pulse_shell():
    """OG key 1 should fire via direct behavior slot 12 without WEAPON_DEMAND."""
    ws = WeaponSystem()
    ws.player_pos = (4950.0, 5100.0, 5.0)
    ws.player_rot = (0.0, 0.0, 0.0)
    ws.player_team = 2
    ws.behavior_slots[12] = 1.0

    projectiles, energy_spent = ws.update(dt=1.0, available_energy=100.0)

    assert len(projectiles) == 1
    assert projectiles[0].entity_type == EntityType.PULSE_SHELL
    assert energy_spent > 0.0
    print("test_weapon_system_og_direct_trigger_slot_fires_pulse_shell: PASSED")
    return True


def test_weapon_system_held_fire_repeats_on_cooldown():
    """Held fire should repeat when cooldown expires instead of only on rising edge."""
    ws = WeaponSystem()
    ws.current_weapon = WeaponType.CHAIN_GUN
    fired = []
    ws.on_chain_gun_fire = lambda **kwargs: fired.append(kwargs)
    ws.behavior_slots[8] = 1.0

    ws.update(dt=1.0, available_energy=100.0)
    ws.update(dt=0.05, available_energy=100.0)
    ws.update(dt=0.05, available_energy=100.0)

    assert len(fired) == 2, fired
    print("test_weapon_system_held_fire_repeats_on_cooldown: PASSED")
    return True


def test_send_entity_create_uses_udp_only():
    """Remote entity-create UPDATE_ARRAY should avoid TCP for OG safety."""
    server = WulframServer.__new__(WulframServer)
    server.remote_yaw_negate = False
    server.remote_yaw_offset = 0.0
    server.update_local_state_mode = "wf"
    server.spawn_tank_weapon_type = 2
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678
    calls = []

    def fake_send_packet(ctx, payload, *, prefer_tcp=True):
        calls.append((ctx.client_id, prefer_tcp, payload[0], len(payload)))
        return True

    server._send_packet_to_client = fake_send_packet

    target_session = Session()
    target_session.translation_ack_received = True
    target_ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=target_session,
        entity_id=1337,
    )

    player_session = Session()
    player_session.entity_id = 1338
    player_session.team_id = 2
    player_ctx = ClientContext(
        client_id=2,
        client_addr=("127.0.0.1", 50001),
        session=player_session,
        entity_id=1338,
    )
    player_ctx.player_pos = (4980.0, 5100.0, 5.0)
    player_ctx.player_pose = {"roll": 0.0}
    player_ctx.player_heading = 0.0
    player_ctx.entity_type = 0

    server._send_entity_create(target_ctx, player_ctx)

    assert len(calls) == 1, calls
    assert calls[0][1] is False, calls
    assert calls[0][2] == 0x0E, calls
    assert 1338 in target_ctx.known_entity_ids
    print("test_send_entity_create_uses_udp_only: PASSED")
    return True


def test_transient_fx_stays_off_for_remote_og_by_default():
    """Remote OG viewers should not receive crash-prone cosmetic TRANSIENT_ARRAY by default."""
    sent = []
    server = WulframServer.__new__(WulframServer)
    server.remote_transient_fx = False
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: sent.append((payload, addr)))

    remote_session = Session()
    remote_session.in_game = True
    remote_session.udp_addr = ("10.10.10.2", 50000)
    remote_ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=remote_session,
        entity_id=1337,
    )

    loopback_session = Session()
    loopback_session.in_game = True
    loopback_session.udp_addr = ("127.0.0.1", 50001)
    loopback_ctx = ClientContext(
        client_id=2,
        client_addr=("127.0.0.1", 50001),
        session=loopback_session,
        entity_id=1338,
    )

    server._snapshot_in_game_clients = lambda: [remote_ctx, loopback_ctx]

    pkt = server._broadcast_transient_fx([{
        "type": 12,
        "pos": (4950.0, 5100.0, 5.0),
    }])

    assert pkt and pkt[0] == 0x0D
    assert sent == [(pkt, ("127.0.0.1", 50001))], sent
    print("test_transient_fx_stays_off_for_remote_og_by_default: PASSED")
    return True


def test_transient_fx_can_be_enabled_for_remote_clients():
    """Loopback/Python should keep FX, and remote delivery can be explicitly re-enabled."""
    sent = []
    server = WulframServer.__new__(WulframServer)
    server.remote_transient_fx = True
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: sent.append((payload, addr)))

    remote_session = Session()
    remote_session.in_game = True
    remote_session.udp_addr = ("10.10.10.2", 50000)
    remote_ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=remote_session,
        entity_id=1337,
    )

    loopback_session = Session()
    loopback_session.in_game = True
    loopback_session.udp_addr = ("127.0.0.1", 50001)
    loopback_ctx = ClientContext(
        client_id=2,
        client_addr=("127.0.0.1", 50001),
        session=loopback_session,
        entity_id=1338,
    )

    server._snapshot_in_game_clients = lambda: [remote_ctx, loopback_ctx]

    pkt = server._broadcast_transient_fx([{
        "type": 12,
        "pos": (4950.0, 5100.0, 5.0),
    }], exclude_client=loopback_ctx)

    assert pkt and pkt[0] == 0x0D
    assert sent == [(pkt, ("10.10.10.2", 50000))], sent
    print("test_transient_fx_can_be_enabled_for_remote_clients: PASSED")
    return True


def test_entity_create_uses_spawn_safe_local_state_for_og_viewer():
    """OG viewer entity-create packets must stay on the short-form-safe local-state."""
    server = WulframServer.__new__(WulframServer)
    server.remote_yaw_negate = False
    server.remote_yaw_offset = 0.0
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos

    viewer_session = Session()
    viewer_session.translation_ack_received = True
    viewer_ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=viewer_session,
        entity_id=1337,
    )
    viewer_ctx.remote_full_local_state_ready = True

    include_local_state, local_state = server._get_update_array_local_state_for_viewer(viewer_ctx)
    payload = build_update_array_create_tank(
        tick=0x12345678,
        entity_id=1338,
        entity_type=0,
        team=2,
        pos=(4980.0, 5100.0, 5.0),
        is_manned=True,
        rot=(0.0, 0.0, 0.0),
        include_health=include_local_state,
        **local_state,
    )

    tick, decoded_local_state, entities = decode_update_array(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert decoded_local_state is not None
    assert decoded_local_state.weapon_id == 2
    assert len(entities) == 1, entities
    assert entities[0].entity_id == 1338
    print("test_entity_create_uses_spawn_safe_local_state_for_og_viewer: PASSED")
    return True


def test_remote_player_update_uses_spawn_safe_viewer_local_state():
    """Remote player updates must use the viewer's short safe local-state."""
    server = WulframServer.__new__(WulframServer)
    server.remote_update_mode = "pos_vel_rot"
    server.remote_yaw_negate = False
    server.remote_yaw_offset = 0.0
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    captured = []
    server._send_packet_to_client = lambda ctx, payload, prefer_tcp=True: captured.append(payload) or True

    viewer_session = Session()
    viewer_session.translation_ack_received = True
    viewer_session.in_game = True
    viewer_session.entity_id = 1337
    viewer_ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=viewer_session,
        entity_id=1337,
    )
    viewer_ctx.remote_full_local_state_ready = True
    viewer_ctx.known_entity_ids.add(1338)
    viewer_ctx.player_pos = (4950.0, 5100.0, 5.0)
    viewer_ctx.player_vel = (0.0, 0.0, 0.0)
    viewer_ctx.player_pose = {"roll": 0.0}
    viewer_ctx.player_heading = 0.0

    other_session = Session()
    other_session.translation_ack_received = True
    other_session.in_game = True
    other_session.entity_id = 1338
    other_ctx = ClientContext(
        client_id=2,
        client_addr=("127.0.0.1", 50001),
        session=other_session,
        entity_id=1338,
    )
    other_ctx.player_pos = (4980.0, 5100.0, 5.0)
    other_ctx.player_vel = (0.0, 0.0, 0.0)
    other_ctx.player_pose = {"roll": 0.0}
    other_ctx.player_heading = 0.0
    other_ctx.angular_vel_yaw = 0.0
    other_ctx.entity_type = 0

    server._snapshot_in_game_clients = lambda: [viewer_ctx, other_ctx]

    server._send_remote_player_updates(viewer_ctx, tick=0x12345678, prefer_tcp=False)

    assert len(captured) == 1, captured
    tick, local_state, entities = decode_update_array(
        captured[0],
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert len(entities) == 1, entities
    assert entities[0].entity_id == 1338
    print("test_remote_player_update_uses_spawn_safe_viewer_local_state: PASSED")
    return True


def test_loopback_entity_create_decodes_roundtrip():
    """Loopback entity-create packets must decode cleanly on the Python client."""
    server = WulframServer.__new__(WulframServer)
    server.remote_yaw_negate = False
    server.remote_yaw_offset = 0.0
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos

    viewer_session = Session()
    viewer_session.translation_ack_received = True
    viewer_ctx = ClientContext(
        client_id=2,
        client_addr=("127.0.0.1", 50001),
        session=viewer_session,
        entity_id=1339,
    )
    viewer_ctx.entity_type = 0

    payload = build_update_array_create_tank(
        tick=0x12345678,
        entity_id=1338,
        entity_type=0,
        team=1,
        pos=(4950.0, 5100.0, 5.0),
        is_manned=True,
        rot=(0.0, 0.0, 0.0),
        **server._get_local_state_kwargs(viewer_ctx),
    )

    tick, local_state, entities = decode_update_array(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None
    assert local_state.weapon_id == 0
    assert len(entities) == 1, entities
    assert entities[0].entity_id == 1338
    print("test_loopback_entity_create_decodes_roundtrip: PASSED")
    return True


def test_loopback_remote_player_update_decodes_roundtrip():
    """Loopback remote updates must decode cleanly on the entity-only path."""
    server = WulframServer.__new__(WulframServer)
    server.remote_update_mode = "pos_vel_rot"
    server.remote_yaw_negate = False
    server.remote_yaw_offset = 0.0
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    captured = []
    server._send_packet_to_client = lambda ctx, payload, prefer_tcp=True: captured.append(payload) or True

    viewer_session = Session()
    viewer_session.translation_ack_received = True
    viewer_session.in_game = True
    viewer_session.entity_id = 1339
    viewer_ctx = ClientContext(
        client_id=2,
        client_addr=("127.0.0.1", 50001),
        session=viewer_session,
        entity_id=1339,
    )
    viewer_ctx.known_entity_ids.add(1338)
    viewer_ctx.player_pos = (4980.0, 5100.0, 5.0)
    viewer_ctx.player_vel = (0.0, 0.0, 0.0)
    viewer_ctx.player_pose = {"roll": 0.0}
    viewer_ctx.player_heading = 0.0
    viewer_ctx.entity_type = 0

    other_session = Session()
    other_session.translation_ack_received = True
    other_session.in_game = True
    other_session.entity_id = 1338
    other_ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=other_session,
        entity_id=1338,
    )
    other_ctx.player_pos = (4950.0, 5100.0, 5.0)
    other_ctx.player_vel = (0.0, 0.0, 0.0)
    other_ctx.player_pose = {"roll": 0.0}
    other_ctx.player_heading = 0.0
    other_ctx.angular_vel_yaw = 0.0
    other_ctx.entity_type = 0

    server._snapshot_in_game_clients = lambda: [viewer_ctx, other_ctx]
    server._send_remote_player_updates(viewer_ctx, tick=0x12345678, prefer_tcp=False)

    assert len(captured) == 1, captured
    tick, local_state, entities = decode_update_array(
        captured[0],
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is None
    assert len(entities) == 1, entities
    assert entities[0].entity_id == 1338
    assert entities[0].velocity is not None
    assert entities[0].rotation is not None
    assert entities[0].angular_velocity is not None
    print("test_loopback_remote_player_update_decodes_roundtrip: PASSED")
    return True


def test_loopback_heartbeat_decodes_roundtrip():
    """Loopback heartbeats must decode cleanly with local-state + dummy entity block."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.local_state_weapon_type = 0
    server.spawn_tank_weapon_type = 2
    server.local_state_ammo_override = False
    server.local_state_ammo_from_behavior = True
    server.local_state_primary_override = ""
    server.local_state_secondary_override = ""
    server.local_state_turret_bits = 16
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.behavior_weapon_caps = [(0, 0, 9, 0)] * 32
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server.heartbeat_view_update = False

    session = Session()
    session.translation_ack_received = True
    ctx = ClientContext(
        client_id=2,
        client_addr=("127.0.0.1", 50001),
        session=session,
        entity_id=0x14EE,
    )
    ctx.entity_type = 0

    payload = server._build_local_state_heartbeat(
        ctx,
        tick=0x12345678,
        entity_id=0x14EE,
        include_health=True,
        health=1.0,
        fuel=1.0,
    )

    tick, local_state, entities = decode_update_array(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None
    assert local_state.weapon_id == 0
    assert len(entities) == 1, entities
    assert entities[0].entity_id == 0xFFFFFFFE
    print("test_loopback_heartbeat_decodes_roundtrip: PASSED")
    return True


def test_server_network_strafe_decode_matches_og_sign():
    """OG slot-3 input is negated before physics: negative=right, positive=left."""
    server = WulframServer.__new__(WulframServer)
    server.strafe_sign = -1.0

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.weapon_system = SimpleNamespace(control_max=1000.0)

    left_input = server._decode_network_strafe_input(ctx, -0.5800)
    right_input = server._decode_network_strafe_input(ctx, 0.6409)

    assert left_input > 0.0, left_input
    assert right_input < 0.0, right_input
    print("test_server_network_strafe_decode_matches_og_sign: PASSED")
    return True


def test_remote_client_promotes_full_local_state_after_spawn_delay():
    """Remote OG clients should leave the spawn-safe path once stable in-game."""
    import time

    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.remote_full_local_state_delay = 0.75

    session = Session()
    session.in_game = True
    session.last_spawn_time = time.monotonic() - 1.0
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = False

    assert server._wf_minimal_local_state_for_client(ctx) is True
    assert server._maybe_promote_remote_full_local_state(ctx, reason="post_spawn") is True
    assert ctx.remote_full_local_state_ready is True
    assert server._wf_minimal_local_state_for_client(ctx) is False
    print("test_remote_client_promotes_full_local_state_after_spawn_delay: PASSED")
    return True


def test_remote_client_suppresses_periodic_spawn_safe_heartbeat_until_promoted():
    """Periodic remote heartbeats should stay off until targeted sync promotion."""
    import time

    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.remote_full_local_state_delay = 0.75

    session = Session()
    session.in_game = True
    session.last_spawn_time = time.monotonic()
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = False

    assert server._suppress_remote_spawn_safe_heartbeat(ctx) is True
    ctx.remote_full_local_state_ready = True
    assert server._suppress_remote_spawn_safe_heartbeat(ctx) is False
    print("test_remote_client_suppresses_periodic_spawn_safe_heartbeat_until_promoted: PASSED")
    return True


def test_remote_client_suppresses_spawn_bootstrap_heartbeat_until_promoted():
    """The one-off post-spawn heartbeat should stay off while remote OG is still spawn-safe."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"

    session = Session()
    session.in_game = True
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = False

    assert server._suppress_remote_spawn_bootstrap_heartbeat(ctx) is True
    ctx.remote_full_local_state_ready = True
    assert server._suppress_remote_spawn_bootstrap_heartbeat(ctx) is False
    print("test_remote_client_suppresses_spawn_bootstrap_heartbeat_until_promoted: PASSED")
    return True


def test_remote_client_does_not_promote_full_local_state_on_heartbeat_reason():
    """Remote full local-state promotion must wait for targeted sync, not heartbeat/action traffic."""
    import time

    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.remote_full_local_state_delay = 0.75

    session = Session()
    session.in_game = True
    session.last_spawn_time = time.monotonic() - 1.0
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = False

    assert server._maybe_promote_remote_full_local_state(ctx, reason="heartbeat") is False
    assert ctx.remote_full_local_state_ready is False
    assert server._maybe_promote_remote_full_local_state(ctx, reason="test") is True
    assert ctx.remote_full_local_state_ready is True
    print("test_remote_client_does_not_promote_full_local_state_on_heartbeat_reason: PASSED")
    return True


def test_remote_respawn_restores_promoted_local_state_after_spawn():
    """Previously promoted remote clients should resume full local-state heartbeat immediately after respawn."""
    captured = []
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server._get_network_tick = lambda ctx: 0x12345678
    server._build_local_state_heartbeat = lambda *args, **kwargs: b"HB"
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: captured.append((payload, addr)))

    session = Session()
    session.in_game = True
    session.udp_addr = ("10.10.10.2", 50000)
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = False
    ctx._spawn_safe_heartbeat_suppressed_logged = True
    ctx.last_state_sync_send = 99.0

    resumed = server._resume_remote_full_local_state_after_spawn(
        ctx,
        entity_id=0x14EA,
        previously_promoted=True,
    )

    assert resumed is True
    assert ctx.remote_full_local_state_ready is True
    assert ctx._spawn_safe_heartbeat_suppressed_logged is False
    assert ctx.last_state_sync_send == 0.0
    assert captured == [(b"HB", ("10.10.10.2", 50000))], captured
    print("test_remote_respawn_restores_promoted_local_state_after_spawn: PASSED")
    return True


def test_remote_initial_spawn_keeps_minimal_path_when_never_promoted():
    """Fresh remote spawns must not auto-resume full local-state sync before promotion."""
    captured = []
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server._build_local_state_heartbeat = lambda *args, **kwargs: b"HB"
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: captured.append((payload, addr)))

    session = Session()
    session.in_game = True
    session.udp_addr = ("10.10.10.2", 50000)
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.remote_full_local_state_ready = False
    ctx._spawn_safe_heartbeat_suppressed_logged = True
    ctx.last_state_sync_send = 77.0

    resumed = server._resume_remote_full_local_state_after_spawn(
        ctx,
        entity_id=0x14EA,
        previously_promoted=False,
    )

    assert resumed is False
    assert ctx.remote_full_local_state_ready is False
    assert ctx._spawn_safe_heartbeat_suppressed_logged is True
    assert ctx.last_state_sync_send == 77.0
    assert captured == [], captured
    print("test_remote_initial_spawn_keeps_minimal_path_when_never_promoted: PASSED")
    return True


def test_server_tank_motion_uses_low_speed_mobility_factor():
    """Tank forward movement should ramp from the OG 0.4 mobility floor at rest."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 45.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.up_axis = "z"
    server.terrain = None
    server.terrain_pitch_enabled = False
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.f32_physics = False
    server._resolve_entity_world_collision = (
        lambda ctx, px, py, pz, vx, vy, vz: (px, py, pz, vx, vy, vz)
    )
    server._check_building_collisions = (
        lambda ctx, px, py, pz, vx, vy: (px, py, vx, vy)
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.injected_input = (1.0, 0.0)
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (0.0, 0.0, 10.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}

    server._update_player_position(ctx, dt_override=1.0)

    vx, vy, vz = ctx.player_vel
    assert abs(vx - 34.0) < 1e-4, vx
    assert abs(vy) < 1e-4, vy
    assert abs(vz) < 1e-4, vz
    print("test_server_tank_motion_uses_low_speed_mobility_factor: PASSED")
    return True


def test_server_motion_clamps_to_move_adjust():
    """Movement vector should clamp to move_adjust before integration."""
    import math

    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 45.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.up_axis = "z"
    server.terrain = None
    server.terrain_pitch_enabled = False
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.f32_physics = False
    server._resolve_entity_world_collision = (
        lambda ctx, px, py, pz, vx, vy, vz: (px, py, pz, vx, vy, vz)
    )
    server._check_building_collisions = (
        lambda ctx, px, py, pz, vx, vy: (px, py, vx, vy)
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.injected_input = (1.0, 1.0)
    ctx.entity_type = EntityType.ASSAULT_PLATFORM
    ctx.player_pos = (0.0, 0.0, 10.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}

    server._update_player_position(ctx, dt_override=1.0)

    vx, vy, vz = ctx.player_vel
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    assert abs(speed - 85.0) < 1e-4, speed
    print("test_server_motion_clamps_to_move_adjust: PASSED")
    return True


def test_server_motion_reclamps_below_ground_after_collision_response():
    """Final authoritative pose should not remain below terrain after collision response pushes it down."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 45.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.up_axis = "z"
    server.terrain = SimpleNamespace(get_height=lambda x, y: 0.0)
    server.terrain_height_offset = 5.0
    server.terrain_pitch_enabled = False
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.f32_physics = False
    server._resolve_entity_world_collision = (
        lambda ctx, px, py, pz, vx, vy, vz: (px, py, 1.0, vx, vy, -7.0)
    )
    server._check_building_collisions = (
        lambda ctx, px, py, pz, vx, vy: (px, py, vx, vy)
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.injected_input = (0.0, 0.0)
    ctx.entity_type = EntityType.ASSAULT_PLATFORM
    ctx.player_pos = (10.0, 20.0, 5.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}

    server._update_player_position(ctx, dt_override=1.0)

    assert ctx.player_pos[2] == 5.0, ctx.player_pos
    assert ctx.player_vel[2] == 0.0, ctx.player_vel
    print("test_server_motion_reclamps_below_ground_after_collision_response: PASSED")
    return True


def test_ghost_rejoin_skips_loopback_clients():
    """Loopback/Python clients should not use ghost rejoin, which can poison local sync with stale localhost ghosts."""
    server = WulframServer.__new__(WulframServer)
    server.ghost_rejoin = True
    server._ghost_rejoin_attempted = set()

    assert server._ghost_rejoin(("127.0.0.1", 50000)) is None
    assert ("127.0.0.1", 50000) not in server._ghost_rejoin_attempted
    print("test_ghost_rejoin_skips_loopback_clients: PASSED")
    return True


def test_projectile_world_hit_skips_aabb_for_mesh_backed_building():
    """Projectile raycast should not fall back to coarse AABB when mesh exists and is clear."""
    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = None
    server.terrain = None
    server.terrain_height_offset = 0.0
    server.ground_level = 0.0
    server._from_client_pos = lambda pos: pos
    server._building_entities = {
        10069: SimpleNamespace(
            x=0.0,
            y=0.0,
            z=0.0,
            entity_type=EntityType.ENERGY_BUILDING,
            team_id=2,
            heading=0.0,
        )
    }
    server._building_collision = SimpleNamespace(
        available=True,
        has_collision_model=lambda entity_type, team_id: True,
        get_model_half_extents=lambda entity_type, team_id: (5.0, 5.0, 5.0),
        get_model_bounding_radius=lambda entity_type, team_id: 1.0,
        test_segment_collision=lambda building, start_pos, end_pos: False,
    )

    hit = server._check_projectile_world_hit(
        (-10.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        proj=None,
    )
    assert hit is None, hit
    print("test_projectile_world_hit_skips_aabb_for_mesh_backed_building: PASSED")
    return True


def test_projectile_world_hit_prefers_closest_building_before_terrain():
    """Unified world ray should select a closer static-world hit before a farther terrain hit."""
    server = WulframServer.__new__(WulframServer)
    server._from_client_pos = lambda pos: pos
    server._terrain_grid_collision = SimpleNamespace(
        raycast=lambda start, end: TerrainRaycastHit(
            position=(10.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            sector_index=0,
            cell=(0, 0),
            distance=10.0,
        )
    )
    server._building_entities = {
        20001: SimpleNamespace(
            x=5.0,
            y=0.0,
            z=0.0,
            entity_type=EntityType.ENERGY_BUILDING,
            team_id=1,
            heading=0.0,
        )
    }
    server._building_collision = SimpleNamespace(
        available=True,
        has_collision_model=lambda entity_type, team_id: True,
        get_model_half_extents=lambda entity_type, team_id: (1.0, 1.0, 1.0),
        get_model_bounding_radius=lambda entity_type, team_id: 1.0,
        test_segment_collision=lambda building, start_pos, end_pos: True,
    )

    hit = server._check_projectile_world_hit(
        (0.0, 0.0, 0.0),
        (12.0, 0.0, 0.0),
        proj=None,
    )
    assert hit is not None, hit
    assert hit[0] == "building", hit
    assert hit[2] == 20001, hit
    assert hit[1][0] < 10.0, hit
    print("test_projectile_world_hit_prefers_closest_building_before_terrain: PASSED")
    return True


def test_projectile_world_hit_clips_static_world_raycast_to_terrain():
    """Terrain hit should shorten the static-world ray so farther buildings do not win spuriously."""
    server = WulframServer.__new__(WulframServer)
    server._from_client_pos = lambda pos: pos
    server._terrain_grid_collision = SimpleNamespace(
        raycast=lambda start, end: TerrainRaycastHit(
            position=(5.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            sector_index=2,
            cell=(0, 0),
            distance=5.0,
        )
    )
    server._building_entities = {
        20002: SimpleNamespace(
            x=8.0,
            y=0.0,
            z=0.0,
            entity_type=EntityType.ENERGY_BUILDING,
            team_id=1,
            heading=0.0,
        )
    }
    server._building_collision = SimpleNamespace(
        available=True,
        has_collision_model=lambda entity_type, team_id: True,
        get_model_half_extents=lambda entity_type, team_id: (1.0, 1.0, 1.0),
        get_model_bounding_radius=lambda entity_type, team_id: 1.0,
        test_segment_collision=lambda building, start_pos, end_pos: True,
    )

    hit = server._check_projectile_world_hit(
        (0.0, 0.0, 0.0),
        (12.0, 0.0, 0.0),
        proj=None,
    )
    assert hit == ("terrain", (5.0, 0.0, 0.0), 2), hit
    print("test_projectile_world_hit_clips_static_world_raycast_to_terrain: PASSED")
    return True


def test_projectile_world_hit_prefers_terrain_when_static_world_reports_farther_hit():
    """Top-level world-ray selection should still prefer terrain if a clipped static-world hit reports farther distance."""
    server = WulframServer.__new__(WulframServer)
    server._from_client_pos = lambda pos: pos
    server._terrain_grid_collision = SimpleNamespace(
        raycast=lambda start, end: TerrainRaycastHit(
            position=(5.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            sector_index=7,
            cell=(0, 0),
            distance=5.0,
        )
    )
    server._raycast_static_buildings = lambda start_pos, end_pos: ("building", (5.0, 0.0, 0.0), 12345, 9.0)

    hit = server._check_projectile_world_hit(
        (0.0, 0.0, 0.0),
        (12.0, 0.0, 0.0),
        proj=None,
    )
    assert hit == ("terrain", (5.0, 0.0, 0.0), 7), hit
    print("test_projectile_world_hit_prefers_terrain_when_static_world_reports_farther_hit: PASSED")
    return True


def test_projectile_world_hit_uses_exact_mesh_raycast_position():
    """Mesh-backed world rays should use the exact mesh hit position, not the AABB broadphase t."""
    server = WulframServer.__new__(WulframServer)
    server._from_client_pos = lambda pos: pos
    server._terrain_grid_collision = None
    server._building_entities = {
        20003: SimpleNamespace(
            x=5.0,
            y=0.0,
            z=0.0,
            entity_type=EntityType.ENERGY_BUILDING,
            team_id=1,
            heading=0.0,
        )
    }
    server._building_collision = SimpleNamespace(
        available=True,
        has_collision_model=lambda entity_type, team_id: True,
        get_model_half_extents=lambda entity_type, team_id: (5.0, 5.0, 5.0),
        get_model_bounding_radius=lambda entity_type, team_id: 6.0,
        raycast_segment_collision=lambda building, start_pos, end_pos: (
            (4.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            14.0,
        ),
    )

    hit = server._check_projectile_world_hit(
        (-10.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        proj=None,
    )
    assert hit == ("building", (4.0, 0.0, 0.0), 20003), hit
    print("test_projectile_world_hit_uses_exact_mesh_raycast_position: PASSED")
    return True


def test_projectile_world_hit_mesh_broadphase_uses_bounding_sphere():
    """Mesh-backed world rays should reject on the model bounding sphere before precise mesh raycast."""
    server = WulframServer.__new__(WulframServer)
    server._from_client_pos = lambda pos: pos
    server._terrain_grid_collision = None
    server._building_entities = {
        20004: SimpleNamespace(
            x=0.0,
            y=0.0,
            z=0.0,
            entity_type=EntityType.ENERGY_BUILDING,
            team_id=1,
            heading=0.0,
        )
    }
    raycast_calls = []
    server._building_collision = SimpleNamespace(
        available=True,
        has_collision_model=lambda entity_type, team_id: True,
        get_model_half_extents=lambda entity_type, team_id: (10.0, 10.0, 10.0),
        get_model_bounding_radius=lambda entity_type, team_id: 1.0,
        raycast_segment_collision=lambda building, start_pos, end_pos: raycast_calls.append((start_pos, end_pos)),
    )

    hit = server._check_projectile_world_hit(
        (-10.0, 9.0, 0.0),
        (10.0, 9.0, 0.0),
        proj=None,
    )
    assert hit is None, hit
    assert raycast_calls == [], raycast_calls
    print("test_projectile_world_hit_mesh_broadphase_uses_bounding_sphere: PASSED")
    return True


def test_static_world_raycast_uses_quadtree_front_to_back_order():
    """Static-world ray broadphase should traverse quadtree children in decompile-shaped front-to-back quadrant order."""
    server = WulframServer.__new__(WulframServer)
    server._building_entities = {}
    server._static_world_raycast_root = _StaticWorldRayNode(
        0.0,
        4.0,
        0.0,
        4.0,
        (
            _StaticWorldRayNode(2.0, 4.0, 2.0, 4.0, None, (1,)),
            _StaticWorldRayNode(2.0, 4.0, 0.0, 2.0, None, (2,)),
            _StaticWorldRayNode(0.0, 2.0, 2.0, 4.0, None, (3,)),
            _StaticWorldRayNode(0.0, 2.0, 0.0, 2.0, None, (4,)),
        ),
        (),
    )
    visited = []

    server._ray_misses_static_world_node = lambda start, end, node: False

    def fake_leaf(node, start, end):
        visited.append(node.building_ids[0])
        if node.building_ids[0] == 2:
            return ("building", (3.0, 1.0, 0.0), 2, 1.0)
        return None

    server._raycast_static_world_leaf = fake_leaf
    hit = server._raycast_static_buildings((0.5, 0.5, 0.0), (3.5, 3.5, 0.0))
    assert hit == ("building", (3.0, 1.0, 0.0), 2, 1.0), hit
    assert visited == [4, 3, 2], visited
    print("test_static_world_raycast_uses_quadtree_front_to_back_order: PASSED")
    return True


def test_static_world_raycast_uses_point_query_for_zero_horizontal_direction():
    """Zero-horizontal-direction static-world rays should take the point-query path."""
    server = WulframServer.__new__(WulframServer)
    server._building_entities = {10001: SimpleNamespace(x=1.0, y=1.0, z=0.0)}
    server._static_world_raycast_root = _StaticWorldRayNode(0.0, 4.0, 0.0, 4.0, None, (10001,))
    calls = []

    def fake_point_query(start_pos, root):
        calls.append((start_pos, root.building_ids))
        return ("building-aabb", start_pos, 10001, 0.0)

    server._point_query_static_world = fake_point_query
    hit = server._raycast_static_buildings((1.0, 1.0, 0.0), (1.0, 1.0, 5.0))
    assert hit == ("building-aabb", (1.0, 1.0, 0.0), 10001, 0.0), hit
    assert calls == [((1.0, 1.0, 0.0), (10001,))], calls
    print("test_static_world_raycast_uses_point_query_for_zero_horizontal_direction: PASSED")
    return True


def test_static_world_quadtree_uses_bounding_radius_for_overlap_distribution():
    """Overlap-spanning blockers must distribute by quadtree radius, not only by half-extents or center child."""
    server = WulframServer.__new__(WulframServer)
    server._building_entities = {
        10001: SimpleNamespace(entity_type=1000, team_id=1, heading=0.0, x=90.0, y=90.0, z=0.0),
        10002: SimpleNamespace(entity_type=1000, team_id=2, heading=0.0, x=90.0, y=10.0, z=0.0),
        10003: SimpleNamespace(entity_type=1000, team_id=2, heading=0.0, x=10.0, y=90.0, z=0.0),
        10004: SimpleNamespace(entity_type=1000, team_id=2, heading=0.0, x=35.0, y=35.0, z=0.0),
        10005: SimpleNamespace(entity_type=1000, team_id=2, heading=0.0, x=70.0, y=70.0, z=0.0),
        10006: SimpleNamespace(entity_type=1000, team_id=2, heading=0.0, x=70.0, y=10.0, z=0.0),
        10007: SimpleNamespace(entity_type=1000, team_id=2, heading=0.0, x=10.0, y=70.0, z=0.0),
        10008: SimpleNamespace(entity_type=1000, team_id=2, heading=0.0, x=70.0, y=35.0, z=0.0),
        10009: SimpleNamespace(entity_type=1000, team_id=2, heading=0.0, x=35.0, y=70.0, z=0.0),
    }

    def fake_half_extents(building):
        return (3.0, 3.0, 5.0)

    server._get_building_world_half_extents = fake_half_extents
    server._get_building_quadtree_radius = lambda building: 30.0 if building.team_id == 1 else 6.0
    server._point_hits_static_building = lambda building, point: (
        "building-aabb",
        point,
        0.0,
    ) if building.team_id == 1 else None
    server._rebuild_static_world_raycast_index()
    root = server._static_world_raycast_root
    assert root is not None and root.children is not None

    point_hit = server._point_query_static_world((10.0, 10.0, 5.0), root)
    ray_hit = server._raycast_static_buildings((10.0, 10.0, 5.0), (10.0, 10.0, 12.0))
    assert point_hit == ("building-aabb", (10.0, 10.0, 5.0), 10001, 0.0), point_hit
    assert ray_hit == ("building-aabb", (10.0, 10.0, 5.0), 10001, 0.0), ray_hit
    print("test_static_world_quadtree_uses_bounding_radius_for_overlap_distribution: PASSED")
    return True


def test_static_world_raycast_stops_once_endpoint_leaf_is_reached():
    """Traversal should stop once the segment endpoint lies inside a leaf, even with no hit there."""
    server = WulframServer.__new__(WulframServer)
    server._building_entities = {}
    server._static_world_raycast_root = _StaticWorldRayNode(
        0.0,
        4.0,
        0.0,
        4.0,
        (
            _StaticWorldRayNode(2.0, 4.0, 2.0, 4.0, None, (1,)),
            _StaticWorldRayNode(2.0, 4.0, 0.0, 2.0, None, (2,)),
            _StaticWorldRayNode(0.0, 2.0, 2.0, 4.0, None, (3,)),
            _StaticWorldRayNode(0.0, 2.0, 0.0, 2.0, None, (4,)),
        ),
        (),
    )
    visited = []

    server._ray_misses_static_world_node = lambda start, end, node: False

    def fake_leaf(node, start, end):
        visited.append(node.building_ids[0])
        if node.building_ids[0] == 3:
            return ("building", (1.0, 3.0, 0.0), 3, 1.0)
        return None

    server._raycast_static_world_leaf = fake_leaf
    hit = server._raycast_static_buildings((0.5, 0.5, 0.0), (1.5, 1.5, 0.0))
    assert hit is None, hit
    assert visited == [4], visited
    print("test_static_world_raycast_stops_once_endpoint_leaf_is_reached: PASSED")
    return True


def test_static_world_raycast_node_cull_uses_signbit_zero_side_semantics():
    """Node line-cull should treat zero-valued corners as part of the nonnegative side, matching the decompile sign-bit test."""
    node = _StaticWorldRayNode(0.0, 1.0, 0.0, 1.0, None, ())
    misses = WulframServer._ray_misses_static_world_node(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        node,
    )
    assert misses is True, misses
    print("test_static_world_raycast_node_cull_uses_signbit_zero_side_semantics: PASSED")
    return True


def test_segment_raycast_cbsp_tree_uses_split_plane_normal():
    """Rebuilt CBSP tree raycasts should return the node split-plane normal, not triangle winding."""
    vertices = [
        Vec3(0.0, -1.0, -1.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, -1.0),
    ]
    tree = CBSPTree(
        nodes=[
            CBSPTreeNode(
                radius=2.0,
                half_extent_x=1.0,
                half_extent_y=1.0,
                half_extent_z=1.0,
                center=Vec3(0.0, 0.0, 0.0),
                triangles=[(0, 1, 2)],
                split_normal=Vec3(1.0, 0.0, 0.0),
                split_plane_d=0.0,
                child_neg=-1,
                child_pos=-1,
            )
        ],
        root_index=0,
    )

    hit = segment_raycast_cbsp_tree(
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        vertices,
        tree,
    )
    assert hit is not None
    hit_point, normal, dist_sq = hit
    assert hit_point == (0.0, 0.0, 0.0), hit_point
    assert normal == (1.0, 0.0, 0.0), normal
    assert abs(dist_sq - 1.0) <= 1e-6, dist_sq
    print("test_segment_raycast_cbsp_tree_uses_split_plane_normal: PASSED")
    return True


def test_segment_hits_cbsp_tree_detects_plane_hit():
    """Boolean rebuilt-tree segment hits should follow the same BSP traversal path."""
    vertices = [
        Vec3(0.0, -1.0, -1.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, -1.0),
    ]
    tree = CBSPTree(
        nodes=[
            CBSPTreeNode(
                radius=2.0,
                half_extent_x=1.0,
                half_extent_y=1.0,
                half_extent_z=1.0,
                center=Vec3(0.0, 0.0, 0.0),
                triangles=[(0, 1, 2)],
                split_normal=Vec3(1.0, 0.0, 0.0),
                split_plane_d=0.0,
                child_neg=-1,
                child_pos=-1,
            )
        ],
        root_index=0,
    )

    assert segment_hits_cbsp_tree(
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        vertices,
        tree,
    )
    print("test_segment_hits_cbsp_tree_detects_plane_hit: PASSED")
    return True


def test_triangle_cbsp_contact_uses_node_split_normal():
    """Terrain-vs-CBSP contacts should store the node split-plane normal, not the terrain face normal."""

    class FlatTerrain:
        cell_x = 1.0
        cell_z = 1.0
        world_w = 2.0
        world_h = 2.0
        num_x = 2
        num_z = 2

        @staticmethod
        def _get_raw_height(cell_x, cell_y):
            return 0.0

    grid = TerrainGridCollision(FlatTerrain(), 0.0)
    vertices = [
        Vec3(0.0, -1.0, -1.0),
        Vec3(0.0, 1.0, -1.0),
        Vec3(0.0, 0.0, 1.0),
    ]
    tree = CBSPTree(
        nodes=[
            CBSPTreeNode(
                radius=2.0,
                half_extent_x=1.0,
                half_extent_y=1.0,
                half_extent_z=1.0,
                center=Vec3(0.0, 0.0, 0.0),
                triangles=[(0, 1, 2)],
                split_normal=Vec3(1.0, 0.0, 0.0),
                split_plane_d=0.0,
                child_neg=-1,
                child_pos=-1,
            )
        ],
        root_index=0,
    )

    contact = grid._triangle_cbsp_contact(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.5, 0.0),
            (1.0, -0.5, 0.0),
        ),
        vertices,
        tree,
        2.0,
    )
    assert contact is not None
    _, normal, penetration = contact
    assert normal == (1.0, 0.0, 0.0), normal
    assert penetration > 0.0, penetration
    print("test_triangle_cbsp_contact_uses_node_split_normal: PASSED")
    return True


def test_triangle_cbsp_contact_uses_entity_bounding_radius_for_plane_reject():
    """Terrain-vs-CBSP entry should reject on the entity bounding sphere, not the root-node radius."""

    class FlatTerrain:
        cell_x = 1.0
        cell_z = 1.0
        world_w = 2.0
        world_h = 2.0
        num_x = 2
        num_z = 2

        @staticmethod
        def _get_raw_height(cell_x, cell_y):
            return 0.0

    grid = TerrainGridCollision(FlatTerrain(), 0.0)
    vertices = [
        Vec3(0.0, -2.0, 1.0),
        Vec3(0.0, 2.0, 1.0),
        Vec3(0.0, 0.0, 2.0),
    ]
    tree = CBSPTree(
        nodes=[
            CBSPTreeNode(
                radius=1.0,
                half_extent_x=1.0,
                half_extent_y=1.0,
                half_extent_z=1.0,
                center=Vec3(0.0, 0.0, 1.5),
                triangles=[(0, 1, 2)],
                split_normal=Vec3(1.0, 0.0, 0.0),
                split_plane_d=0.0,
                child_neg=-1,
                child_pos=-1,
            )
        ],
        root_index=0,
    )

    contact = grid._triangle_cbsp_contact(
        (
            (-1.0, 0.0, 1.5),
            (1.0, 0.5, 1.5),
            (1.0, -0.5, 1.5),
        ),
        vertices,
        tree,
        2.0,
    )
    assert contact is not None
    print("test_triangle_cbsp_contact_uses_entity_bounding_radius_for_plane_reject: PASSED")
    return True


def test_box_collision_returns_first_contact_in_grid_order():
    """Terrain box contact should return on the first hit instead of searching for the deepest one."""

    class FlatTerrain:
        cell_x = 1.0
        cell_z = 1.0
        world_w = 2.0
        world_h = 2.0
        num_x = 3
        num_z = 2

        @staticmethod
        def _get_raw_height(cell_x, cell_y):
            return 0.0

    grid = TerrainGridCollision(FlatTerrain(), 0.0, sector_rows=1, sector_cols=1)
    tri = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    calls = []

    grid._iter_aabb_sectors = lambda aabb_min, aabb_max: [SimpleNamespace(index=0)]
    grid._iter_sector_cells = lambda aabb_min, aabb_max, sector: [(0, 0), (1, 0)]
    grid._iter_cell_triangles = lambda cell_x, cell_y: [tri]
    grid._triangle_overlaps_xy_bounds = lambda *args, **kwargs: True

    def fake_triangle_box_contact(tri_local, half_extents):
        calls.append((tri_local, half_extents))
        return ((0.0, 0.0, 1.0), 0.1 if len(calls) == 1 else 2.0)

    grid._triangle_box_contact = fake_triangle_box_contact

    contact = grid.test_box_collision((0.5, 0.5, 0.5), (1.0, 1.0, 1.0), 0.0)
    assert contact is not None
    assert len(calls) == 1, calls
    assert contact.cell == (0, 0), contact.cell
    assert abs(contact.penetration - 0.1) <= 1e-6, contact.penetration
    print("test_box_collision_returns_first_contact_in_grid_order: PASSED")
    return True


def test_model_collision_returns_first_contact_in_grid_order():
    """Terrain mesh contact should return on the first hit instead of searching for the deepest one."""

    class FlatTerrain:
        cell_x = 1.0
        cell_z = 1.0
        world_w = 2.0
        world_h = 2.0
        num_x = 3
        num_z = 2

        @staticmethod
        def _get_raw_height(cell_x, cell_y):
            return 0.0

    grid = TerrainGridCollision(FlatTerrain(), 0.0, sector_rows=1, sector_cols=1)
    tri = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    calls = []

    grid._iter_aabb_sectors = lambda aabb_min, aabb_max: [SimpleNamespace(index=0)]
    grid._iter_sector_cells = lambda aabb_min, aabb_max, sector: [(0, 0), (1, 0)]
    grid._iter_cell_triangles = lambda cell_x, cell_y: [tri]
    grid._triangle_overlaps_aabb = lambda *args, **kwargs: True

    def fake_triangle_cbsp_contact(tri_local, vertices, cbsp_tree, bounding_radius):
        calls.append((tri_local, bounding_radius))
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1 if len(calls) == 1 else 2.0)

    grid._triangle_cbsp_contact = fake_triangle_cbsp_contact

    contact = grid.test_model_collision(
        (0.5, 0.5, 0.5),
        0.0,
        [],
        CBSPTree(nodes=[CBSPTreeNode(
            radius=1.0,
            half_extent_x=1.0,
            half_extent_y=1.0,
            half_extent_z=1.0,
            center=Vec3(0.0, 0.0, 0.0),
            triangles=[],
            split_normal=Vec3(1.0, 0.0, 0.0),
            split_plane_d=0.0,
            child_neg=-1,
            child_pos=-1,
        )]),
        1.0,
    )
    assert contact is not None
    assert len(calls) == 1, calls
    assert contact.cell == (0, 0), contact.cell
    assert abs(contact.penetration - 0.1) <= 1e-6, contact.penetration
    print("test_model_collision_returns_first_contact_in_grid_order: PASSED")
    return True


def test_triangle_cbsp_contact_returns_first_leaf_hit():
    """CBSP leaf hit recording should stop on the first hit instead of picking a later deeper triangle."""

    class FlatTerrain:
        cell_x = 1.0
        cell_z = 1.0
        world_w = 2.0
        world_h = 2.0
        num_x = 2
        num_z = 2

        @staticmethod
        def _get_raw_height(cell_x, cell_y):
            return 0.0

    grid = TerrainGridCollision(FlatTerrain(), 0.0)
    estimate_calls = []

    def fake_estimate(tri_a, tri_b, normal):
        estimate_calls.append((tri_b, normal))
        return 0.1 if len(estimate_calls) == 1 else 2.0

    grid._estimate_triangle_penetration = fake_estimate

    vertices = [
        Vec3(0.0, -1.0, -1.0),
        Vec3(0.0, 1.0, -1.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, -1.0, -1.0),
        Vec3(0.0, 1.0, -1.0),
        Vec3(0.0, 0.0, 1.0),
    ]
    tree = CBSPTree(
        nodes=[
            CBSPTreeNode(
                radius=2.0,
                half_extent_x=1.0,
                half_extent_y=1.0,
                half_extent_z=1.0,
                center=Vec3(0.0, 0.0, 0.0),
                triangles=[(0, 1, 2), (3, 4, 5)],
                split_normal=Vec3(1.0, 0.0, 0.0),
                split_plane_d=0.0,
                child_neg=-1,
                child_pos=-1,
            )
        ],
        root_index=0,
    )

    contact = grid._triangle_cbsp_contact(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.5, 0.0),
            (1.0, -0.5, 0.0),
        ),
        vertices,
        tree,
        2.0,
    )
    assert contact is not None
    _, normal, penetration = contact
    assert normal == (1.0, 0.0, 0.0), normal
    assert abs(penetration - 0.1) <= 1e-6, penetration
    assert len(estimate_calls) == 1, estimate_calls
    print("test_triangle_cbsp_contact_returns_first_leaf_hit: PASSED")
    return True


def test_building_collision_skips_aabb_for_mesh_backed_building():
    """Tank world collision should not use AABB when a real building mesh exists and is clear."""
    server = WulframServer.__new__(WulframServer)
    server._building_entities = {
        10069: SimpleNamespace(
            x=5047.6796875,
            y=5078.9921875,
            z=2.313780069351,
            entity_type=EntityType.ENERGY_BUILDING,
            team_id=2,
            heading=0.564635634422,
        )
    }
    server._building_collision = SimpleNamespace(
        available=True,
        has_collision_model=lambda entity_type, team_id: True,
        test_sphere_collision=lambda building, sphere_pos, sphere_radius: (0.0, None),
    )
    server._snapshot_in_game_clients = lambda: []

    px, py, vx, vy = server._check_building_collisions(
        None,
        5035.1,
        5090.5,
        6.2,
        4.0,
        -3.0,
    )
    assert abs(px - 5035.1) < 1e-6, px
    assert abs(py - 5090.5) < 1e-6, py
    assert abs(vx - 4.0) < 1e-6, vx
    assert abs(vy + 3.0) < 1e-6, vy
    print("test_building_collision_skips_aabb_for_mesh_backed_building: PASSED")
    return True


def test_repair_pad_collision_does_not_block_vehicle_movement():
    """Repair pads are service/spawn pads, not solid vehicle blockers."""
    old_env = os.environ.pop("WULFRAM_REPAIR_PAD_BLOCKS_VEHICLES", None)
    try:
        server = WulframServer.__new__(WulframServer)
        server._building_entities = {
            10001: SimpleNamespace(
                x=100.0,
                y=100.0,
                z=5.0,
                entity_type=EntityType.REPAIR_BUILDING,
                team_id=1,
                heading=0.0,
            )
        }
        server._building_collision = SimpleNamespace(
            available=True,
            has_collision_model=lambda entity_type, team_id: True,
            test_sphere_collision=lambda building, sphere_pos, sphere_radius: (
                10.0,
                (1.0, 0.0, 0.0),
            ),
        )
        server._snapshot_in_game_clients = lambda: []

        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(),
            entity_id=0x14EA,
        )

        px, py, vx, vy = server._check_building_collisions(
            ctx,
            100.0,
            100.0,
            5.0,
            3.0,
            4.0,
        )

        assert (px, py, vx, vy) == (100.0, 100.0, 3.0, 4.0), (px, py, vx, vy)
        assert ctx.debug_last_collision == {}, ctx.debug_last_collision
    finally:
        if old_env is not None:
            os.environ["WULFRAM_REPAIR_PAD_BLOCKS_VEHICLES"] = old_env

    print("test_repair_pad_collision_does_not_block_vehicle_movement: PASSED")
    return True


def test_building_collision_team_variant_matches_client_helper():
    """Server building collision must resolve the same team variant as the client."""
    assert BuildingCollisionAssets.get_model_name(EntityType.PAD, 1) == "skypump_2"
    assert BuildingCollisionAssets.get_model_name(EntityType.PAD, 2) == "skypump_1"
    assert BuildingCollisionAssets.get_model_name(EntityType.DARK_LIGHT, 1) == "darklight_2"
    assert BuildingCollisionAssets.get_model_name(EntityType.DARK_LIGHT, 2) == "darklight_1"
    print("test_building_collision_team_variant_matches_client_helper: PASSED")
    return True


def test_server_team_model_name_matches_client_helper():
    """Server non-building model selection must use the same team-variant semantics."""
    server = WulframServer(host="127.0.0.1", port=0)
    assert server._select_team_model_name(("tank_1", "tank_2"), 1) == "tank_2"
    assert server._select_team_model_name(("tank_1", "tank_2"), 2) == "tank_1"
    assert server._select_team_model_name(("s_missile_1", "s_missile_2"), 1) == "s_missile_2"
    assert server._select_team_model_name(("s_missile_1", "s_missile_2"), 2) == "s_missile_1"
    print("test_server_team_model_name_matches_client_helper: PASSED")
    return True


def test_effective_inactivity_timeout_extends_remote_ingame_clients():
    """Idle remote in-game sessions should not trip the short generic inactivity timeout."""
    server = WulframServer(host="127.0.0.1", port=0)
    server.inactivity_timeout = 120.0
    server.remote_idle_timeout = 900.0

    remote_ctx = ClientContext(client_id=1, client_addr=("10.10.10.2", 50011))
    remote_ctx.session = Session()
    remote_ctx.session.phase = Phase.IN_GAME

    local_ctx = ClientContext(client_id=2, client_addr=("127.0.0.1", 50011))
    local_ctx.session = Session()
    local_ctx.session.phase = Phase.IN_GAME

    assert server._effective_inactivity_timeout(remote_ctx) == 900.0
    assert server._effective_inactivity_timeout(local_ctx) == 120.0
    print("test_effective_inactivity_timeout_extends_remote_ingame_clients: PASSED")
    return True


def test_entity_world_collision_prefers_mesh_contact_when_collision_model_exists():
    """Mesh-backed entities should resolve terrain contact from the CBSP path, not SAT box fallback."""
    box_calls = 0
    model_calls = 0

    def fake_box_collision(*args, **kwargs):
        nonlocal box_calls
        box_calls += 1
        return TerrainContact(
            position=(10.0, 0.0, 0.0),
            normal=(1.0, 0.0, 0.0),
            penetration=2.0,
            sector_index=0,
            cell=(0, 0),
        )

    def fake_model_collision(*args, **kwargs):
        nonlocal model_calls
        model_calls += 1
        return TerrainContact(
            position=(100.0, 200.0, 10.5),
            normal=(0.0, 0.0, 1.0),
            penetration=0.5,
            sector_index=0,
            cell=(0, 0),
        )

    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = SimpleNamespace(
        test_box_collision=fake_box_collision,
        test_model_collision=fake_model_collision,
    )
    server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 1.0)
    server._get_entity_world_collision_model = lambda ctx: (
        [],
        SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
        5.0,
        0.0,
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        100.0,
        200.0,
        10.0,
        3.0,
        4.0,
        -6.0,
    )

    assert box_calls == 0, box_calls
    assert model_calls == 1, model_calls
    assert abs(px - 100.0) < 1e-6, px
    assert abs(py - 200.0) < 1e-6, py
    assert pz > 10.0, pz
    assert abs(vx - 3.0) < 1e-6, vx
    assert abs(vy - 4.0) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_prefers_mesh_contact_when_collision_model_exists: PASSED")
    return True


def test_entity_world_collision_falls_back_to_box_without_collision_model():
    """Non-mesh entities should keep the existing SAT-box terrain fallback."""
    box_calls = 0

    def fake_box_collision(*args, **kwargs):
        nonlocal box_calls
        box_calls += 1
        return TerrainContact(
            position=(50.0, 75.0, 6.0),
            normal=(1.0, 0.0, 0.0),
            penetration=1.25,
            sector_index=0,
            cell=(0, 0),
        )

    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = SimpleNamespace(
        test_box_collision=fake_box_collision,
        test_model_collision=lambda *args, **kwargs: None,
    )
    server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 1.0)
    server._get_entity_world_collision_model = lambda ctx: None

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        50.0,
        75.0,
        6.0,
        -2.0,
        1.0,
        0.0,
    )

    assert box_calls == 1, box_calls
    assert px > 50.0, px
    assert abs(py - 75.0) < 1e-6, py
    assert abs(pz - 6.0) < 1e-6, pz
    assert abs(vx) < 1e-6, vx
    assert abs(vy - 1.0) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_falls_back_to_box_without_collision_model: PASSED")
    return True


def test_entity_world_collision_uses_dirty_terrain_raycast_branch():
    """Large movement should route through the decompile-shaped terrain raycast branch before CBSP."""
    raycast_calls = 0
    model_calls = 0

    def fake_raycast(start, end):
        nonlocal raycast_calls
        raycast_calls += 1
        return TerrainRaycastHit(
            position=(4.0, 0.0, 5.0),
            normal=(0.0, 0.0, 1.0),
            sector_index=0,
            cell=(0, 0),
            distance=4.0,
        )

    def fake_model_collision(*args, **kwargs):
        nonlocal model_calls
        model_calls += 1
        return None

    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = SimpleNamespace(
        test_bounds_intersection=lambda *args, **kwargs: False,
        raycast=fake_raycast,
        test_box_collision=lambda *args, **kwargs: None,
        test_model_collision=fake_model_collision,
    )
    server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
    server._get_entity_world_collision_model = lambda ctx: (
        [],
        SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
        5.0,
        0.0,
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (0.0, 0.0, 5.0)
    ctx.world_collision_ref_pos = (0.0, 0.0, 5.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        10.0,
        0.0,
        5.0,
        3.0,
        0.0,
        -4.0,
    )

    assert raycast_calls == 1, raycast_calls
    assert model_calls == 0, model_calls
    assert px < 0.0, px
    assert abs(py) < 1e-6, py
    assert pz > 5.0, pz
    assert abs(vx - 3.0) < 1e-6, vx
    assert abs(vy) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_uses_dirty_terrain_raycast_branch: PASSED")
    return True


def test_entity_world_collision_uses_dirty_contact_before_raycast():
    """Dirty large-displacement motion should resolve overlapping terrain contact before raycast fallback."""
    raycast_calls = 0
    model_calls = 0

    def fake_raycast(start, end):
        nonlocal raycast_calls
        raycast_calls += 1
        return None

    def fake_model_collision(*args, **kwargs):
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return TerrainContact(
                position=(10.0, 0.0, 4.0),
                normal=(0.0, 0.0, 1.0),
                penetration=1.5,
                sector_index=0,
                cell=(0, 0),
            )
        return None

    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = SimpleNamespace(
        test_bounds_intersection=lambda *args, **kwargs: True,
        raycast=fake_raycast,
        test_box_collision=lambda *args, **kwargs: None,
        test_model_collision=fake_model_collision,
    )
    server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
    server._get_entity_world_collision_model = lambda ctx: (
        [],
        SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
        5.0,
        0.0,
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (0.0, 0.0, 4.0)
    ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        10.0,
        0.0,
        4.0,
        1.0,
        0.0,
        -5.0,
    )

    assert model_calls == 1, model_calls
    assert raycast_calls == 0, raycast_calls
    assert abs(px - 10.0) < 1e-6, px
    assert abs(py) < 1e-6, py
    assert pz > 5.0, pz
    assert abs(vx - 1.0) < 1e-6, vx
    assert abs(vy) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_uses_dirty_contact_before_raycast: PASSED")
    return True


def test_entity_world_collision_uses_dirty_bounds_contact_store():
    """Dirty terrain-bounds overlap should use the stored bounds contact directly before any fallback resample or raycast."""
    raycast_calls = 0
    model_calls = 0
    model_bounds_calls = []

    def fake_raycast(start, end):
        nonlocal raycast_calls
        raycast_calls += 1
        return None

    def fake_model_collision(*args, **kwargs):
        nonlocal model_calls
        model_calls += 1
        return None

    server = WulframServer.__new__(WulframServer)
    def fake_model_bounds_contact(*args, **kwargs):
        model_bounds_calls.append(args)
        return TerrainContact(
            position=(10.0, 0.0, 4.0),
            normal=(0.0, 0.0, 1.0),
            penetration=1.5,
            sector_index=0,
            cell=(0, 0),
        )

    server._terrain_grid_collision = SimpleNamespace(
        test_bounds_intersection=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy bounds gate should not run")),
        test_model_bounds_contact=fake_model_bounds_contact,
        raycast=fake_raycast,
        test_box_collision=lambda *args, **kwargs: None,
        test_model_collision=fake_model_collision,
    )
    server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
    server._get_entity_world_collision_model = lambda ctx: (
        [],
        SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
        5.0,
        0.0,
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (0.0, 0.0, 4.0)
    ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        10.0,
        0.0,
        4.0,
        1.0,
        0.0,
        -5.0,
    )

    assert model_calls == 0, model_calls
    assert raycast_calls == 0, raycast_calls
    assert model_bounds_calls, model_bounds_calls
    assert model_bounds_calls[0][0] == (10.0, 0.0, 4.0), model_bounds_calls[0]
    assert model_bounds_calls[0][1] == (10.0, 0.0, 4.0), model_bounds_calls[0]
    assert abs(px - 10.0) < 1e-6, px
    assert abs(py) < 1e-6, py
    assert pz > 5.0, pz
    assert abs(vx - 1.0) < 1e-6, vx
    assert abs(vy) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_uses_dirty_bounds_contact_store: PASSED")
    return True


def test_entity_world_collision_dirty_bounds_store_accepts_tiny_contact():
    """Dirty bounds-phase stored contacts should apply even when penetration is below the clean-path epsilon."""
    raycast_calls = 0

    def fake_raycast(start, end):
        nonlocal raycast_calls
        raycast_calls += 1
        return None

    server = WulframServer.__new__(WulframServer)

    def fake_model_bounds_contact(*args, **kwargs):
        return TerrainContact(
            position=(10.0, 0.0, 4.0),
            normal=(0.0, 0.0, 1.0),
            penetration=0.0001,
            sector_index=0,
            cell=(0, 0),
        )

    server._terrain_grid_collision = SimpleNamespace(
        test_model_bounds_contact=fake_model_bounds_contact,
        raycast=fake_raycast,
        test_box_collision=lambda *args, **kwargs: None,
        test_model_collision=lambda *args, **kwargs: None,
    )
    server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
    server._get_entity_world_collision_model = lambda ctx: (
        [],
        SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
        5.0,
        0.0,
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (0.0, 0.0, 4.0)
    ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        10.0,
        0.0,
        4.0,
        1.0,
        0.0,
        -5.0,
    )

    assert raycast_calls == 0, raycast_calls
    assert abs(px - 10.0) < 1e-6, px
    assert abs(py) < 1e-6, py
    assert pz > 4.0, pz
    assert abs(vx - 1.0) < 1e-6, vx
    assert abs(vy) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_dirty_bounds_store_accepts_tiny_contact: PASSED")
    return True


def test_entity_world_collision_dirty_bounds_store_uses_contact_point_radius_resolution():
    """Dirty stored bounds contacts should resolve from the stored contact point plus bounding radius, not SAT penetration depth."""
    server = WulframServer.__new__(WulframServer)

    def fake_model_bounds_contact(*args, **kwargs):
        return TerrainContact(
            position=(10.0, 0.0, 4.0),
            normal=(0.0, 0.0, 1.0),
            penetration=0.25,
            sector_index=0,
            cell=(0, 0),
        )

    server._terrain_grid_collision = SimpleNamespace(
        test_model_bounds_contact=fake_model_bounds_contact,
        raycast=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dirty bounds contact should win before raycast")),
        test_box_collision=lambda *args, **kwargs: None,
        test_model_collision=lambda *args, **kwargs: None,
    )
    server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
    server._get_entity_world_collision_model = lambda ctx: (
        [],
        SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
        5.0,
        0.0,
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (0.0, 0.0, 4.0)
    ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        10.0,
        0.0,
        4.0,
        1.0,
        0.0,
        -5.0,
    )

    assert abs(px - 10.0) < 1e-6, px
    assert abs(py) < 1e-6, py
    assert abs(pz - 9.01) < 1e-6, pz
    assert abs(vx - 1.0) < 1e-6, vx
    assert abs(vy) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_dirty_bounds_store_uses_contact_point_radius_resolution: PASSED")
    return True


def test_entity_world_collision_pathological_dirty_bounds_contact_falls_back_to_raycast():
    """Pathological dirty-bounds contacts should defer to terrain raycast instead of launching sideways."""
    raycast_calls = 0

    def fake_model_bounds_contact(*args, **kwargs):
        return TerrainContact(
            position=(10.0, 0.0, 4.0),
            normal=(0.0, -0.99995, 0.01),
            penetration=50.0,
            sector_index=0,
            cell=(0, 0),
        )

    def fake_raycast(start, end):
        nonlocal raycast_calls
        raycast_calls += 1
        return TerrainRaycastHit(
            position=(9.0, 0.0, 4.0),
            normal=(0.0, 0.0, 1.0),
            distance=1.0,
            sector_index=0,
            cell=(0, 0),
        )

    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = SimpleNamespace(
        test_model_bounds_contact=fake_model_bounds_contact,
        raycast=fake_raycast,
        test_box_collision=lambda *args, **kwargs: None,
        test_model_collision=lambda *args, **kwargs: None,
    )
    server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
    server._get_entity_world_collision_model = lambda ctx: (
        [],
        SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
        5.0,
        0.0,
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (0.0, 0.0, 4.0)
    ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        10.0,
        0.0,
        4.0,
        1.0,
        0.0,
        -5.0,
    )

    assert raycast_calls == 1, raycast_calls
    assert abs(px - 4.0) < 1e-6, px
    assert abs(py) < 1e-6, py
    assert 4.0 < pz < 5.0, pz
    assert ctx.debug_last_collision["kind"] == "terrain_dirty_raycast", ctx.debug_last_collision
    assert abs(vx - 1.0) < 1e-6, vx
    assert abs(vy) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_pathological_dirty_bounds_contact_falls_back_to_raycast: PASSED")
    return True


def test_entity_world_collision_far_dirty_bounds_contact_falls_back_to_raycast():
    """Dirty-bounds contacts with far-away stored contact points should defer to raycast instead of teleporting."""
    raycast_calls = 0

    def fake_model_bounds_contact(*args, **kwargs):
        return TerrainContact(
            position=(20.0, 0.0, 4.0),
            normal=(0.0, 0.0, 1.0),
            penetration=1.0,
            sector_index=0,
            cell=(0, 0),
        )

    def fake_raycast(start, end):
        nonlocal raycast_calls
        raycast_calls += 1
        return TerrainRaycastHit(
            position=(9.0, 0.0, 4.0),
            normal=(0.0, 0.0, 1.0),
            distance=1.0,
            sector_index=0,
            cell=(0, 0),
        )

    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = SimpleNamespace(
        test_model_bounds_contact=fake_model_bounds_contact,
        raycast=fake_raycast,
        test_box_collision=lambda *args, **kwargs: None,
        test_model_collision=lambda *args, **kwargs: None,
    )
    server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
    server._get_entity_world_collision_model = lambda ctx: (
        [],
        SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
        5.0,
        0.0,
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (0.0, 0.0, 4.0)
    ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        10.0,
        0.0,
        4.0,
        1.0,
        0.0,
        -5.0,
    )

    assert raycast_calls == 1, raycast_calls
    assert abs(px - 4.0) < 1e-6, px
    assert abs(py) < 1e-6, py
    assert 4.0 < pz < 5.0, pz
    assert ctx.debug_last_collision["kind"] == "terrain_dirty_raycast", ctx.debug_last_collision
    assert abs(vx - 1.0) < 1e-6, vx
    assert abs(vy) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_far_dirty_bounds_contact_falls_back_to_raycast: PASSED")
    return True


def test_entity_world_collision_horizontal_dirty_bounds_contact_falls_back_to_raycast():
    """Dirty-bounds contacts with effectively horizontal normals and large stored penetration should defer to raycast."""
    raycast_calls = 0

    def fake_model_bounds_contact(*args, **kwargs):
        return TerrainContact(
            position=(10.0, 0.0, 4.0),
            normal=(-0.96, 0.28, 0.0),
            penetration=14.0,
            sector_index=0,
            cell=(0, 0),
        )

    def fake_raycast(start, end):
        nonlocal raycast_calls
        raycast_calls += 1
        return TerrainRaycastHit(
            position=(9.0, 0.0, 4.0),
            normal=(0.0, 0.0, 1.0),
            distance=1.0,
            sector_index=0,
            cell=(0, 0),
        )

    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = SimpleNamespace(
        test_model_bounds_contact=fake_model_bounds_contact,
        raycast=fake_raycast,
        test_box_collision=lambda *args, **kwargs: None,
        test_model_collision=lambda *args, **kwargs: None,
    )
    server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
    server._get_entity_world_collision_model = lambda ctx: (
        [],
        SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
        5.0,
        0.0,
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (0.0, 0.0, 4.0)
    ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        10.0,
        0.0,
        4.0,
        1.0,
        0.0,
        -5.0,
    )

    assert raycast_calls == 1, raycast_calls
    assert abs(px - 4.0) < 1e-6, px
    assert abs(py) < 1e-6, py
    assert 4.0 < pz < 5.0, pz
    assert ctx.debug_last_collision["kind"] == "terrain_dirty_raycast", ctx.debug_last_collision
    assert abs(vx - 1.0) < 1e-6, vx
    assert abs(vy) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_horizontal_dirty_bounds_contact_falls_back_to_raycast: PASSED")
    return True


def test_entity_world_collision_dirty_bounds_phase_uses_xy_broadphase():
    """Dirty bounds broadphase should ignore Z separation, matching the decompile's XY-only terrain bounds test."""
    tri = (
        (0.0, 0.0, 100.0),
        (5.0, 0.0, 100.0),
        (0.0, 5.0, 100.0),
    )
    aabb_min = (0.0, 0.0, 0.0)
    aabb_max = (5.0, 5.0, 1.0)

    assert TerrainGridCollision._triangle_overlaps_aabb(tri, aabb_min, aabb_max) is False
    assert TerrainGridCollision._triangle_overlaps_xy_bounds(
        tri,
        aabb_min[0],
        aabb_min[1],
        aabb_max[0],
        aabb_max[1],
    ) is True
    print("test_entity_world_collision_dirty_bounds_phase_uses_xy_broadphase: PASSED")
    return True


def test_dirty_bounds_contact_helpers_skip_triangle_prefilter():
    """Helper-backed dirty bounds contact should iterate selected cells directly without extra triangle prefilters."""
    terrain = SimpleNamespace(
        cell_x=1.0,
        cell_z=1.0,
        world_w=2.0,
        world_h=2.0,
        num_x=2,
        num_z=2,
        _get_raw_height=lambda x, y: 0.0,
    )
    grid = TerrainGridCollision(terrain, 0.0, sector_rows=1, sector_cols=1)
    tri = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    grid._iter_aabb_sectors = lambda aabb_min, aabb_max: [SimpleNamespace(index=0)]
    grid._iter_sector_cells = lambda aabb_min, aabb_max, sector: [(0, 0)]
    grid._iter_cell_triangles = lambda cell_x, cell_y: [tri]
    grid._triangle_overlaps_xy_bounds = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("xy prefilter should not run")
    )
    grid._triangle_box_contact = lambda tri_local, half_extents: ((0.0, 0.0, 1.0), 1.0)

    box_contact = grid.test_box_bounds_contact(
        (0.5, 0.5, 0.0),
        (0.5, 0.5, 0.0),
        (1.0, 1.0, 1.0),
        0.0,
        1.0,
    )
    assert box_contact is not None

    grid._triangle_overlaps_aabb = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("3d prefilter should not run")
    )
    grid._triangle_cbsp_contact = lambda tri_local, vertices, cbsp_tree, bounding_radius: (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        1.0,
    )
    model_contact = grid.test_model_bounds_contact(
        (0.5, 0.5, 0.0),
        (0.5, 0.5, 0.0),
        0.0,
        [],
        object(),
        1.0,
    )
    assert model_contact is not None
    print("test_dirty_bounds_contact_helpers_skip_triangle_prefilter: PASSED")
    return True


def test_terrain_cell_triangles_match_decompile_order():
    """Terrain cell triangle splitting should match the original quad-triangle order, not just the diagonal parity."""
    terrain = SimpleNamespace(
        cell_x=1.0,
        cell_z=1.0,
        world_w=3.0,
        world_h=3.0,
        num_x=3,
        num_z=3,
        _get_raw_height=lambda x, y: 0.0,
    )
    grid = TerrainGridCollision(terrain, 0.0, sector_rows=1, sector_cols=1)

    even_tris = list(grid._iter_cell_triangles(0, 0))
    odd_tris = list(grid._iter_cell_triangles(1, 0))

    assert even_tris == [
        ((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
    ], even_tris
    assert odd_tris == [
        ((1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        ((2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
    ], odd_tris
    print("test_terrain_cell_triangles_match_decompile_order: PASSED")
    return True


def test_terrain_raycast_patch_traverse_uses_start_to_end_sector_sweep():
    """Terrain patch traversal should sweep the 3x3 sector rectangle from start sector toward end sector."""
    terrain = SimpleNamespace(
        cell_x=1.0,
        cell_z=1.0,
        world_w=9.0,
        world_h=9.0,
        num_x=10,
        num_z=10,
        _get_raw_height=lambda x, y: 0.0,
    )
    grid = TerrainGridCollision(terrain, 0.0)
    visited = []

    def fake_sector_raycast(start, end, sector):
        visited.append((sector.row, sector.col))
        return None

    grid._raycast_sector_cells = fake_sector_raycast
    hit = grid.raycast((8.5, 0.5, 1.0), (3.5, 8.5, -1.0))
    assert hit is None
    assert visited == [
        (2, 0),
        (2, 1),
        (2, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ], visited
    print("test_terrain_raycast_patch_traverse_uses_start_to_end_sector_sweep: PASSED")
    return True


def test_terrain_patch_raycast_cells_uses_decompile_dda_order():
    """Per-patch terrain raycast should visit cells in the decompile DDA order and stop on first hit."""
    terrain = SimpleNamespace(
        cell_x=1.0,
        cell_z=1.0,
        world_w=3.0,
        world_h=3.0,
        num_x=4,
        num_z=4,
        _get_raw_height=lambda x, y: 0.0,
    )
    grid = TerrainGridCollision(terrain, 0.0, sector_rows=1, sector_cols=1)
    sector = grid.sectors[0]
    visited = []
    expected_hit = TerrainRaycastHit(
        position=(1.5, 1.5, 0.0),
        normal=(0.0, 0.0, 1.0),
        sector_index=sector.index,
        cell=(1, 1),
        distance=math.sqrt(3.125),
    )

    def fake_cell_raycast(start, end, sector_arg, cell_x, cell_y):
        visited.append((cell_x, cell_y))
        if (cell_x, cell_y) == (1, 1):
            return expected_hit
        return None

    grid._raycast_cell_triangles = fake_cell_raycast
    hit = grid._raycast_sector_cells((0.25, 0.25, 1.0), (2.75, 2.75, -1.0), sector)
    assert hit == expected_hit, hit
    assert visited == [
        (0, 0),
        (1, 0),
        (1, 1),
    ], visited
    print("test_terrain_patch_raycast_cells_uses_decompile_dda_order: PASSED")
    return True


def test_terrain_patch_raycast_cells_uses_decompile_axis_flag_step_policy():
    """Per-patch terrain raycast should follow the decompile axis-flag step policy in x-only cases."""
    terrain = SimpleNamespace(
        cell_x=1.0,
        cell_z=1.0,
        world_w=3.0,
        world_h=3.0,
        num_x=4,
        num_z=4,
        _get_raw_height=lambda x, y: 0.0,
    )
    grid = TerrainGridCollision(terrain, 0.0, sector_rows=1, sector_cols=1)
    sector = grid.sectors[0]
    visited = []

    def fake_cell_raycast(start, end, sector_arg, cell_x, cell_y):
        visited.append((cell_x, cell_y))
        return None

    grid._raycast_cell_triangles = fake_cell_raycast
    hit = grid._raycast_sector_cells((0.25, 0.25, 1.0), (2.75, 0.25, -1.0), sector)
    assert hit is None
    assert visited == [
        (0, 0),
        (1, 0),
        (2, 0),
    ], visited
    print("test_terrain_patch_raycast_cells_uses_decompile_axis_flag_step_policy: PASSED")
    return True


def test_entity_world_collision_uses_persistent_reference_pos_for_dirty_branch():
    """Dirty terrain-ray testing should use the persistent collision reference position, not only last frame pos."""
    raycast_calls = []

    def fake_raycast(start, end):
        raycast_calls.append((start, end))
        return None

    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = SimpleNamespace(
        test_bounds_intersection=lambda *args, **kwargs: False,
        raycast=fake_raycast,
        test_box_collision=lambda *args, **kwargs: None,
        test_model_collision=lambda *args, **kwargs: None,
    )
    server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
    server._get_entity_world_collision_model = lambda ctx: None

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (9.0, 0.0, 5.0)
    ctx.world_collision_ref_pos = (0.0, 0.0, 5.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        10.0,
        0.0,
        5.0,
        0.0,
        0.0,
        0.0,
    )

    assert raycast_calls == [((0.0, 0.0, 5.0), (10.0, 0.0, 5.0))], raycast_calls
    assert ctx.world_collision_bounds_dirty is True, ctx.world_collision_bounds_dirty
    assert ctx.world_collision_ref_pos == (10.0, 0.0, 5.0), ctx.world_collision_ref_pos
    assert (px, py, pz, vx, vy, vz) == (10.0, 0.0, 5.0, 0.0, 0.0, 0.0)
    print("test_entity_world_collision_uses_persistent_reference_pos_for_dirty_branch: PASSED")
    return True


def test_entity_world_collision_refreshes_reference_on_clean_contact():
    """Repeated flat-ground clean contacts should refresh the dirty reference instead of escalating into raycast."""
    raycast_calls = []

    def fake_raycast(start, end):
        raycast_calls.append((start, end))
        return None

    def fake_box_collision(center, half_extents, heading):
        return SimpleNamespace(
            position=(center[0], center[1], center[2] - half_extents[2]),
            normal=(0.0, 0.0, 1.0),
            penetration=0.1,
        )

    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = SimpleNamespace(
        test_bounds_intersection=lambda *args, **kwargs: False,
        raycast=fake_raycast,
        test_box_collision=fake_box_collision,
        test_model_collision=lambda *args, **kwargs: None,
    )
    server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
    server._get_entity_world_collision_model = lambda ctx: None
    server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1.0
    server._get_static_separation_from_contact = lambda entity_pos, contact_point: 0.0

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (0.0, 0.0, 5.0)
    ctx.world_collision_ref_pos = (0.0, 0.0, 5.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        0.75,
        0.0,
        5.0,
        0.0,
        0.0,
        0.0,
    )
    assert ctx.world_collision_bounds_dirty is False, ctx.world_collision_bounds_dirty
    assert raycast_calls == [], raycast_calls
    assert abs(px - 0.75) < 1e-6, px
    assert abs(pz - 5.1) < 1e-6, pz
    assert ctx.world_collision_ref_pos == (px, py, pz), ctx.world_collision_ref_pos

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        1.5,
        0.0,
        5.0,
        0.0,
        0.0,
        0.0,
    )
    assert ctx.world_collision_bounds_dirty is False, ctx.world_collision_bounds_dirty
    assert raycast_calls == [], raycast_calls
    assert abs(px - 1.5) < 1e-6, px
    assert abs(pz - 5.1) < 1e-6, pz
    assert ctx.world_collision_ref_pos == (px, py, pz), ctx.world_collision_ref_pos
    print("test_entity_world_collision_refreshes_reference_on_clean_contact: PASSED")
    return True


def test_entity_world_collision_dirty_threshold_uses_mesh_min_half_extent():
    """Dirty threshold should use the assigned collision mesh min half-extent, including Z, per the decompile."""
    server = WulframServer.__new__(WulframServer)
    server._building_collision = SimpleNamespace(
        available=True,
        models={
            "tank_2": SimpleNamespace(
                collision_mesh=SimpleNamespace(
                    vertices=[
                        SimpleNamespace(x=-6.0, y=0.0, z=0.0),
                        SimpleNamespace(x=6.0, y=0.0, z=0.0),
                        SimpleNamespace(x=0.0, y=-5.0, z=0.0),
                        SimpleNamespace(x=0.0, y=5.0, z=0.0),
                        SimpleNamespace(x=0.0, y=0.0, z=-2.0),
                        SimpleNamespace(x=0.0, y=0.0, z=2.0),
                    ]
                )
            )
        },
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(team_id=1),
        entity_id=0x14EA,
        entity_type=EntityType.TANK,
    )
    threshold_sq = server._get_entity_dirty_threshold_sq(ctx, (4.0, 4.0, 4.0))

    assert abs(threshold_sq - 2.56) < 1e-6, threshold_sq
    print("test_entity_world_collision_dirty_threshold_uses_mesh_min_half_extent: PASSED")
    return True


def test_entity_world_collision_dirty_raycast_uses_contact_separation():
    """Dirty terrain-ray response should use contact-derived static separation instead of a fixed epsilon."""
    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = SimpleNamespace(
        test_bounds_intersection=lambda *args, **kwargs: False,
        raycast=lambda start, end: TerrainRaycastHit(
            position=(4.0, 0.0, 5.0),
            normal=(0.0, 0.0, 1.0),
            sector_index=0,
            cell=(0, 0),
            distance=4.0,
        ),
        test_box_collision=lambda *args, **kwargs: None,
        test_model_collision=lambda *args, **kwargs: None,
    )
    server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
    server._get_entity_world_collision_model = lambda ctx: (
        [],
        SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
        5.0,
        0.0,
    )
    server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1e-8

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (0.0, 0.0, 5.0)
    ctx.world_collision_ref_pos = (0.0, 0.0, 5.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        10.0,
        0.0,
        5.0,
        3.0,
        0.0,
        -4.0,
    )

    assert abs(px + 1.0) < 1e-6, px
    assert abs(py) < 1e-6, py
    assert abs(pz - 5.15) < 1e-6, pz
    assert abs(vx - 3.0) < 1e-6, vx
    assert abs(vy) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_dirty_raycast_uses_contact_separation: PASSED")
    return True


def test_entity_world_collision_dirty_raycast_uses_decompile_degenerate_threshold():
    """Dirty terrain-ray contact should take the decompile's <=0.001 degenerate path, not normalize tiny rays."""
    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = SimpleNamespace(
        test_bounds_intersection=lambda *args, **kwargs: False,
        raycast=lambda start, end: TerrainRaycastHit(
            position=(9.0, 0.0, 5.0),
            normal=(0.0, 0.0, 1.0),
            sector_index=0,
            cell=(0, 0),
            distance=1.0,
        ),
        test_box_collision=lambda *args, **kwargs: None,
        test_model_collision=lambda *args, **kwargs: None,
    )
    server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
    server._get_entity_world_collision_model = lambda ctx: (
        [],
        SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
        5.0,
        0.0,
    )
    server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1e-8

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (10.0, 0.0, 5.0)
    ctx.world_collision_ref_pos = (10.0005, 0.0, 5.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        10.0,
        0.0,
        5.0,
        3.0,
        0.0,
        -4.0,
    )

    assert abs(px - 9.0) < 1e-6, px
    assert abs(py) < 1e-6, py
    assert abs(pz - 10.03) < 1e-6, pz
    assert abs(vx - 3.0) < 1e-6, vx
    assert abs(vy) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_dirty_raycast_uses_decompile_degenerate_threshold: PASSED")
    return True


def test_entity_world_collision_static_separation_matches_decompile_clamp():
    """Static separation should use the decompile's distance*0.03 rule clamped to [0.01, 0.5]."""
    server = WulframServer.__new__(WulframServer)

    assert abs(server._get_static_separation_from_contact((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)) - 0.3) < 1e-6
    assert abs(server._get_static_separation_from_contact((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)) - 0.5) < 1e-6
    assert abs(server._get_static_separation_from_contact((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)) - 0.01) < 1e-6
    print("test_entity_world_collision_static_separation_matches_decompile_clamp: PASSED")
    return True


def test_entity_world_collision_clean_path_uses_single_contact_store():
    """Clean-bounds entity/world contact should apply the first stored contact once, not iterate multiple pushes."""
    server = WulframServer.__new__(WulframServer)
    server._get_entity_world_half_extents = lambda ctx: (1.0, 1.0, 1.0)
    server._get_entity_world_collision_model = lambda ctx: None
    server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

    calls = []

    def fake_box_collision(center, half_extents, heading):
        calls.append(center)
        if len(calls) == 1:
            return TerrainContact(
                position=(0.0, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
                penetration=1.0,
                sector_index=0,
                cell=(0, 0),
            )
        return TerrainContact(
            position=(0.0, 0.0, 2.0),
            normal=(0.0, 0.0, 1.0),
            penetration=5.0,
            sector_index=0,
            cell=(0, 1),
        )

    server._terrain_grid_collision = SimpleNamespace(
        test_box_collision=fake_box_collision,
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.player_pos = (0.0, 0.0, 0.0)

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -3.0,
    )

    assert len(calls) == 1, calls
    assert abs(px) < 1e-6, px
    assert abs(py) < 1e-6, py
    assert abs(pz - 1.01) < 1e-6, pz
    assert abs(vx) < 1e-6, vx
    assert abs(vy) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_clean_path_uses_single_contact_store: PASSED")
    return True


def test_roster_entry_stays_tcp_only():
    """ADD_TO_ROSTER must not leak onto UDP when TCP is unavailable."""
    sent_udp = []

    server = WulframServer.__new__(WulframServer)
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: sent_udp.append((payload, addr)))

    target_session = Session()
    target_session.udp_addr = ("172.18.84.98", 62479)
    target_ctx = ClientContext(
        client_id=1,
        client_addr=("172.18.84.98", 50000),
        session=target_session,
        entity_id=0x14EA,
    )
    target_ctx.tcp_handler = None
    target_ctx.known_roster_ids = set()

    player_session = Session()
    player_session.player_id = 0x053B
    player_session.username = "easystar"
    player_session.team_id = 1
    player_ctx = ClientContext(
        client_id=3,
        client_addr=("172.18.84.98", 50002),
        session=player_session,
        entity_id=0x053B,
    )
    player_ctx.kills = 0
    player_ctx.deaths = 0

    server._send_roster_entry(target_ctx, player_ctx)

    assert sent_udp == [], sent_udp
    assert target_ctx.known_roster_ids == set(), target_ctx.known_roster_ids
    print("test_roster_entry_stays_tcp_only: PASSED")
    return True


def test_broadcast_player_stats_stays_tcp_only():
    """UPDATE_STATS must not fall back to UDP when a client TCP stream is unavailable."""
    sent_udp = []

    server = WulframServer.__new__(WulframServer)
    server.udp_handler = SimpleNamespace(send_to=lambda payload, addr: sent_udp.append((payload, addr)))

    target_session = Session()
    target_session.udp_addr = ("172.18.84.98", 62479)
    target_ctx = ClientContext(
        client_id=1,
        client_addr=("172.18.84.98", 50000),
        session=target_session,
        entity_id=0x14EA,
    )
    target_ctx.tcp_handler = None
    server._snapshot_clients = lambda: [target_ctx]

    player_session = Session()
    player_session.player_id = 0x053B
    player_session.team_id = 1
    player_ctx = ClientContext(
        client_id=3,
        client_addr=("172.18.84.98", 50002),
        session=player_session,
        entity_id=0x053B,
    )
    player_ctx.kills = 2
    player_ctx.deaths = 1

    server._broadcast_player_stats(player_ctx)

    assert sent_udp == [], sent_udp
    print("test_broadcast_player_stats_stays_tcp_only: PASSED")
    return True


def test_control_pos_exact_reset_targets_specific_client():
    """Control `pos c<id> ...` must reset authoritative motion state exactly."""
    server = SimpleNamespace(
        clients_lock=threading.Lock(),
        clients={},
        _get_network_tick=lambda _ctx: 777,
    )

    session = Session()
    session.udp_addr = ("127.0.0.1", 30000)
    ctx = ClientContext(
        client_id=7,
        client_addr=("127.0.0.1", 30000),
        session=session,
        entity_id=0x1234,
    )
    ctx.player_pos = (10.0, 20.0, 30.0)
    ctx.player_vel = (4.0, 5.0, 6.0)
    ctx.player_speed = 11.0
    ctx.player_heading = 1.0
    ctx.player_yaw = -1.0
    ctx.player_angular_vel = 0.7
    ctx.angular_vel_yaw = 0.5
    ctx.world_collision_ref_pos = (1.0, 2.0, 3.0)
    ctx.world_collision_bounds_dirty = True
    ctx.last_sent_pos = (99.0, 98.0, 97.0)
    ctx.last_sent_vel = (1.0, 1.0, 1.0)
    ctx.last_sent_yaw = 0.3
    ctx.last_state_sync_vel = (9.0, 9.0, 9.0)
    ctx.last_state_sync_rot = (0.1, 0.2, 0.3)
    ctx.injected_input = (1.0, -1.0)
    ctx.injected_turn = 0.4
    ctx.prev_raw_turn_input = 0.4
    ctx.player_pose["pos"] = ctx.player_pos
    ctx.player_pose["vel"] = ctx.player_vel
    ctx.player_pose["yaw"] = ctx.player_yaw
    ctx.weapon_system = WeaponSystem()
    ctx.weapon_system.behavior_slots[1] = 1.0
    ctx.weapon_system.behavior_slots[2] = 1.0
    ctx.weapon_system.prev_fire_state = 1.0
    ctx.weapon_system.fire_cooldown = 0.5
    server.clients[ctx.client_id] = ctx

    control = ControlServer(port=0)
    control.server = server

    result = control._cmd_player_pos(["c7", "100", "200", "300", "45"])

    assert "client 7" in result.lower(), result
    assert ctx.player_pos == (100.0, 200.0, 300.0), ctx.player_pos
    assert ctx.player_vel == (0.0, 0.0, 0.0), ctx.player_vel
    assert ctx.player_speed == 0.0, ctx.player_speed
    assert abs(ctx.player_heading - math.radians(45.0)) < 1e-6, ctx.player_heading
    assert abs(ctx.player_yaw + math.radians(45.0)) < 1e-6, ctx.player_yaw
    assert ctx.player_angular_vel == 0.0, ctx.player_angular_vel
    assert ctx.angular_vel_yaw == 0.0, ctx.angular_vel_yaw
    assert ctx.world_collision_ref_pos == (100.0, 200.0, 300.0), ctx.world_collision_ref_pos
    assert ctx.world_collision_bounds_dirty is False, ctx.world_collision_bounds_dirty
    assert ctx.last_sent_pos == (100.0, 200.0, 300.0), ctx.last_sent_pos
    assert ctx.last_sent_vel == (0.0, 0.0, 0.0), ctx.last_sent_vel
    assert abs(ctx.last_sent_yaw + math.radians(45.0)) < 1e-6, ctx.last_sent_yaw
    assert ctx.last_state_sync_vel is None, ctx.last_state_sync_vel
    assert ctx.last_state_sync_rot is None, ctx.last_state_sync_rot
    assert ctx.injected_input is None, ctx.injected_input
    assert ctx.injected_turn is None, ctx.injected_turn
    assert ctx.prev_raw_turn_input == 0.0, ctx.prev_raw_turn_input
    assert ctx.player_pose["pos"] == (100.0, 200.0, 300.0), ctx.player_pose["pos"]
    assert ctx.player_pose["vel"] == (0.0, 0.0, 0.0), ctx.player_pose["vel"]
    assert abs(ctx.player_pose["yaw"] + math.radians(45.0)) < 1e-6, ctx.player_pose["yaw"]
    assert max(abs(v) for v in ctx.weapon_system.behavior_slots) == 0.0, ctx.weapon_system.behavior_slots
    assert ctx.weapon_system.prev_fire_state == 0.0, ctx.weapon_system.prev_fire_state
    assert ctx.weapon_system.fire_cooldown == 0.0, ctx.weapon_system.fire_cooldown
    print("test_control_pos_exact_reset_targets_specific_client: PASSED")
    return True


def test_control_heading_set_preserves_yaw_sign_convention():
    """Control `heading set` must keep player_yaw = -player_heading."""
    server = SimpleNamespace(
        clients_lock=threading.Lock(),
        clients={},
    )

    ctx = ClientContext(
        client_id=7,
        client_addr=("127.0.0.1", 30000),
        session=Session(),
        entity_id=0x1234,
    )
    ctx.player_heading = 0.0
    ctx.player_yaw = 0.0
    ctx.player_pose["yaw"] = 0.0
    ctx.vehicle_physics = SimpleNamespace(heading=0.0, _angular_velocity=1.25)
    server.clients[ctx.client_id] = ctx

    control = ControlServer(port=0)
    control.server = server

    result = control._cmd_heading(["set", "45", "c7"])

    assert "45" in result, result
    assert abs(ctx.player_heading - math.radians(45.0)) < 1e-6, ctx.player_heading
    assert abs(ctx.player_yaw + math.radians(45.0)) < 1e-6, ctx.player_yaw
    assert abs(ctx.player_pose["yaw"] + math.radians(45.0)) < 1e-6, ctx.player_pose["yaw"]
    assert abs(ctx.vehicle_physics.heading - math.radians(45.0)) < 1e-6, ctx.vehicle_physics.heading
    assert ctx.vehicle_physics._angular_velocity == 0.0, ctx.vehicle_physics._angular_velocity
    print("test_control_heading_set_preserves_yaw_sign_convention: PASSED")
    return True


def test_solo_local_player_keepalive_shape_triggers_og_state_request_gate():
    """The keepalive packet must decode as entity_count=1 with the local player OID.

    OG's organic STATE_REQUEST trigger at Replication.c:1173-1177 is
    `if (entity_count == 1 && final_entity_ptr == g_local_player_entity)`.
    The keepalive uses build_update_array_player_update which always emits
    exactly one entity; lock that invariant here so a builder change can't
    silently add a dummy entity and break the organic correction loop.
    """
    from wulfram.packets import build_update_array_player_update, build_behavior_packet
    client_root = Path(__file__).resolve().parent.parent / "client"
    if str(client_root) not in sys.path:
        sys.path.insert(0, str(client_root))
    from wulfram_client.network.decoder import decode_update_array
    from wulfram_client.network.behavior import parse_behavior

    behavior = parse_behavior(build_behavior_packet())
    entity_id = 0x14EA
    payload = build_update_array_player_update(
        tick=0x1234,
        entity_id=entity_id,
        pos=(5100.0, 4300.0, 5.0),
        vel=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0),
        include_pos=True,
        include_vel=True,
        include_rot=True,
        include_local_state=True,
        weapon_id=2,
    )
    _, _local, entities = decode_update_array(payload, behavior_config=behavior)
    assert len(entities) == 1, f"keepalive must be solo-local, got {len(entities)} entities"
    assert entities[0].entity_id == entity_id
    assert entities[0].position is not None, "pos required — satisfies Entity_apply_network_transform assert"
    assert entities[0].rotation is not None, "rot required — satisfies Entity_apply_network_transform assert"
    print("test_solo_local_player_keepalive_shape_triggers_og_state_request_gate: PASSED")
    return True


def test_view_update_pos_without_rot_clamps_to_safe_shape():
    """OG's Entity_apply_network_transform exits on pos-without-rot for non-static entities."""
    from wulfram.packets import build_view_update_player_update, build_view_update_multi, build_behavior_packet
    client_root = Path(__file__).resolve().parent.parent / "client"
    if str(client_root) not in sys.path:
        sys.path.insert(0, str(client_root))
    from wulfram_client.network.decoder import decode_view_update
    from wulfram_client.network.behavior import parse_behavior

    behavior = parse_behavior(build_behavior_packet())

    # Single-entity builder: pos=True, rot=False requested — should coerce rot=True.
    payload = build_view_update_player_update(
        tick=0x1234,
        entity_id=0x14EA,
        pos=(5100.0, 4300.0, 5.0),
        vel=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.5),
        include_pos=True,
        include_vel=False,
        include_rot=False,
        include_local_state=False,
    )
    _, _, _local, entities = decode_view_update(payload, behavior_config=behavior)
    assert len(entities) == 1
    assert entities[0].position is not None, "pos must still be included"
    assert entities[0].rotation is not None, "rotation must be coerced on when pos is requested"

    # Multi-entity builder: per-entity clamp.
    entity = {
        "entity_id": 0x14EA,
        "is_manned": True,
        "pos": (5100.0, 4300.0, 5.0),
        "vel": (0.0, 0.0, 0.0),
        "rot": (0.0, 0.0, 0.5),
        "include_pos": True,
        "include_vel": False,
        "include_rot": False,
    }
    payload = build_view_update_multi(
        tick=0x1234,
        include_local_state=False,
        entities=[entity],
    )
    _, _, _local, entities = decode_view_update(payload, behavior_config=behavior)
    assert len(entities) == 1
    assert entities[0].position is not None
    assert entities[0].rotation is not None
    print("test_view_update_pos_without_rot_clamps_to_safe_shape: PASSED")
    return True


def test_players_json_includes_transport_addresses():
    """Control `players json` should expose client and UDP addresses for diagnostics."""
    server = SimpleNamespace(
        clients_lock=threading.Lock(),
        clients={},
    )
    session = Session()
    session.username = "Player9"
    session.team_id = 1
    session.entity_id = 0x541
    session.udp_addr = ("10.10.10.2", 52732)
    session.phase = Session().phase.__class__.IN_GAME
    ctx = ClientContext(
        client_id=9,
        client_addr=("10.10.10.2", 52731),
        session=session,
        entity_id=0x541,
    )
    ctx.player_pos = (5172.77, 5093.27, 5.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_heading = math.radians(169.8)
    ctx.action_packet_count = 3
    ctx.nonzero_move_input_count = 1
    ctx.state_request_count = 2
    ctx.state_sync_reply_count = 2
    ctx.state_sync_view_reply_count = 2
    ctx.last_decoded_input = {"fwd": 0.5, "strafe": 0.0}
    server.clients[ctx.client_id] = ctx

    control = ControlServer(port=0)
    control.server = server

    payload = control._cmd_players(["json"])
    entries = json.loads(payload)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["client_addr"] == ["10.10.10.2", 52731]
    assert entry["udp_addr"] == ["10.10.10.2", 52732]
    assert entry["telemetry"]["action_packets"] == 3
    assert entry["telemetry"]["nonzero_move_inputs"] == 1
    assert entry["telemetry"]["state_requests"] == 2
    assert entry["telemetry"]["state_sync_replies"] == 2
    assert entry["telemetry"]["state_sync_view_replies"] == 2
    assert entry["telemetry"]["last_input"]["fwd"] == 0.5
    print("test_players_json_includes_transport_addresses: PASSED")
    return True


def main():
    print("=" * 60)
    print("Handler Tests")
    print("=" * 60)

    tests = [
        test_decode_lp_string_basic,
        test_decode_lp_string_offset,
        test_decode_lp_string_empty,
        test_decode_lp_string_truncated,
        test_handlers_import,
        test_input_sync_diagnosis_distinguishes_idle_snapback_from_correction_failure,
        test_input_sync_diagnosis_reports_movement_without_targeted_corrections,
        test_spawn_override_wins_over_map_spawn_points,
        test_spawn_at_point_honors_clicked_pad_when_default_configured,
        test_map_entity_z_aligns_buried_entities_to_terrain,
        test_map_entity_z_preserves_elevated_entities,
        test_control_pose_reset_updates_ground_override,
        test_pulse_shell_default_spawn_uses_recovered_muzzle_offset,
        test_remote_spawn_points_use_udp_not_tcp,
        test_send_initial_game_data_og_bootstrap_order,
        test_remote_want_updates_suppresses_empty_tcp_update_array,
        test_build_chat_message_comm_layout,
        test_build_update_array_remote_heartbeat_shape,
        test_server_remote_heartbeat_helper_keeps_full_local_state,
        test_server_remote_heartbeat_helper_pre_state_request_is_spawn_safe,
        test_remote_state_sync_reply_uses_safe_local_player_shape_when_ready,
        test_remote_state_sync_reply_stays_spawn_safe_immediately_after_spawn,
        test_remote_state_sync_reply_stays_spawn_safe_after_spawn_delay,
        test_remote_state_sync_reply_stays_safe_without_post_spawn_input_after_delay,
        test_remote_state_sync_reply_emits_view_update_with_request_timestamp,
        test_loopback_state_sync_reply_keeps_request_timestamp,
        test_remote_state_sync_reply_uses_request_aligned_authoritative_pose,
        test_remote_promoted_heartbeat_stays_short_form_safe,
        test_remote_state_sync_reply_keeps_full_motion_when_stable,
        test_local_player_sync_rotation_uses_heading_not_player_yaw,
        test_send_tank_uses_local_player_sync_rotation,
        test_spawn_wf_minimal_uses_local_player_sync_rotation,
        test_player_body_rotation_preserves_pitch_for_remote_entities,
        test_remote_sync_heartbeat_helper_uses_heading_not_player_yaw,
        test_remote_spawn_bootstrap_heartbeat_uses_safe_rot_only_shape,
        test_state_request_does_not_overwrite_client_tick_offset,
        test_remote_udp_ping_request_gets_og_safe_reply,
        test_udp_ping_reply_default_policy_is_loopback_only,
        test_jump_velocity_update_packet_uses_spawn_safe_local_state_for_remote_og,
        test_translation_velocity_quantizer_matches_decompile_defaults,
        test_server_remote_local_state_kwargs_use_full_tank_shape,
        test_server_remote_entity_packets_use_safe_local_state_after_promotion,
        test_server_remote_projectile_spawn_uses_viewer_local_state,
        test_server_remote_projectile_update_uses_safe_local_state_after_promotion,
        test_loopback_projectile_update_stays_entity_only,
        test_server_remote_player_info_uses_spawn_safe_local_state,
        test_remote_player_info_packet_short_local_state_layout,
        test_remote_spawn_entry_transition_sends_canonical_packets,
        test_spawn_entry_transition_stays_off_by_default,
        test_control_game_clock_builder_matches_packet_signature,
        test_weapon_system_og_direct_trigger_slot_fires_pulse_shell,
        test_weapon_system_held_fire_repeats_on_cooldown,
        test_send_entity_create_uses_udp_only,
        test_transient_fx_stays_off_for_remote_og_by_default,
        test_transient_fx_can_be_enabled_for_remote_clients,
        test_entity_create_uses_spawn_safe_local_state_for_og_viewer,
        test_remote_player_update_uses_spawn_safe_viewer_local_state,
        test_loopback_entity_create_decodes_roundtrip,
        test_loopback_remote_player_update_decodes_roundtrip,
        test_loopback_heartbeat_decodes_roundtrip,
        test_server_network_strafe_decode_matches_og_sign,
        test_remote_client_promotes_full_local_state_after_spawn_delay,
        test_remote_client_suppresses_periodic_spawn_safe_heartbeat_until_promoted,
        test_remote_client_suppresses_spawn_bootstrap_heartbeat_until_promoted,
        test_remote_client_does_not_promote_full_local_state_on_heartbeat_reason,
        test_remote_respawn_restores_promoted_local_state_after_spawn,
        test_remote_initial_spawn_keeps_minimal_path_when_never_promoted,
        test_server_tank_motion_uses_low_speed_mobility_factor,
        test_server_motion_clamps_to_move_adjust,
        test_server_motion_reclamps_below_ground_after_collision_response,
        test_ghost_rejoin_skips_loopback_clients,
        test_projectile_world_hit_skips_aabb_for_mesh_backed_building,
        test_projectile_world_hit_prefers_closest_building_before_terrain,
        test_projectile_world_hit_clips_static_world_raycast_to_terrain,
        test_projectile_world_hit_prefers_terrain_when_static_world_reports_farther_hit,
        test_projectile_world_hit_uses_exact_mesh_raycast_position,
        test_static_world_raycast_uses_quadtree_front_to_back_order,
        test_static_world_raycast_uses_point_query_for_zero_horizontal_direction,
        test_static_world_quadtree_uses_bounding_radius_for_overlap_distribution,
        test_static_world_raycast_stops_once_endpoint_leaf_is_reached,
        test_static_world_raycast_node_cull_uses_signbit_zero_side_semantics,
        test_segment_raycast_cbsp_tree_uses_split_plane_normal,
        test_segment_hits_cbsp_tree_detects_plane_hit,
        test_triangle_cbsp_contact_uses_node_split_normal,
        test_triangle_cbsp_contact_uses_entity_bounding_radius_for_plane_reject,
        test_box_collision_returns_first_contact_in_grid_order,
        test_model_collision_returns_first_contact_in_grid_order,
        test_triangle_cbsp_contact_returns_first_leaf_hit,
        test_building_collision_skips_aabb_for_mesh_backed_building,
        test_repair_pad_collision_does_not_block_vehicle_movement,
        test_building_collision_team_variant_matches_client_helper,
        test_server_team_model_name_matches_client_helper,
        test_effective_inactivity_timeout_extends_remote_ingame_clients,
        test_entity_world_collision_prefers_mesh_contact_when_collision_model_exists,
        test_entity_world_collision_falls_back_to_box_without_collision_model,
        test_entity_world_collision_uses_dirty_terrain_raycast_branch,
        test_entity_world_collision_uses_dirty_contact_before_raycast,
        test_entity_world_collision_uses_dirty_bounds_contact_store,
        test_entity_world_collision_dirty_bounds_store_accepts_tiny_contact,
        test_entity_world_collision_dirty_bounds_store_uses_contact_point_radius_resolution,
        test_entity_world_collision_pathological_dirty_bounds_contact_falls_back_to_raycast,
        test_entity_world_collision_far_dirty_bounds_contact_falls_back_to_raycast,
        test_entity_world_collision_horizontal_dirty_bounds_contact_falls_back_to_raycast,
        test_entity_world_collision_dirty_bounds_phase_uses_xy_broadphase,
        test_dirty_bounds_contact_helpers_skip_triangle_prefilter,
        test_terrain_cell_triangles_match_decompile_order,
        test_terrain_raycast_patch_traverse_uses_start_to_end_sector_sweep,
        test_terrain_patch_raycast_cells_uses_decompile_dda_order,
        test_terrain_patch_raycast_cells_uses_decompile_axis_flag_step_policy,
        test_entity_world_collision_uses_persistent_reference_pos_for_dirty_branch,
        test_entity_world_collision_refreshes_reference_on_clean_contact,
        test_entity_world_collision_dirty_threshold_uses_mesh_min_half_extent,
        test_entity_world_collision_dirty_raycast_uses_contact_separation,
        test_entity_world_collision_dirty_raycast_uses_decompile_degenerate_threshold,
        test_entity_world_collision_static_separation_matches_decompile_clamp,
        test_entity_world_collision_clean_path_uses_single_contact_store,
        test_roster_entry_stays_tcp_only,
        test_broadcast_player_stats_stays_tcp_only,
        test_control_pos_exact_reset_targets_specific_client,
        test_control_heading_set_preserves_yaw_sign_convention,
        test_solo_local_player_keepalive_shape_triggers_og_state_request_gate,
        test_view_update_pos_without_rot_clamps_to_safe_shape,
        test_players_json_includes_transport_addresses,
    ]

    passed = 0
    failed = 0

    for test in tests:
        print()
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {test.__name__}: FAILED - {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
