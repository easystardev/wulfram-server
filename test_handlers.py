#!/usr/bin/env python3
"""
Tests for handler functions extracted from server.py.
"""

import os
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from wulfram.handlers import decode_lp_string
from wulfram.handlers import send_initial_game_data
from wulfram.client import ClientContext
from wulfram.session import Session
from wulfram.server import WulframServer, _StaticWorldRayNode
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
from client.wulfram_client.network.decoder import decode_update_array, decode_view_update
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


def test_send_initial_game_data_og_bootstrap_order():
    """Remote OG bootstrap should match the verified packet order."""
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
        assert opcodes == [0x28, 0x22, 0x17, 0x2F, 0x23, 0x24, 0x32, 0x1A, 0x16], opcodes
        assert session.behavior_sent is True
        assert session.translation_sent is True
        assert session.roster_sent is True
        assert session.world_stats_sent is True
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
    """Promoted remote OG heartbeats must use a real local-player update shape."""
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
    assert entities[0].position is not None
    assert entities[0].velocity is not None
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


def test_remote_state_sync_reply_uses_safe_local_player_shape():
    """Remote OG STATE_REQUEST replies must stay on the safe 43-byte local-player shape."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
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

    server._send_state_sync_snapshot(ctx, include_view_update=False, reason="test")

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
    assert len(entities) == 1, entities
    assert entities[0].entity_id == 0x14EA
    assert entities[0].position is not None
    assert entities[0].velocity is not None
    assert entities[0].rotation is not None
    assert entities[0].angular_velocity is not None
    print("test_remote_state_sync_reply_uses_safe_local_player_shape: PASSED")
    return True


def test_remote_state_sync_reply_emits_view_update_with_request_timestamp():
    """Remote STATE_REQUEST replies should include a replay VIEW_UPDATE companion."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
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
    print("test_remote_state_sync_reply_emits_view_update_with_request_timestamp: PASSED")
    return True


def test_state_request_does_not_overwrite_client_tick_offset():
    """STATE_REQUEST request_id must not replace the input-tick sync domain."""
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
    assert ctx.remote_full_local_state_ready is True
    print("test_state_request_does_not_overwrite_client_tick_offset: PASSED")
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
    assert server._maybe_promote_remote_full_local_state(ctx, reason="test") is True
    assert ctx.remote_full_local_state_ready is True
    assert server._wf_minimal_local_state_for_client(ctx) is False
    print("test_remote_client_promotes_full_local_state_after_spawn_delay: PASSED")
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


def test_server_motion_clamps_to_max_velocity():
    """Movement vector should still clamp to config max_velocity before integration."""
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
    assert abs(speed - 80.0) < 1e-4, speed
    print("test_server_motion_clamps_to_max_velocity: PASSED")
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
    assert model_calls == 2, model_calls
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

    assert box_calls == 2, box_calls
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
            normal=(0.0, 0.0, -1.0),
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


def test_entity_world_collision_dirty_threshold_uses_mesh_min_half_extent():
    """Dirty threshold should use the assigned collision mesh min half-extent, including Z, per the decompile."""
    server = WulframServer.__new__(WulframServer)
    server._building_collision = SimpleNamespace(
        available=True,
        models={
            "tank_1": SimpleNamespace(
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
            normal=(0.0, 0.0, -1.0),
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
            normal=(0.0, 0.0, -1.0),
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
        test_send_initial_game_data_og_bootstrap_order,
        test_build_chat_message_comm_layout,
        test_build_update_array_remote_heartbeat_shape,
        test_server_remote_heartbeat_helper_keeps_full_local_state,
        test_server_remote_heartbeat_helper_pre_state_request_is_spawn_safe,
        test_remote_state_sync_reply_uses_safe_local_player_shape,
        test_remote_state_sync_reply_emits_view_update_with_request_timestamp,
        test_state_request_does_not_overwrite_client_tick_offset,
        test_translation_velocity_quantizer_matches_decompile_defaults,
        test_server_remote_local_state_kwargs_use_full_tank_shape,
        test_server_remote_entity_packets_use_safe_local_state_after_promotion,
        test_server_remote_projectile_spawn_uses_viewer_local_state,
        test_server_remote_projectile_update_uses_safe_local_state_after_promotion,
        test_loopback_projectile_update_stays_entity_only,
        test_server_remote_player_info_uses_spawn_safe_local_state,
        test_remote_player_info_packet_short_local_state_layout,
        test_weapon_system_og_direct_trigger_slot_fires_pulse_shell,
        test_weapon_system_held_fire_repeats_on_cooldown,
        test_send_entity_create_uses_udp_only,
        test_entity_create_uses_spawn_safe_local_state_for_og_viewer,
        test_remote_player_update_uses_spawn_safe_viewer_local_state,
        test_loopback_entity_create_decodes_roundtrip,
        test_loopback_remote_player_update_decodes_roundtrip,
        test_loopback_heartbeat_decodes_roundtrip,
        test_server_network_strafe_decode_matches_og_sign,
        test_remote_client_promotes_full_local_state_after_spawn_delay,
        test_server_tank_motion_uses_low_speed_mobility_factor,
        test_server_motion_clamps_to_max_velocity,
        test_projectile_world_hit_skips_aabb_for_mesh_backed_building,
        test_projectile_world_hit_prefers_closest_building_before_terrain,
        test_projectile_world_hit_clips_static_world_raycast_to_terrain,
        test_projectile_world_hit_uses_exact_mesh_raycast_position,
        test_static_world_raycast_uses_quadtree_front_to_back_order,
        test_static_world_raycast_uses_point_query_for_zero_horizontal_direction,
        test_segment_raycast_cbsp_tree_uses_split_plane_normal,
        test_segment_hits_cbsp_tree_detects_plane_hit,
        test_triangle_cbsp_contact_uses_node_split_normal,
        test_triangle_cbsp_contact_uses_entity_bounding_radius_for_plane_reject,
        test_box_collision_returns_first_contact_in_grid_order,
        test_model_collision_returns_first_contact_in_grid_order,
        test_triangle_cbsp_contact_returns_first_leaf_hit,
        test_building_collision_skips_aabb_for_mesh_backed_building,
        test_entity_world_collision_prefers_mesh_contact_when_collision_model_exists,
        test_entity_world_collision_falls_back_to_box_without_collision_model,
        test_entity_world_collision_uses_dirty_terrain_raycast_branch,
        test_entity_world_collision_uses_dirty_contact_before_raycast,
        test_entity_world_collision_uses_dirty_bounds_contact_store,
        test_entity_world_collision_dirty_bounds_phase_uses_xy_broadphase,
        test_dirty_bounds_contact_helpers_skip_triangle_prefilter,
        test_terrain_cell_triangles_match_decompile_order,
        test_terrain_raycast_patch_traverse_uses_start_to_end_sector_sweep,
        test_terrain_patch_raycast_cells_uses_decompile_dda_order,
        test_terrain_patch_raycast_cells_uses_decompile_axis_flag_step_policy,
        test_entity_world_collision_uses_persistent_reference_pos_for_dirty_branch,
        test_entity_world_collision_dirty_threshold_uses_mesh_min_half_extent,
        test_entity_world_collision_dirty_raycast_uses_contact_separation,
        test_entity_world_collision_dirty_raycast_uses_decompile_degenerate_threshold,
        test_entity_world_collision_static_separation_matches_decompile_clamp,
        test_roster_entry_stays_tcp_only,
        test_broadcast_player_stats_stays_tcp_only,
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
