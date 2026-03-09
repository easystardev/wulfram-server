#!/usr/bin/env python3
"""
Tests for handler functions extracted from server.py.
"""

import os
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
from wulfram.server import WulframServer
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
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678
    server._get_local_state_kwargs = lambda ctx: {
        "include_health": True,
        "weapon_id": 0,
        "health": 1.0,
        "fuel": 1.0,
    }
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


def test_entity_create_uses_viewer_full_local_state_for_og_viewer():
    """OG viewer entity-create packets must carry the viewer's full tank local-state."""
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

    local_state = server._get_local_state_kwargs(viewer_ctx)
    payload = build_update_array_create_tank(
        tick=0x12345678,
        entity_id=1338,
        entity_type=0,
        team=2,
        pos=(4980.0, 5100.0, 5.0),
        is_manned=True,
        rot=(0.0, 0.0, 0.0),
        **local_state,
    )

    tick, decoded_local_state, entities = decode_update_array(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert decoded_local_state is not None
    assert decoded_local_state.weapon_id == 0
    assert len(entities) == 1, entities
    assert entities[0].entity_id == 1338
    print("test_entity_create_uses_viewer_full_local_state_for_og_viewer: PASSED")
    return True


def test_remote_player_update_uses_viewer_local_state():
    """Remote updates must carry the viewer local-state, not the other player's."""
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
    assert local_state.weapon_id == 0
    assert len(entities) == 1, entities
    assert entities[0].entity_id == 1338
    print("test_remote_player_update_uses_viewer_local_state: PASSED")
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
    """Loopback remote updates must decode cleanly with local-state + entity payload."""
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
    assert local_state is not None
    assert local_state.weapon_id == 0
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
    """Negative slot-3 input should decode as world-left, positive as world-right."""
    server = WulframServer.__new__(WulframServer)
    server.strafe_sign = 1.0

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.weapon_system = SimpleNamespace(control_max=1000.0)

    left_input = server._decode_network_strafe_input(ctx, -0.5800)
    right_input = server._decode_network_strafe_input(ctx, 0.6409)

    assert left_input < 0.0, left_input
    assert right_input > 0.0, right_input
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


def test_server_tank_motion_clamps_to_max_velocity():
    """Tank movement vector should clamp to config max_velocity before integration."""
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
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (0.0, 0.0, 10.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}

    server._update_player_position(ctx, dt_override=1.0)

    vx, vy, vz = ctx.player_vel
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    assert abs(speed - 80.0) < 1e-4, speed
    print("test_server_tank_motion_clamps_to_max_velocity: PASSED")
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
        test_server_remote_projectile_spawn_uses_viewer_local_state,
        test_server_remote_projectile_update_uses_safe_local_state_after_promotion,
        test_loopback_projectile_update_stays_entity_only,
        test_server_remote_player_info_uses_spawn_safe_local_state,
        test_remote_player_info_packet_short_local_state_layout,
        test_weapon_system_og_direct_trigger_slot_fires_pulse_shell,
        test_weapon_system_held_fire_repeats_on_cooldown,
        test_send_entity_create_uses_udp_only,
        test_entity_create_uses_viewer_full_local_state_for_og_viewer,
        test_remote_player_update_uses_viewer_local_state,
        test_loopback_entity_create_decodes_roundtrip,
        test_loopback_remote_player_update_decodes_roundtrip,
        test_loopback_heartbeat_decodes_roundtrip,
        test_server_network_strafe_decode_matches_og_sign,
        test_remote_client_promotes_full_local_state_after_spawn_delay,
        test_server_tank_motion_clamps_to_max_velocity,
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
