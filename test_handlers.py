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
from wulfram.control import ControlServer, build_input_sync_diagnosis, build_player_terrain_probe
from wulfram.session import Session, Phase, FEATURES
from wulfram.server import WulframServer, _StaticWorldRayNode
from wulfram.physics import _matrix3_from_euler_xyz
from wulfram.terrain import Terrain
from wulfram.building_collision import BuildingCollisionAssets, BuildingEntity
from wulfram.world_collision import (
    TerrainContact,
    TerrainGridCollision,
    TerrainRaycastHit,
    _cross3,
    _normalize3,
    _sub3,
)
from wulfram.weapons import (
    WeaponSystem,
    WeaponType,
    EntityType,
    OG_DIRECT_TRIGGER_WEAPON_SLOTS,
    VEHICLE_PHYSICS_CONFIGS,
    Projectile,
    build_projectile_spawn_packet,
    build_projectile_update_packet,
)


def _legacy_contact_response_test(fn):
    """Run a legacy projection-path unit test without changing global defaults."""
    def wrapper():
        old_response = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE")
        try:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = "legacy"
            return fn()
        finally:
            if old_response is None:
                os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
            else:
                os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = old_response

    wrapper.__name__ = fn.__name__
    return wrapper
from wulfram.packets import (
    build_chat_message,
    build_behavior_packet,
    get_ticks,
    build_player_info,
    build_translation_packet,
    build_update_array_create_tank,
    build_view_update_create_tank,
    build_update_array_heartbeat,
    build_world_stats,
    build_update_stats_team_first,
    build_ship_status,
    build_carrying_info,
    build_uplink_info,
    build_supply_ship_info,
)
from wulfram2_protocol.codec import BitReader, BitWriter, quantize_float
from client.wulfram_client.network.behavior import parse_behavior
from client.wulfram_client.network.decoder import decode_update_array, decode_view_update, decode_tank_packet
from client.wulfram_client.network.quantizer import parse_translation
from wulfram2_protocol.entities import (
    ACTION_ANALOG_SLOTS,
    ACTION_DUMP_CONTROL_SLOTS,
    BehaviorSlot,
    JUMP_JET_CONFIGS,
    OG_TANK_SOFTBODY_FLAT_AVERAGE_HEIGHT,
    OG_TANK_SOFTBODY_IDLE_SLOT5,
    OG_TANK_SOFTBODY_Q_SLOT5,
    OG_TANK_SOFTBODY_Z_SLOT5,
    OG_TANK_SPRING_STRETCH_SPEED_DENOMINATOR,
    TANK_SOFTBODY_CONTROL_SLOT,
    entity_interp_factor,
    entity_interpolate_toward_target_decision,
    solve_static_terrain_constraint,
    tank_softbody_control_slot_value,
    tank_softbody_suspension_force,
    tank_suspension_lift_accel,
)
from client.wulfram_client.data.models import CBSPTree, CBSPTreeNode, Vec3
from client.wulfram_client.simulation.collision import (
    segment_hits_cbsp_tree,
    segment_raycast_cbsp_tree,
)


def _remote_view_timestamp(tick: int) -> int:
    return (int(tick) + 1000) & 0xFFFFFFFF


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


def test_behavior_spawn_enabled_defaults_on_for_entry_map_spawn():
    """OG map clicks only emit default spawn REINCARNATE when BEHAVIOR enables spawning."""
    cfg = parse_behavior(build_behavior_packet())
    assert cfg.spawn_enabled is True
    print("test_behavior_spawn_enabled_defaults_on_for_entry_map_spawn: PASSED")
    return True


def test_input_sync_diagnosis_distinguishes_idle_snapback_from_correction_failure():
    """Live telemetry should call out idle controls while correction packets are active."""
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

    assert diagnosis["status"] == "idle_input_correction_packets_active"
    assert diagnosis["corrections_active"] is True
    assert diagnosis["correction_packets_active"] is True
    assert diagnosis["correction_application_verified"] is False
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


def test_input_sync_diagnosis_counts_unsolicited_correction_stream():
    """Live movement correction packets should not be reported as no packet stream."""
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
        last_correction_age_s=0.05,
        last_movement_correction_age_s=0.05,
        movement_corrections=3,
    )

    assert diagnosis["status"] == "moving_with_correction_packets"
    assert diagnosis["corrections_active"] is True
    assert diagnosis["correction_packets_active"] is True
    assert diagnosis["correction_application_verified"] is False
    assert diagnosis["state_request_corrections_active"] is False
    assert diagnosis["correction_stream_active"] is True
    print("test_input_sync_diagnosis_counts_unsolicited_correction_stream: PASSED")
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


def test_spawn_at_point_honors_vehicle_selection():
    """handle_spawn_at_point threads the client's selected vehicle_type to
    _spawn_wf_style for supported types (0=Tank, 1=Scout only); 2/3/4 fall back
    to Tank; the whole feature gates on vehicle_select_enabled. Bomber (3) is a
    fallback because it crashes the live OG client on spawn (see memory
    vehicle-select-bomber-crashes-og)."""
    old = os.environ.get("WULFRAM_SPAWN_POS")
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
        server.vehicle_select_enabled = True
        captured = {}
        server._spawn_wf_style = lambda ctx, team_id, pos=None, unit_type=0, **kw: \
            captured.update(unit_type=unit_type)

        def spawn(veh):
            captured.clear()
            ctx = ClientContext(client_id=1, client_addr=("127.0.0.1", 50000),
                                session=Session(), entity_id=0x14EA)
            handle_spawn_at_point(server, ctx, 7001, veh, ("127.0.0.1", 50000))
            return captured.get("unit_type")

        assert spawn(0) == 0, "Tank"
        assert spawn(1) == 1, "Scout honored"
        assert spawn(2) == 0, "Assault -> Tank fallback"
        assert spawn(3) == 0, "Bomber -> Tank fallback (crashes live OG client)"
        assert spawn(4) == 0, "Transport -> Tank fallback"
        server.vehicle_select_enabled = False
        assert spawn(1) == 0, "gated off -> Tank"
        print("  test_spawn_at_point_honors_vehicle_selection: PASSED")
        return True
    finally:
        if old is None:
            os.environ.pop("WULFRAM_SPAWN_POS", None)
        else:
            os.environ["WULFRAM_SPAWN_POS"] = old


def test_recent_control_pose_blocks_in_game_spawn_override():
    """Late spawn-point packets must not undo a control-plane setup pose."""
    old_block = os.environ.get("WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S")
    try:
        os.environ["WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S"] = "45"
        server = WulframServer.__new__(WulframServer)
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

        session = Session(phase=Phase.IN_GAME, in_game=True, team_id=2, last_spawn_time=0.0)
        ctx = ClientContext(
            client_id=1,
            client_addr=("127.0.0.1", 50000),
            session=session,
            entity_id=0x14EA,
        )
        ctx.control_pose_reset_time = time.monotonic()
        ctx.control_pose_reset_pos = (3160.4343, 2463.2375, -33.4864)

        handle_spawn_at_point(server, ctx, 7001, 0, ("127.0.0.1", 50000))

        assert captured == {}, captured
        assert session.phase == Phase.IN_GAME
        print("test_recent_control_pose_blocks_in_game_spawn_override: PASSED")
        return True
    finally:
        if old_block is None:
            os.environ.pop("WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S", None)
        else:
            os.environ["WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S"] = old_block


def test_recent_control_pose_blocks_delayed_auto_spawn():
    """A delayed auto-spawn must not undo a control-plane setup pose."""
    old_block = os.environ.get("WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S")
    try:
        os.environ["WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S"] = "45"
        server = WulframServer.__new__(WulframServer)
        captured = {}
        server._spawn_wf_style = lambda ctx, team_id, **kwargs: captured.update(
            team_id=team_id,
            kwargs=kwargs,
        )

        session = Session(phase=Phase.IN_GAME, in_game=True, team_id=2)
        session.delayed_spawn_team = 2
        session.delayed_spawn_time = time.monotonic() - 1.0
        ctx = ClientContext(
            client_id=1,
            client_addr=("127.0.0.1", 50000),
            session=session,
            entity_id=0x14EA,
        )
        ctx.control_pose_reset_time = time.monotonic()
        ctx.control_pose_reset_pos = (3160.4343, 2463.2375, -33.4864)

        server._auto_join_team(ctx, 2)

        assert captured == {}, captured
        assert session.delayed_spawn_team == 0
        assert session.delayed_spawn_time == 0
        print("test_recent_control_pose_blocks_delayed_auto_spawn: PASSED")
        return True
    finally:
        if old_block is None:
            os.environ.pop("WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S", None)
        else:
            os.environ["WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S"] = old_block


def test_recent_control_pose_repairs_unstamped_large_jump():
    """A recent setup pose can recover from an unstamped jump back to spawn."""
    old_block = os.environ.get("WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S")
    old_distance = os.environ.get("WULFRAM_CONTROL_POSE_REPAIR_DISTANCE")
    try:
        os.environ["WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S"] = "45"
        os.environ["WULFRAM_CONTROL_POSE_REPAIR_DISTANCE"] = "100"
        server = WulframServer.__new__(WulframServer)
        server.spawn_sets_ground_level = False

        session = Session(phase=Phase.IN_GAME, in_game=True, team_id=2)
        ctx = ClientContext(
            client_id=1,
            client_addr=("127.0.0.1", 50000),
            session=session,
            entity_id=0x14EA,
        )
        ctx.control_pose_reset_time = time.monotonic()
        ctx.control_pose_reset_pos = (3160.4343, 2463.2375, -33.4864)
        ctx.player_pos = (4950.0, 5100.0, 3.25)
        ctx.player_vel = (1.0, 2.0, 3.0)
        ctx.player_pose["pos"] = ctx.player_pos
        ctx.player_pose["vel"] = ctx.player_vel

        repaired = server._repair_recent_control_pose_jump(ctx, "unit_test")

        assert repaired is True
        assert ctx.player_pos == ctx.control_pose_reset_pos
        assert ctx.player_vel == (0.0, 0.0, 0.0)
        assert ctx.last_pose_reset_source == "control_pose_jump_repair"
        assert ctx.debug_last_control_pose_repair["source"] == "unit_test"
        assert ctx.debug_last_control_pose_repair["distance"] > 3000.0
        print("test_recent_control_pose_repairs_unstamped_large_jump: PASSED")
        return True
    finally:
        if old_block is None:
            os.environ.pop("WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S", None)
        else:
            os.environ["WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S"] = old_block
        if old_distance is None:
            os.environ.pop("WULFRAM_CONTROL_POSE_REPAIR_DISTANCE", None)
        else:
            os.environ["WULFRAM_CONTROL_POSE_REPAIR_DISTANCE"] = old_distance


def test_recent_control_pose_repair_can_be_disabled():
    """Zero repair distance disables the harness-only jump repair."""
    old_block = os.environ.get("WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S")
    old_distance = os.environ.get("WULFRAM_CONTROL_POSE_REPAIR_DISTANCE")
    try:
        os.environ["WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S"] = "45"
        os.environ["WULFRAM_CONTROL_POSE_REPAIR_DISTANCE"] = "0"
        server = WulframServer.__new__(WulframServer)

        session = Session(phase=Phase.IN_GAME, in_game=True, team_id=2)
        ctx = ClientContext(
            client_id=1,
            client_addr=("127.0.0.1", 50000),
            session=session,
            entity_id=0x14EA,
        )
        ctx.control_pose_reset_time = time.monotonic()
        ctx.control_pose_reset_pos = (3160.4343, 2463.2375, -33.4864)
        ctx.player_pos = (4950.0, 5100.0, 3.25)
        ctx.player_pose["pos"] = ctx.player_pos

        repaired = server._repair_recent_control_pose_jump(ctx, "unit_test")

        assert repaired is False
        assert ctx.player_pos == (4950.0, 5100.0, 3.25)
        assert ctx.last_pose_reset_source == ""
        print("test_recent_control_pose_repair_can_be_disabled: PASSED")
        return True
    finally:
        if old_block is None:
            os.environ.pop("WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S", None)
        else:
            os.environ["WULFRAM_CONTROL_POSE_BLOCK_SPAWN_OVERRIDE_S"] = old_block
        if old_distance is None:
            os.environ.pop("WULFRAM_CONTROL_POSE_REPAIR_DISTANCE", None)
        else:
            os.environ["WULFRAM_CONTROL_POSE_REPAIR_DISTANCE"] = old_distance


def test_map_entity_z_aligns_buried_entities_to_terrain():
    """Map-state buildings/spawn points below physics terrain should be lifted."""
    server = WulframServer.__new__(WulframServer)
    server.up_axis = "z"
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 0.0
    server.terrain = SimpleNamespace(get_height=lambda x, y: 60.0)

    z, ground_z, aligned = server._align_map_entity_z_to_terrain(2578.7, 3040.0, 58.0)

    assert aligned is True
    assert ground_z == 60.0
    assert z == 60.0
    print("test_map_entity_z_aligns_buried_entities_to_terrain: PASSED")
    return True


def test_map_entity_z_preserves_raw_z_above_physics_terrain():
    """The +5 map display offset must not lift physics collision blockers."""
    server = WulframServer.__new__(WulframServer)
    server.up_axis = "z"
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 0.0
    server.terrain = SimpleNamespace(
        get_height=lambda x, y: 0.0,
        get_slope=lambda x, y: (0.0, 0.0),
    )

    z, ground_z, aligned = server._align_map_entity_z_to_terrain(5064.28, 5103.49, 3.65)

    assert aligned is False
    assert ground_z == 0.0
    assert z == 3.65
    print("test_map_entity_z_preserves_raw_z_above_physics_terrain: PASSED")
    return True


def test_map_entity_z_preserves_elevated_entities():
    """Terrain alignment should not pull already-elevated map entries down."""
    server = WulframServer.__new__(WulframServer)
    server.up_axis = "z"
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 0.0
    server.terrain = SimpleNamespace(
        get_height=lambda x, y: 0.0,
        get_slope=lambda x, y: (0.0, 0.0),
    )

    z, ground_z, aligned = server._align_map_entity_z_to_terrain(5150.1, 5241.3, 7.7)

    assert aligned is False
    assert ground_z == 0.0
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


def test_enter_game_can_target_team_select_tcp_only_client():
    """The automation spawn path must resolve TEAM_SELECT clients before UDP exists."""
    session = Session()
    ctx = ClientContext(
        client_id=5,
        client_addr=("10.10.10.2", 52911),
        session=session,
        entity_id=0x153D,
    )
    ctx.tcp_handler = object()
    ctx.player_pos = (4950.0, 5100.0, 5.0)

    captured = {}

    def spawn_wf_style(spawn_ctx, team_id, net_id, unit_type, pos, announce):
        captured.update(
            ctx=spawn_ctx,
            team_id=team_id,
            net_id=net_id,
            unit_type=unit_type,
            pos=pos,
            announce=announce,
        )

    server = SimpleNamespace(
        clients={5: ctx},
        clients_lock=threading.Lock(),
        next_entity_id=0x2000,
        _spawn_wf_style=spawn_wf_style,
    )
    control = ControlServer.__new__(ControlServer)
    control.server = server
    control._sync_to_active_client = lambda: None

    output = control._cmd_enter_game(["c5", "t2"])

    assert "Entered game: client=5" in output, output
    assert "udp=no" in output, output
    assert captured["ctx"] is ctx, captured
    assert captured["team_id"] == 2, captured
    assert captured["net_id"] == 0x153D, captured
    assert session.team_id == 2, session.team_id
    print("test_enter_game_can_target_team_select_tcp_only_client: PASSED")
    return True


def test_tank_softbody_spawn_pose_does_not_pin_ground_override():
    """Softbody tanks should settle on terrain suspension, not a spawn Z clamp."""
    server = WulframServer.__new__(WulframServer)
    server.spawn_sets_ground_level = True
    server.up_axis = "z"
    server.terrain = SimpleNamespace(
        get_height=lambda x, y: 0.0,
        get_slope=lambda x, y: (0.0, 0.0),
    )
    server.terrain_physics_height_offset = 0.0
    server.tank_suspension_enabled = True
    server.tank_suspension_model = "softbody"

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.entity_type = EntityType.TANK
    ctx.ground_level_override = 65.0

    server._set_ground_level_override_for_pose(ctx, (100.0, 200.0, 2.5))

    assert ctx.ground_level_override is None
    assert ctx.ground_override_ref_terrain_level is None
    print("test_tank_softbody_spawn_pose_does_not_pin_ground_override: PASSED")
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


def test_projectile_fire_pose_uses_replay_history_when_available():
    """Moving fire should use the pose aligned to the input packet tick."""
    server = WulframServer.__new__(WulframServer)
    server.up_axis = "z"
    server.pos_offset = 0.0
    server.projectile_aim_source = "body"
    server.projectile_body_pitch = False
    server.use_client_ticks = False
    server.viewpoint_timeout = 0.5
    server.aim_hold_time = 0.5

    ctx = ClientContext(
        client_id=1,
        client_addr=("127.0.0.1", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_pos = (50.0, 60.0, 7.0)
    ctx.player_heading = 2.0
    ctx.player_pose["roll"] = 0.4
    ctx.player_pose["pitch"] = 0.2
    ctx.player_aim_source = "init"
    ctx.player_aim_time = 0.0
    ctx.authoritative_state_history.append({
        "tick": 1000,
        "time": time.monotonic(),
        "pos": (10.0, 20.0, 3.0),
        "vel": (1.0, 0.0, 0.0),
        "rot": (0.1, 0.2, 1.25),
    })

    pos, rot, aim_src, pose_src = server._select_weapon_fire_pose(ctx, 1000)

    assert pos == (10.0, 20.0, 3.0), pos
    assert rot == (0.1, 0.0, 1.25), rot
    assert aim_src == "body"
    assert pose_src == "history"
    print("test_projectile_fire_pose_uses_replay_history_when_available: PASSED")
    return True


def test_projectile_body_source_can_opt_into_body_pitch():
    """Body-pitch gate should preserve replay pose pitch when enabled."""
    server = WulframServer.__new__(WulframServer)
    server.up_axis = "z"
    server.pos_offset = 0.0
    server.projectile_aim_source = "body"
    server.projectile_body_pitch = True
    server.use_client_ticks = False
    server.viewpoint_timeout = 0.5
    server.aim_hold_time = 0.5

    ctx = ClientContext(
        client_id=1,
        client_addr=("127.0.0.1", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_pos = (50.0, 60.0, 7.0)
    ctx.player_heading = 2.0
    ctx.player_pose["roll"] = 0.4
    ctx.player_pose["pitch"] = 0.2
    ctx.player_aim_source = "init"
    ctx.player_aim_time = 0.0
    ctx.authoritative_state_history.append({
        "tick": 1000,
        "time": time.monotonic(),
        "pos": (10.0, 20.0, 3.0),
        "vel": (1.0, 0.0, 0.0),
        "rot": (0.1, 0.2, 1.25),
    })

    pos, rot, aim_src, pose_src = server._select_weapon_fire_pose(ctx, 1000)

    assert pos == (10.0, 20.0, 3.0), pos
    assert rot == (0.1, 0.2, 1.25), rot
    assert aim_src == "body_pitch"
    assert pose_src == "history"
    print("test_projectile_body_source_can_opt_into_body_pitch: PASSED")
    return True


def test_projectile_fire_pose_rejects_stale_fallback_yaw():
    """A fire whose tick misses the replay window must NOT spawn at the stale
    history yaw (the 'pulse fires the wrong way' bug). The selector's unbounded
    best_abs fallback returns a pre-turn pose; the stale-yaw guard keeps the live
    body yaw for direction while still using the history muzzle position."""
    server = WulframServer.__new__(WulframServer)
    server.up_axis = "z"
    server.pos_offset = 0.0
    server.projectile_aim_source = "body"
    server.projectile_body_pitch = False
    server.use_client_ticks = False
    server.viewpoint_timeout = 0.5
    server.aim_hold_time = 0.5
    server.fire_pose_stale_yaw_guard = True
    server.fire_pose_stale_yaw_deg = 30.0

    ctx = ClientContext(
        client_id=1,
        client_addr=("73.185.157.225", 50000),  # non-loopback (real remote)
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_pos = (50.0, 60.0, 7.0)
    ctx.player_heading = 2.0  # live body yaw (where the firer is actually pointing)
    ctx.player_pose["roll"] = 0.4
    ctx.player_pose["pitch"] = 0.2
    ctx.player_aim_source = "init"
    ctx.player_aim_time = 0.0
    # History snapshot at tick 1000 with a PRE-TURN yaw (1.25 rad ~ 72deg, ~43deg off
    # the live 2.0). The fire requests tick 9000 -> 8000ms past the snapshot, far outside
    # the 250ms replay window, so the selector returns this via its stale best_abs fallback.
    ctx.authoritative_state_history.append({
        "tick": 1000,
        "time": time.monotonic(),
        "pos": (10.0, 20.0, 3.0),
        "vel": (1.0, 0.0, 0.0),
        "rot": (0.1, 0.2, 1.25),
    })

    pos, rot, aim_src, pose_src = server._select_weapon_fire_pose(ctx, 9000)

    assert pos == (10.0, 20.0, 3.0), pos          # muzzle position still from history
    assert rot == (0.1, 0.0, 2.0), rot            # DIRECTION uses live yaw (2.0), not 1.25
    assert pose_src == "history+liveyaw", pose_src
    assert getattr(ctx, "_last_auth_snapshot_stale", None) is True

    # Guard OFF restores the legacy behavior (stale history yaw used) -- documents the gate.
    server.fire_pose_stale_yaw_guard = False
    _, rot_legacy, _, pose_legacy = server._select_weapon_fire_pose(ctx, 9000)
    assert rot_legacy == (0.1, 0.0, 1.25), rot_legacy
    assert pose_legacy == "history", pose_legacy

    print("test_projectile_fire_pose_rejects_stale_fallback_yaw: PASSED")
    return True


def test_projectile_body_pitch_defaults_on_with_env_optout():
    """Pulse/body projectile pitch should be canonical by default, with explicit opt-out."""
    old_env = os.environ.get("WULFRAM_PROJECTILE_BODY_PITCH")
    try:
        os.environ.pop("WULFRAM_PROJECTILE_BODY_PITCH", None)
        assert WulframServer._projectile_body_pitch_enabled_from_env() is True

        os.environ["WULFRAM_PROJECTILE_BODY_PITCH"] = "1"
        assert WulframServer._projectile_body_pitch_enabled_from_env() is True

        os.environ["WULFRAM_PROJECTILE_BODY_PITCH"] = "0"
        assert WulframServer._projectile_body_pitch_enabled_from_env() is False
    finally:
        if old_env is None:
            os.environ.pop("WULFRAM_PROJECTILE_BODY_PITCH", None)
        else:
            os.environ["WULFRAM_PROJECTILE_BODY_PITCH"] = old_env

    print("test_projectile_body_pitch_defaults_on_with_env_optout: PASSED")
    return True


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


def _build_login_request(username: str, password: str = "", sub_type: int = 0x00) -> bytes:
    """Build a LOGIN_REQUEST (0x21) packet with length-prefixed strings."""
    def lp(s: str) -> bytes:
        encoded = s.encode("ascii") + b"\x00"
        return struct.pack(">H", len(encoded)) + encoded

    return bytes([0x21, sub_type]) + lp(username) + lp(password)


def test_relogin_honors_changed_username_and_remembers_last():
    """Behavior (c): a re-login on the same connection must HONOR a changed
    username (not stay stuck on the first/cached one) while remembering the
    last name as a default."""
    from wulfram.handlers import handle_login_request

    class DummyTCP:
        def __init__(self):
            self.sent = []

        def send(self, payload):
            self.sent.append(payload)

    session = Session()
    ctx = ClientContext(
        client_id=3,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=1337,
    )
    ctx.tcp_handler = DummyTCP()
    server = SimpleNamespace()

    # First login: client submits "Alice" (sub_type 0 -> server requests password).
    handle_login_request(server, ctx, _build_login_request("Alice"))
    assert session.username == "Alice", session.username
    assert session.last_username == "Alice", session.last_username
    # First-login path still works: server asked for the password (LOGIN_STATUS 1).
    assert any(p[:1] == b"\x22" and p[-1] == 1 for p in ctx.tcp_handler.sent), ctx.tcp_handler.sent

    # Complete login, then the player returns to the handle screen and re-logs
    # in with a DIFFERENT username on the SAME connection.
    session.login_complete = True
    ctx.tcp_handler.sent.clear()
    handle_login_request(server, ctx, _build_login_request("Bob"))

    # The NEW username must be applied (not silently kept as "Alice").
    assert session.username == "Bob", f"sticky username: {session.username!r}"
    assert session.last_username == "Bob", session.last_username
    # And the client is re-acked with LOGIN_STATUS(8) so it doesn't time out.
    assert any(p[:1] == b"\x22" and p[-1] == 8 for p in ctx.tcp_handler.sent), ctx.tcp_handler.sent

    print("test_relogin_honors_changed_username_and_remembers_last: PASSED")
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


def test_login_bootstrap_mode_routes_og_client_by_login_flow():
    """A real OG client (game-service login, sub_type 0x01->0x03) must get the
    OG bootstrap even on loopback — otherwise it lacks the spectator PLAYER
    identity and hangs at 'Processing Player Map'. Since the loopback
    client-type fork was retired (9ea5dbd, 2026-06-02: _is_loopback_client
    always False; the Python client is an OG clone), hybrid mode resolves to
    the OG bootstrap for EVERY client; only the explicit env override selects
    minimal."""
    from wulfram.handlers import _get_login_bootstrap_mode

    old = os.environ.get("WULFRAM_LOGIN_BOOTSTRAP")
    try:
        os.environ.pop("WULFRAM_LOGIN_BOOTSTRAP", None)  # default = hybrid

        def _ctx(addr, game_service):
            s = Session()
            s.login_game_service_requested = game_service
            return SimpleNamespace(client_addr=addr, session=s)

        # OG client (game-service) on loopback -> og (the fix).
        assert _get_login_bootstrap_mode(_ctx(("127.0.0.1", 5), True)) == "og"
        # Loopback fork retired: non-game-service loopback client also gets og.
        assert _get_login_bootstrap_mode(_ctx(("127.0.0.1", 5), False)) == "og"
        # Remote client -> og regardless (preserved).
        assert _get_login_bootstrap_mode(_ctx(("10.10.10.2", 5), False)) == "og"
        # Explicit override still wins over the heuristic.
        os.environ["WULFRAM_LOGIN_BOOTSTRAP"] = "minimal"
        assert _get_login_bootstrap_mode(_ctx(("10.10.10.2", 5), True)) == "minimal"
        print("test_login_bootstrap_mode_routes_og_client_by_login_flow: PASSED")
        return True
    finally:
        if old is None:
            os.environ.pop("WULFRAM_LOGIN_BOOTSTRAP", None)
        else:
            os.environ["WULFRAM_LOGIN_BOOTSTRAP"] = old


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


def _build_comm_req_packet(text: str, seq: int = 1, source: int = 0x1000) -> bytes:
    """Build a COMM_REQ (0x20) UDP packet carrying a chat string.

    Layout mirrors handle_udp_chat's parser: opcode, seq(2), length(2),
    source(2), 2 filler bytes, then a length-prefixed string at offset 9.
    """
    msg_bytes = text.encode("ascii")
    head = struct.pack(">H", seq) + struct.pack(">H", 0) + struct.pack(">H", source) + b"\x00\x00"
    payload = struct.pack(">H", len(msg_bytes)) + msg_bytes
    return b"\x20" + head + payload


def _make_spawned_chat_ctx(sent_packets: list, wonked_pos):
    """Build a minimal in-game ClientContext with a packet-capturing tcp_handler."""
    session = Session(phase=Phase.IN_GAME, in_game=True, team_id=2)
    session.player_id = 0x2222
    ctx = ClientContext(
        client_id=1,
        client_addr=("127.0.0.1", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.player_pos = wonked_pos
    ctx.vehicle_physics = None
    ctx.tcp_handler = SimpleNamespace(send=lambda pkt: sent_packets.append(pkt))
    ctx.pending_respawn_pos = wonked_pos  # the stuck location to be cleared
    return ctx


def test_relay_player_chat_routes_by_mode():
    """Player chat relays as COMM_MESSAGE (0x1F): ALL -> everyone incl sender,
    TEAM -> same-team incl sender, whisper(5) -> target only; SERVER/empty -> none."""
    def mkctx(cid, team, pid, name):
        s = Session(phase=Phase.IN_GAME, in_game=True, team_id=team)
        s.player_id = pid
        s.username = name
        return ClientContext(client_id=cid, client_addr=("127.0.0.1", 50000 + cid),
                             session=s, entity_id=pid)
    c1 = mkctx(1, 1, 0x1001, "Alice")  # sender, team 1
    c2 = mkctx(2, 1, 0x1002, "Bob")    # team 1
    c3 = mkctx(3, 2, 0x1003, "Carol")  # team 2

    server = WulframServer.__new__(WulframServer)
    server._snapshot_clients = lambda: [c1, c2, c3]
    sends: list = []
    server._send_packet_to_client = lambda c, pkt, **k: sends.append((c.client_id, pkt))

    sends.clear(); server._relay_player_chat(c1, 4, 0, "hi all")
    assert {s[0] for s in sends} == {1, 2, 3}, f"ALL should reach everyone: {sends}"
    assert all(p[:1] == b"\x1F" for _, p in sends)
    assert b"Alice: hi all" in sends[0][1]

    sends.clear(); server._relay_player_chat(c1, 3, 0, "go left")
    assert {s[0] for s in sends} == {1, 2}, f"TEAM should reach team 1 only: {sends}"
    assert b"[team] Alice: go left" in sends[0][1]

    sends.clear(); server._relay_player_chat(c1, 5, 0x1003, "secret")
    assert {s[0] for s in sends} == {3}, f"whisper should reach target only: {sends}"
    assert b"[whisper] Alice: secret" in sends[0][1]

    sends.clear(); server._relay_player_chat(c1, 1, 0, "x")
    assert sends == [], "SERVER channel must not relay"
    sends.clear(); server._relay_player_chat(c1, 4, 0, "   ")
    assert sends == [], "empty message must not relay"
    print("  test_relay_player_chat_routes_by_mode: PASSED")
    return True


def test_kill_feed_broadcasts_to_all_in_game():
    """Kill notices ride server chat (COMM_MESSAGE 0x1F, system source_id=0) to all
    in-game clients; gated by kill_feed_enabled."""
    def mkctx(cid):
        s = Session(phase=Phase.IN_GAME, in_game=True, team_id=1)
        s.player_id = 0x100 + cid
        return ClientContext(client_id=cid, client_addr=("127.0.0.1", 50000 + cid),
                             session=s, entity_id=0x100 + cid)
    a, b = mkctx(1), mkctx(2)
    server = WulframServer.__new__(WulframServer)
    server.kill_feed_enabled = True
    server._snapshot_in_game_clients = lambda: [a, b]
    sends: list = []
    server._send_packet_to_client = lambda c, pkt, **k: sends.append((c.client_id, pkt))

    assert server._broadcast_kill_feed("Alice destroyed Bob") == 2
    assert {s[0] for s in sends} == {1, 2}
    assert all(p[:1] == b"\x1F" for _, p in sends)
    assert b"Alice destroyed Bob" in sends[0][1]

    server.kill_feed_enabled = False
    sends.clear()
    assert server._broadcast_kill_feed("x") == 0 and sends == []
    print("  test_kill_feed_broadcasts_to_all_in_game: PASSED")
    return True


def test_player_chat_respawn_despawns_and_clears_cached_spawn():
    """/respawn from a player must despawn their tank, clear the cached spawn
    position so the next spawn uses the team map spawn point, and confirm via
    chat — while leaving normal (non-slash) chat unaffected."""
    from wulfram.handlers import handle_udp_chat

    wonked_pos = (4242.0, 1313.0, -77.0)

    server = WulframServer.__new__(WulframServer)
    server.udp_handler = None  # skip the UDP ACK send path
    server.up_axis = "z"
    server.spawn_height = 5.0
    server._get_network_tick = lambda ctx: 1234
    server._snapshot_in_game_clients = lambda: [active_ctx]
    server._snapshot_clients = lambda: []  # normal chat now relays via 0x1F; no peers here

    control = ControlServer.__new__(ControlServer)
    control.server = server
    server.control_server = control

    sent = []
    active_ctx = _make_spawned_chat_ctx(sent, wonked_pos)
    addr = ("127.0.0.1", 50000)

    # 1) /respawn through the real chat handler.
    handle_udp_chat(server, active_ctx, _build_comm_req_packet("/respawn"), addr)

    delete_pkts = [p for p in sent if p[:1] == b"\x15"]  # DELETE_OBJECT (0x15)
    chat_pkts = [p for p in sent if p[:1] == b"\x1F"]    # COMM_MESSAGE (0x1F)
    assert delete_pkts, f"expected a DELETE_OBJECT, sent opcodes={[p[0] for p in sent]}"
    # Cached/pending spawn position cleared so the next spawn is map-resolved.
    assert active_ctx.pending_respawn_pos is None, active_ctx.pending_respawn_pos
    # Returned to the team/map screen for a fresh re-pick, no scheduled respawn.
    assert active_ctx.session.phase == Phase.TEAM_SELECT
    assert active_ctx.session.in_game is False
    assert active_ctx.session.delayed_spawn_time == 0
    # A confirmation chat was queued back to the player.
    assert chat_pkts, "expected a chat confirmation packet"
    assert b"Despawning" in chat_pkts[-1], chat_pkts[-1]

    # _auto_join_team would now resolve a fresh map spawn (pending pos is gone),
    # not the pre-despawn wonked location.
    assert active_ctx.pending_respawn_pos != wonked_pos

    # 2) The alias /rs behaves identically.
    sent2 = []
    ctx2 = _make_spawned_chat_ctx(sent2, wonked_pos)
    server._snapshot_in_game_clients = lambda: [ctx2]
    handle_udp_chat(server, ctx2, _build_comm_req_packet("/rs"), addr)
    assert any(p[:1] == b"\x15" for p in sent2), "/rs should despawn too"
    assert ctx2.pending_respawn_pos is None

    # 3) A normal (non-slash) chat message must NOT despawn the player.
    sent3 = []
    ctx3 = _make_spawned_chat_ctx(sent3, wonked_pos)
    server._snapshot_in_game_clients = lambda: [ctx3]
    handle_udp_chat(server, ctx3, _build_comm_req_packet("hello team"), addr)
    assert not any(p[:1] == b"\x15" for p in sent3), "plain chat must not despawn"
    assert ctx3.session.phase == Phase.IN_GAME
    assert ctx3.pending_respawn_pos == wonked_pos  # untouched

    # 4) Unknown PREFIXED commands get a polite reply, not a despawn.
    sent4 = []
    ctx4 = _make_spawned_chat_ctx(sent4, wonked_pos)
    server._snapshot_in_game_clients = lambda: [ctx4]
    handle_udp_chat(server, ctx4, _build_comm_req_packet("!wibble"), addr)
    assert not any(p[:1] == b"\x15" for p in sent4), "unknown command must not despawn"
    unknown_chat = [p for p in sent4 if p[:1] == b"\x1F"]
    assert unknown_chat and b"Unknown command" in unknown_chat[-1], unknown_chat

    # 5) BARE keyword 'respawn' (the OG-client path — no leading '/', which the
    #    OG client would eat as a whisper destination) must despawn.
    sent5 = []
    ctx5 = _make_spawned_chat_ctx(sent5, wonked_pos)
    server._snapshot_in_game_clients = lambda: [ctx5]
    handle_udp_chat(server, ctx5, _build_comm_req_packet("respawn"), addr)
    assert any(p[:1] == b"\x15" for p in sent5), "bare 'respawn' should despawn"
    assert ctx5.pending_respawn_pos is None

    # 6) '!respawn' prefix also works.
    sent6 = []
    ctx6 = _make_spawned_chat_ctx(sent6, wonked_pos)
    server._snapshot_in_game_clients = lambda: [ctx6]
    handle_udp_chat(server, ctx6, _build_comm_req_packet("!respawn"), addr)
    assert any(p[:1] == b"\x15" for p in sent6), "'!respawn' should despawn"

    # 7) A bare UNKNOWN word is ordinary chat — no despawn, no 'unknown' reply.
    sent7 = []
    ctx7 = _make_spawned_chat_ctx(sent7, wonked_pos)
    server._snapshot_in_game_clients = lambda: [ctx7]
    handle_udp_chat(server, ctx7, _build_comm_req_packet("respawning soon lol"), addr)
    assert not any(p[:1] == b"\x15" for p in sent7), "multi-word chat must not despawn"
    assert ctx7.session.phase == Phase.IN_GAME

    print("test_player_chat_respawn_despawns_and_clears_cached_spawn: PASSED")
    return True


def test_player_chat_respawn_via_tcp_comm_handler():
    """The real Python client sends COMM_MESSAGE_REQUEST over TCP, and the OG
    client only sends plain text — both arrive via the shared comm handler.
    A plain 'respawn' through that TCP path must despawn the player."""
    wonked_pos = (1111.0, 2222.0, -3.0)

    server = WulframServer.__new__(WulframServer)
    server.up_axis = "z"
    server.spawn_height = 5.0
    server._get_network_tick = lambda ctx: 99
    server._snapshot_in_game_clients = lambda: [ctx]
    server.build_uplink_mvp = False  # respawn must work even with MVP disabled
    server._build_uplink_command_events = []

    control = ControlServer.__new__(ControlServer)
    control.server = server
    server.control_server = control

    sent = []
    ctx = _make_spawned_chat_ctx(sent, wonked_pos)

    # TCP COMM_MESSAGE_REQUEST body: message_type(u16) + flags(u16) + lp_string.
    # Use message_type=4 (ALL) to prove it is NOT gated to team(2) like uplink.
    text = b"respawn"
    body = struct.pack(">H", 4) + struct.pack(">H", 0) + struct.pack(">H", len(text)) + text
    packet = b"\x20" + body

    event = server._handle_tcp_comm_message_request(ctx, packet)

    assert event.get("handled") is True, event
    assert any(p[:1] == b"\x15" for p in sent), "TCP 'respawn' should despawn"
    assert ctx.pending_respawn_pos is None
    assert ctx.session.phase == Phase.TEAM_SELECT
    assert any(p[:1] == b"\x1F" and b"Despawning" in p for p in sent), "expected chat reply"

    print("test_player_chat_respawn_via_tcp_comm_handler: PASSED")
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
    """Promoted remote OG heartbeats keep the FULL local-state HUD shape.

    GOAL 7 (16a3bfb, 2026-06-04): steady-state heartbeats (no transform to
    deliver) now carry ZERO entity records — the per-player mask=0 record made
    the OG client zero its predicted angular velocity (~10x under-rotation).
    The promoted full local-state (real weapon/ammo/turret bits) is preserved.
    """
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
    # Promoted/full form uses the HUD weapon type (local_state_weapon_type=0),
    # not the spawn-safe short-form weapon (spawn_tank_weapon_type=2).
    assert local_state.weapon_id == 0
    # Full form carries the live turret aim (short form zeroes it).
    assert abs(local_state.primary_turret - 1.234) < 0.01, local_state
    # GOAL 7: no transform requested -> no entity record (stomp-entity dropped).
    assert len(entities) == 0, entities
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
    assert timestamp == _remote_view_timestamp(0x12345678)
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
    assert timestamp == _remote_view_timestamp(0x12345678)
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


def test_remote_state_sync_reply_emits_view_update_with_fresh_remote_timestamp():
    """Remote OG STATE_REQUEST replies keep replay wrappers fresh for client admission."""
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
    assert timestamp == _remote_view_timestamp(0x12345678)
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
    assert ctx.last_state_sync_view_timestamp == _remote_view_timestamp(0x12345678)
    assert ctx.last_state_sync_reason == "test"
    assert ctx.last_state_sync_update_hex == update_payload[:32].hex()
    assert ctx.last_state_sync_view_hex == view_payload[:32].hex()
    print("test_remote_state_sync_reply_emits_view_update_with_fresh_remote_timestamp: PASSED")
    return True


def test_loopback_state_sync_reply_keeps_request_timestamp():
    """Loopback STATE_REQUEST replies take the SAME unified path as remote.

    The loopback client-type fork was retired (9ea5dbd, 2026-06-02): a
    127.0.0.1 client gets the same fresh remote-admission VIEW_UPDATE
    timestamp as a remote OG client (UpdateArray_check_eligible rejects stale
    interp_record+0x08 values), not the echoed request id.
    """
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
    # Unified path: fresh remote-admission timestamp, not the request id.
    assert timestamp == _remote_view_timestamp(0x12345678), hex(timestamp)
    assert view_tick == 0x12345678
    assert len(view_entities) == 1
    assert view_entities[0].entity_id == 0x14EA
    assert view_entities[0].position is not None
    print("test_loopback_state_sync_reply_keeps_request_timestamp: PASSED")
    return True


def test_remote_state_request_queues_visible_correction_burst():
    """Remote OG STATE_REQUEST replies should be followed by a short settle stream."""
    server = WulframServer.__new__(WulframServer)
    server.state_sync_correction_burst_count = 6
    server.state_sync_correction_burst_interval = 0.10

    remote = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    remote.correction_burst_remaining = 2
    assert server._queue_state_sync_correction_burst(remote) is True
    assert remote.correction_burst_remaining == 6
    assert remote.correction_burst_interval_s == 0.10

    # The loopback client-type fork is retired (2026-06-02): a 127.0.0.1
    # client queues exactly like any other — flood safety is the rate cap,
    # not a client-type branch.
    loopback = ClientContext(
        client_id=2,
        client_addr=("127.0.0.1", 50000),
        session=Session(),
        entity_id=0x14EB,
    )
    assert server._queue_state_sync_correction_burst(loopback) is True
    assert loopback.correction_burst_remaining == 6
    print("test_remote_state_request_queues_visible_correction_burst: PASSED")
    return True


def test_state_request_burst_rate_cap():
    """Auto-burst on STATE_REQUEST is capped to one queue per min_interval."""
    server = WulframServer.__new__(WulframServer)
    server.state_sync_correction_burst_count = 6
    server.state_sync_correction_burst_interval = 0.10
    server.state_request_burst_enabled = True
    server.state_request_burst_min_interval = 1.75
    server.active_input_correction_suppress_window = 0.35

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    assert server._maybe_queue_state_request_burst(ctx, now=100.0) is True
    assert ctx.correction_burst_remaining == 6
    assert ctx.last_state_request_burst_queue == 100.0

    # Drain the burst, then spam requests inside the cap window: no re-queue.
    ctx.correction_burst_remaining = 0
    assert server._maybe_queue_state_request_burst(ctx, now=100.5) is False
    assert server._maybe_queue_state_request_burst(ctx, now=101.7) is False
    assert ctx.correction_burst_remaining == 0

    # Past the cap window the next request queues again.
    assert server._maybe_queue_state_request_burst(ctx, now=101.8) is True
    assert ctx.correction_burst_remaining == 6
    print("test_state_request_burst_rate_cap: PASSED")
    return True


def test_correction_burst_due_drains_under_gate():
    """Queued bursts must drain in the GATED tick branch (2026-06-09 bug:
    `correction now c1 10 0.1` emitted a single packet because the gated
    branch never consulted correction_burst_remaining)."""
    server = WulframServer.__new__(WulframServer)

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.correction_burst_remaining = 3
    ctx.correction_burst_interval_s = 0.10
    ctx.last_correction_send = 0.0

    assert server._correction_burst_due(ctx, now=10.0, movement_suppressed=False) is True
    # Movement pauses the drain (corrections land right after key release).
    assert server._correction_burst_due(ctx, now=10.0, movement_suppressed=True) is False
    # Burst spacing (~10Hz) is honored between packets...
    ctx.last_correction_send = 10.0
    assert server._correction_burst_due(ctx, now=10.05, movement_suppressed=False) is False
    assert server._correction_burst_due(ctx, now=10.11, movement_suppressed=False) is True
    # ...and an empty queue never fires.
    ctx.correction_burst_remaining = 0
    assert server._correction_burst_due(ctx, now=11.0, movement_suppressed=False) is False
    print("test_correction_burst_due_drains_under_gate: PASSED")
    return True


def test_state_request_queues_burst_under_default_gate():
    """End-to-end trigger: STATE_REQUEST under the default correction gate
    queues the rate-capped settle burst (the 2026-06-09 correction-trigger
    fix) in addition to the plain snapshot reply."""
    snapshots = []
    server = WulframServer.__new__(WulframServer)
    server.correction_gate_enabled = True
    server.state_request_burst_enabled = True
    server.state_request_burst_min_interval = 1.75
    server.state_sync_correction_burst_count = 6
    server.state_sync_correction_burst_interval = 0.10
    server.active_input_correction_suppress_window = 0.35
    server.state_sync_reply_allow_all = True
    server.state_sync_reply_hosts = set()
    server._state_sync_blocked_clients = set()
    server.remote_full_local_state_delay = 2.0
    server.update_local_state_mode = "off"
    server._send_state_sync_snapshot = lambda *a, **k: snapshots.append(k)

    session = Session()
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
    ctx.last_decoded_input = {"fwd": 0.0, "strafe": 0.0}

    payload = struct.pack(">BII", 0x0C, 0x00ABCDEF, 0)
    server._handle_state_request(ctx, payload, ctx.client_addr)

    assert len(snapshots) == 1, snapshots
    assert snapshots[0].get("include_view_update") is False
    assert ctx.correction_burst_remaining == 6
    assert ctx.correction_burst_interval_s == 0.10
    assert ctx.last_state_request_burst_queue > 0.0

    # Second request inside the cap window: snapshot replies, burst does not re-queue.
    ctx.correction_burst_remaining = 0
    server._handle_state_request(ctx, payload, ctx.client_addr)
    assert len(snapshots) == 2
    assert ctx.correction_burst_remaining == 0
    print("test_state_request_queues_burst_under_default_gate: PASSED")
    return True


def test_remote_active_movement_suppresses_visible_correction_burst():
    """Do not queue hard replay corrections while OG is actively driving."""
    server = WulframServer.__new__(WulframServer)
    server.state_sync_correction_burst_count = 6
    server.state_sync_correction_burst_interval = 0.10
    server.active_input_correction_suppress_window = 0.35

    remote = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    now = time.monotonic()
    remote.last_action_packet_time = now
    remote.last_decoded_input = {"fwd": 0.61, "strafe": 0.0}
    assert server._remote_movement_input_active(remote, now=now) is True
    assert server._queue_state_sync_correction_burst(remote) is False
    assert remote.correction_burst_remaining == 0

    remote.last_decoded_input = {"fwd": 0.0, "strafe": 0.0}
    assert server._remote_movement_input_active(remote, now=now) is False
    assert server._queue_state_sync_correction_burst(remote) is True
    assert remote.correction_burst_remaining == 6

    # FREEZE FIX (2026-06-27): a HELD movement axis means actively driving even when
    # the last ACTION packet is stale -- the OG client only sends ACTION_UPDATE on input
    # CHANGE, so holding forward yields no fresh packets. The old packet-recency window
    # reported "not moving" here and let the velocity-zeroing burst fire mid-drive =
    # the persistent freeze. Held fwd now suppresses the burst regardless of recency.
    remote.correction_burst_remaining = 0
    remote.last_decoded_input = {"fwd": 0.61, "strafe": 0.0}
    remote.last_action_packet_time = now - 1.0
    assert server._remote_movement_input_active(remote, now=now) is True
    assert server._queue_state_sync_correction_burst(remote) is False
    assert remote.correction_burst_remaining == 0

    # Held TURN (no fwd/strafe) is also active driving and must suppress corrections.
    remote.last_decoded_input = {"fwd": 0.0, "strafe": 0.0, "turn": -0.64}
    assert server._remote_movement_input_active(remote, now=now) is True
    print("test_remote_active_movement_suppresses_visible_correction_burst: PASSED")
    return True


def test_state_request_active_movement_skips_view_update_correction():
    """STATE_REQUEST during active drive should not inject local sync packets."""
    captured = []
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "off"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
    server.state_sync_reply_allow_all = True
    server.state_sync_reply_hosts = set()
    server._state_sync_blocked_clients = set()
    server.state_sync_view_mode = "all"
    server.state_sync_snapshot_mode = "remote_live"
    server.state_sync_correction_burst_count = 6
    server.state_sync_correction_burst_interval = 0.10
    server.active_input_correction_suppress_window = 0.35
    server.remote_full_local_state_delay = 2.0
    server.spawn_tank_weapon_type = 2
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos
    server._get_network_tick = lambda ctx: 0x12345678
    server.debug_viewpoint = False
    server.debug_udp_raw = False
    server.pktlog = SimpleNamespace(enabled=False)
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
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (7.0, 1.0, 0.0)
    ctx.player_pose = {"roll": 0.0, "pitch": 0.0}
    ctx.player_heading = 0.25
    ctx.player_yaw = -0.25
    ctx.last_state_sync_send = 0.0
    ctx.last_action_packet_time = time.monotonic()
    ctx.last_decoded_input = {"fwd": 0.61, "strafe": 0.0}

    payload = struct.pack(">BII", 0x0C, 0x00ABCDEF, 0)
    server._handle_state_request(ctx, payload, ctx.client_addr)

    assert len(captured) == 0, captured
    assert ctx.state_sync_reply_count == 0
    assert ctx.state_sync_view_reply_count == 0
    assert ctx.correction_burst_remaining == 0
    assert ctx.state_request_count == 1
    print("test_state_request_active_movement_skips_view_update_correction: PASSED")
    return True


def test_batched_state_request_sees_later_action_update_movement():
    """Datagram pre-scan should suppress sync before a later batched ACTION_UPDATE is handled."""
    server = WulframServer.__new__(WulframServer)
    server.active_input_correction_suppress_window = 0.35

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.weapon_system = WeaponSystem()
    state_request = struct.pack(">BII", 0x0C, 0x00ABCDEF, 0)
    action_update = _build_single_slot_action_update(
        ctx.weapon_system,
        BehaviorSlot.MOVING_FORWARD,
        0.61,
    )

    assert server._udp_packets_have_active_movement_input(
        ctx,
        [state_request, action_update],
    ) is True
    ctx._datagram_active_movement_input = True
    assert server._remote_movement_input_active(ctx, now=time.monotonic()) is True
    print("test_batched_state_request_sees_later_action_update_movement: PASSED")
    return True


def test_remote_empirical_view_update_correction_uses_fresh_remote_timestamp():
    """Explicit OG correction bursts should use a fresh remote replay wrapper."""
    server = WulframServer.__new__(WulframServer)
    server.correction_mode = "view_update"
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.spawn_tank_weapon_type = 2
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos

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
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (1.0, 2.0, 3.0)
    ctx.player_pose = {"roll": 0.125, "pitch": -0.25}
    ctx.player_heading = 0.5
    ctx.last_state_request_id = 0x00ABCDEF
    ctx.last_state_request_time = time.monotonic()

    payload, label, corr_pos, corr_rot, inc_pos, inc_rot = server._build_empirical_correction_payload(
        ctx,
        tick=0x00123456,
        include_local_state=True,
        health=1.0,
        fuel=1.0,
        weapon_type=0,
        ammo_bits=6,
        ammo_mask=0x3F,
        pt_bits=16,
        pt_angle=1.0,
        st_bits=16,
        st_angle=-1.0,
    )

    timestamp, tick, local_state, entities = decode_view_update(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert label == "CORRECTION(view_update)"
    assert inc_pos is True
    assert inc_rot is True
    assert timestamp == _remote_view_timestamp(0x00123456)
    assert tick == 0x00123456
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert local_state.ammo_mask == 0
    assert len(entities) == 1
    assert entities[0].entity_id == 0x14EA
    assert entities[0].position is not None
    assert entities[0].velocity is not None
    assert entities[0].rotation is not None
    assert all(abs(a - b) < 0.3 for a, b in zip(entities[0].position, corr_pos))
    assert all(abs(a - b) < 0.01 for a, b in zip(entities[0].rotation, corr_rot))
    print("test_remote_empirical_view_update_correction_uses_fresh_remote_timestamp: PASSED")
    return True


def test_view_update_create_tank_decodes_definition_shape():
    """Definition-bearing VIEW_UPDATE should roundtrip with bit 0 plus pos/rot."""
    payload = build_view_update_create_tank(
        tick=0x00123456,
        timestamp=0x00ABCDEF,
        entity_id=0x14EA,
        entity_type=EntityType.TANK,
        team=2,
        pos=(4950.0, 5100.0, 5.0),
        behavior_type=2,
        include_health=True,
        health=0.875,
        fuel=0.625,
        weapon_id=2,
        rot=(0.125, -0.25, 0.5),
    )

    timestamp, tick, local_state, entities = decode_view_update(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert timestamp == 0x00ABCDEF
    assert tick == 0x00123456
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert len(entities) == 1
    assert entities[0].entity_id == 0x14EA
    assert entities[0].is_manned is True
    assert entities[0].entity_type == EntityType.TANK
    assert entities[0].team_id == 2
    assert entities[0].position is not None
    assert entities[0].velocity is None
    assert entities[0].rotation is not None
    print("test_view_update_create_tank_decodes_definition_shape: PASSED")
    return True


def test_remote_empirical_view_update_define_correction_uses_definition_shape():
    """The experimental OG correction mode should set definition, pos, and rot under VIEW_UPDATE."""
    server = WulframServer.__new__(WulframServer)
    server.correction_mode = "view_update_define"
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.spawn_tank_weapon_type = 2
    server._get_health_value = lambda ctx: 1.0
    server._get_energy_value = lambda ctx: 1.0
    server._to_client_pos = lambda pos: pos

    session = Session()
    session.translation_ack_received = True
    session.in_game = True
    session.entity_id = 0x14EA
    session.team_id = 2
    session.udp_addr = ("10.10.10.2", 50000)
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=session,
        entity_id=0x14EA,
    )
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (4950.0, 5100.0, 5.0)
    ctx.player_vel = (1.0, 2.0, 3.0)
    ctx.player_pose = {"roll": 0.125, "pitch": -0.25}
    ctx.player_heading = 0.5
    ctx.last_state_request_id = 0x00ABCDEF
    ctx.last_state_request_time = time.monotonic()

    payload, label, corr_pos, corr_rot, inc_pos, inc_rot = server._build_empirical_correction_payload(
        ctx,
        tick=0x00123456,
        include_local_state=True,
        health=1.0,
        fuel=1.0,
        weapon_type=0,
        ammo_bits=6,
        ammo_mask=0x3F,
        pt_bits=16,
        pt_angle=1.0,
        st_bits=16,
        st_angle=-1.0,
    )

    timestamp, tick, local_state, entities = decode_view_update(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert label == "CORRECTION(view_update_define)"
    assert inc_pos is True
    assert inc_rot is True
    assert timestamp == _remote_view_timestamp(0x00123456)
    assert tick == 0x00123456
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert local_state.ammo_mask == 0
    assert len(entities) == 1
    assert entities[0].entity_id == 0x14EA
    assert entities[0].entity_type == EntityType.TANK
    assert entities[0].team_id == 2
    assert entities[0].position is not None
    assert entities[0].velocity is None
    assert entities[0].rotation is not None
    assert all(abs(a - b) < 0.3 for a, b in zip(entities[0].position, corr_pos))
    assert all(abs(a - b) < 0.01 for a, b in zip(entities[0].rotation, corr_rot))
    print("test_remote_empirical_view_update_define_correction_uses_definition_shape: PASSED")
    return True


def test_remote_state_sync_defaults_to_live_snapshot_for_remote_og():
    """Remote OG STATE_REQUEST replies should not drag live movement to stale history."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
    server.state_sync_snapshot_mode = "remote_live"
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
    _, _, update_entities = decode_update_array(
        update_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    _, _, _, view_entities = decode_view_update(
        view_payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )

    assert ctx.last_state_sync_snapshot_source == "live"
    assert all(abs(a - b) < 0.25 for a, b in zip(update_entities[0].position, (4950.0, 5100.0, 5.0)))
    assert all(abs(a - b) < 0.25 for a, b in zip(view_entities[0].position, (4950.0, 5100.0, 5.0)))
    assert all(abs(a - b) < 0.05 for a, b in zip(update_entities[0].velocity, (9.0, 9.0, 0.0)))
    assert update_entities[0].position == view_entities[0].position
    print("test_remote_state_sync_defaults_to_live_snapshot_for_remote_og: PASSED")
    return True


def test_remote_state_sync_reply_uses_request_aligned_authoritative_pose():
    """STATE_REQUEST replies should use the cached authoritative pose nearest the replay tick."""
    server = WulframServer.__new__(WulframServer)
    server.update_local_state_mode = "wf"
    server.update_entity_vitals = False
    server.view_update_local_stats = False
    server.view_update_entity_vitals = False
    server.state_sync_snapshot_mode = "history"
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
    assert timestamp == _remote_view_timestamp(0x12345678)
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
    server.state_sync_snapshot_mode = "history"
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
    assert timestamp == _remote_view_timestamp(0x12345678)
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
    """Promoted transform-bearing heartbeats stay on the short-form-safe local-state.

    GOAL 7 (16a3bfb, 2026-06-04): an ORDINARY steady-state heartbeat (no
    transform) now carries zero entity records (the mask=0 stomp-entity zeroed
    the OG client's predicted angular velocity). A transform-bearing heartbeat
    (correction shape) keeps the full pos+vel+rot entity record AND the
    spawn-safe short-form local-state prefix (weapon=spawn_tank_weapon_type,
    zeroed ammo/turret bits).
    """
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

    # Steady-state heartbeat (no transform): GOAL-7 zero-entity HUD form.
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
    assert len(entities) == 0, entities

    # Transform-bearing heartbeat (correction shape): full transform entity
    # plus the short-form-safe local-state prefix.
    payload = server._build_local_state_heartbeat(
        ctx,
        tick=0x12345678,
        entity_id=0x14EA,
        include_health=True,
        health=1.0,
        fuel=1.0,
        pos=ctx.player_pos,
        rot=(0.0, 0.0, 0.0),
    )
    tick, local_state, entities = decode_update_array(
        payload,
        behavior_config=parse_behavior(build_behavior_packet()),
    )
    assert tick == 0x12345678
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert len(entities) == 1
    assert entities[0].position is not None
    assert entities[0].velocity is not None
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
    assert entities[0].position is not None
    assert entities[0].velocity is not None
    assert entities[0].rotation is not None
    assert abs(entities[0].rotation[0] - 0.125) < 1e-3
    assert abs(entities[0].rotation[1] + 0.375) < 1e-3
    assert abs(entities[0].rotation[2] - 0.25) < 1e-3
    print("test_remote_sync_heartbeat_helper_uses_heading_not_player_yaw: PASSED")
    return True


def test_remote_spawn_bootstrap_heartbeat_uses_safe_full_transform_shape():
    """Fresh remote OG spawn bootstrap should get a complete transform heartbeat."""
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
    assert entities[0].position is not None
    assert entities[0].velocity is not None
    assert entities[0].rotation is not None
    assert entities[0].angular_velocity is not None
    assert abs(entities[0].rotation[0] - 0.125) < 1e-3
    assert abs(entities[0].rotation[1] + 0.375) < 1e-3
    assert abs(entities[0].rotation[2] - 0.25) < 1e-3
    print("test_remote_spawn_bootstrap_heartbeat_uses_safe_full_transform_shape: PASSED")
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


def test_caltrop_projectile_spawn_and_update_decode_promoted_wire_shape():
    """Caltrop spawn packets should define type 13 and updates should carry motion."""
    proj = Projectile(
        entity_id=21015,
        entity_type=EntityType.CALTROP,
        owner_id=0x14EA,
        team=2,
        pos=(2632.83, 3039.84, 64.75),
        vel=(80.0, 0.0, 0.0),
        spawn_time=0.0,
        lifetime=30.0,
    )

    spawn_payload = build_projectile_spawn_packet(proj, 0x12345670)
    update_payload = build_projectile_update_packet(proj, 0x12345671, dt=0.1)

    spawn_tick, spawn_local, spawn_entities = decode_update_array(spawn_payload)
    update_tick, update_local, update_entities = decode_update_array(update_payload)

    assert spawn_tick == 0x12345670
    assert update_tick == 0x12345671
    assert spawn_local is None
    assert update_local is None
    assert len(spawn_entities) == 1
    assert len(update_entities) == 1
    assert spawn_entities[0].entity_id == 21015
    assert update_entities[0].entity_id == 21015
    assert spawn_entities[0].entity_type == EntityType.CALTROP
    # UPDATE_ARRAY motion updates identify the existing object; the type is
    # established by the spawn/definition packet above.
    assert update_entities[0].entity_type in (-1, None, EntityType.CALTROP)
    assert spawn_entities[0].velocity is not None
    assert update_entities[0].velocity is not None
    assert math.isclose(spawn_entities[0].velocity[0], 80.0, abs_tol=0.1)
    assert math.isclose(update_entities[0].velocity[0], 80.0, abs_tol=0.1)
    print("test_caltrop_projectile_spawn_and_update_decode_promoted_wire_shape: PASSED")
    return True


def test_loopback_projectile_update_stays_entity_only():
    """Loopback projectile updates carry the same OG local-state prefix as remote.

    The loopback client-type fork was retired (9ea5dbd, 2026-06-02) and the
    viewer discriminator is _is_og_client (login bootstrap mode, og by
    default), so a 127.0.0.1 viewer gets the short-form-safe local-state
    prefix — the OG client reads garbage health from an entity-only packet.
    """
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
    # Unified OG path: short-form-safe local-state prefix present.
    assert local_state is not None
    assert local_state.weapon_id == 2
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
    # roster_sent must be False to exercise the roster-send path: _send_team_switch_roster
    # now guards against re-sending when roster_sent is already True (the duplicate
    # "copy of yourself as spectator" row fix), so a True value here would correctly
    # send nothing. (Test updated 2026-06-26 — was orphaned with a stale True.)
    session.roster_sent = False
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
    assert ctx.tcp_handler.sent[0][1:5] == b"\x00\x00\x05\x39"  # f1 = player_id
    # GOAL 3 fix: the OG scoreboard team filter (PlayerEntry+0x08) is fed by wire
    # field-3 (u16), NOT field-2. f2 (bytes 5-9) is now 0 (clan_id); the team (2)
    # lives in the u16 at bytes 9-11.
    assert ctx.tcp_handler.sent[0][5:9] == b"\x00\x00\x00\x00"  # f2 = clan_id (unused)
    assert ctx.tcp_handler.sent[0][9:11] == b"\x00\x02"          # f3 = team filter
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


def test_weapon_system_pulse_shell_respects_pitch_when_enabled():
    """The server body-pitch gate must reach the projectile velocity math."""
    ws = WeaponSystem()
    ws.player_pos = (4950.0, 5100.0, 5.0)
    ws.player_rot = (0.0, -0.2, 0.0)
    ws.player_team = 2
    ws.use_pitch = True
    ws.behavior_slots[12] = 1.0

    projectiles, energy_spent = ws.update(dt=1.0, available_energy=100.0)

    assert len(projectiles) == 1
    projectile = projectiles[0]
    assert projectile.entity_type == EntityType.PULSE_SHELL
    assert projectile.vel[2] > 10.0, projectile.vel
    assert abs(projectile.vel[2] - (75.0 * math.sin(0.2))) < 0.001, projectile.vel
    assert energy_spent > 0.0
    print("test_weapon_system_pulse_shell_respects_pitch_when_enabled: PASSED")
    return True


def test_weapon_system_og_direct_trigger_slots_fire_promoted_projectiles():
    """Promoted direct-fire slots should spawn the expected server projectile type."""
    cases = [
        (12, WeaponType.PULSE_CANNON, EntityType.PULSE_SHELL, "Pulse Shell"),
        (13, WeaponType.PIERCER, EntityType.PIERCER, "Piercer"),
        (14, WeaponType.THUMPER, EntityType.THUMPER, "Thumper"),
        (15, EntityType.CALTROP, EntityType.CALTROP, "Caltrop"),
        (16, WeaponType.HUNTER_SEEKER, EntityType.HUNTER, "Hunter"),
        (17, WeaponType.MINE, EntityType.MINE, "Mine"),
    ]
    for trigger_slot, weapon_slot, entity_type, label in cases:
        ws = WeaponSystem()
        ws.player_pos = (4950.0, 5100.0, 5.0)
        ws.player_rot = (0.0, 0.0, 0.0)
        ws.player_team = 2
        ws.behavior_slots[trigger_slot] = 1.0

        projectiles, energy_spent = ws.update(dt=1.0, available_energy=100.0)

        assert OG_DIRECT_TRIGGER_WEAPON_SLOTS[trigger_slot] == weapon_slot, (trigger_slot, weapon_slot)
        assert len(projectiles) == 1, (label, projectiles)
        assert projectiles[0].entity_type == entity_type, (label, projectiles[0])
        assert energy_spent > 0.0, (label, energy_spent)
        if entity_type == EntityType.MINE:
            assert projectiles[0].vel == (0.0, 0.0, 0.0), projectiles[0].vel
        else:
            assert math.sqrt(sum(float(v) * float(v) for v in projectiles[0].vel)) > 0.0, projectiles[0].vel

    print("test_weapon_system_og_direct_trigger_slots_fire_promoted_projectiles: PASSED")
    return True


def test_weapon_system_caltrop_uses_promoted_lifecycle_constants():
    """OG key 5 / Caltrop should spawn a real short-lived projectile."""
    ws = WeaponSystem()
    ws.player_pos = (4950.0, 5100.0, 5.0)
    ws.player_rot = (0.0, 0.0, 0.0)
    ws.player_team = 2
    ws.player_id = 0xCA17
    ws.behavior_slots[15] = 1.0

    projectiles, energy_spent = ws.update(dt=1.0, available_energy=100.0)

    assert OG_DIRECT_TRIGGER_WEAPON_SLOTS[15] == EntityType.CALTROP
    assert len(projectiles) == 1, projectiles
    projectile = projectiles[0]
    assert projectile.entity_type == EntityType.CALTROP, projectile
    assert projectile.owner_id == 0xCA17, projectile
    assert projectile.team == 2, projectile
    assert projectile.lifetime == 30.0, projectile
    assert abs(math.sqrt(sum(float(v) * float(v) for v in projectile.vel)) - 80.0) < 0.001, projectile.vel
    assert energy_spent == 4.0, energy_spent
    assert ws.fire_cooldown == 1.0, ws.fire_cooldown
    assert ws.projectiles == [projectile], ws.projectiles
    print("test_weapon_system_caltrop_uses_promoted_lifecycle_constants: PASSED")
    return True


def test_weapon_system_chain_gun_autocannon_fire_slot_hitscan_path():
    """Chain Gun/autocannon is the current fire-slot hitscan path, not a number-key projectile."""
    ws = WeaponSystem()
    ws.player_pos = (4950.0, 5100.0, 5.0)
    ws.player_rot = (0.0, 0.0, 0.0)
    ws.player_team = 2
    ws.current_weapon = WeaponType.CHAIN_GUN
    fired = []
    ws.on_chain_gun_fire = lambda **kwargs: fired.append(kwargs)
    ws.behavior_slots[BehaviorSlot.FIRE] = 1.0

    projectiles, energy_spent = ws.update(dt=1.0, available_energy=100.0)

    assert WeaponType.CHAIN_GUN not in OG_DIRECT_TRIGGER_WEAPON_SLOTS.values()
    assert projectiles == [], projectiles
    assert len(fired) == 1, fired
    assert fired[0]["pos"] == ws.player_pos, fired
    assert energy_spent == 2.0, energy_spent
    print("test_weapon_system_chain_gun_autocannon_fire_slot_hitscan_path: PASSED")
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


def test_weapon_system_accepts_empty_action_update_keepalive():
    """Count=0 ACTION_UPDATE packets still carry a valid tick/frame keepalive."""
    ws = WeaponSystem()
    packet = b"\x0A\x00" + struct.pack(">II", 0x12345678, 77)

    assert ws.decode_action_update(packet) is True
    assert ws.client_frame_counter == 77
    assert ws.prev_action_client_tick == 0x12345678
    assert abs(ws.behavior_slots[TANK_SOFTBODY_CONTROL_SLOT] - OG_TANK_SOFTBODY_IDLE_SLOT5) < 1e-6
    print("test_weapon_system_accepts_empty_action_update_keepalive: PASSED")
    return True


def _build_single_slot_action_update(ws: WeaponSystem, slot_idx: int, value: float) -> bytes:
    bw = BitWriter()
    bw.write_bits(8, 1)
    bw.write_bits(32, 0x12345679)
    bw.write_bits(32, 78)
    bw.write_bits(ws.slot_index_bits, int(slot_idx))
    if slot_idx == BehaviorSlot.UPWARD_THRUST:
        raw = quantize_float(value, ws.zoom_max, ws.zoom_range, ws.zoom_bits)
        bw.write_bits(ws.zoom_bits, raw)
    elif slot_idx in ACTION_ANALOG_SLOTS:
        raw = quantize_float(value, ws.control_max, ws.control_range, ws.control_bits)
        bw.write_bits(ws.control_bits, raw)
    else:
        bw.write_bits(1, 1 if value >= 0.5 else 0)
    return b"\x0A" + bw.get_bytes()


def _build_action_dump(ws: WeaponSystem, slots: dict[int, float]) -> bytes:
    bw = BitWriter()
    bw.write_bits(32, 0x1234567A)
    bw.write_bits(32, 79)
    for slot_idx in range(1, 22):
        value = float(slots.get(slot_idx, 0.0))
        if slot_idx == BehaviorSlot.UPWARD_THRUST:
            raw = quantize_float(value, ws.zoom_max, ws.zoom_range, ws.zoom_bits)
            bw.write_bits(ws.zoom_bits, raw)
        elif slot_idx in ACTION_DUMP_CONTROL_SLOTS:
            raw = quantize_float(value, ws.control_max, ws.control_range, ws.control_bits)
            bw.write_bits(ws.control_bits, raw)
        else:
            bw.write_bits(1, 1 if value >= 0.5 else 0)
    return b"\x09" + bw.get_bytes()


def test_weapon_system_slot5_release_preserves_og_slider_value():
    """ACTION_UPDATE slot-5 zero is a key release, not an OG softbody reset."""
    ws = WeaponSystem()
    ws.behavior_slots[TANK_SOFTBODY_CONTROL_SLOT] = OG_TANK_SOFTBODY_Z_SLOT5

    packet = _build_single_slot_action_update(ws, BehaviorSlot.UPWARD_THRUST, 0.0)

    assert ws.decode_action_update(packet) is True
    assert abs(ws.behavior_slots[TANK_SOFTBODY_CONTROL_SLOT] - OG_TANK_SOFTBODY_Z_SLOT5) < 1e-6
    assert abs(tank_softbody_control_slot_value(ws.behavior_slots) - OG_TANK_SOFTBODY_Z_SLOT5) < 1e-6
    print("test_weapon_system_slot5_release_preserves_og_slider_value: PASSED")
    return True


def test_weapon_system_action_dump_slot5_zero_preserves_og_slider_value():
    """ACTION_DUMP slot-5 zero should not erase the persistent OG Q/Z slider."""
    ws = WeaponSystem()
    ws.behavior_slots[TANK_SOFTBODY_CONTROL_SLOT] = OG_TANK_SOFTBODY_Z_SLOT5

    packet = _build_action_dump(ws, {BehaviorSlot.UPWARD_THRUST: 0.0})

    assert ws.decode_action_dump(packet) is True
    assert abs(ws.behavior_slots[TANK_SOFTBODY_CONTROL_SLOT] - OG_TANK_SOFTBODY_Z_SLOT5) < 1e-6
    assert abs(tank_softbody_control_slot_value(ws.behavior_slots) - OG_TANK_SOFTBODY_Z_SLOT5) < 1e-6
    print("test_weapon_system_action_dump_slot5_zero_preserves_og_slider_value: PASSED")
    return True


def test_weapon_system_action_dump_slot5_nonzero_updates_og_slider_value():
    """ACTION_DUMP should still accept explicit nonzero slot-5 Q/Z values."""
    ws = WeaponSystem()
    ws.behavior_slots[TANK_SOFTBODY_CONTROL_SLOT] = OG_TANK_SOFTBODY_IDLE_SLOT5

    packet = _build_action_dump(ws, {BehaviorSlot.UPWARD_THRUST: OG_TANK_SOFTBODY_Q_SLOT5})

    assert ws.decode_action_dump(packet) is True
    assert abs(ws.behavior_slots[TANK_SOFTBODY_CONTROL_SLOT] - OG_TANK_SOFTBODY_Q_SLOT5) < 0.01
    assert tank_softbody_control_slot_value(ws.behavior_slots) > OG_TANK_SOFTBODY_IDLE_SLOT5
    print("test_weapon_system_action_dump_slot5_nonzero_updates_og_slider_value: PASSED")
    return True


def test_tank_softbody_control_ignores_live_slot6_lean_by_default():
    """Slot 6 is live OG lean data; it is only a legacy opt-in fallback."""
    slots = [0.0] * 22
    slots[BehaviorSlot.SLOT6] = -0.7324

    assert abs(tank_softbody_control_slot_value(slots) - OG_TANK_SOFTBODY_IDLE_SLOT5) < 1e-6
    assert abs(
        tank_softbody_control_slot_value(slots, allow_legacy_slot6=True) + 0.7324
    ) < 1e-6
    print("test_tank_softbody_control_ignores_live_slot6_lean_by_default: PASSED")
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


def test_og_viewer_replication_gates_skip_remote_only():
    """T3 isolation gates apply uniformly — loopback is no longer exempt.

    The loopback client-type fork was retired (9ea5dbd, 2026-06-02:
    _is_loopback_client always False), so the og_viewer_* gates suppress the
    streams for EVERY client, 127.0.0.1 included.
    """
    server = WulframServer.__new__(WulframServer)
    server.og_viewer_roster_entry = False
    server.og_viewer_entity_create = False
    server.og_viewer_remote_updates = False

    remote_session = Session()
    remote_session.translation_ack_received = True
    remote_ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=remote_session,
        entity_id=1337,
    )
    remote_ctx.known_roster_ids = set()
    remote_ctx.known_entity_ids = set()

    loopback_session = Session()
    loopback_session.translation_ack_received = True
    loopback_ctx = ClientContext(
        client_id=2,
        client_addr=("127.0.0.1", 50001),
        session=loopback_session,
        entity_id=1338,
    )

    assert server._og_viewer_replication_enabled(remote_ctx, "roster") is False
    assert server._og_viewer_replication_enabled(remote_ctx, "entity_create") is False
    assert server._og_viewer_replication_enabled(remote_ctx, "remote_updates") is False
    # Unified path: loopback clients honor the same gates.
    assert server._og_viewer_replication_enabled(loopback_ctx, "roster") is False
    assert server._og_viewer_replication_enabled(loopback_ctx, "entity_create") is False
    assert server._og_viewer_replication_enabled(loopback_ctx, "remote_updates") is False

    player_session = Session()
    player_session.player_id = 1338
    player_session.entity_id = 1338
    player_session.username = "Target"
    player_session.team_id = 2
    player_ctx = ClientContext(
        client_id=3,
        client_addr=("127.0.0.1", 50002),
        session=player_session,
        entity_id=1338,
    )
    player_ctx.kills = 0
    player_ctx.deaths = 0

    sent = []
    server._send_packet_to_client = lambda *args, **kwargs: sent.append((args, kwargs)) or True
    server._send_roster_entry(remote_ctx, player_ctx)
    server._send_entity_create(remote_ctx, player_ctx)
    server._send_remote_player_updates(remote_ctx, tick=0x12345678)

    assert sent == [], sent
    assert remote_ctx.known_roster_ids == set(), remote_ctx.known_roster_ids
    assert remote_ctx.known_entity_ids == set(), remote_ctx.known_entity_ids
    print("test_og_viewer_replication_gates_skip_remote_only: PASSED")
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
    # Loopback fork retired (9ea5dbd, 2026-06-02): the remote_transient_fx
    # gate applies to ALL clients, so nothing is delivered while it is off.
    assert sent == [], sent
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
    # Loopback fork retired (9ea5dbd): an unpromoted viewer gets the
    # spawn-safe short-form weapon (spawn_tank_weapon_type), like remote OG.
    assert local_state.weapon_id == 2
    assert len(entities) == 1, entities
    assert entities[0].entity_id == 1338
    print("test_loopback_entity_create_decodes_roundtrip: PASSED")
    return True


def test_loopback_remote_player_update_decodes_roundtrip():
    """Loopback remote updates decode cleanly on the unified OG viewer path.

    Loopback fork retired (9ea5dbd, 2026-06-02): the viewer gets the same
    short-form-safe local-state prefix as a remote OG client, not the old
    entity-only shape.
    """
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
    # Unified OG path: short-form-safe local-state prefix present.
    assert local_state is not None
    assert local_state.weapon_id == 2
    assert len(entities) == 1, entities
    assert entities[0].entity_id == 1338
    assert entities[0].velocity is not None
    assert entities[0].rotation is not None
    assert entities[0].angular_velocity is not None
    print("test_loopback_remote_player_update_decodes_roundtrip: PASSED")
    return True


def test_loopback_heartbeat_decodes_roundtrip():
    """Loopback heartbeats decode cleanly on the unified spawn-safe form.

    Loopback fork retired (9ea5dbd): an unpromoted client stays on the
    10-byte spawn-safe short form (weapon=spawn_tank_weapon_type), and
    GOAL 7 (16a3bfb) dropped the dummy/stomp entity record, so the
    heartbeat carries zero entities.
    """
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
    assert local_state.weapon_id == 2
    assert len(entities) == 0, entities
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


def test_server_tank_motion_uses_fuel_mobility_factor():
    """Full-fuel tanks should not get the low-fuel 0.4 mobility floor."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 45.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.turn_deadzone = 0.05
    server.turn_sign = -1.0
    server.up_axis = "z"
    server.terrain = None
    server.terrain_pitch_enabled = False
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
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
    ctx.player_fuel = 33000.0
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}

    server._update_player_position(ctx, dt_override=1.0)

    vx, vy, vz = ctx.player_vel
    assert abs(vx - 85.0) < 1e-4, vx
    assert abs(vy) < 1e-4, vy
    assert abs(vz) < 1e-4, vz
    print("test_server_tank_motion_uses_fuel_mobility_factor: PASSED")
    return True


def test_server_tank_motion_reduces_mobility_when_low_fuel():
    """Only low raw fuel should activate the OG 0.4-1.0 mobility ramp."""
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
    ctx.player_fuel = 0.0
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}

    server._update_player_position(ctx, dt_override=1.0)

    vx, vy, vz = ctx.player_vel
    assert abs(vx - 34.0) < 1e-4, vx
    assert abs(vy) < 1e-4, vy
    assert abs(vz) < 1e-4, vz
    print("test_server_tank_motion_reduces_mobility_when_low_fuel: PASSED")
    return True


def test_tank_ground_contact_damping_limits_low_hover_speed():
    """Low Z hover should keep the extra terrain-contact horizontal loss."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 30.0
    server.linear_damp_driving = 1.5
    server.linear_damp_coasting = 1.5
    server.tank_ground_contact_damp = 6.0
    server.turn_deadzone = 0.05
    server.turn_sign = -1.0
    server.up_axis = "z"
    server.terrain = SimpleNamespace(
        get_height=lambda x, y: 0.0,
        get_slope=lambda x, y: (0.0, 0.0),
    )
    server.terrain_height_offset = 0.0
    server.terrain_physics_height_offset = 0.0
    server.terrain_pitch_enabled = False
    server.tank_suspension_enabled = True
    server.tank_suspension_model = "softbody"
    server.tank_suspension_damping = 6.0
    server.tank_spring_base_offset = 2.0
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.jump_jets_enabled = False
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
    ctx.injected_input = (0.5188, 0.0)
    ctx.weapon_system = WeaponSystem()
    ctx.weapon_system.behavior_slots[5] = OG_TANK_SOFTBODY_Z_SLOT5
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (0.0, 0.0, 1.628)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_fuel = 33000.0
    ctx.player_energy = 100.0
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}

    for _ in range(60):
        server._update_player_position(ctx, dt_override=1.0 / 30.0)

    vx, vy, vz = ctx.player_vel
    controller = ctx.debug_last_controller_step
    assert abs(controller["linear_damp"] - 1.5) < 1e-6, controller
    assert abs(controller["horizontal_damp"] - 6.0) < 1e-6, controller
    assert abs(controller["tank_ground_contact_damp"] - 6.0) < 1e-6, controller
    assert 7.0 <= vx <= 9.0, vx
    assert abs(vy) < 1e-4, vy
    assert abs(vz) < 1e-4, vz
    print("test_tank_ground_contact_damping_limits_low_hover_speed: PASSED")
    return True


def test_tank_high_hover_uses_linear_damping_for_w_motion():
    """High Q hover should move like OG with the measured 1.5 linear damping."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 30.0
    server.linear_damp_driving = 1.5
    server.linear_damp_coasting = 1.5
    server.tank_ground_contact_damp = 6.0
    server.turn_deadzone = 0.05
    server.turn_sign = -1.0
    server.up_axis = "z"
    server.terrain = SimpleNamespace(
        get_height=lambda x, y: 0.0,
        get_slope=lambda x, y: (0.0, 0.0),
    )
    server.terrain_height_offset = 0.0
    server.terrain_physics_height_offset = 0.0
    server.terrain_pitch_enabled = False
    server.tank_suspension_enabled = True
    server.tank_suspension_model = "softbody"
    server.tank_suspension_damping = 6.0
    server.tank_spring_base_offset = 2.0
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.jump_jets_enabled = False
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
    ctx.injected_input = (0.5188, 0.0)
    ctx.weapon_system = WeaponSystem()
    ctx.weapon_system.behavior_slots[5] = OG_TANK_SOFTBODY_Q_SLOT5
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (0.0, 0.0, 3.952)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_fuel = 33000.0
    ctx.player_energy = 100.0
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}

    for _ in range(60):
        server._update_player_position(ctx, dt_override=1.0 / 30.0)

    vx, vy, vz = ctx.player_vel
    controller = ctx.debug_last_controller_step
    assert abs(controller["linear_damp"] - 1.5) < 1e-6, controller
    assert abs(controller["horizontal_damp"] - 1.5) < 1e-6, controller
    assert abs(controller["tank_ground_contact_damp"]) < 1e-6, controller
    assert controller["softbody_scalar_stretch_source"] == "entity_velocity", controller
    assert abs(
        controller["softbody_scalar_stretch_ratio"]
        - abs(controller["pre_vel"][0]) / OG_TANK_SPRING_STRETCH_SPEED_DENOMINATOR
    ) < 1e-6, controller
    assert max(controller["softbody_point_blend_factors"]) > 0.0, controller
    assert 26.0 <= vx <= 30.0, vx
    assert abs(vy) < 1e-4, vy
    assert abs(vz) < 1e-4, vz
    print("test_tank_high_hover_uses_linear_damping_for_w_motion: PASSED")
    return True


def test_remote_og_movement_input_delay_replays_prior_axis_sample():
    """Remote OG movement should use delayed action history, not immediate slots."""
    server = WulframServer.__new__(WulframServer)
    server.remote_og_movement_input_delay = 0.20
    server.remote_og_movement_input_selection = "latest_before_target"

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    now = time.monotonic()
    ctx.movement_input_history.append({"time": now - 0.40, "fwd": 0.0, "strafe": 0.0})
    ctx.movement_input_history.append({"time": now - 0.05, "fwd": 0.5188, "strafe": 0.0})

    delay = server._remote_og_movement_input_delay_for_ctx(ctx)
    fwd, strafe, source = server._select_delayed_movement_input(
        ctx,
        current_fwd=0.5188,
        current_strafe=0.0,
        delay_s=delay,
    )
    assert abs(delay - 0.20) < 1e-6, delay
    assert fwd == 0.0, (fwd, source)
    assert strafe == 0.0, strafe
    assert source == "delayed_remote_og_action_history", source
    selection = getattr(ctx, "debug_last_movement_input_selection", {})
    assert selection["source"] == "delayed_remote_og_action_history", selection
    assert selection["history_len"] == 2, selection
    assert 0.35 <= selection["selected_age_s"] <= 0.45, selection
    assert selection["selected_fwd"] == 0.0, selection
    assert selection["latest_fwd"] == 0.5188, selection
    assert selection["time_probe_before_found"] is True, selection
    assert selection["time_probe_after_found"] is True, selection
    assert selection["time_probe_nearest_found"] is True, selection
    assert selection["time_probe_before_fwd"] == 0.0, selection
    assert selection["time_probe_after_fwd"] == 0.5188, selection
    assert selection["selected_interval_contains_target"] is True, selection
    assert selection["selected_interval_end_target_error_s"] > 0.0, selection
    assert selection["movement_history_window_s"] == 0.75, selection
    assert selection["movement_history_window_count"] == 2, selection
    assert selection["movement_history_window_total_count"] == 2, selection
    assert selection["movement_history_window_truncated"] is False, selection
    window = selection["movement_history_window"]
    assert window[0]["fwd"] == 0.0 and window[0]["selected"] is True, window
    assert window[1]["fwd"] == 0.5188 and window[1]["future_of_target"] is True, window

    # Loopback fork retired (9ea5dbd, 2026-06-02): the replay delay applies
    # to loopback clients too — only injected control-port input bypasses it.
    ctx.client_addr = ("127.0.0.1", 50000)
    assert abs(server._remote_og_movement_input_delay_for_ctx(ctx) - 0.20) < 1e-6
    ctx.injected_input = (0.0, 0.0)
    assert server._remote_og_movement_input_delay_for_ctx(ctx) == 0.0
    print("test_remote_og_movement_input_delay_replays_prior_axis_sample: PASSED")
    return True


def test_remote_og_movement_input_delay_can_probe_nearest_axis_sample():
    """Default-off nearest selection bounds replay jitter from sparse OG dumps."""
    server = WulframServer.__new__(WulframServer)
    server.remote_og_movement_input_delay = 0.20
    server.remote_og_movement_input_selection = "nearest_to_target"

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    now = time.monotonic()
    ctx.movement_input_history.append({"time": now - 0.40, "fwd": 0.0, "strafe": 0.0})
    ctx.movement_input_history.append({"time": now - 0.05, "fwd": 0.5188, "strafe": 0.0})

    delay = server._remote_og_movement_input_delay_for_ctx(ctx)
    fwd, strafe, source = server._select_delayed_movement_input(
        ctx,
        current_fwd=0.5188,
        current_strafe=0.0,
        delay_s=delay,
    )

    assert abs(delay - 0.20) < 1e-6, delay
    assert fwd == 0.5188, (fwd, source)
    assert strafe == 0.0, strafe
    assert source == "delayed_remote_og_action_history", source
    selection = getattr(ctx, "debug_last_movement_input_selection", {})
    assert selection["selection_policy"] == "nearest_to_target", selection
    assert selection["selected_future_of_target"] is True, selection
    assert abs(selection["selected_fwd"] - 0.5188) < 1e-6, selection
    assert selection["selected_abs_target_error_s"] < 0.20, selection
    assert selection["time_probe_before_found"] is True, selection
    assert selection["time_probe_after_found"] is True, selection
    assert selection["time_probe_nearest_fwd"] == 0.5188, selection
    assert selection["selected_interval_contains_target"] is False, selection

    print("test_remote_og_movement_input_delay_can_probe_nearest_axis_sample: PASSED")
    return True


def test_remote_og_movement_input_can_select_bounded_after_target():
    """Default-off bounded-after policy can probe late OG key consumption."""
    server = WulframServer.__new__(WulframServer)
    server.remote_og_movement_input_delay = 0.20
    server.remote_og_movement_input_selection = "bounded_after_target"
    server.remote_og_movement_input_after_max = 0.12

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    now = time.monotonic()
    ctx.movement_input_history.append({"time": now - 0.25, "fwd": 0.0, "strafe": 0.0})
    ctx.movement_input_history.append({"time": now - 0.09, "fwd": 0.5188, "strafe": 0.0})

    fwd, strafe, source = server._select_delayed_movement_input(
        ctx,
        current_fwd=0.5188,
        current_strafe=0.0,
        delay_s=server.remote_og_movement_input_delay,
    )

    selection = getattr(ctx, "debug_last_movement_input_selection", {})
    assert source == "delayed_remote_og_action_history", source
    assert abs(fwd - 0.5188) < 1e-6, selection
    assert strafe == 0.0, selection
    assert selection["selection_policy"] == "bounded_after_target", selection
    assert selection["bounded_after_max_s"] == 0.12, selection
    assert selection["bounded_after_applied"] is True, selection
    assert selection["bounded_after_reason"] == (
        "after_sample_within_bound_matches_current_input"
    ), selection
    assert selection["selected_future_of_target"] is True, selection
    assert 0.10 <= selection["selected_abs_target_error_s"] <= 0.12, selection
    assert selection["time_probe_before_fwd"] == 0.0, selection
    assert selection["time_probe_after_fwd"] == 0.5188, selection

    server.remote_og_movement_input_after_max = 0.05
    fwd, strafe, source = server._select_delayed_movement_input(
        ctx,
        current_fwd=0.5188,
        current_strafe=0.0,
        delay_s=server.remote_og_movement_input_delay,
    )
    selection = getattr(ctx, "debug_last_movement_input_selection", {})
    assert source == "delayed_remote_og_action_history", source
    assert fwd == 0.0, selection
    assert strafe == 0.0, selection
    assert selection["bounded_after_applied"] is False, selection
    assert selection["bounded_after_reason"] == "after_sample_outside_bound", selection

    print("test_remote_og_movement_input_can_select_bounded_after_target: PASSED")
    return True


def test_remote_og_movement_input_reports_nonzero_time_candidates():
    """Debug fields should expose late nonzero samples hidden behind neutral rows."""
    server = WulframServer.__new__(WulframServer)
    server.remote_og_movement_input_delay = 0.20
    server.remote_og_movement_input_selection = "latest_before_target"
    server.remote_og_movement_input_after_max = 0.12

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    now = time.monotonic()
    ctx.movement_input_history.append({"time": now - 0.766, "fwd": 0.0, "strafe": 0.0})
    ctx.movement_input_history.append({"time": now - 0.125, "fwd": 0.0, "strafe": 0.0})
    ctx.movement_input_history.append({"time": now - 0.050, "fwd": 0.549333, "strafe": 0.0})

    fwd, strafe, source = server._select_delayed_movement_input(
        ctx,
        current_fwd=0.549333,
        current_strafe=0.0,
        delay_s=server.remote_og_movement_input_delay,
    )

    selection = getattr(ctx, "debug_last_movement_input_selection", {})
    assert source == "delayed_remote_og_action_history", source
    assert fwd == 0.0, selection
    assert strafe == 0.0, selection
    assert selection["time_probe_after_found"] is True, selection
    assert selection["time_probe_after_fwd"] == 0.0, selection
    assert selection["time_probe_nearest_fwd"] == 0.0, selection
    assert selection["time_probe_nonzero_before_found"] is False, selection
    assert selection["time_probe_nonzero_after_found"] is True, selection
    assert abs(selection["time_probe_nonzero_after_fwd"] - 0.549333) < 1e-6, selection
    assert 0.14 <= selection["time_probe_nonzero_after_target_error_s"] <= 0.16, selection
    assert selection["time_probe_nonzero_nearest_found"] is True, selection
    assert selection["nonzero_after_within_bounded_after_max"] is False, selection

    print("test_remote_og_movement_input_reports_nonzero_time_candidates: PASSED")
    return True


def test_remote_og_movement_input_history_window_is_bounded_near_target():
    """Selection debug should carry enough nearby samples for offline policy replay."""
    server = WulframServer.__new__(WulframServer)
    server.remote_og_movement_input_delay = 0.20
    server.remote_og_movement_input_selection = "latest_before_target"
    server.remote_og_movement_input_after_max = 0.12

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    now = time.monotonic()
    for index in range(24):
        age = 0.95 - index * 0.04
        ctx.movement_input_history.append(
            {
                "time": now - age,
                "fwd": 0.5 if index >= 18 else 0.0,
                "strafe": 0.0,
                "packet_type": "ACTION_UPDATE",
                "client_tick": 200000 + index,
            }
        )

    fwd, strafe, source = server._select_delayed_movement_input(
        ctx,
        current_fwd=0.5,
        current_strafe=0.0,
        delay_s=server.remote_og_movement_input_delay,
    )

    selection = getattr(ctx, "debug_last_movement_input_selection", {})
    assert source == "delayed_remote_og_action_history", source
    assert fwd == 0.5, selection
    assert strafe == 0.0, selection
    window = selection["movement_history_window"]
    assert selection["movement_history_window_total_count"] > 16, selection
    assert selection["movement_history_window_count"] == 16, selection
    assert selection["movement_history_window_truncated"] is True, selection
    assert any(item["selected"] for item in window), window
    assert all(abs(item["target_error_s"]) <= 0.75 for item in window), window
    assert [item["index"] for item in window] == sorted(item["index"] for item in window), window

    print("test_remote_og_movement_input_history_window_is_bounded_near_target: PASSED")
    return True


def test_remote_og_movement_input_tick_probe_reports_tick_domain_candidates():
    """Default-off tick probe should compare client-tick candidates without changing input."""
    server = WulframServer.__new__(WulframServer)
    server.remote_og_movement_input_delay = 0.10
    server.remote_og_movement_input_selection = "latest_before_target"
    server.remote_og_movement_input_tick_probe = True

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    current_client_tick = 100000
    ctx.tick_offset = current_client_tick - int(get_ticks())
    ctx.last_client_tick = current_client_tick - 5
    now = time.monotonic()
    ctx.movement_input_history.append(
        {
            "time": now - 0.40,
            "fwd": 0.0,
            "strafe": 0.0,
            "client_tick": 99700,
        }
    )
    ctx.movement_input_history.append(
        {
            "time": now - 0.05,
            "fwd": 0.5188,
            "strafe": 0.0,
            "client_tick": 99950,
        }
    )

    fwd, strafe, source = server._select_delayed_movement_input(
        ctx,
        current_fwd=0.5188,
        current_strafe=0.0,
        delay_s=server.remote_og_movement_input_delay,
    )

    assert fwd == 0.0, (fwd, source)
    assert strafe == 0.0, strafe
    assert source == "delayed_remote_og_action_history", source
    selection = getattr(ctx, "debug_last_movement_input_selection", {})
    assert selection["tick_probe_enabled"] is True, selection
    assert selection["tick_probe_target_client_tick"] == 99900, selection
    assert selection["tick_probe_before_client_tick"] == 99700, selection
    assert selection["tick_probe_before_fwd"] == 0.0, selection
    assert selection["tick_probe_nearest_client_tick"] == 99950, selection
    assert selection["tick_probe_nearest_future_of_target"] is True, selection
    assert abs(selection["tick_probe_nearest_fwd"] - 0.5188) < 1e-6, selection
    assert selection["tick_probe_nearest_abs_target_error_ms"] == 50, selection

    print(
        "test_remote_og_movement_input_tick_probe_reports_tick_domain_candidates: PASSED"
    )
    return True


def test_remote_og_movement_input_can_select_tick_domain_candidates():
    """Default-off tick-domain policies can drive replay from client tick timing."""
    server = WulframServer.__new__(WulframServer)
    server.remote_og_movement_input_delay = 0.10
    server.remote_og_movement_input_tick_probe = False
    server.remote_og_movement_input_stale_clamp = 0.0

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.last_client_tick = 100000
    now = time.monotonic()
    ctx.movement_input_history.append(
        {
            "time": now - 0.40,
            "fwd": 0.0,
            "strafe": 0.0,
            "client_tick": 99700,
        }
    )
    ctx.movement_input_history.append(
        {
            "time": now - 0.05,
            "fwd": 0.5188,
            "strafe": 0.0,
            "client_tick": 99950,
        }
    )

    server.remote_og_movement_input_selection = "latest_before_tick_target"
    fwd, strafe, source = server._select_delayed_movement_input(
        ctx,
        current_fwd=0.5188,
        current_strafe=0.0,
        delay_s=server.remote_og_movement_input_delay,
    )
    selection = getattr(ctx, "debug_last_movement_input_selection", {})
    assert source == "delayed_remote_og_action_history", source
    assert fwd == 0.0, selection
    assert strafe == 0.0, selection
    assert selection["selection_policy"] == "latest_before_tick_target", selection
    assert selection["tick_probe_enabled"] is True, selection
    assert selection["tick_probe_target_client_tick"] == 99900, selection
    assert selection["selected_client_tick"] == 99700, selection
    assert selection["selected_tick_target_error_ms"] == -200, selection
    assert selection["selected_tick_abs_target_error_ms"] == 200, selection

    server.remote_og_movement_input_selection = "nearest_tick_target"
    fwd, strafe, source = server._select_delayed_movement_input(
        ctx,
        current_fwd=0.5188,
        current_strafe=0.0,
        delay_s=server.remote_og_movement_input_delay,
    )
    selection = getattr(ctx, "debug_last_movement_input_selection", {})
    assert source == "delayed_remote_og_action_history", source
    assert abs(fwd - 0.5188) < 1e-6, selection
    assert strafe == 0.0, selection
    assert selection["selection_policy"] == "nearest_tick_target", selection
    assert selection["selected_client_tick"] == 99950, selection
    assert selection["selected_tick_target_error_ms"] == 50, selection
    assert selection["selected_tick_abs_target_error_ms"] == 50, selection

    print("test_remote_og_movement_input_can_select_tick_domain_candidates: PASSED")
    return True


def test_server_jump_jets_apply_fixed_step_rising_edge():
    """Opt-in jump jets should fire once on thrust rising edge inside motion step."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 30.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.turn_deadzone = 0.05
    server.turn_sign = -1.0
    server.up_axis = "z"
    server.terrain = None
    server.terrain_pitch_enabled = False
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.jump_jets_enabled = True
    server.weapon_energy_enabled = True
    server.player_energy_max = 100.0
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
    ctx.injected_input = (0.0, 0.0)
    ctx.injected_jumpjet = 1.0
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (0.0, 0.0, 10.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_fuel = 33000.0
    ctx.player_energy = 100.0
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}
    ctx.jump_spawn_lockout = 0.0
    ctx.jump_cooldown_remaining = 0.0
    ctx.jump_prev_thrust_input = 0.0

    server._update_player_position(ctx, dt_override=1.0 / 30.0)

    tank_jump = JUMP_JET_CONFIGS[EntityType.TANK]
    assert ctx.debug_last_controller_step["jump_jet_fired"] is True
    assert abs(ctx.player_vel[2] - tank_jump.impulse) < 1e-4, ctx.player_vel
    assert abs(ctx.player_energy - (100.0 - tank_jump.fuel_cost)) < 1e-4, ctx.player_energy

    server._update_player_position(ctx, dt_override=1.0 / 30.0)

    assert ctx.debug_last_controller_step["jump_jet_fired"] is False
    assert abs(ctx.player_vel[2] - tank_jump.impulse) < 1e-4, ctx.player_vel
    print("test_server_jump_jets_apply_fixed_step_rising_edge: PASSED")
    return True


def test_server_jump_jets_have_visible_peak_under_default_gravity():
    """Opt-in tank jumpjets should produce a visible altitude change, not a tiny hop."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 30.0
    server.linear_damp_driving = 0.8
    server.linear_damp_coasting = 2.0
    server.up_axis = "z"
    server.terrain = None
    server.terrain_pitch_enabled = False
    server.gravity = -50.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.jump_jets_enabled = True
    server.weapon_energy_enabled = True
    server.player_energy_max = 100.0
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
    ctx.injected_input = (0.0, 0.0)
    ctx.injected_jumpjet = 1.0
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (0.0, 0.0, 0.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_fuel = 33000.0
    ctx.player_energy = 100.0
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}
    ctx.jump_spawn_lockout = 0.0
    ctx.jump_cooldown_remaining = 0.0
    ctx.jump_prev_thrust_input = 0.0

    peak_z = 0.0
    for step in range(45):
        if step == 1:
            ctx.injected_jumpjet = None
        server._update_player_position(ctx, dt_override=1.0 / 30.0)
        peak_z = max(peak_z, ctx.player_pos[2])

    assert peak_z >= 8.0, peak_z
    print("test_server_jump_jets_have_visible_peak_under_default_gravity: PASSED")
    return True


def test_server_jump_jets_use_tank_body_up_direction():
    """Tank jumpjets should follow body pitch/roll instead of hardcoded world Z."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 30.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.turn_deadzone = 0.05
    server.turn_sign = -1.0
    server.up_axis = "z"
    server.terrain = None
    server.terrain_pitch_enabled = False
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.jump_jets_enabled = True
    server.jump_jet_direction = "body"
    server.weapon_energy_enabled = True
    server.player_energy_max = 100.0
    server._resolve_entity_world_collision = (
        lambda ctx, px, py, pz, vx, vy, vz: (px, py, pz, vx, vy, vz)
    )
    server._check_building_collisions = (
        lambda ctx, px, py, pz, vx, vy: (px, py, vx, vy)
    )

    pitch = math.radians(30.0)
    body_matrix = _matrix3_from_euler_xyz(0.0, pitch, 0.0)
    expected_direction = (body_matrix[2], body_matrix[5], body_matrix[8])
    tank_jump = JUMP_JET_CONFIGS[EntityType.TANK]

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.injected_input = (0.0, 0.0)
    ctx.injected_jumpjet = 1.0
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (0.0, 0.0, 10.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_fuel = 33000.0
    ctx.player_energy = 100.0
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {"roll": 0.0, "pitch": pitch, "yaw": 0.0}
    ctx.spring_body_matrix = body_matrix
    ctx.jump_spawn_lockout = 0.0
    ctx.jump_cooldown_remaining = 0.0
    ctx.jump_prev_thrust_input = 0.0

    server._update_player_position(ctx, dt_override=1.0 / 30.0)

    assert ctx.debug_last_controller_step["jump_jet_fired"] is True
    assert ctx.debug_last_controller_step["jump_jet_direction_mode"] == "body"
    assert abs(ctx.player_vel[0] - tank_jump.impulse * expected_direction[0]) < 1e-4, ctx.player_vel
    assert abs(ctx.player_vel[1] - tank_jump.impulse * expected_direction[1]) < 1e-4, ctx.player_vel
    assert abs(ctx.player_vel[2] - tank_jump.impulse * expected_direction[2]) < 1e-4, ctx.player_vel
    print("test_server_jump_jets_use_tank_body_up_direction: PASSED")
    return True


def test_server_jump_jet_landing_guard_rejects_large_world_collision_projection():
    """Jumpjet landing should not accept terrain-model side shoves as authority."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 30.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.turn_deadzone = 0.05
    server.turn_sign = -1.0
    server.up_axis = "z"
    server.terrain = None
    server.terrain_pitch_enabled = False
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.jump_jets_enabled = True
    server.jump_jet_direction = "body"
    server.jump_jet_collision_guard = True
    server.jump_jet_collision_guard_xy = 1.0
    server.jump_jet_collision_guard_zpop = 2.0
    server.jump_jet_landing_clearance = 1.85
    server.tank_suspension_enabled = False
    server.weapon_energy_enabled = True
    server.player_energy_max = 100.0
    server._resolve_entity_world_collision = (
        lambda ctx, px, py, pz, vx, vy, vz: (px - 4.0, py - 9.0, pz + 8.0, vx, vy, vz)
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
    ctx.injected_jumpjet = 0.0
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (10.0, 20.0, 0.5)
    ctx.player_vel = (0.0, 0.0, -40.0)
    ctx.player_fuel = 33000.0
    ctx.player_energy = 100.0
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}
    ctx.jump_spawn_lockout = 0.0
    ctx.jump_cooldown_remaining = 2.0
    ctx.jump_prev_thrust_input = 0.0

    server._update_player_position(ctx, dt_override=1.0 / 30.0)

    guard = ctx.debug_last_controller_step["jump_jet_collision_guard"]
    assert guard["applied"] is True, guard
    assert guard["collision_xy"] > 9.0, guard
    assert guard["landing_floor_applied"] is True, guard
    assert abs(ctx.player_pos[0] - 10.0) < 1e-6, ctx.player_pos
    assert abs(ctx.player_pos[1] - 20.0) < 1e-6, ctx.player_pos
    assert abs(ctx.player_pos[2] - 1.85) < 1e-6, ctx.player_pos
    assert ctx.player_vel == (0.0, 0.0, 0.0), ctx.player_vel
    print("test_server_jump_jet_landing_guard_rejects_large_world_collision_projection: PASSED")
    return True


def test_server_tank_terrain_projection_guard_rejects_fast_straight_side_shove():
    """Fast tank travel should not accept large terrain-model side projections."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 30.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.turn_deadzone = 0.05
    server.turn_sign = -1.0
    server.up_axis = "z"
    server.terrain = None
    server.terrain_pitch_enabled = False
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.jump_jets_enabled = True
    server.jump_jet_direction = "body"
    server.tank_terrain_projection_guard = True
    server.tank_terrain_projection_guard_xy = 1.0
    server.tank_terrain_projection_guard_zpop = 2.0
    server.tank_terrain_projection_guard_min_clearance = 0.5
    server.tank_suspension_enabled = False
    server.weapon_energy_enabled = True
    server.player_energy_max = 100.0
    server._resolve_entity_world_collision = (
        lambda ctx, px, py, pz, vx, vy, vz: (px, py - 8.75, pz, vx, vy, vz)
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
    ctx.injected_jumpjet = 0.0
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (10.0, 20.0, 3.25)
    ctx.player_vel = (25.0, 0.0, 0.0)
    ctx.player_fuel = 33000.0
    ctx.player_energy = 100.0
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}
    ctx.jump_spawn_lockout = 0.0
    ctx.jump_cooldown_remaining = 0.0
    ctx.jump_prev_thrust_input = 0.0

    server._update_player_position(ctx, dt_override=1.0 / 30.0)

    guard = ctx.debug_last_controller_step["tank_terrain_projection_guard"]
    assert guard["applied"] is True, guard
    assert guard["collision_xy"] > 8.0, guard
    assert guard["projection_clearance"] >= 3.0, guard
    assert abs(ctx.player_pos[0] - guard["pre_world_collision_pos"][0]) < 1e-6, ctx.player_pos
    assert abs(ctx.player_pos[1] - guard["pre_world_collision_pos"][1]) < 1e-6, ctx.player_pos
    assert abs(ctx.player_pos[2] - guard["pre_world_collision_pos"][2]) < 1e-6, ctx.player_pos
    assert ctx.player_vel == tuple(guard["pre_world_collision_vel"]), ctx.player_vel
    print("test_server_tank_terrain_projection_guard_rejects_fast_straight_side_shove: PASSED")
    return True


def test_server_jump_jets_queue_remote_og_correction_burst():
    """Remote OG clients need correction bursts because OG has no local jump impulse."""
    server = WulframServer.__new__(WulframServer)
    server.jump_jet_correction_burst_count = 12
    server.jump_jet_correction_burst_interval = 0.05

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )

    tank_jump = JUMP_JET_CONFIGS[EntityType.TANK]
    server._on_jump_jet_triggered(ctx, player_id=ctx.entity_id, impulse=tank_jump.impulse, new_vel_z=tank_jump.impulse)

    assert ctx.force_correction_once is True
    assert ctx.correction_burst_remaining == 11
    assert abs(ctx.correction_burst_interval_s - 0.05) < 1e-6
    assert ctx.last_correction_send == 0.0

    # Loopback fork retired (9ea5dbd, 2026-06-02): a 127.0.0.1 client queues
    # the same correction burst as a remote OG client.
    loopback_ctx = ClientContext(
        client_id=2,
        client_addr=("127.0.0.1", 50001),
        session=Session(),
        entity_id=0x14EB,
    )
    server._on_jump_jet_triggered(
        loopback_ctx,
        player_id=loopback_ctx.entity_id,
        impulse=tank_jump.impulse,
        new_vel_z=tank_jump.impulse,
    )

    assert loopback_ctx.force_correction_once is True
    assert loopback_ctx.correction_burst_remaining == 11
    print("test_server_jump_jets_queue_remote_og_correction_burst: PASSED")
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
    server.terrain_physics_height_offset = 5.0
    server.terrain_pitch_enabled = False
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
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


def test_server_motion_uses_physics_terrain_offset_for_vehicle_ground():
    """Vehicle collision should use physics terrain offset, not map-entity Z offset."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 30.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.up_axis = "z"
    server.terrain = SimpleNamespace(
        get_height=lambda x, y: 10.0,
        get_slope=lambda x, y: (0.0, 0.0),
    )
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 0.0
    server.terrain_pitch_enabled = False
    server.tank_suspension_enabled = False
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
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
    ctx.injected_input = (0.0, 0.0)
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (0.0, 0.0, 9.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}

    server._update_player_position(ctx, dt_override=1.0 / 30.0)

    assert ctx.player_pos[2] == 10.0, ctx.player_pos
    assert ctx.debug_last_controller_step["ground_level"] == 10.0
    assert ctx.debug_last_controller_step["terrain_ground_level"] == 10.0
    print("test_server_motion_uses_physics_terrain_offset_for_vehicle_ground: PASSED")
    return True


def test_server_motion_releases_spawn_ground_override_on_terrain_departure():
    """Spawn-pad ground pins should not become a permanent flat plane on sloped terrain."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 45.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.up_axis = "z"
    server.terrain = SimpleNamespace(
        get_height=lambda x, y: 42.0,
        get_slope=lambda x, y: (0.0, 0.0),
    )
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 5.0
    server.terrain_pitch_enabled = False
    server.gravity = -50.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.ground_override_release_distance = 24.0
    server.ground_override_release_height = 4.0
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
    ctx.injected_input = (0.0, 0.0)
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (130.0, 0.0, 63.7)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_heading = 0.0
    ctx.ground_level_override = 63.7
    ctx.world_collision_ref_pos = (0.0, 0.0, 63.7)
    ctx.player_pose = {}

    server._update_player_position(ctx, dt_override=1.0)

    assert ctx.ground_level_override is None
    assert ctx.debug_last_controller_step["ground_level_source"] == "terrain"
    assert ctx.debug_last_controller_step["ground_override_released"] is True
    assert ctx.player_pos[2] == 47.0, ctx.player_pos
    assert ctx.player_vel[2] == 0.0, ctx.player_vel
    print("test_server_motion_releases_spawn_ground_override_on_terrain_departure: PASSED")
    return True


def test_server_motion_releases_ground_override_when_terrain_changes_under_tank():
    """Exact-pose ground pins should release once the tank drives onto changing terrain."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 45.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.up_axis = "z"
    server.terrain = SimpleNamespace(
        get_height=lambda x, y: 42.0 + x * 0.25,
        get_slope=lambda x, y: (0.25, 0.0),
    )
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 5.0
    server.terrain_pitch_enabled = False
    server.gravity = -50.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.ground_override_release_distance = 24.0
    server.ground_override_release_height = 4.0
    server.ground_override_release_terrain_distance = 4.0
    server.ground_override_release_terrain_height = 0.75
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
    ctx.injected_input = (0.0, 0.0)
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (6.0, 0.0, 47.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_heading = 0.0
    ctx.ground_level_override = 47.0
    ctx.ground_override_ref_terrain_level = 47.0
    ctx.world_collision_ref_pos = (0.0, 0.0, 47.0)
    ctx.player_pose = {}

    server._update_player_position(ctx, dt_override=0.1)

    assert ctx.ground_level_override is None
    assert ctx.debug_last_controller_step["ground_override_released"] is True
    assert ctx.debug_last_controller_step["ground_override_release_reason"] == "terrain_change"
    assert ctx.debug_last_controller_step["ground_level_source"] == "terrain"
    assert abs(ctx.player_pos[2] - 48.5) < 1e-6, ctx.player_pos
    print("test_server_motion_releases_ground_override_when_terrain_changes_under_tank: PASSED")
    return True


def test_remote_team_select_uses_remote_idle_timeout():
    """Remote OG clients can sit on entry-map UI while the harness probes spawn clicks."""
    server = WulframServer.__new__(WulframServer)
    server.inactivity_timeout = 120.0
    server.remote_idle_timeout = 900.0

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.session.phase = Phase.TEAM_SELECT

    assert server._effective_inactivity_timeout(ctx) == 900.0

    ctx.client_addr = ("127.0.0.1", 50000)
    assert server._effective_inactivity_timeout(ctx) == 120.0
    print("test_remote_team_select_uses_remote_idle_timeout: PASSED")
    return True


def test_tank_surface_state_uses_spring_base_clearance_target():
    """Tank altitude ratio should use the decompile spring height denominator."""
    server = WulframServer.__new__(WulframServer)
    server.terrain = SimpleNamespace(
        get_height=lambda x, y: 10.0,
        get_slope=lambda x, y: (0.0, 0.0),
    )
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 0.0
    server.tank_spring_base_offset = 2.0

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.entity_type = EntityType.TANK
    target_clearance = 2.0 + VEHICLE_PHYSICS_CONFIGS[EntityType.TANK].max_altitude
    ctx.player_pos = (0.0, 0.0, 10.0 + target_clearance)
    ctx.player_heading = 0.0

    _up, clearance_ratio = server._sample_tank_surface_state(ctx)

    # Spring_update_world_state stores height_sum / (point_count - 1), so a
    # 4-point spring at nominal per-point clearance reports 4/3.
    assert abs(clearance_ratio - (4.0 / 3.0)) < 1e-6, clearance_ratio
    print("test_tank_surface_state_uses_spring_base_clearance_target: PASSED")
    return True


def test_tank_surface_state_uses_behavior_spring_offsets():
    """Terrain samples should follow BEHAVIOR Section-5 local spring points."""
    server = WulframServer.__new__(WulframServer)
    sampled = []

    def sample_height_normal(x, y):
        sampled.append((x, y))
        return 10.0, (0.0, 0.0, 1.0)

    server.terrain = SimpleNamespace(sample_height_normal=sample_height_normal)
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 0.0
    server.tank_spring_base_offset = 2.0
    server.tank_spring_sample_local_offsets = (
        (5.0, 1.0),
        (5.0, -1.0),
        (-5.0, 1.0),
        (-5.0, -1.0),
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (100.0, 200.0, 15.25)
    ctx.player_heading = 0.0

    server._sample_tank_surface_state(ctx)

    expected = [(105.0, 201.0), (105.0, 199.0), (95.0, 201.0), (95.0, 199.0)]
    assert sampled == expected, sampled
    spring_debug = ctx.debug_last_spring_state
    assert spring_debug["source"] == "Spring_update_world_state"
    assert spring_debug["point_count"] == 4
    assert spring_debug["clearance_denominator"] == 3
    assert spring_debug["height_sum"] == 21.0
    assert spring_debug["average_clearance"] == 7.0
    assert spring_debug["samples"][0]["local_offset"] == [5.0, 1.0]
    assert spring_debug["samples"][0]["sample_xy"] == [105.0, 201.0]
    print("test_tank_surface_state_uses_behavior_spring_offsets: PASSED")
    return True


def test_tank_surface_state_rotates_spring_points_through_body_pose():
    """Spring_update_world_state rotates local points before terrain sampling."""
    server = WulframServer.__new__(WulframServer)
    sampled = []

    def sample_height_normal(x, y):
        sampled.append((x, y))
        return 10.0, (0.0, 0.0, 1.0)

    server.terrain = SimpleNamespace(sample_height_normal=sample_height_normal)
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 0.0
    server.tank_spring_base_offset = 2.0
    server.terrain_pitch_enabled = True
    server.tank_spring_sample_local_offsets = (
        (5.0, 1.0),
        (5.0, -1.0),
        (-5.0, 1.0),
        (-5.0, -1.0),
    )

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (100.0, 200.0, 15.25)
    ctx.player_heading = 0.0
    ctx.player_pose["roll"] = math.radians(8.0)
    ctx.player_pose["pitch"] = 0.0

    server._sample_tank_surface_state(ctx)

    cos_roll = math.cos(math.radians(8.0))
    sin_roll = math.sin(math.radians(8.0))
    assert abs(sampled[0][0] - 105.0) < 1e-6, sampled
    assert abs(sampled[0][1] - (200.0 + cos_roll)) < 1e-6, sampled
    spring_debug = ctx.debug_last_spring_state
    assert spring_debug["rotation_source"] == "body_matrix"
    assert abs(spring_debug["samples"][0]["world_offset_z"] - sin_roll) < 1e-5
    assert abs(spring_debug["samples"][0]["clearance"] - (5.25 + sin_roll)) < 1e-5
    assert abs(spring_debug["samples"][1]["world_offset_z"] + sin_roll) < 1e-5
    print("test_tank_surface_state_rotates_spring_points_through_body_pose: PASSED")
    return True


def test_server_tank_drive_uses_body_matrix_when_body_pose_live():
    """Tank drive should rotate through the live body pose before integration."""

    def sample_height_normal(_x, _y):
        nx, ny, nz = -0.25, -0.1, 1.0
        mag = math.sqrt(nx * nx + ny * ny + nz * nz)
        return 0.0, (nx / mag, ny / mag, nz / mag)

    def make_server(body_matrix_enabled: bool):
        server = WulframServer.__new__(WulframServer)
        server.tick_rate_hz = 30.0
        server.linear_damp_driving = 0.0
        server.linear_damp_coasting = 0.0
        server.turn_deadzone = 0.05
        server.turn_sign = -1.0
        server.up_axis = "z"
        server.terrain = SimpleNamespace(
            get_height=lambda _x, _y: 0.0,
            get_slope=lambda _x, _y: (0.25, 0.1),
            sample_height_normal=sample_height_normal,
        )
        server.terrain_height_offset = 0.0
        server.terrain_physics_height_offset = 0.0
        server.terrain_pitch_enabled = True
        server.tank_drive_terrain_aligned = False
        server.tank_drive_body_matrix = body_matrix_enabled
        server.tank_terrain_contact_coupling_enabled = False
        server.tank_suspension_enabled = False
        server.tank_spring_base_offset = 2.0
        server.tank_spring_attitude_model = "target"
        server.gravity = 0.0
        server.ground_level = 0.0
        server.world_bound = 100000.0
        server.jump_jets_enabled = False
        server._resolve_entity_world_collision = (
            lambda ctx, px, py, pz, vx, vy, vz: (px, py, pz, vx, vy, vz)
        )
        server._check_building_collisions = (
            lambda ctx, px, py, pz, vx, vy: (px, py, vx, vy)
        )
        return server

    def make_ctx():
        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(),
            entity_id=0x14EA,
        )
        ctx.injected_input = (1.0, 0.0)
        ctx.entity_type = EntityType.TANK
        ctx.player_pos = (100.0, 200.0, 20.0)
        ctx.player_vel = (0.0, 0.0, 0.0)
        ctx.player_speed = 0.0
        ctx.player_fuel = 33000.0
        ctx.player_energy = 100.0
        ctx.player_heading = 0.0
        ctx.ground_level_override = None
        ctx.player_pose = {
            "roll": math.radians(4.0),
            "pitch": math.radians(-6.0),
            "yaw": 0.0,
            "pos": ctx.player_pos,
            "vel": ctx.player_vel,
        }
        ctx.spring_body_ang_vel = (0.0, 0.0)
        return ctx

    server = make_server(True)
    ctx = make_ctx()
    server._update_player_position(ctx, dt_override=1.0 / 30.0)
    debug = ctx.debug_last_controller_step
    assert debug["drive_basis_source"] == "entity_body_matrix", debug
    assert abs(debug["basis_forward"][2]) > 0.01, debug
    assert abs(debug["move_impulse"][2]) > 0.01, debug
    assert debug["tank_vehicle_impulse"] == debug["move_impulse"], debug
    assert debug["gravity_impulse"] == (0.0, 0.0, 0.0), debug
    assert debug["suspension_impulse"] == (0.0, 0.0, 0.0), debug

    stale_server = make_server(True)
    stale_ctx = make_ctx()
    stale_ctx.player_heading = math.pi
    stale_ctx.player_yaw = -math.pi
    stale_ctx.player_pose["yaw"] = -math.pi
    stale_ctx.spring_body_matrix = _matrix3_from_euler_xyz(
        stale_ctx.player_pose["roll"],
        stale_ctx.player_pose["pitch"],
        0.0,
    )
    stale_server._update_player_position(stale_ctx, dt_override=1.0 / 30.0)
    stale_debug = stale_ctx.debug_last_controller_step
    assert stale_debug["drive_basis_source"] == "entity_body_matrix", stale_debug
    assert stale_debug["basis_forward"][0] < -0.99, stale_debug
    assert stale_debug["drive_impulse_capped"][0] < 0.0, stale_debug

    flat_server = make_server(False)
    flat_ctx = make_ctx()
    flat_server._update_player_position(flat_ctx, dt_override=1.0 / 30.0)
    flat_debug = flat_ctx.debug_last_controller_step
    assert flat_debug["drive_basis_source"] == "entity_yaw_flat", flat_debug
    assert abs(flat_debug["basis_forward"][2]) < 1e-6, flat_debug
    assert abs(flat_debug["move_impulse"][2]) < 1e-6, flat_debug
    assert flat_debug["tank_vehicle_impulse"] == flat_debug["move_impulse"], flat_debug
    print("test_server_tank_drive_uses_body_matrix_when_body_pose_live: PASSED")
    return True


def test_contact_yaw_velocity_feeds_next_vehicle_physics_tick():
    """Terrain contact yaw velocity should feed the next steering integration step."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 30.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.turn_deadzone = 0.05
    server.turn_sign = -1.0
    server.strafe_sign = -1.0
    server.up_axis = "z"
    server.terrain = None
    server.terrain_pitch_enabled = False
    server.tank_drive_terrain_aligned = False
    server.tank_drive_body_matrix = True
    server.tank_terrain_contact_coupling_enabled = False
    server.tank_suspension_enabled = False
    server.tank_spring_base_offset = 2.0
    server.tank_spring_attitude_model = "force"
    server.tank_spring_attitude_damping = 2.0
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.jump_jets_enabled = False

    def fake_contact(ctx, px, py, pz, vx, vy, vz):
        ctx.angular_vel_yaw = 0.75
        return px, py, pz, vx, vy, vz

    server._resolve_entity_world_collision = fake_contact
    server._check_building_collisions = lambda ctx, px, py, pz, vx, vy: (px, py, vx, vy)

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.injected_input = (0.0, 0.0)
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (100.0, 200.0, 20.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_speed = 0.0
    ctx.player_fuel = 33000.0
    ctx.player_energy = 100.0
    ctx.player_heading = 0.25
    ctx.ground_level_override = None
    ctx.vehicle_physics = SimpleNamespace(heading=ctx.player_heading, angular_velocity=0.0)

    server._update_player_position(ctx, dt_override=1.0 / 30.0)

    assert abs(ctx.vehicle_physics.heading - ctx.player_heading) < 1e-9
    assert abs(ctx.vehicle_physics.angular_velocity - 0.75) < 1e-9
    print("test_contact_yaw_velocity_feeds_next_vehicle_physics_tick: PASSED")
    return True


def test_surface_attitude_uses_post_yaw_heading_after_drive_step():
    """Drive impulse uses old heading, but spring/body attitude should use post-yaw heading."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 30.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.turn_deadzone = 0.05
    server.turn_sign = -1.0
    server.strafe_sign = -1.0
    server.up_axis = "z"
    server.terrain = None
    server.terrain_pitch_enabled = False
    server.tank_drive_terrain_aligned = False
    server.tank_drive_body_matrix = True
    server.tank_terrain_contact_coupling_enabled = False
    server.tank_suspension_enabled = False
    server.tank_spring_base_offset = 2.0
    server.tank_spring_attitude_model = "force"
    server.tank_spring_attitude_damping = 2.0
    server.gravity = 0.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
    server.jump_jets_enabled = False
    server._resolve_entity_world_collision = (
        lambda ctx, px, py, pz, vx, vy, vz: (px, py, pz, vx, vy, vz)
    )
    server._check_building_collisions = lambda ctx, px, py, pz, vx, vy: (px, py, vx, vy)

    headings_seen = []

    def fake_surface_attitude(ctx, heading=None, **kwargs):
        headings_seen.append(heading)
        return {
            "source": "test",
            "rotation": (0.0, 0.0, heading),
            "up": (0.0, 0.0, 1.0),
            "matrix": None,
            "target_rotation": (0.0, 0.0, heading),
            "angular_velocity": (0.0, 0.0),
            "spring_attitude": {},
        }

    server._update_player_surface_attitude = fake_surface_attitude

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.injected_input = (0.0, 0.0)
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (100.0, 200.0, 20.0)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_speed = 0.0
    ctx.player_fuel = 33000.0
    ctx.player_energy = 100.0
    old_drive_heading = 0.25
    ctx.player_heading = 0.5
    ctx.player_yaw = -ctx.player_heading
    ctx.angular_vel_yaw = 0.0
    ctx.ground_level_override = None

    server._update_player_position(
        ctx,
        dt_override=1.0 / 30.0,
        heading_override=old_drive_heading,
    )

    assert headings_seen == [ctx.player_heading], headings_seen
    print("test_surface_attitude_uses_post_yaw_heading_after_drive_step: PASSED")
    return True


def test_tank_surface_attitude_uses_spring_normal_for_replication():
    """Replicated tank roll/pitch should follow the spring terrain normal."""
    server = WulframServer.__new__(WulframServer)
    slope = (0.3, 0.1)

    def sample_height_normal(_x, _y):
        nx, ny, nz = -slope[0], -slope[1], 1.0
        mag = math.sqrt(nx * nx + ny * ny + nz * nz)
        return 10.0, (nx / mag, ny / mag, nz / mag)

    server.terrain = SimpleNamespace(
        sample_height_normal=sample_height_normal,
        get_slope=lambda _x, _y: slope,
    )
    server.up_axis = "z"
    server.terrain_pitch_enabled = True
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 0.0
    server.tank_spring_base_offset = 2.0
    server.tank_spring_sample_local_offsets = (
        (5.0, 1.0),
        (5.0, -1.0),
        (-5.0, 1.0),
        (-5.0, -1.0),
    )
    # 7bca231 (CH2 attitude debug command) stashes this attribute into
    # ctx.debug_attitude_target unconditionally; __init__ defaults it to "force".
    server.tank_spring_attitude_model = "force"

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (100.0, 200.0, 15.25)
    ctx.player_heading = 0.0

    attitude = server._update_player_surface_attitude(ctx)

    assert attitude["source"] == "terrain_surface"
    assert attitude["up"][2] < 1.0
    assert abs(ctx.player_pose["roll"]) > 0.01 or abs(ctx.player_pose["pitch"]) > 0.01
    assert abs(ctx.player_pose["yaw"] + ctx.player_heading) < 1e-6
    print("test_tank_surface_attitude_uses_spring_normal_for_replication: PASSED")
    return True


def test_tank_surface_attitude_steps_toward_spring_normal():
    """Live ticks should advance spring body pose by torque, not snap it."""
    server = WulframServer.__new__(WulframServer)
    slope = (0.3, 0.0)

    def sample_height_normal(_x, _y):
        nx, ny, nz = -slope[0], -slope[1], 1.0
        mag = math.sqrt(nx * nx + ny * ny + nz * nz)
        return 10.0, (nx / mag, ny / mag, nz / mag)

    server.terrain = SimpleNamespace(
        sample_height_normal=sample_height_normal,
        get_slope=lambda _x, _y: slope,
    )
    server.up_axis = "z"
    server.terrain_pitch_enabled = True
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 0.0
    server.tank_spring_base_offset = 2.0
    server.tank_spring_sample_local_offsets = (
        (5.0, 1.0),
        (5.0, -1.0),
        (-5.0, 1.0),
        (-5.0, -1.0),
    )
    server.tank_spring_attitude_stiffness = 40.0
    server.tank_spring_attitude_damping = 2.0
    # 7bca231 (CH2 attitude debug command) stashes this attribute into
    # ctx.debug_attitude_target unconditionally; __init__ defaults it to "force".
    server.tank_spring_attitude_model = "force"

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (100.0, 200.0, 15.25)
    ctx.player_heading = 0.0

    snap = server._update_player_surface_attitude(ctx, snap=True)
    ctx.player_pose["roll"] = 0.0
    ctx.player_pose["pitch"] = 0.0
    ctx.spring_body_ang_vel = (0.0, 0.0)

    stepped = server._update_player_surface_attitude(ctx, dt=1.0 / 30.0)
    initial_error = abs(stepped["spring_attitude"]["error"][1])
    stepped_delta = abs((ctx.player_pose["pitch"] - 0.0 + math.pi) % (2.0 * math.pi) - math.pi)

    assert stepped["source"] == "terrain_surface"
    assert stepped_delta > 0.0
    assert stepped_delta < initial_error
    assert abs(ctx.spring_body_ang_vel[1]) > 0.0
    assert stepped["spring_attitude"]["target"] == snap["target_rotation"]
    print("test_tank_surface_attitude_steps_toward_spring_normal: PASSED")
    return True


def test_tank_surface_attitude_force_path_uses_point_clearance_torque():
    """Live spring attitude can use per-point force torque instead of target snap."""
    server = WulframServer.__new__(WulframServer)

    def sample_height_normal(x, _y):
        raw_height = 10.0 + 0.2 * (x - 100.0)
        nx, ny, nz = -0.2, 0.0, 1.0
        mag = math.sqrt(nx * nx + ny * ny + nz * nz)
        return raw_height, (nx / mag, ny / mag, nz / mag)

    server.terrain = SimpleNamespace(
        sample_height_normal=sample_height_normal,
        get_slope=lambda _x, _y: (0.2, 0.0),
    )
    server.up_axis = "z"
    server.terrain_pitch_enabled = True
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 0.0
    server.tank_spring_base_offset = 2.0
    server.tank_spring_sample_local_offsets = (
        (5.0, 1.0),
        (5.0, -1.0),
        (-5.0, 1.0),
        (-5.0, -1.0),
    )
    server.tank_spring_attitude_model = "force"
    server.tank_spring_attitude_damping = 2.0

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (100.0, 200.0, 15.25)
    ctx.player_heading = 0.0

    attitude = server._update_player_surface_attitude(
        ctx,
        dt=1.0 / 30.0,
        suspension_lift=50.0,
    )
    spring = attitude["spring_attitude"]

    assert spring["model"] == "force", spring
    assert spring["integration_model"] == "decompile_accel", spring
    assert abs(ctx.spring_body_ang_vel[1]) > 0.0, ctx.spring_body_ang_vel
    assert spring["point_forces"][0] > spring["point_forces"][2], spring["point_forces"]
    assert abs(spring["local_torque"][1]) > 0.0, spring["local_torque"]
    assert abs(spring["spring_angular_delta"][1] - spring["local_torque"][1] / 30.0) < 1e-9, spring
    print("test_tank_surface_attitude_force_path_uses_point_clearance_torque: PASSED")
    return True


def test_tank_surface_attitude_reuses_force_sample_state_without_resampling():
    """Spring attitude should use the same Spring_update_world_state rows as force."""
    server = WulframServer.__new__(WulframServer)
    sampled = []

    def sample_height_normal(x, y):
        sampled.append((x, y))
        nx, ny, nz = -0.2, 0.05, 1.0
        mag = math.sqrt(nx * nx + ny * ny + nz * nz)
        return 10.0 + 0.1 * (x - 100.0), (nx / mag, ny / mag, nz / mag)

    server.terrain = SimpleNamespace(
        sample_height_normal=sample_height_normal,
        get_slope=lambda _x, _y: (0.2, -0.05),
    )
    server.up_axis = "z"
    server.terrain_pitch_enabled = True
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 0.0
    server.tank_spring_base_offset = 2.0
    server.tank_spring_sample_local_offsets = (
        (5.0, 1.0),
        (5.0, -1.0),
        (-5.0, 1.0),
        (-5.0, -1.0),
    )
    server.tank_spring_attitude_model = "force"
    server.tank_spring_attitude_damping = 2.0

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (100.0, 200.0, 15.25)
    ctx.player_heading = 0.0

    server._sample_tank_surface_state(ctx)
    force_state = dict(ctx.debug_last_spring_state)
    sampled_count = len(sampled)
    ctx.player_pos = (125.0, 200.0, 15.25)

    attitude = server._update_player_surface_attitude(
        ctx,
        dt=1.0 / 30.0,
        suspension_lift=50.0,
        spring_state_override=force_state,
    )

    assert len(sampled) == sampled_count, sampled
    assert ctx.debug_last_spring_state == force_state
    assert len(force_state["body_matrix"]) == 9
    assert tuple(round(float(v), 8) for v in ctx.spring_body_matrix) == tuple(
        round(float(v), 8) for v in attitude["spring_attitude"]["rotation_matrix"]
    )
    assert attitude["spring_attitude"]["model"] == "force"
    assert attitude["spring_attitude"]["spring_state_source"] == "force_sample"
    print("test_tank_surface_attitude_reuses_force_sample_state_without_resampling: PASSED")
    return True


def test_heading_physics_sync_preserves_spring_body_pose():
    """Yaw-only VehiclePhysics must not flatten spring-derived tank pitch/roll."""
    server = WulframServer.__new__(WulframServer)
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_pose["roll"] = math.radians(7.0)
    ctx.player_pose["pitch"] = math.radians(-3.0)
    ctx.spring_body_matrix = _matrix3_from_euler_xyz(
        ctx.player_pose["roll"],
        ctx.player_pose["pitch"],
        0.0,
    )
    physics = SimpleNamespace(
        heading=math.radians(12.0),
        angular_velocity=0.25,
        rotation=(0.0, 0.0, math.radians(12.0)),
    )

    server._sync_heading_physics_to_context(ctx, physics)

    assert abs(ctx.player_heading - physics.heading) < 1e-9
    assert abs(ctx.angular_vel_yaw - physics.angular_velocity) < 1e-9
    assert abs(ctx.player_pose["roll"] - math.radians(7.0)) < 1e-9
    assert abs(ctx.player_pose["pitch"] - math.radians(-3.0)) < 1e-9
    assert abs(ctx.player_pose["yaw"] + physics.heading) < 1e-9
    assert abs(ctx.spring_body_matrix[0] - math.cos(physics.heading)) < 0.02
    assert abs(ctx.spring_body_matrix[3] - math.sin(physics.heading)) < 0.13
    print("test_heading_physics_sync_preserves_spring_body_pose: PASSED")
    return True


def test_tank_softbody_support_pulls_down_from_compact_equilibrium():
    """Default tank support should not hold the old compact 5.25-target Z."""
    server = WulframServer.__new__(WulframServer)
    server.tick_rate_hz = 30.0
    server.linear_damp_driving = 0.0
    server.linear_damp_coasting = 0.0
    server.turn_deadzone = 0.05
    server.turn_sign = -1.0
    server.up_axis = "z"
    server.terrain = SimpleNamespace(
        get_height=lambda x, y: 10.0,
        get_slope=lambda x, y: (0.0, 0.0),
    )
    server.terrain_height_offset = 5.0
    server.terrain_physics_height_offset = 0.0
    server.tank_spring_base_offset = 2.0
    server.terrain_pitch_enabled = False
    server.tank_suspension_enabled = True
    server.tank_suspension_model = "softbody"
    server.tank_suspension_stiffness = 40.0
    server.tank_suspension_damping = 1.5
    server.tank_suspension_lift_cap = 120.0
    server.gravity = -50.0
    server.ground_level = 0.0
    server.world_bound = 100000.0
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
    ctx.injected_input = (0.0, 0.0)
    ctx.weapon_system = WeaponSystem()
    ctx.weapon_system.behavior_slots[5] = OG_TANK_SOFTBODY_IDLE_SLOT5
    ctx.entity_type = EntityType.TANK
    ctx.player_pos = (0.0, 0.0, 13.9375)
    ctx.player_vel = (0.0, 0.0, 0.0)
    ctx.player_heading = 0.0
    ctx.ground_level_override = None
    ctx.player_pose = {}

    server._update_player_position(ctx, dt_override=1.0 / 30.0)

    assert ctx.player_vel[2] < 0.0, ctx.player_vel
    assert ctx.debug_last_controller_step["suspension_model"] == "softbody_per_point_piecewise_probe"
    assert ctx.debug_last_controller_step["softbody_point_count"] == 4
    assert "yaw_angular_velocity" in ctx.debug_last_controller_step
    assert abs(ctx.debug_last_controller_step["yaw_angular_velocity"] - ctx.angular_vel_yaw) < 1e-9
    assert (
        ctx.debug_last_controller_step["vehicle_physics_angular_velocity"]
        == ctx.debug_last_controller_step["yaw_angular_velocity"]
    )
    assert ctx.debug_last_controller_step["softbody_point_velocity_z"] == (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert (
        ctx.debug_last_controller_step["suspension_lift"]
        < ctx.debug_last_controller_step["softbody_support_accel"]
    )
    assert ctx.debug_last_controller_step["pre_ground_vertical_impulse"] < 0.0
    assert abs(
        ctx.debug_last_controller_step["suspension_target_clearance"]
        - OG_TANK_SOFTBODY_FLAT_AVERAGE_HEIGHT
    ) < 1e-6
    assert ctx.debug_last_controller_step["suspension_target_clearance"] != 5.25
    print("test_tank_softbody_support_pulls_down_from_compact_equilibrium: PASSED")
    return True


def test_tank_softbody_supports_gravity_at_og_flat_height():
    """The softbody stand-in should expose OG's spring force before gravity."""
    result = tank_softbody_suspension_force(
        OG_TANK_SOFTBODY_FLAT_AVERAGE_HEIGHT,
        0.0,
        OG_TANK_SOFTBODY_IDLE_SLOT5,
        gravity=-50.0,
    )

    assert result.model == "softbody_empirical_flat"
    assert abs(result.lift_accel - 100.0) < 1e-6, result
    assert abs(result.force_bias_accel) < 1e-6, result
    assert abs(result.height_ratio - 0.3324) < 0.001, result.height_ratio
    print("test_tank_softbody_supports_gravity_at_og_flat_height: PASSED")
    return True


def test_tank_softbody_slot5_changes_response_without_jumpjet():
    """Q/Z slot 5 should change the softbody force path, not jumpjet input."""
    low = tank_softbody_suspension_force(
        OG_TANK_SOFTBODY_FLAT_AVERAGE_HEIGHT,
        0.0,
        OG_TANK_SOFTBODY_Z_SLOT5,
    )
    idle = tank_softbody_suspension_force(
        OG_TANK_SOFTBODY_FLAT_AVERAGE_HEIGHT,
        0.0,
        OG_TANK_SOFTBODY_IDLE_SLOT5,
    )
    high = tank_softbody_suspension_force(
        OG_TANK_SOFTBODY_FLAT_AVERAGE_HEIGHT,
        0.0,
        OG_TANK_SOFTBODY_Q_SLOT5,
    )

    assert low.response_scale < idle.response_scale < high.response_scale
    assert low.softbody_stiffness < idle.softbody_stiffness < high.softbody_stiffness
    assert low.force_curve_input < idle.force_curve_input < high.force_curve_input
    assert low.force_bias_accel < idle.force_bias_accel < high.force_bias_accel
    assert low.target_average_height < idle.target_average_height < high.target_average_height
    assert high.target_average_height - low.target_average_height > 1.0
    assert low.lift_accel < idle.lift_accel < high.lift_accel
    assert high.lift_accel - idle.lift_accel < 40.0, high
    assert idle.lift_accel - low.lift_accel < 40.0, low
    print("test_tank_softbody_slot5_changes_response_without_jumpjet: PASSED")
    return True


def test_tank_softbody_control_prefers_live_og_slot5():
    """Live OG Q/Z telemetry uses behavior slot 5; slot 6 is only fallback lean data."""
    slots = [0.0] * 22
    slots[BehaviorSlot.UPWARD_THRUST] = OG_TANK_SOFTBODY_Q_SLOT5
    slots[BehaviorSlot.SLOT6] = -0.0916

    assert abs(tank_softbody_control_slot_value(slots) - OG_TANK_SOFTBODY_Q_SLOT5) < 1e-6
    slots[BehaviorSlot.UPWARD_THRUST] = 0.0
    slots[BehaviorSlot.SLOT6] = 0.1831
    assert abs(
        tank_softbody_control_slot_value(slots, allow_legacy_slot6=True) - 0.1831
    ) < 1e-6
    print("test_tank_softbody_control_prefers_live_og_slot5: PASSED")
    return True


def test_weapon_system_upward_thrust_defaults_to_og_idle_slot5():
    """Slot 5 is a positive OG softbody control with an idle baseline."""
    ws = WeaponSystem()
    assert abs(ws.behavior_slots[TANK_SOFTBODY_CONTROL_SLOT] - OG_TANK_SOFTBODY_IDLE_SLOT5) < 1e-6
    print("test_weapon_system_upward_thrust_defaults_to_og_idle_slot5: PASSED")
    return True


def test_tank_suspension_legacy_compact_stiffness_matches_old_equilibrium():
    """Legacy compact spring behavior remains available for blocker probes."""
    lift = tank_suspension_lift_accel(4.0, 5.25, 0.0)

    assert abs(lift - 50.0) < 1e-6, lift
    print("test_tank_suspension_legacy_compact_stiffness_matches_old_equilibrium: PASSED")
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
    assert contact.cbsp_split_normal == (1.0, 0.0, 0.0), contact
    assert contact.terrain_face_normal == (0.0, 0.0, -1.0), contact
    assert contact.mesh_face_normal == (1.0, 0.0, 0.0), contact
    assert contact.store_normal0 == (0.0, 0.0, -1.0), contact
    assert contact.store_normal1 == (1.0, 0.0, 0.0), contact
    assert contact.record_hit_source == "cbsp_leaf_clip_point", contact
    assert contact.mesh_triangle_indices == (0, 1, 2), contact
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


def test_triangle_cbsp_mesh_vertex_probe_reports_embedded_vertex():
    """Default-off CBSP mesh-vertex probe should return a mesh vertex, not a projected fallback point."""

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
        Vec3(0.25, 0.25, -0.001),
        Vec3(0.8, 0.25, 0.2),
        Vec3(0.25, 0.8, 0.2),
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

    contact = grid._triangle_cbsp_mesh_vertex_contact(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        vertices,
        tree,
    )
    assert contact is not None
    assert contact.position == (0.25, 0.25, -0.001), contact
    assert contact.record_hit_source == "cbsp_mesh_vertex_inside_terrain_probe", contact
    assert contact.mesh_triangle_indices == (0, 1, 2), contact
    assert abs(contact.penetration - 0.01) <= 1e-6, contact.penetration
    traversal_contact = grid._triangle_cbsp_mesh_vertex_contact(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        vertices,
        CBSPTree(
            nodes=[
                CBSPTreeNode(
                    radius=2.0,
                    half_extent_x=1.0,
                    half_extent_y=1.0,
                    half_extent_z=1.0,
                    center=Vec3(0.0, 0.0, -0.5),
                    triangles=[(0, 1, 2)],
                    split_normal=Vec3(1.0, 0.0, 0.0),
                    split_plane_d=0.0,
                    child_neg=-1,
                    child_pos=-1,
                )
            ],
            root_index=0,
        ),
        traversal_order=True,
    )
    assert traversal_contact is not None
    assert traversal_contact.position == (0.25, 0.25, -0.001), traversal_contact
    assert (
        traversal_contact.record_hit_source
        == "cbsp_mesh_vertex_inside_terrain_traversal_probe"
    ), traversal_contact
    print("test_triangle_cbsp_mesh_vertex_probe_reports_embedded_vertex: PASSED")
    return True


def test_triangle_cbsp_mesh_edge_terrain_plane_probe_reports_crossing_edge():
    """Default-off mesh-edge probe should report a mesh edge crossing the terrain plane."""

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
        Vec3(0.25, 0.25, -0.1),
        Vec3(0.75, 0.25, 0.1),
        Vec3(0.25, 0.75, 0.1),
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

    contact = grid._triangle_cbsp_mesh_edge_terrain_plane_contact(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        vertices,
        tree,
    )

    assert contact is not None
    assert contact.position == (0.5, 0.25, 0.0), contact
    assert contact.record_hit_source == "cbsp_mesh_edge_terrain_plane_probe_ab", contact
    assert contact.mesh_triangle_indices == (0, 1, 2), contact
    assert contact.store_normal0 == (0.0, 0.0, 1.0), contact
    assert contact.store_normal1 == (1.0, 0.0, 0.0), contact
    assert contact.cbsp_split_normal == (1.0, 0.0, 0.0), contact
    assert contact.guess7_order == (1, 0, 2), contact
    assert tuple(round(term, 6) for term in contact.guess7_terms) == (
        0.25,
        0.25,
        0.5,
    ), contact
    print("test_triangle_cbsp_mesh_edge_terrain_plane_probe_reports_crossing_edge: PASSED")
    return True


def test_triangle_cbsp_mesh_edge_terrain_plane_probe_reports_second_tilted_fixture():
    """A second static terrain fixture should use the same exact edge-intersect lane."""

    class TiltedTerrain:
        cell_x = 1.0
        cell_z = 1.0
        world_w = 2.0
        world_h = 2.0
        num_x = 2
        num_z = 2

        @staticmethod
        def _get_raw_height(cell_x, cell_y):
            return 0.0

    grid = TerrainGridCollision(TiltedTerrain(), 0.0)
    vertices = [
        Vec3(0.25, 0.25, -0.1),
        Vec3(0.75, 0.25, 0.3),
        Vec3(0.25, 0.75, 0.3),
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
                split_normal=Vec3(0.0, 1.0, 0.0),
                split_plane_d=0.0,
                child_neg=-1,
                child_pos=-1,
            )
        ],
        root_index=0,
    )

    contact = grid._triangle_cbsp_mesh_edge_terrain_plane_contact(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.2),
        ),
        vertices,
        tree,
    )

    assert contact is not None
    assert all(
        abs(actual - expected) <= 1e-6
        for actual, expected in zip(contact.position, (0.4375, 0.25, 0.05))
    ), contact
    assert contact.record_hit_source == "cbsp_mesh_edge_terrain_plane_probe_ab", contact
    assert contact.mesh_triangle_indices == (0, 1, 2), contact
    assert contact.store_normal1 == (0.0, 1.0, 0.0), contact
    assert contact.guess7_order == (1, 0, 2), contact
    assert all(term >= 0.0 for term in contact.guess7_terms), contact
    print("test_triangle_cbsp_mesh_edge_terrain_plane_probe_reports_second_tilted_fixture: PASSED")
    return True


def test_triangle_cbsp_mesh_edge_terrain_plane_probe_preserves_endpoint_hit():
    """The edge-plane lane should accept near/on-plane endpoints, not only sign crossings."""

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
        Vec3(0.25, 0.25, 0.1),
        Vec3(0.75, 0.25, 0.0),
        Vec3(0.25, 0.75, 0.2),
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
                split_normal=Vec3(0.0, 1.0, 0.0),
                split_plane_d=0.0,
                child_neg=-1,
                child_pos=-1,
            )
        ],
        root_index=0,
    )

    contact = grid._triangle_cbsp_mesh_edge_terrain_plane_contact(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        vertices,
        tree,
    )

    assert contact is not None
    assert contact.position == (0.75, 0.25, 0.0), contact
    assert contact.record_hit_source == "cbsp_mesh_edge_terrain_plane_probe_ab", contact
    assert contact.edge_hit_kind == "end_on_terrain_plane", contact
    assert contact.edge_t == 1.0, contact
    assert contact.mesh_triangle_indices == (0, 1, 2), contact
    assert contact.store_normal1 == (0.0, 1.0, 0.0), contact
    assert contact.guess7_order == (1, 0, 2), contact
    assert all(term >= 0.0 for term in contact.guess7_terms), contact
    print("test_triangle_cbsp_mesh_edge_terrain_plane_probe_preserves_endpoint_hit: PASSED")
    return True


def test_triangle_cbsp_mesh_edge_endpoint_probe_prefers_deeper_node():
    """Report-only endpoint probe should surface the deeper target-node endpoint row."""

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
        Vec3(0.25, 0.25, 0.0),
        Vec3(0.75, 0.25, 0.2),
        Vec3(0.25, 0.75, 0.2),
        Vec3(0.4, 0.4, 0.0),
        Vec3(0.2, 0.4, 0.2),
        Vec3(0.8, 0.4, 0.2),
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
                child_pos=1,
            ),
            CBSPTreeNode(
                radius=2.0,
                half_extent_x=1.0,
                half_extent_y=1.0,
                half_extent_z=1.0,
                center=Vec3(0.0, 0.0, 0.0),
                triangles=[(3, 4, 5)],
                split_normal=Vec3(0.0, 1.0, 0.0),
                split_plane_d=0.0,
                child_neg=-1,
                child_pos=-1,
            ),
        ],
        root_index=0,
    )

    contact = grid._triangle_cbsp_mesh_edge_terrain_plane_contact(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        vertices,
        tree,
        endpoint_only=True,
        prefer_deep_endpoint=True,
    )

    assert contact is not None
    assert contact.position == (0.4, 0.4, 0.0), contact
    assert contact.record_hit_source == "cbsp_mesh_edge_endpoint_terrain_plane_probe_ab", contact
    assert contact.mesh_triangle_indices == (3, 4, 5), contact
    assert contact.cbsp_split_normal == (0.0, 1.0, 0.0), contact
    assert contact.edge_hit_kind == "start_on_terrain_plane", contact
    assert contact.node_index == 1, contact
    assert contact.node_depth == 1, contact
    assert abs(contact.node_mesh_normal_angle_deg) <= 1e-6, contact
    assert contact.guess7_order == (1, 0, 2), contact
    print("test_triangle_cbsp_mesh_edge_endpoint_probe_prefers_deeper_node: PASSED")
    return True


def test_triangle_cbsp_node_plane_vertex_probe_uses_split_normal():
    """Default-off node-plane probe should use the CBSP node normal, not mesh face normal."""

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
        Vec3(0.25, 0.25, -0.02),
        Vec3(0.75, 0.25, 0.1),
        Vec3(0.25, 0.75, 0.1),
    ]
    tree = CBSPTree(
        nodes=[
            CBSPTreeNode(
                radius=2.0,
                half_extent_x=1.0,
                half_extent_y=1.0,
                half_extent_z=1.0,
                center=Vec3(0.0, 0.0, -0.5),
                triangles=[(0, 1, 2)],
                split_normal=Vec3(1.0, 0.0, 0.0),
                split_plane_d=-0.25,
                child_neg=-1,
                child_pos=-1,
            )
        ],
        root_index=0,
    )

    contact = grid._triangle_cbsp_node_plane_vertex_contact(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        vertices,
        tree,
    )

    assert contact is not None
    assert contact.position == (0.25, 0.25, 0.0), contact
    assert contact.store_normal0 == (0.0, 0.0, 1.0), contact
    assert contact.store_normal1 == (1.0, 0.0, 0.0), contact
    assert contact.record_hit_source == "cbsp_node_plane_vertex_probe", contact

    traversal_contact = grid._triangle_cbsp_node_plane_vertex_contact(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        vertices,
        tree,
        traversal_order=True,
    )
    assert traversal_contact is not None
    assert traversal_contact.record_hit_source == "cbsp_node_plane_vertex_traversal_probe"
    print("test_triangle_cbsp_node_plane_vertex_probe_uses_split_normal: PASSED")
    return True


def test_triangle_cbsp_strict_probe_skips_heuristic_vertex_fallback():
    """Strict CBSP probe should report only decompile leaf/edge-triangle hits."""

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
        Vec3(0.25, 0.25, -0.1),
        Vec3(0.8, 0.25, 0.2),
        Vec3(0.25, 0.8, 0.2),
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
    terrain_tri = (
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )

    default_contact = grid._triangle_cbsp_contact(
        terrain_tri,
        vertices,
        tree,
        2.0,
    )
    strict_contact = grid._triangle_cbsp_contact(
        terrain_tri,
        vertices,
        tree,
        2.0,
        include_heuristic_fallbacks=False,
    )

    assert default_contact is not None
    assert default_contact.record_hit_source == "cbsp_vertex_contact", default_contact
    assert strict_contact is None, strict_contact
    print("test_triangle_cbsp_strict_probe_skips_heuristic_vertex_fallback: PASSED")
    return True


def test_triangle_cbsp_guess7_order_probe_rejects_target_point_shortcut():
    """Default-off GUESS7 probe should reject the old non-point-dependent shortcut."""

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
    point = (5.920485, -3.373555, -1.851819)
    vertices = [
        Vec3(5.934570, -3.314636, -1.856628),
        Vec3(5.934570, -1.390167, -1.856628),
        Vec3(6.796585, -0.727112, -1.193542),
    ]
    split_normal = Vec3(0.609709, 0.0, -0.792625)
    tree = CBSPTree(
        nodes=[
            CBSPTreeNode(
                radius=8.0,
                half_extent_x=4.0,
                half_extent_y=4.0,
                half_extent_z=4.0,
                center=Vec3(5.0, -2.0, -1.5),
                triangles=[(0, 1, 2)],
                split_normal=split_normal,
                split_plane_d=-(
                    split_normal.x * point[0]
                    + split_normal.y * point[1]
                    + split_normal.z * point[2]
                ),
                child_neg=-1,
                child_pos=-1,
            )
        ],
        root_index=0,
    )
    tri_local = (
        (
            point[0] - split_normal.x,
            point[1] - 1.0,
            point[2] - split_normal.z,
        ),
        (
            point[0] + split_normal.x,
            point[1] + 1.0,
            point[2] + split_normal.z,
        ),
        (6.5, -2.0, -1.0),
    )

    strict_contact = grid._triangle_cbsp_contact(
        tri_local,
        vertices,
        tree,
        8.0,
        include_heuristic_fallbacks=False,
    )
    guess7_contact = grid._triangle_cbsp_contact(
        tri_local,
        vertices,
        tree,
        8.0,
        include_heuristic_fallbacks=False,
        point_inside_mode="guess7_order_probe",
    )

    assert strict_contact is None, strict_contact
    assert guess7_contact is None, guess7_contact

    ordered = (
        (vertices[0].x, vertices[0].y, vertices[0].z),
        (vertices[1].x, vertices[1].y, vertices[1].z),
        (vertices[2].x, vertices[2].y, vertices[2].z),
    )
    normal = _normalize3(_cross3(_sub3(ordered[1], ordered[0]), _sub3(ordered[2], ordered[0])))
    terms = TerrainGridCollision._guess7_point_in_triangle_terms(point, ordered, normal)
    assert tuple(round(term, 6) for term in terms) == (
        0.015842,
        0.042776,
        -2.151564,
    ), terms
    print("test_triangle_cbsp_guess7_order_probe_rejects_target_point_shortcut: PASSED")
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


def test_model_collision_can_probe_upward_min_depth_selection():
    """Default-off contact selection can skip side splits and choose a shallow upward hit."""

    class FlatTerrain:
        cell_x = 1.0
        cell_z = 1.0
        world_w = 3.0
        world_h = 2.0
        num_x = 4
        num_z = 2

        @staticmethod
        def _get_raw_height(cell_x, cell_y):
            return 0.0

    grid = TerrainGridCollision(FlatTerrain(), 0.0, sector_rows=1, sector_cols=1)
    tri = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    calls = []

    grid._iter_aabb_sectors = lambda aabb_min, aabb_max: [SimpleNamespace(index=0)]
    grid._iter_sector_cells = lambda aabb_min, aabb_max, sector: [(0, 0), (1, 0), (2, 0)]
    grid._iter_cell_triangles = lambda cell_x, cell_y: [tri]
    grid._triangle_overlaps_aabb = lambda *args, **kwargs: True

    contacts = [
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.1),
        ((0.0, 0.0, 0.0), (0.0, 0.1, 0.995), 30.0),
        ((0.0, 0.0, 0.0), (0.0, 0.2, 0.98), 4.0),
    ]

    def fake_triangle_cbsp_contact(tri_local, vertices, cbsp_tree, bounding_radius):
        calls.append((tri_local, bounding_radius))
        return contacts[len(calls) - 1]

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
        contact_selection="upward_min_depth",
    )
    assert contact is not None
    assert len(calls) == 3, calls
    assert contact.cell == (2, 0), contact.cell
    assert math.isclose(contact.penetration, 4.0, abs_tol=1e-6), contact.penetration
    assert contact.normal[2] > 0.9, contact.normal
    print("test_model_collision_can_probe_upward_min_depth_selection: PASSED")
    return True


def test_model_bounds_contact_can_probe_upward_min_depth_selection():
    """Dirty bounds model contact selection should match the clean model contact selector."""

    class FlatTerrain:
        cell_x = 1.0
        cell_z = 1.0
        world_w = 3.0
        world_h = 2.0
        num_x = 4
        num_z = 2

        @staticmethod
        def _get_raw_height(cell_x, cell_y):
            return 0.0

    grid = TerrainGridCollision(FlatTerrain(), 0.0, sector_rows=1, sector_cols=1)
    tri = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    calls = []

    grid._iter_aabb_sectors = lambda aabb_min, aabb_max: [SimpleNamespace(index=0)]
    grid._iter_sector_cells = lambda aabb_min, aabb_max, sector: [(0, 0), (1, 0), (2, 0)]
    grid._iter_cell_triangles = lambda cell_x, cell_y: [tri]
    grid._triangle_overlaps_aabb = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("dirty bounds model contact should not use the 3d prefilter")
    )

    contacts = [
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.1),
        ((0.0, 0.0, 0.0), (0.0, 0.1, 0.995), 30.0),
        ((0.0, 0.0, 0.0), (0.0, 0.2, 0.98), 4.0),
    ]

    def fake_triangle_cbsp_contact(tri_local, vertices, cbsp_tree, bounding_radius):
        calls.append((tri_local, bounding_radius))
        return contacts[len(calls) - 1]

    grid._triangle_cbsp_contact = fake_triangle_cbsp_contact

    contact = grid.test_model_bounds_contact(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        0.0,
        [],
        object(),
        1.0,
        contact_selection="upward_min_depth",
    )
    assert contact is not None
    assert len(calls) == 3, calls
    assert contact.cell == (2, 0), contact.cell
    assert math.isclose(contact.penetration, 4.0, abs_tol=1e-6), contact.penetration
    assert contact.normal[2] > 0.9, contact.normal
    print("test_model_bounds_contact_can_probe_upward_min_depth_selection: PASSED")
    return True


def test_model_collision_response_normal_can_probe_terrain_triangle_normal():
    """The terrain-face response probe should be opt-in; the CBSP split normal remains default."""

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

    tri = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0))

    def install_fake_grid(grid):
        grid._iter_aabb_sectors = lambda aabb_min, aabb_max: [SimpleNamespace(index=0)]
        grid._iter_sector_cells = lambda aabb_min, aabb_max, sector: [(0, 0)]
        grid._iter_cell_triangles = lambda cell_x, cell_y: [tri]
        grid._triangle_overlaps_aabb = lambda *args, **kwargs: True
        grid._triangle_cbsp_contact = lambda *args, **kwargs: (
            (0.25, 0.0, -0.5),
            (1.0, 0.0, 0.0),
            0.1,
        )

    def make_tree():
        return CBSPTree(nodes=[CBSPTreeNode(
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
        )])

    grid = TerrainGridCollision(FlatTerrain(), 0.0, sector_rows=1, sector_cols=1)
    install_fake_grid(grid)
    contact = grid.test_model_collision((0.5, 0.5, 0.5), 0.0, [], make_tree(), 1.0)
    assert contact is not None
    assert contact.normal_source == "entity_cbsp_split", contact.normal_source
    assert contact.normal == (1.0, 0.0, 0.0), contact.normal
    assert contact.cbsp_split_normal == (1.0, 0.0, 0.0), contact.cbsp_split_normal
    assert math.isclose(contact.terrain_face_normal[0], 0.0, abs_tol=1e-6), contact
    assert math.isclose(contact.terrain_face_normal[1], -math.sqrt(0.5), abs_tol=1e-6), contact
    assert math.isclose(contact.terrain_face_normal[2], math.sqrt(0.5), abs_tol=1e-6), contact
    assert contact.entity_radial_normal is not None, contact
    assert math.isclose(
        math.sqrt(sum(component * component for component in contact.entity_radial_normal)),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ), contact.entity_radial_normal

    terrain_grid = TerrainGridCollision(
        FlatTerrain(),
        0.0,
        sector_rows=1,
        sector_cols=1,
        model_contact_normal_source="terrain",
    )
    install_fake_grid(terrain_grid)
    terrain_contact = terrain_grid.test_model_collision((0.5, 0.5, 0.5), 0.0, [], make_tree(), 1.0)
    assert terrain_contact is not None
    assert terrain_contact.normal_source == "terrain_triangle", terrain_contact.normal_source
    assert math.isclose(terrain_contact.normal[0], 0.0, abs_tol=1e-6), terrain_contact.normal
    assert math.isclose(terrain_contact.normal[1], -math.sqrt(0.5), abs_tol=1e-6), terrain_contact.normal
    assert math.isclose(terrain_contact.normal[2], math.sqrt(0.5), abs_tol=1e-6), terrain_contact.normal
    assert terrain_contact.cbsp_split_normal == (1.0, 0.0, 0.0), terrain_contact.cbsp_split_normal
    assert math.isclose(terrain_contact.terrain_face_normal[2], math.sqrt(0.5), abs_tol=1e-6), terrain_contact
    radial_grid = TerrainGridCollision(
        FlatTerrain(),
        0.0,
        sector_rows=1,
        sector_cols=1,
        model_contact_normal_source="entity_radial",
    )
    install_fake_grid(radial_grid)
    radial_contact = radial_grid.test_model_collision((0.5, 0.5, 0.5), 0.0, [], make_tree(), 1.0)
    assert radial_contact is not None
    assert radial_contact.normal_source == "entity_radial", radial_contact.normal_source
    assert radial_contact.entity_radial_normal is not None, radial_contact
    for got, expected in zip(radial_contact.normal, radial_contact.entity_radial_normal):
        assert math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-6), radial_contact
    print("test_model_collision_response_normal_can_probe_terrain_triangle_normal: PASSED")
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


def test_entity_world_collision_defaults_enabled_with_env_optout():
    """Entity terrain collision should be live by default, with an explicit debug opt-out."""
    old_env = os.environ.get("WULFRAM_ENTITY_TERRAIN_COLLISION")
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_COLLISION", None)
        server = WulframServer(host="127.0.0.1", port=0)
        assert server.entity_terrain_collision_enabled is True

        os.environ["WULFRAM_ENTITY_TERRAIN_COLLISION"] = "0"
        server = WulframServer(host="127.0.0.1", port=0)
        assert server.entity_terrain_collision_enabled is False

        os.environ["WULFRAM_ENTITY_TERRAIN_COLLISION"] = "false"
        server = WulframServer(host="127.0.0.1", port=0)
        assert server.entity_terrain_collision_enabled is False
    finally:
        if old_env is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_COLLISION", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_COLLISION"] = old_env
    print("test_entity_world_collision_defaults_enabled_with_env_optout: PASSED")
    return True


def test_entity_world_collision_can_be_disabled_for_player_sync():
    """Server can leave terrain shape collision out of player motion while keeping other terrain users."""
    collision_calls = 0

    def fake_box_collision(*args, **kwargs):
        nonlocal collision_calls
        collision_calls += 1
        return TerrainContact(
            position=(50.0, 75.0, 6.0),
            normal=(-1.0, 0.0, 0.0),
            penetration=8.0,
            sector_index=0,
            cell=(0, 0),
        )

    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = SimpleNamespace(test_box_collision=fake_box_collision)
    server.entity_terrain_collision_enabled = False

    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
    )
    ctx.player_heading = 0.0
    ctx.world_collision_bounds_dirty = True

    result = server._resolve_entity_world_collision(
        ctx,
        100.0,
        200.0,
        10.0,
        3.0,
        4.0,
        -6.0,
    )

    assert result == (100.0, 200.0, 10.0, 3.0, 4.0, -6.0), result
    assert collision_calls == 0, collision_calls
    assert ctx.world_collision_bounds_dirty is False, ctx.world_collision_bounds_dirty
    print("test_entity_world_collision_can_be_disabled_for_player_sync: PASSED")
    return True


@_legacy_contact_response_test
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


def _fake_tank_collision_server():
    server = WulframServer.__new__(WulframServer)
    server._entity_collision_model_cache = {}
    server._building_collision = SimpleNamespace(
        available=True,
        models={
            "tank_1": SimpleNamespace(
                collision_mesh=SimpleNamespace(
                    vertices=[
                        SimpleNamespace(x=-1.0, y=-2.0, z=-3.0),
                        SimpleNamespace(x=1.0, y=2.0, z=4.0),
                    ]
                ),
                cbsp_tree=SimpleNamespace(
                    nodes=[object()],
                    root=SimpleNamespace(radius=7.5),
                ),
            ),
            "tank_1_s": SimpleNamespace(
                collision_mesh=SimpleNamespace(
                    vertices=[
                        SimpleNamespace(x=-0.5, y=-1.0, z=-1.5),
                        SimpleNamespace(x=0.5, y=1.0, z=2.0),
                    ]
                ),
                cbsp_tree=SimpleNamespace(
                    nodes=[object()],
                    root=SimpleNamespace(radius=3.75),
                ),
            ),
        },
    )
    return server


def _fake_tank_collision_context():
    ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=Session(),
        entity_id=0x14EA,
        entity_type=int(EntityType.TANK),
    )
    ctx.session.team_id = 2
    return ctx


def test_entity_world_collision_model_defaults_to_legacy_lift():
    """Default vehicle terrain CBSP origin stays on the live-stable lifted mesh path."""
    old_env = os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN")
    try:
        os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        server = _fake_tank_collision_server()
        ctx = _fake_tank_collision_context()

        _vertices, _tree, radius, z_lift = server._get_entity_world_collision_model(ctx)

        assert abs(radius - 7.5) < 1e-6, radius
        assert abs(z_lift - 3.0) < 1e-6, z_lift
    finally:
        if old_env is None:
            os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        else:
            os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = old_env
    print("test_entity_world_collision_model_defaults_to_legacy_lift: PASSED")
    return True


def test_entity_world_collision_model_entity_origin_can_be_requested():
    """The entity-origin terrain transform remains available as an opt-in probe."""
    old_env = os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN")
    try:
        os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = "entity"
        server = _fake_tank_collision_server()
        ctx = _fake_tank_collision_context()

        _vertices, _tree, radius, z_lift = server._get_entity_world_collision_model(ctx)

        assert abs(radius - 7.5) < 1e-6, radius
        assert z_lift == 0.0, z_lift
    finally:
        if old_env is None:
            os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        else:
            os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = old_env
    print("test_entity_world_collision_model_entity_origin_can_be_requested: PASSED")
    return True


def test_entity_world_collision_model_simplified_variant_is_opt_in():
    """Default-off model variant can select the OG-style simplified `_s` CBSP mesh."""
    env_key = "WULFRAM_ENTITY_COLLISION_MODEL_VARIANT"
    old_env = os.environ.get(env_key)
    try:
        os.environ[env_key] = "simplified"
        server = _fake_tank_collision_server()
        ctx = _fake_tank_collision_context()

        vertices, _tree, radius, z_lift = server._get_entity_world_collision_model(ctx)

        assert len(vertices) == 2, vertices
        assert abs(radius - 3.75) < 1e-6, radius
        assert abs(z_lift - 1.5) < 1e-6, z_lift
        assert abs(vertices[0].x + 0.5) < 1e-6, vertices[0]
    finally:
        if old_env is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_env
    print("test_entity_world_collision_model_simplified_variant_is_opt_in: PASSED")
    return True


def test_entity_world_collision_model_contact_can_use_body_matrix_probe():
    """Default-off terrain model contact probing can use full body rotation."""
    env_key = "WULFRAM_ENTITY_TERRAIN_MODEL_CONTACT_ROTATION"
    old_env = os.environ.get(env_key)
    os.environ[env_key] = "body"
    calls = []
    try:
        def fake_model_collision(center, heading, vertices, cbsp_tree, radius, **kwargs):
            calls.append(
                {
                    "center": center,
                    "heading": heading,
                    "rotation_matrix": kwargs.get("rotation_matrix"),
                }
            )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.5
        ctx.player_pose = {"roll": 0.1, "pitch": -0.2}
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.1, -0.2, 0.0)
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)

        server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            1.0,
            0.0,
            0.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(1.0, 0.0, 0.0),
            dt=1.0 / 30.0,
        )

        assert calls, calls
        assert all(call["rotation_matrix"] is not None for call in calls), calls
        assert all(len(call["rotation_matrix"]) == 9 for call in calls), calls
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["model_contact_rotation_source"] == "body_matrix", probe
        assert probe["model_contact_rotation_mode"] == "body", probe
    finally:
        if old_env is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_env
    print("test_entity_world_collision_model_contact_can_use_body_matrix_probe: PASSED")
    return True


def test_entity_world_collision_records_raw_origin_probe_without_applying_contact():
    """Default lifted misses should expose raw-origin contact evidence without changing physics."""
    old_fallback = os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK")
    os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
    calls = []

    try:
        def fake_model_collision(center, *args, **kwargs):
            calls.append(center)
            if abs(center[2] - 10.0) < 1e-6:
                return TerrainContact(
                    position=(1.0, 2.0, 3.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=4.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="terrain_triangle",
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)

        result = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            1.0,
            2.0,
            3.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(1.0, 2.0, 3.0),
            dt=1.0 / 30.0,
        )

        assert result == (0.0, 0.0, 10.0, 1.0, 2.0, 3.0), result
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["reason"] == "lifted_clear_raw_origin_contact", probe
        assert probe["model_z_lift"] == 3.0
        assert probe["lifted_contact"] is None
        assert probe["raw_origin_contact"]["depth"] == 4.0
        assert probe["raw_origin_contact"]["contact_cell"] == (7, 8)
        assert probe["raw_origin_fallback_enabled"] is False
        assert probe["raw_origin_fallback_reject"] == "disabled"
        assert calls[0] == (0.0, 0.0, 13.0)
        assert calls[1] == (0.0, 0.0, 10.0)
    finally:
        if old_fallback is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = old_fallback
    print("test_entity_world_collision_records_raw_origin_probe_without_applying_contact: PASSED")
    return True


def test_entity_world_collision_reference_pose_probe_records_pre_step_contact_when_enabled():
    """The reference-pose probe should expose pre-step raw/pair contacts without applying them."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_PROBE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_PROBE"] = "1"

        def fake_model_collision(center, *args, **kwargs):
            if (
                abs(center[0]) < 1e-6
                and abs(center[1]) < 1e-6
                and abs(center[2] - 10.0) < 1e-6
            ):
                return TerrainContact(
                    position=(0.0, 0.0, 6.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=4.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="terrain_triangle",
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)

        result = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            10.0,
            1.0,
            0.0,
            0.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(1.0, 0.0, 0.0),
            dt=1.0 / 30.0,
        )

        assert result == (10.0, 0.0, 10.0, 1.0, 0.0, 0.0), result
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["reference_pose_probe_enabled"] is True, probe
        assert probe["pre_step_pos"] == (0.0, 0.0, 10.0), probe
        assert probe["pre_step_vel"] == (1.0, 0.0, 0.0), probe
        assert abs(probe["step_dt"] - (1.0 / 30.0)) < 1e-9, probe
        assert probe["timing_ready"] is True, probe
        assert probe["pair_record_timed_contact_enabled"] is False, probe
        assert probe["pair_record_continue_remaining_enabled"] is False, probe
        references = probe["reference_pose_contacts"]
        assert "pre_pos" in references, references
        pre_ref = references["pre_pos"]
        assert pre_ref["raw_contact_any"] is True, pre_ref
        assert pre_ref["pair_record_contact_accept"] is True, pre_ref
        assert pre_ref["pair_record_contact"]["contact_cell"] == (7, 8), pre_ref
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_reference_pose_probe_records_pre_step_contact_when_enabled: PASSED"
    )
    return True


def test_entity_world_collision_reference_pose_pair_record_contact_can_apply_when_enabled():
    """Opt-in reference-pose response should use the latest accepted pre/current pair contact."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_CONTACT_ORDER",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_CONTACT"] = "1"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_CONTACT_ORDER", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ[
            "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"
        ] = "entity_radial_terrain_face_forward_up"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        calls = []

        def fake_model_collision(center, *args, **kwargs):
            calls.append((tuple(center), kwargs.get("contact_selection")))
            if (
                abs(center[2] - 10.0) < 1e-6
                and 0.0 <= center[0] < 9.0
            ):
                return TerrainContact(
                    position=(center[0], 0.0, 6.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=4.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.6, 0.0, 0.8),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        _px, _py, pz, _vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            10.0,
            10.0,
            0.0,
            -10.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(10.0, 0.0, -10.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_reference_pose_pair_record_contact", debug
        assert debug["reference_pose_pair_record_contact"] is True, debug
        assert debug["reference_pose_contact_label"] == "pre_to_current_75", debug
        assert debug["reference_pose_current_pair_reject"] == "no_raw_origin_contact", debug
        assert debug["pair_record_contact"] is True, debug
        assert debug["pair_record_contact_reason"] == "reference_pose_pair_record_contact", debug
        assert debug["pair_record_contact_reject"] == "", debug
        assert debug["raw_origin_fallback_velocity_delta_mag_after_safety"] > 0.0, debug
        assert debug["raw_origin_fallback_angular_preserved"] is True, debug
        assert pz > 10.0 and vz > -10.0, (pz, vz, debug)
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["reference_pose_probe_enabled"] is True, probe
        assert probe["reference_pose_contact_response_enabled"] is True, probe
        assert probe["pair_record_contact_reject"] == "no_raw_origin_contact", probe
        assert probe["reference_pose_contacts"]["pre_to_current_75"][
            "pair_record_contact_accept"
        ] is True, probe
        assert any(
            call == ((7.5, 0.0, 10.0), "upward_min_depth")
            for call in calls
        ), calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_reference_pose_pair_record_contact_can_apply_when_enabled: PASSED"
    )
    return True


def test_entity_world_collision_reference_pose_pair_response_preserves_position():
    """Reference-pose pair response should apply only velocity delta at current pose."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_PAIR_RESPONSE",
        "WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_PAIR_RESPONSE_MAX_DISTANCE",
        "WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_CONTACT_ORDER",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_PAIR_RESPONSE"] = "apply"
        os.environ[
            "WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_PAIR_RESPONSE_MAX_DISTANCE"
        ] = "3.0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_REFERENCE_POSE_CONTACT_ORDER", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        calls = []

        def fake_model_collision(center, *args, **kwargs):
            calls.append((tuple(center), kwargs.get("contact_selection")))
            if (
                abs(center[2] - 10.0) < 1e-6
                and 0.0 <= center[0] < 9.0
            ):
                return TerrainContact(
                    position=(center[0], 0.0, 6.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=4.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.6, 0.0, 0.8),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        result = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            10.0,
            10.0,
            0.0,
            -10.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(2.0, 0.0, -4.0),
            dt=1.0 / 30.0,
        )

        px, py, pz, _vx, _vy, vz = result
        assert (px, py, pz) == (10.0, 0.0, 10.0), result
        assert vz > -10.0, result
        assert vz - (-10.0) <= 0.500001, result
        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_reference_pose_pair_response", debug
        assert debug["pair_record_contact_reason"] == "reference_pose_pair_response", debug
        assert debug["reference_pose_pair_response_preserved_position"] is True, debug
        assert debug["reference_pose_pair_response_label"] == "pre_to_current_75", debug
        assert debug["reference_pose_pair_response_max_distance"] == 3.0, debug
        assert abs(debug["reference_pose_pair_response_current_distance"] - 2.5) < 1e-6, debug
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["reference_pose_probe_enabled"] is True, probe
        assert probe["reference_pose_pair_response_enabled"] is True, probe
        response = probe["reference_pose_pair_response"]
        assert response["applied"] is True, response
        assert response["applied_to_current_state"] is True, response
        assert response["label"] == "pre_to_current_75", response
        assert response["max_distance"] == 3.0, response
        assert abs(response["current_distance"] - 2.5) < 1e-6, response
        assert abs(response["current_xy_distance"] - 2.5) < 1e-6, response
        assert response["current_z_delta"] == 0.0, response
        assert response["current_pos"] == (10.0, 0.0, 10.0), response
        assert response["velocity_before"] == (8.0, 0.0, -8.5), response
        assert response["velocity_fraction"] == 0.75, response
        assert response["velocity_source"] == "pre_to_current_fraction", response
        assert response["final_vel"][2] == vz, response
        assert any(
            call == ((7.5, 0.0, 10.0), "upward_min_depth")
            for call in calls
        ), calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_reference_pose_pair_response_preserves_position: PASSED"
    )
    return True


def test_entity_world_collision_cached_pair_record_contact_can_bridge_lifted_clear_when_enabled():
    """Opt-in cached pair-record contact can replay a short-lived OG-style world pair."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CACHE_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CACHE_MAX_AGE_STEPS",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CACHE_MAX_DISTANCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CACHE_CONTACT"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CACHE_MAX_AGE_STEPS"] = "4"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CACHE_MAX_DISTANCE"] = "3.0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        calls = []

        def fake_model_collision(center, *args, **kwargs):
            calls.append((tuple(center), kwargs.get("contact_selection")))
            if abs(center[2]) < 1e-6 and center[0] < 0.5:
                return TerrainContact(
                    position=(0.0, 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=2.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        _px, _py, _pz, _vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(0.0, 0.0, -1.0),
            dt=1.0 / 30.0,
        )
        first_debug = ctx.debug_last_motion_collision
        assert first_debug["kind"] == "terrain_pair_record_contact", first_debug
        assert first_debug["pair_record_contact_reason"] == "lifted_clear_raw_origin_contact", first_debug
        assert vz > -1.0, (vz, first_debug)

        _px, _py, pz, _vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -1.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(0.0, 0.0, -1.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_cached_pair_record_contact", debug
        assert debug["cached_pair_record_contact"] is True, debug
        assert debug["pair_record_contact_reason"] == "cached_pair_record_contact", debug
        assert debug["pair_record_cached_contact_age_steps"] == 1, debug
        assert debug["pair_record_cached_contact_distance"] < 3.0, debug
        assert pz >= 0.0 and vz > -1.0, (pz, vz, debug)
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_contact_reject"] == "no_raw_origin_contact", probe
        assert probe["pair_record_cached_contact_reject"] == "", probe
        assert any(call[1] == "upward_min_depth" for call in calls), calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_cached_pair_record_contact_can_bridge_lifted_clear_when_enabled: PASSED"
    )
    return True


def test_entity_world_collision_deferred_prestep_pair_record_contact_when_enabled():
    """Opt-in pre-step pair-record contact should resolve at the start pose then step forward."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP_MAX_DISTANCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP_MAX_DISTANCE"] = "2.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        calls = []

        def fake_model_collision(center, *args, **kwargs):
            calls.append((tuple(center), kwargs.get("contact_selection")))
            if abs(center[0]) < 1e-6 and abs(center[2]) < 1e-6:
                return TerrainContact(
                    position=(0.0, 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        px, _py, pz, _vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            -1.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(1.0, 0.0, -1.0),
            dt=1.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_deferred_prestep_pair_record_contact", debug
        assert debug["deferred_prestep_pair_record_contact"] is True, debug
        assert debug["pair_record_contact_reason"] == "deferred_prestep_pair_record_contact", debug
        assert debug["pair_record_deferred_prestep_pos"] == (0.0, 0.0, 0.0), debug
        assert debug["pair_record_deferred_prestep_endpoint_pos"] == (1.0, 0.0, -1.0), debug
        assert debug["pair_record_deferred_prestep_remaining_time_s"] == 1.0, debug
        assert debug["contact_events"][0]["pair_record_deferred_prestep_contact"] is True, debug
        assert px > 0.9 and pz > -1.0 and vz > -1.0, (px, pz, vz, debug)
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_contact_reject"] == "no_raw_origin_contact", probe
        assert probe["pair_record_deferred_prestep_reject"] == "", probe
        assert probe["pair_record_deferred_prestep_contact"]["contact_cell"] == (7, 8), probe
        assert any(call == ((0.0, 0.0, 0.0), "upward_min_depth") for call in calls), calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_deferred_prestep_pair_record_contact_when_enabled: PASSED"
    )
    return True


def test_entity_world_collision_deferred_prestep_pair_record_probe_is_read_only():
    """Read-only pre-step pair-record probe should report contact without resolving it."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP_PROBE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP_MAX_DISTANCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP_PROBE"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DEFERRED_PRESTEP_MAX_DISTANCE"] = "2.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        calls = []

        def fake_model_collision(center, *args, **kwargs):
            calls.append((tuple(center), kwargs.get("contact_selection")))
            if abs(center[0]) < 1e-6 and abs(center[2]) < 1e-6:
                return TerrainContact(
                    position=(0.0, 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        result = server._resolve_entity_world_collision(
            ctx,
            1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            -1.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(1.0, 0.0, -1.0),
            dt=1.0,
        )

        assert result == (1.0, 0.0, -1.0, 1.0, 0.0, -1.0), result
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}, (
            getattr(ctx, "debug_last_motion_collision", {})
        )
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_contact_reject"] == "no_raw_origin_contact", probe
        assert probe["pair_record_deferred_prestep_enabled"] is False, probe
        assert probe["pair_record_deferred_prestep_probe_enabled"] is True, probe
        assert probe["pair_record_deferred_prestep_reject"] == "", probe
        assert probe["pair_record_deferred_prestep_distance"] == 1.4142135623730951, probe
        assert probe["pair_record_deferred_prestep_pos"] == (0.0, 0.0, 0.0), probe
        assert probe["pair_record_deferred_prestep_contact"]["contact_cell"] == (7, 8), probe
        assert any(call == ((0.0, 0.0, 0.0), "upward_min_depth") for call in calls), calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_deferred_prestep_pair_record_probe_is_read_only: PASSED"
    )
    return True


def test_entity_world_collision_pair_record_contact_can_continue_remaining_step_when_enabled():
    """Opt-in direct pair-record contacts can apply at estimated TOI then step the remainder."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING"] = "1"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        calls = []

        def fake_model_collision(center, *args, **kwargs):
            calls.append((tuple(center), kwargs.get("contact_selection")))
            if center[0] >= 0.5 and center[2] < 1.0:
                return TerrainContact(
                    position=(center[0], 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        px, _py, pz, _vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            -1.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(1.0, 0.0, -1.0),
            dt=1.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_pair_record_continued_contact", debug
        assert debug["pair_record_continued_contact"] is True, debug
        assert debug["pair_record_continue_remaining_enabled"] is True, debug
        assert 0.49 <= debug["pair_record_continue_collision_time_s"] <= 0.51, debug
        assert 0.49 <= debug["pair_record_continue_remaining_time_s"] <= 0.51, debug
        assert debug["pair_record_contact"] is True, debug
        assert debug["pair_record_contact_reject"] == "", debug
        assert debug["contact_events"][0]["pair_record_continued_contact"] is True, debug
        assert debug["contact_events"][0]["velocity_before"][2] == -1.0, debug
        assert debug["contact_events"][0]["velocity_after"][2] > -1.0, debug
        assert px > 0.5 and pz > -1.0 and vz > -1.0, (px, pz, vz, debug)
        assert any(call[1] == "upward_min_depth" for call in calls), calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_pair_record_contact_can_continue_remaining_step_when_enabled: PASSED"
    )
    return True


def test_entity_world_collision_pair_record_continue_sweeps_transient_contact():
    """Continue-remaining pair-record contact should catch a mid-step contact even when the endpoint is clear."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN_STEPS",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN_STEPS"] = "9"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        def fake_model_collision(center, *args, **kwargs):
            if 0.45 <= center[0] <= 0.55:
                return TerrainContact(
                    position=(center[0], 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        px, _py, pz, _vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            -1.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(1.0, 0.0, -1.0),
            dt=1.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_pair_record_continued_contact", debug
        assert debug["pair_record_continued_contact"] is True, debug
        assert debug["pair_record_continue_contact_sweep_scan"] is True, debug
        assert 0.44 <= debug["pair_record_continue_collision_time_s"] <= 0.46, debug
        assert 0.54 <= debug["pair_record_continue_remaining_time_s"] <= 0.56, debug
        event = debug["contact_events"][0]
        assert event["contact_sweep_scan"] is True, event
        assert event["pair_record_continued_contact"] is True, event
        assert event["velocity_before"][2] == -1.0, event
        assert event["velocity_after"][2] > -1.0, event
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_contact_reject"] == "no_raw_origin_contact", probe
        assert probe["pair_record_continue_probe_result"] == "interval_contact", probe
        assert probe["pair_record_continue_contact_sweep_scan"] is True, probe
        assert px > 0.45 and pz > -1.0 and vz > -1.0, (px, pz, vz, debug)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_pair_record_continue_sweeps_transient_contact: PASSED"
    )
    return True


def test_entity_world_collision_pair_record_schedule_probe_reports_without_applying():
    """Default-off schedule probe should report interval contact without changing motion."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN_STEPS",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SCHEDULE_PROBE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SCHEDULE_RESPONSE_PROBE",
        "WULFRAM_ENTITY_TERRAIN_SELECTED_ROW_PHASE_TRACE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN_STEPS"] = "9"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SCHEDULE_PROBE"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SCHEDULE_RESPONSE_PROBE"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_SELECTED_ROW_PHASE_TRACE"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        def fake_model_collision(center, *args, **kwargs):
            if 0.45 <= center[0] <= 0.55:
                return TerrainContact(
                    position=(center[0], 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        result = server._resolve_entity_world_collision(
            ctx,
            1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            -1.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(1.0, 0.0, -1.0),
            dt=1.0,
        )

        assert result == (1.0, 0.0, -1.0, 1.0, 0.0, -1.0), result
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_schedule_probe_enabled"] is True, probe
        assert probe["pair_record_continue_remaining_enabled"] is False, probe
        assert probe["pair_record_schedule_probe_result"] == "interval_contact", probe
        assert 0.44 <= probe["pair_record_schedule_collision_time_s"] <= 0.46, probe
        assert probe["pair_record_schedule_decompile_pool_model"] == (
            "CollisionPairPool_30_bucket_time_order"
        ), probe
        assert probe["pair_record_schedule_step_dt_s"] == 1.0, probe
        assert probe["pair_record_schedule_bucket_count"] == 30, probe
        assert probe["pair_record_schedule_bucket_rate_hz"] == 30.0, probe
        assert probe["pair_record_schedule_bucket_index"] == 13, probe
        assert 0.433 <= probe["pair_record_schedule_bucket_start_s"] <= 0.434, probe
        assert 0.466 <= probe["pair_record_schedule_bucket_end_s"] <= 0.467, probe
        assert probe[
            "pair_record_schedule_resolve_remaining_to_tick_end_s"
        ] == probe["pair_record_schedule_remaining_time_s"], probe
        assert probe["pair_record_schedule_contact_sweep_scan"] is True, probe
        assert probe["pair_record_schedule_contact"]["contact_cell"] == (7, 8), probe
        assert probe["selected_row_phase_trace_enabled"] is True, probe
        trace = probe["selected_row_phase_trace"]
        assert trace["runtime_default"] == "off", trace
        assert trace["selected_pos"] == (1.0, 0.0, -1.0), trace
        assert trace["pre_step_pos"] == (0.0, 0.0, 0.0), trace
        assert trace["pair_record_contact_reject"] == "no_raw_origin_contact", trace
        assert trace["pair_record_schedule_probe_result"] == "interval_contact", trace
        assert 0.44 <= trace["pair_record_schedule_collision_time_s"] <= 0.46, trace
        assert trace["pair_record_schedule_bucket_index"] == 13, trace
        assert trace["pair_record_schedule_contact"]["contact_cell"] == (7, 8), trace
        assert trace["pair_record_schedule_response_probe_enabled"] is True, trace
        assert "pair_record_schedule_response_probe_result" in trace, trace
        assert "pair_record_continue_probe_result" not in probe, probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_pair_record_schedule_probe_reports_without_applying: PASSED"
    )
    return True


def test_entity_world_collision_frame_phase_report_first_probe_is_read_only():
    """Frame-phase report-first probe should sample CBSP candidates without applying."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_FRAME_PHASE_PROBE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_FRAME_PHASE_RESOLVE_PREVIEW",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ.pop(
            "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_FRAME_PHASE_RESOLVE_PREVIEW",
            None,
        )
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_FRAME_PHASE_PROBE"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"

        def fake_model_collision(center, *args, **kwargs):
            selection = kwargs.get("contact_selection")
            if (
                selection == "cbsp_mesh_edge_terrain_plane_probe"
                and 0.49 <= center[0] <= 0.51
            ):
                return TerrainContact(
                    position=(center[0], 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                    cbsp_store_normal0=(0.0, 0.0, 1.0),
                    cbsp_store_normal1=(0.0, 0.0, -1.0),
                    cbsp_record_hit_source="cbsp_mesh_edge_terrain_plane_probe_ab",
                    cbsp_mesh_triangle_indices=(6, 8, 9),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        result = server._resolve_entity_world_collision(
            ctx,
            1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            -1.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(1.0, 0.0, -1.0),
            dt=1.0,
        )

        assert result == (1.0, 0.0, -1.0, 1.0, 0.0, -1.0), result
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_frame_phase_probe_enabled"] is True, probe
        frame_probe = probe["pair_record_frame_phase_probe"]
        assert frame_probe["runtime_default"] == "off", frame_probe
        assert frame_probe["frame_pose_start"] == (0.0, 0.0, 0.0), frame_probe
        assert frame_probe["frame_pose_end"] == (1.0, 0.0, -1.0), frame_probe
        assert frame_probe["frame_pose_delta"] == (1.0, 0.0, -1.0), frame_probe
        assert math.isclose(
            frame_probe["frame_pose_span_u"],
            math.sqrt(2.0),
            rel_tol=0.0,
            abs_tol=1e-12,
        ), frame_probe
        assert frame_probe["frame_pose_span_verdict"] == "frame_pose_varies", frame_probe
        assert frame_probe["frame_velocity_start"] == (1.0, 0.0, -1.0), frame_probe
        assert frame_probe["frame_velocity_end"] == (1.0, 0.0, -1.0), frame_probe
        assert frame_probe["frame_velocity_delta"] == (0.0, 0.0, 0.0), frame_probe
        assert frame_probe["frame_velocity_span_u"] == 0.0, frame_probe
        assert frame_probe["frame_pose_velocity_integrated_end"] == (
            1.0,
            0.0,
            -1.0,
        ), frame_probe
        assert frame_probe["frame_pose_integrated_delta_from_source_end"] == (
            0.0,
            0.0,
            0.0,
        ), frame_probe
        assert frame_probe["frame_pose_integrated_error_u"] == 0.0, frame_probe
        assert frame_probe["frame_pose_motion_consistency_verdict"] == (
            "frame_pose_matches_velocity_integration"
        ), frame_probe
        assert frame_probe["bucket_center_sample_count"] == 30, frame_probe
        assert frame_probe["accepted_count"] == 1, frame_probe
        accepted = frame_probe["first_accepted"]
        assert accepted["label"] == "pre_to_current_50", accepted
        assert accepted["server_report_time_s"] == 0.5, accepted
        assert accepted["bucket_index"] == 15, accepted
        assert accepted["contact_selection"] == "cbsp_mesh_edge_terrain_plane_probe", accepted
        assert accepted["contact"]["contact_cell"] == (7, 8), accepted
        assert accepted["contact"]["contact_cbsp_mesh_triangle_indices"] == (6, 8, 9), accepted
        preview = accepted["response_preview"]
        assert preview["runtime_default"] == "off", preview
        assert preview["apply_enabled"] is False, preview
        assert preview["preserved_position"] is True, preview
        assert preview["label"] == "pre_to_current_50", preview
        assert preview["pos"] == accepted["pos"], preview
        assert preview["velocity_before"] == accepted["velocity"], preview
        assert "velocity_delta" in preview, preview
        assert "angular_delta" in preview, preview
        resolve_preview = accepted["resolve_phase_preview"]
        assert resolve_preview["runtime_default"] == "off", resolve_preview
        assert resolve_preview["bucket_order_only"] is True, resolve_preview
        assert resolve_preview["bucket_center_probe_is_diagnostic"] is True, resolve_preview
        assert resolve_preview["collision_time_s"] == 0.5, resolve_preview
        assert resolve_preview["frame_dt_s"] == 1.0, resolve_preview
        assert resolve_preview["collision_pair_bucket"] == 15, resolve_preview
        assert resolve_preview["remaining_after_resolve_s"] == 0.5, resolve_preview
        assert resolve_preview["collision_time_pose"] == accepted["pos"], resolve_preview
        assert resolve_preview["resolve_retest_pose"] == (1.0, 0.0, -1.0), resolve_preview
        assert resolve_preview["resolve_retest_accept"] is False, resolve_preview
        assert resolve_preview["response_preview_applied"] is True, resolve_preview
        assert "post_response_remaining_endpoint_pos" in resolve_preview, resolve_preview
        assert "remaining_endpoint_without_response_pos" in resolve_preview, resolve_preview
        assert "full frame_dt" in resolve_preview["decompile_source"], resolve_preview
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}
        bucket_row = frame_probe["results"]["pre_to_current_bucket_06_center"][
            "cbsp_mesh_edge_terrain_plane_probe"
        ]
        assert bucket_row["bucket_index"] == 6, bucket_row
        assert 0.216 <= bucket_row["velocity_fraction"] <= 0.217, bucket_row
        assert bucket_row["velocity_source"] == "pre_to_current_bucket_center_fraction", bucket_row
        assert bucket_row["accepted"] is False, bucket_row
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_frame_phase_report_first_probe_is_read_only: PASSED"
    )
    return True


def test_entity_world_collision_spatial_ref_schedule_probe_reports_without_applying():
    """Spatial-reference schedule probe should report from ref pose without response."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN_STEPS",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SCHEDULE_PROBE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SPATIAL_REF_SCHEDULE_PROBE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN_STEPS"] = "9"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SCHEDULE_PROBE"] = "0"
        os.environ[
            "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SPATIAL_REF_SCHEDULE_PROBE"
        ] = "1"
        os.environ[
            "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"
        ] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        def fake_model_collision(center, *args, **kwargs):
            if 0.45 <= center[0] <= 0.55:
                return TerrainContact(
                    position=(center[0], 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (1.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.5, 0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        result = server._resolve_entity_world_collision(
            ctx,
            1.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            -1.0,
            pre_pos=(1.0, 0.0, 0.0),
            pre_vel=(0.0, 0.0, -1.0),
            dt=1.0,
        )

        assert result == (1.0, 0.0, -1.0, 0.0, 0.0, -1.0), result
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_schedule_probe_enabled"] is False, probe
        assert probe["pair_record_spatial_ref_schedule_probe_enabled"] is True, probe
        assert probe["pair_record_spatial_ref_schedule_probe_result"] == (
            "interval_contact"
        ), probe
        assert probe["pair_record_spatial_ref_schedule_collision_time_s"] == 0.0, probe
        assert probe["pair_record_spatial_ref_schedule_decompile_pool_model"] == (
            "CollisionPairPool_30_bucket_spatial_ref"
        ), probe
        assert probe["pair_record_spatial_ref_schedule_step_dt_s"] == 1.0, probe
        assert probe["pair_record_spatial_ref_schedule_bucket_count"] == 30, probe
        assert probe["pair_record_spatial_ref_schedule_bucket_index"] == 0, probe
        assert probe["pair_record_spatial_ref_schedule_ref_pos"] == (
            0.5,
            0.0,
            0.0,
        ), probe
        assert probe["pair_record_spatial_ref_schedule_current_pos"] == (
            1.0,
            0.0,
            -1.0,
        ), probe
        assert 1.11 <= probe[
            "pair_record_spatial_ref_schedule_ref_to_current_distance"
        ] <= 1.12, probe
        assert probe[
            "pair_record_spatial_ref_schedule_ref_to_current_z_delta"
        ] == -1.0, probe
        assert probe["pair_record_spatial_ref_schedule_contact"]["contact_cell"] == (
            7,
            8,
        ), probe
        assert "pair_record_schedule_probe_result" not in probe, probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_spatial_ref_schedule_probe_reports_without_applying: PASSED"
    )
    return True


def test_entity_world_collision_pair_record_schedule_response_probe_is_read_only():
    """Schedule response probe should simulate the guarded response without applying it."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN_STEPS",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SCHEDULE_PROBE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SCHEDULE_RESPONSE_PROBE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN_STEPS"] = "9"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTINUE_REMAINING"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SCHEDULE_PROBE"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_SCHEDULE_RESPONSE_PROBE"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA"] = "3.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA"] = "1.0"

        def fake_model_collision(center, *args, **kwargs):
            if 0.45 <= center[0] <= 0.55:
                return TerrainContact(
                    position=(center[0], 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)
        ctx.spring_body_ang_vel = (0.12, -0.08)
        ctx.angular_vel_yaw = 0.04
        ctx.rigid_body_last_interp_tick = 77

        result = server._resolve_entity_world_collision(
            ctx,
            1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            -1.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(1.0, 0.0, -1.0),
            dt=1.0,
        )

        assert result == (1.0, 0.0, -1.0, 1.0, 0.0, -1.0), result
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}
        assert ctx.spring_body_ang_vel == (0.12, -0.08), ctx.spring_body_ang_vel
        assert ctx.angular_vel_yaw == 0.04, ctx.angular_vel_yaw
        assert ctx.rigid_body_last_interp_tick == 77, ctx.rigid_body_last_interp_tick
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_schedule_probe_result"] == "interval_contact", probe
        assert probe["pair_record_schedule_response_probe_enabled"] is True, probe
        assert probe["pair_record_schedule_response_probe_result"] == "applied", probe
        assert probe["pair_record_schedule_response_probe_applied"] is True, probe
        assert probe["pair_record_schedule_response_probe_vel_before"] == (
            1.0,
            0.0,
            -1.0,
        ), probe
        assert probe["pair_record_schedule_response_probe_contact_pos"] == (
            0.45,
            0.0,
            -0.45,
        ), probe
        assert probe["pair_record_schedule_response_probe_post_contact_vel"][2] > -1.0, probe
        assert probe["pair_record_schedule_response_probe_endpoint_vel"][2] > -1.0, probe
        assert (
            probe[
                "pair_record_schedule_response_probe_velocity_delta_after_safety"
            ][2]
            > 0.0
        ), probe
        assert probe["pair_record_schedule_response_probe_angular_preserved"] is True, probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_pair_record_schedule_response_probe_is_read_only: PASSED"
    )
    return True


def test_entity_world_collision_phase_lookahead_probe_reports_future_pair_contact():
    """Default-off phase lookahead should report a near-future pair lane without moving the tank."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_TIME",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_STEPS",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_DISTANCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD"] = "probe"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_TIME"] = "0.1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_STEPS"] = "5"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_DISTANCE"] = "2.0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        calls = []

        def fake_model_collision(center, *args, **kwargs):
            calls.append((tuple(center), kwargs.get("contact_selection")))
            if (
                kwargs.get("contact_selection") == "upward_min_depth"
                and 0.49 <= float(center[0]) <= 0.61
                and float(center[2]) < 0.0
            ):
                return TerrainContact(
                    position=(float(center[0]), 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(71, 56),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, -1.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, -1.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        result = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            -1.0,
            10.0,
            0.0,
            0.0,
            pre_pos=(-0.333333, 0.0, -1.0),
            pre_vel=(10.0, 0.0, 0.0),
            dt=1.0 / 30.0,
        )

        assert result == (0.0, 0.0, -1.0, 10.0, 0.0, 0.0), result
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_phase_lookahead_enabled"] is True, probe
        assert probe["pair_record_phase_lookahead_apply_enabled"] is False, probe
        assert probe["pair_record_phase_lookahead_reject"] == "", probe
        assert 0.049 <= probe["pair_record_phase_lookahead_collision_time_s"] <= 0.061, probe
        assert probe["pair_record_phase_lookahead_distance"] < 0.7, probe
        assert probe["pair_record_phase_lookahead_contact"]["contact_cell"] == (71, 56), probe
        assert any(call[1] == "upward_min_depth" for call in calls), calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_phase_lookahead_probe_reports_future_pair_contact: PASSED"
    )
    return True


def test_entity_world_collision_phase_lookahead_contact_applies_without_future_teleport():
    """Apply mode uses the future contact as an impulse source while preserving current position."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_TIME",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_STEPS",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_DISTANCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD"] = "apply"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_TIME"] = "0.1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_STEPS"] = "5"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_DISTANCE"] = "2.0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        def fake_model_collision(center, *args, **kwargs):
            if (
                kwargs.get("contact_selection") == "upward_min_depth"
                and 0.49 <= float(center[0]) <= 0.61
                and float(center[2]) < 0.0
            ):
                return TerrainContact(
                    position=(float(center[0]), 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(71, 56),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, -1.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, -1.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            -1.0,
            10.0,
            0.0,
            -1.0,
            pre_pos=(-0.333333, 0.0, -1.0),
            pre_vel=(10.0, 0.0, -1.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_phase_lookahead_pair_record_contact", debug
        assert debug["phase_lookahead_pair_record_contact"] is True, debug
        assert debug["pair_record_phase_lookahead_preserved_position"] is True, debug
        assert 0.049 <= debug["pair_record_phase_lookahead_collision_time_s"] <= 0.061, debug
        assert debug["pair_record_phase_lookahead_current_pos"] == (0.0, 0.0, -1.0), debug
        assert abs(px) < 1e-6 and abs(py) < 1e-6 and abs(pz + 1.0) < 1e-6, (px, py, pz, debug)
        assert vx == debug["velocity_after"][0], debug
        assert vz > -1.0, (vx, vy, vz, debug)
        assert debug["pair_record_phase_lookahead_velocity_delta"][2] > 0.0, debug
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_phase_lookahead_contact_applies_without_future_teleport: PASSED"
    )
    return True


def test_entity_world_collision_phase_lookahead_queue_resolves_when_due():
    """Queue mode stores a future pair contact and resolves it in a later frame."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_TIME",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_STEPS",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_DISTANCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD"] = "queue"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_TIME"] = "0.1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_STEPS"] = "5"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_LOOKAHEAD_MAX_DISTANCE"] = "2.0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        def fake_model_collision(center, *args, **kwargs):
            if (
                kwargs.get("contact_selection") == "upward_min_depth"
                and 0.49 <= float(center[0]) <= 0.61
                and float(center[2]) < 0.0
            ):
                return TerrainContact(
                    position=(float(center[0]), 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(71, 56),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, -1.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, -1.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        result = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            -1.0,
            10.0,
            0.0,
            -1.0,
            pre_pos=(-0.333333, 0.0, -1.0),
            pre_vel=(10.0, 0.0, -1.0),
            dt=1.0 / 30.0,
        )
        assert result == (0.0, 0.0, -1.0, 10.0, 0.0, -1.0), result
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}
        first_probe = ctx.debug_last_terrain_contact_probe
        assert first_probe["pair_record_phase_lookahead_queue_enabled"] is True, first_probe
        assert first_probe["pair_record_phase_lookahead_queue_stored"] is True, first_probe
        queued = getattr(ctx, "terrain_pair_record_phase_lookahead_queue")
        assert 0.049 <= queued["time_to_contact_s"] <= 0.061, queued

        result = server._resolve_entity_world_collision(
            ctx,
            0.333333,
            0.0,
            -1.033333,
            10.0,
            0.0,
            -1.0,
            pre_pos=(0.0, 0.0, -1.0),
            pre_vel=(10.0, 0.0, -1.0),
            dt=1.0 / 30.0,
        )
        assert result == (0.333333, 0.0, -1.033333, 10.0, 0.0, -1.0), result
        pending_probe = ctx.debug_last_terrain_contact_probe
        assert pending_probe["pair_record_phase_lookahead_queue_pending"] is True, pending_probe
        queued = getattr(ctx, "terrain_pair_record_phase_lookahead_queue")
        assert 0.01 <= queued["time_to_contact_s"] <= 0.03, queued
        assert queued["age_steps"] == 1, queued

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.666666,
            0.0,
            -1.066666,
            10.0,
            0.0,
            -1.0,
            pre_pos=(0.333333, 0.0, -1.033333),
            pre_vel=(10.0, 0.0, -1.0),
            dt=1.0 / 30.0,
        )
        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_phase_lookahead_queued_pair_record_contact", debug
        assert debug["phase_lookahead_queued_pair_record_contact"] is True, debug
        assert debug["pair_record_phase_lookahead_queue_enabled"] is True, debug
        assert 0.01 <= debug["pair_record_phase_lookahead_queued_collision_time_s"] <= 0.03, debug
        assert debug["pair_record_phase_lookahead_queued_contact_pos"][0] > 0.49, debug
        assert vx == debug["velocity_after"][0], debug
        assert vz > -1.0, (px, py, pz, vx, vy, vz, debug)
        assert getattr(ctx, "terrain_pair_record_phase_lookahead_queue") is None
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_phase_lookahead_queue_resolves_when_due: PASSED"
    )
    return True


def test_entity_world_collision_phase_backtrack_probe_reports_prior_pair_contact():
    """Default-off phase backtrack should report a recently passed pair lane without moving the tank."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_MAX_TIME",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_STEPS",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_MAX_DISTANCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK"] = "probe"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_MAX_TIME"] = "0.1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_STEPS"] = "5"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_MAX_DISTANCE"] = "2.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_SOURCE"] = "pre"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"

        calls = []

        def fake_model_collision(center, *args, **kwargs):
            calls.append((tuple(center), kwargs.get("contact_selection")))
            if (
                kwargs.get("contact_selection") == "upward_min_depth"
                and -0.61 <= float(center[0]) <= -0.49
                and float(center[2]) < 0.0
            ):
                return TerrainContact(
                    position=(float(center[0]), 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(71, 56),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, -1.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, -1.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        result = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            -1.0,
            10.0,
            0.0,
            0.0,
            pre_pos=(0.0, 0.0, -1.0),
            pre_vel=(10.0, 0.0, 0.0),
            dt=1.0 / 30.0,
        )

        assert result == (0.0, 0.0, -1.0, 10.0, 0.0, 0.0), result
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_phase_backtrack_enabled"] is True, probe
        assert probe["pair_record_phase_backtrack_reject"] == "", probe
        assert 0.049 <= probe["pair_record_phase_backtrack_time_s"] <= 0.061, probe
        assert probe["pair_record_phase_backtrack_distance"] < 0.7, probe
        assert probe["pair_record_phase_backtrack_source"] == "pre", probe
        assert probe["pair_record_phase_backtrack_contact"]["contact_cell"] == (71, 56), probe
        assert any(call[1] == "upward_min_depth" for call in calls), calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_phase_backtrack_probe_reports_prior_pair_contact: PASSED"
    )
    return True


def test_entity_world_collision_phase_backtrack_contact_replays_to_endpoint():
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_MAX_TIME",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_STEPS",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_MAX_DISTANCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK"] = "apply"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_MAX_TIME"] = "0.1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_STEPS"] = "5"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_MAX_DISTANCE"] = "2.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_PHASE_BACKTRACK_SOURCE"] = "current"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE"] = "component"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA"] = "4.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA"] = "2.0"

        def fake_model_collision(center, *args, **kwargs):
            if (
                kwargs.get("contact_selection") == "upward_min_depth"
                and -0.61 <= float(center[0]) <= -0.49
                and float(center[2]) < 0.0
            ):
                return TerrainContact(
                    position=(float(center[0]), 0.0, -2.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=2,
                    cell=(71, 56),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, -1.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, -1.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        result = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            -1.0,
            10.0,
            0.0,
            -1.0,
            pre_pos=(0.0, 0.0, -1.0),
            pre_vel=(10.0, 0.0, -1.0),
            dt=1.0 / 30.0,
        )

        assert result[0] > -0.05, result
        assert result[5] > -1.0, result
        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_phase_backtrack_pair_record_contact", debug
        assert debug["phase_backtrack_pair_record_contact"] is True, debug
        assert debug["pair_record_phase_backtrack_apply_enabled"] is True, debug
        assert debug["pair_record_phase_backtrack_source"] == "current", debug
        assert debug["pair_record_phase_backtrack_replayed_position"] is True, debug
        assert debug["pair_record_phase_backtrack_contact_pos"][0] < -0.49, debug
        assert debug["pair_record_phase_backtrack_endpoint_pos"][0] > -0.05, debug
        assert debug["contact_events"][0]["phase_backtrack_pair_record_contact"] is True, debug
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_phase_backtrack_apply_enabled"] is True, probe
        assert probe["pair_record_phase_backtrack_reject"] == "", probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_phase_backtrack_contact_replays_to_endpoint: PASSED"
    )
    return True


def test_entity_world_collision_pair_record_contact_applies_decompile_face_gated_contact():
    """Default pair-record contact should activate only on raw contacts carrying terrain-face evidence."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA"] = "3.0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA", None)
        terrain_face_normal = (0.25, 0.0, 0.9682458365518543)
        calls = []

        def fake_model_collision(center, *args, **kwargs):
            calls.append((center, kwargs.get("contact_selection")))
            if abs(center[2] - 10.0) < 1e-6:
                return TerrainContact(
                    position=(1.0, 2.0, 3.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=4.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=terrain_face_normal,
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.6, 0.8, 0.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        _px, _py, pz, _vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            -10.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(0.0, 0.0, -10.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_pair_record_contact", debug
        assert debug["pair_record_contact"] is True, debug
        assert debug["pair_record_contact_reject"] == "", debug
        assert debug["pair_record_contact_selection"] == "upward_min_depth", debug
        assert debug["pair_record_contact_normal_source"] == "mesh", debug
        assert debug["pair_record_solver_normal_source"] == "entity_cbsp_split", debug
        assert (
            debug["pair_record_contact_delta_normal_source"]
            == "entity_radial_terrain_face_blend"
        ), debug
        assert debug["pair_record_delta_normal_source"] == (
            "entity_radial_terrain_face_blend"
        ), debug
        assert debug["pair_record_terrain_face_normal"] == terrain_face_normal, debug
        assert debug["raw_origin_fallback_delta_mode"] == "closing_velocity", debug
        assert debug["raw_origin_fallback_delta_normal_source"] == (
            "entity_radial_terrain_face_blend"
        ), debug
        assert debug["raw_origin_fallback_angular_mode"] == "preserve", debug
        assert debug["raw_origin_fallback_vertical_delta_mode"] == "scale", debug
        assert debug["raw_origin_fallback_angular_preserved"] is True, debug
        assert debug["raw_origin_fallback_velocity_delta_mag_after_safety"] <= 3.000001, debug
        assert debug["raw_origin_fallback_velocity_safety_max_vertical_delta"] == 1.0, debug
        assert debug["raw_origin_fallback_velocity_delta_after_safety"][2] <= 1.000001, debug
        assert pz > 10.0 and vz > -10.0, (pz, vz)
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_contact_enabled"] is True, probe
        assert probe["pair_record_contact_reject"] == "", probe
        assert probe["pair_record_contact_selection"] == "upward_min_depth", probe
        assert probe["pair_record_contact_delta_normal_source"] == (
            "entity_radial_terrain_face_blend"
        ), probe
        assert probe["pair_record_contact_max_vertical_delta"] == 1.0, probe
        assert probe["pair_record_contact"]["contact_terrain_face_normal"] == terrain_face_normal, probe
        assert calls[0] == ((0.0, 0.0, 13.0), "first")
        assert calls[1] == ((0.0, 0.0, 10.0), "first")
        assert calls[2] == ((0.0, 0.0, 10.0), "upward_min_depth")
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_entity_world_collision_pair_record_contact_applies_decompile_face_gated_contact: PASSED")
    return True


def test_entity_world_collision_pair_record_can_probe_raw_solver_linear_response():
    """Default-off pair-record response profile should preserve raw solver linear delta."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_RESPONSE_PROFILE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_RESPONSE_PROFILE"] = (
            "decompile_linear_solver"
        )
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = (
            "upward_min_depth"
        )
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA", None)

        def fake_model_collision(center, *args, **kwargs):
            if abs(center[2] - 10.0) >= 1e-6:
                return None
            return TerrainContact(
                position=(1.0, 2.0, 3.0),
                normal=(0.0, 0.0, 1.0),
                penetration=4.0,
                sector_index=2,
                cell=(71, 56),
                normal_source="entity_cbsp_split",
                cbsp_split_normal=(0.0, 0.0, 1.0),
                terrain_face_normal=(0.25, 0.0, 0.9682458365518543),
                mesh_face_normal=(0.0, 0.0, 1.0),
                entity_radial_normal=(0.6, 0.8, 0.0),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        _px, _py, _pz, _vx, _vy, _vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            -10.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(0.0, 0.0, -10.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_pair_record_contact", debug
        assert debug["pair_record_contact_response_profile"] == (
            "decompile_linear_solver"
        ), debug
        assert debug["raw_origin_fallback_delta_mode"] == "solver_vector", debug
        assert debug["raw_origin_fallback_velocity_safety_max_delta"] == 0.0, debug
        assert debug["raw_origin_fallback_velocity_safety_max_vertical_delta"] == 0.0, debug
        assert debug["raw_origin_fallback_velocity_delta_clamped"] is False, debug
        assert debug["raw_origin_fallback_vertical_delta_clamped"] is False, debug
        assert debug["raw_origin_fallback_angular_preserved"] is True, debug
        assert (
            debug["raw_origin_fallback_velocity_delta_after_safety"]
            == debug["raw_origin_fallback_velocity_delta_unclamped"]
        ), debug
        assert (
            debug["raw_origin_fallback_solver_velocity_delta_unprojected"]
            == debug["raw_origin_fallback_velocity_delta_unclamped"]
        ), debug
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_contact_response_profile"] == (
            "decompile_linear_solver"
        ), probe
        assert probe["pair_record_contact_max_velocity_delta"] == 0.0, probe
        assert probe["pair_record_contact_max_vertical_delta"] == 0.0, probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_pair_record_can_probe_raw_solver_linear_response: PASSED"
    )
    return True


def test_entity_world_collision_pair_record_contact_uses_shallow_upward_selection():
    """Pair-record contact selection should skip a deep side split for a shallow upward hit."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ[
            "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"
        ] = "entity_radial_terrain_face_forward_up"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA"] = "3.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA"] = "1.0"
        deep_face_normal = (-0.743293, 0.0, 0.668966)
        shallow_face_normal = (0.0, 0.2, 0.9797958971132712)
        calls = []

        def fake_model_collision(center, *args, **kwargs):
            selection = kwargs.get("contact_selection")
            calls.append((center, selection))
            if abs(center[2] - 10.0) >= 1e-6:
                return None
            if selection == "upward_min_depth":
                return TerrainContact(
                    position=(1.0, 2.0, 3.0),
                    normal=(0.0, 0.2, 0.9797958971132712),
                    penetration=4.0,
                    sector_index=2,
                    cell=(71, 56),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.2, 0.9797958971132712),
                    terrain_face_normal=shallow_face_normal,
                    mesh_face_normal=(0.0, 0.2, 0.9797958971132712),
                    entity_radial_normal=(0.2, 0.0, 0.9797958971132712),
                )
            return TerrainContact(
                position=(0.0, 0.0, 0.0),
                normal=deep_face_normal,
                penetration=29.9,
                sector_index=2,
                cell=(71, 56),
                normal_source="entity_cbsp_split",
                cbsp_split_normal=deep_face_normal,
                terrain_face_normal=deep_face_normal,
                mesh_face_normal=deep_face_normal,
                entity_radial_normal=(0.95, 0.0, 0.3122498999199199),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        _px, _py, pz, _vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            -10.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(0.0, 0.0, -10.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_pair_record_contact", debug
        assert debug["depth"] == 4.0, debug
        assert debug["pair_record_contact_selection"] == "upward_min_depth", debug
        assert debug["pair_record_raw_normal"] == deep_face_normal, debug
        assert debug["pair_record_selected_raw_normal"] == (
            0.0,
            0.2,
            0.9797958971132712,
        ), debug
        assert debug["pair_record_terrain_face_normal"] == shallow_face_normal, debug
        assert debug["raw_origin_fallback_velocity_delta_after_safety"][2] <= 1.000001, debug
        assert pz > 10.0 and vz > -10.0, (pz, vz)
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["raw_origin_contact"]["depth"] == 29.9, probe
        assert probe["pair_record_raw_contact"]["depth"] == 29.9, probe
        assert probe["pair_record_selected_raw_contact"]["depth"] == 4.0, probe
        assert probe["pair_record_selected_raw_bounds_contact"] is None, probe
        assert probe["pair_record_selected_raw_error"] is None, probe
        assert probe["pair_record_selected_pair_contact_source"] == "entity_cbsp_split", probe
        assert probe["pair_record_contact"]["depth"] == 4.0, probe
        assert probe["pair_record_contact_selection"] == "upward_min_depth", probe
        assert probe["pair_record_contact_reject"] == "", probe
        assert calls[0] == ((0.0, 0.0, 13.0), "first")
        assert calls[1] == ((0.0, 0.0, 10.0), "first")
        assert calls[2] == ((0.0, 0.0, 10.0), "upward_min_depth")
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_entity_world_collision_pair_record_contact_uses_shallow_upward_selection: PASSED")
    return True


def test_entity_world_collision_pair_record_contact_accepts_og_straightaway_face():
    """Empirical straightaway contact should pass the default pair-record gates."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_DEPTH",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MIN_FACE_NORMAL_Z",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_DEPTH", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MIN_FACE_NORMAL_Z", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA", None)
        terrain_face_normal = (-0.900996, -0.000522, 0.433826)
        mesh_normal = (-0.061263, 0.294716, 0.95362)
        calls = []

        def fake_model_collision(center, *args, **kwargs):
            calls.append((center, kwargs.get("contact_selection")))
            if abs(center[2] - 10.0) >= 1e-6:
                return None
            return TerrainContact(
                position=(2.0, -1.0, 8.0),
                normal=mesh_normal,
                penetration=8.541382,
                sector_index=7,
                cell=(120, 70),
                normal_source="entity_cbsp_split",
                cbsp_split_normal=mesh_normal,
                terrain_face_normal=terrain_face_normal,
                mesh_face_normal=mesh_normal,
                entity_radial_normal=(0.132776, 0.951725, 0.276748),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        _px, _py, _pz, vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            24.0,
            0.0,
            -0.02,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(24.0, 0.0, -0.02),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_pair_record_contact", debug
        assert debug["pair_record_contact"] is True, debug
        assert debug["pair_record_contact_reject"] == "", debug
        assert debug["pair_record_contact_max_depth"] == 10.0, debug
        assert debug["pair_record_contact_min_face_normal_z"] == 0.4, debug
        assert debug["pair_record_terrain_face_normal"] == terrain_face_normal, debug
        assert debug["depth"] == 8.541382, debug
        assert debug["raw_origin_fallback_velocity_delta_mag_after_safety"] > 0.0, debug
        assert vx < 24.0 and vz > -0.02, (vx, vz, debug)
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_contact_reject"] == "", probe
        assert calls[0] == ((0.0, 0.0, 13.0), "first")
        assert calls[1] == ((0.0, 0.0, 10.0), "first")
        assert calls[2] == ((0.0, 0.0, 10.0), "upward_min_depth")
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_pair_record_contact_accepts_og_straightaway_face: PASSED"
    )
    return True


def test_tank_clean_terrain_contact_uses_pair_solver_by_default():
    """Tank auto-response should not use the legacy projection path for clean terrain hits."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE",
        "WULFRAM_TANK_CLEAN_TERRAIN_PAIR_SOLVER",
        "WULFRAM_ENTITY_TERRAIN_PROJECTION_ORDER",
        "WULFRAM_TANK_CLEAN_TERRAIN_PROJECTION_ORDER",
        "WULFRAM_ENTITY_CONTACT_POSITION_CORRECTION_CAP",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        os.environ.pop("WULFRAM_TANK_CLEAN_TERRAIN_PAIR_SOLVER", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PROJECTION_ORDER", None)
        os.environ.pop("WULFRAM_TANK_CLEAN_TERRAIN_PROJECTION_ORDER", None)
        os.environ.pop("WULFRAM_ENTITY_CONTACT_POSITION_CORRECTION_CAP", None)
        terrain_face_normal = (-0.900996, -0.000522, 0.433826)
        contact_normal = (-0.901, -0.0005, 0.4338)

        def fake_model_collision(center, *args, **kwargs):
            if abs(center[2] - 13.0) >= 1e-6:
                return None
            return TerrainContact(
                position=(7.0, 0.0, 4.0),
                normal=contact_normal,
                penetration=8.541382,
                sector_index=7,
                cell=(120, 70),
                normal_source="entity_cbsp_split",
                cbsp_split_normal=contact_normal,
                terrain_face_normal=terrain_face_normal,
                mesh_face_normal=contact_normal,
                entity_radial_normal=(0.9, 0.0, 0.4338),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        px, _py, pz, vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            24.0,
            0.0,
            -0.02,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(24.0, 0.0, -0.02),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_clean_contact", debug
        assert debug["response"] == "terrain_contact_constraint_solver", debug
        assert debug["constraint_projection_order"] == "opposite_if_separating", debug
        assert debug["position_correction"] <= 0.005001, debug
        assert debug["contact_terrain_face_normal"] == terrain_face_normal, debug
        assert px < 0.0 and pz > 10.0, (px, pz, debug)
        assert vx < 24.0 and vz > -0.02, (vx, vz, debug)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_tank_clean_terrain_contact_uses_pair_solver_by_default: PASSED")
    return True


def test_tank_clean_terrain_projection_order_uses_opposite_probe_by_default():
    """Tank clean contacts should activate the OG opposite projection when needed."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE",
        "WULFRAM_TANK_CLEAN_TERRAIN_PAIR_SOLVER",
        "WULFRAM_ENTITY_TERRAIN_PROJECTION_ORDER",
        "WULFRAM_TANK_CLEAN_TERRAIN_PROJECTION_ORDER",
        "WULFRAM_ENTITY_CONTACT_POSITION_CORRECTION_CAP",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        os.environ.pop("WULFRAM_TANK_CLEAN_TERRAIN_PAIR_SOLVER", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PROJECTION_ORDER", None)
        os.environ.pop("WULFRAM_TANK_CLEAN_TERRAIN_PROJECTION_ORDER", None)
        os.environ.pop("WULFRAM_ENTITY_CONTACT_POSITION_CORRECTION_CAP", None)

        def fake_model_collision(center, *args, **kwargs):
            if abs(center[2] - 13.0) >= 1e-6:
                return None
            return TerrainContact(
                position=(0.0, 0.0, 9.0),
                normal=(0.0, 0.0, 1.0),
                penetration=4.0,
                sector_index=2,
                cell=(7, 8),
                normal_source="entity_cbsp_split",
                cbsp_split_normal=(0.0, 0.0, 1.0),
                terrain_face_normal=(0.0, 0.0, 1.0),
                mesh_face_normal=(0.0, 0.0, 1.0),
                entity_radial_normal=(0.0, 0.0, 1.0),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        _px, _py, _pz, _vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            0.006,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(0.0, 0.0, 0.006),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_clean_contact", debug
        assert debug["response"] == "terrain_contact_constraint_solver", debug
        assert debug["constraint_projection_order"] == "opposite_if_separating", debug
        assert (
            debug["constraint_primary_projection_speed_source"]
            == "world_minus_body_if_body_separating"
        ), debug
        assert debug["normal_iterations"] > 0, debug
        assert debug["normal_impulse"] > 0.0, debug
        assert vz > 0.006, (vz, debug)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_tank_clean_terrain_projection_order_uses_opposite_probe_by_default: PASSED"
    )
    return True


def test_tank_default_raw_origin_fallback_uses_empirical_straightaway_face():
    """Default tank auto mode should catch the OG straightaway raw-origin face contact."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_TANK_RAW_ORIGIN_FALLBACK",
        "WULFRAM_TANK_RAW_ORIGIN_NORMAL_SOURCE",
        "WULFRAM_TANK_RAW_ORIGIN_MAX_DEPTH",
        "WULFRAM_TANK_RAW_ORIGIN_MIN_NORMAL_Z",
        "WULFRAM_TANK_RAW_ORIGIN_MIN_FACE_NORMAL_Z",
        "WULFRAM_TANK_RAW_ORIGIN_MAX_VELOCITY_DELTA",
        "WULFRAM_TANK_RAW_ORIGIN_MAX_VERTICAL_DELTA",
        "WULFRAM_TANK_RAW_ORIGIN_DELTA_MODE",
        "WULFRAM_TANK_RAW_ORIGIN_DELTA_NORMAL",
        "WULFRAM_TANK_RAW_ORIGIN_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        for key in env_keys:
            os.environ.pop(key, None)
        terrain_face_normal = (-0.900996, -0.000522, 0.433826)

        def fake_model_collision(center, *args, **kwargs):
            if abs(center[2] - 3.21) >= 1e-6:
                return None
            return TerrainContact(
                position=(5251.66, 3071.04, 3.41),
                normal=(0.0, -1.0, 0.0),
                penetration=8.541382,
                sector_index=7,
                cell=(120, 70),
                normal_source="entity_cbsp_split",
                cbsp_split_normal=(0.0, -1.0, 0.0),
                terrain_face_normal=terrain_face_normal,
                mesh_face_normal=(0.0, 1.0, 0.0),
                entity_radial_normal=(0.951729, -0.3053, 0.031683),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (5245.68, 3072.96, 3.21)
        ctx.world_collision_ref_pos = (5245.68, 3072.96, 3.21)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)
        ctx.spring_body_ang_vel = (0.12, -0.08)

        _px, _py, _pz, vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            5245.68,
            3072.96,
            3.21,
            26.4,
            0.0,
            -0.01,
            pre_pos=(5244.8, 3072.96, 3.21),
            pre_vel=(26.3, 0.0, -0.01),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_raw_origin_fallback_contact", debug
        assert debug["tank_raw_origin_fallback"] is True, debug
        assert debug["contact_normal_source"] == "terrain_triangle_contact_face", debug
        assert debug["raw_origin_fallback_normal_source"] == "terrain_face", debug
        assert debug["raw_origin_fallback_delta_mode"] == "closing_velocity", debug
        assert debug["tank_raw_origin_fallback_delta_normal_mode"] == "horizontal_face", debug
        assert debug["raw_origin_fallback_delta_normal"][2] == 0.0, debug
        assert debug["raw_origin_fallback_delta_normal_source"] == "terrain_triangle_contact_face_horizontal", debug
        assert debug["raw_origin_fallback_angular_mode"] == "preserve", debug
        assert debug["raw_origin_fallback_angular_preserved"] is True, debug
        assert debug["raw_origin_fallback_reject"] == "", debug
        assert debug["raw_origin_fallback_velocity_delta_mag_after_safety"] <= 18.001, debug
        assert vx < 26.4 and abs(vz + 0.01) < 1e-6, (vx, vz, debug)
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["tank_raw_origin_fallback"] is True, probe
        assert probe["raw_origin_fallback_enabled"] is True, probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_tank_default_raw_origin_fallback_uses_empirical_straightaway_face: PASSED"
    )
    return True


def test_tank_clean_side_contact_uses_face_fallback_without_angular_blowup():
    """Lifted side-split contacts should use the guarded face-normal tank path."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE",
        "WULFRAM_TANK_CLEAN_TERRAIN_FACE_FALLBACK",
        "WULFRAM_TANK_CLEAN_TERRAIN_FACE_FALLBACK_MAX_CONTACT_NORMAL_Z",
        "WULFRAM_TANK_CLEAN_TERRAIN_FACE_FALLBACK_MIN_FACE_NORMAL_Z",
        "WULFRAM_TANK_CLEAN_TERRAIN_FACE_FALLBACK_DELTA_MODE",
        "WULFRAM_TANK_CLEAN_TERRAIN_FACE_FALLBACK_DELTA_NORMAL",
        "WULFRAM_TANK_FACE_FALLBACK_LATCH",
        "WULFRAM_TANK_RAW_ORIGIN_NORMAL_SOURCE",
        "WULFRAM_TANK_RAW_ORIGIN_MAX_VELOCITY_DELTA",
        "WULFRAM_TANK_RAW_ORIGIN_MAX_VERTICAL_DELTA",
        "WULFRAM_TANK_RAW_ORIGIN_DELTA_MODE",
        "WULFRAM_TANK_RAW_ORIGIN_ANGULAR_MODE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        for key in env_keys:
            os.environ.pop(key, None)
        terrain_face_normal = (-0.900996, -0.000522, 0.433826)

        def fake_model_collision(center, *args, **kwargs):
            if abs(center[2] - 6.21) >= 1e-6:
                return None
            return TerrainContact(
                position=(5253.0156, 3071.0239, 6.2153),
                normal=(0.0, -1.0, 0.0),
                penetration=8.524048,
                sector_index=7,
                cell=(120, 70),
                normal_source="entity_cbsp_split",
                cbsp_split_normal=(0.0, -1.0, 0.0),
                terrain_face_normal=terrain_face_normal,
                mesh_face_normal=(0.0, 1.0, 0.0),
                entity_radial_normal=(0.95433, -0.298151, 0.018981),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (5246.72, 3072.954, 3.21)
        ctx.world_collision_ref_pos = ctx.player_pos
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)
        ctx.spring_body_ang_vel = (0.02, -0.82)
        ctx.angular_vel_yaw = 0.0

        _px, _py, _pz, vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            5246.72,
            3072.954,
            3.21,
            3.26,
            -0.19,
            -6.35,
            pre_pos=(5246.6, 3072.96, 3.4),
            pre_vel=(3.26, -0.19, -6.35),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_clean_face_fallback_contact", debug
        assert debug["tank_clean_face_fallback"] is True, debug
        assert debug["clean_contact_original_normal"] == (0.0, -1.0, 0.0), debug
        assert debug["contact_normal_source"] == "terrain_triangle_contact_face", debug
        assert debug["raw_origin_fallback_delta_mode"] == "center_closing_velocity", debug
        assert debug["tank_clean_face_fallback_delta_normal_mode"] == "horizontal_face", debug
        assert debug["raw_origin_fallback_delta_normal"][2] == 0.0, debug
        assert debug["raw_origin_fallback_delta_normal_source"] == "terrain_triangle_contact_face_horizontal", debug
        assert debug["raw_origin_fallback_before_normal_speed_source"] == "center_velocity", debug
        assert debug["raw_origin_fallback_angular_mode"] == "preserve", debug
        assert debug["raw_origin_fallback_angular_preserved"] is True, debug
        assert debug["raw_origin_fallback_velocity_delta_mag_after_safety"] <= 18.001, debug
        assert abs(debug["angular_velocity_after"][0] - 0.02) < 1e-6, debug
        assert abs(debug["angular_velocity_after"][1] + 0.82) < 1e-6, debug
        assert vx < 3.26 and abs(vz + 6.35) < 1e-6, (vx, vz, debug)

        second = server._resolve_entity_world_collision(
            ctx,
            5246.72,
            3072.954,
            3.21,
            -4.0,
            -0.19,
            2.0,
            pre_pos=(5246.6, 3072.96, 3.4),
            pre_vel=(-4.0, -0.19, 2.0),
            dt=1.0 / 30.0,
        )
        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_clean_face_fallback_contact", debug
        assert debug["raw_origin_fallback_delta_mode"] == "center_closing_velocity", debug
        assert debug["raw_origin_fallback_delta_normal"][2] == 0.0, debug
        assert debug["raw_origin_fallback_before_normal_speed_source"] == "center_velocity", debug
        assert debug["raw_origin_fallback_before_normal_speed"] > 0.0, debug
        assert debug["raw_origin_fallback_normal_delta_skip_reason"] == "separating_before_velocity", debug
        assert debug["raw_origin_fallback_velocity_delta_mag_after_safety"] == 0.0, debug
        assert abs(second[3] + 4.0) < 1e-6, (second, debug)
        assert abs(second[5] - 2.0) < 1e-6, (second, debug)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_tank_clean_side_contact_uses_face_fallback_without_angular_blowup: PASSED"
    )
    return True


def test_tank_clean_terrain_contact_rejects_pathological_depth_by_default():
    """Deep lifted side contacts should not feed the unconstrained clean solver."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE",
        "WULFRAM_TANK_CLEAN_TERRAIN_MAX_DEPTH",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        os.environ.pop("WULFRAM_TANK_CLEAN_TERRAIN_MAX_DEPTH", None)

        def fake_model_collision(center, *args, **kwargs):
            if abs(center[2] - 13.0) >= 1e-6:
                return None
            return TerrainContact(
                position=(-3.0, 0.0, 10.0),
                normal=(-1.0, 0.0, 0.0),
                penetration=36.0,
                sector_index=7,
                cell=(122, 68),
                normal_source="entity_cbsp_split",
                terrain_face_normal=(0.91213, 0.0, 0.40989),
                mesh_face_normal=(-1.0, 0.0, 0.0),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        result = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            14.0,
            -1.0,
            -60.0,
            pre_pos=(0.0, 0.0, 12.0),
            pre_vel=(14.0, -1.0, -60.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_clean_contact", debug
        assert debug["response"] == "terrain_contact_constraint_solver_depth_rejected", debug
        assert debug["terrain_contact_depth_rejected"] is True, debug
        assert debug["terrain_contact_max_depth"] == 10.0, debug
        assert result == (0.0, 0.0, 10.0, 14.0, -1.0, -60.0), (result, debug)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_tank_clean_terrain_contact_rejects_pathological_depth_by_default: PASSED")
    return True


def test_entity_world_collision_raw_origin_fallback_applies_guarded_pair_solver():
    """Opt-in raw-origin fallback should turn eligible lifted misses into capped contacts."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_PROJECTION_ORDER",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = "1"
        os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA"] = "20.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED"] = "200.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE"] = "solver"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE"] = "solver"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION"] = "default"
        calls = []

        def fake_model_collision(center, *args, **kwargs):
            calls.append(center)
            if abs(center[2] - 10.0) < 1e-6:
                return TerrainContact(
                    position=(1.0, 2.0, 3.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=4.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="terrain_triangle",
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            6.0,
            0.0,
            2.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(6.0, 0.0, 2.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_raw_origin_fallback_contact", debug
        assert debug["raw_origin_fallback"] is True, debug
        assert debug["response"] == "terrain_contact_constraint_solver", debug
        assert debug["constraint_projection_order"] == "opposite_if_separating", debug
        assert debug["constraint_primary_projection_speed_source"] == "world_minus_body_if_body_separating", debug
        assert debug["raw_origin_fallback_velocity_delta_clamped"] is False, debug
        assert debug["contact_cell"] == (7, 8), debug
        assert pz > 10.0 and pz < 10.01, (px, py, pz)
        assert vz > 2.0, (vx, vy, vz)
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["raw_origin_fallback_enabled"] is True, probe
        assert probe["raw_origin_fallback_reject"] == "", probe
        assert calls[0] == (0.0, 0.0, 13.0)
        assert calls[1] == (0.0, 0.0, 10.0)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_entity_world_collision_raw_origin_fallback_applies_guarded_pair_solver: PASSED")
    return True


def test_entity_world_collision_raw_origin_fallback_clamps_solver_velocity_delta():
    """The experimental raw-origin fallback must not amplify one contact into runaway speed."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = "1"
        os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA"] = "5.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED"] = "200.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE"] = "solver"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE"] = "solver"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION"] = "default"
        calls = []

        def fake_model_collision(center, *args, **kwargs):
            calls.append(center)
            if abs(center[2] - 10.0) < 1e-6:
                return TerrainContact(
                    position=(0.0, 0.0, 9.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=4.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="terrain_triangle",
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            -50.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(0.0, 0.0, -50.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_raw_origin_fallback_contact", debug
        assert debug["raw_origin_fallback"] is True, debug
        assert debug["raw_origin_fallback_velocity_delta_clamped"] is True, debug
        assert debug["raw_origin_fallback_velocity_delta_mag_after_safety"] <= 5.000001, debug
        assert math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)
        assert math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)
        assert abs(vz - (-45.0)) < 1e-6, (vx, vy, vz)
        assert calls[0] == (0.0, 0.0, 13.0)
        assert calls[1] == (0.0, 0.0, 10.0)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_entity_world_collision_raw_origin_fallback_clamps_solver_velocity_delta: PASSED")
    return True


def test_entity_world_collision_raw_origin_fallback_can_use_terrain_normal_probe():
    """Raw-origin fallback can A/B terrain normals without changing the global grid mode."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = "1"
        os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA"] = "10.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED"] = "200.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE"] = "terrain"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE"] = "normal"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE"] = "solver"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION"] = "0.0"
        terrain_normal = (0.5, 0.0, 0.8660254037844386)

        def fake_model_collision(center, *args, **kwargs):
            if abs(center[2] - 10.0) < 1e-6:
                return TerrainContact(
                    position=(1.0, 2.0, 3.0),
                    normal=(1.0, 0.0, 0.0),
                    penetration=4.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server.terrain = SimpleNamespace(
            sample_height_normal=lambda wx, wy: (2.5, terrain_normal)
        )
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            -8.0,
            0.0,
            -3.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(-8.0, 0.0, -3.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_raw_origin_fallback_contact", debug
        assert debug["contact_normal_source"] == "terrain_triangle", debug
        assert debug["normal"] == terrain_normal, debug
        assert debug["raw_origin_fallback_normal_source"] == "terrain", debug
        assert debug["raw_origin_fallback_delta_mode"] == "normal", debug
        assert debug["raw_origin_fallback_angular_mode"] == "solver", debug
        assert debug["raw_origin_fallback_closing_only"] is True, debug
        assert debug["raw_origin_fallback_before_normal_speed"] < 0.0, debug
        assert debug["raw_origin_fallback_normal_delta_skip_reason"] == "", debug
        assert debug["raw_origin_fallback_normal_delta_projected"] is True, debug
        assert debug["pair_friction_coeff"] == 0.0, debug
        assert debug["friction_skip_reason"] == "nonpositive_pair_friction", debug
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["raw_origin_contact"]["contact_normal_source"] == "terrain_triangle", probe
        assert probe["raw_origin_fallback_reject"] == "", probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_entity_world_collision_raw_origin_fallback_can_use_terrain_normal_probe: PASSED")
    return True


def test_entity_world_collision_raw_origin_fallback_can_use_contact_face_normal():
    """Raw-origin fallback can use the decompile terrain-face normal carried by the CBSP contact."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = "1"
        os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA"] = "10.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED"] = "200.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE"] = "terrain_face"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE"] = "normal"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE"] = "preserve"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION"] = "0.0"
        contact_face_normal = (0.25, 0.0, 0.9682458365518543)
        sampled_terrain_normal = (0.0, 1.0, 0.0)

        def fake_model_collision(center, *args, **kwargs):
            if abs(center[2] - 10.0) < 1e-6:
                return TerrainContact(
                    position=(1.0, 2.0, 3.0),
                    normal=(1.0, 0.0, 0.0),
                    penetration=4.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(1.0, 0.0, 0.0),
                    terrain_face_normal=contact_face_normal,
                    mesh_face_normal=(1.0, 0.0, 0.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server.terrain = SimpleNamespace(
            sample_height_normal=lambda wx, wy: (2.5, sampled_terrain_normal)
        )
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            -8.0,
            0.0,
            -3.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(-8.0, 0.0, -3.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_raw_origin_fallback_contact", debug
        assert debug["contact_normal_source"] == "terrain_triangle_contact_face", debug
        assert debug["normal"] == contact_face_normal, debug
        assert debug["normal"] != sampled_terrain_normal, debug
        assert debug["contact_terrain_face_normal"] == contact_face_normal, debug
        assert debug["raw_origin_fallback_normal_source"] == "terrain_face", debug
        assert debug["raw_origin_fallback_angular_preserved"] is True, debug
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["raw_origin_contact"]["contact_normal_source"] == "terrain_triangle_contact_face", probe
        assert probe["raw_origin_contact"]["contact_terrain_face_normal"] == contact_face_normal, probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_entity_world_collision_raw_origin_fallback_can_use_contact_face_normal: PASSED")
    return True


def test_entity_world_collision_raw_origin_fallback_can_blend_radial_face_forward_up():
    """Raw-origin fallback can A/B decompile-context/terrain-face normals without side impulse."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ENTITY_RADIAL_WEIGHT",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_TERRAIN_FACE_WEIGHT",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = "1"
        os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA"] = "10.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED"] = "200.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA"] = "0.5"
        os.environ[
            "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE"
        ] = "entity_radial_terrain_face_forward_up"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ENTITY_RADIAL_WEIGHT"] = "1.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_TERRAIN_FACE_WEIGHT"] = "1.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE"] = "normal"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE"] = "preserve"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION"] = "0.0"
        entity_radial_normal = (0.6, 0.8, 0.0)
        terrain_face_normal = (0.0, 0.0, 1.0)
        expected_normal = (0.5144957554275266, 0.0, 0.8574929257125442)

        def fake_model_collision(center, *args, **kwargs):
            if abs(center[2] - 10.0) < 1e-6:
                return TerrainContact(
                    position=(1.0, 2.0, 3.0),
                    normal=(1.0, 0.0, 0.0),
                    penetration=4.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(1.0, 0.0, 0.0),
                    terrain_face_normal=terrain_face_normal,
                    mesh_face_normal=(1.0, 0.0, 0.0),
                    entity_radial_normal=entity_radial_normal,
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            -5.0,
            0.0,
            -5.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(-5.0, 0.0, -5.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_raw_origin_fallback_contact", debug
        assert (
            debug["contact_normal_source"]
            == "entity_radial_terrain_face_forward_up"
        ), debug
        assert debug["raw_origin_fallback_normal_source"] == (
            "entity_radial_terrain_face_forward_up"
        ), debug
        assert debug["contact_entity_radial_normal"] == entity_radial_normal, debug
        assert debug["contact_terrain_face_normal"] == terrain_face_normal, debug
        for got, expected in zip(debug["normal"], expected_normal):
            assert math.isclose(got, expected, abs_tol=1e-6), debug
        assert math.isclose(debug["normal"][1], 0.0, abs_tol=1e-8), debug
        assert debug["raw_origin_fallback_angular_preserved"] is True, debug
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["raw_origin_contact"]["contact_normal_source"] == (
            "entity_radial_terrain_face_forward_up"
        ), probe
        assert probe["raw_origin_fallback_reject"] == "", probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_raw_origin_fallback_can_blend_radial_face_forward_up: PASSED"
    )
    return True


def test_entity_world_collision_raw_origin_fallback_can_use_closing_velocity_delta():
    """Raw-origin fallback can bypass solver magnitude and cancel projected closing speed."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = "1"
        os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_SPEED"] = "0.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA"] = "20.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED"] = "200.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE"] = "terrain_face"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE"] = "preserve"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION"] = "0.0"

        def fake_model_collision(center, *args, **kwargs):
            if abs(center[2] - 10.0) < 1e-6:
                return TerrainContact(
                    position=(0.0, 0.0, 9.0),
                    normal=(1.0, 0.0, 0.0),
                    penetration=3.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    terrain_face_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            -3.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(0.0, 0.0, -3.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_raw_origin_fallback_contact", debug
        assert debug["contact_normal_source"] == "terrain_triangle_contact_face", debug
        assert debug["raw_origin_fallback_delta_mode"] == "closing_velocity", debug
        assert debug["raw_origin_fallback_normal_delta_projected"] is True, debug
        assert debug["raw_origin_fallback_normal_delta_skip_reason"] == "", debug
        assert debug["raw_origin_fallback_before_normal_speed"] < 0.0, debug
        assert 0.0 <= vz <= 0.01, (px, py, pz, vx, vy, vz, debug)
        assert 3.0 <= debug["raw_origin_fallback_velocity_delta_mag_after_safety"] <= 3.01, debug
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_entity_world_collision_raw_origin_fallback_can_use_closing_velocity_delta: PASSED")
    return True


def test_entity_world_collision_raw_origin_fallback_can_preserve_angular_velocity():
    """Raw-origin fallback can run as a linear-only A/B without synthetic angular feedback."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = "1"
        os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA"] = "10.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED"] = "200.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE"] = "normal"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE"] = "preserve"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION"] = "0.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_PROJECTION_ORDER"] = "body_minus_world"

        def fake_model_collision(center, *args, **kwargs):
            if abs(center[2] - 10.0) < 1e-6:
                return TerrainContact(
                    position=(2.0, -3.0, 5.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=5.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="terrain_triangle",
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)
        ctx.spring_body_ang_vel = (0.25, -0.125)
        ctx.angular_vel_yaw = 0.0

        server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            4.0,
            0.0,
            -8.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(4.0, 0.0, -8.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_raw_origin_fallback_contact", debug
        assert debug["raw_origin_fallback_delta_mode"] == "normal", debug
        assert debug["raw_origin_fallback_angular_mode"] == "preserve", debug
        assert debug["raw_origin_fallback_closing_only"] is True, debug
        assert debug["raw_origin_fallback_normal_delta_skip_reason"] == "", debug
        assert debug["raw_origin_fallback_angular_preserved"] is True, debug
        assert debug["raw_origin_fallback_angular_delta_after_safety"] == (0.0, 0.0, 0.0), debug
        assert debug["angular_velocity_after"] == (0.25, -0.125, 0.0), debug
        assert ctx.spring_body_ang_vel == (0.25, -0.125), ctx.spring_body_ang_vel
        assert ctx.angular_vel_yaw == 0.0, ctx.angular_vel_yaw
        assert debug["raw_origin_fallback_velocity_delta_mag_after_safety"] > 0.0, debug
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_entity_world_collision_raw_origin_fallback_can_preserve_angular_velocity: PASSED")
    return True


def test_entity_world_collision_raw_origin_fallback_skips_normal_delta_when_separating():
    """Normal-projected fallback contacts must not add rebound when already separating."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = "1"
        os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA"] = "10.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED"] = "200.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE"] = "normal"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE"] = "preserve"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION"] = "0.0"

        def fake_model_collision(center, *args, **kwargs):
            if abs(center[2] - 10.0) < 1e-6:
                return TerrainContact(
                    position=(2.0, -3.0, 5.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=5.0,
                    sector_index=2,
                    cell=(7, 8),
                    normal_source="terrain_triangle",
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (2.0, 2.0, 3.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=7.5)),
            7.5,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999999.0

        ctx = _fake_tank_collision_context()
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)

        server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            12.0,
            pre_pos=(0.0, 0.0, 10.0),
            pre_vel=(0.0, 0.0, 12.0),
            dt=1.0 / 30.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_raw_origin_fallback_contact", debug
        assert debug["raw_origin_fallback_before_normal_speed"] > 0.0, debug
        assert debug["raw_origin_fallback_before_normal_speed_source"] == "constraint_selected_separation_speed_before", debug
        assert debug["raw_origin_fallback_normal_delta_skip_reason"] in {
            "separating_before_velocity",
            "nonpositive_solver_normal_delta",
        }, debug
        assert debug["raw_origin_fallback_velocity_delta_after_safety"] == (0.0, 0.0, 0.0), debug
        assert debug["velocity_after"] == (0.0, 0.0, 12.0), debug
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_entity_world_collision_raw_origin_fallback_skips_normal_delta_when_separating: PASSED")
    return True


def test_entity_world_collision_restores_previous_motion_after_nonfinite_input():
    """Bad timed-contact output should not feed non-finite positions into terrain lookup."""
    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = object()
    server.ground_level = 0.0

    ctx = _fake_tank_collision_context()
    ctx.player_pos = (1.0, 2.0, 3.0)
    ctx.player_vel = (4.0, 5.0, 6.0)

    result = server._resolve_entity_world_collision(
        ctx,
        math.inf,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        pre_pos=(7.0, 8.0, 9.0),
        pre_vel=(0.1, 0.2, 0.3),
        dt=1.0 / 30.0,
    )

    assert result == (7.0, 8.0, 9.0, 0.1, 0.2, 0.3), result
    assert ctx.debug_last_motion_collision["kind"] == "terrain_motion_nonfinite_input"
    assert ctx.world_collision_ref_pos == (7.0, 8.0, 9.0)
    print("test_entity_world_collision_restores_previous_motion_after_nonfinite_input: PASSED")
    return True


def test_entity_world_collision_rejects_pathological_finite_fallback_state():
    """Huge finite fallback state should fail closed instead of poisoning terrain sampling."""
    server = WulframServer.__new__(WulframServer)
    server._terrain_grid_collision = object()
    server.ground_level = 4.0
    server.world_bound = 8192.0

    ctx = _fake_tank_collision_context()
    ctx.player_pos = (1.0e46, -2.0e46, 3.0e45)
    ctx.player_vel = (4.0e47, -5.0e47, 6.0e46)

    result = server._resolve_entity_world_collision(
        ctx,
        -8192.0,
        -8192.0,
        -2030.0,
        -math.inf,
        -math.inf,
        0.0,
        pre_pos=(1.0e46, -2.0e46, 3.0e45),
        pre_vel=(4.0e47, -5.0e47, 6.0e46),
        dt=1.0 / 30.0,
    )

    assert result == (0.0, 0.0, 4.0, 0.0, 0.0, 0.0), result
    debug = ctx.debug_last_motion_collision
    assert debug["kind"] == "terrain_motion_nonfinite_input", debug
    assert debug["fallback_pos"] == (0.0, 0.0, 4.0), debug
    assert debug["fallback_vel"] == (0.0, 0.0, 0.0), debug
    assert ctx.world_collision_ref_pos == (0.0, 0.0, 4.0)
    print("test_entity_world_collision_rejects_pathological_finite_fallback_state: PASSED")
    return True


def test_entity_world_half_extents_preserves_mesh_z_extent():
    """Mesh-backed vehicle contact should not inflate Z to the tank radius."""
    server = WulframServer.__new__(WulframServer)
    server._entity_collision_extents_cache = {}
    server._building_collision = SimpleNamespace(
        available=True,
        models={
            "tank_1": SimpleNamespace(
                collision_mesh=SimpleNamespace(
                    vertices=[
                        SimpleNamespace(x=-1.0, y=-2.0, z=-1.5),
                        SimpleNamespace(x=1.0, y=2.0, z=1.25),
                    ]
                )
            )
        },
    )
    ctx = _fake_tank_collision_context()

    half_extents = server._get_entity_world_half_extents(ctx)

    assert half_extents == (4.0, 4.0, 1.5), half_extents
    print("test_entity_world_half_extents_preserves_mesh_z_extent: PASSED")
    return True


def test_entity_origin_probe_uses_capped_pair_solver_contact_response():
    """Entity-origin terrain probes should use capped solver response instead of full penetration snap."""
    old_origin = os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN")
    old_response = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE")
    try:
        os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = "entity"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)

        def fake_model_collision(*args, **kwargs):
            return TerrainContact(
                position=(0.0, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
                penetration=26.0,
                sector_index=0,
                cell=(0, 0),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999.0
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
            entity_type=int(EntityType.TANK),
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.spring_body_matrix = _matrix3_from_euler_xyz(0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            0.0,
            10.0,
            0.0,
            -5.0,
        )

        assert abs(px) < 1e-6, px
        assert abs(py) < 1e-6, py
        assert 0.004 < pz < 0.006, pz
        assert 0.0 < vx < 0.01, vx
        assert abs(vy) < 1e-6, vy
        assert 0.49 < vz < 0.52, vz
        assert ctx.debug_last_motion_collision["contact_sector_index"] == 0
        assert ctx.debug_last_motion_collision["contact_cell"] == (0, 0)
        assert ctx.debug_last_motion_collision["response"] == "terrain_contact_constraint_solver"
        assert ctx.debug_last_motion_collision["position_correction"] <= 0.005
        assert (
            ctx.debug_last_motion_collision["constraint_model"]
            == "decompile_static_terrain_sequential_impulse"
        )
        assert ctx.debug_last_motion_collision["rotation_source"] == "rotation_matrix"
        assert ctx.debug_last_motion_collision["torque_delta_frame"] == "entity_local"
        assert ctx.debug_last_motion_collision["constraint_record_order"] == "body_static_world"
        assert (
            ctx.debug_last_motion_collision["constraint_projection_model"]
            == "Constraint_compute_velocity_projection_body_minus_world"
        )
        assert ctx.debug_last_motion_collision["constraint_world_point_velocity_before"] == (
            0.0,
            0.0,
            0.0,
        )
        assert (
            ctx.debug_last_motion_collision["constraint_relative_velocity_before"]
            == ctx.debug_last_motion_collision["point_velocity_before"]
        )
        assert math.isclose(
            ctx.debug_last_motion_collision["constraint_separation_speed_before"],
            ctx.debug_last_motion_collision["point_normal_velocity_before"],
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        assert math.isclose(
            ctx.debug_last_motion_collision["constraint_opposite_separation_speed_before"],
            -ctx.debug_last_motion_collision["point_normal_velocity_before"],
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        assert ctx.debug_last_motion_collision["normal_impulse_body_direction"] == (
            0.0,
            0.0,
            1.0,
        )
        assert (
            ctx.debug_last_motion_collision["inertia_model"]
            == "decompile_mesh_bounds_full_extents"
        )
        assert abs(
            ctx.debug_last_motion_collision["inertia_diagonal"][0]
            - ((8.0 * 8.0 * 8.0 * 8.0 * 6700.0) / 12.0)
        ) < 1e-6
        assert (
            ctx.debug_last_motion_collision["friction_model"]
            == "decompile_constraint_apply_friction_min_pair"
        )
        assert abs(ctx.debug_last_motion_collision["pair_friction_coeff"] - 0.2) < 1e-6
        assert abs(ctx.debug_last_motion_collision["terrain_friction_coeff"] - 1.0) < 1e-6
        assert ctx.debug_last_motion_collision["body_should_sleep"] is False
        assert ctx.debug_last_motion_collision["body_is_sleeping"] is False
        assert ctx.debug_last_motion_collision["effective_mass_sleep_scale"] == 1.0
        assert ctx.debug_last_motion_collision["impulse_sleep_scale"] == 1.0
        assert ctx.debug_last_motion_collision["friction_skip_reason"] is None
        assert (
            ctx.debug_last_motion_collision["entity_interpolation_model"]
            == "decompile_entity_interpolate_toward_target"
        )
        assert ctx.debug_last_motion_collision["interpolation_action"] == "target_unavailable"
        assert ctx.debug_last_motion_collision["effective_mass_normal"] > 0.0
        assert ctx.debug_last_motion_collision["normal_iterations"] > 1
        assert ctx.debug_last_motion_collision["friction_iterations"] > 1
        assert ctx.debug_last_motion_collision["restitution_impulse"] > 0.0
    finally:
        if old_origin is None:
            os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        else:
            os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = old_origin
        if old_response is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = old_response
    print("test_entity_origin_probe_uses_capped_pair_solver_contact_response: PASSED")
    return True


def test_entity_origin_probe_can_retest_inactive_penetrating_contact_when_enabled():
    """Entity-origin terrain contacts can rerun inactive constraints when enabled."""
    old_origin = os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN")
    old_response = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE")
    old_retest = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONSTRAINT_RETEST")
    try:
        os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = "entity"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_CONSTRAINT_RETEST"] = "1"

        def fake_model_collision(*args, **kwargs):
            return TerrainContact(
                position=(0.0, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
                penetration=1.0,
                sector_index=0,
                cell=(0, 0),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999.0
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
            entity_type=int(EntityType.TANK),
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)

        _px, _py, _pz, _vx, _vy, _vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
        )
        debug = ctx.debug_last_motion_collision

        assert debug["response"] == "terrain_contact_constraint_solver"
        assert debug["primary_start_separation_speed"] > 0.005
        assert debug["primary_normal_iterations"] == 0
        assert debug["inactive_retest_enabled"] is True
        assert debug["inactive_retest_applied"] is True
        assert debug["inactive_retest_iterations"] > 0
        assert debug["normal_impulse"] > 0.0
        assert debug["point_normal_velocity_after"] > debug["point_normal_velocity_before"]
        assert math.isclose(
            debug["inactive_retest_target_separation"],
            debug["inactive_retest_start_separation_speed"] + 0.1,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        assert debug["velocity_before"] != debug["velocity_after"]
    finally:
        if old_origin is None:
            os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        else:
            os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = old_origin
        if old_response is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = old_response
        if old_retest is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONSTRAINT_RETEST", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONSTRAINT_RETEST"] = old_retest
    print("test_entity_origin_probe_can_retest_inactive_penetrating_contact_when_enabled: PASSED")
    return True


def test_entity_origin_probe_suppresses_static_terrain_yaw_feedback_by_default():
    """Tank terrain contact should not feed solver yaw impulse into steering yaw."""
    old_origin = os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN")
    old_response = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE")
    old_yaw_feedback = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_YAW_FEEDBACK")
    try:
        os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = "entity"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_YAW_FEEDBACK", None)

        def fake_model_collision(*args, **kwargs):
            return TerrainContact(
                position=(2.0, 0.0, 1.0),
                normal=(0.0, 0.0, 1.0),
                penetration=0.1,
                sector_index=0,
                cell=(0, 0),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999.0
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
            entity_type=int(EntityType.TANK),
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)
        ctx.angular_vel_yaw = 0.25
        ctx.spring_body_ang_vel = (0.0, 0.0)

        server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            0.0,
            0.0,
            10.0,
            -2.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["response"] == "terrain_contact_constraint_solver", debug
        assert debug["contact_yaw_feedback_enabled"] is False, debug
        assert abs(debug["constraint_angular_delta_raw"][2]) > 0.01, debug
        assert abs(debug["contact_yaw_delta_suppressed"]) > 0.01, debug
        assert abs(ctx.angular_vel_yaw - 0.25) < 1e-9, ctx.angular_vel_yaw
        assert abs(debug["angular_velocity_after"][2] - 0.25) < 1e-9, debug
        assert abs(ctx.spring_body_ang_vel[1]) > 0.0, ctx.spring_body_ang_vel
    finally:
        if old_origin is None:
            os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        else:
            os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = old_origin
        if old_response is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = old_response
        if old_yaw_feedback is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_YAW_FEEDBACK", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_YAW_FEEDBACK"] = old_yaw_feedback
    print("test_entity_origin_probe_suppresses_static_terrain_yaw_feedback_by_default: PASSED")
    return True


def test_entity_origin_probe_uses_fed_target_for_interpolation_decision():
    """Server entity-origin probes should feed RigidBody target state into interpolation gates."""
    old_origin = os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN")
    old_response = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE")
    try:
        os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = "entity"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)

        def fake_model_collision(*args, **kwargs):
            return TerrainContact(
                position=(0.0, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
                penetration=26.0,
                sector_index=0,
                cell=(0, 0),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999.0
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
            entity_type=int(EntityType.TANK),
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)
        ctx.last_client_tick = 123
        ctx.rigid_body_target_pos = (0.0, 0.0, 0.005)
        ctx.rigid_body_target_rot = (0.0, 0.0, 0.0)
        ctx.rigid_body_interp_tolerance = 0.003
        ctx.rigid_body_last_interp_tick = 0

        server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            0.0,
            10.0,
            0.0,
            -5.0,
        )

        assert (
            ctx.debug_last_motion_collision["entity_interpolation_model"]
            == "decompile_entity_interpolate_toward_target"
        )
        assert ctx.debug_last_motion_collision["interpolation_action"] == "wake_update_last_interp_tick"
        assert ctx.debug_last_motion_collision["interpolation_update_last_interp_tick"] is True
        assert ctx.rigid_body_last_interp_tick == 123
        assert ctx.debug_last_motion_collision["radius_gate"] is False
    finally:
        if old_origin is None:
            os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        else:
            os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = old_origin
        if old_response is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = old_response
    print("test_entity_origin_probe_uses_fed_target_for_interpolation_decision: PASSED")
    return True


def test_static_terrain_constraint_sleeping_body_uses_decompile_scaling():
    """Persistent +0xAE sleep should halve K and applied impulses, or freeze to zero."""
    base_kwargs = dict(
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, -2.0),
        angular_velocity=(0.0, 0.0, 0.0),
        contact_point=(2.0, 0.0, 0.0),
        contact_normal=(0.0, 0.0, 1.0),
        penetration=0.1,
        half_extents=(4.0, 4.0, 4.0),
        inertia_half_extents=(4.0, 4.0, 4.0),
        mass=6700.0,
        friction=0.2,
        constraint_iterations=1,
        restitution_fraction=0.0,
    )

    awake = solve_static_terrain_constraint(**base_kwargs)
    sleeping = solve_static_terrain_constraint(**base_kwargs, body_is_sleeping=True)
    frozen_sleeping = solve_static_terrain_constraint(
        **base_kwargs,
        body_is_sleeping=True,
        constraint_frozen=True,
    )

    assert awake.debug["effective_mass_sleep_scale"] == 1.0
    assert awake.debug["impulse_sleep_scale"] == 1.0
    assert awake.debug["body_is_sleeping"] is False
    assert sleeping.debug["body_is_sleeping"] is True
    assert sleeping.debug["effective_mass_sleep_scale"] == 0.5
    assert sleeping.debug["impulse_sleep_scale"] == 0.5
    assert math.isclose(
        sleeping.debug["effective_mass_normal"],
        awake.debug["effective_mass_normal"] * 0.5,
        rel_tol=1e-6,
    )
    assert frozen_sleeping.debug["constraint_frozen"] is True
    assert frozen_sleeping.debug["effective_mass_normal"] == 0.0
    assert frozen_sleeping.debug["effective_mass_sleep_scale"] == 0.0
    assert frozen_sleeping.debug["impulse_sleep_scale"] == 0.0
    assert frozen_sleeping.debug["normal_iterations"] == 0
    assert frozen_sleeping.velocity == (0.0, 0.0, -2.0)
    print("test_static_terrain_constraint_sleeping_body_uses_decompile_scaling: PASSED")
    return True


def test_static_terrain_constraint_retests_inactive_penetrating_contact():
    """Inactive first pass should rerun with cached separation +0.1 like OG."""
    base_kwargs = dict(
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.2),
        angular_velocity=(0.0, -1.0, 0.0),
        contact_point=(2.0, 0.0, 1.0),
        contact_normal=(0.0, 0.0, 1.0),
        penetration=0.1,
        half_extents=(4.0, 4.0, 4.0),
        inertia_half_extents=(4.0, 4.0, 4.0),
        mass=6700.0,
        friction=0.2,
        restitution_fraction=0.0,
    )

    first_pass_only = solve_static_terrain_constraint(
        **base_kwargs,
        enable_inactive_retest=False,
    )
    retested = solve_static_terrain_constraint(
        **base_kwargs,
        enable_inactive_retest=True,
    )

    assert first_pass_only.debug["primary_start_separation_speed"] > 0.005
    assert first_pass_only.debug["normal_iterations"] == 0
    assert first_pass_only.debug["normal_impulse"] == 0.0
    assert first_pass_only.debug["inactive_retest_applied"] is False
    assert retested.debug["primary_normal_iterations"] == 0
    assert retested.debug["inactive_retest_applied"] is True
    assert retested.debug["inactive_retest_iterations"] > 0
    assert retested.debug["normal_iterations"] == retested.debug["inactive_retest_iterations"]
    assert retested.debug["normal_impulse"] > 0.0
    assert math.isclose(
        retested.debug["inactive_retest_target_separation"],
        retested.debug["inactive_retest_start_separation_speed"] + 0.1,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    print("test_static_terrain_constraint_retests_inactive_penetrating_contact: PASSED")
    return True


def test_static_terrain_constraint_friction_uses_pre_normal_projection_buffer():
    """Friction should consume the pre-normal relative velocity buffer."""
    result = solve_static_terrain_constraint(
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, -2.0),
        angular_velocity=(0.0, 0.0, 0.0),
        contact_point=(2.0, 0.0, 1.0),
        contact_normal=(0.0, 0.0, 1.0),
        penetration=0.1,
        half_extents=(4.0, 4.0, 4.0),
        inertia_half_extents=(4.0, 4.0, 4.0),
        mass=6700.0,
        friction=0.2,
        constraint_iterations=1,
        restitution_fraction=0.0,
    )

    assert result.debug["normal_iterations"] == 1
    assert result.debug["friction_velocity_source"] == "pre_normal_projection_buffer"
    assert result.debug["friction_iterations"] == 0
    assert result.debug["friction_impulse"] == 0.0
    assert result.debug["post_normal_tangent_speed_abs_max"] > 0.001
    print("test_static_terrain_constraint_friction_uses_pre_normal_projection_buffer: PASSED")
    return True


def test_static_terrain_constraint_opposite_projection_probe_activates_separating_penetration():
    """The rough-terrain probe can test OG's opposite projection without changing defaults."""
    base_kwargs = dict(
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 2.0),
        angular_velocity=(0.0, 0.0, 0.0),
        contact_point=(0.0, 0.0, -1.0),
        contact_normal=(0.0, 0.0, 1.0),
        penetration=0.1,
        half_extents=(4.0, 4.0, 4.0),
        inertia_half_extents=(4.0, 4.0, 4.0),
        mass=6700.0,
        friction=0.2,
        constraint_iterations=1,
        restitution_fraction=0.0,
    )

    default = solve_static_terrain_constraint(**base_kwargs)
    opposite_probe = solve_static_terrain_constraint(
        **base_kwargs,
        projection_order="opposite_if_separating",
    )
    signed_opposite_probe = solve_static_terrain_constraint(
        **base_kwargs,
        projection_order="opposite_if_separating_signed",
    )
    falling = solve_static_terrain_constraint(
        **{**base_kwargs, "velocity": (0.0, 0.0, -2.0)},
        projection_order="opposite_if_separating",
    )

    assert default.debug["constraint_projection_order"] == "body_minus_world"
    assert default.debug["normal_iterations"] == 0
    assert default.debug["normal_impulse"] == 0.0
    assert opposite_probe.debug["constraint_projection_order"] == "opposite_if_separating"
    assert (
        opposite_probe.debug["constraint_primary_projection_speed_source"]
        == "world_minus_body_if_body_separating"
    )
    assert math.isclose(
        opposite_probe.debug["constraint_body_minus_world_speed_before"],
        2.0,
        rel_tol=1e-6,
    )
    assert math.isclose(
        opposite_probe.debug["constraint_selected_separation_speed_before"],
        -2.0,
        rel_tol=1e-6,
    )
    assert opposite_probe.debug["normal_iterations"] == 1
    assert opposite_probe.debug["normal_impulse"] > 0.0
    assert opposite_probe.velocity[2] > base_kwargs["velocity"][2]
    assert (
        signed_opposite_probe.debug["constraint_projection_order"]
        == "opposite_if_separating"
    )
    assert (
        signed_opposite_probe.debug["constraint_projection_impulse_sign_mode"]
        == "match_projection"
    )
    assert (
        signed_opposite_probe.debug["constraint_primary_projection_speed_source"]
        == "world_minus_body_if_body_separating"
    )
    assert signed_opposite_probe.debug["normal_iterations"] == 1
    assert signed_opposite_probe.debug["normal_impulse"] > 0.0
    assert signed_opposite_probe.debug["normal_impulse_body_sign"] == -1.0
    assert signed_opposite_probe.velocity[2] < base_kwargs["velocity"][2]
    assert falling.debug["constraint_primary_projection_speed_source"] == "body_minus_world"
    assert falling.debug["normal_impulse"] > 0.0
    print("test_static_terrain_constraint_opposite_projection_probe_activates_separating_penetration: PASSED")
    return True


def test_static_terrain_constraint_can_probe_entity_rotation_for_angular_frame():
    """Constraint point velocity and angular impulse should use the entity frame."""
    base_kwargs = dict(
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, -2.0),
        angular_velocity=(0.0, 0.75, 0.0),
        contact_point=(2.0, 0.0, 1.0),
        contact_normal=(0.0, 0.0, 1.0),
        penetration=0.1,
        half_extents=(4.0, 4.0, 4.0),
        inertia_half_extents=(4.0, 4.0, 4.0),
        mass=6700.0,
        friction=0.2,
        constraint_iterations=1,
        restitution_fraction=0.0,
    )

    flat = solve_static_terrain_constraint(**base_kwargs, body_rotation=(0.0, 0.0, 0.0))
    roll = math.radians(30.0)
    tilted = solve_static_terrain_constraint(
        **base_kwargs,
        body_rotation=(roll, 0.0, 0.0),
    )

    assert tilted.debug["rotation_source"] == "body_rotation_euler"
    assert tilted.debug["angular_velocity_frame"] == "entity_local"
    assert tilted.debug["torque_delta_frame"] == "entity_local"
    assert tilted.debug["contact_lever"] == (2.0, 0.0, 1.0)

    matrix = tilted.debug["rotation_matrix"]
    expected_lever = (
        1.0 * matrix[2] + 2.0 * matrix[0] + 0.0 * matrix[1],
        1.0 * matrix[5] + 0.0 * matrix[4] + 2.0 * matrix[3],
        1.0 * matrix[8] + 0.0 * matrix[7] + 2.0 * matrix[6],
    )
    for got, expected in zip(
        tilted.debug["contact_lever_point_velocity_frame"],
        expected_lever,
    ):
        assert math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-6)

    assert not all(
        math.isclose(got, base, rel_tol=1e-6, abs_tol=1e-6)
        for got, base in zip(
            tilted.debug["contact_lever_point_velocity_frame"],
            tilted.debug["contact_lever"],
        )
    )
    assert not all(
        math.isclose(got, base, rel_tol=1e-6, abs_tol=1e-6)
        for got, base in zip(
            tilted.debug["angular_velocity_after"],
            flat.debug["angular_velocity_after"],
        )
    )
    print("test_static_terrain_constraint_can_probe_entity_rotation_for_angular_frame: PASSED")
    return True


def test_static_terrain_constraint_can_probe_contact_iterative_solver_shape():
    """The direct-contact solver probe should expose OG's aggressive threshold path."""
    base_kwargs = dict(
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, -2.0),
        angular_velocity=(0.0, 0.0, 0.0),
        contact_point=(0.0, 0.0, 0.0),
        contact_normal=(0.0, 0.0, 1.0),
        penetration=0.1,
        half_extents=(4.0, 4.0, 4.0),
        inertia_half_extents=(4.0, 4.0, 4.0),
        mass=6700.0,
        friction=0.0,
        constraint_iterations=1,
        restitution_fraction=0.0,
    )

    constraint = solve_static_terrain_constraint(**base_kwargs)
    contact = solve_static_terrain_constraint(
        **base_kwargs,
        solver_variant="contact",
    )

    assert constraint.debug["constraint_solver_variant"] == "constraint_iterative"
    assert constraint.debug["constraint_iteration_limit"] == 1
    assert math.isclose(
        constraint.debug["constraint_min_correction_initial"],
        0.005,
        rel_tol=1e-6,
    )
    assert constraint.debug["constraint_progressive_scaling"] is True
    assert contact.debug["constraint_solver_variant"] == "contact_iterative"
    assert contact.debug["constraint_iteration_limit"] == 1
    assert math.isclose(
        contact.debug["constraint_min_correction_initial"],
        0.1,
        rel_tol=1e-6,
    )
    assert math.isclose(
        contact.debug["constraint_min_correction_increment"],
        0.002,
        rel_tol=1e-6,
    )
    assert contact.debug["constraint_progressive_scaling"] is False
    assert contact.velocity[2] > constraint.velocity[2] + 1.0
    print("test_static_terrain_constraint_can_probe_contact_iterative_solver_shape: PASSED")
    return True


def test_entity_interpolation_decision_matches_decompile_gates():
    """Entity_interpolate_toward_target should expose reset/wake/tick-update gates."""
    assert entity_interp_factor(0.03) == 1.0
    assert entity_interp_factor(0.25) == 0.0
    assert math.isclose(entity_interp_factor(0.1), 0.25, rel_tol=1e-6)

    reset = entity_interpolate_toward_target_decision(
        current_position=(0.0, 0.0, 0.0),
        target_position=(0.001, 0.001, 0.001),
        current_rotation=(0.0, 0.0, 0.0),
        target_rotation=(0.0, 0.0, 0.001),
        tolerance=5.0,
        combined_radius=5.0,
        current_tick=20,
        last_interp_tick=0,
        delta_seconds=1.0 / 30.0,
    )
    assert reset.action == "reset_physics"
    assert reset.reset_physics is True
    assert reset.wake is False
    assert reset.update_last_interp_tick is False

    wait = entity_interpolate_toward_target_decision(
        current_position=(0.0, 0.0, 0.0),
        target_position=(0.001, 0.001, 0.001),
        current_rotation=(0.0, 0.0, 0.0),
        target_rotation=(0.0, 0.0, 0.001),
        tolerance=5.0,
        combined_radius=5.0,
        current_tick=2,
        last_interp_tick=0,
        delta_seconds=1.0 / 30.0,
    )
    assert wait.action == "wake_without_tick_update"
    assert wait.wake is True
    assert wait.update_last_interp_tick is False

    far = entity_interpolate_toward_target_decision(
        current_position=(0.0, 0.0, 0.0),
        target_position=(20.0, 0.0, 0.0),
        current_rotation=(0.0, 0.0, 0.0),
        target_rotation=(0.0, 0.0, 0.0),
        tolerance=0.005,
        combined_radius=5.0,
        current_tick=20,
        last_interp_tick=0,
        delta_seconds=1.0 / 30.0,
    )
    assert far.action == "wake_update_last_interp_tick"
    assert far.reset_physics is False
    assert far.wake is True
    assert far.update_last_interp_tick is True

    forced = entity_interpolate_toward_target_decision(
        current_position=(0.0, 0.0, 0.0),
        target_position=(20.0, 0.0, 0.0),
        current_rotation=(0.0, 0.0, 0.0),
        target_rotation=(0.0, 0.0, 0.0),
        tolerance=0.005,
        combined_radius=5.0,
        current_tick=20,
        last_interp_tick=0,
        delta_seconds=1.0 / 30.0,
        wake_override=True,
    )
    assert forced.action == "wake_forced"
    assert forced.wake is True
    print("test_entity_interpolation_decision_matches_decompile_gates: PASSED")
    return True


def test_entity_origin_probe_applies_pair_solver_at_contact_time():
    """Entity-origin pair probes should resolve at estimated TOI and continue the remaining step."""
    old_origin = os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN")
    old_response = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE")
    old_timing = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING")
    try:
        os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = "entity"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING"] = "probe"

        def fake_model_collision(model_center, *args, **kwargs):
            if model_center[0] < 5.0:
                return None
            return TerrainContact(
                position=(5.0, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
                penetration=26.0,
                sector_index=0,
                cell=(0, 0),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999.0
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
            entity_type=int(EntityType.TANK),
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(10.0, 0.0, 0.0),
            dt=1.0,
        )

        assert 8.8 < px < 9.2, px
        assert abs(py) < 1e-6, py
        assert 0.006 < pz < 0.010, pz
        assert abs(vx - 8.0) < 1e-6, vx
        assert abs(vy) < 1e-6, vy
        assert 0.005 < vz < 0.006, vz
        assert ctx.debug_last_motion_collision["timing_response"] == "terrain_contact_pair_toi_single_step"
        assert 0.49 < ctx.debug_last_motion_collision["collision_time_s"] < 0.51
        assert 0.49 < ctx.debug_last_motion_collision["remaining_time_s"] < 0.51
    finally:
        if old_origin is None:
            os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        else:
            os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = old_origin
        if old_response is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = old_response
        if old_timing is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING"] = old_timing
    print("test_entity_origin_probe_applies_pair_solver_at_contact_time: PASSED")
    return True


def test_entity_origin_probe_can_repeat_bucketed_pair_contacts():
    """Bucketed entity-origin probes should re-solve repeated contacts in one frame."""
    old_origin = os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN")
    old_response = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE")
    old_timing = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING")
    old_iterations = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS")
    old_start_clamp = os.environ.get("WULFRAM_ENTITY_TERRAIN_START_TIME_CLAMP")
    try:
        os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = "entity"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING"] = "bucket"
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS"] = "3"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_START_TIME_CLAMP", None)

        def fake_model_collision(model_center, *args, **kwargs):
            if model_center[0] < 5.0:
                return None
            return TerrainContact(
                position=(5.0, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
                penetration=26.0,
                sector_index=0,
                cell=(0, 0),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999.0
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
            entity_type=int(EntityType.TANK),
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(10.0, 0.0, 0.0),
            dt=1.0,
        )

        assert 8.8 < px < 9.2, px
        assert abs(py) < 1e-6, py
        assert 0.017 < pz < 0.019, pz
        assert abs(vx - 8.0) < 1e-6, vx
        assert abs(vy) < 1e-6, vy
        assert 0.005 < vz < 0.006, vz
        assert ctx.debug_last_motion_collision["timing_response"] == "terrain_contact_pair_bucketed_step"
        assert ctx.debug_last_motion_collision["contact_iteration_count"] == 3
        assert ctx.debug_last_motion_collision["contact_events"][0]["collision_at_start"] is False
        assert ctx.debug_last_motion_collision["contact_events"][1]["collision_at_start"] is True
        assert ctx.debug_last_motion_collision["contact_events"][1]["collision_time_s"] == 0.0
        assert (
            ctx.debug_last_motion_collision["contact_events"][1].get("start_time_clamped", False)
            is False
        )
        first_event = ctx.debug_last_motion_collision["contact_events"][0]
        assert ctx.debug_last_motion_collision["response"] == "terrain_contact_constraint_solver"
        assert first_event["effective_mass_normal"] > 0.0
        assert first_event["inertia_model"] == "decompile_mesh_bounds_full_extents"
        assert first_event["friction_model"] == "decompile_constraint_apply_friction_min_pair"
        assert first_event["constraint_record_order"] == "body_static_world"
        assert (
            first_event["constraint_projection_model"]
            == "Constraint_compute_velocity_projection_body_minus_world"
        )
        assert "constraint_relative_velocity_before" in first_event
        assert "constraint_opposite_separation_speed_before" in first_event
        assert abs(first_event["pair_friction_coeff"] - 0.2) < 1e-6
        assert first_event["body_is_sleeping"] is False
        assert first_event["effective_mass_sleep_scale"] == 1.0
        assert first_event["interpolation_action"] == "target_unavailable"
        assert first_event["normal_impulse"] > 0.0
        assert first_event["restitution_impulse"] > 0.0
        assert first_event["velocity_before"] != first_event["velocity_after"]
    finally:
        if old_origin is None:
            os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        else:
            os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = old_origin
        if old_response is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = old_response
        if old_timing is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING"] = old_timing
        if old_iterations is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS"] = old_iterations
        if old_start_clamp is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_START_TIME_CLAMP", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_START_TIME_CLAMP"] = old_start_clamp
    print("test_entity_origin_probe_can_repeat_bucketed_pair_contacts: PASSED")
    return True


def test_lifted_timed_probe_can_use_guarded_raw_origin_contact():
    """Opt-in timed raw-origin fallback should fill lifted sweep misses at the contact time."""
    env_keys = [
        "WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_TIMED_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_DEPTH",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_DEPTH",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = "pair"
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING"] = "bucket"
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_TIMED_FALLBACK"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_DEPTH"] = "0.1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_DEPTH"] = "40.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_SPEED"] = "0.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA"] = "20.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED"] = "200.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE"] = "solver"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE"] = "solver"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FRICTION"] = "default"
        calls = []

        def fake_model_collision(model_center, *args, **kwargs):
            calls.append(model_center)
            if abs(model_center[2]) > 1e-6 or model_center[0] < 5.0:
                return None
            return TerrainContact(
                position=(5.0, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
                penetration=26.0,
                sector_index=0,
                cell=(0, 0),
                normal_source="terrain_triangle",
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999.0
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )

        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(),
            entity_id=0x14EA,
            entity_type=int(EntityType.TANK),
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(10.0, 0.0, 0.0),
            dt=1.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["timing_response"] == "terrain_contact_pair_toi_single_step", debug
        assert debug["raw_origin_timed_fallback_event_count"] == 1, debug
        assert debug["raw_origin_timed_fallback"] is True, debug
        assert debug["raw_origin_fallback"] is True, debug
        assert debug["raw_origin_fallback_reject"] == "", debug
        assert debug["contact_events"][0]["raw_origin_timed_fallback"] is True, debug
        assert debug["contact_events"][0]["raw_origin_fallback_reject"] == "", debug
        assert 0.49 < debug["collision_time_s"] < 0.51, debug
        assert pz > 0.0, (px, py, pz)
        assert vz > 0.0, (vx, vy, vz)
        assert any(abs(call[2] - 3.0) < 1e-6 for call in calls), calls
        assert any(abs(call[2]) < 1e-6 for call in calls), calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_lifted_timed_probe_can_use_guarded_raw_origin_contact: PASSED")
    return True


def test_raw_origin_closing_gate_uses_solver_contact_projection():
    """Raw fallback safety should follow the contact-point projection, not center velocity."""
    env_keys = [
        "WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_DEPTH",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_DEPTH",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_PROJECTION_ORDER",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = "pair"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_DEPTH"] = "0.1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_DEPTH"] = "40.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MIN_SPEED"] = "0.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_VELOCITY_DELTA"] = "20.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_SPEED"] = "200.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_MAX_ANGULAR_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_DELTA_MODE"] = "normal"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_ANGULAR_MODE"] = "preserve"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_CLOSING_ONLY"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_PROJECTION_ORDER"] = "opposite_if_separating"

        def fake_model_collision(model_center, *args, **kwargs):
            if abs(model_center[2]) > 1e-6:
                return None
            return TerrainContact(
                position=(0.0, 0.0, -1.0),
                normal=(0.0, 0.0, 1.0),
                penetration=1.0,
                sector_index=0,
                cell=(71, 56),
                normal_source="raw_origin_test",
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
            test_model_bounds_contact=lambda *args, **kwargs: None,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999.0
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )

        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(),
            entity_id=0x14EA,
            entity_type=int(EntityType.TANK),
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)

        _px, _py, _pz, _vx, _vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["raw_origin_fallback"] is True, debug
        assert debug["raw_origin_fallback_closing_only"] is True, debug
        assert debug["constraint_projection_order"] == "opposite_if_separating", debug
        assert (
            debug["constraint_primary_projection_speed_source"]
            == "world_minus_body_if_body_separating"
        ), debug
        assert debug["raw_origin_fallback_before_center_normal_speed"] > 0.0, debug
        assert debug["raw_origin_fallback_before_normal_speed"] < 0.0, debug
        assert (
            debug["raw_origin_fallback_before_normal_speed_source"]
            == "constraint_selected_separation_speed_before"
        ), debug
        assert debug["raw_origin_fallback_normal_delta_projected"] is True, debug
        assert debug["raw_origin_fallback_normal_delta_skip_reason"] == "", debug
        assert debug["raw_origin_fallback_velocity_delta_mag_after_safety"] > 0.0, debug
        assert vz > 2.0, (vz, debug)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_raw_origin_closing_gate_uses_solver_contact_projection: PASSED")
    return True


def test_lifted_timed_probe_can_use_guarded_raycast_contact():
    """Opt-in timed raycast fallback should fill lifted sweep misses."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS",
        "WULFRAM_ENTITY_TERRAIN_RAYCAST_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_RAYCAST_TIMED_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_RAYCAST_MIN_PENETRATION",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_TIMED_FALLBACK",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = "pair"
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING"] = "probe"
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAYCAST_FALLBACK"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAYCAST_TIMED_FALLBACK"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAYCAST_MIN_PENETRATION"] = "0.001"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = "0"
        os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_TIMED_FALLBACK"] = "0"
        raycast_calls = []

        def fake_model_collision(*args, **kwargs):
            return None

        def fake_raycast(start, end):
            raycast_calls.append((start, end))
            if start[2] <= end[2]:
                return None
            return TerrainRaycastHit(
                position=(0.0, 0.0, 5.0),
                normal=(0.0, 0.0, 1.0),
                sector_index=0,
                cell=(71, 56),
                distance=4.0,
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
            test_model_bounds_contact=lambda *args, **kwargs: None,
            raycast=fake_raycast,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999.0
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )

        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(),
            entity_id=0x14EA,
            entity_type=int(EntityType.TANK),
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 10.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 10.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -10.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(0.0, 0.0, -10.0),
            dt=1.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["timing_response"] == "terrain_contact_pair_toi_single_step", debug
        assert debug["raycast_timed_fallback_enabled"] is True, debug
        assert debug["raycast_timed_fallback_event_count"] == 1, debug
        assert debug["terrain_raycast_fallback"] is True, debug
        assert debug["raycast_fallback_reject"] == "", debug
        event = debug["contact_events"][0]
        assert event["terrain_raycast_fallback"] is True, debug
        assert event["raycast_fallback_reject"] == "", debug
        assert event["raycast_fallback_probe_reason"] == "timed_lifted_clear_raycast_contact", debug
        assert debug["contact_normal_source"] == "terrain_capsule_raycast", debug
        assert debug["contact_cell"] == (71, 56), debug
        assert pz > 0.0, (px, py, pz)
        assert vz > -10.0, (vx, vy, vz)
        assert raycast_calls, raycast_calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_lifted_timed_probe_can_use_guarded_raycast_contact: PASSED")
    return True


def test_timed_pair_sweep_scan_finds_transient_midframe_contact():
    """Opt-in sweep scan can find clear-contact-clear terrain intersections."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN_STEPS",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = "pair"
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING"] = "probe"
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN_STEPS"] = "4"
        calls = []

        def fake_model_collision(model_center, *args, **kwargs):
            calls.append(model_center)
            if 0.39 <= model_center[0] <= 0.61:
                return TerrainContact(
                    position=(model_center[0], 0.0, 0.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=0,
                    cell=(71, 56),
                    normal_source="transient_test",
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999.0
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
            entity_type=int(EntityType.TANK),
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(1.0, 0.0, 0.0),
            dt=1.0,
        )

        debug = ctx.debug_last_motion_collision
        assert debug["timing_response"] == "terrain_contact_pair_toi_single_step", debug
        assert debug["contact_sweep_scan_enabled"] is True, debug
        assert debug["contact_sweep_scan_event_count"] == 1, debug
        assert debug["contact_events"][0]["contact_sweep_scan"] is True, debug
        assert 0.39 <= debug["contact_sweep_scan_hit_time_s"] <= 0.41, debug
        assert debug["contact_cell"] == (71, 56), debug
        assert pz > 0.0, (px, py, pz)
        assert any(0.39 <= call[0] <= 0.61 for call in calls), calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_timed_pair_sweep_scan_finds_transient_midframe_contact: PASSED")
    return True


def test_pair_record_timed_sweep_uses_shallow_selection_when_enabled():
    """Opt-in pair-record timing should catch transient raw-origin contacts only through its gated path."""
    env_keys = [
        "WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS",
        "WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_SWEEP",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_TIMED_FALLBACK",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_SWEEP_SCAN", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_CONTACT"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_TIMED_SWEEP"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = "upward_min_depth"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ[
            "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"
        ] = "entity_radial_terrain_face_forward_up"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_VERTICAL_DELTA_MODE", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA"] = "3.0"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA"] = "0.0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_TIMED_FALLBACK", None)
        calls = []

        def fake_model_collision(model_center, *args, **kwargs):
            selection = kwargs.get("contact_selection")
            calls.append((model_center, selection))
            if (
                selection == "upward_min_depth"
                and 0.39 <= float(model_center[0]) <= 0.61
            ):
                return TerrainContact(
                    position=(float(model_center[0]), 0.0, 0.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.0,
                    sector_index=0,
                    cell=(71, 56),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
            test_model_bounds_contact=lambda *args, **kwargs: None,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999.0
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )

        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(),
            entity_id=0x14EA,
            entity_type=int(EntityType.TANK),
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 0.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            -1.0,
            pre_pos=(0.0, 0.0, 0.0),
            pre_vel=(1.0, 0.0, -1.0),
            dt=1.0,
        )

        debug = ctx.debug_last_motion_collision
        event = debug["contact_events"][0]
        assert debug["kind"] == "terrain_pair_record_timed_contact", debug
        assert debug["timing_response"] == "terrain_contact_pair_toi_single_step", debug
        assert debug["contact_iteration_limit"] == 1, debug
        assert debug["contact_sweep_scan_enabled"] is True, debug
        assert debug["contact_sweep_scan_event_count"] == 1, debug
        assert debug["pair_record_timed_contact_enabled"] is True, debug
        assert debug["pair_record_timed_sweep_enabled"] is True, debug
        assert debug["pair_record_timed_contact_event_count"] == 1, debug
        assert debug["pair_record_contact"] is True, debug
        assert debug["pair_record_timed_contact"] is True, debug
        assert debug["pair_record_contact_reject"] == "", debug
        assert debug["pair_record_contact_selection"] == "upward_min_depth", debug
        assert event["pair_record_timed_contact"] is True, debug
        assert event["pair_record_contact_selection"] == "upward_min_depth", debug
        assert 0.39 <= debug["contact_sweep_scan_hit_time_s"] <= 0.61, debug
        assert debug["raw_origin_timed_fallback_event_count"] == 0, debug
        assert debug["raycast_timed_fallback_event_count"] == 0, debug
        assert pz > -1.0 and vz > -1.0, (px, py, pz, vx, vy, vz, debug)
        assert any(call[1] == "first" for call in calls), calls
        assert any(call[1] == "upward_min_depth" for call in calls), calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_pair_record_timed_sweep_uses_shallow_selection_when_enabled: PASSED")
    return True


def test_collision_at_start_can_use_iterative_world_separation():
    """Collision-at-start pair records can route to the OG iterative separation branch."""
    old_origin = os.environ.get("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN")
    old_response = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE")
    old_timing = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING")
    old_iterations = os.environ.get("WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS")
    old_start_iterative = os.environ.get("WULFRAM_ENTITY_TERRAIN_START_ITERATIVE")
    try:
        os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = "entity"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING"] = "bucket"
        os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS"] = "3"
        os.environ["WULFRAM_ENTITY_TERRAIN_START_ITERATIVE"] = "1"

        def fake_model_collision(model_center, *args, **kwargs):
            if model_center[0] < 5.0 or model_center[2] >= 1.0:
                return None
            return TerrainContact(
                position=(model_center[0], model_center[1], 0.0),
                normal=(0.0, 0.0, 1.0),
                penetration=1.0 - model_center[2],
                sector_index=0,
                cell=(0, 0),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 999.0
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
            entity_type=int(EntityType.TANK),
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (10.0, 0.0, 0.0)
        ctx.world_collision_ref_pos = (10.0, 0.0, 0.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            pre_pos=(10.0, 0.0, 0.0),
            pre_vel=(0.0, 0.0, 0.0),
            dt=1.0,
        )

        assert abs(px - 10.0) < 1e-6, px
        assert abs(py) < 1e-6, py
        assert abs(pz - 1.0) < 1e-6, pz
        assert (vx, vy, vz) == (0.0, 0.0, 0.0)
        assert (
            ctx.debug_last_motion_collision["response"]
            == "terrain_contact_iterative_position_rollback"
        )
        assert ctx.debug_last_motion_collision["collision_at_start"] is True
        assert ctx.debug_last_motion_collision["contact_iteration_count"] == 1
        assert ctx.debug_last_motion_collision["iterative_cleared"] is True
        assert ctx.debug_last_motion_collision["iterative_iterations"] == 1
        assert (
            ctx.debug_last_motion_collision["contact_events"][0]["response"]
            == "terrain_contact_iterative_position_rollback"
        )
    finally:
        if old_origin is None:
            os.environ.pop("WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN", None)
        else:
            os.environ["WULFRAM_ENTITY_COLLISION_MODEL_ORIGIN"] = old_origin
        if old_response is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_RESPONSE"] = old_response
        if old_timing is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_TIMING"] = old_timing
        if old_iterations is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_CONTACT_ITERATIONS"] = old_iterations
        if old_start_iterative is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_START_ITERATIVE", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_START_ITERATIVE"] = old_start_iterative
    print("test_collision_at_start_can_use_iterative_world_separation: PASSED")
    return True


@_legacy_contact_response_test
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


@_legacy_contact_response_test
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


@_legacy_contact_response_test
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


def _run_dirty_model_center_probe(dirty_model_center_mode):
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_COLLISION_SHAPE",
        "WULFRAM_ENTITY_TERRAIN_DIRTY_MODEL_CENTER",
        "WULFRAM_ENTITY_TERRAIN_MODEL_CONTACT_SELECTION",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_COLLISION_SHAPE"] = "model"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_MODEL_CONTACT_SELECTION", None)
        if dirty_model_center_mode is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_DIRTY_MODEL_CENTER", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_DIRTY_MODEL_CENTER"] = dirty_model_center_mode

        calls = []

        def fake_model_bounds_contact(*args, **kwargs):
            calls.append((args, kwargs))
            return TerrainContact(
                position=(10.0, 0.0, 4.0),
                normal=(0.0, 0.0, 1.0),
                penetration=1.5,
                sector_index=0,
                cell=(0, 0),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_model_bounds_contact=fake_model_bounds_contact,
            raycast=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("dirty bounds contact should resolve before raycast")
            ),
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=lambda *args, **kwargs: None,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1.0
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
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

        result = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            4.0,
            1.0,
            0.0,
            -5.0,
        )
        return calls, ctx, result
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@_legacy_contact_response_test
def test_entity_world_collision_dirty_model_center_defaults_to_lifted_center():
    """Dirty model bounds should preserve the current lifted collision center by default."""
    calls, ctx, result = _run_dirty_model_center_probe(None)
    assert calls, calls
    args, kwargs = calls[0]
    assert args[0] == (10.0, 0.0, 4.0), args
    assert args[1] == (10.0, 0.0, 7.0), args
    assert kwargs["contact_selection"] == "first", kwargs
    assert ctx.debug_last_motion_collision["dirty_model_center_mode"] == "lift", ctx.debug_last_motion_collision
    assert ctx.debug_last_motion_collision["dirty_collision_center"] == (10.0, 0.0, 7.0), ctx.debug_last_motion_collision
    px, py, pz, vx, vy, vz = result
    assert abs(px - 10.0) < 1e-6, result
    assert abs(py) < 1e-6, result
    assert pz > 4.0, result
    assert abs(vx - 1.0) < 1e-6, result
    assert abs(vy) < 1e-6, result
    assert abs(vz) < 1e-6, result
    print("test_entity_world_collision_dirty_model_center_defaults_to_lifted_center: PASSED")
    return True


@_legacy_contact_response_test
def test_entity_world_collision_dirty_model_center_can_use_raw_center():
    """Opt-in dirty model bounds can use the raw entity center seen in the decompile path."""
    calls, ctx, result = _run_dirty_model_center_probe("raw")
    assert calls, calls
    args, kwargs = calls[0]
    assert args[0] == (10.0, 0.0, 4.0), args
    assert args[1] == (10.0, 0.0, 4.0), args
    assert kwargs["contact_selection"] == "first", kwargs
    assert ctx.debug_last_motion_collision["dirty_model_center_mode"] == "raw", ctx.debug_last_motion_collision
    assert ctx.debug_last_motion_collision["dirty_collision_center"] == (10.0, 0.0, 4.0), ctx.debug_last_motion_collision
    px, py, pz, vx, vy, vz = result
    assert abs(px - 10.0) < 1e-6, result
    assert abs(py) < 1e-6, result
    assert pz > 4.0, result
    assert abs(vx - 1.0) < 1e-6, result
    assert abs(vy) < 1e-6, result
    assert abs(vz) < 1e-6, result
    print("test_entity_world_collision_dirty_model_center_can_use_raw_center: PASSED")
    return True


@_legacy_contact_response_test
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


@_legacy_contact_response_test
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


def test_entity_world_collision_downward_dirty_bounds_contact_falls_back_to_raycast():
    """Dirty terrain contacts with downward normals should not push idle tanks under rough terrain."""
    raycast_calls = 0

    def fake_model_bounds_contact(*args, **kwargs):
        return TerrainContact(
            position=(10.0, 0.0, 4.0),
            normal=(0.2, 0.0, -0.98),
            penetration=2.0,
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
    assert ctx.debug_last_collision["kind"] == "terrain_dirty_raycast", ctx.debug_last_collision
    assert abs(px - 4.0) < 1e-6, px
    assert abs(py) < 1e-6, py
    assert 4.0 < pz < 5.0, pz
    assert abs(vx - 1.0) < 1e-6, vx
    assert abs(vy) < 1e-6, vy
    assert abs(vz) < 1e-6, vz
    print("test_entity_world_collision_downward_dirty_bounds_contact_falls_back_to_raycast: PASSED")
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


def test_decompile_bounds_contact_reports_sat_metadata():
    """Box-bounds SAT helper should expose a report-only decompile terrain lane."""
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

    grid._iter_aabb_sectors = lambda aabb_min, aabb_max: [SimpleNamespace(index=7)]
    grid._iter_sector_cells = lambda aabb_min, aabb_max, sector: [(3, 4)]
    grid._iter_cell_triangles = lambda cell_x, cell_y: [tri]
    grid._triangle_box_contact = lambda tri_local, half_extents: ((0.0, 0.0, 1.0), 0.75)

    contact = grid.test_box_bounds_contact(
        (0.5, 0.5, 1.0),
        (0.5, 0.5, 1.0),
        (1.0, 1.0, 1.0),
        0.0,
        1.0,
        rotation_matrix=(
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        contact_selection="upward_min_depth",
    )

    assert contact is not None
    assert contact.normal_source == "terrain_bounds_sat"
    assert contact.cell == (3, 4)
    assert contact.sector_index == 7
    assert contact.terrain_face_normal is not None
    assert contact.terrain_face_normal[2] > 0.99
    assert contact.entity_radial_normal is not None
    assert contact.entity_radial_normal[2] > 0.99
    assert abs(contact.penetration - 0.75) < 1e-6
    print("test_decompile_bounds_contact_reports_sat_metadata: PASSED")
    return True


def _make_test_terrain(size_x: int, size_z: int, heights: list[float]) -> Terrain:
    terrain = Terrain.__new__(Terrain)
    terrain.num_x = size_x
    terrain.num_z = size_z
    terrain.world_w = float(size_x - 1)
    terrain.world_h = float(size_z - 1)
    terrain.cell_x = 1.0
    terrain.cell_z = 1.0
    terrain.inv_cell_x = 1.0
    terrain.inv_cell_z = 1.0
    terrain._heights = list(heights)
    terrain._cell_types = [0] * (size_x * size_z)
    return terrain


def test_terrain_height_uses_decompile_triangle_plane_not_bilinear():
    """Rough-cell height sampling should use the same triangle split as the client spring query."""
    # Heights are stored [x * num_z + z]. On odd cell (1, 0), the upper-left
    # triangle excludes h11, so this point should remain at zero. Bilinear
    # sampling would incorrectly blend h11 in and return 6.25.
    terrain = _make_test_terrain(
        3,
        2,
        [
            0.0, 0.0,
            0.0, 0.0,
            0.0, 100.0,
        ],
    )

    assert terrain.get_height(1.25, 0.25) == 0.0
    assert terrain.get_height(1.75, 0.75) == 50.0
    print("test_terrain_height_uses_decompile_triangle_plane_not_bilinear: PASSED")
    return True


def test_terrain_slope_uses_active_triangle_plane_normal():
    """Terrain slope should come from the active triangle plane, not central bilinear smoothing."""
    terrain = _make_test_terrain(
        2,
        2,
        [
            0.0, 3.0,
            2.0, 5.0,
        ],
    )

    height, normal = terrain.sample_height_normal(0.25, 0.75)
    dh_dx, dh_dy = terrain.get_slope(0.25, 0.75)

    assert abs(height - 2.75) < 1e-6, height
    assert abs(dh_dx - 2.0) < 1e-6, dh_dx
    assert abs(dh_dy - 3.0) < 1e-6, dh_dy
    assert normal[2] > 0.0, normal
    print("test_terrain_slope_uses_active_triangle_plane_normal: PASSED")
    return True


def test_player_terrain_probe_reports_decompile_triangle_state():
    """Live player telemetry should preserve the terrain data needed for rough-terrain diagnosis."""
    terrain = _make_test_terrain(
        2,
        2,
        [
            0.0, 3.0,
            2.0, 5.0,
        ],
    )
    terrain._cell_types = [7, 8, 9, 10]
    server = SimpleNamespace(
        terrain=terrain,
        terrain_height_offset=5.0,
        terrain_physics_height_offset=0.5,
    )

    probe = build_player_terrain_probe(server, (0.25, 0.75, 4.0), 0.0)

    assert probe["source"] == "GUESS3_Terrain_interpolate_grid_height"
    assert probe["cell"] == [0, 0]
    assert probe["cell_type"] == 7
    assert abs(probe["raw_height"] - 2.75) < 1e-5
    assert abs(probe["physics_ground_z"] - 3.25) < 1e-5
    assert abs(probe["clearance_z"] - 0.75) < 1e-5
    assert probe["normal"][2] > 0.0
    assert probe["slope"] == [2.0, 3.0]
    print("test_player_terrain_probe_reports_decompile_triangle_state: PASSED")
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


@_legacy_contact_response_test
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


def test_entity_world_collision_preserves_reference_on_clean_miss_until_dirty():
    """Clean lifted misses should accumulate against the persistent dirty reference."""
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
    server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1.0

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
        0.5,
        0.0,
        5.0,
        0.0,
        0.0,
        0.0,
    )
    assert ctx.world_collision_bounds_dirty is False, ctx.world_collision_bounds_dirty
    assert raycast_calls == [], raycast_calls
    assert (px, py, pz, vx, vy, vz) == (0.5, 0.0, 5.0, 0.0, 0.0, 0.0)
    assert ctx.world_collision_ref_pos == (0.0, 0.0, 5.0), ctx.world_collision_ref_pos

    px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
        ctx,
        1.1,
        0.0,
        5.0,
        0.0,
        0.0,
        0.0,
    )
    assert ctx.world_collision_bounds_dirty is True, ctx.world_collision_bounds_dirty
    assert raycast_calls == [((0.0, 0.0, 5.0), (1.1, 0.0, 5.0))], raycast_calls
    assert (px, py, pz, vx, vy, vz) == (1.1, 0.0, 5.0, 0.0, 0.0, 0.0)
    assert ctx.world_collision_ref_pos == (1.1, 0.0, 5.0), ctx.world_collision_ref_pos
    print("test_entity_world_collision_preserves_reference_on_clean_miss_until_dirty: PASSED")
    return True


def test_entity_world_collision_can_preserve_dirty_miss_reference_for_probe():
    """The dirty-miss reference refresh is default-on but can be held for decompile A/Bs."""
    old_refresh = os.environ.get("WULFRAM_ENTITY_TERRAIN_DIRTY_MISS_REFRESH")
    old_fallback = os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK")
    os.environ["WULFRAM_ENTITY_TERRAIN_DIRTY_MISS_REFRESH"] = "0"
    os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
    try:
        raycast_calls = []
        model_calls = []

        def fake_raycast(start, end):
            raycast_calls.append((start, end))
            return None

        def fake_model_collision(center, *args, **kwargs):
            model_calls.append(center)
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_model_bounds_contact=lambda *args, **kwargs: None,
            raycast=fake_raycast,
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1.0

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
            0.0,
        )

        assert ctx.world_collision_bounds_dirty is True, ctx.world_collision_bounds_dirty
        assert (px, py, pz, vx, vy, vz) == (10.0, 0.0, 4.0, 1.0, 0.0, 0.0)
        assert ctx.world_collision_ref_pos == (0.0, 0.0, 4.0), ctx.world_collision_ref_pos
        assert raycast_calls == [((0.0, 0.0, 4.0), (10.0, 0.0, 4.0))], raycast_calls
        assert model_calls[0] == (10.0, 0.0, 7.0), model_calls
        assert model_calls[1] == (10.0, 0.0, 4.0), model_calls
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["reason"] == "lifted_clear", probe
        assert probe["dirty_bounds_active"] is True, probe
        assert probe["dirty_miss_refresh_enabled"] is False, probe
        assert probe["dirty_miss_ref_action"] == "preserved", probe
        assert probe["dirty_miss_reason"] == "dirty_bounds_clear", probe
        assert probe["dirty_raycast_reject"] == "no_terrain_raycast_hit", probe
        assert probe["dirty_reference_pos"] == (0.0, 0.0, 4.0), probe
        assert probe["dirty_current_pos"] == (10.0, 0.0, 4.0), probe
        assert probe["dirty_displacement_sq"] == 100.0, probe
        assert probe["dirty_threshold_sq"] == 1.0, probe
    finally:
        if old_refresh is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_DIRTY_MISS_REFRESH", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_DIRTY_MISS_REFRESH"] = old_refresh
        if old_fallback is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = old_fallback
    print("test_entity_world_collision_can_preserve_dirty_miss_reference_for_probe: PASSED")
    return True


def test_entity_world_collision_dirty_reference_pair_probe_is_read_only():
    """The dirty-reference pair probe reports missed pair contacts without applying them."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_PROBE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_PROBE"] = "1"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = (
            "upward_min_depth"
        )
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)

        model_calls = []
        raycast_calls = []

        def fake_model_collision(center, *args, **kwargs):
            model_calls.append((tuple(center), kwargs.get("contact_selection")))
            if (
                abs(center[0]) < 1e-6
                and abs(center[1]) < 1e-6
                and abs(center[2] - 4.0) < 1e-6
            ):
                return TerrainContact(
                    position=(0.0, 0.0, 2.5),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.5,
                    sector_index=0,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        def fake_raycast(start, end):
            raycast_calls.append((start, end))
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_model_bounds_contact=lambda *args, **kwargs: None,
            test_bounds_intersection=lambda *args, **kwargs: True,
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
            raycast=fake_raycast,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1.0

        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(),
            entity_id=0x14EA,
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 4.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

        result = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            4.0,
            1.0,
            0.0,
            0.0,
            pre_pos=(0.0, 0.0, 4.0),
            pre_vel=(1.0, 0.0, 0.0),
            dt=1.0 / 30.0,
        )

        assert result == (10.0, 0.0, 4.0, 1.0, 0.0, 0.0), result
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}, (
            getattr(ctx, "debug_last_motion_collision", {})
        )
        assert ctx.world_collision_ref_pos == (10.0, 0.0, 4.0), (
            ctx.world_collision_ref_pos
        )
        assert raycast_calls == [((0.0, 0.0, 4.0), (10.0, 0.0, 4.0))], (
            raycast_calls
        )
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["dirty_bounds_active"] is True, probe
        assert probe["dirty_bounds_xy_overlap"] is True, probe
        assert probe["dirty_miss_reason"] == "dirty_bounds_clear", probe
        assert probe["dirty_reference_pair_probe_enabled"] is True, probe
        dirty_probe = probe["dirty_reference_pair_probe"]
        assert dirty_probe["reject"] == "", dirty_probe
        assert dirty_probe["accept_labels"] == ["dirty_reference_pos"], dirty_probe
        reference = dirty_probe["results"]["dirty_reference_pos"]
        assert reference["accepted"] is True, reference
        assert reference["contact"]["contact_cell"] == (7, 8), reference
        assert dirty_probe["results"]["dirty_current_pos"]["reject"] == (
            "no_raw_origin_contact"
        ), dirty_probe
        assert any(
            call == ((0.0, 0.0, 4.0), "upward_min_depth") for call in model_calls
        ), model_calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_entity_world_collision_dirty_reference_pair_probe_is_read_only: PASSED")
    return True


def test_entity_world_collision_dirty_reference_pair_response_preserves_position():
    """Dirty-reference pair response applies only the contact velocity delta."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_RESPONSE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_RESPONSE_MAX_DISTANCE",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_RESPONSE"] = "apply"
        os.environ[
            "WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_RESPONSE_MAX_DISTANCE"
        ] = "20.0"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = (
            "upward_min_depth"
        )
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)

        model_calls = []
        raycast_calls = []

        def fake_model_collision(center, *args, **kwargs):
            model_calls.append((tuple(center), kwargs.get("contact_selection")))
            if (
                abs(center[0]) < 1e-6
                and abs(center[1]) < 1e-6
                and abs(center[2] - 4.0) < 1e-6
            ):
                return TerrainContact(
                    position=(0.0, 0.0, 2.5),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.5,
                    sector_index=0,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        def fake_raycast(start, end):
            raycast_calls.append((start, end))
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_model_bounds_contact=lambda *args, **kwargs: None,
            test_bounds_intersection=lambda *args, **kwargs: True,
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
            raycast=fake_raycast,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1.0

        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(),
            entity_id=0x14EA,
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 4.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

        result = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            4.0,
            1.0,
            0.0,
            -5.0,
            pre_pos=(0.0, 0.0, 4.0),
            pre_vel=(1.0, 0.0, -5.0),
            dt=1.0 / 30.0,
        )

        px, py, pz, vx, vy, vz = result
        assert (px, py, pz) == (10.0, 0.0, 4.0), result
        assert vz > -5.0, result
        assert vz - (-5.0) <= 0.500001, result
        assert raycast_calls == [], raycast_calls
        assert ctx.world_collision_ref_pos == (10.0, 0.0, 4.0), (
            ctx.world_collision_ref_pos
        )
        collision = ctx.debug_last_collision
        assert collision["kind"] == "terrain_dirty_reference_pair_response", collision
        assert collision["pair_record_contact_reason"] == (
            "dirty_reference_pair_response"
        ), collision
        assert collision["dirty_reference_pair_response_preserved_position"] is True, (
            collision
        )
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["dirty_reference_pair_response_enabled"] is True, probe
        assert probe["dirty_reference_pair_response_apply_enabled"] is True, probe
        response = probe["dirty_reference_pair_response"]
        assert response["applied"] is True, response
        assert response["applied_to_current_state"] is True, response
        assert response["label"] == "dirty_reference_pos", response
        assert response["max_distance"] == 20.0, response
        assert abs(response["current_distance"] - 10.0) < 1e-6, response
        assert abs(response["current_xy_distance"] - 10.0) < 1e-6, response
        assert response["current_z_delta"] == 0.0, response
        assert response["final_vel"][2] == vz, response
        assert collision["dirty_reference_pair_response_max_distance"] == 20.0, (
            collision
        )
        assert abs(
            collision["dirty_reference_pair_response_current_distance"] - 10.0
        ) < 1e-6, collision
        assert any(
            call == ((0.0, 0.0, 4.0), "upward_min_depth") for call in model_calls
        ), model_calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_dirty_reference_pair_response_preserves_position: PASSED"
    )
    return True


def test_entity_world_collision_dirty_reference_pair_response_max_distance_rejects():
    """Dirty-reference pair response can be distance-gated without moving state."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_RESPONSE",
        "WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_RESPONSE_MAX_DISTANCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_RESPONSE"] = "apply"
        os.environ[
            "WULFRAM_ENTITY_TERRAIN_DIRTY_REFERENCE_PAIR_RESPONSE_MAX_DISTANCE"
        ] = "1.5"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT", None)
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_CONTACT_SELECTION"] = (
            "upward_min_depth"
        )
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_NORMAL_SOURCE"] = "mesh"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA"] = "0.5"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)

        model_calls = []
        raycast_calls = []

        def fake_model_collision(center, *args, **kwargs):
            model_calls.append((tuple(center), kwargs.get("contact_selection")))
            if (
                abs(center[0]) < 1e-6
                and abs(center[1]) < 1e-6
                and abs(center[2] - 4.0) < 1e-6
            ):
                return TerrainContact(
                    position=(0.0, 0.0, 2.5),
                    normal=(0.0, 0.0, 1.0),
                    penetration=1.5,
                    sector_index=0,
                    cell=(7, 8),
                    normal_source="entity_cbsp_split",
                    cbsp_split_normal=(0.0, 0.0, 1.0),
                    terrain_face_normal=(0.0, 0.0, 1.0),
                    mesh_face_normal=(0.0, 0.0, 1.0),
                    entity_radial_normal=(0.0, 0.0, 1.0),
                )
            return None

        def fake_raycast(start, end):
            raycast_calls.append((start, end))
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_model_bounds_contact=lambda *args, **kwargs: None,
            test_bounds_intersection=lambda *args, **kwargs: True,
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
            raycast=fake_raycast,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1.0

        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(),
            entity_id=0x14EA,
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 4.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

        result = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            4.0,
            1.0,
            0.0,
            -5.0,
            pre_pos=(0.0, 0.0, 4.0),
            pre_vel=(1.0, 0.0, -5.0),
            dt=1.0 / 30.0,
        )

        assert result == (10.0, 0.0, 4.0, 1.0, 0.0, -5.0), result
        probe = ctx.debug_last_terrain_contact_probe
        response = probe["dirty_reference_pair_response"]
        assert response["reject"] == "dirty_reference_pair_response_too_far", (
            response
        )
        assert response["applied"] is False, response
        assert response["max_distance"] == 1.5, response
        assert abs(response["current_distance"] - 10.0) < 1e-6, response
        assert abs(response["current_xy_distance"] - 10.0) < 1e-6, response
        assert response["current_z_delta"] == 0.0, response
        assert raycast_calls == [((0.0, 0.0, 4.0), (10.0, 0.0, 4.0))], (
            raycast_calls
        )
        assert any(
            call == ((0.0, 0.0, 4.0), "upward_min_depth") for call in model_calls
        ), model_calls
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_dirty_reference_pair_response_max_distance_rejects: PASSED"
    )
    return True


def test_entity_world_collision_dirty_bounds_safe_response_limits_box_fallback():
    """Dirty AABB fallback can reuse the safety-limited pair response instead of raw solver output."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_SAFE_RESPONSE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_FALLBACK"] = "decompile"
        os.environ["WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_SAFE_RESPONSE"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA"] = "0.25"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA"] = "0.25"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)

        box_calls = []
        raycast_calls = []

        def fake_box_bounds_contact(*args, **kwargs):
            box_calls.append(args)
            return TerrainContact(
                position=(10.0, 0.0, 3.0),
                normal=(0.0, 0.0, 1.0),
                penetration=2.0,
                sector_index=0,
                cell=(2, 3),
                normal_source="dirty_aabb_test",
            )

        def fake_raycast(start, end):
            raycast_calls.append((start, end))
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_model_bounds_contact=lambda *args, **kwargs: None,
            test_box_bounds_contact=fake_box_bounds_contact,
            test_bounds_intersection=lambda *args, **kwargs: True,
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=lambda *args, **kwargs: None,
            raycast=fake_raycast,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1.0

        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(),
            entity_id=0x14EA,
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 4.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

        result = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            4.0,
            0.0,
            0.0,
            -10.0,
            pre_pos=(0.0, 0.0, 4.0),
            pre_vel=(0.0, 0.0, -10.0),
            dt=1.0 / 30.0,
        )

        assert box_calls, box_calls
        assert raycast_calls == [], raycast_calls
        px, py, pz, vx, vy, vz = result
        assert all(math.isfinite(value) for value in result), result
        assert vz - (-10.0) <= 0.250001, result
        assert ctx.world_collision_ref_pos == (px, py, pz), ctx.world_collision_ref_pos
        collision = ctx.debug_last_collision
        assert collision["kind"] == "terrain_dirty_bounds_safety_limited", collision
        assert collision["dirty_bounds_safe_response"] is True, collision
        assert collision["dirty_bounds_safe_response_applied"] is True, collision
        assert collision["raw_origin_fallback_velocity_delta_clamped"] in {
            True,
            False,
        }, collision
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["dirty_bounds_safe_response_enabled"] is True, probe
        assert probe["dirty_bounds_box_fallback_applied"] is True, probe
        assert probe["dirty_bounds_box_contact"]["contact_cell"] == (2, 3), probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "test_entity_world_collision_dirty_bounds_safe_response_limits_box_fallback: PASSED"
    )
    return True


def test_pair_record_bounds_sat_can_feed_safety_limited_contact():
    """Default-off terrain-bounds SAT contact can feed the pair-record response path."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_BOUNDS_SAT",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE",
        "WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_BOUNDS_SAT"] = "apply"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VELOCITY_DELTA"] = "0.25"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_MAX_VERTICAL_DELTA"] = "0.25"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_DELTA_MODE"] = "closing_velocity"
        os.environ["WULFRAM_ENTITY_TERRAIN_PAIR_RECORD_ANGULAR_MODE"] = "preserve"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)

        box_calls = []

        def fake_box_bounds_contact(*args, **kwargs):
            box_calls.append((args, kwargs))
            return TerrainContact(
                position=(0.0, 0.0, 3.0),
                normal=(0.0, 0.0, 1.0),
                penetration=1.0,
                sector_index=0,
                cell=(2, 3),
                normal_source="terrain_bounds_sat",
                terrain_face_normal=(0.0, 0.0, 1.0),
                entity_radial_normal=(0.6, 0.0, 0.8),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_model_bounds_contact=lambda *args, **kwargs: None,
            test_box_bounds_contact=fake_box_bounds_contact,
            test_bounds_intersection=lambda *args, **kwargs: False,
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=lambda *args, **kwargs: None,
            raycast=lambda *args, **kwargs: None,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 9999.0

        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(),
            entity_id=0x14EA,
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 4.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

        result = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            4.0,
            0.0,
            0.0,
            -10.0,
            pre_pos=(0.0, 0.0, 4.0),
            pre_vel=(0.0, 0.0, -10.0),
            dt=1.0 / 30.0,
        )

        assert box_calls, box_calls
        assert any(
            call_kwargs.get("contact_selection") == "upward_min_depth"
            for _call_args, call_kwargs in box_calls
        ), box_calls
        px, py, pz, vx, vy, vz = result
        assert all(math.isfinite(value) for value in result), result
        assert vz - (-10.0) <= 0.250001, result
        collision = ctx.debug_last_collision
        assert collision["kind"] == "terrain_pair_record_contact", collision
        assert collision["pair_record_contact_reason"] == (
            "lifted_clear_raw_origin_bounds_contact"
        ), collision
        assert collision["pair_record_bounds_sat_apply_enabled"] is True, collision
        assert collision["pair_record_solver_normal_source"] == "terrain_bounds_sat", collision
        assert collision["pair_record_delta_normal_source"] == (
            "entity_radial_terrain_face_blend"
        ), collision
        assert collision["raw_origin_fallback_delta_normal_source"] == (
            "entity_radial_terrain_face_blend"
        ), collision
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["pair_record_bounds_sat_enabled"] is True, probe
        assert probe["pair_record_bounds_sat_apply_enabled"] is True, probe
        assert probe["pair_record_contact_delta_normal_source"] == (
            "entity_radial_terrain_face_blend"
        ), probe
        assert probe["raw_origin_bounds_contact"]["contact_normal_source"] == (
            "terrain_bounds_sat"
        ), probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_pair_record_bounds_sat_can_feed_safety_limited_contact: PASSED")
    return True


def test_entity_world_collision_dirty_miss_still_records_raw_origin_probe():
    """Dirty lifted misses should still expose the clean/raw-origin probe reason."""
    old_fallback = os.environ.get("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK")
    os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
    try:
        model_calls = []

        def fake_model_collision(center, *args, **kwargs):
            model_calls.append(center)
            if abs(center[2] - 4.0) < 1e-6:
                return TerrainContact(
                    position=(10.0, 0.0, 3.0),
                    normal=(0.0, 0.0, 1.0),
                    penetration=2.0,
                    sector_index=0,
                    cell=(0, 0),
                    normal_source="terrain_triangle",
                )
            return None

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_model_bounds_contact=lambda *args, **kwargs: None,
            raycast=lambda *args, **kwargs: None,
            test_box_collision=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1.0

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
            0.0,
        )

        assert ctx.world_collision_bounds_dirty is True, ctx.world_collision_bounds_dirty
        assert (px, py, pz, vx, vy, vz) == (10.0, 0.0, 4.0, 1.0, 0.0, 0.0)
        assert ctx.world_collision_ref_pos == (10.0, 0.0, 4.0), ctx.world_collision_ref_pos
        assert model_calls[0] == (10.0, 0.0, 7.0), model_calls
        assert model_calls[1] == (10.0, 0.0, 4.0), model_calls
        assert getattr(ctx, "debug_last_motion_collision", {}) == {}
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["reason"] == "lifted_clear_raw_origin_contact", probe
        assert probe["dirty_bounds_active"] is True, probe
        assert probe["dirty_miss_refresh_enabled"] is True, probe
        assert probe["dirty_miss_ref_action"] == "refreshed", probe
        assert probe["dirty_miss_reason"] == "dirty_bounds_clear", probe
        assert probe["dirty_raycast_reject"] == "no_terrain_raycast_hit", probe
        assert probe["raw_origin_fallback_enabled"] is False, probe
        assert probe["raw_origin_fallback_reject"] == "disabled", probe
        assert probe["raw_origin_contact"]["depth"] == 2.0, probe
    finally:
        if old_fallback is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK"] = old_fallback
    print("test_entity_world_collision_dirty_miss_still_records_raw_origin_probe: PASSED")
    return True


def test_entity_world_collision_dirty_model_clear_skips_box_fallback_by_default():
    """A model-bounds miss must not silently switch to the box dirty phase without the probe env."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_COLLISION_SHAPE",
        "WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_COLLISION_SHAPE"] = "model"
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_FALLBACK", None)
        os.environ.pop("WULFRAM_ENTITY_TERRAIN_RAW_ORIGIN_FALLBACK", None)
        raycast_calls = []
        box_bounds_calls = []
        bounds_overlap_calls = []

        def fake_raycast(start, end):
            raycast_calls.append((start, end))
            return None

        def fake_bounds_overlap(aabb_min, aabb_max):
            bounds_overlap_calls.append((aabb_min, aabb_max))
            return True

        def fake_box_bounds_contact(*args, **kwargs):
            box_bounds_calls.append((args, kwargs))
            return TerrainContact(
                position=(10.0, 0.0, 4.0),
                normal=(0.0, 0.0, 1.0),
                penetration=1.5,
                sector_index=0,
                cell=(0, 0),
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_model_bounds_contact=lambda *args, **kwargs: None,
            test_bounds_intersection=fake_bounds_overlap,
            test_box_bounds_contact=fake_box_bounds_contact,
            raycast=fake_raycast,
            test_model_collision=lambda *args, **kwargs: None,
            test_box_collision=lambda *args, **kwargs: None,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1.0

        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(),
            entity_id=0x14EA,
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 4.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 4.0)

        result = server._resolve_entity_world_collision(
            ctx,
            10.0,
            0.0,
            4.0,
            1.0,
            0.0,
            0.0,
        )

        assert result == (10.0, 0.0, 4.0, 1.0, 0.0, 0.0), result
        assert box_bounds_calls == [], box_bounds_calls
        assert bounds_overlap_calls == [((5.0, -5.0, -1.0), (15.0, 5.0, 9.0))], bounds_overlap_calls
        assert raycast_calls == [((0.0, 0.0, 4.0), (10.0, 0.0, 4.0))], raycast_calls
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["dirty_bounds_aabb_min"] == (5.0, -5.0, -1.0), probe
        assert probe["dirty_bounds_aabb_max"] == (15.0, 5.0, 9.0), probe
        assert probe["dirty_bounds_xy_overlap"] is True, probe
        assert probe["dirty_bounds_box_fallback_enabled"] is False, probe
        assert "dirty_bounds_box_fallback_attempted" not in probe or probe[
            "dirty_bounds_box_fallback_attempted"
        ] is None, probe
        assert probe["dirty_miss_reason"] == "dirty_bounds_clear", probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_entity_world_collision_dirty_model_clear_skips_box_fallback_by_default: PASSED")
    return True


@_legacy_contact_response_test
def test_entity_world_collision_dirty_model_clear_can_use_box_fallback():
    """Opt-in rough-terrain probe can test the decompile-style box dirty phase after model bounds miss."""
    env_keys = [
        "WULFRAM_ENTITY_TERRAIN_COLLISION_SHAPE",
        "WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_FALLBACK",
        "WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_SHAPE",
        "WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_CENTER",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_COLLISION_SHAPE"] = "model"
        os.environ["WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_FALLBACK"] = "1"
        os.environ["WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_SHAPE"] = "inertia"
        os.environ["WULFRAM_ENTITY_TERRAIN_DIRTY_BOUNDS_BOX_CENTER"] = "raw"
        raycast_calls = []
        model_bounds_calls = []
        box_bounds_calls = []

        def fake_raycast(start, end):
            raycast_calls.append((start, end))
            return None

        def fake_model_bounds_contact(*args, **kwargs):
            model_bounds_calls.append((args, kwargs))
            return None

        def fake_box_bounds_contact(*args, **kwargs):
            box_bounds_calls.append((args, kwargs))
            return TerrainContact(
                position=(10.0, 0.0, 4.0),
                normal=(0.0, 0.0, 1.0),
                penetration=1.5,
                sector_index=0,
                cell=(0, 0),
                normal_source="terrain_box_sat",
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_model_bounds_contact=fake_model_bounds_contact,
            test_box_bounds_contact=fake_box_bounds_contact,
            raycast=fake_raycast,
            test_model_collision=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("clean model contact should not run after dirty fallback")
            ),
            test_box_collision=lambda *args, **kwargs: None,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=5.0)),
            5.0,
            3.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 1.0

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

        assert model_bounds_calls, model_bounds_calls
        assert box_bounds_calls, box_bounds_calls
        args, kwargs = box_bounds_calls[0]
        assert args == ((10.0, 0.0, 4.0), (10.0, 0.0, 4.0), (4.0, 4.0, 4.0), 0.0, 5.0), args
        assert kwargs == {
            "rotation_matrix": None,
            "contact_selection": "upward_min_depth",
        }, kwargs
        assert raycast_calls == [], raycast_calls
        assert abs(px - 10.0) < 1e-6, (px, py, pz, vx, vy, vz)
        assert abs(py) < 1e-6, (px, py, pz, vx, vy, vz)
        assert pz > 5.0, (px, py, pz, vx, vy, vz)
        assert abs(vx - 1.0) < 1e-6, (px, py, pz, vx, vy, vz)
        assert abs(vy) < 1e-6, (px, py, pz, vx, vy, vz)
        assert abs(vz) < 1e-6, (px, py, pz, vx, vy, vz)
        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_dirty_bounds", debug
        assert debug["dirty_bounds_contact_source"] == "box_bounds_fallback", debug
        assert debug["dirty_bounds_box_fallback_applied"] is True, debug
        assert debug["dirty_bounds_box_half_extents_source"] == "inertia_half_extents", debug
        probe = ctx.debug_last_terrain_contact_probe
        assert probe["reason"] == "box_bounds_fallback_contact", probe
        assert probe["dirty_bounds_box_fallback_applied"] is True, probe
        assert probe["dirty_bounds_box_contact"]["contact_normal_source"] == "terrain_box_sat", probe
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("test_entity_world_collision_dirty_model_clear_can_use_box_fallback: PASSED")
    return True


def test_entity_world_collision_can_probe_decompile_box_shape_with_model_loaded():
    """The rough-terrain A/B can use the decompile terrain-vs-entity hull path instead of model CBSP."""
    old_shape = os.environ.get("WULFRAM_ENTITY_TERRAIN_COLLISION_SHAPE")
    try:
        os.environ["WULFRAM_ENTITY_TERRAIN_COLLISION_SHAPE"] = "box"
        model_calls = []
        box_calls = []

        def fake_model_collision(center, *args, **kwargs):
            model_calls.append(center)
            return TerrainContact(
                position=(0.0, 0.0, 5.0),
                normal=(0.0, 0.0, 1.0),
                penetration=1.0,
                sector_index=0,
                cell=(0, 0),
                normal_source="model_should_not_be_used",
            )

        def fake_box_collision(center, half_extents, heading):
            box_calls.append((center, half_extents, heading))
            return TerrainContact(
                position=(0.0, 0.0, 4.5),
                normal=(0.0, 0.0, 1.0),
                penetration=0.1,
                sector_index=0,
                cell=(0, 0),
                normal_source="terrain_box_sat",
            )

        server = WulframServer.__new__(WulframServer)
        server._terrain_grid_collision = SimpleNamespace(
            test_model_bounds_contact=lambda *args, **kwargs: None,
            test_model_collision=fake_model_collision,
            test_box_bounds_contact=lambda *args, **kwargs: None,
            test_box_collision=fake_box_collision,
            raycast=lambda *args, **kwargs: None,
        )
        server._get_entity_world_half_extents = lambda ctx: (4.0, 4.0, 4.0)
        server._get_entity_world_collision_model = lambda ctx: (
            [
                SimpleNamespace(x=-6.0, y=-7.0, z=-2.0),
                SimpleNamespace(x=6.0, y=7.0, z=2.0),
            ],
            SimpleNamespace(nodes=[object()], root=SimpleNamespace(radius=9.0)),
            9.0,
            2.0,
        )
        server._get_entity_dirty_threshold_sq = lambda ctx, half_extents: 10000.0

        ctx = ClientContext(
            client_id=1,
            client_addr=("10.10.10.2", 50000),
            session=Session(team_id=1),
            entity_id=0x14EA,
        )
        ctx.player_heading = 0.0
        ctx.player_pos = (0.0, 0.0, 2.0)
        ctx.world_collision_ref_pos = (0.0, 0.0, 2.0)

        px, py, pz, vx, vy, vz = server._resolve_entity_world_collision(
            ctx,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            -1.0,
        )

        assert model_calls == [], model_calls
        assert box_calls == [((0.0, 0.0, 4.0), (6.0, 7.0, 2.0), 0.0)], box_calls
        assert pz > 2.0, (px, py, pz, vx, vy, vz)
        debug = ctx.debug_last_motion_collision
        assert debug["kind"] == "terrain_clean_contact", debug
        assert debug["terrain_collision_shape"] == "box", debug
        assert debug["contact_normal_source"] == "terrain_box_sat", debug
    finally:
        if old_shape is None:
            os.environ.pop("WULFRAM_ENTITY_TERRAIN_COLLISION_SHAPE", None)
        else:
            os.environ["WULFRAM_ENTITY_TERRAIN_COLLISION_SHAPE"] = old_shape
    print("test_entity_world_collision_can_probe_decompile_box_shape_with_model_loaded: PASSED")
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


@_legacy_contact_response_test
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


def test_remote_combat_observer_stats_gate_skips_nonparticipant_og():
    """Combat stat packets can exclude remote OG observers while keeping participants."""

    class TcpSink:
        def __init__(self):
            self.sent = []

        def send(self, payload, log=True):
            self.sent.append(payload)

    server = WulframServer.__new__(WulframServer)
    server.udp_handler = None
    server.remote_combat_observer_packets = False

    observer_session = Session()
    observer_session.in_game = True
    observer_ctx = ClientContext(
        client_id=1,
        client_addr=("10.10.10.2", 50000),
        session=observer_session,
        entity_id=0x14EA,
    )
    observer_ctx.tcp_handler = TcpSink()

    participant_session = Session()
    participant_session.in_game = True
    participant_session.player_id = 0x053B
    participant_session.team_id = 1
    participant_ctx = ClientContext(
        client_id=2,
        client_addr=("10.10.10.3", 50001),
        session=participant_session,
        entity_id=0x053B,
    )
    participant_ctx.tcp_handler = TcpSink()
    participant_ctx.kills = 1
    participant_ctx.deaths = 0

    loopback_session = Session()
    loopback_session.in_game = True
    loopback_ctx = ClientContext(
        client_id=3,
        client_addr=("127.0.0.1", 50002),
        session=loopback_session,
        entity_id=0x053C,
    )
    loopback_ctx.tcp_handler = TcpSink()

    server._snapshot_clients = lambda: [observer_ctx, participant_ctx, loopback_ctx]

    server._broadcast_player_stats(participant_ctx, participants=(participant_ctx,))

    assert observer_ctx.tcp_handler.sent == [], observer_ctx.tcp_handler.sent
    assert len(participant_ctx.tcp_handler.sent) == 1
    # Loopback fork retired (9ea5dbd, 2026-06-02): a nonparticipant loopback
    # observer is gated exactly like a remote OG observer.
    assert loopback_ctx.tcp_handler.sent == [], loopback_ctx.tcp_handler.sent
    print("test_remote_combat_observer_stats_gate_skips_nonparticipant_og: PASSED")
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
    assert ctx.control_pose_reset_pos == (100.0, 200.0, 300.0), ctx.control_pose_reset_pos
    assert ctx.control_pose_reset_time > 0.0, ctx.control_pose_reset_time
    assert ctx.last_pose_reset_source == "control_pos", ctx.last_pose_reset
    assert ctx.last_pose_reset["pos"] == [100.0, 200.0, 300.0], ctx.last_pose_reset
    assert list(ctx.pose_reset_history)[-1]["source"] == "control_pos", ctx.pose_reset_history
    assert ctx.player_pose["pos"] == (100.0, 200.0, 300.0), ctx.player_pose["pos"]
    assert ctx.player_pose["vel"] == (0.0, 0.0, 0.0), ctx.player_pose["vel"]
    assert abs(ctx.player_pose["yaw"] + math.radians(45.0)) < 1e-6, ctx.player_pose["yaw"]
    assert tuple(round(float(v), 6) for v in ctx.spring_body_matrix[:3]) == (
        round(math.cos(math.radians(45.0)), 6),
        round(-math.sin(math.radians(45.0)), 6),
        0.0,
    ), ctx.spring_body_matrix
    softbody_slots = {int(BehaviorSlot.UPWARD_THRUST), int(TANK_SOFTBODY_CONTROL_SLOT)}
    assert abs(tank_softbody_control_slot_value(ctx.weapon_system.behavior_slots) - OG_TANK_SOFTBODY_IDLE_SLOT5) < 1e-6
    for idx, value in enumerate(ctx.weapon_system.behavior_slots):
        if idx in softbody_slots:
            assert abs(value - OG_TANK_SOFTBODY_IDLE_SLOT5) < 1e-6, ctx.weapon_system.behavior_slots
        else:
            assert value == 0.0, ctx.weapon_system.behavior_slots
    assert ctx.weapon_system.prev_fire_state == 0.0, ctx.weapon_system.prev_fire_state
    assert ctx.weapon_system.fire_cooldown == 0.0, ctx.weapon_system.fire_cooldown
    print("test_control_pos_exact_reset_targets_specific_client: PASSED")
    return True


def test_control_pos_can_apply_live_tap_velocity():
    """Control `pos ... vel vx vy vz` should preserve live tap velocity."""
    server = SimpleNamespace(
        clients_lock=threading.Lock(),
        clients={},
        _get_network_tick=lambda _ctx: 778,
    )

    session = Session()
    session.udp_addr = ("127.0.0.1", 30000)
    ctx = ClientContext(
        client_id=8,
        client_addr=("127.0.0.1", 30000),
        session=session,
        entity_id=0x1235,
    )
    server.clients[ctx.client_id] = ctx

    control = ControlServer(port=0)
    control.server = server

    result = control._cmd_player_pos(["c8", "10", "20", "30", "vel", "1.5", "-2.0", "0.25"])

    assert "vel=(1.50, -2.00, 0.25)" in result, result
    assert ctx.player_pos == (10.0, 20.0, 30.0), ctx.player_pos
    assert ctx.player_vel == (1.5, -2.0, 0.25), ctx.player_vel
    assert ctx.player_pose["vel"] == (1.5, -2.0, 0.25), ctx.player_pose["vel"]
    assert ctx.last_sent_vel == (1.5, -2.0, 0.25), ctx.last_sent_vel
    assert abs(ctx.player_speed - math.sqrt(1.5 * 1.5 + 2.0 * 2.0 + 0.25 * 0.25)) < 1e-6
    print("test_control_pos_can_apply_live_tap_velocity: PASSED")
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
    assert abs(ctx.spring_body_matrix[0] - math.cos(math.radians(45.0))) < 1e-6
    assert abs(ctx.spring_body_matrix[3] - math.sin(math.radians(45.0))) < 1e-6
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


def test_tick_loop_start_guard_allows_one_live_thread():
    """Concurrent spawn paths should not start duplicate physics/update loops."""
    server = WulframServer.__new__(WulframServer)
    ctx = ClientContext(
        client_id=22,
        client_addr=("127.0.0.1", 2627),
        session=Session(),
        entity_id=1337,
    )
    entered = threading.Event()
    stop = threading.Event()
    calls = []
    old_tick_enabled = FEATURES.tick_loop_enabled

    def fake_tick_loop(loop_ctx):
        current = threading.current_thread()
        with loop_ctx.tick_lock:
            if loop_ctx.tick_thread is not current:
                return
        calls.append(current)
        entered.set()
        stop.wait(1.0)
        with loop_ctx.tick_lock:
            if loop_ctx.tick_thread is current:
                loop_ctx.tick_thread = None

    try:
        FEATURES.tick_loop_enabled = True
        server._tick_loop = fake_tick_loop

        assert server._ensure_tick_loop(ctx) is True
        assert entered.wait(1.0), "tick loop did not start"
        first_thread = ctx.tick_thread
        assert first_thread is not None
        assert server._ensure_tick_loop(ctx) is False
        stop.set()
        first_thread.join(timeout=1.0)

        assert len(calls) == 1
        assert ctx.tick_thread is None
    finally:
        stop.set()
        FEATURES.tick_loop_enabled = old_tick_enabled
    print("test_tick_loop_start_guard_allows_one_live_thread: PASSED")
    return True


def test_tick_pacer_preserves_capped_catchup_backlog():
    period = 1.0 / 30.0
    next_tick, sleep_dt = WulframServer._advance_tick_pacer(
        0.0,
        period,
        now=0.08,
        max_catchup_steps=5,
    )
    assert abs(next_tick - period) < 1e-9, (next_tick, sleep_dt)
    assert sleep_dt == 0.0, (next_tick, sleep_dt)

    capped_next, capped_sleep = WulframServer._advance_tick_pacer(
        0.0,
        period,
        now=1.0,
        max_catchup_steps=5,
    )
    expected = 1.0 - period * 4.0
    assert abs(capped_next - expected) < 1e-9, (capped_next, expected)
    assert capped_sleep == 0.0, (capped_next, capped_sleep)
    print("test_tick_pacer_preserves_capped_catchup_backlog: PASSED")
    return True


def test_client_weapon_fire_telemetry_records_input_projectiles():
    """Client-input projectile fire should leave structured control-plane evidence."""
    server = WulframServer.__new__(WulframServer)
    session = Session()
    session.username = "OgShooter"
    session.team_id = 2
    ctx = ClientContext(
        client_id=11,
        client_addr=("10.10.10.2", 52731),
        session=session,
        entity_id=0x611,
    )
    ctx.weapon_system = WeaponSystem()
    ctx.weapon_system.behavior_slots[12] = 1.0

    projectile = SimpleNamespace(entity_id=7001, entity_type=EntityType.PULSE_SHELL)
    server._record_client_weapon_fire(ctx, "ACTION_UPDATE", 1234, [projectile], 8.0)

    assert ctx.weapon_fire_count == 1
    assert ctx.last_weapon_fire_source == "ACTION_UPDATE"
    assert ctx.last_weapon_fire_client_tick == 1234
    assert ctx.last_weapon_fire_projectile_ids == [7001]
    assert ctx.last_weapon_fire_projectile_types == ["PULSE_SHELL"]
    assert ctx.last_weapon_fire_energy_spent == 8.0
    assert ctx.last_weapon_fire_input["direct_slots"]["12"] == 1.0
    assert ctx.last_weapon_fire_input["active_slots"]["12"] == 1.0
    print("test_client_weapon_fire_telemetry_records_input_projectiles: PASSED")
    return True


def test_client_hitscan_fire_telemetry_records_fire():
    """Instant-hit fire should leave structured audit telemetry."""
    server = WulframServer.__new__(WulframServer)
    server.clients_lock = threading.Lock()
    session = Session()
    session.username = "OgChain"
    session.team_id = 2
    session.in_game = True
    ctx = ClientContext(
        client_id=12,
        client_addr=("10.10.10.2", 52731),
        session=session,
        entity_id=0x612,
    )
    ctx.weapon_system = WeaponSystem()
    ctx.weapon_system.behavior_slots[BehaviorSlot.FIRE] = 1.0
    server.clients = {ctx.client_id: ctx}

    server._on_chain_gun_fire(ctx, ctx.player_pos, (0.0, 0.0, 0.0), ctx.session.team_id)

    assert ctx.hitscan_fire_count == 1
    assert ctx.last_hitscan_weapon_name == "Chain Gun"
    assert ctx.last_hitscan_fire_input["fire"] == 1.0
    assert ctx.last_hitscan_fire_input["active_slots"][str(BehaviorSlot.FIRE)] == 1.0
    assert ctx.last_hitscan_fire_input["direct_slots"] == {}
    print("test_client_hitscan_fire_telemetry_records_fire: PASSED")
    return True


def test_client_hitscan_fire_damages_lane_target():
    """Chain Gun should damage the nearest in-lane target without projectile traffic."""
    server = WulframServer.__new__(WulframServer)
    server.clients_lock = threading.Lock()

    attacker_session = Session(phase=Phase.IN_GAME, in_game=True, team_id=2)
    attacker_session.username = "OgChain"
    attacker = ClientContext(
        client_id=12,
        client_addr=("10.10.10.2", 52731),
        session=attacker_session,
        entity_id=0x612,
    )
    attacker.player_pos = (2600.0, 3040.0, 63.0)
    attacker.weapon_system = WeaponSystem()
    attacker.weapon_system.behavior_slots[BehaviorSlot.FIRE] = 1.0

    target_session = Session(phase=Phase.IN_GAME, in_game=True, team_id=2)
    target_session.username = "Target"
    target = ClientContext(
        client_id=13,
        client_addr=("127.0.0.1", 52732),
        session=target_session,
        entity_id=0x613,
    )
    target.player_pos = (2635.0, 3040.0, 63.0)
    server.clients = {attacker.client_id: attacker, target.client_id: target}

    server._on_chain_gun_fire(attacker, attacker.player_pos, (0.0, 0.0, 0.0), attacker.session.team_id)

    assert attacker.hitscan_fire_count == 1
    assert target.player_health == 0.8
    assert target.last_damage_source == "hitscan:Chain Gun"
    assert target.last_damage_amount == 0.20
    print("test_client_hitscan_fire_damages_lane_target: PASSED")
    return True


def test_caltrop_projectile_steers_toward_nearest_target_in_help_range():
    """Server Caltrop steering should pick a nearby enemy-shaped in-game target."""
    server = WulframServer.__new__(WulframServer)
    server.clients_lock = threading.Lock()
    server.up_axis = "z"
    server.pos_offset = 0.0

    owner = ClientContext(
        client_id=21,
        client_addr=("10.10.10.2", 52731),
        session=Session(phase=Phase.IN_GAME, in_game=True, team_id=2),
        entity_id=0x621,
    )
    owner.player_pos = (0.0, 0.0, 0.0)

    target = ClientContext(
        client_id=22,
        client_addr=("10.10.10.3", 52732),
        session=Session(phase=Phase.IN_GAME, in_game=True, team_id=1),
        entity_id=0x622,
    )
    target.player_pos = (60.0, 0.0, 0.0)

    far_target = ClientContext(
        client_id=23,
        client_addr=("10.10.10.4", 52733),
        session=Session(phase=Phase.IN_GAME, in_game=True, team_id=1),
        entity_id=0x623,
    )
    far_target.player_pos = (260.0, 0.0, 0.0)

    server.clients = {
        owner.client_id: owner,
        target.client_id: target,
        far_target.client_id: far_target,
    }
    proj = Projectile(
        entity_id=7015,
        entity_type=EntityType.CALTROP,
        owner_id=owner.entity_id,
        team=owner.session.team_id,
        pos=(0.0, 0.0, 0.0),
        vel=(0.0, 0.0, 0.0),
        spawn_time=time.monotonic(),
        lifetime=30.0,
    )

    selected = server._steer_caltrop_projectile(proj, owner, dt=1.0 / 15.0)

    assert selected is target, selected
    assert proj.vel[0] > 0.0, proj.vel
    assert abs(math.sqrt(sum(float(v) * float(v) for v in proj.vel)) - 32.0) < 0.001, proj.vel
    print("test_caltrop_projectile_steers_toward_nearest_target_in_help_range: PASSED")
    return True


def test_caltrop_projectile_damage_uses_light_bomblet_amount_and_cleanup():
    """Caltrop projectile hits should apply light damage and request projectile cleanup."""
    server = WulframServer.__new__(WulframServer)
    server.clients_lock = threading.Lock()
    server.clients = {}
    server.udp_handler = None
    server.pktlog = SimpleNamespace(enabled=False)
    server._get_network_tick = lambda _ctx: 0xCA17
    server._snapshot_in_game_clients = lambda: []
    server._broadcast_transient_fx = lambda _events: None
    server._projectile_packets_allowed_for_client = lambda _client: True
    server._send_packet_to_client = lambda *_args, **_kwargs: True
    server._debug_comm_allowed_for_client = lambda _client: True

    attacker_session = Session(phase=Phase.IN_GAME, in_game=True, team_id=2)
    attacker_session.username = "Caltropper"
    attacker = ClientContext(
        client_id=31,
        client_addr=("10.10.10.2", 52731),
        session=attacker_session,
        entity_id=0x631,
    )

    target_session = Session(phase=Phase.IN_GAME, in_game=True, team_id=1)
    target_session.username = "Target"
    target = ClientContext(
        client_id=32,
        client_addr=("10.10.10.3", 52732),
        session=target_session,
        entity_id=0x632,
    )
    target.player_health = 1.0

    proj = Projectile(
        entity_id=7016,
        entity_type=EntityType.CALTROP,
        owner_id=attacker.entity_id,
        team=attacker.session.team_id,
        pos=target.player_pos,
        vel=(0.0, 0.0, 0.0),
        spawn_time=time.monotonic(),
        lifetime=30.0,
    )

    server._apply_damage(target, proj, attacker)

    assert target.player_health == 0.9, target.player_health
    assert target.last_damage_source == "projectile:CALTROP"
    assert target.last_damage_amount == 0.10
    assert target.session.in_game is True
    print("test_caltrop_projectile_damage_uses_light_bomblet_amount_and_cleanup: PASSED")
    return True


def test_players_json_includes_transport_addresses():
    """Control `players json` should expose client and UDP addresses for diagnostics."""
    server = SimpleNamespace(
        clients_lock=threading.Lock(),
        clients={},
        _get_energy_value=lambda ctx: max(0.0, min(100.0, ctx.player_energy)) / 100.0,
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
    ctx.player_energy = 42.5
    ctx.player_fuel = 16500.0
    ctx.action_packet_count = 3
    ctx.nonzero_move_input_count = 1
    ctx.state_request_count = 2
    ctx.state_sync_reply_count = 2
    ctx.state_sync_view_reply_count = 2
    ctx.last_decoded_input = {"fwd": 0.5, "strafe": 0.0}
    ctx.movement_input_history.append(
        {
            "time": time.monotonic() - 0.5,
            "packet_type": "ACTION_DUMP",
            "client_tick": 1100,
            "action_sequence": 2,
            "fwd": 0.0,
            "strafe": 0.0,
        }
    )
    ctx.movement_input_history.append(
        {
            "time": time.monotonic() - 0.25,
            "packet_type": "ACTION_UPDATE",
            "client_tick": 1122,
            "action_sequence": 3,
            "fwd": 0.5,
            "strafe": 0.0,
        }
    )
    ctx.weapon_fire_count = 1
    ctx.last_weapon_fire_time = time.monotonic() - 0.5
    ctx.last_weapon_fire_source = "ACTION_UPDATE"
    ctx.last_weapon_fire_client_tick = 1122
    ctx.last_weapon_fire_projectile_ids = [7001]
    ctx.last_weapon_fire_projectile_types = ["PULSE_SHELL"]
    ctx.last_weapon_fire_energy_spent = 8.0
    ctx.last_weapon_fire_input = {"direct_slots": {"12": 1.0}}
    ctx.hitscan_fire_count = 1
    ctx.last_hitscan_fire_time = time.monotonic() - 0.4
    ctx.last_hitscan_weapon_name = "Chain Gun"
    ctx.last_hitscan_fire_input = {"fire": 1.0}
    ctx.projectile_update_packet_count = 3
    ctx.last_projectile_update_time = time.monotonic() - 0.25
    ctx.last_projectile_update_id = 7001
    ctx.last_projectile_update_targets = 2
    server.clients[ctx.client_id] = ctx

    control = ControlServer(port=0)
    control.server = server

    payload = control._cmd_players(["json"])
    entries = json.loads(payload)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["client_addr"] == ["10.10.10.2", 52731]
    assert entry["udp_addr"] == ["10.10.10.2", 52732]
    assert entry["energy"] == 42.5
    assert entry["energy_pct"] == 42.5
    assert entry["fuel_pct"] == 50.0
    assert entry["telemetry"]["action_packets"] == 3
    assert entry["telemetry"]["nonzero_move_inputs"] == 1
    assert entry["telemetry"]["state_requests"] == 2
    assert entry["telemetry"]["state_sync_replies"] == 2
    assert entry["telemetry"]["state_sync_view_replies"] == 2
    assert entry["telemetry"]["last_input"]["fwd"] == 0.5
    assert entry["telemetry"]["movement_input_history_len"] == 2
    assert entry["telemetry"]["movement_input_history_nonzero_count"] == 1
    assert entry["telemetry"]["movement_input_history_latest_nonzero_age_s"] is not None
    assert entry["telemetry"]["movement_input_history_latest_nonzero"]["fwd"] == 0.5
    assert (
        entry["telemetry"]["movement_input_history_latest_nonzero"][
            "action_sequence"
        ]
        == 3
    )
    assert entry["telemetry"]["movement_input_history_tail"][-1]["client_tick"] == 1122
    assert entry["telemetry"]["weapon_fire_count"] == 1
    assert entry["telemetry"]["last_weapon_fire_age_s"] is not None
    assert entry["telemetry"]["last_weapon_fire_source"] == "ACTION_UPDATE"
    assert entry["telemetry"]["last_weapon_fire_client_tick"] == 1122
    assert entry["telemetry"]["last_weapon_fire_projectile_ids"] == [7001]
    assert entry["telemetry"]["last_weapon_fire_projectile_types"] == ["PULSE_SHELL"]
    assert entry["telemetry"]["last_weapon_fire_energy_spent"] == 8.0
    assert entry["telemetry"]["last_weapon_fire_input"]["direct_slots"]["12"] == 1.0
    assert entry["telemetry"]["hitscan_fire_count"] == 1
    assert entry["telemetry"]["last_hitscan_fire_age_s"] is not None
    assert entry["telemetry"]["last_hitscan_weapon_name"] == "Chain Gun"
    assert entry["telemetry"]["last_hitscan_fire_input"]["fire"] == 1.0
    assert entry["telemetry"]["projectile_update_packets"] == 3
    assert entry["telemetry"]["last_projectile_update_age_s"] is not None
    assert entry["telemetry"]["last_projectile_update_id"] == 7001
    assert entry["telemetry"]["last_projectile_update_targets"] == 2
    print("test_players_json_includes_transport_addresses: PASSED")
    return True


def test_health_control_uses_server_heartbeat_helper():
    """Control `health set` should use the OG-safe server heartbeat helper."""
    sent = []
    helper_calls = []

    def build_safe_hb(ctx, **kwargs):
        helper_calls.append((ctx.client_id, kwargs))
        return b"SAFE-HB"

    server = SimpleNamespace(
        clients_lock=threading.Lock(),
        clients={},
        udp_handler=SimpleNamespace(send_to=lambda payload, addr: sent.append((payload, addr))),
        _get_network_tick=lambda _ctx: 0x12345678,
        _build_local_state_heartbeat=build_safe_hb,
        _get_health_value=lambda ctx: ctx.player_health,
        _get_energy_value=lambda _ctx: 0.75,
    )
    session = Session()
    session.username = "HealthBot"
    session.team_id = 1
    session.entity_id = 0x543
    session.udp_addr = ("10.10.10.2", 30000)
    ctx = ClientContext(
        client_id=12,
        client_addr=("10.10.10.2", 30000),
        session=session,
        entity_id=0x543,
    )
    ctx.player_health = 1.0
    server.clients[ctx.client_id] = ctx

    control = ControlServer(port=0)
    control.server = server

    result = control._cmd_send_health(["set", "0.5", "c12"])

    assert "Client 12" in result, result
    assert ctx.player_health == 0.5, ctx.player_health
    assert sent == [(b"SAFE-HB", ("10.10.10.2", 30000))], sent
    assert helper_calls == [
        (
            12,
            {
                "tick": 0x12345678,
                "entity_id": 0x543,
                "include_health": True,
                "health": 0.5,
                "fuel": 0.75,
            },
        )
    ], helper_calls
    print("test_health_control_uses_server_heartbeat_helper: PASSED")
    return True


def test_energy_control_sets_absolute_or_fractional_energy():
    """Control `energy set` should expose fuel/energy-pad recovery setup."""
    server = SimpleNamespace(
        clients_lock=threading.Lock(),
        clients={},
        player_energy_max=100.0,
        udp_handler=None,
    )
    session = Session()
    session.username = "EnergyBot"
    session.team_id = 1
    session.entity_id = 0x542
    session.udp_addr = ("127.0.0.1", 30000)
    session.phase = Session().phase.__class__.IN_GAME
    ctx = ClientContext(
        client_id=10,
        client_addr=("127.0.0.1", 30000),
        session=session,
        entity_id=0x542,
    )
    ctx.player_energy = 100.0
    server.clients[ctx.client_id] = ctx

    control = ControlServer(port=0)
    control.server = server

    result = control._cmd_energy(["set", "25", "c10"])
    assert "25.0/100.0" in result, result
    assert ctx.player_energy == 25.0, ctx.player_energy
    result = control._cmd_energy(["set", "0.5", "c10"])
    assert "50.0/100.0" in result, result
    assert ctx.player_energy == 50.0, ctx.player_energy
    text = control._cmd_energy(["c10"])
    assert "EnergyBot" in text and "50%" in text, text
    print("test_energy_control_sets_absolute_or_fractional_energy: PASSED")
    return True


def test_buildings_json_exposes_playable_slice_observability():
    """Control `buildings json` should expose service/turret state for slice smokes."""
    server = SimpleNamespace(
        clients_lock=threading.Lock(),
        clients={},
        _building_entities={
            5001: BuildingEntity(
                x=10.0,
                y=20.0,
                z=5.0,
                entity_type=EntityType.REPAIR_BUILDING,
                team_id=2,
            ),
            5002: BuildingEntity(
                x=100.0,
                y=100.0,
                z=8.0,
                entity_type=EntityType.GUN_TURRET,
                team_id=2,
            ),
        },
        _building_health={5001: 1500.0, 5002: 600.0},
        _building_max_health={5001: 2000.0, 5002: 1200.0},
        _turret_last_fire={5002: time.monotonic() - 1.0},
        _dynamic_building_ids={5002},
        _dynamic_building_sources={5002: {"command": {"action": "build"}, "slot": 0}},
    )
    server._building_blocks_vehicle_collision = (
        lambda building: int(building.entity_type) != int(EntityType.REPAIR_BUILDING)
    )
    server._building_has_mesh_collision = (
        lambda building: int(building.entity_type) == int(EntityType.GUN_TURRET)
    )
    server._get_building_world_half_extents = lambda _building: (4.0, 5.0, 6.0)
    server._terrain_ground_z_at = lambda _x, _y: 4.0

    friendly_session = Session()
    friendly_session.phase = Phase.IN_GAME
    friendly_session.team_id = 2
    friendly_ctx = ClientContext(
        client_id=1,
        client_addr=("127.0.0.1", 30001),
        session=friendly_session,
        entity_id=0x1001,
    )
    friendly_ctx.player_pos = (20.0, 20.0, 5.0)

    enemy_session = Session()
    enemy_session.phase = Phase.IN_GAME
    enemy_session.team_id = 1
    enemy_ctx = ClientContext(
        client_id=2,
        client_addr=("127.0.0.1", 30002),
        session=enemy_session,
        entity_id=0x1002,
    )
    enemy_ctx.player_pos = (110.0, 100.0, 5.0)

    server.clients[1] = friendly_ctx
    server.clients[2] = enemy_ctx

    control = ControlServer(port=0)
    control.server = server

    entries = json.loads(control._cmd_buildings(["json"]))
    by_oid = {entry["oid"]: entry for entry in entries}

    repair = by_oid[5001]
    assert repair["entity_type_name"] == "REPAIR_BUILDING", repair
    assert repair["service_kind"] == "repair", repair
    assert repair["blocks_vehicles"] is False, repair
    assert repair["clients_in_service_range"] == [1], repair
    assert repair["health_pct"] == 75.0, repair
    assert repair["terrain_delta_z"] == 1.0, repair
    assert repair["dynamic"] is False, repair

    turret = by_oid[5002]
    assert turret["entity_type_name"] == "GUN_TURRET", turret
    assert turret["turret_kind"] == "gun", turret
    assert turret["has_mesh_collision"] is True, turret
    assert turret["enemy_clients_in_turret_range"] == [2], turret
    assert turret["last_turret_fire_age_s"] is not None, turret
    assert turret["health_pct"] == 50.0, turret
    assert turret["dynamic"] is True, turret
    assert turret["dynamic_source"]["command"]["action"] == "build", turret

    text = control._cmd_buildings([])
    assert "REPAIR_BUILDING" in text and "nonblocking" in text, text
    assert "GUN_TURRET" in text and "turret=gun" in text, text
    print("test_buildings_json_exposes_playable_slice_observability: PASSED")
    return True


def _comm_request_body(text: str) -> bytes:
    encoded = (text + "\x00").encode("ascii")
    return struct.pack(">HHH", 2, 0, len(encoded)) + encoded


CAPTURED_OG_POWER_CELL_COMM_REQUEST_HEX = (
    "200001002600020000001b"
    "6275696c642032393030322022506f7765722043656c6c22203000"
)


def _minimal_build_uplink_server(ctx: ClientContext) -> WulframServer:
    class DummyUDP:
        def __init__(self):
            self.sent = []

        def send_to(self, payload, addr):
            self.sent.append((payload, addr))

    server = WulframServer.__new__(WulframServer)
    server.clients_lock = threading.Lock()
    server.clients = {ctx.client_id: ctx}
    server.udp_handler = DummyUDP()
    server.pktlog = SimpleNamespace(enabled=False)
    server.up_axis = "z"
    server.pos_offset = 0.0
    server.use_client_ticks = False
    server.debug_health_value = 1.0
    server.debug_health_pattern = False
    server.player_energy_max = 100.0
    server.spawn_tank_weapon_type = 2
    server.local_state_turret_max = 6.3
    server.local_state_turret_range = 12.6
    server.build_uplink_mvp = True
    server._building_entities = {}
    server._building_health = {}
    server._building_max_health = {}
    server._turret_last_fire = {}
    server._dynamic_building_ids = set()
    server._dynamic_building_sources = {}
    server._dynamic_building_next_oid = 30000
    server._build_uplink_command_events = []
    server._building_lifecycle_events = []
    server._uplink_ships = {}
    server._building_collision = SimpleNamespace(available=False)
    server._terrain_grid_collision = None
    server._static_world_raycast_root = None
    server.remote_combat_observer_packets = True
    server._terrain_ground_z_at = lambda _x, _y: 4.0
    server._rebuild_static_world_raycast_index = lambda: None
    return server


def _in_game_context(client_id: int = 1) -> ClientContext:
    session = Session()
    session.phase = Phase.IN_GAME
    session.in_game = True
    session.translation_ack_received = True
    session.team_id = 2
    session.entity_id = 0x14EA
    session.udp_addr = ("10.10.10.2", 62588)
    ctx = ClientContext(
        client_id=client_id,
        client_addr=("10.10.10.2", 50000 + client_id),
        session=session,
        entity_id=0x14EA,
    )
    ctx.player_pos = (2600.0, 3040.0, 6.0)
    ctx.player_heading = 0.0
    return ctx


def test_comm_message_request_decodes_og_type2_build_command():
    """Decompile-backed uplink commands are type-2 COMM_MESSAGE_REQUEST strings."""
    server = WulframServer.__new__(WulframServer)
    body = _comm_request_body('build 29002 "Repair Building" 1')

    decoded = server._decode_comm_message_request_body(body)
    parsed = server._parse_build_uplink_command(decoded["text"])

    assert decoded["ok"] is True, decoded
    assert decoded["message_type"] == 2, decoded
    assert decoded["flags_or_target"] == 0, decoded
    assert decoded["text"] == 'build 29002 "Repair Building" 1', decoded
    assert parsed["ok"] is True, parsed
    assert parsed["action"] == "build", parsed
    assert parsed["ship_oid"] == 29002, parsed
    assert parsed["entity_type"] == int(EntityType.REPAIR_BUILDING), parsed
    assert parsed["slot"] == 1, parsed
    captured = server._parse_build_uplink_command('build 29002 "Power Cell" 0')
    assert captured["ok"] is True, captured
    assert captured["entity_type"] == int(EntityType.ENERGY_BUILDING), captured
    assert captured["slot"] == 0, captured
    print("test_comm_message_request_decodes_og_type2_build_command: PASSED")
    return True


def test_build_uplink_parser_accepts_slice_buildable_names():
    """The four Crossroads T3 buildables should parse before live UI promotion."""
    server = WulframServer.__new__(WulframServer)
    cases = [
        ('build 29002 "Power Cell" 0', EntityType.ENERGY_BUILDING, 0),
        ('build 29002 "Repair Building" 1', EntityType.REPAIR_BUILDING, 1),
        ('build 29002 "Fuel Building" 2', EntityType.FUEL_BUILDING, 2),
        ('build 29002 "Gun Turret" 3', EntityType.GUN_TURRET, 3),
    ]
    for text, entity_type, slot in cases:
        parsed = server._parse_build_uplink_command(text)
        assert parsed["ok"] is True, (text, parsed)
        assert parsed["action"] == "build", parsed
        assert parsed["ship_oid"] == 29002, parsed
        assert parsed["entity_type"] == int(entity_type), parsed
        assert parsed["slot"] == slot, parsed

    print("test_build_uplink_parser_accepts_slice_buildable_names: PASSED")
    return True


def test_build_uplink_command_creates_dynamic_building():
    """Accepted OG build commands should create and replicate a dynamic building."""
    ctx = _in_game_context()
    server = _minimal_build_uplink_server(ctx)
    packet = b"\x20\x00\x44\x00\x00" + _comm_request_body('build 29002 "Repair Building" 1')

    event = server._handle_comm_message_request(
        ctx,
        packet,
        transport="udp",
        body=packet[5:],
        addr=ctx.session.udp_addr,
        sequence=0x44,
    )

    assert event["handled"] is True, event
    assert event["decoded"]["ok"] is True, event
    assert event["build_uplink_command"]["action"] == "build", event
    assert event["result"]["ok"] is True, event
    oid = int(event["result"]["oid"])
    assert oid in server._dynamic_building_ids, event
    assert oid in server._building_entities, event
    assert int(server._building_entities[oid].entity_type) == int(EntityType.REPAIR_BUILDING), event
    assert server._dynamic_building_sources[oid]["slot"] == 1, event
    assert ctx.build_uplink_command_count == 1, ctx.last_build_uplink_command
    assert any(payload and payload[0] == 0x0E for payload, _addr in server.udp_handler.sent), server.udp_handler.sent
    assert any(payload and payload[0] == 0x2D for payload, _addr in server.udp_handler.sent), server.udp_handler.sent
    print("test_build_uplink_command_creates_dynamic_building: PASSED")
    return True


def test_build_uplink_commands_create_all_slice_buildable_types():
    """All requested slice buildable types should enter the dynamic entity lifecycle."""
    ctx = _in_game_context()
    server = _minimal_build_uplink_server(ctx)
    cases = [
        ("Power Cell", EntityType.ENERGY_BUILDING, "energy", 0),
        ("Repair Building", EntityType.REPAIR_BUILDING, "repair", 1),
        ("Fuel Building", EntityType.FUEL_BUILDING, "fuel", 2),
        ("Gun Turret", EntityType.GUN_TURRET, "turret", 3),
    ]

    created_oids = []
    for sequence, (name, entity_type, service_kind, slot) in enumerate(cases, start=1):
        packet = b"\x20\x00" + bytes([0x44 + sequence]) + b"\x00\x00" + _comm_request_body(
            f'build 29002 "{name}" {slot}'
        )
        event = server._handle_comm_message_request(
            ctx,
            packet,
            transport="udp",
            body=packet[5:],
            addr=ctx.session.udp_addr,
            sequence=0x44 + sequence,
        )

        assert event["handled"] is True, (name, event)
        assert event["result"]["ok"] is True, (name, event)
        oid = int(event["result"]["oid"])
        building = server._building_entities[oid]
        source = server._dynamic_building_sources[oid]
        assert int(building.entity_type) == int(entity_type), (name, event)
        assert source["command"]["entity_name"] == name, source
        assert source["slot"] == slot, source
        if int(entity_type) == int(EntityType.GUN_TURRET):
            classification = "turret"
        elif int(entity_type) == int(EntityType.REPAIR_BUILDING):
            classification = "repair"
        elif int(entity_type) == int(EntityType.FUEL_BUILDING):
            classification = "fuel"
        elif int(entity_type) == int(EntityType.ENERGY_BUILDING):
            classification = "energy"
        else:
            classification = ""
        assert classification == service_kind, (name, classification, service_kind)
        created_oids.append(oid)

    assert len(created_oids) == 4, created_oids
    assert set(created_oids) <= server._dynamic_building_ids
    assert ctx.build_uplink_command_count == 4, ctx.build_uplink_command_count
    assert len(server._building_lifecycle_events) == 4, server._building_lifecycle_events
    print("test_build_uplink_commands_create_all_slice_buildable_types: PASSED")
    return True


def test_dynamic_powercell_service_restores_energy_with_ambient_regen_disabled():
    """Captured OG Power Cell packet should restore energy without ambient regen."""
    ctx = _in_game_context()
    ctx.player_energy = 10.0
    server = _minimal_build_uplink_server(ctx)
    server.player_energy_regen = 0.0
    packet = bytes.fromhex(CAPTURED_OG_POWER_CELL_COMM_REQUEST_HEX)

    event = server._handle_comm_message_request(
        ctx,
        packet,
        transport="udp",
        body=packet[5:],
        addr=ctx.session.udp_addr,
        sequence=1,
    )
    oid = int(event["result"]["oid"])
    building = server._building_entities[oid]

    assert event["result"]["ok"] is True, event
    assert int(building.entity_type) == int(EntityType.ENERGY_BUILDING), event
    assert math.hypot(ctx.player_pos[0] - building.x, ctx.player_pos[1] - building.y) <= 40.0

    server._regen_player_energy(ctx, 1.0)
    assert ctx.player_energy == 10.0, ctx.player_energy

    server._update_supply_buildings(ctx, 1.0 / 30.0)
    assert ctx.player_energy == 13.0, ctx.player_energy
    for _ in range(20):
        server._update_supply_buildings(ctx, 1.0 / 30.0)
    assert ctx.player_energy >= 70.0, ctx.player_energy
    assert ctx.player_energy <= server.player_energy_max, ctx.player_energy
    print("test_dynamic_powercell_service_restores_energy_with_ambient_regen_disabled: PASSED")
    return True


def test_dynamic_building_damage_control_destroys_and_records_delete():
    """Demo lifecycle gate should have explicit damage, destroy, and delete evidence."""
    ctx = _in_game_context()
    server = _minimal_build_uplink_server(ctx)
    packet = b"\x20\x00\x44\x00\x00" + _comm_request_body('build 29002 "Power Cell" 0')
    event = server._handle_comm_message_request(
        ctx,
        packet,
        transport="udp",
        body=packet[5:],
        addr=ctx.session.udp_addr,
        sequence=0x44,
    )
    oid = int(event["result"]["oid"])

    control = ControlServer(port=0)
    control.server = server

    first = json.loads(control._cmd_building_damage([str(oid), "250"]))
    assert first["ok"] is True, first
    assert first["action"] == "damage", first
    assert first["old_health"] == 2000.0, first
    assert first["new_health"] == 1750.0, first
    assert oid in server._building_entities, first

    destroyed = json.loads(control._cmd_building_damage([str(oid), "destroy"]))
    assert destroyed["ok"] is True, destroyed
    assert destroyed["action"] == "destroy", destroyed
    assert destroyed["new_health"] == 0.0, destroyed
    assert destroyed["delete_sent"] == 1, destroyed
    assert destroyed["removed"] is True, destroyed
    assert oid not in server._building_entities, destroyed
    assert oid not in server._dynamic_building_ids, destroyed
    assert server._uplink_ships[2]["cargo"][0] == 40, server._uplink_ships[2]
    assert any(payload and payload[0] == 0x15 and oid.to_bytes(4, "big") in payload for payload, _addr in server.udp_handler.sent), server.udp_handler.sent

    lifecycle = json.loads(control._cmd_building_events(["json"]))
    actions = [item.get("action") for item in lifecycle if int(item.get("oid", 0) or 0) == oid]
    assert actions == ["create", "damage", "destroy"], lifecycle
    print("test_dynamic_building_damage_control_destroys_and_records_delete: PASSED")
    return True


def test_dynamic_building_projectile_damage_destroys_and_records_delete():
    """Projectile-code-path lifecycle gate should record projectile-sourced destroy evidence."""
    ctx = _in_game_context()
    ctx.session.username = "ProjTester"
    server = _minimal_build_uplink_server(ctx)
    server._rebuild_static_world_raycast_index = WulframServer._rebuild_static_world_raycast_index.__get__(
        server,
        WulframServer,
    )
    packet = b"\x20\x00\x45\x00\x00" + _comm_request_body('build 29002 "Power Cell" 0')
    event = server._handle_comm_message_request(
        ctx,
        packet,
        transport="udp",
        body=packet[5:],
        addr=ctx.session.udp_addr,
        sequence=0x45,
    )
    oid = int(event["result"]["oid"])
    server._rebuild_static_world_raycast_index()

    control = ControlServer(port=0)
    control.server = server
    result = json.loads(control._cmd_building_projectile_damage([str(oid), "destroy", "heavy", "c1"]))

    assert result["ok"] is True, result
    assert result["removed"] is True, result
    assert result["projectile_type"] == "HEAVY_MISSILE", result
    assert len(result["shots"]) >= 2, result
    assert oid not in server._building_entities, result
    assert oid not in server._dynamic_building_ids, result
    assert server._uplink_ships[2]["cargo"][0] == 40, server._uplink_ships[2]
    assert any(payload and payload[0] == 0x15 and oid.to_bytes(4, "big") in payload for payload, _addr in server.udp_handler.sent), server.udp_handler.sent

    lifecycle = json.loads(control._cmd_building_events(["json"]))
    events = [item for item in lifecycle if int(item.get("oid", 0) or 0) == oid]
    actions = [item.get("action") for item in events]
    sources = [str(item.get("source") or "") for item in events if item.get("action") in {"damage", "destroy"}]
    assert actions[0] == "create", lifecycle
    assert "damage" in actions, lifecycle
    assert actions[-1] == "destroy", lifecycle
    assert all(source.startswith("projectile:") for source in sources), sources
    assert events[-1]["delete_sent"] == 1, events[-1]
    assert events[-1]["removed"] is True, events[-1]
    print("test_dynamic_building_projectile_damage_destroys_and_records_delete: PASSED")
    return True


def test_uplink_mvp_bootstrap_sends_minimal_status_packets():
    """Default-off build/uplink bootstrap should send local team and uplink state."""
    ctx = _in_game_context()
    server = _minimal_build_uplink_server(ctx)

    server._ensure_uplink_mvp_state(ctx)

    sent_payloads = [payload for payload, _addr in server.udp_handler.sent]
    opcodes = [payload[0] for payload in sent_payloads if payload]
    assert ctx.uplink_mvp_bootstrap_sent is True, opcodes
    assert 0x0E in opcodes, opcodes
    assert 0x1C in opcodes, opcodes
    assert 0x27 in opcodes, opcodes
    assert 0x2D in opcodes, opcodes
    assert 0x29 in opcodes, opcodes
    assert 0x2A in opcodes, opcodes
    assert build_update_stats_team_first(player_id=0x14EA, entity_id=0x14EA, team_id=2) in sent_payloads
    ship_status = next(payload for payload in sent_payloads if payload and payload[0] == 0x27)
    assert ship_status == build_ship_status(29002, 2, "Team 2 Supply Ship"), ship_status.hex()
    supply_ship_info = next(payload for payload in sent_payloads if payload and payload[0] == 0x2D)
    assert supply_ship_info == build_supply_ship_info(29002, shield_pct=100, status_template=0, build_mode=3), supply_ship_info.hex()
    assert build_carrying_info(0x14EA, has_uplink=True) in sent_payloads
    assert build_uplink_info(2, 0x14EA, 3) in sent_payloads
    print("test_uplink_mvp_bootstrap_sends_minimal_status_packets: PASSED")
    return True


def test_lead_extrapolated_correction_pose():
    """Prediction-lead correction extrapolation (WULFRAM_CORRECTION_LEAD_TICKS).

    Characterization (tools/wulftap_turn_capture.py) showed the OG client predicts
    ~1 server tick AHEAD of the server's confirmed pose, with the turn RATE
    lockstep. `_lead_extrapolated_correction_pose` aims corrections at that
    predicted pose so they stop snapping the client backward. lead=0 must be a
    byte-identical no-op vs the old `_to_client_pos(player_pos)` /
    `_local_player_sync_rotation`; lead>0 advances pos by `player_vel`*dt and yaw
    by `angular_vel_yaw`*dt (the very rates the integrator advances pos/heading
    with); at rest both rates are 0 so it is a no-op.
    """
    from wulfram.server import WulframServer

    def make_self(lead):
        return SimpleNamespace(
            correction_lead_ticks=lead,
            tick_rate_hz=30.0,
            # mimic _to_client_pos (pos_offset=5) and _local_player_sync_rotation
            _to_client_pos=lambda p: (p[0], p[1], p[2] + 5.0),
            _local_player_sync_rotation=lambda c: (
                c.player_pose["roll"], c.player_pose["pitch"], c.player_heading),
        )

    pose = {"roll": 0.1, "pitch": -0.2}
    moving = SimpleNamespace(
        player_pos=(100.0, 200.0, 3.0), player_vel=(6.0, -3.0, 0.3),
        angular_vel_yaw=0.9, player_heading=1.5, player_pose=pose)

    # lead=0 -> identity with the legacy path
    pos0, rot0 = WulframServer._lead_extrapolated_correction_pose(make_self(0.0), moving)
    assert pos0 == (100.0, 200.0, 8.0), pos0
    assert rot0 == (0.1, -0.2, 1.5), rot0

    # lead=1 tick (1/30 s)
    dt = 1.0 / 30.0
    pos1, rot1 = WulframServer._lead_extrapolated_correction_pose(make_self(1.0), moving)
    assert abs(pos1[0] - (100.0 + 6.0 * dt)) < 1e-9, pos1
    assert abs(pos1[1] - (200.0 - 3.0 * dt)) < 1e-9, pos1
    assert abs(pos1[2] - (3.0 + 0.3 * dt + 5.0)) < 1e-9, pos1
    assert rot1[0] == 0.1 and rot1[1] == -0.2, rot1
    assert abs(rot1[2] - (1.5 + 0.9 * dt)) < 1e-9, rot1

    # at rest -> no-op even with lead>0
    rest = SimpleNamespace(
        player_pos=(10.0, 20.0, 3.0), player_vel=(0.0, 0.0, 0.0),
        angular_vel_yaw=0.0, player_heading=2.0, player_pose=pose)
    posr, rotr = WulframServer._lead_extrapolated_correction_pose(make_self(1.0), rest)
    assert posr == (10.0, 20.0, 8.0), posr
    assert rotr == (0.1, -0.2, 2.0), rotr

    print("test_lead_extrapolated_correction_pose: PASSED")
    return True


def test_carrying_info_carry_state():
    """Cargo carry state + CARRYING_INFO broadcast (construction Phase 1).

    `set_player_carry` updates the ctx cargo fields and broadcasts CARRYING_INFO
    (0x29: u32 player_oid, u8 cargo_type, u8 has_uplink, u8 cargo_count) only when
    the state actually changes. `build_carrying_info` encodes the wire format.
    """
    from wulfram import build_uplink
    from wulfram.packets import build_carrying_info

    # wire format
    pkt = build_carrying_info(0x1337, cargo_type=27, has_uplink=True, cargo_count=2)
    assert pkt == b"\x29" + struct.pack(">I", 0x1337) + struct.pack("BBB", 27, 1, 2), pkt.hex()

    sent = []
    ctx = SimpleNamespace(
        cargo_type=0, cargo_count=0, has_uplink=False, entity_id=1339,
        session=SimpleNamespace(player_id=0x1337, in_game=True))

    class StubServer:
        def _snapshot_in_game_clients(self):
            return [ctx]

        def _send_packet_to_client(self, target, payload, prefer_tcp=False):
            sent.append(payload)
            return True

        def _broadcast_carrying_info(self, c):
            return build_uplink.broadcast_carrying_info(self, c)

    srv = StubServer()

    r = build_uplink.set_player_carry(srv, ctx, cargo_type=27, cargo_count=2, has_uplink=True)
    assert ctx.cargo_type == 27 and ctx.cargo_count == 2 and ctx.has_uplink is True, vars(ctx)
    assert r["changed"] is True and r["recipients"] == 1, r
    assert sent and sent[-1] == build_carrying_info(0x1337, cargo_type=27, has_uplink=True, cargo_count=2)

    # unchanged -> no re-broadcast
    sent.clear()
    r2 = build_uplink.set_player_carry(srv, ctx, cargo_type=27, cargo_count=2, has_uplink=True)
    assert r2["changed"] is False and r2["recipients"] == 0 and not sent, (r2, sent)

    # drop
    r3 = build_uplink.set_player_carry(srv, ctx, cargo_type=0, cargo_count=0, has_uplink=False)
    assert ctx.cargo_type == 0 and ctx.has_uplink is False and r3["changed"] is True, (vars(ctx), r3)

    print("test_carrying_info_carry_state: PASSED")
    return True


def test_cargo_pickup_proximity():
    """_try_cargo_pickup grabs a cargo box near the team ship, only empty-handed."""
    from wulfram.server import WulframServer

    carries = []

    def set_carry(ctx, *, cargo_type, cargo_count, has_uplink):
        ctx.cargo_type = cargo_type
        ctx.cargo_count = cargo_count
        ctx.has_uplink = has_uplink
        carries.append((cargo_type, cargo_count))
        return {}

    srv = SimpleNamespace(
        _uplink_ships={2: {"oid": 29002, "pos": (5000.0, 5000.0, 5.0),
                           "cargo_available": 3, "next_replenish": 0.0, "cargo": [27, 27, 27, 40]}},
        cargo_pickup_range=30.0, default_cargo_type=27, ship_cargo_capacity=3,
        ship_replenish_s=0.0, _set_player_carry=set_carry,
        _broadcast_uplink_ship_info=lambda ship: None)
    ctx = SimpleNamespace(
        cargo_type=0, cargo_count=0, has_uplink=False, client_id=1,
        player_pos=(5100.0, 5000.0, 5.0), session=SimpleNamespace(team_id=2, in_game=True))

    # far from ship -> no pickup
    assert WulframServer._try_cargo_pickup(srv, ctx) is False and not carries, carries
    # near ship + empty-handed -> pickup
    ctx.player_pos = (5010.0, 5000.0, 5.0)
    assert WulframServer._try_cargo_pickup(srv, ctx) is True
    assert ctx.cargo_type == 27 and ctx.cargo_count == 1, vars(ctx)
    # already carrying -> no re-pickup
    carries.clear()
    assert WulframServer._try_cargo_pickup(srv, ctx) is False and not carries, carries
    print("test_cargo_pickup_proximity: PASSED")
    return True


def test_build_require_cargo_gate():
    """build_require_cargo: a build with no carried cargo is rejected, no build."""
    from wulfram import build_uplink

    ctx = SimpleNamespace(
        cargo_type=0, cargo_count=0, has_uplink=False, client_id=1, entity_id=1339,
        session=SimpleNamespace(team_id=2, entity_id=1339))
    srv = SimpleNamespace(build_require_cargo=True)
    r = build_uplink.create_dynamic_building_from_uplink(srv, ctx, {"entity_type": 27, "slot": 0})
    assert r["ok"] is False and r["error"] == "no_cargo", r
    print("test_build_require_cargo_gate: PASSED")
    return True


def test_construction_timer_lazy_complete():
    """_building_under_construction is True until the timer elapses, then completes."""
    from wulfram.server import WulframServer
    import time as _time

    srv = SimpleNamespace(_building_construction={})
    # untracked -> not under construction
    assert WulframServer._building_under_construction(srv, 999) is False
    # future completion -> under construction (stays tracked)
    srv._building_construction[30001] = _time.monotonic() + 100.0
    assert WulframServer._building_under_construction(srv, 30001) is True
    assert 30001 in srv._building_construction
    # elapsed completion -> completes + drops from the set
    srv._building_construction[30002] = _time.monotonic() - 1.0
    assert WulframServer._building_under_construction(srv, 30002) is False
    assert 30002 not in srv._building_construction
    print("test_construction_timer_lazy_complete: PASSED")
    return True


def test_deconstruct_refund():
    """Deconstruct removes a player-built structure and refunds it as carried
    cargo when the economy is on and the player's hands are free (Phase 2)."""
    from wulfram import build_uplink

    class Srv:
        build_require_cargo = True

        def __init__(self):
            self._dynamic_building_ids = {30000}
            self._building_entities = {30000: SimpleNamespace(entity_type=27, team_id=2)}
            self._dynamic_building_sources = {30000: {}}
            self._uplink_ships = {}

        def _building_lifecycle_base_event(self, oid, action):
            return {"oid": oid, "action": action}

        def _broadcast_building_delete(self, oid, prefer_tcp=False):
            return 1

        def _remove_dynamic_building_record(self, oid):
            self._building_entities.pop(oid, None)
            self._dynamic_building_ids.discard(oid)

        def _remember_building_lifecycle_event(self, ev):
            return ev

        def _broadcast_uplink_ship_info(self, ship):
            return 0

        def _set_player_carry(self, ctx, *, cargo_type, cargo_count, has_uplink):
            ctx.cargo_type = cargo_type
            ctx.cargo_count = cargo_count
            ctx.has_uplink = has_uplink
            return {}

    srv = Srv()
    ctx = SimpleNamespace(cargo_type=0, cargo_count=0, has_uplink=False, client_id=1,
                          session=SimpleNamespace(team_id=2))
    r = build_uplink.delete_dynamic_building_from_uplink(srv, ctx, {})
    assert r["ok"] is True and r["refunded"] is True, r
    assert ctx.cargo_type == 27 and ctx.cargo_count == 1, vars(ctx)
    assert 30000 not in srv._building_entities, srv._building_entities

    # hands full -> no refund (cargo lost), building still removed
    srv2 = Srv()
    ctx2 = SimpleNamespace(cargo_type=27, cargo_count=1, has_uplink=False, client_id=1,
                           session=SimpleNamespace(team_id=2))
    r2 = build_uplink.delete_dynamic_building_from_uplink(srv2, ctx2, {})
    assert r2["ok"] is True and r2["refunded"] is False, r2
    print("test_deconstruct_refund: PASSED")
    return True


def test_deconstruction_timer():
    """Deconstruct with a timeout marks the building 'deconstructing' (kept, no
    service), then update_deconstruction removes + refunds it when due (Phase 2)."""
    import threading
    import time as _t
    from wulfram import build_uplink

    refunds = []

    class Srv:
        build_require_cargo = True
        deconstruction_timeout = 100.0

        def __init__(self):
            self._dynamic_building_ids = {30000}
            self._building_entities = {30000: SimpleNamespace(entity_type=27, team_id=2)}
            self._dynamic_building_sources = {30000: {}}
            self._uplink_ships = {}
            self._building_deconstruction = {}
            self.clients = {}
            self.clients_lock = threading.Lock()

        def _building_lifecycle_base_event(self, oid, action):
            return {"oid": oid}

        def _broadcast_building_delete(self, oid, prefer_tcp=False):
            return 1

        def _remove_dynamic_building_record(self, oid):
            self._building_entities.pop(oid, None)
            self._dynamic_building_ids.discard(oid)

        def _remember_building_lifecycle_event(self, ev):
            return ev

        def _broadcast_uplink_ship_info(self, ship):
            return 0

        def _set_player_carry(self, ctx, *, cargo_type, cargo_count, has_uplink):
            ctx.cargo_type = cargo_type
            ctx.cargo_count = cargo_count
            ctx.has_uplink = has_uplink
            refunds.append(cargo_type)

    srv = Srv()
    ctx = SimpleNamespace(cargo_type=0, cargo_count=0, has_uplink=False, client_id=7,
                          session=SimpleNamespace(team_id=2))
    srv.clients = {7: ctx}

    # initiate -> deconstructing; building kept, not yet removed/refunded
    r = build_uplink.delete_dynamic_building_from_uplink(srv, ctx, {})
    assert r["deconstructing"] is True and r["ok"] is True, r
    assert 30000 in srv._building_entities and 30000 in srv._building_deconstruction
    assert not refunds and ctx.cargo_type == 0

    # not due -> no completion
    assert build_uplink.update_deconstruction(srv) == 0
    assert 30000 in srv._building_entities, "removed too early"

    # force due -> completes: removed + refunded
    srv._building_deconstruction[30000]["done"] = _t.monotonic() - 1.0
    assert build_uplink.update_deconstruction(srv) == 1
    assert 30000 not in srv._building_entities and 30000 not in srv._building_deconstruction
    assert ctx.cargo_type == 27 and refunds == [27], (vars(ctx), refunds)
    print("test_deconstruction_timer: PASSED")
    return True


def test_removal_clears_timer_dicts():
    """Any building removal clears the construction + deconstruction timer dicts, so
    a building destroyed mid-(de)construction can't dangle / double-refund (Phase 2)."""
    from wulfram import building_lifecycle

    class Srv:
        def __init__(self):
            self._building_entities = {30000: SimpleNamespace(team_id=2)}
            self._building_health = {30000: 0.0}
            self._building_max_health = {30000: 2000.0}
            self._dynamic_building_ids = {30000}
            self._dynamic_building_sources = {30000: {}}
            self._uplink_ships = {}
            self._building_construction = {30000: 123.0}
            self._building_deconstruction = {30000: {"done": 123.0, "entity_type": 27}}
            self.rebuilt = 0

        def _broadcast_uplink_ship_info(self, ship):
            return 0

        def _rebuild_static_world_raycast_index(self):
            self.rebuilt += 1

    srv = Srv()
    building_lifecycle.remove_dynamic_record(srv, 30000)
    assert 30000 not in srv._building_entities
    assert 30000 not in srv._building_construction, "construction timer dangled"
    assert 30000 not in srv._building_deconstruction, "deconstruction timer dangled"
    assert srv.rebuilt == 1
    print("test_removal_clears_timer_dicts: PASSED")
    return True


def test_cargo_drop_and_crate_pickup():
    """A destroyed building drops a CARGO_BOX crate; a player drives over it
    empty-handed and picks it up (crate removed). (Phase 2 cargo-drop-on-destroy)"""
    from wulfram.server import WulframServer

    sent = []
    deleted = []

    class Srv:
        cargo_pickup_range = 30.0
        default_cargo_type = 0  # no ship cargo -> isolate crate pickup
        cargo_box_entity_enabled = True  # exercise the (gated) entity broadcast path

        def __init__(self):
            self._dropped_cargo = {}
            self._dropped_cargo_next_oid = 40000
            self._uplink_ships = {}

        def _broadcast_dynamic_entity_definition(self, **kw):
            sent.append(kw)
            return 1

        def _broadcast_building_delete(self, oid, prefer_tcp=False):
            deleted.append(oid)
            return 1

        def _set_player_carry(self, ctx, *, cargo_type, cargo_count, has_uplink):
            ctx.cargo_type = cargo_type
            ctx.cargo_count = cargo_count
            ctx.has_uplink = has_uplink

    srv = Srv()

    # destroyed building drops a crate at (5000,5000) of type 27
    oid = WulframServer._drop_cargo_crate(srv, (5000.0, 5000.0, 5.0), 27, 2)
    assert oid == 40000 and oid in srv._dropped_cargo, srv._dropped_cargo
    assert srv._dropped_cargo[oid]["cargo_type"] == 27
    assert sent and sent[-1]["entity_type"] == 19, sent  # CARGO_BOX

    ctx = SimpleNamespace(cargo_type=0, cargo_count=0, has_uplink=False, client_id=1,
                          player_pos=(5100.0, 5000.0, 5.0),
                          session=SimpleNamespace(team_id=2, in_game=True))
    # far -> no pickup
    assert WulframServer._try_cargo_pickup(srv, ctx) is False and oid in srv._dropped_cargo
    # near crate + empty-handed -> pickup; crate removed + delete broadcast
    ctx.player_pos = (5010.0, 5000.0, 5.0)
    assert WulframServer._try_cargo_pickup(srv, ctx) is True
    assert ctx.cargo_type == 27, vars(ctx)
    assert oid not in srv._dropped_cargo and oid in deleted, (srv._dropped_cargo, deleted)
    print("test_cargo_drop_and_crate_pickup: PASSED")
    return True


def test_supply_ship_cargo_replenish():
    """Supply-ship cargo is finite + replenishes: ship_set_cargo_available reflects
    the count into the cargo array; replenish_ships restocks one box per interval up
    to capacity. (Phase 2 supply-ship resource)"""
    import time as _t
    from wulfram import build_uplink

    class Srv:
        ship_cargo_capacity = 3
        ship_replenish_s = 10.0
        default_cargo_type = 27

        def __init__(self):
            self._uplink_ships = {2: {"cargo_available": 3, "next_replenish": 0.0,
                                      "cargo": [27, 27, 27, 40]}}
            self.broadcasts = 0

        def _broadcast_uplink_ship_info(self, ship):
            self.broadcasts += 1

    srv = Srv()
    ship = srv._uplink_ships[2]

    # set available -> reflects into the 4-slot array
    build_uplink.ship_set_cargo_available(srv, ship, 1)
    assert ship["cargo_available"] == 1 and ship["cargo"] == [27, 40, 40, 40], ship["cargo"]

    # first replenish (next_replenish=0) arms the timer, no restock
    ship["next_replenish"] = 0.0
    assert build_uplink.replenish_ships(srv) == 0
    assert ship["cargo_available"] == 1 and ship["next_replenish"] > 0

    # not due -> no restock
    ship["next_replenish"] = _t.monotonic() + 100.0
    assert build_uplink.replenish_ships(srv) == 0

    # due -> restock one
    ship["next_replenish"] = _t.monotonic() - 1.0
    assert build_uplink.replenish_ships(srv) == 1 and ship["cargo_available"] == 2

    # at capacity -> no restock
    build_uplink.ship_set_cargo_available(srv, ship, 3)
    assert build_uplink.replenish_ships(srv) == 0
    print("test_supply_ship_cargo_replenish: PASSED")
    return True


def test_cargo_deploy_and_drop_request():
    """DROP_REQUEST (0x2b) handler: mode 1 deploys the carried cargo as a building at
    the player's position + consumes a box; mode 0 drops a pickup-able crate (with a
    re-pickup cooldown) + consumes a box. Captured wire format: mode is a u32 (0=drop,
    1=deploy); the packet carries no type/pos so the server uses carried type + player
    pos. Phase 3 slice 2."""
    import time as _t
    from wulfram import build_uplink

    class Srv:
        construction_timeout = 0.0
        deploy_distance = 12.0
        cargo_drop_repickup_cooldown_s = 5.0

        def __init__(self):
            self._building_entities = {}
            self._building_health = {}
            self._building_max_health = {}
            self._dynamic_building_ids = set()
            self._dynamic_building_sources = {}
            self._building_construction = {}
            self._dropped_cargo = {}
            self._next_oid = 30000
            self.broadcasts = []

        def _allocate_dynamic_building_oid(self):
            oid = self._next_oid
            self._next_oid += 1
            return oid

        def _building_max_health_for_type(self, et):
            return 2000.0

        def _terrain_ground_z_at(self, x, y):
            return 5.0

        def _rebuild_static_world_raycast_index(self):
            return None

        def _broadcast_dynamic_entity_definition(self, *, entity_id, entity_type, team_id,
                                                 pos, heading, is_static, cargo_contained_type=None):
            self.broadcasts.append((entity_id, entity_type, tuple(round(v) for v in pos)))
            return 1

        def _building_lifecycle_base_event(self, oid, action):
            return {"oid": oid, "action": action}

        def _remember_building_lifecycle_event(self, ev):
            return ev

        def _set_player_carry(self, ctx, *, cargo_type, cargo_count, has_uplink):
            ctx.cargo_type = cargo_type
            ctx.cargo_count = cargo_count
            ctx.has_uplink = has_uplink
            return {"changed": True}

        def _drop_cargo_crate(self, pos, cargo_type, team_id):
            oid = self._allocate_dynamic_building_oid()
            self._dropped_cargo[oid] = {"pos": tuple(pos), "cargo_type": cargo_type,
                                        "team_id": team_id}
            return oid

    def mk_ctx():
        return SimpleNamespace(
            client_id=1, entity_id=0x14EA, cargo_type=25, cargo_count=1, has_uplink=False,
            player_heading=0.0, player_pos=(4950.0, 5100.0, 5.0),
            session=SimpleNamespace(team_id=2, entity_id=0x14EA),
        )

    # --- mode 1: deploy builds the carried type in front of the player + consumes cargo
    srv = Srv()
    ctx = mk_ctx()
    r = build_uplink.handle_drop_request(srv, ctx, 1)
    assert r["ok"] is True, r
    oid = r["oid"]
    assert oid in srv._building_entities, "deploy did not create a building"
    b = srv._building_entities[oid]
    assert int(b.entity_type) == 25, b.entity_type           # Power Cell (carried)
    assert round(b.x) == 4962 and round(b.y) == 5100, (b.x, b.y)  # 12u in front (heading 0 -> +x)
    assert ctx.cargo_type == 0 and ctx.cargo_count == 0, "deploy must consume the carried box"
    assert srv.broadcasts and srv.broadcasts[-1][1] == 25, srv.broadcasts

    # not carrying -> deploy is a no-op
    ctx2 = mk_ctx(); ctx2.cargo_type = 0; ctx2.cargo_count = 0
    r2 = build_uplink.handle_drop_request(srv, ctx2, 1)
    assert r2["ok"] is False and r2.get("error") == "not_carrying", r2

    # --- mode 0: drop spawns a crate + consumes cargo + sets a re-pickup cooldown
    srv3 = Srv()
    ctx3 = mk_ctx()
    r3 = build_uplink.handle_drop_request(srv3, ctx3, 0)
    assert r3["ok"] is True, r3
    crate_oid = r3["oid"]
    assert crate_oid in srv3._dropped_cargo, "drop did not spawn a crate"
    crate = srv3._dropped_cargo[crate_oid]
    assert crate["cargo_type"] == 25, crate
    assert crate.get("pickup_after", 0.0) > _t.monotonic(), "drop must set a re-pickup cooldown"
    assert ctx3.cargo_type == 0 and ctx3.cargo_count == 0, "drop must consume the carried box"

    print("test_cargo_deploy_and_drop_request: PASSED")
    return True


def test_death_auto_respawn_schedules_delayed_spawn():
    """death=respawn (user-requested): on death, _enter_death_deploy_state schedules a
    delayed auto-spawn on the PRESERVED team (mirroring the respawn command) instead of
    the manual flag-click flow -- when death_auto_respawn is on. Off -> manual flow
    (timer cleared, phase TEAM_SELECT)."""
    import time as _t

    class Srv:
        death_auto_respawn = True
        death_respawn_delay_s = 7.0

        def _pick_spawn_point(self, team_id):
            return {"oid": 5002, "x": 5050.0, "y": 5050.0, "z": 5.0, "team": team_id}

    def mk_ctx():
        ctx = ClientContext(client_id=1, client_addr=("10.10.10.2", 50000),
                            session=Session(), entity_id=0x14EA)
        ctx.session.team_id = 2
        ctx.session.in_game = True
        ctx.session.phase = Phase.IN_GAME
        return ctx

    # ON: schedules delayed auto-spawn on the preserved team
    srv = Srv()
    ctx = mk_ctx()
    t0 = _t.monotonic()
    team = WulframServer._enter_death_deploy_state(srv, ctx)
    assert team == 2, team
    assert ctx.session.delayed_spawn_team == 2, ctx.session.delayed_spawn_team
    assert ctx.session.delayed_spawn_time >= t0 + 6.5, ctx.session.delayed_spawn_time
    assert ctx.pending_respawn_pos == (5050.0, 5050.0, 5.0), getattr(ctx, "pending_respawn_pos", None)
    assert ctx.session.in_game is False
    # does NOT force TEAM_SELECT (matches _do_respawn so _auto_join_team fires)
    assert ctx.session.phase == Phase.IN_GAME, ctx.session.phase

    # OFF: manual flag-click flow (timer cleared, TEAM_SELECT)
    srv2 = Srv(); srv2.death_auto_respawn = False
    srv2.roster_sent = False
    ctx2 = mk_ctx()
    WulframServer._enter_death_deploy_state(srv2, ctx2)
    assert ctx2.session.delayed_spawn_team == 0, ctx2.session.delayed_spawn_team
    assert ctx2.session.phase == Phase.TEAM_SELECT, ctx2.session.phase

    print("test_death_auto_respawn_schedules_delayed_spawn: PASSED")
    return True


def test_match_flow_clock_and_round_end():
    """Match flow (Phase 3 slice 4): GAME_CLOCK uses the inverted active flag (running
    -> 0), the round timer counts down, and round end announces a winner (most team
    kills) + RESET_GAME + a fresh round (phase bump, timer reset)."""
    import time as _t
    from wulfram import match_flow
    from wulfram.packets import build_game_clock

    # Wire format: running clock sends active_flag=0 (byte 5), paused sends 1.
    assert build_game_clock(running=True, round_time_ms=1000)[5] == 0x00, "running flag must be 0"
    assert build_game_clock(running=False, round_time_ms=0)[5] == 0x01, "paused flag must be 1"

    class Client:
        def __init__(self, team, kills):
            self.session = SimpleNamespace(team_id=team)
            self.kills = kills

    class Srv:
        match_flow_enabled = True
        match_round_duration_s = 600.0
        match_clock_interval_s = 5.0

        def __init__(self, clients):
            self._clients = clients
            self.sent = []

        def _snapshot_in_game_clients(self):
            return list(self._clients)

        def _send_packet_to_client(self, client, pkt, *, prefer_tcp=True, allow_udp_fallback=False):
            self.sent.append(pkt[0])
            return True

    # Winner = team with the most kills (Blue=2 with 5 > Red=1 with 3).
    srv = Srv([Client(1, 3), Client(2, 5)])
    match_flow.init_match_state(srv)
    assert "Blue wins" in match_flow.winner_message(srv), match_flow.winner_message(srv)

    # Round timer counts down from the duration.
    assert match_flow.remaining_ms(srv) > 599_000

    # Force expiry -> end_round announces (0x1f chat) + RESET_GAME (0x3f) + new GAME_CLOCK (0x2f),
    # bumps the phase, and resets the timer.
    srv._match_round_start = _t.monotonic() - 10_000.0  # already expired
    assert match_flow.remaining_ms(srv) == 0
    phase_before = srv._match_phase
    match_flow.update_match_flow(srv)
    assert 0x3F in srv.sent, "RESET_GAME not broadcast on round end"
    assert 0x2F in srv.sent, "new GAME_CLOCK not broadcast on round end"
    assert 0x1F in srv.sent, "winner chat not broadcast on round end"
    assert srv._match_phase == phase_before + 1, "phase did not advance"
    assert match_flow.remaining_ms(srv) > 599_000, "round timer did not reset"

    # Ties / no kills -> draw.
    srv2 = Srv([Client(1, 2), Client(2, 2)])
    match_flow.init_match_state(srv2)
    assert "draw" in match_flow.winner_message(srv2).lower(), match_flow.winner_message(srv2)

    # Gated off -> no-op.
    srv3 = Srv([Client(1, 1)]); srv3.match_flow_enabled = False
    match_flow.init_match_state(srv3)
    srv3._match_round_start = _t.monotonic() - 10_000.0
    match_flow.update_match_flow(srv3)
    assert not srv3.sent, "match flow must be a no-op when disabled"

    # Empty server -> idle (don't burn rounds with no players).
    srv4 = Srv([]); match_flow.init_match_state(srv4)
    srv4._match_round_start = _t.monotonic() - 10_000.0
    match_flow.update_match_flow(srv4)
    assert not srv4.sent, "empty server must not run the round"

    print("test_match_flow_clock_and_round_end: PASSED")
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
        test_lead_extrapolated_correction_pose,
        test_carrying_info_carry_state,
        test_cargo_pickup_proximity,
        test_build_require_cargo_gate,
        test_construction_timer_lazy_complete,
        test_deconstruct_refund,
        test_deconstruction_timer,
        test_removal_clears_timer_dicts,
        test_cargo_drop_and_crate_pickup,
        test_supply_ship_cargo_replenish,
        test_behavior_spawn_enabled_defaults_on_for_entry_map_spawn,
        test_input_sync_diagnosis_distinguishes_idle_snapback_from_correction_failure,
        test_input_sync_diagnosis_reports_movement_without_targeted_corrections,
        test_input_sync_diagnosis_counts_unsolicited_correction_stream,
        test_spawn_override_wins_over_map_spawn_points,
        test_spawn_at_point_honors_clicked_pad_when_default_configured,
        test_spawn_at_point_honors_vehicle_selection,
        test_recent_control_pose_blocks_in_game_spawn_override,
        test_recent_control_pose_blocks_delayed_auto_spawn,
        test_recent_control_pose_repairs_unstamped_large_jump,
        test_recent_control_pose_repair_can_be_disabled,
        test_map_entity_z_aligns_buried_entities_to_terrain,
        test_map_entity_z_preserves_raw_z_above_physics_terrain,
        test_map_entity_z_preserves_elevated_entities,
        test_control_pose_reset_updates_ground_override,
        test_tank_softbody_spawn_pose_does_not_pin_ground_override,
        test_pulse_shell_default_spawn_uses_recovered_muzzle_offset,
        test_projectile_fire_pose_uses_replay_history_when_available,
        test_projectile_body_source_can_opt_into_body_pitch,
        test_projectile_fire_pose_rejects_stale_fallback_yaw,
        test_projectile_body_pitch_defaults_on_with_env_optout,
        test_remote_spawn_points_use_udp_not_tcp,
        test_login_bootstrap_mode_routes_og_client_by_login_flow,
        test_send_initial_game_data_og_bootstrap_order,
        test_remote_want_updates_suppresses_empty_tcp_update_array,
        test_relogin_honors_changed_username_and_remembers_last,
        test_build_chat_message_comm_layout,
        test_player_chat_respawn_despawns_and_clears_cached_spawn,
        test_player_chat_respawn_via_tcp_comm_handler,
        test_relay_player_chat_routes_by_mode,
        test_kill_feed_broadcasts_to_all_in_game,
        test_build_update_array_remote_heartbeat_shape,
        test_server_remote_heartbeat_helper_keeps_full_local_state,
        test_server_remote_heartbeat_helper_pre_state_request_is_spawn_safe,
        test_remote_state_sync_reply_uses_safe_local_player_shape_when_ready,
        test_remote_state_sync_reply_stays_spawn_safe_immediately_after_spawn,
        test_remote_state_sync_reply_stays_spawn_safe_after_spawn_delay,
        test_remote_state_sync_reply_stays_safe_without_post_spawn_input_after_delay,
        test_remote_state_sync_reply_emits_view_update_with_fresh_remote_timestamp,
        test_loopback_state_sync_reply_keeps_request_timestamp,
        test_remote_state_request_queues_visible_correction_burst,
        test_state_request_burst_rate_cap,
        test_correction_burst_due_drains_under_gate,
        test_state_request_queues_burst_under_default_gate,
        test_remote_active_movement_suppresses_visible_correction_burst,
        test_state_request_active_movement_skips_view_update_correction,
        test_batched_state_request_sees_later_action_update_movement,
        test_remote_empirical_view_update_correction_uses_fresh_remote_timestamp,
        test_view_update_create_tank_decodes_definition_shape,
        test_remote_empirical_view_update_define_correction_uses_definition_shape,
        test_remote_state_sync_defaults_to_live_snapshot_for_remote_og,
        test_remote_state_sync_reply_uses_request_aligned_authoritative_pose,
        test_remote_promoted_heartbeat_stays_short_form_safe,
        test_remote_state_sync_reply_keeps_full_motion_when_stable,
        test_local_player_sync_rotation_uses_heading_not_player_yaw,
        test_send_tank_uses_local_player_sync_rotation,
        test_spawn_wf_minimal_uses_local_player_sync_rotation,
        test_player_body_rotation_preserves_pitch_for_remote_entities,
        test_remote_sync_heartbeat_helper_uses_heading_not_player_yaw,
        test_remote_spawn_bootstrap_heartbeat_uses_safe_full_transform_shape,
        test_state_request_does_not_overwrite_client_tick_offset,
        test_remote_udp_ping_request_gets_og_safe_reply,
        test_udp_ping_reply_default_policy_is_loopback_only,
        test_jump_velocity_update_packet_uses_spawn_safe_local_state_for_remote_og,
        test_translation_velocity_quantizer_matches_decompile_defaults,
        test_server_remote_local_state_kwargs_use_full_tank_shape,
        test_server_remote_entity_packets_use_safe_local_state_after_promotion,
        test_server_remote_projectile_spawn_uses_viewer_local_state,
        test_server_remote_projectile_update_uses_safe_local_state_after_promotion,
        test_caltrop_projectile_spawn_and_update_decode_promoted_wire_shape,
        test_loopback_projectile_update_stays_entity_only,
        test_server_remote_player_info_uses_spawn_safe_local_state,
        test_remote_player_info_packet_short_local_state_layout,
        test_remote_spawn_entry_transition_sends_canonical_packets,
        test_spawn_entry_transition_stays_off_by_default,
        test_control_game_clock_builder_matches_packet_signature,
        test_weapon_system_og_direct_trigger_slot_fires_pulse_shell,
        test_weapon_system_pulse_shell_respects_pitch_when_enabled,
        test_weapon_system_og_direct_trigger_slots_fire_promoted_projectiles,
        test_weapon_system_caltrop_uses_promoted_lifecycle_constants,
        test_weapon_system_chain_gun_autocannon_fire_slot_hitscan_path,
        test_weapon_system_held_fire_repeats_on_cooldown,
        test_weapon_system_accepts_empty_action_update_keepalive,
        test_weapon_system_slot5_release_preserves_og_slider_value,
        test_weapon_system_action_dump_slot5_zero_preserves_og_slider_value,
        test_weapon_system_action_dump_slot5_nonzero_updates_og_slider_value,
        test_tank_softbody_control_ignores_live_slot6_lean_by_default,
        test_send_entity_create_uses_udp_only,
        test_og_viewer_replication_gates_skip_remote_only,
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
        test_server_tank_motion_uses_fuel_mobility_factor,
        test_server_tank_motion_reduces_mobility_when_low_fuel,
        test_tank_ground_contact_damping_limits_low_hover_speed,
        test_tank_high_hover_uses_linear_damping_for_w_motion,
        test_remote_og_movement_input_delay_replays_prior_axis_sample,
        test_remote_og_movement_input_delay_can_probe_nearest_axis_sample,
        test_remote_og_movement_input_can_select_bounded_after_target,
        test_remote_og_movement_input_reports_nonzero_time_candidates,
        test_remote_og_movement_input_history_window_is_bounded_near_target,
        test_remote_og_movement_input_tick_probe_reports_tick_domain_candidates,
        test_remote_og_movement_input_can_select_tick_domain_candidates,
        test_server_jump_jets_apply_fixed_step_rising_edge,
        test_server_jump_jets_have_visible_peak_under_default_gravity,
        test_server_jump_jets_use_tank_body_up_direction,
        test_server_jump_jet_landing_guard_rejects_large_world_collision_projection,
        test_server_tank_terrain_projection_guard_rejects_fast_straight_side_shove,
        test_server_jump_jets_queue_remote_og_correction_burst,
        test_server_motion_clamps_to_move_adjust,
        test_server_motion_reclamps_below_ground_after_collision_response,
        test_server_motion_uses_physics_terrain_offset_for_vehicle_ground,
        test_server_motion_releases_spawn_ground_override_on_terrain_departure,
        test_server_motion_releases_ground_override_when_terrain_changes_under_tank,
        test_remote_team_select_uses_remote_idle_timeout,
        test_tank_surface_state_uses_spring_base_clearance_target,
        test_tank_surface_state_uses_behavior_spring_offsets,
        test_tank_surface_state_rotates_spring_points_through_body_pose,
        test_server_tank_drive_uses_body_matrix_when_body_pose_live,
        test_contact_yaw_velocity_feeds_next_vehicle_physics_tick,
        test_surface_attitude_uses_post_yaw_heading_after_drive_step,
        test_tank_surface_attitude_uses_spring_normal_for_replication,
        test_tank_surface_attitude_steps_toward_spring_normal,
        test_tank_surface_attitude_force_path_uses_point_clearance_torque,
        test_tank_surface_attitude_reuses_force_sample_state_without_resampling,
        test_heading_physics_sync_preserves_spring_body_pose,
        test_tank_softbody_support_pulls_down_from_compact_equilibrium,
        test_tank_softbody_supports_gravity_at_og_flat_height,
        test_tank_softbody_slot5_changes_response_without_jumpjet,
        test_tank_softbody_control_prefers_live_og_slot5,
        test_weapon_system_upward_thrust_defaults_to_og_idle_slot5,
        test_tank_suspension_legacy_compact_stiffness_matches_old_equilibrium,
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
        test_triangle_cbsp_mesh_vertex_probe_reports_embedded_vertex,
        test_triangle_cbsp_mesh_edge_terrain_plane_probe_reports_crossing_edge,
        test_triangle_cbsp_mesh_edge_terrain_plane_probe_reports_second_tilted_fixture,
        test_triangle_cbsp_mesh_edge_terrain_plane_probe_preserves_endpoint_hit,
        test_triangle_cbsp_mesh_edge_endpoint_probe_prefers_deeper_node,
        test_triangle_cbsp_node_plane_vertex_probe_uses_split_normal,
        test_triangle_cbsp_strict_probe_skips_heuristic_vertex_fallback,
        test_triangle_cbsp_guess7_order_probe_rejects_target_point_shortcut,
        test_box_collision_returns_first_contact_in_grid_order,
        test_model_collision_returns_first_contact_in_grid_order,
        test_model_collision_can_probe_upward_min_depth_selection,
        test_model_bounds_contact_can_probe_upward_min_depth_selection,
        test_model_collision_response_normal_can_probe_terrain_triangle_normal,
        test_triangle_cbsp_contact_returns_first_leaf_hit,
        test_building_collision_skips_aabb_for_mesh_backed_building,
        test_repair_pad_collision_does_not_block_vehicle_movement,
        test_building_collision_team_variant_matches_client_helper,
        test_server_team_model_name_matches_client_helper,
        test_effective_inactivity_timeout_extends_remote_ingame_clients,
        test_entity_world_collision_defaults_enabled_with_env_optout,
        test_entity_world_collision_can_be_disabled_for_player_sync,
        test_entity_world_collision_prefers_mesh_contact_when_collision_model_exists,
        test_entity_world_collision_model_defaults_to_legacy_lift,
        test_entity_world_collision_model_entity_origin_can_be_requested,
        test_entity_world_collision_model_simplified_variant_is_opt_in,
        test_entity_world_collision_model_contact_can_use_body_matrix_probe,
        test_entity_world_collision_records_raw_origin_probe_without_applying_contact,
        test_entity_world_collision_reference_pose_probe_records_pre_step_contact_when_enabled,
        test_entity_world_collision_reference_pose_pair_record_contact_can_apply_when_enabled,
        test_entity_world_collision_reference_pose_pair_response_preserves_position,
        test_entity_world_collision_cached_pair_record_contact_can_bridge_lifted_clear_when_enabled,
        test_entity_world_collision_deferred_prestep_pair_record_contact_when_enabled,
        test_entity_world_collision_deferred_prestep_pair_record_probe_is_read_only,
        test_entity_world_collision_dirty_reference_pair_probe_is_read_only,
        test_entity_world_collision_dirty_reference_pair_response_preserves_position,
        test_entity_world_collision_dirty_reference_pair_response_max_distance_rejects,
        test_entity_world_collision_dirty_bounds_safe_response_limits_box_fallback,
        test_entity_world_collision_pair_record_contact_can_continue_remaining_step_when_enabled,
        test_entity_world_collision_pair_record_continue_sweeps_transient_contact,
        test_entity_world_collision_pair_record_schedule_probe_reports_without_applying,
        test_entity_world_collision_frame_phase_report_first_probe_is_read_only,
        test_entity_world_collision_spatial_ref_schedule_probe_reports_without_applying,
        test_entity_world_collision_pair_record_schedule_response_probe_is_read_only,
        test_entity_world_collision_phase_lookahead_probe_reports_future_pair_contact,
        test_entity_world_collision_phase_lookahead_contact_applies_without_future_teleport,
        test_entity_world_collision_phase_lookahead_queue_resolves_when_due,
        test_entity_world_collision_phase_backtrack_probe_reports_prior_pair_contact,
        test_entity_world_collision_phase_backtrack_contact_replays_to_endpoint,
        test_entity_world_collision_pair_record_contact_applies_decompile_face_gated_contact,
        test_entity_world_collision_pair_record_can_probe_raw_solver_linear_response,
        test_entity_world_collision_pair_record_contact_uses_shallow_upward_selection,
        test_entity_world_collision_pair_record_contact_accepts_og_straightaway_face,
        test_tank_clean_terrain_contact_uses_pair_solver_by_default,
        test_tank_clean_terrain_projection_order_uses_opposite_probe_by_default,
        test_tank_default_raw_origin_fallback_uses_empirical_straightaway_face,
        test_tank_clean_side_contact_uses_face_fallback_without_angular_blowup,
        test_tank_clean_terrain_contact_rejects_pathological_depth_by_default,
        test_entity_world_collision_raw_origin_fallback_applies_guarded_pair_solver,
        test_entity_world_collision_raw_origin_fallback_clamps_solver_velocity_delta,
        test_entity_world_collision_raw_origin_fallback_can_use_terrain_normal_probe,
        test_entity_world_collision_raw_origin_fallback_can_use_contact_face_normal,
        test_entity_world_collision_raw_origin_fallback_can_blend_radial_face_forward_up,
        test_entity_world_collision_raw_origin_fallback_can_use_closing_velocity_delta,
        test_entity_world_collision_raw_origin_fallback_can_preserve_angular_velocity,
        test_entity_world_collision_raw_origin_fallback_skips_normal_delta_when_separating,
        test_entity_world_collision_restores_previous_motion_after_nonfinite_input,
        test_entity_world_collision_rejects_pathological_finite_fallback_state,
        test_entity_world_half_extents_preserves_mesh_z_extent,
        test_entity_origin_probe_uses_capped_pair_solver_contact_response,
        test_entity_origin_probe_can_retest_inactive_penetrating_contact_when_enabled,
        test_entity_origin_probe_suppresses_static_terrain_yaw_feedback_by_default,
        test_entity_origin_probe_uses_fed_target_for_interpolation_decision,
        test_static_terrain_constraint_sleeping_body_uses_decompile_scaling,
        test_static_terrain_constraint_retests_inactive_penetrating_contact,
        test_static_terrain_constraint_friction_uses_pre_normal_projection_buffer,
        test_static_terrain_constraint_opposite_projection_probe_activates_separating_penetration,
        test_static_terrain_constraint_can_probe_entity_rotation_for_angular_frame,
        test_static_terrain_constraint_can_probe_contact_iterative_solver_shape,
        test_entity_interpolation_decision_matches_decompile_gates,
        test_entity_origin_probe_applies_pair_solver_at_contact_time,
        test_entity_origin_probe_can_repeat_bucketed_pair_contacts,
        test_lifted_timed_probe_can_use_guarded_raw_origin_contact,
        test_raw_origin_closing_gate_uses_solver_contact_projection,
        test_lifted_timed_probe_can_use_guarded_raycast_contact,
        test_timed_pair_sweep_scan_finds_transient_midframe_contact,
        test_pair_record_timed_sweep_uses_shallow_selection_when_enabled,
        test_collision_at_start_can_use_iterative_world_separation,
        test_entity_world_collision_falls_back_to_box_without_collision_model,
        test_entity_world_collision_uses_dirty_terrain_raycast_branch,
        test_entity_world_collision_uses_dirty_contact_before_raycast,
        test_entity_world_collision_uses_dirty_bounds_contact_store,
        test_entity_world_collision_dirty_model_center_defaults_to_lifted_center,
        test_entity_world_collision_dirty_model_center_can_use_raw_center,
        test_entity_world_collision_dirty_bounds_store_accepts_tiny_contact,
        test_pair_record_bounds_sat_can_feed_safety_limited_contact,
        test_entity_world_collision_dirty_bounds_store_uses_contact_point_radius_resolution,
        test_entity_world_collision_pathological_dirty_bounds_contact_falls_back_to_raycast,
        test_entity_world_collision_downward_dirty_bounds_contact_falls_back_to_raycast,
        test_entity_world_collision_far_dirty_bounds_contact_falls_back_to_raycast,
        test_entity_world_collision_horizontal_dirty_bounds_contact_falls_back_to_raycast,
        test_entity_world_collision_dirty_bounds_phase_uses_xy_broadphase,
        test_dirty_bounds_contact_helpers_skip_triangle_prefilter,
        test_decompile_bounds_contact_reports_sat_metadata,
        test_terrain_height_uses_decompile_triangle_plane_not_bilinear,
        test_terrain_slope_uses_active_triangle_plane_normal,
        test_player_terrain_probe_reports_decompile_triangle_state,
        test_terrain_cell_triangles_match_decompile_order,
        test_terrain_raycast_patch_traverse_uses_start_to_end_sector_sweep,
        test_terrain_patch_raycast_cells_uses_decompile_dda_order,
        test_terrain_patch_raycast_cells_uses_decompile_axis_flag_step_policy,
        test_entity_world_collision_uses_persistent_reference_pos_for_dirty_branch,
        test_entity_world_collision_refreshes_reference_on_clean_contact,
        test_entity_world_collision_preserves_reference_on_clean_miss_until_dirty,
        test_entity_world_collision_can_preserve_dirty_miss_reference_for_probe,
        test_entity_world_collision_dirty_miss_still_records_raw_origin_probe,
        test_entity_world_collision_dirty_model_clear_skips_box_fallback_by_default,
        test_entity_world_collision_dirty_model_clear_can_use_box_fallback,
        test_entity_world_collision_can_probe_decompile_box_shape_with_model_loaded,
        test_entity_world_collision_dirty_threshold_uses_mesh_min_half_extent,
        test_entity_world_collision_dirty_raycast_uses_contact_separation,
        test_entity_world_collision_dirty_raycast_uses_decompile_degenerate_threshold,
        test_entity_world_collision_static_separation_matches_decompile_clamp,
        test_entity_world_collision_clean_path_uses_single_contact_store,
        test_roster_entry_stays_tcp_only,
        test_broadcast_player_stats_stays_tcp_only,
        test_remote_combat_observer_stats_gate_skips_nonparticipant_og,
        test_control_pos_exact_reset_targets_specific_client,
        test_enter_game_can_target_team_select_tcp_only_client,
        test_control_pos_can_apply_live_tap_velocity,
        test_control_heading_set_preserves_yaw_sign_convention,
        test_solo_local_player_keepalive_shape_triggers_og_state_request_gate,
        test_view_update_pos_without_rot_clamps_to_safe_shape,
        test_tick_loop_start_guard_allows_one_live_thread,
        test_tick_pacer_preserves_capped_catchup_backlog,
        test_client_weapon_fire_telemetry_records_input_projectiles,
        test_client_hitscan_fire_telemetry_records_fire,
        test_client_hitscan_fire_damages_lane_target,
        test_caltrop_projectile_steers_toward_nearest_target_in_help_range,
        test_caltrop_projectile_damage_uses_light_bomblet_amount_and_cleanup,
        test_players_json_includes_transport_addresses,
        test_health_control_uses_server_heartbeat_helper,
        test_energy_control_sets_absolute_or_fractional_energy,
        test_comm_message_request_decodes_og_type2_build_command,
        test_build_uplink_parser_accepts_slice_buildable_names,
        test_build_uplink_command_creates_dynamic_building,
        test_build_uplink_commands_create_all_slice_buildable_types,
        test_dynamic_powercell_service_restores_energy_with_ambient_regen_disabled,
        test_dynamic_building_damage_control_destroys_and_records_delete,
        test_dynamic_building_projectile_damage_destroys_and_records_delete,
        test_uplink_mvp_bootstrap_sends_minimal_status_packets,
        test_buildings_json_exposes_playable_slice_observability,
        # Previously-orphaned tests (defined but never registered -> silently not
        # run); registered 2026-06-26 during the post-decomposition test audit.
        test_projectile_world_hit_mesh_broadphase_uses_bounding_sphere,
        test_remote_spawn_create_update_array_avoids_tcp,
        test_remote_state_sync_reply_remaps_client_tick_to_server_history,
        test_remote_state_sync_reuses_cached_sample_when_replay_window_misses,
        test_udp_team_switch_can_reassert_roster_without_entry_packets,
        test_udp_team_switch_can_send_update_stats_over_tcp,
        test_udp_team_switch_can_suppress_duplicate_entry_packets,
        test_udp_team_switch_can_use_team_first_update_stats_variant,
        test_udp_team_switch_sends_update_stats_before_reincarnate,
        test_cargo_deploy_and_drop_request,
        test_match_flow_clock_and_round_end,
        test_death_auto_respawn_schedules_delayed_spawn,
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
