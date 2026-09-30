"""Captured wire fixture gates: physics fields, plus explicit approximation override."""
import hashlib
import struct
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'client'))
from wulfram_client.network.behavior import parse_behavior
from wulfram import packets


def test_captured_tank_primary_trajectory_fields():
    fixture = (Path(__file__).parent / 'testdata/production-behavior-20260907.bin').read_bytes()
    assert hashlib.sha256(fixture).hexdigest() == '290658f5f8bc9edfc80a732d69b7ca2e05847d1a9d4e71d87c13b4d1ebbec688'
    local = packets.build_behavior_packet()
    assert local[0] == fixture[0] == 0x24
    # Offsets include opcode, but exclude the TCP length prefix.
    for offset, fixed in ((101, 61584), (129, 22937600), (133, 5898), (137, 5243)):
        assert local[offset:offset + 4] == fixture[offset:offset + 4] == struct.pack('>i', fixed)
    # Adjacent, separately unaudited parameter is deliberately not imported.
    assert local[125:129] == struct.pack('>i', 100 * 65536)
    # The correction must not silently promote any of the other 51 slots.
    for slot in range(1, 52):
        start = 96 + slot * 45
        for relative, value in ((5, 1), (33, 1000), (37, 500), (41, 1)):
            assert local[start + relative:start + relative + 4] == struct.pack('>i', value * 65536)


def test_captured_physics_defaults(monkeypatch):
    for name in ('LONGITUDINAL', 'LATERAL'):
        monkeypatch.delenv('WULFRAM_BEHAVIOR_TANK_SPRING_' + name, raising=False)
    fixture = (Path(__file__).parent / 'testdata/production-behavior-20260907.bin').read_bytes()
    assert hashlib.sha256(fixture).hexdigest() == '290658f5f8bc9edfc80a732d69b7ca2e05847d1a9d4e71d87c13b4d1ebbec688'
    prod = parse_behavior(fixture)
    local = parse_behavior(packets.build_behavior_packet())
    assert local.spring_states == prod.spring_states
    assert local.active_vehicle_physics[0] == prod.active_vehicle_physics[0]


def test_explicit_symmetric_geometry_override(monkeypatch):
    monkeypatch.setenv('WULFRAM_BEHAVIOR_TANK_SPRING_LONGITUDINAL', '3.4')
    local = parse_behavior(packets.build_behavior_packet())
    assert all(p.pos[2] == 0 for state in local.spring_states for p in state.points)
    assert all(state == local.spring_states[0] for state in local.spring_states)
