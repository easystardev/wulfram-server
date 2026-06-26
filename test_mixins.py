"""Guards for the server.py mixin decomposition (2026-06-25/26).

WulframServer was decomposed from a ~23k-line God class into a thin core plus 8
method-only mixins (Config/Raycast/Replication/Spawn/Combat/RemoteSync/
Correction/Tick). These cheap structural checks catch the two regressions the
behavior tests don't surface obviously: an accidentally-dropped base class
(method silently falls back to... nothing) and an import cycle introduced by a
new cross-module import. Run standalone: `uv run python test_mixins.py`.
"""
import sys
import importlib

sys.path.insert(0, "../shared")

EXPECTED_MIXINS = [
    ("server_config", "ConfigMixin"),
    ("server_raycast", "RaycastMixin"),
    ("server_replication", "ReplicationMixin"),
    ("server_spawn", "SpawnMixin"),
    ("server_combat", "CombatMixin"),
    ("server_remote", "RemoteSyncMixin"),
    ("server_corrections", "CorrectionMixin"),
    ("server_tick", "TickMixin"),
]

# A representative method moved into each mixin -- must still resolve on the
# composed class (via the MRO). If a base class is dropped, these vanish.
SAMPLE_METHODS = [
    "_init_correction_config",            # ConfigMixin
    "_raycast_world",                     # RaycastMixin
    "_build_local_state_heartbeat",       # ReplicationMixin
    "_spawn_wf_style",                    # SpawnMixin
    "_apply_damage",                      # CombatMixin
    "_send_remote_player_updates",        # RemoteSyncMixin
    "_send_state_sync_snapshot",          # CorrectionMixin
    "_resolve_entity_world_collision",    # TickMixin
]


def test_every_mixin_module_imports():
    """Each server_<area>.py imports cleanly (catches cycles / missing names)."""
    for mod_name, _cls in EXPECTED_MIXINS:
        importlib.import_module(f"wulfram.{mod_name}")
    print("  test_every_mixin_module_imports: PASSED")
    return True


def test_wulframserver_mro_has_all_mixins():
    """WulframServer must inherit all 8 mixins (catches a dropped base class)."""
    from wulfram.server import WulframServer

    mro_names = {c.__name__ for c in WulframServer.__mro__}
    for mod_name, cls_name in EXPECTED_MIXINS:
        mod = importlib.import_module(f"wulfram.{mod_name}")
        assert hasattr(mod, cls_name), f"{mod_name} missing class {cls_name}"
        assert cls_name in mro_names, f"{cls_name} not in WulframServer MRO"
    print("  test_wulframserver_mro_has_all_mixins: PASSED")
    return True


def test_moved_methods_resolve_on_composed_class():
    """A representative moved method per mixin still resolves on WulframServer."""
    from wulfram.server import WulframServer

    for name in SAMPLE_METHODS:
        assert callable(getattr(WulframServer, name, None)), f"{name} not resolvable"
    print("  test_moved_methods_resolve_on_composed_class: PASSED")
    return True


def main():
    tests = [
        test_every_mixin_module_imports,
        test_wulframserver_mro_has_all_mixins,
        test_moved_methods_resolve_on_composed_class,
    ]
    passed = failed = 0
    for t in tests:
        try:
            if t():
                passed += 1
            else:
                failed += 1
                print(f"  {t.__name__}: FAILED (returned falsy)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  {t.__name__}: FAILED - {e}")
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
