"""Small, map-owned tutorial runtime for original Wulfram clients.

Tutorial files live beside canonical map source as ``tutorial.json``.  The
runtime supports a deliberately small set of typed, ordered triggers.  It
uses authoritative pose plus decoded action state without changing normal game
physics, prediction, or correction behavior.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

from .packets import build_chat_message


def _finite_triplet(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must be a three-number array")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite numbers")
    return result


def _number(value: Any, label: str, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        suffix = f" between {minimum} and {maximum}" if maximum is not None else f" at least {minimum}"
        raise ValueError(f"{label} must be finite and{suffix}")
    return result


def _region(value: Any, label: str) -> dict[str, tuple[float, float, float]]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    minimum = _finite_triplet(value.get("min"), f"{label} min")
    maximum = _finite_triplet(value.get("max"), f"{label} max")
    if any(lo > hi for lo, hi in zip(minimum, maximum)):
        raise ValueError(f"{label} min exceeds max")
    return {"minimum": minimum, "maximum": maximum}


def _inside(pos: tuple[float, float, float], region: dict[str, Any]) -> bool:
    return all(lo <= value <= hi for value, lo, hi in zip(pos, region["minimum"], region["maximum"]))


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _chat_text(value: Any, label: str, *, limit: int = 220) -> str:
    """Validate text before it reaches the original client's ASCII chat packet."""
    text = str(value).strip()[:limit]
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must contain only ASCII characters") from exc
    return text


def parse_definition(raw: Any, *, expected_map: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("tutorial root must be an object")
    if raw.get("format") != "wulfram-tutorial" or raw.get("version") != 1:
        raise ValueError("tutorial must be wulfram-tutorial version 1")
    if str(raw.get("map", "")).lower() != expected_map.lower():
        raise ValueError(f"tutorial map must match {expected_map!r}")
    lessons = raw.get("lessons")
    if not isinstance(lessons, list) or not lessons:
        raise ValueError("tutorial lessons must be a non-empty array")

    targets_raw = raw.get("targets", [])
    if not isinstance(targets_raw, list):
        raise ValueError("tutorial targets must be an array")
    targets: dict[str, dict[str, Any]] = {}
    for index, target in enumerate(targets_raw):
        if not isinstance(target, dict):
            raise ValueError(f"target {index} must be an object")
        target_id = str(target.get("id", "")).strip()
        if not target_id or target_id in targets:
            raise ValueError(f"target {index} has a missing or duplicate id")
        token = str(target.get("token", ""))
        if token not in {"e", "f", "r", "S", "s", "g", "E", "L", "p", "o", "d", "b", "*"}:
            raise ValueError(f"target {target_id} has unsupported token {token!r}")
        targets[target_id] = {
            "id": target_id,
            "label": _chat_text(
                target.get("label", target_id.replace("-", " ").title()),
                f"target {target_id} label", limit=60,
            ),
            "source_entity_id": str(target.get("sourceEntityId", "")).strip(),
            "token": token,
            "team": int(target.get("team", 0)),
            "position": _finite_triplet(target.get("position"), f"target {target_id} position"),
            "protected": bool(target.get("protected", False)),
            "network_visible": bool(target.get("networkVisible", False)),
            "tutorial_health": (
                _number(
                    target.get("tutorialHealth"), f"target {target_id} tutorialHealth",
                    minimum=1.0, maximum=10000.0,
                )
                if target.get("tutorialHealth") is not None else None
            ),
            "runtime_oid": None,
        }

    scenario_raw = raw.get("scenario")
    scenario = None
    if scenario_raw is not None:
        if not isinstance(scenario_raw, dict) or scenario_raw.get("kind") != "free-skirmish":
            raise ValueError("tutorial scenario must be a free-skirmish object")
        objective_target = str(scenario_raw.get("objectiveTarget", "")).strip()
        if objective_target not in targets:
            raise ValueError(f"tutorial scenario references unknown objective target {objective_target!r}")
        threat_target = str(scenario_raw.get("threatTarget", "")).strip()
        if threat_target and threat_target not in targets:
            raise ValueError(f"tutorial scenario references unknown threat target {threat_target!r}")
        scenario = {
            "kind": "free-skirmish",
            "objective_target": objective_target,
            "objective_health": _number(
                scenario_raw.get("objectiveHealth"), "scenario objectiveHealth",
                minimum=1.0, maximum=10000.0,
            ),
            "loss_on_player_death": bool(scenario_raw.get("lossOnPlayerDeath", True)),
            "threat_target": threat_target or None,
            "threat_activation_lesson": str(
                scenario_raw.get("threatActivationLesson", "")
            ).strip() or None,
            "quiet_after_result": bool(scenario_raw.get("quietAfterResult", True)),
            "hit_feedback": bool(scenario_raw.get("hitFeedback", True)),
            "loss_message": _chat_text(
                scenario_raw.get(
                    "lossMessage",
                    "Scenario lost: your tank was destroyed. Redeploy or type 'tutorial reset' to retry.",
                ),
                "scenario lossMessage",
            ),
        }

    normalized = []
    seen_ids: set[str] = set()
    for index, lesson in enumerate(lessons):
        if not isinstance(lesson, dict):
            raise ValueError(f"lesson {index} must be an object")
        lesson_id = str(lesson.get("id", "")).strip()
        prompt = _chat_text(lesson.get("prompt", ""), f"lesson {lesson_id or index} prompt")
        if not lesson_id or lesson_id in seen_ids:
            raise ValueError(f"lesson {index} has a missing or duplicate id")
        if not prompt:
            raise ValueError(f"lesson {lesson_id} needs a prompt")
        seen_ids.add(lesson_id)
        trigger = lesson.get("trigger")
        if not isinstance(trigger, dict):
            raise ValueError(f"lesson {lesson_id} needs a trigger")
        trigger_type = str(trigger.get("type", ""))
        normalized_trigger: dict[str, Any] = {"type": trigger_type}
        if trigger_type == "enter-region":
            normalized_trigger["region"] = _region(trigger.get("region"), f"lesson {lesson_id} region")
            normalized_trigger["require_entry"] = bool(trigger.get("requireEntry", False))
        elif trigger_type == "stop-in-region":
            normalized_trigger["region"] = _region(trigger.get("region"), f"lesson {lesson_id} region")
            normalized_trigger["max_horizontal_speed"] = _number(
                trigger.get("maxHorizontalSpeed"), f"lesson {lesson_id} maxHorizontalSpeed", minimum=0.01
            )
            normalized_trigger["dwell_seconds"] = _number(
                trigger.get("dwellSeconds"), f"lesson {lesson_id} dwellSeconds", minimum=0.1, maximum=30.0
            )
            normalized_trigger["require_entry"] = bool(trigger.get("requireEntry", True))
        elif trigger_type == "strafe-to-region":
            normalized_trigger["start_region"] = _region(
                trigger.get("startRegion"), f"lesson {lesson_id} startRegion"
            )
            normalized_trigger["target_region"] = _region(
                trigger.get("targetRegion"), f"lesson {lesson_id} targetRegion"
            )
            normalized_trigger["min_strafe_input"] = _number(
                trigger.get("minStrafeInput"), f"lesson {lesson_id} minStrafeInput", minimum=0.01, maximum=1.0
            )
            normalized_trigger["min_strafe_seconds"] = _number(
                trigger.get("minStrafeSeconds"), f"lesson {lesson_id} minStrafeSeconds", minimum=0.1, maximum=60.0
            )
            normalized_trigger["min_lateral_distance"] = _number(
                trigger.get("minLateralDistance"), f"lesson {lesson_id} minLateralDistance", minimum=1.0
            )
            normalized_trigger["max_heading_change"] = math.radians(_number(
                trigger.get("maxHeadingChangeDegrees"), f"lesson {lesson_id} maxHeadingChangeDegrees",
                minimum=0.1, maximum=180.0,
            ))
            normalized_trigger["max_forward_input"] = _number(
                trigger.get("maxForwardInput"), f"lesson {lesson_id} maxForwardInput", minimum=0.0, maximum=1.0
            )
        elif trigger_type == "self-confirm":
            normalized_trigger["region"] = _region(trigger.get("region"), f"lesson {lesson_id} region")
        elif trigger_type == "hit-target":
            target_id = str(trigger.get("target", "")).strip()
            if target_id not in targets:
                raise ValueError(f"lesson {lesson_id} references unknown target {target_id!r}")
            weapon = str(trigger.get("weapon", "")).strip().upper()
            if not weapon:
                raise ValueError(f"lesson {lesson_id} needs a weapon")
            normalized_trigger["target"] = target_id
            normalized_trigger["weapon"] = weapon
            normalized_trigger["min_strafe_input"] = _number(
                trigger.get("minStrafeInput", 0.0), f"lesson {lesson_id} minStrafeInput", minimum=0.0, maximum=1.0
            )
        elif trigger_type == "destroy-target":
            target_id = str(trigger.get("target", "")).strip()
            if target_id not in targets:
                raise ValueError(f"lesson {lesson_id} references unknown target {target_id!r}")
            normalized_trigger["target"] = target_id
        elif trigger_type == "turret-damage":
            target_id = str(trigger.get("target", "")).strip()
            if target_id not in targets:
                raise ValueError(f"lesson {lesson_id} references unknown target {target_id!r}")
            normalized_trigger.update({
                "target": target_id,
                "min_damage": _number(
                    trigger.get("minDamage", 0.01), f"lesson {lesson_id} minDamage",
                    minimum=0.001, maximum=1.0,
                ),
                "cease_fire_on_complete": bool(trigger.get("ceaseFireOnComplete", False)),
            })
        elif trigger_type == "service-recovery":
            repair_target = str(trigger.get("repairTarget", "")).strip()
            fuel_target = str(trigger.get("fuelTarget", "")).strip()
            if repair_target not in targets or fuel_target not in targets:
                raise ValueError(f"lesson {lesson_id} references an unknown service target")
            normalized_trigger.update({
                "repair_target": repair_target,
                "fuel_target": fuel_target,
                "min_health_restored": _number(
                    trigger.get("minHealthRestored"), f"lesson {lesson_id} minHealthRestored",
                    minimum=0.01, maximum=1.0,
                ),
                "min_energy_restored": _number(
                    trigger.get("minEnergyRestored"), f"lesson {lesson_id} minEnergyRestored", minimum=0.1,
                ),
                "fixture_health": _number(
                    trigger.get("fixtureHealth", 0.65), f"lesson {lesson_id} fixtureHealth",
                    minimum=0.05, maximum=0.95,
                ),
                "fixture_energy": _number(
                    trigger.get("fixtureEnergy", 15.0), f"lesson {lesson_id} fixtureEnergy", minimum=0.0,
                ),
            })
        elif trigger_type == "cargo-pickup":
            binding = str(trigger.get("binding", "")).strip()
            if not binding:
                raise ValueError(f"lesson {lesson_id} cargo-pickup needs a binding")
            normalized_trigger.update({
                "binding": binding,
                "cargo_type": int(_number(
                    trigger.get("cargoType"), f"lesson {lesson_id} cargoType", minimum=25, maximum=36,
                )),
                "seed_position": _finite_triplet(
                    trigger.get("seedPosition"), f"lesson {lesson_id} seedPosition"
                ),
                "team": int(trigger.get("team", 1)),
                "native_map_fixture": bool(trigger.get("nativeMapFixture", False)),
                "repickup_cooldown_seconds": _number(
                    trigger.get("repickupCooldownSeconds", 5.0),
                    f"lesson {lesson_id} repickupCooldownSeconds", minimum=0.0, maximum=120.0,
                ),
            })
        elif trigger_type == "cargo-carry-to-region":
            binding = str(trigger.get("binding", "")).strip()
            if not binding:
                raise ValueError(f"lesson {lesson_id} cargo carry needs a binding")
            normalized_trigger.update({
                "binding": binding,
                "cargo_type": int(_number(
                    trigger.get("cargoType"), f"lesson {lesson_id} cargoType", minimum=25, maximum=36,
                )),
                "region": _region(trigger.get("region"), f"lesson {lesson_id} region"),
                "require_entry": bool(trigger.get("requireEntry", True)),
            })
        elif trigger_type == "cargo-drop":
            binding = str(trigger.get("binding", "")).strip()
            bind_as = str(trigger.get("bindAs", "")).strip()
            if not binding or not bind_as:
                raise ValueError(f"lesson {lesson_id} cargo drop needs binding and bindAs")
            normalized_trigger.update({
                "binding": binding,
                "bind_as": bind_as,
                "cargo_type": int(_number(
                    trigger.get("cargoType"), f"lesson {lesson_id} cargoType", minimum=25, maximum=36,
                )),
                "region": _region(trigger.get("region"), f"lesson {lesson_id} region"),
            })
        elif trigger_type == "cargo-recover":
            binding = str(trigger.get("binding", "")).strip()
            if not binding:
                raise ValueError(f"lesson {lesson_id} cargo recover needs a binding")
            normalized_trigger.update({
                "binding": binding,
                "cargo_type": int(_number(
                    trigger.get("cargoType"), f"lesson {lesson_id} cargoType", minimum=25, maximum=36,
                )),
                "min_away_distance": _number(
                    trigger.get("minAwayDistance"), f"lesson {lesson_id} minAwayDistance", minimum=1.0,
                ),
            })
        elif trigger_type == "cargo-deploy":
            binding = str(trigger.get("binding", "")).strip()
            bind_as = str(trigger.get("bindAs", "")).strip()
            if not binding or not bind_as:
                raise ValueError(f"lesson {lesson_id} cargo deploy needs binding and bindAs")
            normalized_trigger.update({
                "binding": binding,
                "bind_as": bind_as,
                "cargo_type": int(_number(
                    trigger.get("cargoType"), f"lesson {lesson_id} cargoType", minimum=25, maximum=36,
                )),
                "region": _region(trigger.get("region"), f"lesson {lesson_id} region"),
            })
        elif trigger_type == "deployed-service":
            binding = str(trigger.get("binding", "")).strip()
            service = str(trigger.get("service", "")).strip().lower()
            if not binding:
                raise ValueError(f"lesson {lesson_id} deployed service needs a binding")
            if service not in {"repair", "fuel", "energy"}:
                raise ValueError(f"lesson {lesson_id} has unsupported deployed service {service!r}")
            normalized_trigger.update({
                "binding": binding,
                "service": service,
                "min_restored": _number(
                    trigger.get("minRestored"), f"lesson {lesson_id} minRestored", minimum=0.001,
                ),
                "fixture_health": _number(
                    trigger.get("fixtureHealth", 0.65), f"lesson {lesson_id} fixtureHealth",
                    minimum=0.05, maximum=0.95,
                ),
                "fixture_energy": _number(
                    trigger.get("fixtureEnergy", 15.0), f"lesson {lesson_id} fixtureEnergy", minimum=0.0,
                ),
            })
        else:
            raise ValueError(f"lesson {lesson_id} has unsupported trigger type {trigger_type!r}")
        complete = lesson.get("onComplete", {})
        message = _chat_text(
            complete.get("message", "Completed.") if isinstance(complete, dict) else "Completed.",
            f"lesson {lesson_id} completion message",
        )
        normalized.append({
            "id": lesson_id,
            "prompt": prompt,
            "hint": _chat_text(lesson.get("hint", prompt), f"lesson {lesson_id} hint"),
            "trigger": normalized_trigger,
            "complete_message": message,
        })

    if scenario and scenario["threat_activation_lesson"]:
        activation_lesson = scenario["threat_activation_lesson"]
        if activation_lesson not in seen_ids:
            raise ValueError(
                f"tutorial scenario references unknown activation lesson {activation_lesson!r}"
            )
        scenario["threat_activation_index"] = next(
            index for index, lesson in enumerate(normalized)
            if lesson["id"] == activation_lesson
        )
    elif scenario:
        scenario["threat_activation_index"] = None

    teams_raw = raw.get("teams", [1])
    if not isinstance(teams_raw, list) or not teams_raw:
        raise ValueError("tutorial teams must be a non-empty array")
    teams = {int(team) for team in teams_raw}
    return {
        "map": expected_map,
        "title": str(raw.get("title", expected_map)).strip()[:100],
        "lessons": normalized,
        "teams": teams,
        "targets": targets,
        "scenario": scenario,
        "max_concurrent_learners": int(raw.get("maxConcurrentLearners", 0) or 0),
    }


def definition_path(map_name: str) -> Path:
    override = os.environ.get("WULFRAM_TUTORIAL_FILE", "").strip()
    if override:
        return Path(override).resolve()
    root = Path(__file__).resolve().parents[2]
    source = root / "wulfram-maps" / "maps" / map_name / "tutorial.json"
    if source.exists():
        return source
    return root / "slurpysoft-wulfram" / "data" / "maps" / map_name / "tutorial.json"


def load_for_server(server: object) -> Optional[dict[str, Any]]:
    server.tutorial_definition = None
    server._tutorial_events = []
    server._tutorial_hit_events = []
    server._tutorial_service_events = []
    server._tutorial_logistics_events = []
    if os.environ.get("WULFRAM_TUTORIAL", "auto").strip().lower() in {"0", "false", "off", "no"}:
        return None
    path = definition_path(server.map_name)
    if not path.exists():
        return None
    definition = parse_definition(json.loads(path.read_text(encoding="utf-8")), expected_map=server.map_name)
    server.tutorial_definition = definition
    print(f"[TUTORIAL] Loaded {definition['title']!r}: {len(definition['lessons'])} lesson(s) from {path}")
    return definition


_TARGET_ENTITY_TYPES = {
    "e": 25, "f": 26, "r": 27, "S": 28, "s": 29, "g": 30,
    "E": 31, "L": 32, "p": 33, "o": 34, "d": 35, "b": 36, "*": 37,
}


def resolve_targets(server: object) -> None:
    """Bind stable tutorial source IDs to runtime building OIDs after map load."""
    definition = getattr(server, "tutorial_definition", None)
    if not definition:
        return
    buildings = getattr(server, "_building_entities", {}) or {}
    for target in definition["targets"].values():
        expected_type = _TARGET_ENTITY_TYPES[target["token"]]
        tx, ty, tz = target["position"]
        matches = []
        for oid, building in buildings.items():
            if int(building.entity_type) != expected_type or int(building.team_id) != target["team"]:
                continue
            distance = math.sqrt((building.x - tx) ** 2 + (building.y - ty) ** 2 + (building.z - tz) ** 2)
            if distance <= 8.0:
                matches.append((distance, int(oid)))
        if len(matches) != 1:
            raise ValueError(
                f"tutorial target {target['id']!r} resolved to {len(matches)} buildings; expected exactly one"
            )
        target["runtime_oid"] = min(matches)[1]
        print(f"[TUTORIAL] Target {target['id']!r} -> building oid={target['runtime_oid']}")

    scenario = definition.get("scenario")
    if scenario:
        definition["targets"][scenario["objective_target"]]["tutorial_health"] = float(
            scenario["objective_health"]
        )
    for target in definition["targets"].values():
        health = target.get("tutorial_health")
        if health is None:
            continue
        oid = int(target["runtime_oid"])
        server._building_max_health[oid] = float(health)
        server._building_health[oid] = float(health)
        print(
            f"[TUTORIAL] Target {target['id']!r} oid={oid} "
            f"tutorial health={float(health):.0f}"
        )


def _ensure_network_targets(server: object, ctx: object) -> bool:
    """Replicate selected map-native tutorial landmarks to the OG client.

    Static map buildings participate in server collision, services, and target
    lookup, but the original client does not reliably instantiate their models
    from the map files alone. Tutorial authors can opt specific landmarks into
    the normal network entity-create path so the visible object matches the
    authoritative building used by lesson evidence.
    """
    definition = getattr(server, "tutorial_definition", None)
    if not definition:
        return True
    targets = [target for target in definition["targets"].values() if target["network_visible"]]
    if not targets:
        return True
    sent_oids = getattr(ctx, "tutorial_network_target_oids", None)
    if not isinstance(sent_oids, set):
        sent_oids = set()
        ctx.tutorial_network_target_oids = sent_oids
    buildings = getattr(server, "_building_entities", {}) or {}
    from .build_uplink import send_dynamic_entity_definition

    all_sent = True
    for target in targets:
        oid = target.get("runtime_oid")
        if oid is None or int(oid) in sent_oids:
            continue
        building = buildings.get(int(oid))
        if building is None:
            all_sent = False
            continue
        sent = send_dynamic_entity_definition(
            server,
            ctx,
            entity_id=int(oid),
            entity_type=int(building.entity_type),
            team_id=int(building.team_id),
            pos=(float(building.x), float(building.y), float(building.z)),
            heading=float(getattr(building, "heading", 0.0) or 0.0),
            is_static=True,
        )
        if sent:
            sent_oids.add(int(oid))
        else:
            all_sent = False
    return all_sent


def record_hit(
    server: object,
    attacker: object,
    *,
    target_kind: str,
    target_id: int,
    weapon: str,
    destroyed: bool = False,
    old_health: float | None = None,
    new_health: float | None = None,
    max_health: float | None = None,
) -> None:
    """Record an authoritative damage event for ordered tutorial evidence."""
    if not getattr(server, "tutorial_definition", None):
        return
    decoded = getattr(attacker, "last_decoded_input", {}) or {}
    events = getattr(server, "_tutorial_hit_events", None)
    if events is None:
        server._tutorial_hit_events = events = []
    event = {
        "seq": len(events) + 1,
        "time": time.monotonic(),
        "attacker_client_id": int(attacker.client_id),
        "attacker_epoch": int(getattr(attacker.session, "local_epoch", 0) or 0),
        "target_kind": str(target_kind),
        "target_id": int(target_id),
        "weapon": str(weapon).upper(),
        "destroyed": bool(destroyed),
        "strafe": abs(float(decoded.get("strafe", 0.0) or 0.0)),
    }
    if old_health is not None:
        event["old_health"] = float(old_health)
    if new_health is not None:
        event["new_health"] = float(new_health)
    if max_health is not None:
        event["max_health"] = float(max_health)
    events.append(event)
    if len(events) > 512:
        del events[:-256]
    definition = getattr(server, "tutorial_definition", None) or {}
    scenario = definition.get("scenario")
    if not scenario or not scenario.get("hit_feedback") or str(target_kind) != "building":
        return
    target = next(
        (
            item for item in definition.get("targets", {}).values()
            if item.get("runtime_oid") == int(target_id)
        ),
        None,
    )
    if target is None:
        return
    if new_health is not None and max_health is not None and float(max_health) > 0.0:
        message = (
            f"HIT: {target['label']} has {max(0.0, float(new_health)):.0f}/"
            f"{float(max_health):.0f} HP remaining."
        )
    else:
        message = (
            f"HIT confirmed on {target['label']} with "
            f"{str(weapon).replace('_', ' ')}."
        )
    _send(server, attacker, message)


def record_weapon_miss(server: object, ctx: object, *, weapon: str, reason: str) -> bool:
    """Give bounded, scenario-only feedback when an ordinary shot misses."""
    definition = getattr(server, "tutorial_definition", None) or {}
    scenario = definition.get("scenario")
    state = getattr(ctx, "tutorial_state", None)
    if (
        not scenario
        or not scenario.get("hit_feedback")
        or not isinstance(state, dict)
        or state.get("complete")
    ):
        return False
    now = time.monotonic()
    previous = float(getattr(ctx, "tutorial_last_miss_feedback", 0.0) or 0.0)
    if now - previous < 2.0:
        return False
    ctx.tutorial_last_miss_feedback = now
    if str(weapon).upper() == "CHAIN GUN":
        message = (
            "MISS: Chain Gun hit no structure. Get within 120 units and center "
            "the target in the reticle."
        )
    else:
        weapon_name = str(weapon).replace("_", " ").title()
        message = (
            f"MISS: {weapon_name} struck {reason}. Raise the reticle onto the "
            "target and fire again."
        )
    return _send(server, ctx, message)


def record_service(server: object, ctx: object, *, service: str, target_id: int, delta: float) -> None:
    """Record health/energy actually supplied by a specific friendly building."""
    if not getattr(server, "tutorial_definition", None) or float(delta) <= 0.0:
        return
    events = getattr(server, "_tutorial_service_events", None)
    if events is None:
        server._tutorial_service_events = events = []
    events.append({
        "seq": len(events) + 1,
        "time": time.monotonic(),
        "client_id": int(ctx.client_id),
        "epoch": int(getattr(ctx.session, "local_epoch", 0) or 0),
        "service": str(service).lower(),
        "target_id": int(target_id),
        "delta": float(delta),
    })
    if len(events) > 512:
        del events[:-256]


def record_logistics_event(
    server: object,
    ctx: object,
    *,
    action: str,
    oid: int,
    entity_type: int,
    position: Any,
    fixture_binding: str = "",
) -> None:
    """Record an authoritative cargo/build lifecycle action by one learner."""
    if not getattr(server, "tutorial_definition", None):
        return
    events = getattr(server, "_tutorial_logistics_events", None)
    if events is None:
        server._tutorial_logistics_events = events = []
    try:
        pos = tuple(float(value) for value in position)
    except (TypeError, ValueError):
        pos = (0.0, 0.0, 0.0)
    events.append({
        "seq": len(events) + 1,
        "time": time.monotonic(),
        "client_id": int(ctx.client_id),
        "epoch": int(getattr(ctx.session, "local_epoch", 0) or 0),
        "team": int(getattr(ctx.session, "team_id", 0) or 0),
        "action": str(action),
        "oid": int(oid),
        "entity_type": int(entity_type),
        "position": pos,
        "fixture_binding": str(fixture_binding or ""),
    })
    if len(events) > 512:
        del events[:-256]


def is_protected_target(server: object, oid: int) -> bool:
    definition = getattr(server, "tutorial_definition", None)
    return bool(definition and any(
        target["protected"] and target["runtime_oid"] == int(oid)
        for target in definition["targets"].values()
    ))


def suppress_turret_fire(server: object, ctx: object, turret_oid: int) -> bool:
    """Return true only for an opt-in tutorial turret after its scored contact."""
    definition = getattr(server, "tutorial_definition", None)
    state = getattr(ctx, "tutorial_state", None)
    if not definition or not isinstance(state, dict):
        return False
    scenario = definition.get("scenario")
    if scenario:
        threat_id = scenario.get("threat_target")
        threat = definition["targets"].get(threat_id, {}) if threat_id else {}
        if threat.get("runtime_oid") == int(turret_oid):
            if scenario.get("quiet_after_result") and (
                state.get("complete")
                or getattr(ctx, "tutorial_last_scenario_result", None) in {"win", "loss"}
            ):
                return True
            activation_index = scenario.get("threat_activation_index")
            if (
                activation_index is not None
                and int(state.get("index", 0) or 0) <= int(activation_index)
            ):
                return True
    completed_count = int(state.get("index", 0) or 0)
    if state.get("complete"):
        completed_count = len(definition["lessons"])
    for lesson in definition["lessons"][:completed_count]:
        trigger = lesson["trigger"]
        if trigger.get("type") != "turret-damage" or not trigger.get("cease_fire_on_complete"):
            continue
        target = definition["targets"].get(trigger["target"], {})
        if target.get("runtime_oid") == int(turret_oid):
            return True
    return False


def suppress_passive_energy(server: object, ctx: object) -> bool:
    """Keep the service fixture recoverable until an attributed fuel visit occurs."""
    definition = getattr(server, "tutorial_definition", None)
    state = getattr(ctx, "tutorial_state", None)
    if not definition or not isinstance(state, dict) or state.get("complete"):
        return False
    index = int(state.get("index", 0) or 0)
    if index < 0 or index >= len(definition["lessons"]):
        return False
    trigger = definition["lessons"][index]["trigger"]
    return bool(state.get("evidence", {}).get("fixture_applied")) and (
        trigger["type"] == "service-recovery"
        or (trigger["type"] == "deployed-service" and trigger["service"] in {"fuel", "energy"})
    )


def _send(server: object, ctx: object, message: str) -> bool:
    packet = build_chat_message(f"[TRAINING] {message}", source_id=0)
    return bool(server._send_packet_to_client(ctx, packet, prefer_tcp=True, allow_udp_fallback=False))


def _new_evidence(ctx: object, now: float, lesson: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    pos = tuple(float(value) for value in ctx.player_pos)
    initially_inside = False
    if lesson:
        trigger = lesson["trigger"]
        if trigger["type"] in {"enter-region", "stop-in-region", "cargo-carry-to-region"}:
            initially_inside = _inside(pos, trigger["region"])
    return {
        "activated_at": now,
        "last_check_at": now,
        "start_pos": pos,
        "start_heading": float(getattr(ctx, "player_heading", 0.0) or 0.0),
        "inside_since": None,
        # Required-entry lessons demand an outside -> inside transition after
        # activation.  Initial occupancy is deliberately not an entry event.
        "was_inside": initially_inside,
        "entered": False,
        "armed": False,
        "strafe_seconds": 0.0,
        "max_forward_input": 0.0,
        "max_heading_change": 0.0,
        "max_lateral_distance": 0.0,
        "failure_notified": False,
        "confirmed": False,
        "hit_cursor": 0,
        "service_cursor": 0,
        "health_restored": 0.0,
        "energy_restored": 0.0,
        "fixture_applied": False,
        "logistics_cursor": 0,
        "moved_away": False,
    }


def _activate_lesson(ctx: object, state: dict[str, Any], now: float, lesson: dict[str, Any]) -> None:
    state["evidence"] = _new_evidence(ctx, now, lesson)
    state["prompt_sent"] = False
    state["prompt_attempts"] = 0
    state["prompt_retry_at"] = now


def _send_prompt(server: object, ctx: object, state: dict[str, Any], lesson: dict[str, Any], message: str, now: float) -> bool:
    if int(state.get("prompt_attempts", 0) or 0) >= 5:
        return False
    if now < float(state.get("prompt_retry_at", 0.0) or 0.0):
        return False
    sent = _send(server, ctx, message)
    state["prompt_attempts"] = int(state.get("prompt_attempts", 0) or 0) + 1
    state["prompt_retry_at"] = now + min(5.0, float(state["prompt_attempts"]))
    if not sent:
        print(f"[TUTORIAL] c{ctx.client_id} prompt send failed for {lesson['id']} (attempt {state['prompt_attempts']})")
        return False
    state["prompt_sent"] = True
    server._tutorial_events.append({"event": "prompt", "client_id": ctx.client_id, "lesson": lesson["id"]})
    trigger = lesson["trigger"]
    if trigger["type"] in {"service-recovery", "deployed-service"} and not state["evidence"].get("fixture_applied"):
        evidence = state["evidence"]
        evidence["service_cursor"] = len(getattr(server, "_tutorial_service_events", []))
        evidence["service_cursor_initialized"] = True
        evidence["pre_fixture_health"] = float(getattr(ctx, "player_health", 1.0) or 0.0)
        evidence["pre_fixture_energy"] = float(getattr(ctx, "player_energy", 0.0) or 0.0)
        if trigger["type"] == "service-recovery" or trigger["service"] == "repair":
            ctx.player_health = min(evidence["pre_fixture_health"], trigger["fixture_health"])
        if trigger["type"] == "service-recovery" or trigger["service"] in {"fuel", "energy"}:
            ctx.player_energy = min(evidence["pre_fixture_energy"], trigger["fixture_energy"])
        evidence["fixture_applied"] = True
        server._tutorial_events.append({
            "event": "fixture", "client_id": ctx.client_id, "lesson": lesson["id"],
            "health": ctx.player_health, "energy": ctx.player_energy,
        })
    if trigger["type"] == "cargo-pickup":
        evidence = state["evidence"]
        evidence["logistics_cursor"] = len(getattr(server, "_tutorial_logistics_events", []))
        evidence["logistics_cursor_initialized"] = True
        binding = trigger["binding"]
        if binding not in state["bindings"]:
            native_map_fixture = bool(trigger.get("native_map_fixture", False))
            if native_map_fixture:
                oid = int(server._dropped_cargo_next_oid)
                server._dropped_cargo_next_oid = oid + 1
                server._dropped_cargo[oid] = {
                    "pos": tuple(float(v) for v in trigger["seed_position"]),
                    "cargo_type": int(trigger["cargo_type"]),
                    "team_id": int(trigger["team"]),
                    "tutorial_native_map": True,
                }
            else:
                oid = int(server._drop_cargo_crate(
                    trigger["seed_position"], trigger["cargo_type"], trigger["team"],
                ) or 0)
            if oid <= 0:
                state["prompt_sent"] = False
                print(f"[TUTORIAL] c{ctx.client_id} failed to seed cargo for {lesson['id']}")
                return False
            crate = server._dropped_cargo.get(oid)
            if crate is not None:
                crate.update({
                    "tutorial_map": server.tutorial_definition["map"],
                    "tutorial_owner_client_id": int(ctx.client_id),
                    "tutorial_owner_epoch": int(getattr(ctx.session, "local_epoch", 0) or 0),
                    "tutorial_binding": binding,
                    "tutorial_native_map": native_map_fixture,
                    "tutorial_repickup_cooldown_s": trigger["repickup_cooldown_seconds"],
                })
            state["bindings"][binding] = {
                "oid": oid,
                "position": tuple(trigger["seed_position"]),
                "entity_type": trigger["cargo_type"],
                "kind": "cargo",
            }
            server._tutorial_events.append({
                "event": "cargo_fixture", "client_id": ctx.client_id,
                "lesson": lesson["id"], "binding": binding, "oid": oid,
            })
    return True


def _cleanup_fixture(ctx: object, state: Any) -> None:
    if not isinstance(state, dict):
        return
    evidence = state.get("evidence", {})
    if not evidence.get("fixture_applied"):
        return
    ctx.player_health = float(evidence.get("pre_fixture_health", ctx.player_health))
    ctx.player_energy = float(evidence.get("pre_fixture_energy", ctx.player_energy))


def _cleanup_logistics_fixture(server: object, ctx: object, state: Any) -> None:
    """Remove this learner's tutorial crates/carry state without touching normal cargo."""
    client_id = int(getattr(ctx, "client_id", 0) or 0)
    tutorial_map = str((getattr(server, "tutorial_definition", None) or {}).get("map", ""))
    for oid, crate in list((getattr(server, "_dropped_cargo", None) or {}).items()):
        if (
            str(crate.get("tutorial_map", "")) == tutorial_map
            and int(crate.get("tutorial_owner_client_id", 0) or 0) == client_id
        ):
            if (
                not crate.get("tutorial_native_map")
                and hasattr(server, "_broadcast_building_delete")
            ):
                server._broadcast_building_delete(int(oid), prefer_tcp=False)
            server._dropped_cargo.pop(oid, None)
    marker = getattr(ctx, "tutorial_fixture_cargo", None)
    if isinstance(marker, dict) and str(marker.get("tutorial_map", "")) == tutorial_map:
        if int(getattr(ctx, "cargo_type", 0) or 0) != 0 and hasattr(server, "_set_player_carry"):
            server._set_player_carry(
                ctx,
                cargo_type=0,
                cargo_count=0,
                has_uplink=bool(getattr(ctx, "has_uplink", False)),
            )
        ctx.tutorial_fixture_cargo = None
    for oid, source in list((getattr(server, "_dynamic_building_sources", None) or {}).items()):
        fixture = source.get("tutorial_fixture") if isinstance(source, dict) else None
        if not isinstance(fixture, dict):
            continue
        if (
            str(fixture.get("tutorial_map", "")) != tutorial_map
            or int(fixture.get("tutorial_owner_client_id", 0) or 0) != client_id
        ):
            continue
        if hasattr(server, "_broadcast_building_delete"):
            server._broadcast_building_delete(int(oid), prefer_tcp=False)
        if hasattr(server, "_remove_dynamic_building_record"):
            server._remove_dynamic_building_record(int(oid))


def _restore_scenario_objective(server: object) -> bool:
    """Restore scenario-owned target health only at reset/spawn boundaries."""
    definition = getattr(server, "tutorial_definition", None) or {}
    if not definition.get("scenario"):
        return False
    buildings = getattr(server, "_building_entities", {}) or {}
    snapshot = getattr(server, "_snapshot_in_game_clients", lambda: ())
    restored = False
    for target in definition.get("targets", {}).values():
        health = target.get("tutorial_health")
        oid = target.get("runtime_oid")
        if health is None or oid is None or int(oid) not in buildings:
            continue
        oid = int(oid)
        health = float(health)
        old_health = float((getattr(server, "_building_health", {}) or {}).get(oid, 0.0) or 0.0)
        old_max = float((getattr(server, "_building_max_health", {}) or {}).get(oid, 0.0) or 0.0)
        if old_health == health and old_max == health:
            continue
        if old_health > 0.0 and hasattr(server, "_broadcast_building_delete"):
            server._broadcast_building_delete(oid, prefer_tcp=True)
        server._building_max_health[oid] = health
        server._building_health[oid] = health
        for client in snapshot():
            getattr(client, "known_entity_ids", set()).discard(oid)
            getattr(client, "_entity_create_times", {}).pop(oid, None)
            getattr(client, "tutorial_network_target_oids", set()).discard(oid)
        print(f"[TUTORIAL] Restored target {target['id']!r} oid={oid} to {health:.0f} HP")
        restored = True
    return restored


def record_player_death(server: object, ctx: object) -> bool:
    """Record the honest free-skirmish loss before normal death cleanup."""
    definition = getattr(server, "tutorial_definition", None) or {}
    scenario = definition.get("scenario")
    state = getattr(ctx, "tutorial_state", None)
    if (
        not scenario
        or not scenario["loss_on_player_death"]
        or not isinstance(state, dict)
        or state.get("complete")
    ):
        return False
    ctx.tutorial_last_scenario_result = "loss"
    server._tutorial_events.append({
        "event": "scenario_loss",
        "client_id": int(ctx.client_id),
        "spawn_epoch": int(getattr(ctx.session, "local_epoch", 0) or 0),
        "reason": "player_death",
    })
    _send(server, ctx, scenario["loss_message"])
    print(f"[TUTORIAL] c{ctx.client_id} lost free-skirmish scenario: player death")
    return True


def reset_player(server: object, ctx: object, *, announce: bool = True) -> dict[str, Any]:
    now = time.monotonic()
    _cleanup_fixture(ctx, getattr(ctx, "tutorial_state", None))
    _cleanup_logistics_fixture(server, ctx, getattr(ctx, "tutorial_state", None))
    _restore_scenario_objective(server)
    definition = getattr(server, "tutorial_definition", None)
    first_lesson = definition["lessons"][0] if definition else None
    ctx.tutorial_state = {
        "index": 0,
        "prompt_sent": False,
        "started_at": now,
        "ready_at": now + 1.25,
        "spawn_epoch": int(getattr(ctx.session, "local_epoch", 0) or 0),
        "team_id": int(getattr(ctx.session, "team_id", 0) or 0),
        "complete": False,
        "prompt_attempts": 0,
        "prompt_retry_at": now + 1.25,
        "bindings": {},
        "evidence": _new_evidence(ctx, now, first_lesson),
    }
    ctx.tutorial_last_scenario_result = None
    if announce:
        _send(server, ctx, "Course reset. The first instruction will appear shortly.")
    return ctx.tutorial_state


def cleanup_player(server: object, ctx: object) -> None:
    """Remove one departing learner's tutorial-only fixtures and evidence.

    Normal cargo and normal dynamic buildings carry no tutorial ownership metadata
    and are deliberately left alone.
    """
    state = getattr(ctx, "tutorial_state", None)
    _cleanup_fixture(ctx, state)
    _cleanup_logistics_fixture(server, ctx, state)
    ctx.tutorial_state = None


def status_for_player(server: object, ctx: object) -> str:
    definition = getattr(server, "tutorial_definition", None)
    if not definition:
        return "No tutorial is active on this map."
    state = getattr(ctx, "tutorial_state", None)
    if not isinstance(state, dict):
        return f"{definition['title']}: waiting to start."
    if state.get("complete"):
        return f"{definition['title']}: complete ({len(definition['lessons'])}/{len(definition['lessons'])})."
    index = int(state.get("index", 0))
    lesson = definition["lessons"][min(index, len(definition["lessons"]) - 1)]
    return f"{definition['title']}: step {index + 1}/{len(definition['lessons'])} - {lesson['prompt']}"


def hint_for_player(server: object, ctx: object) -> str:
    definition = getattr(server, "tutorial_definition", None)
    state = getattr(ctx, "tutorial_state", None)
    if not definition or not isinstance(state, dict) or state.get("complete"):
        return status_for_player(server, ctx)
    lesson = definition["lessons"][int(state.get("index", 0))]
    return f"Hint: {lesson['hint']}"


def retry_player(server: object, ctx: object) -> str:
    definition = getattr(server, "tutorial_definition", None)
    state = getattr(ctx, "tutorial_state", None)
    if not definition or not isinstance(state, dict) or state.get("complete"):
        return status_for_player(server, ctx)
    now = time.monotonic()
    lesson = definition["lessons"][int(state.get("index", 0))]
    if definition.get("scenario"):
        reset_player(server, ctx, announce=False)
        return "Scenario reset. The briefing and objective will appear shortly."
    if lesson["trigger"]["type"].startswith("cargo-"):
        reset_player(server, ctx, announce=False)
        return "Cargo exercise reset to step 1. Return to the start and use the newly supplied crate."
    _cleanup_fixture(ctx, state)
    _activate_lesson(ctx, state, now, lesson)
    state["ready_at"] = now + 0.5
    return f"Retrying step {int(state['index']) + 1}. Return to its starting bay. {lesson['hint']}"


def confirm_player(server: object, ctx: object) -> str:
    """Accept an explicit learner acknowledgement only for the active confirm step."""
    definition = getattr(server, "tutorial_definition", None)
    state = getattr(ctx, "tutorial_state", None)
    if not definition or not isinstance(state, dict) or state.get("complete"):
        return status_for_player(server, ctx)
    lesson = definition["lessons"][int(state.get("index", 0))]
    trigger = lesson["trigger"]
    if trigger["type"] != "self-confirm":
        return "This step requires an observed game action, not confirmation."
    if not state.get("prompt_sent"):
        return "Wait for the current training instruction before confirming."
    if not _inside(tuple(float(value) for value in ctx.player_pos), trigger["region"]):
        return "Move into the marked firing line before confirming."
    state["evidence"]["confirmed"] = True
    return "Orientation confirmed."


def _evaluate_trigger(
    server: object,
    ctx: object,
    lesson: dict[str, Any],
    evidence: dict[str, Any],
    state: dict[str, Any],
    now: float,
) -> tuple[bool, str]:
    trigger = lesson["trigger"]
    trigger_type = trigger["type"]
    pos = tuple(float(value) for value in ctx.player_pos)
    dt = max(0.0, min(0.25, now - float(evidence.get("last_check_at", now))))
    evidence["last_check_at"] = now
    if trigger_type == "self-confirm":
        return bool(evidence.get("confirmed")) and _inside(pos, trigger["region"]), ""

    if trigger_type == "hit-target":
        target = server.tutorial_definition["targets"][trigger["target"]]
        events = getattr(server, "_tutorial_hit_events", [])
        cursor = int(evidence.get("hit_cursor", 0) or 0)
        evidence["hit_cursor"] = len(events)
        epoch = int(getattr(ctx.session, "local_epoch", 0) or 0)
        for event in events[cursor:]:
            if (
                event["attacker_client_id"] == int(ctx.client_id)
                and event["attacker_epoch"] == epoch
                and event["target_kind"] == "building"
                and event["target_id"] == target["runtime_oid"]
                and event["weapon"] == trigger["weapon"]
                and event["strafe"] >= trigger["min_strafe_input"]
            ):
                return True, ""
        return False, ""

    if trigger_type == "destroy-target":
        target = server.tutorial_definition["targets"][trigger["target"]]
        events = getattr(server, "_tutorial_hit_events", [])
        cursor = int(evidence.get("hit_cursor", 0) or 0)
        evidence["hit_cursor"] = len(events)
        epoch = int(getattr(ctx.session, "local_epoch", 0) or 0)
        for event in events[cursor:]:
            if (
                event["attacker_client_id"] == int(ctx.client_id)
                and event["attacker_epoch"] == epoch
                and event["target_kind"] == "building"
                and event["target_id"] == target["runtime_oid"]
                and event.get("destroyed", False)
            ):
                return True, ""
        return False, ""

    if trigger_type == "turret-damage":
        target = server.tutorial_definition["targets"][trigger["target"]]
        target_oid = target.get("runtime_oid")
        if target_oid is None:
            return False, ""
        damage_time = float(getattr(ctx, "last_damage_time", 0.0) or 0.0)
        damage_source = str(getattr(ctx, "last_damage_source", "") or "")
        damage_amount = float(getattr(ctx, "last_damage_amount", 0.0) or 0.0)
        return (
            damage_time >= float(evidence.get("activated_at", now))
            and f"oid={int(target_oid)}" in damage_source
            and damage_source.startswith("turret:")
            and damage_amount >= trigger["min_damage"]
            and float(getattr(ctx, "player_health", 0.0) or 0.0) > 0.0
        ), ""

    if trigger_type == "service-recovery":
        repair_oid = server.tutorial_definition["targets"][trigger["repair_target"]]["runtime_oid"]
        fuel_oid = server.tutorial_definition["targets"][trigger["fuel_target"]]["runtime_oid"]
        events = getattr(server, "_tutorial_service_events", [])
        cursor = int(evidence.get("service_cursor", 0) or 0)
        evidence["service_cursor"] = len(events)
        epoch = int(getattr(ctx.session, "local_epoch", 0) or 0)
        for event in events[cursor:]:
            if event["client_id"] != int(ctx.client_id) or event["epoch"] != epoch:
                continue
            if event["service"] == "repair" and event["target_id"] == repair_oid:
                evidence["health_restored"] += event["delta"]
            elif event["service"] == "fuel" and event["target_id"] == fuel_oid:
                evidence["energy_restored"] += event["delta"]
        return (
            evidence["health_restored"] >= trigger["min_health_restored"]
            and evidence["energy_restored"] >= trigger["min_energy_restored"]
        ), ""

    if trigger_type == "deployed-service":
        binding_info = state.get("bindings", {}).get(trigger["binding"])
        if not isinstance(binding_info, dict) or binding_info.get("kind") != "building":
            return False, "The tracked deployed building is missing. Retry this step."
        events = getattr(server, "_tutorial_service_events", [])
        cursor = int(evidence.get("service_cursor", 0) or 0)
        evidence["service_cursor"] = len(events)
        epoch = int(getattr(ctx.session, "local_epoch", 0) or 0)
        for event in events[cursor:]:
            if (
                event["client_id"] == int(ctx.client_id)
                and event["epoch"] == epoch
                and event["service"] == trigger["service"]
                and event["target_id"] == binding_info["oid"]
            ):
                if trigger["service"] == "repair":
                    evidence["health_restored"] += event["delta"]
                else:
                    evidence["energy_restored"] += event["delta"]
        restored = (
            evidence["health_restored"] if trigger["service"] == "repair"
            else evidence["energy_restored"]
        )
        return restored >= trigger["min_restored"], ""

    if trigger_type == "cargo-carry-to-region":
        marker = getattr(ctx, "tutorial_fixture_cargo", None)
        inside = _inside(pos, trigger["region"])
        if inside and not evidence["was_inside"]:
            evidence["entered"] = True
        evidence["was_inside"] = inside
        owns_expected = (
            isinstance(marker, dict)
            and marker.get("tutorial_binding") == trigger["binding"]
            and int(getattr(ctx, "cargo_type", 0) or 0) == trigger["cargo_type"]
            and int(getattr(ctx, "cargo_count", 0) or 0) > 0
        )
        return (
            owns_expected and inside and (evidence["entered"] or not trigger["require_entry"])
        ), ""

    if trigger_type in {"cargo-pickup", "cargo-drop", "cargo-recover", "cargo-deploy"}:
        events = getattr(server, "_tutorial_logistics_events", [])
        cursor = int(evidence.get("logistics_cursor", 0) or 0)
        evidence["logistics_cursor"] = len(events)
        epoch = int(getattr(ctx.session, "local_epoch", 0) or 0)
        binding_info = state.get("bindings", {}).get(trigger["binding"])
        if trigger_type == "cargo-recover" and isinstance(binding_info, dict):
            bx, by, _ = binding_info["position"]
            if math.hypot(pos[0] - bx, pos[1] - by) >= trigger["min_away_distance"]:
                evidence["moved_away"] = True
        for event in events[cursor:]:
            if (
                event["client_id"] != int(ctx.client_id)
                or event["epoch"] != epoch
                or event["entity_type"] != trigger["cargo_type"]
            ):
                continue
            if trigger_type == "cargo-pickup":
                if (
                    isinstance(binding_info, dict)
                    and event["action"] == "cargo_pickup"
                    and event["oid"] == binding_info["oid"]
                    and event["fixture_binding"] == trigger["binding"]
                ):
                    return True, ""
            elif trigger_type == "cargo-drop":
                if (
                    event["action"] == "cargo_drop"
                    and event["fixture_binding"] == trigger["binding"]
                    and _inside(event["position"], trigger["region"])
                ):
                    state["bindings"][trigger["bind_as"]] = {
                        "oid": event["oid"], "position": event["position"],
                        "entity_type": event["entity_type"], "kind": "cargo",
                    }
                    return True, ""
            elif trigger_type == "cargo-deploy":
                if (
                    event["action"] == "cargo_deploy"
                    and event["fixture_binding"] == trigger["binding"]
                    and _inside(event["position"], trigger["region"])
                ):
                    state["bindings"][trigger["bind_as"]] = {
                        "oid": event["oid"], "position": event["position"],
                        "entity_type": event["entity_type"], "kind": "building",
                    }
                    return True, ""
            elif (
                evidence["moved_away"]
                and isinstance(binding_info, dict)
                and event["action"] == "cargo_pickup"
                and event["oid"] == binding_info["oid"]
            ):
                return True, ""
        return False, ""

    if trigger_type in {"enter-region", "stop-in-region"}:
        inside = _inside(pos, trigger["region"])
        if inside and not evidence["was_inside"]:
            evidence["entered"] = True
        evidence["was_inside"] = inside
        if trigger_type == "enter-region":
            return inside and (evidence["entered"] or not trigger["require_entry"]), ""
        speed = math.hypot(float(ctx.player_vel[0]), float(ctx.player_vel[1]))
        if inside and speed <= trigger["max_horizontal_speed"] and (
            evidence["entered"] or not trigger["require_entry"]
        ):
            if evidence["inside_since"] is None:
                evidence["inside_since"] = now
            dwell = now - float(evidence["inside_since"])
            return dwell >= trigger["dwell_seconds"], ""
        evidence["inside_since"] = None
        return False, ""

    if not evidence["armed"]:
        if not _inside(pos, trigger["start_region"]):
            return False, "Return to the starting bay before retrying the strafe."
        evidence["armed"] = True
        evidence["start_pos"] = pos
        evidence["start_heading"] = float(getattr(ctx, "player_heading", 0.0) or 0.0)
        evidence["last_check_at"] = now
        return False, ""

    decoded = getattr(ctx, "last_decoded_input", {}) or {}
    strafe = abs(float(decoded.get("strafe", 0.0) or 0.0))
    forward = abs(float(decoded.get("fwd", 0.0) or 0.0))
    if strafe >= trigger["min_strafe_input"]:
        evidence["strafe_seconds"] += dt
    evidence["max_forward_input"] = max(evidence["max_forward_input"], forward)
    heading = float(getattr(ctx, "player_heading", 0.0) or 0.0)
    evidence["max_heading_change"] = max(
        evidence["max_heading_change"], _angle_delta(heading, evidence["start_heading"])
    )
    dx = pos[0] - evidence["start_pos"][0]
    dy = pos[1] - evidence["start_pos"][1]
    lateral = abs(-math.sin(evidence["start_heading"]) * dx + math.cos(evidence["start_heading"]) * dy)
    evidence["max_lateral_distance"] = max(evidence["max_lateral_distance"], lateral)
    if not _inside(pos, trigger["target_region"]):
        return False, ""
    if evidence["max_heading_change"] > trigger["max_heading_change"]:
        return False, "Heading changed too much. Return to the first bay and use only the strafe action."
    if evidence["max_forward_input"] > trigger["max_forward_input"]:
        return False, "Forward/reverse input was detected. Return to the first bay and use only strafe."
    if evidence["strafe_seconds"] < trigger["min_strafe_seconds"]:
        return False, "Not enough strafe input was observed. Return to the first bay and retry."
    if evidence["max_lateral_distance"] < trigger["min_lateral_distance"]:
        return False, "The lateral movement was too short. Return to the first bay and retry."
    return True, ""


def update_player(server: object, ctx: object, *, now: Optional[float] = None) -> None:
    definition = getattr(server, "tutorial_definition", None)
    if not definition or not getattr(ctx, "session", None) or not ctx.session.in_game:
        return
    limit = int(definition.get("max_concurrent_learners", 0) or 0)
    if limit > 0 and hasattr(server, "_snapshot_in_game_clients"):
        eligible = sorted([
            candidate for candidate in server._snapshot_in_game_clients()
            if getattr(candidate, "session", None)
            and int(candidate.session.team_id or 0) in definition["teams"]
        ], key=lambda candidate: int(candidate.client_id))
        active_ids = {int(candidate.client_id) for candidate in eligible}
        allowed_ids = {int(candidate.client_id) for candidate in eligible[:limit]}
        # Remove abandoned tutorial crates when their learner disconnects. Normal
        # cargo has no tutorial_map tag and is never touched here.
        for oid, crate in list((getattr(server, "_dropped_cargo", None) or {}).items()):
            if (
                str(crate.get("tutorial_map", "")) == definition["map"]
                and int(crate.get("tutorial_owner_client_id", 0) or 0) not in active_ids
            ):
                if not crate.get("tutorial_native_map"):
                    server._broadcast_building_delete(int(oid), prefer_tcp=False)
                server._dropped_cargo.pop(oid, None)
        if int(ctx.client_id) not in allowed_ids:
            if isinstance(getattr(ctx, "tutorial_state", None), dict):
                _cleanup_fixture(ctx, ctx.tutorial_state)
                _cleanup_logistics_fixture(server, ctx, ctx.tutorial_state)
                ctx.tutorial_state = None
            if not getattr(ctx, "tutorial_waiting_notified", False):
                if _send(server, ctx, f"This course supports {limit} active learner at a time. Please wait."):
                    ctx.tutorial_waiting_notified = True
            return
        ctx.tutorial_waiting_notified = False
    if int(ctx.session.team_id or 0) not in definition["teams"]:
        # Do not retain partial evidence while the player is on an ineligible
        # team; returning later starts the course cleanly.
        if hasattr(ctx, "tutorial_state"):
            ctx.tutorial_state = None
        return
    # Make map-native instructional landmarks visible before sending the first
    # prompt. A failed/too-early UDP create is retried on the next tutorial tick.
    if not _ensure_network_targets(server, ctx):
        return
    now = time.monotonic() if now is None else now
    state = getattr(ctx, "tutorial_state", None)
    epoch = int(getattr(ctx.session, "local_epoch", 0) or 0)
    team_id = int(ctx.session.team_id or 0)
    if (
        not isinstance(state, dict)
        or state.get("spawn_epoch") != epoch
        or state.get("team_id") != team_id
    ):
        state = reset_player(server, ctx, announce=False)
    if state.get("complete") or now < float(state["ready_at"]):
        return
    lessons = definition["lessons"]
    index = int(state["index"])
    lesson = lessons[index]
    if not state.get("prompt_sent"):
        if not _send_prompt(
            server, ctx, state, lesson,
            f"Step {index + 1}/{len(lessons)}: {lesson['prompt']}", now,
        ):
            return

    # Ignore combat evidence produced before this lesson became active.
    if lesson["trigger"]["type"] in {"hit-target", "destroy-target"} and not state["evidence"].get("hit_cursor_initialized"):
        state["evidence"]["hit_cursor"] = len(getattr(server, "_tutorial_hit_events", []))
        state["evidence"]["hit_cursor_initialized"] = True
    if lesson["trigger"]["type"] == "service-recovery" and not state["evidence"].get("service_cursor_initialized"):
        state["evidence"]["service_cursor"] = len(getattr(server, "_tutorial_service_events", []))
        state["evidence"]["service_cursor_initialized"] = True
    if lesson["trigger"]["type"] in {"cargo-drop", "cargo-recover", "cargo-deploy"} and not state["evidence"].get("logistics_cursor_initialized"):
        state["evidence"]["logistics_cursor"] = len(getattr(server, "_tutorial_logistics_events", []))
        state["evidence"]["logistics_cursor_initialized"] = True
    complete, failure = _evaluate_trigger(server, ctx, lesson, state["evidence"], state, now)
    if not complete:
        if failure and not state["evidence"].get("failure_notified"):
            if _send(server, ctx, failure):
                state["evidence"]["failure_notified"] = True
        return
    if not _send(server, ctx, lesson["complete_message"]):
        return
    pos = tuple(float(value) for value in ctx.player_pos)
    server._tutorial_events.append({"event": "complete", "client_id": ctx.client_id, "lesson": lesson["id"], "position": pos})
    print(f"[TUTORIAL] c{ctx.client_id} completed {lesson['id']} at {pos}")
    index += 1
    state["index"] = index
    state["prompt_sent"] = False
    if index >= len(lessons):
        state["complete"] = True
        if definition.get("scenario"):
            ctx.tutorial_last_scenario_result = "win"
            server._tutorial_events.append({
                "event": "scenario_win",
                "client_id": int(ctx.client_id),
                "spawn_epoch": int(getattr(ctx.session, "local_epoch", 0) or 0),
                "reason": "objective_destroyed",
            })
        _send(server, ctx, f"{definition['title']} complete. Type 'tutorial reset' to replay.")
    else:
        _activate_lesson(ctx, state, now, lessons[index])
        state["ready_at"] = now + 1.25
