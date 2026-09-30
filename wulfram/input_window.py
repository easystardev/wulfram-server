"""Time-area sampling for controller channels 1-3.

The recovered client and Baffler IntegrateInputs average the channels over a
window, then evaluate the controller once. Arrival time is not client time;
the server keeps this integration in shadow mode until phase alignment is measured.
"""
import math

CHANNELS = ("turn", "fwd", "strafe")


def average_controls(history, start, end, fallback):
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("input window must have positive finite duration")
    events = list(history)
    if not events:
        return {k: float(fallback.get(k, 0)) for k in CHANNELS}
    # Stable ordering handles multiple packets at the same timestamp.
    events.sort(key=lambda e: (e["time"], e.get("action_sequence", 0)))
    current = {k: float(events[0].get("previous", fallback).get(k, 0)) for k in CHANNELS}
    area = dict.fromkeys(CHANNELS, 0.0)
    cursor = start
    for event in events:
        when = float(event["time"])
        if when > end:
            break
        boundary = max(cursor, when)
        for key in CHANNELS:
            area[key] += current[key] * (boundary - cursor)
            current[key] = max(-1.0, min(1.0, float(event.get(key, current[key]))))
        cursor = boundary
    return {key: (area[key] + current[key] * (end - cursor)) / (end - start)
            for key in CHANNELS}
