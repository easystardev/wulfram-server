#!/usr/bin/env python3
"""
Quick log checker for debugging - shows server log, latest trace, and client crash info.
Usage: python check_logs.py [lines]
"""

import sys
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SERVER_LOG = PROJECT_ROOT / "server.log"
TRACES_DIR = Path(__file__).parent / "traces"
CLIENT_ERRORLOG = PROJECT_ROOT / "slurpysoft-wulfram" / "errorlog.txt"

def get_latest_trace():
    """Get most recent trace file."""
    if not TRACES_DIR.exists():
        return None
    traces = list(TRACES_DIR.glob("*.log"))
    if not traces:
        return None
    return max(traces, key=lambda p: p.stat().st_mtime)

def tail_file(path, lines=50):
    """Get last N lines of a file."""
    if not path or not path.exists():
        return f"[File not found: {path}]"
    with open(path, 'r', errors='ignore') as f:
        all_lines = f.readlines()
        return ''.join(all_lines[-lines:])

def find_crash_info(path, context_lines=8):
    """Find crash info in client errorlog."""
    if not path.exists():
        return "[Client errorlog not found]"

    with open(path, 'r', errors='ignore') as f:
        lines = f.readlines()

    # Find last "caused an" or "Access Violation"
    for i in range(len(lines) - 1, -1, -1):
        if 'caused an' in lines[i] or 'Access Violation' in lines[i]:
            start = max(0, i - 1)
            end = min(len(lines), i + context_lines)
            return ''.join(lines[start:end])

    return "[No crash found in errorlog]"

def main():
    lines = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    print("=" * 60)
    print("SERVER LOG (last {} lines)".format(lines))
    print("=" * 60)
    print(tail_file(SERVER_LOG, lines))

    print("\n" + "=" * 60)
    print("LATEST TRACE")
    print("=" * 60)
    trace = get_latest_trace()
    if trace:
        print(f"[{trace.name}]")
        print(tail_file(trace, lines))
    else:
        print("[No trace files found]")

    print("\n" + "=" * 60)
    print("CLIENT CRASH INFO")
    print("=" * 60)
    print(find_crash_info(CLIENT_ERRORLOG))

if __name__ == "__main__":
    main()
