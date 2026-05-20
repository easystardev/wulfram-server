#!/usr/bin/env python3
"""
Server manager - handles process lifecycle, logging, and cleanup.
Ensures unbuffered output and clean process management.
"""

import os
import sys
import subprocess
import time
import argparse
import socket
import ctypes
import msvcrt
from pathlib import Path
from ctypes import wintypes

SERVER_SCRIPT = Path(__file__).parent / "run_server.py"
PID_FILE = Path(__file__).parent / "server.pid"
LOG_FILE = Path(__file__).parent.parent / "server.log"
CLIENT_ERROR_LOG = Path(__file__).parent.parent / "slurpysoft-wulfram" / "errorlog.txt"
PORT = 2627


def _windows_pid_exists(pid):
    """Return whether a Windows process exists without relying on tasklist."""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION,
        False,
        int(pid),
    )
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def kill_processes_on_port(port):
    """Kill any processes using the specified port."""
    if sys.platform == 'win32':
        # Find PIDs using the port
        try:
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True, text=True
            )
            pids_to_kill = set()
            for line in result.stdout.splitlines():
                if f':{port}' in line and ('LISTENING' in line or 'UDP' in line):
                    parts = line.split()
                    if parts:
                        try:
                            pid = int(parts[-1])
                            if pid > 0:
                                pids_to_kill.add(pid)
                        except ValueError:
                            pass

            for pid in pids_to_kill:
                print(f"[MANAGER] Killing process {pid} on port {port}")
                subprocess.run(['taskkill', '/F', '/PID', str(pid)],
                             capture_output=True)

            if pids_to_kill:
                time.sleep(1)  # Wait for ports to be released
                return True
        except Exception as e:
            print(f"[MANAGER] Error killing port processes: {e}")
    return False


def is_port_in_use(port):
    """Check if port is in use."""
    for host in ('127.0.0.1', '0.0.0.0'):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
            except OSError:
                return True
    return False


def get_running_pid():
    """Get PID of running server if any."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if process exists
            if sys.platform == 'win32':
                if _windows_pid_exists(pid):
                    return pid
            else:
                os.kill(pid, 0)
                return pid
        except (ValueError, OSError, ProcessLookupError):
            pass
        PID_FILE.unlink(missing_ok=True)
    return None


def stop_server():
    """Stop the running server and clean up zombies."""
    # First try PID file
    pid = get_running_pid()
    if pid:
        print(f"[MANAGER] Stopping server (PID {pid})...")
        try:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True)
            else:
                import signal
                os.kill(pid, signal.SIGTERM)
        except Exception as e:
            print(f"[MANAGER] Error: {e}")
        PID_FILE.unlink(missing_ok=True)

    # Also kill anything on the port (zombies)
    if is_port_in_use(PORT):
        print(f"[MANAGER] Cleaning up zombies on port {PORT}...")
        kill_processes_on_port(PORT)

    # Verify port is free
    time.sleep(0.5)
    if is_port_in_use(PORT):
        print(f"[MANAGER] WARNING: Port {PORT} still in use!")
    else:
        print("[MANAGER] Server stopped, port free")


def start_server(foreground=False):
    """Start the server."""
    # Always clean up first
    if is_port_in_use(PORT):
        print(f"[MANAGER] Port {PORT} in use, cleaning up...")
        kill_processes_on_port(PORT)
        time.sleep(1)

    if is_port_in_use(PORT):
        print(f"[MANAGER] ERROR: Port {PORT} still in use, cannot start")
        return False

    print(f"[MANAGER] Starting server on port {PORT}...")

    if foreground:
        # Run in foreground with unbuffered output
        print("[MANAGER] Running in foreground (Ctrl+C to stop)")
        print("-" * 60)
        try:
            # Use -u for unbuffered output
            proc = subprocess.Popen(
                [sys.executable, '-u', str(SERVER_SCRIPT)],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            PID_FILE.write_text(str(proc.pid))
            proc.wait()
        except KeyboardInterrupt:
            print("\n[MANAGER] Interrupted")
        finally:
            PID_FILE.unlink(missing_ok=True)
            stop_server()  # Clean up
    else:
        # Background mode - write to log file
        print(f"[MANAGER] Log file: {LOG_FILE}")

        # Clear old log
        LOG_FILE.write_text("")

        def _open_shared_log_file(path: Path):
            if sys.platform != 'win32':
                return open(path, 'w', encoding='utf-8', errors='replace', buffering=1)

            FILE_GENERIC_WRITE = 0x40000000
            FILE_SHARE_READ = 0x00000001
            FILE_SHARE_WRITE = 0x00000002
            CREATE_ALWAYS = 2
            FILE_ATTRIBUTE_NORMAL = 0x80
            HANDLE_FLAG_INHERIT = 0x00000001

            handle = ctypes.windll.kernel32.CreateFileW(
                str(path),
                FILE_GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                None,
                CREATE_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                None,
            )
            if handle in (0, wintypes.HANDLE(-1).value):
                raise OSError("Failed to open log file with shared access")

            ctypes.windll.kernel32.SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT)

            fd = msvcrt.open_osfhandle(handle, os.O_WRONLY)
            return os.fdopen(fd, 'w', encoding='utf-8', errors='replace', buffering=1)

        with _open_shared_log_file(LOG_FILE) as log:
            if sys.platform == 'win32':
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                if hasattr(subprocess, "DETACHED_PROCESS"):
                    creationflags |= subprocess.DETACHED_PROCESS
                proc = subprocess.Popen(
                    [sys.executable, '-u', str(SERVER_SCRIPT)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                    close_fds=True
                )
            else:
                proc = subprocess.Popen(
                    [sys.executable, '-u', str(SERVER_SCRIPT)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )

        PID_FILE.write_text(str(proc.pid))

        # Imports and map/collision setup can take more than a second on a
        # cold run, so poll before reporting a failed background start.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if is_port_in_use(PORT):
                print(f"[MANAGER] Server started (PID {proc.pid})")
                return True
            if proc.poll() is not None:
                break
            time.sleep(0.25)

        print(f"[MANAGER] ERROR: Server failed to start")
        tail_log(20)
        return False


def clear_client_logs():
    """Clear the client error log."""
    if CLIENT_ERROR_LOG.exists():
        CLIENT_ERROR_LOG.write_text("")
        print("[MANAGER] Cleared client error log")


def restart_server():
    """Restart the server."""
    stop_server()
    clear_client_logs()
    time.sleep(1)
    start_server()


def status():
    """Show server status."""
    pid = get_running_pid()
    port_used = is_port_in_use(PORT)

    if pid and port_used:
        print(f"[MANAGER] Server running (PID {pid}, port {PORT})")
    elif port_used:
        print(f"[MANAGER] WARNING: Port {PORT} in use but no PID file (zombie?)")
    elif pid:
        print(f"[MANAGER] WARNING: PID file exists ({pid}) but port {PORT} free")
        PID_FILE.unlink(missing_ok=True)
    else:
        print("[MANAGER] Server not running")


def tail_log(lines=50):
    """Show recent log output."""
    if LOG_FILE.exists():
        content = LOG_FILE.read_text()
        if content.strip():
            log_lines = content.splitlines()
            for line in log_lines[-lines:]:
                print(line)
        else:
            print("[MANAGER] Log file is empty")
    else:
        print("[MANAGER] No log file found")


def main():
    parser = argparse.ArgumentParser(description='Wulfram Server Manager')
    parser.add_argument('command', choices=['start', 'stop', 'restart', 'status', 'log', 'fg', 'clean'],
                       help='Command: start, stop, restart, status, log, fg (foreground), clean (kill zombies)')
    parser.add_argument('-n', '--lines', type=int, default=50,
                       help='Number of log lines to show')
    parser.add_argument('--wf', action='store_true',
                       help='Enable wulf-forge compatibility mode (minimal server)')
    args = parser.parse_args()

    # Set wulf-forge mode environment variable if requested
    if args.wf:
        os.environ['WULFRAM_WULFFORGE_MODE'] = '1'
        print("[MANAGER] Wulf-forge compatibility mode requested")

    if args.command == 'start':
        start_server(foreground=False)
    elif args.command == 'stop':
        stop_server()
    elif args.command == 'restart':
        restart_server()
    elif args.command == 'status':
        status()
    elif args.command == 'log':
        tail_log(args.lines)
    elif args.command == 'fg':
        start_server(foreground=True)
    elif args.command == 'clean':
        print("[MANAGER] Cleaning up all server processes...")
        kill_processes_on_port(PORT)
        PID_FILE.unlink(missing_ok=True)
        if not is_port_in_use(PORT):
            print("[MANAGER] Port is free")
        else:
            print("[MANAGER] WARNING: Port still in use")


if __name__ == '__main__':
    main()
