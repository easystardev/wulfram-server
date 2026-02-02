#!/usr/bin/env python3
"""
Packet injection client for Wulfram control plane.

Usage:
  python inject.py                    # Interactive mode
  python inject.py "send CHAT hello"  # Single command
  python inject.py -f script.txt      # Run script file

Control plane runs on port 2628 (game server port + 1).
"""

import socket
import sys
import argparse
import readline  # Enables arrow keys, history in interactive mode


def connect(host: str = "127.0.0.1", port: int = 2628) -> socket.socket:
    """Connect to control plane."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.settimeout(2.0)
    # Read banner until we get the prompt
    banner = ""
    while "> " not in banner:
        chunk = sock.recv(1024).decode()
        if not chunk:
            break
        banner += chunk
    # Print banner without prompt
    if banner.endswith("> "):
        banner = banner[:-2]
    print(banner.strip())
    return sock


def send_command(sock: socket.socket, cmd: str) -> str:
    """Send command and return response."""
    sock.send((cmd + "\n").encode())
    sock.settimeout(3.0)
    response = ""
    while True:
        try:
            chunk = sock.recv(4096).decode()
            if not chunk:
                break
            response += chunk
            if "> " in response:
                # Got prompt, we have complete response
                break
        except socket.timeout:
            break
    # Remove trailing prompt
    if response.endswith("> "):
        response = response[:-2]
    return response.strip()


def interactive(sock: socket.socket):
    """Interactive REPL."""
    print("Type 'help' for commands, Ctrl+C to exit")
    try:
        while True:
            try:
                cmd = input("> ").strip()
                if not cmd:
                    continue
                if cmd in ('quit', 'exit'):
                    break
                response = send_command(sock, cmd)
                if response:
                    print(response)
            except EOFError:
                break
    except KeyboardInterrupt:
        print("\nBye!")


def run_script(sock: socket.socket, filename: str):
    """Run commands from a file."""
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            print(f"> {line}")
            response = send_command(sock, line)
            if response:
                print(response)


def main():
    parser = argparse.ArgumentParser(description="Wulfram packet injector")
    parser.add_argument("command", nargs="*", help="Command to run")
    parser.add_argument("-f", "--file", help="Script file to run")
    parser.add_argument("-H", "--host", default="127.0.0.1", help="Control server host")
    parser.add_argument("-p", "--port", type=int, default=2628, help="Control server port")
    args = parser.parse_args()

    try:
        sock = connect(args.host, args.port)
    except ConnectionRefusedError:
        print(f"Could not connect to control plane at {args.host}:{args.port}")
        print("Is the game server running?")
        sys.exit(1)

    try:
        if args.file:
            run_script(sock, args.file)
        elif args.command:
            cmd = " ".join(args.command)
            response = send_command(sock, cmd)
            if response:
                print(response)
        else:
            interactive(sock)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
