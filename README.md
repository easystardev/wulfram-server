# Wulfram 2 Server

A Python server emulator for Wulfram 2, implementing server-authoritative physics.

## Download the Game

<!-- TODO: Add download link -->
**Game download:** [LINK NEEDED]

## Current Status

### Working
- TCP/UDP networking and session management
- Player login and team selection
- Entity spawning with server-controlled position/velocity
- Server-authoritative physics (gravity, ground collision)
- UPDATE_ARRAY packets with health/energy state
- Client input reception (ACTION_DUMP packets)
- Packet tracing for debugging

### In Progress
- Player movement from client inputs
- Viewpoint/rotation synchronization
- Weapon firing and projectiles

## Quick Start

### 1. Patch the Client

The game client needs patching to connect to localhost and fix a DirectInput bug:

```powershell
# From the server directory
python patch_slurpysoft.py path/to/wulfram2.exe
```

This creates `wulfram2_fixed.exe` with:
- Server URLs redirected to `127.0.0.1`
- DirectInput NULL HWND bug fixed

### 2. Run the Server

```powershell
# Start the server (TCP port 20100, UDP port 20105)
python manage_server.py start

# Or run directly
python run_server.py
```

### 3. Launch the Game

Run the patched `wulfram2_fixed.exe`. The game will connect to your local server.

## Testing

The `test_spawn.ps1` script automates testing the spawn sequence:

```powershell
# Run with explicit paths
.\test_spawn.ps1 -GamePath "C:\path\to\wulfram2_fixed.exe" -LogPath ".\server.log"

# Or set environment variables
$env:WULFRAM_GAME_PATH = "C:\Games\Wulfram2\wulfram2_fixed.exe"
$env:WULFRAM_LOG_PATH = "C:\path\to\server.log"
.\test_spawn.ps1

# Options:
#   -GamePath       Path to wulfram2_fixed.exe or launch.bat
#   -LogPath        Path to server.log file
#   -Launch         Launch the game (default: true)
#   -KillExisting   Kill existing game processes (default: true)
#   -Screenshot     Capture screenshots for analysis
#   -SkipInput      Don't send test inputs
#   -Verbose        Show detailed click coordinates
```

The test script:
1. Launches the patched game
2. Automatically clicks through login/team select
3. Waits for spawn
4. Sends test inputs (WASD, mouse, fire)
5. Checks for crashes, health issues, movement

## Project Structure

```
server/
├── wulfram/
│   ├── server.py      # Main game server and tick loop
│   ├── session.py     # Player session state
│   ├── packets.py     # Packet building (UPDATE_ARRAY, etc)
│   ├── codec.py       # Binary encoding (BitWriter, quantizers)
│   ├── handlers.py    # Packet handlers
│   ├── transport.py   # TCP/UDP socket handling
│   └── weapons.py     # Weapon/input parsing
├── patch_slurpysoft.py   # Client binary patcher
├── fake_server_list.py   # HTTP server list emulator
├── manage_server.py      # Server management script
└── test_*.py             # Unit tests
```

## Protocol Notes

The server implements the Wulfram 2 network protocol:

- **TCP (port 20100)**: Login, team select, reliable game packets
- **UDP (port 20105)**: Real-time position updates, inputs

Key packet types:
- `0x0E UPDATE_ARRAY`: Entity state (position, velocity, health)
- `0x09 ACTION_DUMP`: Client inputs (movement, firing)
- `0x18 TANK_PACKET`: Initial entity spawn
- `0x35 VIEWPOINT_INFO`: Camera/rotation state

## Development

```powershell
# Run tests
python -m pytest test_*.py -v

# Watch server logs
Get-Content server.log -Wait -Tail 50
```

## License

This project is for educational and preservation purposes.
