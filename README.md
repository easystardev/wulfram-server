# Wulfram II Server Emulator

Work-in-progress server emulator based on protocol docs from https://azurefishy.github.io/Wulfram2Dev/

## Current Status

- [x] TCP listener
- [x] HELLO handshake (version, UDP port, server name)
- [x] LOGIN handling (accept any credentials)
- [x] LOGIN_STATUS success response
- [ ] BEHAVIOR packet (needed for team select)
- [ ] TEAM_INFO packet (needed for team select)
- [ ] UDP protocol
- [ ] Game state sync

## How to Run

### Option 1: Hosts file redirect (requires admin)

1. Edit `C:\Windows\System32\drivers\etc\hosts` (as admin):
   ```
   127.0.0.1 www.wulfram.com wulfram.com
   ```

2. Run the fake HTTP server (as admin for port 80):
   ```
   python fake_server_list.py
   ```

3. Run the game server:
   ```
   python wulfram_server.py
   ```

4. Launch `wulfram2.exe`

### Option 2: Modify wulfemulator DLL (advanced)

The wulfemulator DLL can be modified to redirect `try_server_connect` calls to localhost.
Requires Visual Studio to build.

### Option 3: Binary patch (advanced)

Patch `wulfram2.exe` to change the hardcoded server URLs.

## Protocol Reference

See `../AGENTS.md` for protocol summary or full docs at:
https://azurefishy.github.io/Wulfram2Dev/

## Files

- `wulfram_server.py` - Main TCP game server
- `fake_server_list.py` - HTTP server for fake server list
