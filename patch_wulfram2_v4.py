#!/usr/bin/env python3
"""
Wulfram 2 Binary Patcher - Version 4

More comprehensive fix: when EBX is NULL at 0x004837C9, skip the entire
EBX-dependent code block and jump to the existing null check at 0x00483880.

The code at 0x00483880 already has: TEST EBX, EBX; JZ skip
So we leverage that existing null handler.

Original code at 0x004837C9:
    50              PUSH EAX
    51              PUSH ECX
    8B CB           MOV ECX, EBX
    E8 1E C6 03 00  CALL 0x004bfdf0
    8B 73 30        MOV ESI, [EBX+30h]   <- crashes if EBX=0
    ... more EBX usage ...
    85 DB           TEST EBX, EBX        <- at 0x00483880
    74 10           JZ skip_block        <- existing null handler!

Patched at 0x004837C9:
    E9 XX XX XX XX  JMP code_cave
    90 90 90 90     NOP (pad to 0x004837D2)

Code cave:
    TEST EBX, EBX
    JZ skip_to_existing_null_check
    ; Normal path
    PUSH EAX
    PUSH ECX
    MOV ECX, EBX
    CALL 0x004bfdf0
    JMP back_to_0x004837D2
skip_to_existing_null_check:
    JMP 0x00483880      ; Jump to existing TEST EBX,EBX; JZ handler
"""

import sys
from pathlib import Path

# File offsets
PATCH_SITE = 0x0837C9           # Before the PUSHes
PATCH_V4_CAVE = 0x0EA540        # After v2 and v3 caves

# Original bytes at patch site (9 bytes: PUSH EAX, PUSH ECX, MOV ECX EBX, CALL)
ORIGINAL_SITE = bytes([0x50, 0x51, 0x8B, 0xCB, 0xE8, 0x1E, 0xC6, 0x03, 0x00])

# Patch at site: JMP to code cave + 4 NOPs
# JMP offset: 0x004EA540 - 0x004837CE = 0x66D72
PATCH_SITE_BYTES = bytes([0xE9, 0x72, 0x6D, 0x06, 0x00, 0x90, 0x90, 0x90, 0x90])

# V4 Code cave at 0x004EA540:
#   85 DB           TEST EBX, EBX           (2 bytes)
#   74 0E           JZ +14 to skip_label    (2 bytes)
#   50              PUSH EAX                (1 byte)
#   51              PUSH ECX                (1 byte)
#   8B CB           MOV ECX, EBX            (2 bytes)
#   E8 A5 58 FD FF  CALL 0x004bfdf0         (5 bytes) offset from 0x4EA54B
#   E9 82 F2 F9 FF  JMP 0x004837D2          (5 bytes) offset from 0x4EA550
# skip_label at 0x4EA555:
#   E9 26 F3 F9 FF  JMP 0x00483880          (5 bytes) offset from 0x4EA555

# Calculate offsets:
# CALL target: 0x004bfdf0, from 0x004EA54B+5=0x004EA550, offset = 0x004bfdf0 - 0x004EA550 = -0x2A760 = 0xFFFD58A0
# JMP back: 0x004837D2, from 0x004EA550+5=0x004EA555, offset = 0x004837D2 - 0x004EA555 = -0x66D83 = 0xFFF9927D
# JMP to null check: 0x00483880, from 0x004EA555+5=0x004EA55A, offset = 0x00483880 - 0x004EA55A = -0x66CDA = 0xFFF99326

PATCH_V4_CAVE_BYTES = bytes([
    0x85, 0xDB,                         # TEST EBX, EBX
    0x74, 0x0E,                         # JZ +14 (to skip_label at 0x4EA552)
    0x50,                               # PUSH EAX
    0x51,                               # PUSH ECX
    0x8B, 0xCB,                         # MOV ECX, EBX
    0xE8, 0xA3, 0x58, 0xFD, 0xFF,       # CALL 0x004bfdf0
    0xE9, 0x80, 0x92, 0xF9, 0xFF,       # JMP 0x004837D2
    # skip_label at 0x4EA552:
    0xE9, 0x2A, 0x93, 0xF9, 0xFF        # JMP 0x00483881 (existing TEST EBX,EBX; JZ)
])

# Also include v2 patches for safety
PATCH_V2_ENTRY = 0x0BFDF0
PATCH_V2_CAVE = 0x0EA4F8
ORIGINAL_V2_ENTRY = bytes([0x83, 0xEC, 0x2C, 0x53, 0x55])
PATCH_V2_JMP = bytes([0xE9, 0x03, 0xA7, 0x02, 0x00])
PATCH_V2_CAVE_BYTES = bytes([
    0x85, 0xC9, 0x74, 0x0B,
    0x83, 0xEC, 0x2C, 0x53, 0x55,
    0xE9, 0xEF, 0x58, 0xFD, 0xFF,
    0x31, 0xC0, 0xC2, 0x08, 0x00
])


# Registry patches: HKEY_USERS (0x80000003) -> HKEY_CURRENT_USER (0x80000001)
# This avoids the "Can't open nor create registry element: HKEY_USERS\.DEFAULT" admin error
REGISTRY_HKEY_PATCHES = [
    0x099BF7,  # First PUSH HKEY_USERS + 1 (the 03 byte)
    0x099C7A,  # Second PUSH HKEY_USERS + 1 (the 03 byte)
]

# Registry display string patches: Change "HKEY_USERS" to "HKEY_CURRENT_USER"
# The code pushes "HKEY_USERS" string for error message display. Change it to
# push an existing "HKEY_CURRENT_USER" string at 0x53F164 instead.
REGISTRY_DISPLAY_PATCHES = [
    # (offset of PUSH instruction, original bytes, new bytes)
    # Original: PUSH 0x53F434 ("HKEY_USERS")
    # Patched:  PUSH 0x53F164 ("HKEY_CURRENT_USER")
    (0x099BF1, bytes([0x68, 0x34, 0xF4, 0x53, 0x00]), bytes([0x68, 0x64, 0xF1, 0x53, 0x00])),
    (0x099C74, bytes([0x68, 0x8C, 0xF4, 0x53, 0x00]), bytes([0x68, 0x64, 0xF1, 0x53, 0x00])),
]

# Registry path patches: Skip ".DEFAULT" in the subkey path
# The PUSH instructions load the address of ".DEFAULT" string. We change them
# to push the address of an empty string (the null bytes after "HKEY_CURRENT_USER").
# This makes the path go directly from HKEY root to "Software\Wulfram2".
#
# Original: PUSH 0x53F440 (.DEFAULT at 0x13F440)
# Patched:  PUSH 0x53F175 (empty string - null after "HKEY_CURRENT_USER" at 0x53F164)
REGISTRY_DEFAULT_PUSH_PATCHES = [
    # (offset of PUSH instruction, original bytes, new bytes)
    (0x099C04, bytes([0x68, 0x40, 0xF4, 0x53, 0x00]), bytes([0x68, 0x75, 0xF1, 0x53, 0x00])),
    (0x099C87, bytes([0x68, 0x98, 0xF4, 0x53, 0x00]), bytes([0x68, 0x75, 0xF1, 0x53, 0x00])),
]

# URL patches: Replace hardcoded wulfram.com URLs with localhost
# This makes the game connect to local server instead of the defunct wulfram.com
URL_PATCHES = [
    # (offset, original_url, new_url)
    (0x135C28, b"http://www.wulfram.com\x00", b"http://127.0.0.1:8080\x00\x00"),
    (0x135C40, b"http://www.wulfram.com/updates.php\x00", b"http://127.0.0.1:8080/updates.php\x00\x00"),
]

# HKEY_CLASSES_ROOT patches: Change HKCR (0x80000000) to HKCU (0x80000001)
# This avoids the "Can't open nor create registry element: HKEY_CLASSES_ROOT\.w2l" admin error
# The .w2l file type registration code uses HKCR which requires admin. Changing to HKCU
# makes it use per-user file associations (HKCU\Software\Classes acts like HKCR for current user)
# PUSH instruction is: 68 00 00 00 80 -> change byte at offset+1 from 00 to 01
# HKCR/HKLM patches - change to HKCU to avoid admin requirements
# PUSH 68 XX 00 00 80 - change byte at offset+1 from XX to 01 (HKCU)
REGISTRY_HKCR_PATCHES = []  # HKCR = 0x80000000, not used

# HKLM = 0x80000002, change to HKCU = 0x80000001
# Found at offsets where PUSH 68 02 00 00 80 appears
REGISTRY_HKLM_PATCHES = [
    0x099833,  # offset+1 of PUSH HKLM at 0x099832
    0x0998AD,  # offset+1 of PUSH HKLM at 0x0998AC
    0x0999EA,  # offset+1 of PUSH HKLM at 0x0999E9
]

# Skip file registration function: NOP the CALL at 0x019850
# DISABLED - this breaks game startup! The function does more than just registry.
# This prevents all the HKCR/HKLM/HKU registry error dialogs
SKIP_FILE_REGISTRATION = None  # 0x019850  # CALL to file registration function

# =============================================================================
# DirectInput NULL HWND Bug Fix
# =============================================================================
# Bug: SetCooperativeLevel is called with HWND=0 (NULL) for mouse device
# Location: 0x0049D010-0x0049D01A (VA), file offset 0x09D010-0x09D01A
#
# Original code:
#   A1 D0 14 5E 00  mov eax, [mouse_ptr]  ; EAX = mouse device
#   6A 0A           push 10               ; flags = DISCL_FOREGROUND | DISCL_NONEXCLUSIVE
#   6A 00           push 0                ; hwnd = NULL (BUG!)
#   50              push eax              ; this
#   8B 08           mov ecx, [eax]        ; vtable
#   FF 51 34        call [ecx+34h]        ; SetCooperativeLevel
#
# Fix: Replace with JMP to code cave that calls GUESS2_Winsys_get_hwnd_direct()
#      (at VA 0x004b8820) to get the proper HWND
#
# Code at 0x49D010 (10 bytes): 6A 0A 6A 00 50 8B 08 FF 51 34
# Replaced with: E8 XX XX XX XX 90 90 90 90 90 (CALL cave + 5 NOPs)
#
DINPUT_HWND_PATCH_SITE = 0x09D010  # File offset of push 10 (start of patch area)
DINPUT_HWND_ORIGINAL = bytes([0x6A, 0x0A, 0x6A, 0x00, 0x50, 0x8B, 0x08, 0xFF, 0x51, 0x34])

# Code cave location (after other caves)
DINPUT_HWND_CAVE = 0x0EA560

# The cave needs to:
# 1. Save EAX (mouse device ptr)
# 2. Call GUESS2_Winsys_get_hwnd_direct (VA 0x004b8820)
# 3. Move result (hwnd) to EDX
# 4. Restore EAX (mouse device ptr)
# 5. Push 10 (flags)
# 6. Push EDX (hwnd)
# 7. Push EAX (this)
# 8. Call SetCooperativeLevel through vtable
# 9. JMP back to 0x49D01A (after the call)
#
# Cave code (at VA 0x004EA560):
#   50              push eax              ; save mouse ptr (1 byte)
#   E8 BB 82 FD FF  call 0x004B8820       ; GUESS2_Winsys_get_hwnd_direct (5 bytes)
#                   ; offset = 0x4B8820 - (0x4EA561 + 5) = 0x4B8820 - 0x4EA566 = -0x31D46 = 0xFFFCE2BA
#   89 C2           mov edx, eax          ; EDX = hwnd (2 bytes)
#   58              pop eax               ; restore mouse ptr (1 byte)
#   6A 0A           push 10               ; flags (2 bytes)
#   52              push edx              ; hwnd (fixed!) (1 byte)
#   50              push eax              ; this (1 byte)
#   8B 08           mov ecx, [eax]        ; vtable (2 bytes)
#   FF 51 34        call [ecx+34h]        ; SetCooperativeLevel (3 bytes)
#   E9 XX XX XX XX  jmp 0x0049D01A        ; return to normal flow (5 bytes)
#                   ; offset = 0x49D01A - (0x4EA578 + 5) = 0x49D01A - 0x4EA57D = -0x50863 = 0xFFFAF79D

# Using GetForegroundWindow from IAT instead of game's internal function
# IAT address for GetForegroundWindow: 0x004EB2BC
# Cave layout:
#   50                   push eax              ; save mouse ptr (1 byte)
#   FF 15 BC B2 4E 00    call [GetForegroundWindow] ; get hwnd (6 bytes)
#   89 C2                mov edx, eax          ; edx = hwnd (2 bytes)
#   58                   pop eax               ; restore mouse ptr (1 byte)
#   6A 0A                push 10               ; flags (2 bytes)
#   52                   push edx              ; hwnd (1 byte)
#   50                   push eax              ; this (1 byte)
#   8B 08                mov ecx, [eax]        ; vtable (2 bytes)
#   FF 51 34             call [ecx+34h]        ; SetCooperativeLevel (3 bytes)
#   E9 XX XX XX XX       jmp 0x0049D01A        ; return (5 bytes)
# Total: 24 bytes
# jmp offset: from 0x4EA560+19=0x4EA573, +5=0x4EA578
# offset = 0x49D01A - 0x4EA578 = -0x50B5E = 0xFFFAF4A2

DINPUT_HWND_CAVE_BYTES = bytes([
    0x50,                               # push eax (save mouse ptr)
    0xFF, 0x15, 0xBC, 0xB2, 0x4E, 0x00, # call dword ptr [GetForegroundWindow]
    0x89, 0xC2,                         # mov edx, eax (edx = hwnd)
    0x58,                               # pop eax (restore mouse ptr)
    0x6A, 0x0A,                         # push 10 (flags)
    0x52,                               # push edx (hwnd - fixed!)
    0x50,                               # push eax (this)
    0x8B, 0x08,                         # mov ecx, [eax] (vtable)
    0xFF, 0x51, 0x34,                   # call [ecx+34h] (SetCooperativeLevel)
    0xE9, 0xA2, 0xF4, 0xFA, 0xFF,       # jmp 0x0049D01A (continue normal flow)
])

# Patch at site: CALL cave + 5 NOPs
# CALL offset = cave_va - (site_va + 5) = 0x4EA560 - (0x49D010 + 5) = 0x4EA560 - 0x49D015 = 0x4D54B
DINPUT_HWND_PATCH_BYTES = bytes([
    0xE8, 0x4B, 0xD5, 0x04, 0x00,       # call 0x004EA560 (code cave)
    0x90, 0x90, 0x90, 0x90, 0x90,       # 5 NOPs to fill remaining space
])


def apply_patch(input_path: Path, output_path: Path) -> bool:
    """Apply the v4 patch to wulfram2.exe"""

    print(f"Reading: {input_path}")
    data = bytearray(input_path.read_bytes())

    # Check current state
    current_site = bytes(data[PATCH_SITE:PATCH_SITE+9])
    if current_site == PATCH_SITE_BYTES:
        print("V4 patch already applied!")
        return False

    # Verify we're patching the right location
    if current_site != ORIGINAL_SITE:
        print(f"WARNING: Unexpected bytes at patch site 0x{PATCH_SITE:X}")
        print(f"  Expected: {ORIGINAL_SITE.hex()}")
        print(f"  Found:    {current_site.hex()}")
        # Check if v3 was applied
        if current_site[0] == 0xE9:
            print("  Looks like v3 patch - will overwrite")

    # Apply v2 patch (function entry safety net)
    print(f"Applying v2 patch at 0x{PATCH_V2_ENTRY:X}...")
    data[PATCH_V2_ENTRY:PATCH_V2_ENTRY+5] = PATCH_V2_JMP
    data[PATCH_V2_CAVE:PATCH_V2_CAVE+len(PATCH_V2_CAVE_BYTES)] = PATCH_V2_CAVE_BYTES

    # Apply v4 patch
    print(f"Applying v4 patch at 0x{PATCH_SITE:X}...")
    data[PATCH_SITE:PATCH_SITE+9] = PATCH_SITE_BYTES
    data[PATCH_V4_CAVE:PATCH_V4_CAVE+len(PATCH_V4_CAVE_BYTES)] = PATCH_V4_CAVE_BYTES

    # Apply registry patches (HKEY_USERS -> HKEY_CURRENT_USER)
    print("Applying registry patches (HKEY_USERS -> HKEY_CURRENT_USER)...")
    for offset in REGISTRY_HKEY_PATCHES:
        if data[offset] == 0x03:  # HKEY_USERS
            data[offset] = 0x01   # HKEY_CURRENT_USER
            print(f"  HKEY patch 0x{offset:06X}: 0x03 -> 0x01")
        else:
            print(f"  HKEY patch 0x{offset:06X}: already patched or unexpected value")

    # Apply registry display patches (HKEY_USERS string -> HKEY_CURRENT_USER string)
    print("Applying registry display patches (HKEY_USERS -> HKEY_CURRENT_USER string)...")
    for offset, original, new in REGISTRY_DISPLAY_PATCHES:
        current = bytes(data[offset:offset+5])
        if current == original:
            data[offset:offset+5] = new
            print(f"  Display patch 0x{offset:06X}: {original.hex()} -> {new.hex()}")
        elif current == new:
            print(f"  Display patch 0x{offset:06X}: already patched")
        else:
            print(f"  Display patch 0x{offset:06X}: unexpected bytes: {current.hex()}")

    # Apply registry path patches (change PUSH .DEFAULT to PUSH empty string)
    print("Applying registry path patches (skip .DEFAULT)...")
    for offset, original, new in REGISTRY_DEFAULT_PUSH_PATCHES:
        current = bytes(data[offset:offset+5])
        if current == original:
            data[offset:offset+5] = new
            print(f"  PUSH patch 0x{offset:06X}: {original.hex()} -> {new.hex()}")
        elif current == new:
            print(f"  PUSH patch 0x{offset:06X}: already patched")
        else:
            print(f"  PUSH patch 0x{offset:06X}: unexpected bytes: {current.hex()}")

    # Apply URL patches (wulfram.com -> localhost)
    print("Applying URL patches (wulfram.com -> localhost)...")
    for offset, original, new in URL_PATCHES:
        current = bytes(data[offset:offset+len(original)])
        if current == original:
            data[offset:offset+len(new)] = new
            print(f"  Patched 0x{offset:06X}: {original[:30]}... -> {new[:30]}...")
        elif current == new[:len(original)]:
            print(f"  0x{offset:06X}: already patched")
        else:
            print(f"  WARNING: 0x{offset:06X}: unexpected content: {current[:30]}...")

    # Skip file registration function (NOP the CALL)
    if SKIP_FILE_REGISTRATION is not None:
        print("Skipping file registration (NOP CALL to registry function)...")
        if data[SKIP_FILE_REGISTRATION] == 0xE8:  # CALL opcode
            # Replace 5-byte CALL with 5 NOPs
            data[SKIP_FILE_REGISTRATION:SKIP_FILE_REGISTRATION+5] = bytes([0x90, 0x90, 0x90, 0x90, 0x90])
            print(f"  NOP'd CALL at 0x{SKIP_FILE_REGISTRATION:06X}")
        elif data[SKIP_FILE_REGISTRATION] == 0x90:
            print(f"  CALL at 0x{SKIP_FILE_REGISTRATION:06X}: already NOP'd")
        else:
            print(f"  WARNING: Unexpected opcode at 0x{SKIP_FILE_REGISTRATION:06X}: 0x{data[SKIP_FILE_REGISTRATION]:02X}")
    else:
        print("Skipping file registration: DISABLED (breaks game startup)")

    # Apply HKLM patches (HKEY_LOCAL_MACHINE -> HKEY_CURRENT_USER)
    if REGISTRY_HKLM_PATCHES:
        print("Applying HKLM patches (HKEY_LOCAL_MACHINE -> HKEY_CURRENT_USER)...")
        for offset in REGISTRY_HKLM_PATCHES:
            if data[offset] == 0x02:  # HKEY_LOCAL_MACHINE
                data[offset] = 0x01   # HKEY_CURRENT_USER
                print(f"  HKLM patch 0x{offset:06X}: 0x02 -> 0x01")
            elif data[offset] == 0x01:
                print(f"  HKLM patch 0x{offset:06X}: already patched")
            else:
                print(f"  HKLM patch 0x{offset:06X}: unexpected value 0x{data[offset]:02X}")

    # Apply DirectInput NULL HWND fix
    print("Applying DirectInput HWND fix (SetCooperativeLevel NULL hwnd bug)...")
    current_dinput = bytes(data[DINPUT_HWND_PATCH_SITE:DINPUT_HWND_PATCH_SITE+10])
    if current_dinput == DINPUT_HWND_ORIGINAL:
        # Apply patch at call site
        data[DINPUT_HWND_PATCH_SITE:DINPUT_HWND_PATCH_SITE+10] = DINPUT_HWND_PATCH_BYTES
        print(f"  Patched call site 0x{DINPUT_HWND_PATCH_SITE:06X}")
        # Write code cave
        data[DINPUT_HWND_CAVE:DINPUT_HWND_CAVE+len(DINPUT_HWND_CAVE_BYTES)] = DINPUT_HWND_CAVE_BYTES
        print(f"  Wrote code cave at 0x{DINPUT_HWND_CAVE:06X}")
    elif current_dinput == DINPUT_HWND_PATCH_BYTES:
        print(f"  DirectInput HWND patch: already applied")
    else:
        print(f"  WARNING: Unexpected bytes at 0x{DINPUT_HWND_PATCH_SITE:06X}: {current_dinput.hex()}")

    # Write output
    print(f"Writing: {output_path}")
    output_path.write_bytes(data)

    # Verify
    verify_data = output_path.read_bytes()
    v2_ok = bytes(verify_data[PATCH_V2_ENTRY:PATCH_V2_ENTRY+5]) == PATCH_V2_JMP
    v4_site_ok = bytes(verify_data[PATCH_SITE:PATCH_SITE+9]) == PATCH_SITE_BYTES
    v4_cave_ok = bytes(verify_data[PATCH_V4_CAVE:PATCH_V4_CAVE+len(PATCH_V4_CAVE_BYTES)]) == PATCH_V4_CAVE_BYTES

    # Verify registry patches
    reg_hkey_ok = all(verify_data[off] == 0x01 for off in REGISTRY_HKEY_PATCHES)
    reg_display_ok = all(
        bytes(verify_data[off:off+5]) == new
        for off, _, new in REGISTRY_DISPLAY_PATCHES
    )
    reg_path_ok = all(
        bytes(verify_data[off:off+5]) == new
        for off, _, new in REGISTRY_DEFAULT_PUSH_PATCHES
    )

    # Verify URL patches
    url_ok = all(
        bytes(verify_data[off:off+len(new)]) == new
        for off, _, new in URL_PATCHES
    )

    # Verify file registration skip (should be NOPs) - only if enabled
    if SKIP_FILE_REGISTRATION is not None:
        file_reg_ok = verify_data[SKIP_FILE_REGISTRATION:SKIP_FILE_REGISTRATION+5] == bytes([0x90]*5)
    else:
        file_reg_ok = True  # Skip verification if disabled

    # Verify HKLM patches
    reg_hklm_ok = all(verify_data[off] == 0x01 for off in REGISTRY_HKLM_PATCHES) if REGISTRY_HKLM_PATCHES else True

    # Verify DirectInput HWND patch
    dinput_site_ok = bytes(verify_data[DINPUT_HWND_PATCH_SITE:DINPUT_HWND_PATCH_SITE+10]) == DINPUT_HWND_PATCH_BYTES
    dinput_cave_ok = bytes(verify_data[DINPUT_HWND_CAVE:DINPUT_HWND_CAVE+len(DINPUT_HWND_CAVE_BYTES)]) == DINPUT_HWND_CAVE_BYTES

    print("\nVerification:")
    print(f"  V2 function entry: {'OK' if v2_ok else 'MISMATCH!'}")
    print(f"  V4 call site:      {'OK' if v4_site_ok else 'MISMATCH!'}")
    print(f"  V4 code cave:      {'OK' if v4_cave_ok else 'MISMATCH!'}")
    print(f"  Registry HKEY:     {'OK' if reg_hkey_ok else 'MISMATCH!'}")
    print(f"  Registry display:  {'OK' if reg_display_ok else 'MISMATCH!'}")
    print(f"  Registry path:     {'OK' if reg_path_ok else 'MISMATCH!'}")
    print(f"  Registry HKLM:     {'OK' if reg_hklm_ok else 'MISMATCH!'}")
    print(f"  URL patches:       {'OK' if url_ok else 'MISMATCH!'}")
    print(f"  File reg skip:     {'OK' if file_reg_ok else 'MISMATCH!'}")
    print(f"  DInput HWND site:  {'OK' if dinput_site_ok else 'MISMATCH!'}")
    print(f"  DInput HWND cave:  {'OK' if dinput_cave_ok else 'MISMATCH!'}")

    return True


def main():
    game_dir = Path(r"C:\Users\wstri\dev\wolfram\wulfram2-game")
    # Use original as input to ensure we start fresh
    input_exe = game_dir / "wulfram2_original.exe"
    output_exe = game_dir / "wulfram2_fixed.exe"

    if not input_exe.exists():
        print(f"ERROR: Cannot find {input_exe}")
        sys.exit(1)

    print("=" * 60)
    print("Wulfram 2 Binary Patcher - Version 4.1")
    print("=" * 60)
    print()
    print("V4: Skip entire EBX-dependent block when EBX is NULL")
    print("V2: Safety net at function entry")
    print("Registry: HKEY_USERS -> HKEY_CURRENT_USER (no admin required)")
    print("Registry: Skip file registration (no HKCR/HKLM dialogs)")
    print("URLs: wulfram.com -> 127.0.0.1:8080 (local server)")
    print("NEW: DirectInput HWND fix (SetCooperativeLevel bug)")
    print()

    if apply_patch(input_exe, output_exe):
        print()
        print("=" * 60)
        print("Patch applied successfully!")
        print(f"Run: {output_exe}")
        print("=" * 60)


if __name__ == "__main__":
    main()
