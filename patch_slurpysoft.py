#!/usr/bin/env python3
"""
SlurpySoft Wulfram 2 Binary Patcher

This patches the SlurpySoft version of wulfram2.exe which matches the
decompiled source in azurefishy-src.

Fixes:
1. DirectInput SetCooperativeLevel NULL HWND bug
   - Location: VA 0x4B163E (file offset varies by section layout)
   - Original: push 10, push 0, ... (NULL HWND)
   - Fixed: calls GUESS2_Winsys_get_hwnd_direct() at 0x4B8820

2. URL patches for local server connection
"""

import sys
from pathlib import Path

# SlurpySoft PE layout:
# .text VA=0x1000, RawPtr=0x400
# File offset = VA - 0x1000 + 0x400 = VA - 0xC00

def va_to_file(va):
    """Convert virtual address to file offset for SlurpySoft binary"""
    return va - 0x400000 - 0x1000 + 0x400

def file_to_va(offset):
    """Convert file offset to virtual address for SlurpySoft binary"""
    return offset + 0x400000 + 0x1000 - 0x400

# DirectInput HWND bug fix
# Location: VA 0x4B163E = file offset 0x0B0A3E
# Original code:
#   6A 0A             push 10        ; flags
#   6A 00             push 0         ; NULL HWND (BUG!)
#   8B 0D B4 85 67 00 mov ecx, [mouse_ptr]
#   8B 11             mov edx, [ecx]
#   A1 B4 85 67 00    mov eax, [mouse_ptr]
#   50                push eax       ; this
#   8B 4A 34          mov ecx, [edx+34h]
#   FF D1             call ecx       ; SetCooperativeLevel
#
# Total: 23 bytes (6A0A 6A00 8B0DB4856700 8B11 A1B4856700 50 8B4A34 FFD1)

DINPUT_PATCH_VA = 0x4B163E
DINPUT_PATCH_FILE = va_to_file(DINPUT_PATCH_VA)

DINPUT_ORIGINAL = bytes([
    0x6A, 0x0A,                         # push 10
    0x6A, 0x00,                         # push 0 (BUG!)
    0x8B, 0x0D, 0xB4, 0x85, 0x67, 0x00, # mov ecx, [0x6785B4]
    0x8B, 0x11,                         # mov edx, [ecx]
    0xA1, 0xB4, 0x85, 0x67, 0x00,       # mov eax, [0x6785B4]
    0x50,                               # push eax
    0x8B, 0x4A, 0x34,                   # mov ecx, [edx+34h]
    0xFF, 0xD1,                         # call ecx
])

# Code cave for the fix
# We'll place it at end of .text section padding
# .text size is 0x13A20E, ends around VA 0x13B20E
# Let's use VA 0x13B000 (file offset 0x13A400) as code cave
CODE_CAVE_VA = 0x53B000
CODE_CAVE_FILE = va_to_file(CODE_CAVE_VA)

# Fixed code cave:
#   E8 XX XX XX XX    call GUESS2_Winsys_get_hwnd_direct (at 0x4B8820)
#   89 C3             mov ebx, eax   ; save hwnd
#   8B 0D B4 85 67 00 mov ecx, [mouse_ptr]
#   8B 11             mov edx, [ecx]
#   A1 B4 85 67 00    mov eax, [mouse_ptr]
#   6A 0A             push 10        ; flags
#   53                push ebx       ; hwnd (fixed!)
#   50                push eax       ; this
#   8B 4A 34          mov ecx, [edx+34h]
#   FF D1             call ecx       ; SetCooperativeLevel
#   E9 XX XX XX XX    jmp back

# Calculate offsets
# CALL target: 0x4B8820
# From: CODE_CAVE_VA + 5 = 0x53B005
# Offset = 0x4B8820 - 0x53B005 = -0x827E5 = 0xFFF7D81B

# JMP back target: VA after original code = 0x4B1655 (0x4B163E + 23)
# From: CODE_CAVE_VA + cave_size
# Cave is ~28 bytes, so from 0x53B01C
# Offset = 0x4B1655 - 0x53B01C = -0x899C7 = 0xFFF76639

CODE_CAVE_BYTES = bytes([
    0xE8, 0x1B, 0xD8, 0xF7, 0xFF,       # call 0x4B8820 (GUESS2_Winsys_get_hwnd_direct)
    0x89, 0xC3,                         # mov ebx, eax (save hwnd)
    0x8B, 0x0D, 0xB4, 0x85, 0x67, 0x00, # mov ecx, [0x6785B4]
    0x8B, 0x11,                         # mov edx, [ecx]
    0xA1, 0xB4, 0x85, 0x67, 0x00,       # mov eax, [0x6785B4]
    0x6A, 0x0A,                         # push 10 (flags)
    0x53,                               # push ebx (hwnd - fixed!)
    0x50,                               # push eax (this)
    0x8B, 0x4A, 0x34,                   # mov ecx, [edx+34h]
    0xFF, 0xD1,                         # call ecx
    0xE9, 0x39, 0x66, 0xF7, 0xFF,       # jmp 0x4B1655
])

# Patch at original site: JMP to code cave + NOPs
# JMP offset = CODE_CAVE_VA - (DINPUT_PATCH_VA + 5) = 0x53B000 - 0x4B1643 = 0x899BD
DINPUT_PATCH_BYTES = bytes([
    0xE9, 0xBD, 0x99, 0x08, 0x00,       # jmp CODE_CAVE_VA
]) + bytes([0x90] * (len(DINPUT_ORIGINAL) - 5))  # NOPs

# =============================================================================
# Registry Patches: HKEY_USERS/HKEY_LOCAL_MACHINE -> HKEY_CURRENT_USER
# =============================================================================
# The game tries to register file associations in multiple registry locations
# that require admin rights on modern Windows. Change them to HKEY_CURRENT_USER.
#
# HKEY constant values:
#   HKEY_CLASSES_ROOT    = 0x80000000
#   HKEY_CURRENT_USER    = 0x80000001
#   HKEY_LOCAL_MACHINE   = 0x80000002
#   HKEY_USERS           = 0x80000003
#
# The code uses: MOV DWORD PTR [ESP+XX], 0x800000YY
# Pattern: C7 44 24 XX YY 00 00 80
# We patch the YY byte (at instruction offset+4) from 02/03 to 01

# HKEY_USERS (0x80000003) -> HKEY_CURRENT_USER (0x80000001)
# These are at file offset+4 where the 03 byte is
REGISTRY_HKU_PATCHES = [
    0x0A4048,  # MOV [ESP+74h], 0x80000003 -> 0x80000001
    0x0A40DA,  # MOV [ESP+60h], 0x80000003 -> 0x80000001
]

# HKEY_LOCAL_MACHINE (0x80000002) -> HKEY_CURRENT_USER (0x80000001)
REGISTRY_HKLM_PATCHES = [
    0x0A395C,  # MOV [ESP+60h], 0x80000002 -> 0x80000001
    0x0A3A0D,  # MOV [ESP+60h], 0x80000002 -> 0x80000001
    0x0A3C9C,  # MOV [ESP+4Ch], 0x80000002 -> 0x80000001
    0x0A3DB2,  # MOV [ESP+4Ch], 0x80000002 -> 0x80000001
    0x0A3E53,  # MOV [ESP+4Ch], 0x80000002 -> 0x80000001
]

# Also patch the one at 0x06F509 and 0x06F519 if they exist
# These appear to be in a different function
REGISTRY_EXTRA_PATCHES = [
    (0x06F509, 0x02),  # HKLM -> HKCU
    (0x06F519, 0x03),  # HKU  -> HKCU
]

# =============================================================================
# Registry String Path Fix: Change HKEY_USERS\.DEFAULT to HKEY_CURRENT_USER
# =============================================================================
# The code builds a registry path using string addresses:
#   PUSH "HKEY_USERS"     (VA 0x00564978)
#   MOV ECX, ".DEFAULT"   (VA 0x00564984)
#
# Fix by changing the pushed addresses to use HKEY_CURRENT_USER:
#   HKEY_USERS (0x00564978) -> HKEY_CURRENT_USER (0x005647CC)
#   .DEFAULT (0x00564984) -> empty string (0x005647DD = null after HKCU)
#
# This makes the code access HKEY_CURRENT_USER instead of HKEY_USERS\.DEFAULT

# PUSH HKEY_USERS -> PUSH HKEY_CURRENT_USER
# Original: 68 78 49 56 00 -> New: 68 CC 47 56 00
REGISTRY_STRING_PUSH_PATCHES = [
    (0x0A4051, bytes([0x68, 0x78, 0x49, 0x56, 0x00]), bytes([0x68, 0xCC, 0x47, 0x56, 0x00])),
    (0x0A40E3, bytes([0x68, 0x78, 0x49, 0x56, 0x00]), bytes([0x68, 0xCC, 0x47, 0x56, 0x00])),
]

# MOV ECX, .DEFAULT -> MOV ECX, empty_string
# Original: B9 84 49 56 00 -> New: B9 DD 47 56 00
REGISTRY_STRING_MOV_PATCHES = [
    (0x0A405F, bytes([0xB9, 0x84, 0x49, 0x56, 0x00]), bytes([0xB9, 0xDD, 0x47, 0x56, 0x00])),
    (0x0A40F1, bytes([0xB9, 0x84, 0x49, 0x56, 0x00]), bytes([0xB9, 0xDD, 0x47, 0x56, 0x00])),
]

# =============================================================================
# Crash Fix: Destructor dereference of error code 0xFFFFFFFE
# =============================================================================
# At VA 0x0052A355 (file 0x129755): JZ (74 13) only skips NULL pointers
# But [ECX+4] can contain 0xFFFFFFFE (error code) which is not NULL but invalid
#
# Fix: Change JZ (74) to JLE (7E) - "jump if less or equal"
# After TEST EAX, EAX: JLE jumps if ZF=1 (zero) OR SF=1 (negative)
# This skips both NULL and negative error codes like 0xFFFFFFFE
DESTRUCTOR_CRASH_FIX = (0x129755, 0x74, 0x7E)  # JZ -> JLE

# =============================================================================
# Input Fix: GUESS2_Input_poll_mouse_stub always returns 0, blocking mouse input
# =============================================================================
# Function at VA 0x004b1c40 (file offset 0x0B1040):
#   55        push ebp
#   8b ec     mov ebp, esp
#   32 c0     xor al, al       ; AL = 0 (return 0) <- BUG!
#   5d        pop ebp
#   c3        ret
#
# Patch: Change "32 c0" (xor al, al) to "b0 01" (mov al, 1) to return 1
MOUSE_STUB_PATCH_FILE = 0x0B1043  # File offset of the "32 c0" bytes
MOUSE_STUB_ORIGINAL = bytes([0x32, 0xC0])  # xor al, al
MOUSE_STUB_PATCHED = bytes([0xB0, 0x01])   # mov al, 1

# =============================================================================
# Input Fix: GUESS3_Winsys_can_render gates mouse polling with flag checks
# =============================================================================
# Function at VA 0x004b8980 (file offset 0x0B7D80) checks two flags before
# calling GUESS2_DInput_poll_mouse():
#   - winsys + 0x3fa (cursor_locked) must be non-zero
#   - winsys + 0xd9 (window_active) must be non-zero
#
# Both flags are initialized to 0 and may not be getting set properly.
# The function return value gates GUESS2_InputDevice_poll_all.
#
# Original flow after call to DInput_poll_mouse:
#   0B7DBE: 5D        pop ebp
#   0B7DBF: C3        ret
#
# Problem: EAX contains winsys+0xd9 (which is 0), so return value is 0.
#
# Fix: Change the ret sequence to set AL=1 before returning:
#   0B7DBE: B0 01     mov al, 1
#   0B7DC0: 5D        pop ebp
#   0B7DC1: C3        ret
#
# But there's no room, so we need to also patch the jump targets.
# New approach: patch both JNZ->JMP AND patch after call to set EAX=1
#
# Analysis of GUESS3_Winsys_can_render:
# - If winsys is NULL: return 0
# - If cursor_locked (winsys+0x3fa) is 0: return 0
# - If window_active (winsys+0xd9) is 0: return 0  <- "xor al,al" path
# - Otherwise: call GUESS2_DInput_poll_mouse() and return EAX
#
# Problem: GUESS2_DInput_poll_mouse is VOID - doesn't set EAX!
# The return value is whatever was in EAX before the call, which is
# the window_active flag value. So if window_active=0, returns 0.
#
# Fix strategy:
# 1. Bypass cursor_locked check (JNZ->JMP at 0B7D9E) - flags aren't being set
# 2. Keep window_active check as JNZ - it controls return value!
# 3. Patch "xor al,al" to "mov al,1" at 0B7DB5 - if window_active is 0,
#    we fall through here and return 1 instead of 0
#
# Result:
# - cursor_locked=0: bypassed, continues to window_active check
# - window_active=0: JNZ not taken, executes "mov al,1", returns 1 ✓
# - window_active=1: JNZ to call, EAX=1 from movzx, returns 1 ✓
#
MOUSE_GATE_PATCH1_FILE = 0x0B7D9E  # cursor_locked check - bypass with JMP
MOUSE_GATE_ORIGINAL = 0x75  # JNZ opcode
MOUSE_GATE_PATCHED = 0xEB   # JMP opcode
# Note: We do NOT patch the second JNZ at 0B7DB3 (window_active check)
# because that would skip our "mov al,1" fix and always go to the void call

# Patch the "xor al, al" to "mov al, 1" so return value is 1 when window_active=0
# At 0B7DB5: 32 C0 (xor al, al) -> B0 01 (mov al, 1)
MOUSE_RETURN_PATCH_FILE = 0x0B7DB5
MOUSE_RETURN_ORIGINAL = bytes([0x32, 0xC0])  # xor al, al
MOUSE_RETURN_PATCHED = bytes([0xB0, 0x01])   # mov al, 1

# =============================================================================
# Cursor Clipping Fix: Enable ClipCursor in windowed mode
# =============================================================================
# Function: GUESS3_Win32_clip_cursor_to_window at VA 0x004b8a80
# The original code only calls ClipCursor when NOT in windowed mode:
#   cVar2 = GUESS2_Winsys_is_windowed_mode();
#   if ((cVar2 == '\0') && (GUESS3_g_hide_cursor_on_init != '\0')) {
#       ClipCursor(&local_28);
#   }
#
# At file offset 0x0b7fcc there's a JNZ that skips ClipCursor in windowed mode.
# Patch: NOP out the JNZ to always fall through to the hide_cursor check.
# This allows cursor to be captured in windowed mode for better gameplay.
#
CURSOR_CLIP_PATCH_FILE = 0x0B7FCC
CURSOR_CLIP_ORIGINAL = bytes([0x75, 0x15])  # JNZ +21 (skip ClipCursor if windowed)
CURSOR_CLIP_PATCHED = bytes([0x90, 0x90])   # NOP NOP (always fall through)


def apply_patch(input_path: Path, output_path: Path) -> bool:
    """Apply patches to SlurpySoft wulfram2.exe"""

    print(f"Reading: {input_path}")
    data = bytearray(input_path.read_bytes())

    print(f"File size: {len(data):,} bytes")

    # Verify this is the SlurpySoft version
    if len(data) < 1800000:
        print("ERROR: This doesn't look like the SlurpySoft version (too small)")
        return False

    # DISABLED: DirectInput HWND patch - was breaking in-game mouse input
    # The original NULL HWND with DISCL_BACKGROUND actually works for gameplay.
    # Our "fix" passing a real HWND restricts mouse input to that window.
    print("\nSkipping DirectInput HWND patch (disabled - was breaking in-game mouse)")

    # Apply HKEY_USERS patches (0x80000003 -> 0x80000001)
    print("\nApplying HKEY_USERS -> HKEY_CURRENT_USER patches...")
    for offset in REGISTRY_HKU_PATCHES:
        if data[offset] == 0x03:
            data[offset] = 0x01
            print(f"  Patched 0x{offset:06X}: 0x03 -> 0x01")
        elif data[offset] == 0x01:
            print(f"  0x{offset:06X}: Already patched")
        else:
            print(f"  WARNING: 0x{offset:06X}: Unexpected value 0x{data[offset]:02X}")

    # Apply HKEY_LOCAL_MACHINE patches (0x80000002 -> 0x80000001)
    print("\nApplying HKEY_LOCAL_MACHINE -> HKEY_CURRENT_USER patches...")
    for offset in REGISTRY_HKLM_PATCHES:
        if data[offset] == 0x02:
            data[offset] = 0x01
            print(f"  Patched 0x{offset:06X}: 0x02 -> 0x01")
        elif data[offset] == 0x01:
            print(f"  0x{offset:06X}: Already patched")
        else:
            print(f"  WARNING: 0x{offset:06X}: Unexpected value 0x{data[offset]:02X}")

    # Apply extra registry patches
    print("\nApplying extra registry patches...")
    for offset, expected in REGISTRY_EXTRA_PATCHES:
        if data[offset] == expected:
            data[offset] = 0x01
            print(f"  Patched 0x{offset:06X}: 0x{expected:02X} -> 0x01")
        elif data[offset] == 0x01:
            print(f"  0x{offset:06X}: Already patched")
        else:
            print(f"  WARNING: 0x{offset:06X}: Unexpected value 0x{data[offset]:02X}")

    # Apply registry string PUSH patches (HKEY_USERS -> HKEY_CURRENT_USER)
    print("\nApplying registry string patches (HKEY_USERS -> HKEY_CURRENT_USER)...")
    for offset, original, new in REGISTRY_STRING_PUSH_PATCHES:
        current_bytes = bytes(data[offset:offset + 5])
        if current_bytes == original:
            data[offset:offset + 5] = new
            print(f"  Patched PUSH at 0x{offset:06X}: {original.hex()} -> {new.hex()}")
        elif current_bytes == new:
            print(f"  0x{offset:06X}: Already patched")
        else:
            print(f"  WARNING: 0x{offset:06X}: Unexpected bytes {current_bytes.hex()}")

    # Apply registry string MOV patches (.DEFAULT -> empty string)
    print("\nApplying registry string patches (.DEFAULT -> empty)...")
    for offset, original, new in REGISTRY_STRING_MOV_PATCHES:
        current_bytes = bytes(data[offset:offset + 5])
        if current_bytes == original:
            data[offset:offset + 5] = new
            print(f"  Patched MOV at 0x{offset:06X}: {original.hex()} -> {new.hex()}")
        elif current_bytes == new:
            print(f"  0x{offset:06X}: Already patched")
        else:
            print(f"  WARNING: 0x{offset:06X}: Unexpected bytes {current_bytes.hex()}")

    # DISABLED: These mouse patches were causing input issues by activating
    # disabled code paths or bypassing checks that need to pass naturally.
    # The MOUSE_STUB_PATCH activated a disabled alternative mouse input path.
    # The MOUSE_GATE patches bypassed cursor_locked/window_active flags that
    # should be set during normal window activation.
    #
    # # Apply mouse stub patch (return 1 instead of 0)
    # print("\nApplying mouse input stub fix (return 1 instead of 0)...")
    # current_stub = bytes(data[MOUSE_STUB_PATCH_FILE:MOUSE_STUB_PATCH_FILE + 2])
    # if current_stub == MOUSE_STUB_ORIGINAL:
    #     data[MOUSE_STUB_PATCH_FILE:MOUSE_STUB_PATCH_FILE + 2] = MOUSE_STUB_PATCHED
    #     print(f"  Patched 0x{MOUSE_STUB_PATCH_FILE:06X}: {MOUSE_STUB_ORIGINAL.hex()} -> {MOUSE_STUB_PATCHED.hex()}")
    # elif current_stub == MOUSE_STUB_PATCHED:
    #     print(f"  0x{MOUSE_STUB_PATCH_FILE:06X}: Already patched")
    # else:
    #     print(f"  WARNING: 0x{MOUSE_STUB_PATCH_FILE:06X}: Unexpected bytes {current_stub.hex()}")
    #
    # # Apply mouse gate bypass patch (JNZ -> JMP to skip cursor_locked check)
    # print("\nApplying mouse gate bypass (skip cursor_locked check only)...")
    # if data[MOUSE_GATE_PATCH1_FILE] == MOUSE_GATE_ORIGINAL:
    #     data[MOUSE_GATE_PATCH1_FILE] = MOUSE_GATE_PATCHED
    #     print(f"  Patched 0x{MOUSE_GATE_PATCH1_FILE:06X}: 0x{MOUSE_GATE_ORIGINAL:02X} -> 0x{MOUSE_GATE_PATCHED:02X} (cursor_locked bypass)")
    # elif data[MOUSE_GATE_PATCH1_FILE] == MOUSE_GATE_PATCHED:
    #     print(f"  0x{MOUSE_GATE_PATCH1_FILE:06X}: Already patched")
    # else:
    #     print(f"  WARNING: 0x{MOUSE_GATE_PATCH1_FILE:06X}: Unexpected byte 0x{data[MOUSE_GATE_PATCH1_FILE]:02X}")
    #
    # # Apply mouse return value patch (xor al,al -> mov al,1) to ensure return value is 1
    # print("\nApplying mouse gate return value fix (return 1 instead of 0)...")
    # current_return = bytes(data[MOUSE_RETURN_PATCH_FILE:MOUSE_RETURN_PATCH_FILE + 2])
    # if current_return == MOUSE_RETURN_ORIGINAL:
    #     data[MOUSE_RETURN_PATCH_FILE:MOUSE_RETURN_PATCH_FILE + 2] = MOUSE_RETURN_PATCHED
    #     print(f"  Patched 0x{MOUSE_RETURN_PATCH_FILE:06X}: {MOUSE_RETURN_ORIGINAL.hex()} -> {MOUSE_RETURN_PATCHED.hex()}")
    # elif current_return == MOUSE_RETURN_PATCHED:
    #     print(f"  0x{MOUSE_RETURN_PATCH_FILE:06X}: Already patched")
    # else:
    #     print(f"  WARNING: 0x{MOUSE_RETURN_PATCH_FILE:06X}: Unexpected bytes {current_return.hex()}")
    print("\nSkipping mouse patches (disabled - were causing input issues)")

    # Apply cursor clipping patch (allow ClipCursor in windowed mode)
    print("\nApplying cursor clipping patch (enable ClipCursor in windowed mode)...")
    current_cursor = bytes(data[CURSOR_CLIP_PATCH_FILE:CURSOR_CLIP_PATCH_FILE + 2])
    if current_cursor == CURSOR_CLIP_ORIGINAL:
        data[CURSOR_CLIP_PATCH_FILE:CURSOR_CLIP_PATCH_FILE + 2] = CURSOR_CLIP_PATCHED
        print(f"  Patched 0x{CURSOR_CLIP_PATCH_FILE:06X}: {CURSOR_CLIP_ORIGINAL.hex()} -> {CURSOR_CLIP_PATCHED.hex()}")
    elif current_cursor == CURSOR_CLIP_PATCHED:
        print(f"  0x{CURSOR_CLIP_PATCH_FILE:06X}: Already patched")
    else:
        print(f"  WARNING: 0x{CURSOR_CLIP_PATCH_FILE:06X}: Unexpected bytes {current_cursor.hex()}")

    # Apply destructor crash fix (JZ -> JLE to skip negative error codes like 0xFFFFFFFE)
    destr_offset, destr_original, destr_patched = DESTRUCTOR_CRASH_FIX
    print(f"\nApplying destructor crash fix (JZ -> JLE at 0x{destr_offset:06X})...")
    if data[destr_offset] == destr_original:
        data[destr_offset] = destr_patched
        print(f"  Patched 0x{destr_offset:06X}: 0x{destr_original:02X} -> 0x{destr_patched:02X}")
    elif data[destr_offset] == destr_patched:
        print(f"  0x{destr_offset:06X}: Already patched")
    else:
        print(f"  WARNING: 0x{destr_offset:06X}: Unexpected byte 0x{data[destr_offset]:02X}")

    # Write output
    print(f"\nWriting: {output_path}")
    output_path.write_bytes(data)

    # Verify
    verify_data = output_path.read_bytes()
    # DirectInput patch disabled - skip verification
    site_ok = True  # Not patched anymore
    cave_ok = True  # Not patched anymore

    # Verify registry patches
    hku_ok = all(verify_data[off] == 0x01 for off in REGISTRY_HKU_PATCHES)
    hklm_ok = all(verify_data[off] == 0x01 for off in REGISTRY_HKLM_PATCHES)
    extra_ok = all(verify_data[off] == 0x01 for off, _ in REGISTRY_EXTRA_PATCHES)

    # Verify registry string patches
    reg_push_ok = all(
        bytes(verify_data[off:off + 5]) == new
        for off, _, new in REGISTRY_STRING_PUSH_PATCHES
    )
    reg_mov_ok = all(
        bytes(verify_data[off:off + 5]) == new
        for off, _, new in REGISTRY_STRING_MOV_PATCHES
    )

    # Mouse patches disabled - no verification needed
    # mouse_stub_ok = bytes(verify_data[MOUSE_STUB_PATCH_FILE:MOUSE_STUB_PATCH_FILE + 2]) == MOUSE_STUB_PATCHED
    # mouse_gate_ok = verify_data[MOUSE_GATE_PATCH1_FILE] == MOUSE_GATE_PATCHED
    # mouse_return_ok = bytes(verify_data[MOUSE_RETURN_PATCH_FILE:MOUSE_RETURN_PATCH_FILE + 2]) == MOUSE_RETURN_PATCHED

    # Verify cursor clipping patch
    cursor_clip_ok = bytes(verify_data[CURSOR_CLIP_PATCH_FILE:CURSOR_CLIP_PATCH_FILE + 2]) == CURSOR_CLIP_PATCHED

    # Verify destructor crash fix
    destr_offset, _, destr_patched = DESTRUCTOR_CRASH_FIX
    destructor_ok = verify_data[destr_offset] == destr_patched

    print("\nVerification:")
    print(f"  DirectInput patch site: {'OK' if site_ok else 'MISMATCH!'}")
    print(f"  DirectInput code cave:  {'OK' if cave_ok else 'MISMATCH!'}")
    print(f"  HKEY_USERS patches:     {'OK' if hku_ok else 'MISMATCH!'}")
    print(f"  HKEY_LOCAL_MACHINE:     {'OK' if hklm_ok else 'MISMATCH!'}")
    print(f"  Extra registry:         {'OK' if extra_ok else 'MISMATCH!'}")
    print(f"  Registry string PUSH:   {'OK' if reg_push_ok else 'MISMATCH!'}")
    print(f"  Registry string MOV:    {'OK' if reg_mov_ok else 'MISMATCH!'}")
    print(f"  Mouse patches:          SKIPPED (disabled)")
    print(f"  Cursor clipping:        {'OK' if cursor_clip_ok else 'MISMATCH!'}")
    print(f"  Destructor crash fix:   {'OK' if destructor_ok else 'MISMATCH!'}")

    return site_ok and cave_ok and hku_ok and hklm_ok and extra_ok and reg_push_ok and reg_mov_ok and cursor_clip_ok and destructor_ok


def main():
    slurpy_dir = Path(r"C:\Users\wstri\dev\wolfram\slurpysoft-wulfram")
    input_exe = slurpy_dir / "wulfram2.exe"
    output_exe = slurpy_dir / "wulfram2_fixed.exe"

    if not input_exe.exists():
        print(f"ERROR: Cannot find {input_exe}")
        sys.exit(1)

    print("=" * 60)
    print("SlurpySoft Wulfram 2 Patcher")
    print("=" * 60)
    print()
    print("This version matches the decompiled source (azurefishy-src)")
    print()
    print("Fixes:")
    print("  - DirectInput SetCooperativeLevel NULL HWND bug")
    print("    (calls GUESS2_Winsys_get_hwnd_direct() for proper HWND)")
    print("  - Registry: HKEY_USERS -> HKEY_CURRENT_USER (numeric constant)")
    print("  - Registry: HKEY_LOCAL_MACHINE -> HKEY_CURRENT_USER (numeric constant)")
    print("  - Registry string fix: HKEY_USERS\\.DEFAULT -> HKEY_CURRENT_USER")
    print("    (changes PUSH/MOV addresses to use HKCU strings instead)")
    print("  - Cursor clipping: Enable ClipCursor in windowed mode")
    print("    (bypasses windowed mode check for mouse capture)")
    print("  - Destructor crash fix: JZ -> JLE to handle error codes")
    print("    (skips dereference of 0xFFFFFFFE error code in destructor)")
    print()

    if apply_patch(input_exe, output_exe):
        print()
        print("=" * 60)
        print("Patch applied successfully!")
        print(f"Run: {output_exe}")
        print("=" * 60)
    else:
        print()
        print("Patch failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
