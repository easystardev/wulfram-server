#!/usr/bin/env python3
"""
Wulfram 2 Spawn Debug Patcher

Adds debug hooks to investigate spawn crash:
- EIP=0x004416D9 (GUESS2_Render_prepare_frame) - render context NULL
- EIP=0x0052A358 (GUESS3_List_free_nodes_only) - list corruption

Hook targets:
1. TRANSLATION handler (0x0046e980) - log quantizer allocation
2. PLAYER_INFO handler (0x0046d260) - log packet arrival
3. Render prepare frame (0x004416b0) - null check before crash
4. Global variable watchers

Uses OutputDebugString for logging (view with DebugView or x64dbg).
"""

import sys
import struct
from pathlib import Path

# ============================================================================
# Address Map (from azurefishy-src decompilation)
# ============================================================================

# Virtual addresses (VA) - subtract 0x400000 for file offset in standard PE
IMAGE_BASE = 0x00400000

# Functions to hook
VA_TRANSLATION_HANDLER = 0x0046e980    # GUESS3_PacketHandler_TRANSLATION
VA_PLAYER_INFO_HANDLER = 0x0046d260    # GUESS3_PacketHandler_PLAYER_INFO
VA_RENDER_PREPARE = 0x004416b0         # GUESS2_Render_prepare_frame
VA_QUANTIZER_INIT = 0x0046ead0         # GUESS5_ValueQuantizerArray_init

# Import Address Table entries (for calling Windows APIs)
# These need to be found by scanning the IAT or using existing imports
IAT_OUTPUT_DEBUG_STRING = None  # We'll find or add this

# Global variables (from azurefishy-src renamed_variables_map.json)
VA_G_RENDER_CONTEXT = 0x005f43c4           # g_render_context pointer
VA_G_BEHAVIOR_QUANTIZER_ARRAY = 0x00678134 # g_behavior_quantizer_array pointer

# Code cave location (find empty space in the binary)
# Typical PE files have padding between sections we can use
CODE_CAVE_BASE = 0x004EA600  # After existing patches from v4

# ============================================================================
# Debug Hook Templates
# ============================================================================

def va_to_file_offset(va: int) -> int:
    """Convert virtual address to file offset (assumes standard PE layout)."""
    # This is simplified - real implementation should parse PE headers
    # For wulfram2.exe: .text section starts at 0x401000, file offset 0x1000
    if va >= 0x00401000:
        return va - 0x00400000  # Standard layout
    return va

def calc_rel32(from_va: int, to_va: int) -> int:
    """Calculate relative offset for CALL/JMP (from end of instruction)."""
    # rel32 = target - (current + 5)
    offset = to_va - (from_va + 5)
    return offset & 0xFFFFFFFF  # Ensure unsigned 32-bit

# ============================================================================
# Patch: Add null check to render_prepare_frame
# ============================================================================
#
# Original at 0x004416b0:
#   PUSH EBP
#   MOV EBP, ESP
#   ... setup ...
#   MOV EAX, [g_render_context]   ; Load render context
#   ... use EAX ...
#
# We'll add a check: if render_context is NULL, return early

RENDER_PREPARE_PATCH_SITE = va_to_file_offset(VA_RENDER_PREPARE)
RENDER_PREPARE_CAVE_VA = CODE_CAVE_BASE
RENDER_PREPARE_CAVE_FILE = va_to_file_offset(RENDER_PREPARE_CAVE_VA)

# We need to know the address of g_render_context global
# From decompile: it's accessed like: MOV EAX, [0x005E????]
# Let's search for it in the function


def find_render_context_global(data: bytearray, func_offset: int) -> int:
    """
    Return the known g_render_context address (from decompilation).
    Falls back to searching if the known address doesn't work.
    """
    # Known address from decompilation
    return VA_G_RENDER_CONTEXT


# ============================================================================
# Simple MessageBox-based debug (works without IAT patching)
# ============================================================================
#
# Instead of OutputDebugString, we can use INT 3 breakpoints
# or write to a known memory location that we monitor.
#
# Simplest approach: Use INT 3 (0xCC) as breakpoints at key locations
# Then attach x64dbg and see where it breaks.

def create_breakpoint_patches() -> list:
    """
    Create simple INT 3 breakpoint patches at key locations.

    Returns list of (file_offset, original_byte, description) tuples.
    """
    patches = [
        # TRANSLATION handler entry
        (va_to_file_offset(VA_TRANSLATION_HANDLER), "TRANSLATION_HANDLER_ENTRY"),

        # PLAYER_INFO handler entry
        (va_to_file_offset(VA_PLAYER_INFO_HANDLER), "PLAYER_INFO_HANDLER_ENTRY"),

        # Render prepare frame entry
        (va_to_file_offset(VA_RENDER_PREPARE), "RENDER_PREPARE_FRAME_ENTRY"),

        # Crash site (0x004416D9) - inside render_prepare_frame
        (va_to_file_offset(0x004416D9), "CRASH_SITE_RENDER_CLEAR"),
    ]
    return patches


# ============================================================================
# Code Cave: Null check wrapper for render_prepare_frame
# ============================================================================

def build_render_null_check_cave(render_context_addr: int, original_bytes: bytes) -> bytes:
    """
    Build code cave that checks if render_context is NULL before proceeding.

    Layout:
        PUSH EAX                    ; Save
        MOV EAX, [g_render_context]
        TEST EAX, EAX
        JZ skip_to_ret              ; If NULL, skip function
        POP EAX                     ; Restore
        <original prologue bytes>
        JMP back_to_function+N
    skip_to_ret:
        POP EAX
        RET                         ; Early return if NULL
    """
    cave = bytearray()

    # PUSH EAX (save)
    cave.append(0x50)

    # MOV EAX, [g_render_context]
    cave.append(0xA1)
    cave.extend(struct.pack("<I", render_context_addr))

    # TEST EAX, EAX
    cave.extend([0x85, 0xC0])

    # JZ +offset (to skip_to_ret)
    # Calculate offset after we know size
    jz_offset_pos = len(cave)
    cave.extend([0x74, 0x00])  # Placeholder

    # POP EAX (restore)
    cave.append(0x58)

    # Original prologue bytes
    cave.extend(original_bytes)

    # JMP back to function (after our hook point)
    jmp_back_pos = len(cave)
    cave.extend([0xE9, 0x00, 0x00, 0x00, 0x00])  # Placeholder

    # skip_to_ret label
    skip_label = len(cave)

    # Fix JZ offset
    cave[jz_offset_pos + 1] = skip_label - (jz_offset_pos + 2)

    # POP EAX
    cave.append(0x58)

    # RET
    cave.append(0xC3)

    return bytes(cave)


# ============================================================================
# Main Patching Logic
# ============================================================================

def apply_debug_patches(input_path: Path, output_path: Path, mode: str = "breakpoints") -> bool:
    """
    Apply debug patches to wulfram2.exe.

    Modes:
        - "breakpoints": Add INT 3 at key locations (use with x64dbg)
        - "nullcheck": Add null check to prevent render crash
        - "both": Apply both
    """
    print(f"Reading: {input_path}")
    data = bytearray(input_path.read_bytes())

    patches_applied = []

    if mode in ("breakpoints", "both"):
        print("\n=== Adding Debug Breakpoints ===")
        print("Attach x64dbg and run - it will break at these locations:")

        bp_patches = create_breakpoint_patches()
        for offset, desc in bp_patches:
            if offset < len(data):
                original = data[offset]
                # Don't patch if already INT 3
                if original != 0xCC:
                    data[offset] = 0xCC
                    patches_applied.append((offset, original, desc))
                    print(f"  BP at 0x{offset:06X} (VA 0x{offset + IMAGE_BASE:08X}): {desc}")
                    print(f"       Original byte: 0x{original:02X}")
                else:
                    print(f"  BP at 0x{offset:06X}: already has INT 3")

    if mode in ("nullcheck", "both"):
        print("\n=== Adding Null Check to Render Prepare ===")

        # Find the render context global address
        func_offset = va_to_file_offset(VA_RENDER_PREPARE)
        render_ctx = find_render_context_global(data, func_offset)

        if render_ctx:
            print(f"  Found g_render_context at VA 0x{render_ctx:08X}")

            # Get original bytes at function entry (we'll copy and replace)
            original_prologue = bytes(data[func_offset:func_offset + 5])
            print(f"  Original prologue: {original_prologue.hex()}")

            # Build code cave
            cave_code = build_render_null_check_cave(render_ctx, original_prologue)

            # Calculate JMP offset in cave (back to function)
            # The JMP is at cave_code offset, target is func_offset + 5
            cave_va = RENDER_PREPARE_CAVE_VA
            # Find the JMP instruction in cave (search for E9 pattern)
            for i in range(len(cave_code) - 5):
                if cave_code[i] == 0xE9:
                    jmp_from_va = cave_va + i + 5  # After JMP instruction
                    jmp_to_va = VA_RENDER_PREPARE + 5  # After our hook
                    rel = calc_rel32(cave_va + i, jmp_to_va)
                    cave_code = cave_code[:i+1] + struct.pack("<I", rel) + cave_code[i+5:]
                    break

            # Write cave
            cave_offset = va_to_file_offset(cave_va)
            if cave_offset + len(cave_code) <= len(data):
                data[cave_offset:cave_offset + len(cave_code)] = cave_code
                print(f"  Wrote {len(cave_code)} byte cave at 0x{cave_offset:06X}")

                # Patch function entry to JMP to cave
                jmp_to_cave = b'\xE9' + struct.pack("<I", calc_rel32(VA_RENDER_PREPARE, cave_va))
                data[func_offset:func_offset + 5] = jmp_to_cave
                print(f"  Patched function entry to JMP to cave")
                patches_applied.append((func_offset, original_prologue, "RENDER_NULL_CHECK"))
            else:
                print(f"  ERROR: Cave location 0x{cave_offset:06X} out of bounds")
        else:
            print("  WARNING: Could not find g_render_context address")

    # Write output
    print(f"\nWriting: {output_path}")
    output_path.write_bytes(data)

    # Save patch info for later restoration
    if patches_applied:
        info_path = output_path.with_suffix('.patches.txt')
        with open(info_path, 'w') as f:
            f.write("# Spawn Debug Patches Applied\n")
            f.write("# Format: file_offset, original_byte(s), description\n\n")
            for offset, orig, desc in patches_applied:
                if isinstance(orig, int):
                    f.write(f"0x{offset:06X}, 0x{orig:02X}, {desc}\n")
                else:
                    f.write(f"0x{offset:06X}, {orig.hex()}, {desc}\n")
        print(f"Patch info saved to: {info_path}")

    return True


def restore_patches(input_path: Path, patches_file: Path, output_path: Path) -> bool:
    """Restore original bytes from patch info file."""
    print(f"Reading: {input_path}")
    data = bytearray(input_path.read_bytes())

    print(f"Reading patches from: {patches_file}")
    with open(patches_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                offset = int(parts[0].strip(), 16)
                orig_hex = parts[1].strip()
                if orig_hex.startswith('0x') and len(orig_hex) == 4:
                    # Single byte
                    data[offset] = int(orig_hex, 16)
                else:
                    # Multiple bytes
                    orig_bytes = bytes.fromhex(orig_hex)
                    data[offset:offset + len(orig_bytes)] = orig_bytes
                print(f"  Restored 0x{offset:06X}")

    print(f"Writing: {output_path}")
    output_path.write_bytes(data)
    return True


def show_function_disasm(input_path: Path, va: int, count: int = 32):
    """Show bytes at a function for manual analysis."""
    data = input_path.read_bytes()
    offset = va_to_file_offset(va)

    print(f"\nBytes at VA 0x{va:08X} (file offset 0x{offset:06X}):")
    for i in range(0, min(count, len(data) - offset), 16):
        addr = va + i
        chunk = data[offset + i:offset + i + 16]
        hex_str = ' '.join(f'{b:02X}' for b in chunk)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"  {addr:08X}  {hex_str:<48}  {ascii_str}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Wulfram 2 Spawn Debug Patcher")
    parser.add_argument("action", choices=["patch", "restore", "disasm", "info"],
                       help="Action to perform")
    parser.add_argument("--mode", choices=["breakpoints", "nullcheck", "both"],
                       default="breakpoints",
                       help="Patch mode (default: breakpoints)")
    parser.add_argument("--input", type=Path,
                       default=Path(r"C:\Users\wstri\dev\wolfram\wulfram2-game\wulfram2_fixed.exe"),
                       help="Input executable")
    parser.add_argument("--output", type=Path,
                       default=Path(r"C:\Users\wstri\dev\wolfram\wulfram2-game\wulfram2_debug.exe"),
                       help="Output executable")
    parser.add_argument("--va", type=lambda x: int(x, 0),
                       help="Virtual address for disasm")

    args = parser.parse_args()

    print("=" * 60)
    print("Wulfram 2 Spawn Debug Patcher")
    print("=" * 60)

    if args.action == "patch":
        print(f"\nMode: {args.mode}")
        apply_debug_patches(args.input, args.output, args.mode)

        print("\n" + "=" * 60)
        print("To debug:")
        print("1. Open wulfram2_debug.exe in x64dbg")
        print("2. Run the game - it will break at debug points")
        print("3. Check register/memory state at each break")
        print("4. Press F9 to continue to next breakpoint")
        print("=" * 60)

    elif args.action == "restore":
        patches_file = args.output.with_suffix('.patches.txt')
        if patches_file.exists():
            restore_patches(args.output, patches_file, args.output)
        else:
            print(f"ERROR: Patches file not found: {patches_file}")

    elif args.action == "disasm":
        va = args.va or VA_RENDER_PREPARE
        show_function_disasm(args.input, va, 64)

    elif args.action == "info":
        print("\nKey Addresses:")
        print(f"  TRANSLATION handler:  VA 0x{VA_TRANSLATION_HANDLER:08X}")
        print(f"  PLAYER_INFO handler:  VA 0x{VA_PLAYER_INFO_HANDLER:08X}")
        print(f"  Render prepare:       VA 0x{VA_RENDER_PREPARE:08X}")
        print(f"  Crash site:           VA 0x004416D9")
        print(f"  Quantizer init:       VA 0x{VA_QUANTIZER_INIT:08X}")


if __name__ == "__main__":
    main()
