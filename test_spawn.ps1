param(
    [int]$MaxWaitSeconds = 60,
    [switch]$Launch,
    [switch]$KillExisting,
    [int]$TeamClickX = 200,   # X coordinate to click for team (top-left)
    [int]$TeamClickY = 150,   # Y coordinate to click for team
    [switch]$SkipInput,
    [switch]$LateInput,
    [switch]$Verbose,
    [switch]$Screenshot,      # Capture and analyze screenshots
    [string]$GamePath = "",   # Path to wulfram2_fixed.exe or launch.bat
    [string]$LogPath = ""     # Path to server.log
)

Set-StrictMode -Version Latest

$ProcessName = "wulfram2_fixed"

# Default paths - can be overridden via parameters or env vars
if ($GamePath -eq "") {
    $GamePath = if ($env:WULFRAM_GAME_PATH) { $env:WULFRAM_GAME_PATH } else { "wulfram2_fixed.exe" }
}
$LaunchPath = $GamePath

if ($LogPath -eq "") {
    $LogPath = if ($env:WULFRAM_LOG_PATH) { $env:WULFRAM_LOG_PATH } else { "server.log" }
}
$ServerLogPath = $LogPath

# Default behavior: if no switches were provided, auto launch and kill existing.
if (-not $PSBoundParameters.ContainsKey('Launch')) {
    $Launch = $true
}
if (-not $PSBoundParameters.ContainsKey('KillExisting')) {
    $KillExisting = $true
}

# Track log position
$logStart = 0
if (Test-Path $ServerLogPath) {
    $logStart = (Get-Item $ServerLogPath).Length
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class NativeWin {
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

    [DllImport("user32.dll")]
    public static extern void mouse_event(uint dwFlags, int dx, int dy, uint dwData, int dwExtraInfo);

    [DllImport("user32.dll")]
    public static extern bool SetCursorPos(int X, int Y);

    [DllImport("user32.dll")]
    public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);

    [DllImport("user32.dll")]
    public static extern uint MapVirtualKey(uint uCode, uint uMapType);

    public const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
    public const uint MOUSEEVENTF_LEFTUP = 0x0004;
    public const uint MOUSEEVENTF_MOVE = 0x0001;
    public const uint KEYEVENTF_EXTENDEDKEY = 0x0001;
    public const uint KEYEVENTF_KEYUP = 0x0002;
    public const uint KEYEVENTF_SCANCODE = 0x0008;

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left, Top, Right, Bottom;
    }
}
"@

$CapturesDir = Join-Path $PSScriptRoot "captures"
if (-not (Test-Path $CapturesDir)) {
    New-Item -ItemType Directory -Path $CapturesDir -Force | Out-Null
}

function Capture-Window([IntPtr]$hwnd, [string]$filename) {
    $rect = New-Object NativeWin+RECT
    [NativeWin]::GetWindowRect($hwnd, [ref]$rect) | Out-Null
    $width = $rect.Right - $rect.Left
    $height = $rect.Bottom - $rect.Top

    $bmp = New-Object System.Drawing.Bitmap($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bmp)
    $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, (New-Object System.Drawing.Size($width, $height)))
    $graphics.Dispose()

    $filepath = Join-Path $CapturesDir $filename
    $bmp.Save($filepath, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    return $filepath
}

function Test-RedScreen([string]$imagePath, [float]$threshold = 0.15) {
    # Check for RED SCREEN OVERLAY (damage indicator) on terrain/ground area
    # Avoid the purple/pink sky by sampling BOTTOM HALF of screen only
    $bmp = [System.Drawing.Bitmap]::FromFile($imagePath)

    # Sample bottom half of screen (terrain, not sky)
    $sampleWidth = [int]($bmp.Width * 0.6)
    $sampleHeight = [int]($bmp.Height * 0.35)
    $startX = [int](($bmp.Width - $sampleWidth) / 2)
    $startY = [int]($bmp.Height * 0.55)  # Start at 55% down

    $redTintPixels = 0
    $totalPixels = 0
    $step = 4

    for ($y = $startY; $y -lt [Math]::Min($startY + $sampleHeight, $bmp.Height); $y += $step) {
        for ($x = $startX; $x -lt [Math]::Min($startX + $sampleWidth, $bmp.Width); $x += $step) {
            $pixel = $bmp.GetPixel($x, $y)
            $totalPixels++
            # Red screen overlay: RED is dominant and significantly higher than green
            # Normal terrain is brown (R~G) or green (G>R), red overlay makes R >> G
            if ($pixel.R -gt 120 -and $pixel.R -gt ($pixel.G * 1.5) -and $pixel.R -gt ($pixel.B + 40)) {
                $redTintPixels++
            }
        }
    }
    $bmp.Dispose()

    $redRatio = if ($totalPixels -gt 0) { $redTintPixels / $totalPixels } else { 0 }

    return @{
        IsRed = ($redRatio -gt $threshold)
        RedRatio = $redRatio
        RedPixels = $redTintPixels
        TotalPixels = $totalPixels
    }
}

function Test-ViewpointChanges([int64]$startLen) {
    # Check server log for VIEWPOINT updates (indicates rotation/turning works)
    $text = Get-LogDelta -startLen $startLen
    $matches = [regex]::Matches($text, '\[VIEWPOINT #\d+\].*?yaw=(-?[\d.]+)')
    if ($matches.Count -ge 2) {
        $yaw1 = [float]$matches[0].Groups[1].Value
        $yaw2 = [float]$matches[$matches.Count - 1].Groups[1].Value
        $delta = [math]::Abs($yaw2 - $yaw1)
        return @{
            HasChanges = ($delta -gt 5.0)  # More than 5 degrees change
            YawDelta = $delta
            Updates = $matches.Count
        }
    }
    return @{
        HasChanges = $false
        YawDelta = 0
        Updates = $matches.Count
    }
}

function Test-MovementChanges([int64]$startLen) {
    # Check server log for position changes in TICK entries
    $text = Get-LogDelta -startLen $startLen
    $matches = [regex]::Matches($text, '\[TICK\].*?pos=\((-?[\d.]+),(-?[\d.]+),(-?[\d.]+)\)')
    if ($matches.Count -ge 2) {
        $x1 = [float]$matches[0].Groups[1].Value
        $y1 = [float]$matches[0].Groups[2].Value
        $x2 = [float]$matches[$matches.Count - 1].Groups[1].Value
        $y2 = [float]$matches[$matches.Count - 1].Groups[2].Value
        $deltaX = [math]::Abs($x2 - $x1)
        $deltaY = [math]::Abs($y2 - $y1)
        $totalDelta = [math]::Sqrt($deltaX * $deltaX + $deltaY * $deltaY)
        return @{
            HasMovement = ($totalDelta -gt 0.5)  # More than 0.5 units moved
            Delta = $totalDelta
            StartPos = "($x1,$y1)"
            EndPos = "($x2,$y2)"
            Updates = $matches.Count
        }
    }
    return @{
        HasMovement = $false
        Delta = 0
        StartPos = "unknown"
        EndPos = "unknown"
        Updates = $matches.Count
    }
}

function Test-InputReceived([int64]$startLen) {
    # Check if ACTION_DUMP has non-zero input bits
    $text = Get-LogDelta -startLen $startLen
    # Look for ACTION_DUMP with input_bits that aren't all zeros
    # Format: ACTION_DUMP received: len=23 data=0900017940000000000000000000003ff2800000000000
    # The input bits start after frame number (bytes 1-4), so byte 5 onward
    $matches = [regex]::Matches($text, 'ACTION_DUMP received:.*?data=09[0-9a-f]{8}([0-9a-f]{8})')
    $nonZeroCount = 0
    foreach ($m in $matches) {
        $inputBits = $m.Groups[1].Value
        if ($inputBits -ne "00000000") {
            $nonZeroCount++
        }
    }
    return @{
        HasInput = ($nonZeroCount -gt 0)
        NonZeroPackets = $nonZeroCount
        TotalPackets = $matches.Count
    }
}

function Test-ViewpointPackets([int64]$startLen) {
    # Check for 0x35 VIEWPOINT_INFO packets (even in 0x10 wrappers)
    $text = Get-LogDelta -startLen $startLen
    $raw35 = [regex]::Matches($text, 'Datagram with 0x35')
    return @{
        Count = $raw35.Count
        HasPackets = ($raw35.Count -gt 0)
    }
}

function Test-RotationInput([int64]$startLen, [float]$threshold = 0.05) {
    # Check ACTION_DUMP logs for non-zero turn or slot6/slot7 values (mouse look candidate).
    $text = Get-LogDelta -startLen $startLen
    $matches = [regex]::Matches($text, 'turn: raw=\d+ val=([-0-9.]+).*?s6: raw=\d+ val=([-0-9.]+).*?s7: raw=\d+ val=([-0-9.]+)')
    $nonZero = 0
    foreach ($m in $matches) {
        $turn = [math]::Abs([float]$m.Groups[1].Value)
        $s6 = [math]::Abs([float]$m.Groups[2].Value)
        $s7 = [math]::Abs([float]$m.Groups[3].Value)
        if ($turn -gt $threshold -or $s6 -gt $threshold -or $s7 -gt $threshold) {
            $nonZero++
        }
    }
    return @{
        HasRotationInput = ($nonZero -gt 0)
        NonZeroPackets = $nonZero
        TotalPackets = $matches.Count
    }
}

function Test-ProjectileAlignment([int64]$startLen, [float]$expectedOffset = 2.0, [float]$maxError = 6.0, [float]$minForwardCos = 0.4) {
    # Verify projectile spawn is roughly in front of player by expected offset.
    # Prefer PROJ-AIM logs (actual aim yaw/fwd), fall back to WEAPON-FIRE yaw.
    $text = Get-LogDelta -startLen $startLen
    $aimPlayerMatches = [regex]::Matches($text, '\[PROJ-AIM\] player_srv=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\) player_cli=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\) proj_srv=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\) proj_cli=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\)')
    $aimUsedMatches = [regex]::Matches($text, '\[PROJ-AIM\] aim_used yaw=([-0-9.]+) pitch=([-0-9.]+) fwd=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\).*spawn_offset=([-0-9.]+)')

    if ($aimPlayerMatches.Count -gt 0 -and $aimUsedMatches.Count -gt 0) {
        $p = $aimPlayerMatches[$aimPlayerMatches.Count - 1]
        $a = $aimUsedMatches[$aimUsedMatches.Count - 1]

        $playerCX = [float]$p.Groups[4].Value
        $playerCY = [float]$p.Groups[5].Value
        $playerCZ = [float]$p.Groups[6].Value
        $spawnX = [float]$p.Groups[10].Value
        $spawnY = [float]$p.Groups[11].Value
        $spawnZ = [float]$p.Groups[12].Value

        $fwdX = [float]$a.Groups[3].Value
        $fwdY = [float]$a.Groups[4].Value
        $fwdZ = [float]$a.Groups[5].Value
        $spawnOffset = [float]$a.Groups[6].Value

        $expX = $playerCX + ($spawnOffset * $fwdX)
        $expY = $playerCY + ($spawnOffset * $fwdY)
        $expZ = $playerCZ + ($spawnOffset * $fwdZ)

        $dx = $spawnX - $playerCX
        $dy = $spawnY - $playerCY
        $dz = $spawnZ - $playerCZ
        $dist = [math]::Sqrt(($dx * $dx) + ($dy * $dy) + ($dz * $dz))
        $dot = ($dx * $fwdX) + ($dy * $fwdY) + ($dz * $fwdZ)
        $forwardCos = if ($dist -gt 0) { $dot / $dist } else { 0.0 }

        $errX = $spawnX - $expX
        $errY = $spawnY - $expY
        $errZ = $spawnZ - $expZ
        $posError = [math]::Sqrt(($errX * $errX) + ($errY * $errY) + ($errZ * $errZ))

        $aligned = ($posError -le $maxError) -and ($forwardCos -ge $minForwardCos)

        return @{
            HasSpawn = $true
            IsAligned = $aligned
            Mode = "aim"
            PlayerPos = "($playerCX,$playerCY,$playerCZ)"
            SpawnPos = "($spawnX,$spawnY,$spawnZ)"
            ExpectedPos = "($expX,$expY,$expZ)"
            Distance = $dist
            PosError = $posError
            ForwardCos = $forwardCos
        }
    }

    $fireMatches = [regex]::Matches($text, '\[WEAPON-FIRE\] Firing \d+ proj at yaw=([-0-9.]+) pos=\(([-0-9.]+),\s*([-0-9.]+),\s*([-0-9.]+)\)')
    $spawnMatches = [regex]::Matches($text, 'FIRE! pos=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\)')

    if ($fireMatches.Count -eq 0 -or $spawnMatches.Count -eq 0) {
        return @{
            HasSpawn = $false
            IsAligned = $false
            Reason = "Missing fire/spawn log lines"
        }
    }

    $fire = $fireMatches[$fireMatches.Count - 1]
    $spawn = $spawnMatches[$spawnMatches.Count - 1]

    $yawDeg = [float]$fire.Groups[1].Value
    $playerX = [float]$fire.Groups[2].Value
    $playerY = [float]$fire.Groups[3].Value
    $playerZ = [float]$fire.Groups[4].Value

    $spawnX = [float]$spawn.Groups[1].Value
    $spawnY = [float]$spawn.Groups[2].Value
    $spawnZ = [float]$spawn.Groups[3].Value

    $yawRad = $yawDeg * [math]::PI / 180.0
    $fwdX = [math]::Cos($yawRad)
    $fwdY = [math]::Sin($yawRad)
    $fwdZ = 0.0

    # Projectiles are spawned in server-space coordinates.
    # Apply pos offset only if explicitly configured.
    $upAxis = if ($env:WULFRAM_UP_AXIS) { $env:WULFRAM_UP_AXIS.ToLower() } else { "z" }
    if ($upAxis -ne "y" -and $upAxis -ne "z") { $upAxis = "z" }
    if ($upAxis -eq "z") {
        $posOffset = if ($env:WULFRAM_POS_OFFSET_Z) { [float]$env:WULFRAM_POS_OFFSET_Z } else { 0.0 }
    } else {
        $posOffset = if ($env:WULFRAM_POS_OFFSET_Y) { [float]$env:WULFRAM_POS_OFFSET_Y } else { 0.0 }
    }

    $playerCX = $playerX
    $playerCY = $playerY
    $playerCZ = $playerZ
    if ($upAxis -eq "z") {
        $playerCZ += $posOffset
    } else {
        $playerCY += $posOffset
    }

    $expX = $playerCX + ($expectedOffset * $fwdX)
    $expY = $playerCY + ($expectedOffset * $fwdY)
    $expZ = $playerCZ + ($expectedOffset * $fwdZ)

    $dx = $spawnX - $playerCX
    $dy = $spawnY - $playerCY
    $dz = $spawnZ - $playerCZ
    $dist = [math]::Sqrt(($dx * $dx) + ($dy * $dy) + ($dz * $dz))
    $dot = ($dx * $fwdX) + ($dy * $fwdY) + ($dz * $fwdZ)
    $forwardCos = if ($dist -gt 0) { $dot / $dist } else { 0.0 }

    $errX = $spawnX - $expX
    $errY = $spawnY - $expY
    $errZ = $spawnZ - $expZ
    $posError = [math]::Sqrt(($errX * $errX) + ($errY * $errY) + ($errZ * $errZ))

    $aligned = ($posError -le $maxError) -and ($forwardCos -ge $minForwardCos)

    return @{
        HasSpawn = $true
        IsAligned = $aligned
        Mode = "body"
        YawDeg = $yawDeg
        PlayerPos = "($playerCX,$playerCY,$playerCZ)"
        SpawnPos = "($spawnX,$spawnY,$spawnZ)"
        ExpectedPos = "($expX,$expY,$expZ)"
        Distance = $dist
        PosError = $posError
        ForwardCos = $forwardCos
    }
}

function Test-ProjectileAimSource([int64]$startLen) {
    # Check if projectiles used viewpoint vs input for aim source.
    $text = Get-LogDelta -startLen $startLen
    $srcMatches = [regex]::Matches($text, '\[PROJ-AIM\].*src=([a-z]+)')
    $viewpoint = 0
    $input = 0
    foreach ($m in $srcMatches) {
        $src = $m.Groups[1].Value
        if ($src -eq "viewpoint") { $viewpoint++ }
        elseif ($src -eq "input") { $input++ }
    }
    return @{
        Viewpoint = $viewpoint
        Input = $input
        Total = $srcMatches.Count
    }
}

function Send-Enter([IntPtr]$hwnd) {
    [NativeWin]::SetForegroundWindow($hwnd) | Out-Null
    Start-Sleep -Milliseconds 100
    Send-KeyTap "ENTER"
}

function Send-KeyHold([string]$keyName, [int]$holdMs = 500) {
    Send-KeyEvent $keyName $false
    Start-Sleep -Milliseconds $holdMs
    Send-KeyEvent $keyName $true
}

function Send-KeyTap([string]$keyName, [int]$delayMs = 30) {
    Send-KeyEvent $keyName $false
    Start-Sleep -Milliseconds $delayMs
    Send-KeyEvent $keyName $true
}

function Send-KeyEvent([string]$keyName, [bool]$keyUp) {
    $vk = Get-VirtualKey $keyName
    if ($vk -eq $null) {
        return
    }
    $scan = [byte][NativeWin]::MapVirtualKey([uint32]$vk, 0)
    $flags = [uint32][NativeWin]::KEYEVENTF_SCANCODE
    if ($keyUp) {
        $flags = $flags -bor [NativeWin]::KEYEVENTF_KEYUP
    }
    if (Is-ExtendedKey $keyName) {
        $flags = $flags -bor [NativeWin]::KEYEVENTF_EXTENDEDKEY
    }
    [NativeWin]::keybd_event([byte]$vk, $scan, $flags, [UIntPtr]::Zero)
}

function Get-VirtualKey([string]$keyName) {
    $map = @{
        "W" = 0x57
        "A" = 0x41
        "S" = 0x53
        "D" = 0x44
        "UP" = 0x26
        "DOWN" = 0x28
        "LEFT" = 0x25
        "RIGHT" = 0x27
        "SPACE" = 0x20
        "ENTER" = 0x0D
        "F" = 0x46
    }
    if ($map.ContainsKey($keyName)) {
        return $map[$keyName]
    }
    return $null
}

function Is-ExtendedKey([string]$keyName) {
    return @("UP", "DOWN", "LEFT", "RIGHT") -contains $keyName
}

function Send-TestInputBurst([IntPtr]$hwnd) {
    [NativeWin]::SetForegroundWindow($hwnd) | Out-Null
    Start-Sleep -Milliseconds 100

    # Click center to ensure game has focus before key holds.
    $rect = New-Object NativeWin+RECT
    [NativeWin]::GetWindowRect($hwnd, [ref]$rect) | Out-Null
    $centerX = [int](($rect.Left + $rect.Right) / 2)
    $centerY = [int](($rect.Top + $rect.Bottom) / 2)
    [NativeWin]::SetCursorPos($centerX, $centerY) | Out-Null
    Start-Sleep -Milliseconds 50
    [NativeWin]::mouse_event([NativeWin]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 50
    [NativeWin]::mouse_event([NativeWin]::MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 100

    # Hold keys long enough to show up in ACTION_DUMP.
    Send-KeyHold "W" 600
    Start-Sleep -Milliseconds 150
    Send-KeyHold "A" 400
    Start-Sleep -Milliseconds 150
    Send-KeyHold "D" 400
    Start-Sleep -Milliseconds 150

    # Tank keymap: arrow keys for forward/back and turning.
    Send-KeyHold "UP" 700
    Start-Sleep -Milliseconds 150
    Send-KeyHold "LEFT" 400
    Start-Sleep -Milliseconds 150
    Send-KeyHold "RIGHT" 400

    # Mouse movement to test rotation/viewpoint (relative motion)
    $rect = New-Object NativeWin+RECT
    [NativeWin]::GetWindowRect($hwnd, [ref]$rect) | Out-Null
    $centerX = [int](($rect.Left + $rect.Right) / 2)
    $centerY = [int](($rect.Top + $rect.Bottom) / 2)

    # Move mouse in larger sweeps (absolute) then force relative motion
    [NativeWin]::SetCursorPos($centerX, $centerY) | Out-Null
    Start-Sleep -Milliseconds 100
    [NativeWin]::SetCursorPos($centerX - 200, $centerY) | Out-Null
    Start-Sleep -Milliseconds 300
    [NativeWin]::SetCursorPos($centerX + 200, $centerY) | Out-Null
    Start-Sleep -Milliseconds 300
    [NativeWin]::SetCursorPos($centerX, $centerY - 100) | Out-Null
    Start-Sleep -Milliseconds 200
    Move-MouseRelative -dx 80 -dy 0 -repeat 4 -delayMs 15
    Move-MouseRelative -dx -80 -dy 0 -repeat 4 -delayMs 15
    Move-MouseRelative -dx 0 -dy 60 -repeat 3 -delayMs 15
    Move-MouseRelative -dx 0 -dy -60 -repeat 3 -delayMs 15

    # Wait briefly for VIEWPOINT packets to arrive before firing
    $vpDeadline = (Get-Date).AddSeconds(2)
    while ((Get-Date) -lt $vpDeadline) {
        if (Test-LogPatternRecent "\\[VIEWPOINT #") { break }
        Start-Sleep -Milliseconds 200
    }

    # Space to fire multiple times
    Fire-Primary $hwnd
    Start-Sleep -Milliseconds 200
    Fire-Primary $hwnd
    Start-Sleep -Milliseconds 150
}

function Click-Window([IntPtr]$hwnd, [int]$relX, [int]$relY, [switch]$ShowInfo) {
    $rect = New-Object NativeWin+RECT
    [NativeWin]::GetWindowRect($hwnd, [ref]$rect) | Out-Null

    $winW = $rect.Right - $rect.Left
    $winH = $rect.Bottom - $rect.Top
    $absX = $rect.Left + $relX
    $absY = $rect.Top + $relY

    if ($ShowInfo) {
        Write-Host "  Window: pos=($($rect.Left),$($rect.Top)) size=${winW}x${winH}"
        Write-Host "  Click: rel=($relX,$relY) abs=($absX,$absY)"
    }

    [NativeWin]::SetForegroundWindow($hwnd) | Out-Null
    Start-Sleep -Milliseconds 100
    [NativeWin]::SetCursorPos($absX, $absY) | Out-Null
    Start-Sleep -Milliseconds 100
    [NativeWin]::mouse_event([NativeWin]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 100
    [NativeWin]::mouse_event([NativeWin]::MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
}

function Fire-Primary([IntPtr]$hwnd) {
    # Send a primary fire pulse via mouse click + space as fallback.
    $rect = New-Object NativeWin+RECT
    [NativeWin]::GetWindowRect($hwnd, [ref]$rect) | Out-Null
    $centerX = [int](($rect.Left + $rect.Right) / 2)
    $centerY = [int](($rect.Top + $rect.Bottom) / 2)

    [NativeWin]::SetForegroundWindow($hwnd) | Out-Null
    Start-Sleep -Milliseconds 100
    [NativeWin]::SetCursorPos($centerX, $centerY) | Out-Null
    Start-Sleep -Milliseconds 50
    [NativeWin]::mouse_event([NativeWin]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 30
    [NativeWin]::mouse_event([NativeWin]::MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    # Also send space in case fire is bound to keyboard.
    Start-Sleep -Milliseconds 50
    Send-KeyTap "SPACE"
}

function Move-MouseRelative([int]$dx, [int]$dy, [int]$repeat = 1, [int]$delayMs = 10) {
    for ($i = 0; $i -lt $repeat; $i++) {
        [NativeWin]::mouse_event([NativeWin]::MOUSEEVENTF_MOVE, $dx, $dy, 0, 0)
        Start-Sleep -Milliseconds $delayMs
    }
}

function Get-LogDelta([int64]$startLen) {
    if (-not (Test-Path $ServerLogPath)) { return "" }
    try {
        $fs = [System.IO.File]::Open($ServerLogPath, 'Open', 'Read', 'ReadWrite')
        $bytes = New-Object byte[] $fs.Length
        $fs.Read($bytes, 0, $bytes.Length) | Out-Null
        $fs.Close()
        # If log was truncated (server restart), read from beginning
        if ($bytes.Length -lt $startLen) { $startLen = 0 }
        if ($bytes.Length -le $startLen) { return "" }
        return [System.Text.Encoding]::ASCII.GetString($bytes, $startLen, $bytes.Length - $startLen)
    } catch { return "" }
}

function Test-LogContains([int64]$startLen, [string[]]$needles) {
    $text = Get-LogDelta -startLen $startLen
    foreach ($n in $needles) {
        if ($text.Contains($n)) { return $true }
    }
    return $false
}

function Test-LogPatternRecent([string]$pattern, [int]$tailLines = 100) {
    if (-not (Test-Path $ServerLogPath)) { return $false }
    try {
        $lines = Get-Content $ServerLogPath -Tail $tailLines -ErrorAction SilentlyContinue
        foreach ($line in $lines) {
            if ($line -match $pattern) { return $true }
        }
    } catch {}
    return $false
}

function Test-Desync([int64]$startLen) {
    # Check for desync warnings in server log
    $text = Get-LogDelta -startLen $startLen
    $result = @{
        HasDesync = $false
        DesyncCount = 0
        Details = @()
    }

    # Look for DESYNC markers
    $matches = [regex]::Matches($text, '\[DESYNC\][^\n]+')
    if ($matches.Count -gt 0) {
        $result.HasDesync = $true
        $result.DesyncCount = $matches.Count
        foreach ($m in $matches) {
            $result.Details += $m.Value
        }
    }

    return $result
}

# Kill existing
if ($KillExisting) {
    $toKill = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like "wulfram2*" }
    if ($toKill) {
        Write-Host "Stopping existing processes..."
        $toKill | Stop-Process -Force
        Start-Sleep -Milliseconds 500
    }
}

# Launch game
$proc = $null
if ($Launch) {
    $launchFull = Join-Path $PSScriptRoot $LaunchPath
    $launchDir = Split-Path -Path $launchFull -Parent
    Write-Host "Launching: $launchFull"
    Start-Process -FilePath $launchFull -WorkingDirectory $launchDir | Out-Null
}

# Wait for window
Write-Host "Waiting for game window..."
$deadline = (Get-Date).AddSeconds($MaxWaitSeconds)
while ((Get-Date) -lt $deadline) {
    $candidates = Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -like "$ProcessName*" -and $_.MainWindowHandle -ne 0
    }
    if ($candidates) {
        $proc = $candidates | Select-Object -First 1
        break
    }
    Start-Sleep -Milliseconds 200
}

if (-not $proc) {
    Write-Host "FAIL: Game window not found"
    exit 1
}

Write-Host "Found game window (PID: $($proc.Id))"
$hwnd = $proc.MainWindowHandle

# Focus window and wait for game to load
[void][NativeWin]::ShowWindow($hwnd, 9)
[void][NativeWin]::SetForegroundWindow($hwnd)
Write-Host "Waiting for game to initialize (8 seconds)..."
Start-Sleep -Milliseconds 8000

# Step 1: Press Enter for handle
Write-Host "Step 1: Pressing Enter (confirm handle)..."
Send-Enter $hwnd
Start-Sleep -Milliseconds 2000

# Step 2: Click team (our server doesn't require password like wulf-forge)
Write-Host "Step 2: Clicking team at ($TeamClickX, $TeamClickY)..."
Click-Window $hwnd $TeamClickX $TeamClickY -ShowInfo:$Verbose
Start-Sleep -Milliseconds 1000

# Double-click in case first didn't register
Click-Window $hwnd $TeamClickX $TeamClickY
Start-Sleep -Milliseconds 500

# Step 3: Wait for initialization and auto-spawn
Write-Host "Step 3: Waiting 10 seconds for client init and auto-spawn..."
Start-Sleep -Seconds 10

# Monitor for spawn/crash
Write-Host "Monitoring for spawn/crash..."
$spawnNeedles = @("Auto-spawn (WF) on team", "[SPAWN]", "UDP SEND] TANK", "Packet 0x09 from")
$deadline = (Get-Date).AddSeconds($MaxWaitSeconds)
$spawnDetected = $false
$spawnTime = $null
$inputSent = $false
$lateInputSent = $false
$successAfter = if ($LateInput) { 12 } else { 6 }

while ((Get-Date) -lt $deadline) {
    # Check if process crashed
    try {
        $proc.Refresh()
        if ($proc.HasExited) {
            $exitCode = $proc.ExitCode
            if ($spawnDetected -or (Test-LogPatternRecent "Packet 0x09 from|Packet 0x0B from")) {
                Write-Host "SPAWN+CRASH: Game spawned but then crashed (exit code: $exitCode)"
                Write-Host "Client sent packets after spawn - this is progress!"
            } else {
                Write-Host "CRASH: Game process exited (exit code: $exitCode)"
            }
            $delta = Get-LogDelta -startLen $logStart
            if ($delta) {
                Write-Host "--- Server log delta ---"
                Write-Host $delta
            }
            exit 2
        }
    } catch {
        Write-Host "CRASH: Game process not accessible"
        exit 2
    }

    # Check for spawn success
    if (-not $spawnDetected -and (Test-LogPatternRecent ($spawnNeedles -join "|"))) {
        Write-Host "SPAWN: Detected spawn sequence in server log"
        $spawnDetected = $true
        $spawnTime = Get-Date
    }

    # If spawn detected, send test inputs then monitor for stability
    if ($spawnDetected) {
        $elapsed = ((Get-Date) - $spawnTime).TotalSeconds

        # After 2 seconds, send some input to test for crashes
        if ($elapsed -ge 2 -and $elapsed -lt 4 -and -not $inputSent -and -not $SkipInput) {
            $inputSent = $true
            Write-Host "Sending test inputs (WASD hold + mouse + fire)..."
            Send-TestInputBurst $hwnd

            Write-Host "Test inputs sent (WASD hold + mouse + fire)"
        }
        if ($elapsed -ge 2 -and $elapsed -lt 4 -and -not $inputSent -and $SkipInput) {
            $inputSent = $true
            Write-Host "Skipping input/firing for this run"
        }

        if ($LateInput -and -not $lateInputSent -and $elapsed -ge 8 -and $elapsed -lt 10 -and -not $SkipInput) {
            $lateInputSent = $true
            Write-Host "Sending late input burst to verify post-5s control..."
            Send-TestInputBurst $hwnd
        }

        if ($elapsed -ge $successAfter) {
            $proc.Refresh()
            if (-not $proc.HasExited) {
                Write-Host "SUCCESS: Game still running $([math]::Round($elapsed, 1))s after spawn (with input test)!"

                # Always capture and analyze (not just with -Screenshot flag)
                    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

                    # Take multiple screenshots to catch flicker
                    $screenshotPath1 = Capture-Window $hwnd "spawn_test_${timestamp}_1.png"
                    Start-Sleep -Milliseconds 300
                    $screenshotPath2 = Capture-Window $hwnd "spawn_test_${timestamp}_2.png"
                    Start-Sleep -Milliseconds 300
                    $screenshotPath3 = Capture-Window $hwnd "spawn_test_${timestamp}_3.png"
                    Write-Host "Screenshots saved: ${timestamp}_1/2/3.png"

                    # Check all screenshots for red screen
                    $redResults = @()
                    foreach ($path in @($screenshotPath1, $screenshotPath2, $screenshotPath3)) {
                        $check = Test-RedScreen $path
                        $redResults += $check
                    }
                    $maxRed = 0
                    $anyRed = $false
                    foreach ($r in $redResults) {
                        if ($r.RedRatio -gt $maxRed) { $maxRed = $r.RedRatio }
                        if ($r.IsRed) { $anyRed = $true }
                    }

                    if ($anyRed) {
                        Write-Host "WARNING: Red screen detected! (max ratio: $([math]::Round($maxRed * 100, 1))%)"
                        Write-Host "  This indicates a health packet issue"
                    } else {
                        Write-Host "Health OK: Screen not red (max ratio: $([math]::Round($maxRed * 100, 1))%)"
                    }

                    # Check for viewpoint changes (rotation working via server log)
                    $viewCheck = Test-ViewpointChanges $logStart
                    if ($viewCheck.HasChanges) {
                        Write-Host "Rotation OK: Viewpoint changed by $([math]::Round($viewCheck.YawDelta, 1)) degrees ($($viewCheck.Updates) updates)"
                    } else {
                        Write-Host "WARNING: No viewpoint rotation detected ($($viewCheck.Updates) viewpoint log entries)"
                        # Also check for raw 0x35 packets
                        $vpPackets = Test-ViewpointPackets $logStart
                        if ($vpPackets.HasPackets) {
                            Write-Host "  Note: $($vpPackets.Count) raw 0x35 packets received but not parsed"
                        }
                        # Fall back to ACTION_DUMP rotation inputs
                        $rotInput = Test-RotationInput $logStart
                        if ($rotInput.HasRotationInput) {
                            Write-Host "Rotation INPUT OK: $($rotInput.NonZeroPackets)/$($rotInput.TotalPackets) ACTION_DUMP packets had turn/slot6/slot7 input"
                        } else {
                            Write-Host "WARNING: No rotation input detected ($($rotInput.TotalPackets) ACTION_DUMP packets)"
                        }
                    }

                    # Check for movement (position changes in server log)
                    $moveCheck = Test-MovementChanges $logStart
                    if ($moveCheck.HasMovement) {
                        Write-Host "Movement OK: Position changed by $([math]::Round($moveCheck.Delta, 1)) units"
                        Write-Host "  From $($moveCheck.StartPos) to $($moveCheck.EndPos)"
                    } else {
                        Write-Host "WARNING: No movement detected ($($moveCheck.Updates) tick entries)"
                        Write-Host "  Position stayed at $($moveCheck.StartPos)"
                    }

                    # Check for input reception
                    $inputCheck = Test-InputReceived $logStart
                    if ($inputCheck.HasInput) {
                        Write-Host "Input OK: $($inputCheck.NonZeroPackets)/$($inputCheck.TotalPackets) packets had movement input"
                    } else {
                        Write-Host "WARNING: No movement input received ($($inputCheck.TotalPackets) ACTION_DUMP packets, all zero input)"
                    }

                    # Check for desync
                    $desyncCheck = Test-Desync $logStart
                    if ($desyncCheck.HasDesync) {
                        Write-Host "ERROR: DESYNC detected! ($($desyncCheck.DesyncCount) occurrences)"
                        foreach ($detail in $desyncCheck.Details) {
                            Write-Host "  $detail"
                        }
                    } else {
                        Write-Host "Desync OK: No desync warnings"
                    }

                    # Check projectile spawn alignment vs player pose/yaw
                    $projCheck = Test-ProjectileAlignment $logStart
                    if ($projCheck.HasSpawn) {
                        if ($projCheck.IsAligned) {
                            Write-Host "Projectile OK: spawn aligned ($($projCheck.Mode)) (err=$([math]::Round($projCheck.PosError, 2)) dist=$([math]::Round($projCheck.Distance, 2)) cos=$([math]::Round($projCheck.ForwardCos, 2)))"
                        } else {
                            Write-Host "WARNING: Projectile misaligned ($($projCheck.Mode)) (err=$([math]::Round($projCheck.PosError, 2)) dist=$([math]::Round($projCheck.Distance, 2)) cos=$([math]::Round($projCheck.ForwardCos, 2)))"
                            if ($projCheck.ContainsKey("YawDeg")) {
                                Write-Host "  player=$($projCheck.PlayerPos) spawn=$($projCheck.SpawnPos) expected=$($projCheck.ExpectedPos) yaw=$([math]::Round($projCheck.YawDeg, 1))"
                            } else {
                                Write-Host "  player=$($projCheck.PlayerPos) spawn=$($projCheck.SpawnPos) expected=$($projCheck.ExpectedPos)"
                            }
                        }
                    } else {
                        Write-Host "WARNING: No projectile spawn logs found to verify alignment ($($projCheck.Reason))"
                    }

                    # Check whether projectiles used viewpoint or input for aim
                    $aimSource = Test-ProjectileAimSource $logStart
                    if ($aimSource.Total -gt 0) {
                        Write-Host "Projectile aim source: viewpoint=$($aimSource.Viewpoint) input=$($aimSource.Input) total=$($aimSource.Total)"
                    }

                exit 0
            }
        }
    }

    Start-Sleep -Milliseconds 200
}

Write-Host "TIMEOUT: Test duration exceeded"
$delta = Get-LogDelta -startLen $logStart
if ($delta) {
    Write-Host "--- Server log delta ---"
    Write-Host $delta
}
if ($spawnDetected) {
    Write-Host "Note: Spawn was detected but test timed out"
}
exit 3
