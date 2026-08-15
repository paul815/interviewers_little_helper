<#
    Creates (or refreshes) "Launch ILH.lnk" in the repository root.

    The committed .lnk stores both an absolute and a relative path to
    tools\launch_win.bat; Windows falls back to the relative one, so the
    shortcut keeps working after the folder is moved or cloned elsewhere.
    Run this only if the shortcut goes missing or stops resolving:

        powershell -NoProfile -ExecutionPolicy Bypass -File tools\make_shortcut.ps1

    -Desktop additionally drops a copy on the desktop.
#>
[CmdletBinding()]
param(
    [switch]$Desktop
)

$ErrorActionPreference = 'Stop'

$root   = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$target = Join-Path $root 'tools\launch_win.bat'
$link   = Join-Path $root 'Launch ILH.lnk'

if (-not (Test-Path -LiteralPath $target)) {
    throw "Launcher not found: $target"
}

function New-IlhShortcut {
    param([string]$Path)

    $shell = New-Object -ComObject WScript.Shell
    try {
        $sc = $shell.CreateShortcut($Path)
        $sc.TargetPath       = $target
        $sc.WorkingDirectory = $root
        $sc.Description      = "Interviewer's Little Helper"
        # cmd.exe icon: the repo ships no .ico, and a shortcut with a broken
        # icon path renders as a blank page instead of falling back.
        $sc.IconLocation     = "$env:SystemRoot\System32\cmd.exe,0"
        $sc.WindowStyle      = 1
        $sc.Save()
    } finally {
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($shell)
    }
    Write-Host "Shortcut written: $Path"
}

New-IlhShortcut -Path $link

if ($Desktop) {
    New-IlhShortcut -Path (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Launch ILH.lnk')
}
