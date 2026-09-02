<#
.SYNOPSIS
  Runtime dependency preflight for the openbench-toolkit Windows CUDA build.

.DESCRIPTION
  This project ships SMALL: only our own DLLs + llama-server.exe. The large
  vendor runtimes are NOT bundled - instead this script detects what is missing
  and either remediates from the official source (NVIDIA / Microsoft) or tells
  the user exactly where to get it. See scripts\windows\README.md.

  Three dependency tiers are checked:
    A. NVIDIA driver        -> nvcuda.dll (user installs / updates from NVIDIA)
    B. CUDA runtime + math  -> cudart64_12/cublas64_12/cublasLt64_12.dll
                               (fetched on demand from NVIDIA's official redist)
    C. MSVC runtime         -> vcruntime140(_1)/msvcp140/vcomp140.dll
                               (Microsoft vc_redist.x64.exe, installs as needed)

.PARAMETER AppDir
  Folder containing llama-server.exe and the ggml/llama DLLs. Defaults to the
  script's parent-parent \build\bin, else the script directory.

.PARAMETER AutoFix
  Remediate missing tiers automatically from the official sources instead of
  only reporting them.

.PARAMETER Quiet
  Suppress the OK lines; only warn/error output is shown.

.OUTPUTS
  Exit code 0 = all satisfied (or fixed). Non-zero = something is still missing.
#>
[CmdletBinding()]
param(
    [string] $AppDir,
    [switch] $AutoFix,
    [switch] $Quiet
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $AppDir) {
    $candidate = Join-Path $scriptDir '..\..\build\bin'
    if (Test-Path (Join-Path $candidate 'llama-server.exe')) {
        $AppDir = (Resolve-Path $candidate).Path
    } else {
        $AppDir = $scriptDir
    }
}

function Write-Ok   ($m) { if (-not $Quiet) { Write-Host "  [ ok ] $m" -ForegroundColor Green } }
function Write-Miss ($m) { Write-Host "  [MISS] $m" -ForegroundColor Yellow }
function Write-Head ($m) { Write-Host "`n$m" -ForegroundColor Cyan }

# A DLL counts as resolvable if it sits next to the exe or anywhere on PATH or
# in the system directory (the Windows loader search order we actually rely on).
function Resolve-Dll ([string] $name) {
    $local = Join-Path $AppDir $name
    if (Test-Path $local) { return $local }
    $sys = Join-Path $env:WINDIR "System32\$name"
    if (Test-Path $sys) { return $sys }
    foreach ($p in ($env:PATH -split ';')) {
        if ($p -and (Test-Path (Join-Path $p $name))) { return (Join-Path $p $name) }
    }
    return $null
}

Write-Host "openbench-toolkit preflight" -ForegroundColor White
Write-Host "app dir: $AppDir"

$missing = @{}

# ---- Tier A: NVIDIA driver ------------------------------------------------
Write-Head "NVIDIA driver (provides nvcuda.dll):"
$nvcuda = Resolve-Dll 'nvcuda.dll'
if ($nvcuda) {
    $drv = $null
    $smi = Join-Path $env:WINDIR 'System32\nvidia-smi.exe'
    if (Test-Path $smi) {
        try { $drv = (& $smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1).Trim() } catch {}
    }
    if ($drv) { Write-Ok "nvcuda.dll present (driver $drv)" } else { Write-Ok "nvcuda.dll present" }
} else {
    Write-Miss "nvcuda.dll not found - install/upgrade the NVIDIA GPU driver."
    $missing['driver'] = $true
}

# ---- Tier B: CUDA runtime + math -----------------------------------------
Write-Head "CUDA runtime (bundled on demand from NVIDIA):"
$cudaDlls = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')
$cudaMissing = @()
foreach ($d in $cudaDlls) {
    if (Resolve-Dll $d) { Write-Ok $d } else { Write-Miss $d; $cudaMissing += $d }
}
if ($cudaMissing.Count -gt 0) { $missing['cuda'] = $true }

# ---- Tier C: MSVC runtime -------------------------------------------------
Write-Head "MSVC runtime (Microsoft vc_redist):"
$vcDlls = @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'vcomp140.dll')
$vcMissing = @()
foreach ($d in $vcDlls) {
    if (Resolve-Dll $d) { Write-Ok $d } else { Write-Miss $d; $vcMissing += $d }
}
if ($vcMissing.Count -gt 0) { $missing['vcredist'] = $true }

# ---- Tier D: OpenSSL (only if this build was linked against it) -----------
# The server's HTTP client needs libssl/libcrypto for HTTPS (Models page,
# -hf downloads). Only flag them missing if llama-server.exe actually imports
# them - a build without HTTPS won't, and shouldn't be nagged about it.
$needsSsl = $false
foreach ($exe in 'llama-server.exe', 'llama-server-impl.dll', 'llama-common.dll') {
    $p = Join-Path $AppDir $exe
    if ((Test-Path $p) -and (Select-String -Path $p -Pattern 'libssl-\d+-x64\.dll' -Quiet -ErrorAction SilentlyContinue)) {
        $needsSsl = $true; break
    }
}
if ($needsSsl) {
    Write-Head "OpenSSL runtime (this build does HTTPS):"
    $sslMissing = @()
    foreach ($d in 'libssl', 'libcrypto') {
        if (Get-ChildItem $AppDir -Filter "$d-*-x64.dll" -ErrorAction SilentlyContinue) { Write-Ok "$d-*-x64.dll" }
        else { Write-Miss "$d-*-x64.dll"; $sslMissing += $d }
    }
    if ($sslMissing.Count -gt 0) { $missing['openssl'] = $true }
}

# ---- Remediation ----------------------------------------------------------
if ($missing.Count -eq 0) {
    Write-Host "`nAll runtime dependencies satisfied." -ForegroundColor Green
    exit 0
}

Write-Host "`nMissing dependencies detected." -ForegroundColor Yellow

$driverUrl = 'https://www.nvidia.com/Download/index.aspx'
$vcRedistUrl = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'

if ($missing['driver']) {
    Write-Host "  - NVIDIA driver: download from $driverUrl"
    if ($AutoFix) { Start-Process $driverUrl }
}

if ($missing['vcredist']) {
    Write-Host "  - MSVC runtime: Microsoft VC++ redistributable ($vcRedistUrl)"
    if ($AutoFix) {
        $tmp = Join-Path $env:TEMP 'vc_redist.x64.exe'
        Write-Host "    downloading vc_redist.x64.exe ..."
        Invoke-WebRequest -Uri $vcRedistUrl -OutFile $tmp -UseBasicParsing
        Write-Host "    launching Microsoft installer (installs as needed) ..."
        Start-Process -FilePath $tmp -ArgumentList '/install', '/passive', '/norestart' -Wait
    }
}

if ($missing['openssl']) {
    Write-Host "  - OpenSSL: install ShiningLight OpenSSL and copy its 2 DLLs next to llama-server.exe"
    if ($AutoFix) {
        winget install --id ShiningLight.OpenSSL.Light --accept-source-agreements --accept-package-agreements
        foreach ($d in 'libssl-*-x64.dll', 'libcrypto-*-x64.dll') {
            Get-ChildItem 'C:\Program Files\OpenSSL-Win64' -Filter $d -ErrorAction SilentlyContinue |
                Copy-Item -Destination $AppDir -Force
        }
    }
}

if ($missing['cuda']) {
    $fetch = Join-Path $scriptDir 'fetch-cuda-runtime.ps1'
    Write-Host "  - CUDA runtime: fetch just the needed DLLs from NVIDIA's official redist server"
    Write-Host "      $fetch -AppDir `"$AppDir`""
    if ($AutoFix) {
        & $fetch -AppDir $AppDir
    }
}

if ($AutoFix) {
    Write-Host "`nRe-checking after remediation ..." -ForegroundColor Cyan
    & $MyInvocation.MyCommand.Path -AppDir $AppDir -Quiet
    exit $LASTEXITCODE
}

exit 1
