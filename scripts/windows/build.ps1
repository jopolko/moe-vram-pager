<#
.SYNOPSIS
  Build the openbench-toolkit CUDA llama-server from source on Windows, checking
  (and optionally installing) the required build toolchain first.

.DESCRIPTION
  Answers "what else do I need to build from source?" by preflighting the build
  toolchain and, with -InstallDeps, installing whatever is missing via winget
  (it prompts per package). Then it configures + builds with the correct CUDA
  toolkit for the GPU in this machine.

  Build prerequisites (distinct from the RUNTIME deps in preflight.ps1):
    - Visual Studio 2022 Build Tools  (MSVC v143, "Desktop development with C++")
    - CUDA Toolkit                    (12.9 for legacy sm_5x/6x/7.0 GPUs;
                                        13.x is fine for Turing sm_75+)
    - CMake >= 3.24
    - Ninja                           (generator used here)
    - Node.js + npm                   (only if -WithUI)

  Legacy-GPU note: CUDA 13 dropped offline codegen below sm_75. If this machine's
  GPU is pre-Turing (e.g. GTX 10-series sm_61) the script looks for a CUDA 12.x
  toolkit and builds against it; the CMake preflight guard fails fast otherwise.

.PARAMETER InstallDeps
  Install any missing build prerequisites via winget before building.

.PARAMETER WithUI
  Also build the embedded web UI (requires Node.js). Passes -DLLAMA_BUILD_UI=ON.

.PARAMETER Target
  CMake target to build. Default 'llama-server'.

.PARAMETER Jobs
  Parallel build jobs. Default = number of logical processors.
#>
[CmdletBinding()]
param(
    [switch] $InstallDeps,
    [switch] $WithUI,
    [string] $Target = 'llama-server',
    [int]    $Jobs = $env:NUMBER_OF_PROCESSORS
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..\..')).Path

function Have ($exe) { [bool](Get-Command $exe -ErrorAction SilentlyContinue) }
function Winget-Install ($id) {
    if (-not (Have 'winget')) { throw "winget not available - install '$id' manually." }
    Write-Host "  winget install $id" -ForegroundColor Cyan
    winget install --id $id --accept-source-agreements --accept-package-agreements
}

# ---- Locate VS 2022 (Build Tools or full IDE) + its vcvars ----------------
function Find-VcVars {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) {
        $inst = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null | Select-Object -First 1
        if ($inst) {
            $vc = Join-Path $inst 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path $vc) { return $vc }
        }
    }
    return $null
}

# ---- Find a CUDA toolkit appropriate for this GPU -------------------------
function Get-GpuArch {
    $smi = Join-Path $env:WINDIR 'System32\nvidia-smi.exe'
    if (-not (Test-Path $smi)) { return $null }
    try {
        $cc = (& $smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        return [int]($cc -replace '\.', '')   # "6.1" -> 61
    } catch { return $null }
}
function Find-CudaToolkit ([int] $arch) {
    $root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (-not (Test-Path $root)) { return $null }
    $vers = Get-ChildItem $root -Directory | Where-Object { $_.Name -match '^v(\d+)\.(\d+)$' } |
        ForEach-Object {
            [pscustomobject]@{ Path = $_.FullName;
                Major = [int]$Matches[1]; Minor = [int]$Matches[2];
                Ver = [version]("{0}.{1}" -f $Matches[1], $Matches[2]) }
        } | Sort-Object Ver
    if (-not $vers) { return $null }
    # Pre-Turing needs CUDA < 13; otherwise prefer the newest.
    if ($arch -and $arch -lt 75) {
        $ok = $vers | Where-Object { $_.Major -lt 13 } | Select-Object -Last 1
        if ($ok) { return $ok.Path }
        Write-Warning "GPU is sm_$arch (pre-Turing) but no CUDA 12.x toolkit found; CUDA 13 cannot build it."
        return $null
    }
    return ($vers | Select-Object -Last 1).Path
}

Write-Host "openbench-toolkit build (from source)" -ForegroundColor White
Write-Host "repo: $repo`n"

# ---- Preflight the toolchain ----------------------------------------------
$arch = Get-GpuArch
if ($arch) { Write-Host "detected GPU compute capability: sm_$arch" }

$vcvars = Find-VcVars
if (-not $vcvars) {
    Write-Warning "Visual Studio 2022 C++ Build Tools not found."
    if ($InstallDeps) {
        Winget-Install 'Microsoft.VisualStudio.2022.BuildTools'
        Write-Host "NOTE: add the 'Desktop development with C++' workload in the VS Installer if the build can't find cl.exe." -ForegroundColor Yellow
        $vcvars = Find-VcVars
    }
    if (-not $vcvars) { throw "Need VS 2022 Build Tools (MSVC v143 + C++ workload). Re-run with -InstallDeps or install manually." }
}
Write-Host "vcvars: $vcvars"

if (-not (Have 'cmake')) {
    if ($InstallDeps) { Winget-Install 'Kitware.CMake' } else { throw "cmake not found. Re-run with -InstallDeps or install CMake >= 3.24." }
}
if (-not (Have 'ninja')) {
    if ($InstallDeps) { Winget-Install 'Ninja-build.Ninja' } else { throw "ninja not found. Re-run with -InstallDeps or 'pip install ninja'." }
}

$cuda = Find-CudaToolkit $arch
if (-not $cuda) {
    if ($InstallDeps -and $arch -and $arch -lt 75) {
        Winget-Install 'Nvidia.CUDA --version 12.9 --force'   # legacy GPU needs 12.x
        $cuda = Find-CudaToolkit $arch
    } elseif ($InstallDeps) {
        Winget-Install 'Nvidia.CUDA'
        $cuda = Find-CudaToolkit $arch
    }
    if (-not $cuda) { throw "No suitable CUDA Toolkit found. See scripts\windows\README.md." }
}
Write-Host "CUDA toolkit: $cuda"

if ($WithUI -and -not (Have 'node')) {
    if ($InstallDeps) { Winget-Install 'OpenJS.NodeJS.LTS' } else { throw "-WithUI needs Node.js. Re-run with -InstallDeps or install Node LTS." }
}

# ---- Configure + build via a vcvars-initialized child cmd -----------------
$uiFlag = if ($WithUI) { 'ON' } else { 'OFF' }
$ninjaDir = Split-Path (Get-Command ninja -ErrorAction SilentlyContinue).Source -ErrorAction SilentlyContinue

$bat = @"
@echo off
call "$vcvars" >nul 2>&1
set "CUDA_PATH=$cuda"
set "CUDAToolkit_ROOT=%CUDA_PATH%"
set "CUDACXX=%CUDA_PATH%\bin\nvcc.exe"
set "PATH=%CUDA_PATH%\bin;$ninjaDir;%PATH%"
cd /d "$repo"
cmake -B build -G Ninja -DGGML_CUDA=ON -DLLAMA_BUILD_UI=$uiFlag -DCMAKE_BUILD_TYPE=Release || exit /b 1
cmake --build build --target $Target -j $Jobs || exit /b 1
"@
$tmp = Join-Path $env:TEMP ("obk_build_" + [guid]::NewGuid().ToString('N') + '.bat')
Set-Content -Path $tmp -Value $bat -Encoding ascii
try {
    & cmd /c $tmp
    if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)." }
} finally { Remove-Item $tmp -ErrorAction SilentlyContinue }

Write-Host "`nBuild complete. Binaries in build\bin\." -ForegroundColor Green
Write-Host "Before running, satisfy runtime deps:  scripts\windows\preflight.ps1 -AutoFix" -ForegroundColor Cyan
