<#
.SYNOPSIS
  Fetch the minimal CUDA runtime DLLs from NVIDIA's official redist server.

.DESCRIPTION
  The openbench-toolkit Windows binary links CUDA dynamically but does NOT bundle
  the (large) NVIDIA runtime. This downloads ONLY the two redistributable
  archives we need - cuda_cudart and libcublas - straight from NVIDIA's official
  redist server, extracts the three DLLs (cudart64_12, cublas64_12,
  cublasLt64_12) into the app folder, and deletes the archives.

  Source: https://developer.download.nvidia.com/compute/cuda/redist/
  These archives are the NVIDIA CUDA redistributables (licensed for redist).

.PARAMETER AppDir
  Destination folder (where llama-server.exe / ggml-cuda.dll live). Required.

.PARAMETER CudaMajor
  CUDA major.minor to match the build. Default '12.9' (the toolkit this project
  builds against for legacy sm_61 GPUs; the DLLs are ABI 12.x so cudart64_12 /
  cublas64_12 also load fine for any CUDA 12.x-built binary).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $AppDir,
    [string] $CudaMajor = '12.9'
)

$ErrorActionPreference = 'Stop'
$base = 'https://developer.download.nvidia.com/compute/cuda/redist'

if (-not (Test-Path $AppDir)) { throw "AppDir does not exist: $AppDir" }

# Pick the newest archive matching $CudaMajor from a library's listing.
function Get-LatestArchive ([string] $lib) {
    $url = "$base/$lib/windows-x86_64/"
    $html = (Invoke-WebRequest -Uri $url -UseBasicParsing).Content
    $pattern = [regex]::Escape("$lib-windows-x86_64-$CudaMajor") + "[0-9.]*-archive\.zip"
    $names = [regex]::Matches($html, $pattern) | ForEach-Object { $_.Value } | Sort-Object -Unique
    if (-not $names) { throw "No $lib archive for CUDA $CudaMajor at $url" }
    # Sort by the numeric version embedded in the name, take the highest.
    $latest = $names | Sort-Object {
        $v = ($_ -replace "^$lib-windows-x86_64-", '') -replace '-archive\.zip$', ''
        try { [version]$v } catch { [version]'0.0' }
    } | Select-Object -Last 1
    return @{ Name = $latest; Url = "$url$latest" }
}

$work = Join-Path $env:TEMP ("obk_cuda_redist_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
$wanted = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')

try {
    foreach ($lib in @('cuda_cudart', 'libcublas')) {
        $a = Get-LatestArchive $lib
        Write-Host "Downloading $($a.Name) ..." -ForegroundColor Cyan
        $zip = Join-Path $work $a.Name
        Invoke-WebRequest -Uri $a.Url -OutFile $zip -UseBasicParsing
        $ex = Join-Path $work ($lib + '_x')
        Expand-Archive -Path $zip -DestinationPath $ex -Force
        Get-ChildItem -Path $ex -Recurse -Filter '*.dll' |
            Where-Object { $wanted -contains $_.Name } |
            ForEach-Object {
                Copy-Item $_.FullName -Destination (Join-Path $AppDir $_.Name) -Force
                Write-Host "  installed $($_.Name)" -ForegroundColor Green
            }
    }
    $still = $wanted | Where-Object { -not (Test-Path (Join-Path $AppDir $_)) }
    if ($still) { throw "Still missing after fetch: $($still -join ', ')" }
    Write-Host "CUDA runtime DLLs installed into $AppDir" -ForegroundColor Green
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
