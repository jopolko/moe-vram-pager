<#
.SYNOPSIS
  One launcher for the whole openbench appliance on Windows:
  llama-server (+ webUI + MCP proxy)  [+ Metasploit MCP]  [+ interpretability viewer].

.DESCRIPTION
  Resolves the model (an ollama name like "josiefied", a .gguf path, or a
  sha256- blob), starts llama-server with sane flags, and optionally brings up
  the pentest MCP stack and the interp results sidecar. Waits for each to be
  reachable and prints the URLs. `-Stop` tears it all down.

  Logs: %USERPROFILE%\.openbench\logs\

.EXAMPLE
  # just the model + webUI
  powershell -ExecutionPolicy Bypass -File scripts\windows\start-openbench.ps1

.EXAMPLE
  # everything: a specific model, the Metasploit MCP, the interp viewer
  ... start-openbench.ps1 -Model huihui_ai/qwen3.5-abliterated:9b -Pentest -Interp

.EXAMPLE
  # a big MoE with expert streaming, 128k context
  ... start-openbench.ps1 -Model C:\models\GLM-4.5-Air-Q4.gguf -MoeStream -Ctx 131072

.EXAMPLE
  ... start-openbench.ps1 -Stop
#>
[CmdletBinding()]
param(
    [string]   $Model      = 'josiefied',
    [int]      $Ctx        = 32768,
    [int]      $Ngl        = 99,
    [int]      $Port       = 8080,
    [string]   $Bind       = '127.0.0.1',
    [switch]   $MoeStream,
    [switch]   $NoKvQuant,
    [switch]   $Router,        # multi-model / model-picker mode (download+load from the webUI)
    [switch]   $NoMoeStream,   # router mode: don't pass --moe-stream
    [string]   $ModelsDir  = (Join-Path $env:USERPROFILE '.openbench\models'),
    [string]   $PresetIni  = (Join-Path $env:USERPROFILE '.openbench\models.ini'),
    [int]      $ModelsMax  = 1,
    [switch]   $Pentest,
    [switch]   $Interp,
    [switch]   $Stop,
    [int]      $InterpPort = 8087,
    [string[]] $ExtraArgs
)

$ErrorActionPreference = 'Stop'
$RepoDir   = Split-Path (Split-Path $PSScriptRoot)          # scripts\windows -> repo root
$Server    = Join-Path $RepoDir 'build\bin\llama-server.exe'
$LogDir    = Join-Path $env:USERPROFILE '.openbench\logs'
$OllamaDir = if ($env:OLLAMA_MODELS) { $env:OLLAMA_MODELS } else { Join-Path $env:USERPROFILE '.ollama\models' }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Info($m) { Write-Host "  $m" }
function Step($m) { Write-Host "`n== $m ==" -ForegroundColor Cyan }
function Warn($m) { Write-Host "  ! $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "  x $m" -ForegroundColor Red; exit 1 }

function Test-Port([int] $p) {
    Test-NetConnection -ComputerName 127.0.0.1 -Port $p -InformationLevel Quiet -WarningAction SilentlyContinue
}
function Stop-Port([int] $p, [string] $label) {
    $ids = (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue).OwningProcess |
           Sort-Object -Unique
    foreach ($id in $ids) {
        if ($id -and $id -ne 0) {
            try { Stop-Process -Id $id -Force -ErrorAction Stop; Info "stopped $label (pid $id)" } catch {}
        }
    }
}

# ---------------------------------------------------------------------------
if ($Stop) {
    Step 'stopping openbench'
    Stop-Port $Port       'llama-server'
    Stop-Port 8085        'MetasploitMCP'
    Stop-Port 55553       'msfrpcd'
    Stop-Port $InterpPort 'interp sidecar'
    # msfrpcd is a detached ruby with no listening socket left if it half-died
    Get-Process ruby -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -like '*metasploit*' } |
        ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue; Info "stopped ruby (pid $($_.Id))" }
    Write-Host "`nstopped." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path $Server)) {
    Die "llama-server not built: $Server`n      build it: scripts\windows\build.ps1 -InstallDeps -WithUI"
}

# --- resolve the model ---------------------------------------------------
function Resolve-Model([string] $m) {
    if (Test-Path $m) { return @{ path = (Resolve-Path $m).Path; name = [IO.Path]::GetFileNameWithoutExtension($m) } }
    if ($m -match '^sha256-[0-9a-f]{64}$') {
        $b = Join-Path $OllamaDir "blobs\$m"
        if (Test-Path $b) { return @{ path = $b; name = $m.Substring(0, 14) } }
        Die "blob not found: $b"
    }
    # ollama ref:  [registry/][namespace/]name[:tag]
    $ref = $m
    $tag = 'latest'
    if ($ref -match '^(.*):([^:/]+)$') { $ref = $Matches[1]; $tag = $Matches[2] }
    $manifestRoots = @(
        (Join-Path $OllamaDir "manifests\registry.ollama.ai\$ref\$tag"),
        (Join-Path $OllamaDir "manifests\registry.ollama.ai\library\$ref\$tag")
    )
    # bare name like "josiefied" - search every manifest for a path match
    $mf = $manifestRoots | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $mf) {
        $mf = Get-ChildItem (Join-Path $OllamaDir 'manifests') -Recurse -File -ErrorAction SilentlyContinue |
              Where-Object { $_.FullName -match [Regex]::Escape($ref) -and $_.Name -eq $tag } |
              Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $mf -or -not (Test-Path $mf)) {
        Warn "no ollama model matched '$m'. Installed:"
        Get-ChildItem (Join-Path $OllamaDir 'manifests') -Recurse -File -ErrorAction SilentlyContinue |
            ForEach-Object { '    ' + ($_.FullName -replace [Regex]::Escape((Join-Path $OllamaDir 'manifests\registry.ollama.ai') + '\'), '' -replace '\\', '/' -replace '/([^/]+)$', ':$1') } |
            Sort-Object -Unique | Write-Host
        Die "pass -Model <name|path|sha256->"
    }
    $json = Get-Content $mf -Raw | ConvertFrom-Json
    $layer = $json.layers | Where-Object { $_.mediaType -eq 'application/vnd.ollama.image.model' } | Select-Object -First 1
    if (-not $layer) { Die "manifest $mf has no model layer" }
    $blob = Join-Path $OllamaDir ('blobs\' + ($layer.digest -replace ':', '-'))
    if (-not (Test-Path $blob)) { Die "model blob missing: $blob" }
    @{ path = $blob; name = ($ref -replace '.*/', '') }
}

# --- llama-server --------------------------------------------------------
Step "llama-server  ($Bind`:$Port$(if($Router){'  [router mode]'}))"
Stop-Port $Port 'old llama-server'
Start-Sleep 1

$srvArgs = @('-fa', 'on', '--ui-mcp-proxy', '--host', $Bind, '--port', "$Port")
if (-not $NoKvQuant) { $srvArgs += @('-ctk', 'q8_0', '-ctv', 'q8_0') }

if ($Router) {
    # multi-model: the webUI's Models page can download + hot-load. router_available
    # gates on --models-preset, so the ini must exist (the picker appends to it).
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
    if (-not (Test-Path $PresetIni)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $PresetIni) | Out-Null
        @('[*]', 'ctx-size  = 0', 'mmap      = 1', 'kv-unified = 0') |
            Set-Content -Encoding ascii $PresetIni
        Info "created $PresetIni"
    }
    # if -Model names a real file/blob, make it pickable in the dir
    if ($Model -and $Model -ne 'josiefied' -and (Test-Path $Model -PathType Leaf)) {
        $dst = Join-Path $ModelsDir ([IO.Path]::GetFileName($Model))
        if ($dst -notlike '*.gguf') { $dst += '.gguf' }
        if (-not (Test-Path $dst)) {
            try { New-Item -ItemType HardLink -Path $dst -Target (Resolve-Path $Model) | Out-Null; Info "linked $(Split-Path $dst -Leaf) into models dir" }
            catch { Copy-Item $Model $dst; Info "copied $(Split-Path $dst -Leaf) into models dir" }
        }
    }
    $srvArgs += @('--models-dir', $ModelsDir, '--models-preset', $PresetIni, '--models-max', "$ModelsMax")
    if (-not $NoMoeStream) { $srvArgs += '--moe-stream' }
    Info "models dir: $ModelsDir"
    Info "preset:     $PresetIni"
    $gguf = @(Get-ChildItem $ModelsDir -Filter *.gguf -ErrorAction SilentlyContinue)
    Info "$($gguf.Count) local GGUF$(if($gguf.Count -ne 1){'s'}) - browse/download more from the webUI Models page"
} else {
    Step 'model'
    $resolved = Resolve-Model $Model
    Info "$($resolved.name)"
    Info "$($resolved.path)"
    $srvArgs += @('-m', $resolved.path, '-a', $resolved.name, '-ngl', "$Ngl", '-c', "$Ctx", '-np', '1')
    if ($MoeStream) { $srvArgs += '--moe-stream' }
}
if ($ExtraArgs) { $srvArgs += $ExtraArgs }

$srvLog = Join-Path $LogDir ("llama-server-{0}.log" -f (Get-Date -Format yyyyMMdd-HHmmss))
$srv = Start-Process $Server -PassThru -WindowStyle Hidden -RedirectStandardOutput $srvLog `
    -RedirectStandardError ($srvLog -replace '\.log$', '.err.log') -ArgumentList $srvArgs
Info "pid $($srv.Id)  |  log $srvLog"
Info "waiting for /health ..."
$ok = $false
foreach ($i in 1..120) {
    Start-Sleep 1
    try { if ((Invoke-WebRequest "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $ok = $true; break } } catch {}
    if ($srv.HasExited) { Die "llama-server exited (code $($srv.ExitCode)) - see $srvLog" }
}
if (-not $ok) { Die "llama-server didn't answer /health in 120s - see $srvLog" }
Info "up"

# --- pentest MCP stack -------------------------------------------------
if ($Pentest) {
    Step 'Metasploit MCP'
    & (Join-Path $PSScriptRoot 'start-pentest-mcp.ps1') -Detach
}

# --- interp sidecar -------------------------------------------------
if ($Interp) {
    Step "interpretability viewer  (:$InterpPort)"
    if (Test-Port $InterpPort) {
        Info "already listening on $InterpPort"
    } else {
        $py = (Get-Command py -ErrorAction SilentlyContinue)
        $ex = if ($py) { 'py' } else { 'python' }
        $iLog = Join-Path $LogDir 'interp-ui-api.log'
        Start-Process $ex -PassThru -WindowStyle Hidden -RedirectStandardOutput $iLog `
            -RedirectStandardError ($iLog -replace '\.log$', '.err.log') `
            -ArgumentList @((Join-Path $RepoDir 'tools\interp_ui_api.py'), '--port', "$InterpPort") | Out-Null
        Start-Sleep 2
        if (Test-Port $InterpPort) { Info "up  |  log $iLog" } else { Warn "didn't come up - see $iLog" }
    }
}

# --- summary ---------------------------------------------------------------
Step 'ready'
$modelLine = if ($Router) {
    "router mode - pick/download models on the webUI Models page$(if(-not $NoMoeStream){' (--moe-stream on)'})"
} else {
    "model: $($resolved.name), ctx $Ctx$(if($MoeStream){', --moe-stream'})"
}
Write-Host @"
  webUI + API      http://127.0.0.1:$Port           ($modelLine)
$(if ($Router)  { "  Models page      #/models - browse UGI-ranked derestricted MoE models, download + run`n" })$(if ($Pentest) { "  Metasploit MCP   http://127.0.0.1:8085/sse       (via the webUI's CORS proxy)`n" })$(if ($Interp)  { "  interp viewer    webUI -> flask icon -> Open results folder`n" })
  stop everything: scripts\windows\start-openbench.ps1 -Stop
"@ -ForegroundColor Green
