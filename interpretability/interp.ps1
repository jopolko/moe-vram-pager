<#
  Windows wrapper for the interpretability pipeline - PowerShell equivalent of ./interp.
  Activates the isolated venv, pins the HF cache into this dir, sources the HF token,
  then runs obench-interp.

    .\interp.ps1 doctor
    .\interp.ps1 pull                 # gemma-2-2b + Gemma Scope SAE (layer 12)
    .\interp.ps1 pull --instruct      # also gemma-2-2b-it (phase 3)
    .\interp.ps1 list
    .\interp.ps1 run exp1             # multilingual concept sharing
    .\interp.ps1 run exp2             # planning ahead (rhyming couplets)
    .\interp.ps1 run exp3             # chain-of-thought faithfulness (needs --instruct pull)
#>
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot

if (-not (Test-Path "$here\.venv")) {
    Write-Error "No venv. Create it:  cd $here ; uv venv .venv ; uv pip install -e ."
    exit 1
}

$obench = Join-Path $here '.venv\Scripts\obench-interp.exe'
$env:HF_HOME = Join-Path $here 'hf_cache'
# os.symlink needs admin or Developer Mode on Windows; without it snapshot_download
# dies with WinError 1314. Copy blobs instead (costs disk, always works).
$env:HF_HUB_DISABLE_SYMLINKS = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

# gemma-2-2b is gated: transformers re-checks tokenizer files on the hub even when
# weights are cached, so every model-loading subcommand needs the token, not just
# `pull`. First existing file wins. Values are loaded into the process env, never printed.
$envFiles = @(
    (Join-Path $env:USERPROFILE 'secrets\openbench.env'),
    (Join-Path $env:USERPROFILE 'secrets\nowservingto.env'),
    'C:\var\secrets\nowservingto.env'
)
foreach ($f in $envFiles) {
    if (Test-Path $f) {
        Get-Content $f | ForEach-Object {
            $line = $_.Trim()
            if ($line -and -not $line.StartsWith('#')) {
                $line = $line -replace '^export\s+', ''
                $kv = $line.Split('=', 2)
                if ($kv.Count -eq 2) {
                    [Environment]::SetEnvironmentVariable($kv[0].Trim(), $kv[1].Trim(), 'Process')
                }
            }
        }
        break
    }
}

& $obench @args
exit $LASTEXITCODE
