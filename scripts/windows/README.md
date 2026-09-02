# Windows scripts

Small helpers for running and building the CUDA `llama-server` on Windows. The
distribution philosophy is **ship small**: only our own DLLs + `llama-server.exe`
are bundled. The large vendor runtimes (NVIDIA CUDA, Microsoft VC++) are **not**
bundled - they are fetched on demand from their official sources.

## One launcher for everything

```powershell
# model + webUI (default model: the ollama "josiefied" tag)
powershell -ExecutionPolicy Bypass -File scripts\windows\start-openbench.ps1

# + Metasploit MCP + the interpretability viewer sidecar
... start-openbench.ps1 -Pentest -Interp

# a different model (ollama name, .gguf path, or sha256- blob), bigger context
... start-openbench.ps1 -Model huihui_ai/qwen3.5-abliterated:9b -Ctx 16384 -Pentest

# a big MoE with expert streaming
... start-openbench.ps1 -Model C:\models\GLM-4.5-Air-Q4.gguf -MoeStream -Ctx 131072

# router mode: browse + download + hot-swap models from the webUI Models page
... start-openbench.ps1 -Router -Pentest

# tear it all down
... start-openbench.ps1 -Stop
```

`start-openbench.ps1` resolves the model, starts `llama-server` with sane flags
(`-fa on`, `-ctk/-ctv q8_0` unless `-NoKvQuant`, **`--ui-mcp-proxy`** which the
webUI's Metasploit MCP entry needs, `--moe-stream` only with `-MoeStream`), waits
for `/health`, then optionally brings up the pentest MCP stack (`-Pentest`) and
the interp results sidecar (`-Interp`). Logs land in `%USERPROFILE%\.openbench\logs\`.
`-ExtraArgs` passes anything else straight through to `llama-server`.

**`-Router`** starts multi-model mode instead: no single `-m`, but
`--models-dir %USERPROFILE%\.openbench\models` + `--models-preset ...\models.ini`
(created if absent - the picker appends to it), which is what makes the webUI
Models page's **download + load** buttons work (`router_available` gates on the
preset). `--moe-stream` is on by default here (`-NoMoeStream` to disable). The
Models page needs the HTTPS-capable build - see **OpenSSL** below.

## Running a prebuilt binary

```bat
scripts\windows\llama-server.cmd --moe-stream --host 0.0.0.0 --port 8080
```

`llama-server.cmd` runs `preflight.ps1` first, fetches anything missing from the
official source, then launches the server with your arguments.

### `preflight.ps1`
Checks the three runtime dependency tiers and (with `-AutoFix`) remediates:

| Tier | What | Source when missing |
|---|---|---|
| A. NVIDIA driver | `nvcuda.dll` | You install/update it from NVIDIA |
| B. CUDA runtime | `cudart64_12`, `cublas64_12`, `cublasLt64_12` | `fetch-cuda-runtime.ps1` (NVIDIA redist) |
| C. MSVC runtime | `vcruntime140(_1)`, `msvcp140`, `vcomp140` | Microsoft `vc_redist.x64.exe` |

```powershell
# report only
powershell -ExecutionPolicy Bypass -File scripts\windows\preflight.ps1
# report + fix from official sources
powershell -ExecutionPolicy Bypass -File scripts\windows\preflight.ps1 -AutoFix
```

### `fetch-cuda-runtime.ps1`
Downloads only the `cuda_cudart` + `libcublas` redistributable archives from
NVIDIA's official redist server
(`developer.download.nvidia.com/compute/cuda/redist/`), extracts the three DLLs
into the app folder, and cleans up. No 3 GB toolkit required.

## Building from source

```powershell
# check the build toolchain, install anything missing via winget, then build
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -InstallDeps
# also build the embedded web UI (needs Node.js)
powershell -ExecutionPolicy Bypass -File scripts\windows\build.ps1 -InstallDeps -WithUI
```

`build.ps1` preflights the **build** toolchain (VS 2022 C++ Build Tools, CUDA
Toolkit, CMake >= 3.24, Ninja, and Node.js for the UI). With `-InstallDeps` it
installs missing pieces via `winget` (prompting per package); without it, it
tells you exactly what to install and stops. It auto-selects the right CUDA
toolkit for the GPU in the machine - a **CUDA 12.x** toolkit for pre-Turing cards
(e.g. GTX 10-series `sm_61`), since CUDA 13 dropped offline codegen below
`sm_75`.

## Pentest MCP (Metasploit)

The webui's **Metasploit** MCP entry (`http://127.0.0.1:8085/sse`) needs a local
`MetasploitMCP` server, which needs Metasploit Framework and its RPC daemon.
`setup-pentest-mcp.ps1` does the whole install; it self-elevates for the parts
that need Administrator.

```powershell
# install Metasploit + Defender exclusion + firewall rules + MetasploitMCP venv + msfdb init
powershell -ExecutionPolicy Bypass -File scripts\windows\setup-pentest-mcp.ps1
# also open :8080/:8085/:55553/... to other machines on the LAN
powershell -ExecutionPolicy Bypass -File scripts\windows\setup-pentest-mcp.ps1 -AllowLocalSubnet
```

What it does, all idempotent:

| Step | Detail |
|---|---|
| Metasploit Framework | from `%USERPROFILE%\Downloads\metasploitframework-latest.msi` if present, else downloads it; `msiexec /passive` |
| Defender | `-ExclusionPath` for the MSF dir (before install, so it isn't quarantined mid-flight) + the `MetasploitMCP` checkout + `ruby.exe` process |
| Firewall | rules grouped `OpenBench Pentest MCP`: inbound for `ruby.exe` (reverse-shell handlers) + TCP 4444-4460 on Private/Domain; with `-AllowLocalSubnet`, inbound TCP 8080/8085/8087/8086/55553 from `LocalSubnet` |
| MetasploitMCP | `py -3.12 -m venv`, `pip install -r requirements.txt`, writes `.env` with `MSF_PASSWORD` from `%USERPROFILE%\secrets\moe-vram-pager.env` (generates one if absent) |
| Database | `msfdb init` (bundled PostgreSQL) |

> Pure loopback (`127.0.0.1` ↔ `127.0.0.1`) is never filtered by Windows
> Firewall, so if everything stays local you don't strictly need the firewall
> step - it's for reverse shells coming back from a target and `-AllowLocalSubnet`.
> Remove the rules any time: `Get-NetFirewallRule -Group 'OpenBench Pentest MCP' | Remove-NetFirewallRule`

Then start the stack (no elevation):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\start-pentest-mcp.ps1
# add -Bind 0.0.0.0 to reach it from the LAN (needs setup ... -AllowLocalSubnet)
# add -Detach to leave it running and get the shell back
```

`start-pentest-mcp.ps1` brings up `msfrpcd` (127.0.0.1:55553, no SSL) then
`MetasploitMCP.py --transport http` (SSE on :8085). In the webui: **MCP Servers
→ Metasploit → Connect** - it routes through `llama-server`'s CORS proxy
(`useProxy: true` is already set on that entry).
