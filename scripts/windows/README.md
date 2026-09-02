# Windows scripts

Small helpers for running and building the CUDA `llama-server` on Windows. The
distribution philosophy is **ship small**: only our own DLLs + `llama-server.exe`
are bundled. The large vendor runtimes (NVIDIA CUDA, Microsoft VC++) are **not**
bundled - they are fetched on demand from their official sources.

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
