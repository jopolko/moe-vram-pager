@echo off
rem ---------------------------------------------------------------------------
rem  openbench-toolkit - Windows launcher for llama-server
rem
rem  Runs the runtime preflight (checks NVIDIA driver / CUDA runtime / MSVC
rem  runtime, fetching the small CUDA DLLs from NVIDIA on first run if needed),
rem  then starts llama-server.exe. All arguments are passed straight through.
rem
rem  Usage:   llama-server.cmd [any llama-server args]
rem  Example: llama-server.cmd --moe-stream --host 0.0.0.0 --port 8080
rem ---------------------------------------------------------------------------
setlocal
set "HERE=%~dp0"
set "BIN=%HERE%..\..\build\bin"
if not exist "%BIN%\llama-server.exe" set "BIN=%HERE%"

rem Preflight; auto-remediate from official sources if anything is missing.
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%preflight.ps1" -AppDir "%BIN%" -Quiet
if errorlevel 1 (
    echo.
    echo Some runtime dependencies are missing. Attempting to fetch them from the
    echo official NVIDIA / Microsoft sources...
    echo.
    powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%preflight.ps1" -AppDir "%BIN%" -AutoFix
    if errorlevel 1 (
        echo.
        echo Could not satisfy all runtime dependencies automatically. See messages above.
        exit /b 1
    )
)

set "PATH=%BIN%;%PATH%"
"%BIN%\llama-server.exe" %*
