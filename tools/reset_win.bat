@echo off
REM ===========================================================================
REM  Interviewer's Little Helper - factory reset (Windows).
REM
REM  Returns the working copy to a pristine source tree: stops the app, then
REM  removes everything at the repository root that is not part of the sources
REM  (.venv, logs, sessions, projects, guides, config.json, caches, stray files).
REM
REM  NOT removed - these live outside the repository and cost hours to refetch:
REM    * ASR weights in the HuggingFace cache (%%USERPROFILE%%\.cache\huggingface)
REM    * Ollama models
REM
REM  Keep this file pure ASCII. Under chcp 65001 a single multi-byte character
REM  makes cmd.exe lose its place in the file and run the wrong branch.
REM ===========================================================================
setlocal EnableDelayedExpansion

for %%D in ("%~dp0..") do set "ROOT=%%~fD\"
cd /d "%ROOT%"

REM --- Options ---------------------------------------------------------------
REM --keep-venv   DEV ONLY. Keeps .venv, so the multi-gigabyte stack (torch,
REM               onnxruntime, pywebview) survives the reset and the next run
REM               starts in seconds instead of minutes. Because launch_win.bat
REM               only runs setup.py when .venv is missing, this ALSO skips the
REM               installer - so it does not exercise first-run setup. Must be
REM               OFF when verifying a release.
set "KEEP_VENV="
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--keep-venv" goto arg_keep_venv
echo.
echo   Unknown option: %~1
echo   Usage: reset_win.bat [--keep-venv]
exit /b 2
:arg_keep_venv
set "KEEP_VENV=1"
shift
goto parse_args
:args_done

REM Source entries that survive the reset. Everything else at the root goes.
set "KEEP_LIST=".claude" ".editorconfig" ".git" ".github" ".gitattributes" ".gitignore" ".pre-commit-config.yaml" ".python-version" "AGENTS.md" "CLAUDE.md" "LICENSE" "PLAN.md" "README.md" "app" "config.example.json" "docs" "examples" "Launch ILH.lnk" "requirements-ci.txt" "requirements-common.txt" "requirements-dev.txt" "requirements-macos.txt" "requirements-windows.txt" "setup.py" "tests" "tools""

echo.
echo ============================================================
echo   WARNING: this PERMANENTLY removes all local app data.
echo   Projects, sessions, transcripts, reports, guides, logs and
echo   your config.json will be deleted. This cannot be undone.
echo ============================================================
echo.
echo   Root: %ROOT%
echo.
echo   Will be removed:
call :list_targets
echo.
echo   Kept: sources, git history, ASR weights in the HuggingFace cache,
echo         and Ollama models ^(both live outside this folder^).
if defined KEEP_VENV (
    echo.
    echo   DEV MODE ^(--keep-venv^): .venv is PRESERVED, so setup.py will NOT
    echo   re-run on the next launch and first-run setup is not re-tested.
)
echo.
set /p CONFIRM="Type Yes and press Enter to proceed: "
if /I not "%CONFIRM%"=="Yes" (
    echo Reset cancelled.
    exit /b 0
)
echo.

REM Stop ONLY this install's Python processes. Never blanket-kill python.exe or
REM pythonw.exe by image name - that would also take down the user's unrelated
REM Python (Jupyter, other apps). The kill is scoped two ways:
REM   1) the launcher PID recorded in logs\launcher.pid, plus its children (/t),
REM   2) any python(w).exe whose command line runs from THIS directory.
echo Stopping the application...
set "ILH_PIDFILE=%ROOT%logs\launcher.pid"
if exist "%ILH_PIDFILE%" (
    set "ILH_PID="
    set /p ILH_PID=<"%ILH_PIDFILE%"
    if defined ILH_PID call :kill_launcher_tree !ILH_PID!
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$dir=[regex]::Escape('%ROOT%'); Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match $dir } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {} }" >nul 2>&1
powershell -NoProfile -Command "Start-Sleep -Seconds 2" >nul 2>nul

set FAIL=0
for /f "delims=" %%I in ('dir /b /a "%ROOT%."') do (
    set "NAME=%%~I"
    call :is_kept "%%~I"
    if "!KEEP!"=="0" call :remove_entry "%%~I"
)

REM Purge bytecode and test caches so the reset leaves a pristine source tree.
REM With --keep-venv the sweep is scoped to the source folders: recursing into a
REM preserved .venv would churn thousands of site-packages caches for no gain
REM and spend exactly the time the flag exists to save.
echo Removing Python caches...
if defined KEEP_VENV (
    REM FOR /R will not take a FOR variable as its root - it fails to parse - so
    REM each source folder is swept in a subroutine where the root is a plain
    REM parameter.
    for %%S in (app tests tools docs examples) do if exist "%%S\" call :purge_caches "%%S"
) else (
    for /d /r "%ROOT%." %%D in (__pycache__ .pytest_cache) do if exist "%%~D" rd /s /q "%%~D" >nul 2>nul
    del /s /q "%ROOT%*.pyc" >nul 2>nul
    del /s /q "%ROOT%*.pyo" >nul 2>nul
)

echo.
if %FAIL%==1 (
    echo Reset INCOMPLETE. Some entries could not be removed.
    echo Close the app, any terminals and editors holding those files, then retry.
    exit /b 1
)
if defined KEEP_VENV (
    echo Reset complete. .venv preserved - the next launch skips setup.py.
    echo This is a DEV shortcut - run without --keep-venv before a release.
) else (
    echo Reset complete. Double-click "Launch ILH.lnk" to set the app up again.
)
echo ASR weights and Ollama models were left untouched.
endlocal
exit /b 0

REM --- helpers ---------------------------------------------------------------

REM Kill the recorded launcher and its children - but only after confirming the
REM PID still belongs to THIS install. Windows recycles PIDs, and a stale
REM logs\launcher.pid must never take down an unrelated process.
:kill_launcher_tree
powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process -Filter \"ProcessId=%~1\" -ErrorAction SilentlyContinue; if ($p -and $p.CommandLine -and ($p.CommandLine -match [regex]::Escape('%ROOT%'))) { exit 0 } else { exit 1 }" >nul 2>&1
if errorlevel 1 goto :eof
taskkill /f /t /pid %~1 >nul 2>&1
goto :eof

REM Sets KEEP=1 when the root entry %1 is part of the sources.
:is_kept
set "KEEP=0"
for %%K in (%KEEP_LIST%) do if /I "%~1"=="%%~K" set "KEEP=1"
if defined KEEP_VENV if /I "%~1"==".venv" set "KEEP=1"
goto :eof

REM Prints what a reset would delete, without touching anything.
:list_targets
set "FOUND=0"
for /f "delims=" %%I in ('dir /b /a "%ROOT%."') do (
    call :is_kept "%%~I"
    if "!KEEP!"=="0" (
        echo     - %%~I
        set "FOUND=1"
    )
)
if "!FOUND!"=="0" echo     ^(nothing - the tree is already clean^)
goto :eof

:remove_entry
if exist "%ROOT%%~1\" (
    echo Removing %~1...
    rd /s /q "%ROOT%%~1"
    if exist "%ROOT%%~1\" (
        echo [ERROR] Could not remove %~1 - files may still be in use.
        set FAIL=1
    )
    goto :eof
)
echo Removing %~1...
del /f /q "%ROOT%%~1" >nul 2>nul
if exist "%ROOT%%~1" (
    echo [ERROR] Could not remove %~1 - files may still be in use.
    set FAIL=1
)
goto :eof

REM Sweep __pycache__/.pytest_cache and stray bytecode under one source folder.
REM Only reached from the --keep-venv purge branch above.
:purge_caches
for /d /r "%~1" %%D in (__pycache__ .pytest_cache) do if exist "%%~D" rd /s /q "%%~D" >nul 2>nul
del /s /q "%~1\*.pyc" >nul 2>nul
del /s /q "%~1\*.pyo" >nul 2>nul
exit /b 0
