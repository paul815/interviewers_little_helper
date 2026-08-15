@echo off
REM ===========================================================================
REM  Interviewer's Little Helper - Windows launcher.
REM
REM  Double-click target of "Launch ILH.lnk" in the repository root.
REM
REM  Flow: check .venv (run setup.py if missing) -> hand off to a hidden copy of
REM  this script that owns the app process -> this visible window polls the port
REM  and closes itself once the app is up. If the app never comes up, the window
REM  stays open with log paths and the tail of the runner log.
REM
REM  Keep this file pure ASCII. Under chcp 65001 a single multi-byte character
REM  makes cmd.exe lose its place in the file and run the wrong branch.
REM ===========================================================================
setlocal EnableDelayedExpansion
title Interviewer's Little Helper

REM Repository root is one level up from tools\
for %%D in ("%~dp0..") do set "ROOT=%%~fD\"
cd /d "%ROOT%"

set "VENV_DIR=%ROOT%.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "LOG_DIR=%ROOT%logs"
REM Seconds to wait for the app to answer before giving up and showing logs.
REM Pre-set in the environment to shorten it when testing the failure path.
if not defined ILH_WAIT_SECONDS set "ILH_WAIT_SECONDS=120"
set "ILH_RUN_ID=%RANDOM%%RANDOM%"
set "ILH_STAGE_FILE=%LOG_DIR%\launcher-stage.txt"
set "ILH_LAUNCHER_LOG=%LOG_DIR%\launcher.log"
set "ILH_RUNNER_LOG=%LOG_DIR%\launcher-runner.log"
set "ILH_PID_FILE=%LOG_DIR%\launcher.pid"
set "ILH_HIDDEN_CMD=%LOG_DIR%\launcher-hidden-%ILH_RUN_ID%.cmd"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>nul

if not exist "%VENV_PY%" call :run_installer
if not exist "%VENV_PY%" exit /b 1

call :detect_port
set "ILH_URL=http://127.0.0.1:%ILH_PORT%"

if /I "%ILH_DIAGNOSTIC_LAUNCH%"=="1" goto :diagnostic_main
if /I "%ILH_HIDDEN_LAUNCH%"=="1" goto :hidden_main

REM --- visible branch: hand off, then watch --------------------------------
call :reset_launcher_artifacts
call :probe_app
if not errorlevel 1 goto :already_running
call :relaunch_hidden
if errorlevel 1 exit /b 1
call :wait_for_background_launch
exit /b 0

:already_running
echo.
echo   Interviewer's Little Helper is already running on %ILH_URL%.
echo   Its window is small and always-on-top - it may be hiding behind Zoom.
echo   Opening a browser tab as a fallback view.
call :open_browser "%ILH_URL%"
call :sleep 4
exit /b 0

REM --- hidden branch: this process owns the app -----------------------------
:hidden_main
call :log_event "hidden launcher entered"
call :write_stage "starting application"
"%VENV_PY%" -m app.main
set "APP_EXIT=%errorlevel%"
call :log_event "application exited with code %APP_EXIT%"
call :write_stage "application exited with code %APP_EXIT%"
del /q "%ILH_PID_FILE%" >nul 2>nul
exit /b %APP_EXIT%

REM --- diagnostic branch: run visibly, keep the window ----------------------
:diagnostic_main
title Interviewer's Little Helper - Startup Diagnostics
echo.
echo   Diagnostic mode: starting the app in this window so failures are visible.
call :print_log_paths
echo.
call :probe_app
if not errorlevel 1 (
    echo   Something is already listening on %ILH_URL% - stop it first.
    pause
    exit /b 1
)
call :write_stage "diagnostic launch entered"
"%VENV_PY%" -m app.main
echo.
echo   Application exited with code %errorlevel%.
call :print_log_paths
echo.
echo   This window stays open for diagnostics.
pause
exit /b 0

REM --- setup ----------------------------------------------------------------
:run_installer
echo.
echo   No Python environment yet ^(.venv is missing^).
echo   Running setup.py - this downloads dependencies and ASR weights.
echo.
where python >nul 2>nul
if not errorlevel 1 (
    python "%ROOT%setup.py"
    goto :installer_done
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%ROOT%setup.py"
    goto :installer_done
)
echo.
echo   Python 3.11+ was not found in PATH.
echo   Install it from https://www.python.org/downloads/ ^(tick "Add python.exe
echo   to PATH"^), then double-click the shortcut again.
echo.
pause
goto :eof

:installer_done
if not exist "%VENV_PY%" (
    echo.
    echo   Setup did not finish - .venv still has no interpreter.
    echo   Re-run it by hand to see the error:  python setup.py
    echo.
    pause
)
goto :eof

REM --- helpers --------------------------------------------------------------
REM Port comes from tools\launcher_port.py via a temp file, not from a FOR /F:
REM cmd.exe strips commas out of a FOR /F command line, which silently mangles
REM any python one-liner long enough to be useful. An empty/failed run leaves
REM the default in place, because SET /P keeps the old value on empty input.
:detect_port
set "ILH_PORT=8756"
"%VENV_PY%" -m tools.launcher_port > "%LOG_DIR%\launcher-port.txt" 2>nul
if exist "%LOG_DIR%\launcher-port.txt" set /p ILH_PORT=<"%LOG_DIR%\launcher-port.txt"
del /q "%LOG_DIR%\launcher-port.txt" >nul 2>nul
goto :eof

:reset_launcher_artifacts
del /q "%LOG_DIR%\launcher-hidden-*.cmd" >nul 2>nul
del /q "%ILH_STAGE_FILE%" >nul 2>nul
del /q "%ILH_RUNNER_LOG%" >nul 2>nul
goto :eof

:log_event
set "ILH_EVENT=%~1"
>> "%ILH_LAUNCHER_LOG%" echo [%date% %time%] %ILH_EVENT%
goto :eof

:write_stage
set "ILH_STAGE=%~1"
> "%ILH_STAGE_FILE%" echo %ILH_STAGE%
call :log_event "stage: %ILH_STAGE%"
goto :eof

:relaunch_hidden
call :write_stage "launcher handoff started"
(
    echo @echo off
    echo setlocal
    echo set "ILH_HIDDEN_LAUNCH=1"
    echo call "%~f0" ^>^> "%ILH_RUNNER_LOG%" 2^>^&1
) > "%ILH_HIDDEN_CMD%"
REM The PID is written straight to a file rather than captured with FOR /F:
REM cmd.exe mangles semicolons and parentheses inside a FOR /F command line,
REM and this PowerShell one-liner is full of both.
set "ILH_PID="
del /q "%ILH_PID_FILE%" >nul 2>nul
powershell -NoProfile -Command "$p = Start-Process -WindowStyle Hidden -WorkingDirectory '%ROOT%' -FilePath '%ILH_HIDDEN_CMD%' -PassThru; if ($p) { Write-Output $p.Id }" > "%ILH_PID_FILE%" 2>nul
if exist "%ILH_PID_FILE%" set /p ILH_PID=<"%ILH_PID_FILE%"
if not defined ILH_PID (
    call :write_stage "launcher handoff failed"
    echo   Could not start the background launcher. See %ILH_LAUNCHER_LOG%.
    pause
    exit /b 1
)
call :log_event "launcher handoff pid !ILH_PID!"
goto :eof

:wait_for_background_launch
echo.
echo   Starting Interviewer's Little Helper...
call :print_stage_if_changed
set WAIT_COUNT=0
:waitbgloop
call :probe_app
if not errorlevel 1 goto :launch_ready
if !WAIT_COUNT! GEQ %ILH_WAIT_SECONDS% goto :launch_timeout
call :sleep 1
set /a WAIT_COUNT+=1
call :print_stage_if_changed
goto :waitbgloop

:launch_ready
call :log_event "app ready on %ILH_URL%"
echo   Ready. The app window is 600x400 and stays on top of other windows.
call :sleep 2
goto :eof

:launch_timeout
call :write_stage "timed out waiting for the app to answer on %ILH_URL%"
echo.
echo   The app did not answer on %ILH_URL% within %ILH_WAIT_SECONDS% seconds.
call :print_log_paths
call :print_runner_tail
echo.
echo   To watch the failure live, run this from a terminal:
echo       set ILH_DIAGNOSTIC_LAUNCH=1 ^&^& "%~f0"
echo.
pause
goto :eof

:print_stage_if_changed
set "ILH_STATUS=starting..."
if exist "%ILH_STAGE_FILE%" set /p ILH_STATUS=<"%ILH_STAGE_FILE%"
if not defined LAST_STATUS (
    echo   Status: !ILH_STATUS!
) else if /I not "!LAST_STATUS!"=="!ILH_STATUS!" (
    echo   Status: !ILH_STATUS!
)
set "LAST_STATUS=!ILH_STATUS!"
goto :eof

:print_log_paths
echo   Launcher log:     %ILH_LAUNCHER_LOG%
echo   Runner log:       %ILH_RUNNER_LOG%
echo   Application log:  %LOG_DIR%\app.log
goto :eof

:print_runner_tail
if not exist "%ILH_RUNNER_LOG%" goto :eof
echo.
echo   --- last lines of the runner log ---
powershell -NoProfile -Command "Get-Content -LiteralPath '%ILH_RUNNER_LOG%' -Tail 20"
echo   --- end ---
goto :eof

:probe_app
powershell -NoProfile -Command "try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1', %ILH_PORT%); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>nul
goto :eof

REM TIMEOUT.EXE refuses to run when stdin is redirected ("Input redirection is
REM not supported"), which happens whenever this script is driven by another
REM tool rather than double-clicked. Start-Sleep has no such restriction.
:sleep
powershell -NoProfile -Command "Start-Sleep -Seconds %~1" >nul 2>nul
goto :eof

:open_browser
powershell -NoProfile -Command "Start-Process -FilePath '%~1'" >nul 2>nul
if not errorlevel 1 goto :eof
start "" "%~1" >nul 2>nul
goto :eof
