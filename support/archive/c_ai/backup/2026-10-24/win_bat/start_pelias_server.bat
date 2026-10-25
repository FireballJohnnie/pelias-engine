@echo off
title Pelias server - stuart-creator (Q4_K_M, tier build)
setlocal

set "BUILD=C:\AI\win-build\llama-pelias-tier"
set "EXE=%BUILD%\build-win\bin\llama-server.exe"
set "MODEL=<project>\gguf\test-model-Q4_K_M.gguf"
set "HOTSET=<project>\logs\pelias_hot_set_b7.0.txt"
set "CUDABIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64"

if not exist "%EXE%" echo Missing: %EXE% & goto fail
if not exist "%MODEL%" echo Missing: %MODEL% & goto fail
if not exist "%HOTSET%" echo Missing: %HOTSET% & goto fail
if not exist "%CUDABIN%" echo Missing: %CUDABIN% & goto fail

netstat -ano | findstr /R /C:":8012 .*LISTENING" >nul
if not errorlevel 1 echo Port 8012 is already in use. Stop the other server first. & goto fail

rem CUDA 13.4 DLLs for this window only. Your main PATH is not touched.
set "PATH=%CUDABIN%;%PATH%"

set "PELIAS_HOT_SET=%HOTSET%"
set "PELIAS_TIER_BYPASS_N=32"
set "PELIAS_TIER_MAX_MIB=7300"
set "PELIAS_TIER_VERBOSE=1"
set "PELIAS_TIER_STATS_FILE=C:\AI\win-build\tier_stats.json"

cd /d "%BUILD%"

echo.
echo Starting stuart-creator on http://127.0.0.1:8012  (model load takes a while)
echo Ctrl+C in this window = clean stop, writes tier_stats.json.
echo Closing this window also ends the server, without the stats file.
echo.

powershell -NoProfile -NoExit -Command ".\build-win\bin\llama-server.exe -m %MODEL% --alias stuart-creator --host 127.0.0.1 --port 8012 --jinja -np 1 -c 32768 -n -1 --no-context-shift -ngl 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --n-cpu-moe 40 --fit off -lv 4 -t 6 -tb 8 --load-mode none -b 4096 -ub 4096"

endlocal
exit /b 0

:fail
echo.
echo Not started.
pause
exit /b 1
