@echo off
title Stop Pelias server (Windows)

tasklist /FI "IMAGENAME eq llama-server.exe" | find /I "llama-server.exe" >nul
if errorlevel 1 echo No Windows llama-server.exe is running. & pause & exit /b 0

echo Running now:
tasklist /FI "IMAGENAME eq llama-server.exe"
echo.
echo This ends the server at once and does NOT write tier_stats.json.
echo For a clean stop press Ctrl+C in the server window instead.
echo A WSL server is not affected. In WSL use: llama --stop
echo.
choice /M "End the Windows server now"
if errorlevel 2 exit /b 0

taskkill /F /IM llama-server.exe
echo.
pause
