@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_runpod_pod.ps1" -TrackerName bytetrack -Deploy
if errorlevel 1 (
  echo.
  echo RunPod ByteTrack launch failed.
  pause
  exit /b 1
)
echo.
echo RunPod ByteTrack pod created successfully.
pause
