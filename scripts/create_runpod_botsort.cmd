@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_runpod_pod.ps1" -TrackerName botsort -Deploy
if errorlevel 1 (
  echo.
  echo RunPod BoT-SORT launch failed.
  pause
  exit /b 1
)
echo.
echo RunPod BoT-SORT pod created successfully.
pause
