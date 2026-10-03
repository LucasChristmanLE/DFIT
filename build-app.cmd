@echo off
REM Double-click this to build the FracClosure distribution zip.
REM Runs the .ps1 with execution policy bypassed and keeps the window open at the end.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-app.ps1"
pause
