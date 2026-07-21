@echo off
REM Double-click this to launch the DFIT tool with no file loaded.
REM Runs the .ps1 with execution policy bypassed and keeps the window open on error.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-app.ps1"
