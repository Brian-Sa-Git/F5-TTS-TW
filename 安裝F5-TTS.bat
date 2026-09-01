@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo   F5-TTS Universal Installer
echo ==========================================
echo.

if not exist "%~dp0installer.ps1" (
    echo ERROR: installer.ps1 was not found in this folder.
    echo Please extract the whole ZIP before running this file.
    echo.
    pause
    exit /b 1
)

set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

if not exist "%PS%" (
    echo ERROR: Windows PowerShell was not found.
    echo.
    pause
    exit /b 1
)

"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer.ps1"

set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
    echo Installer exited with error code: %RC%
) else (
    echo Installer finished.
)
echo.
pause
exit /b %RC%
