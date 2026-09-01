@echo off
setlocal
cd /d "%~dp0"

set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
set "INSTALLER=%~dp0installer.ps1"
set "TARGET=C:\AI\F5-TTS-Universal\F5-TTS-src\src\f5_tts\infer\infer_gradio.py"

echo ==========================================
echo   F5-TTS Traditional Chinese v1.0.2
echo   Install / Update
echo ==========================================
echo.

if not exist "%INSTALLER%" (
    echo ERROR: installer.ps1 was not found.
    echo Please extract the whole ZIP before running.
    echo.
    pause
    exit /b 1
)

if not exist "%PS%" (
    echo ERROR: Windows PowerShell was not found.
    echo.
    pause
    exit /b 1
)

if exist "%TARGET%" (
    echo Existing F5-TTS Universal installation detected.
    echo Mode: UPDATE
    echo.
    "%PS%" -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER%" -UpdateOnly
) else (
    echo Existing installation was not detected.
    echo Mode: NEW INSTALL
    echo.
    "%PS%" -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER%"
)

set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
    echo Operation exited with error code: %RC%
) else (
    echo Operation finished successfully.
)
echo.
pause
exit /b %RC%
