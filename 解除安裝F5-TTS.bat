@echo off
setlocal
title F5-TTS Uninstaller

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "TMPBAT=%TEMP%\F5TTS-Uninstall-%RANDOM%-%RANDOM%.bat"

echo ==========================================
echo   F5-TTS Uninstaller
echo ==========================================
echo.
echo Install folder to remove:
echo   %ROOT%
echo.
echo This also removes desktop shortcuts that point to this installation.
echo.
echo It will NOT remove:
echo   Python
echo   Git
echo   NVIDIA Driver
echo   Other F5-TTS installations
echo.
choice /C YN /N /M "Continue uninstall? [Y/N]: "
if errorlevel 2 exit /b 0

> "%TMPBAT%" echo @echo off
>>"%TMPBAT%" echo timeout /t 2 /nobreak ^>nul
>>"%TMPBAT%" echo powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop');$w=New-Object -ComObject WScript.Shell;Get-ChildItem $d -Filter '*.lnk' -ErrorAction SilentlyContinue ^| ForEach-Object {try {$s=$w.CreateShortcut($_.FullName); if($s.TargetPath -like '%ROOT%*'){Remove-Item $_.FullName -Force}} catch {}}"
>>"%TMPBAT%" echo rmdir /s /q "%ROOT%"
>>"%TMPBAT%" echo del /q "%%~f0"

start "" /min cmd /c "%TMPBAT%"
exit /b 0
