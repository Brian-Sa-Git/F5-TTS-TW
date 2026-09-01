@echo off
setlocal
title F5-TTS Universal Uninstaller

set "ROOT=C:\AI\F5-TTS-Universal"
set "TMPPS=%TEMP%\F5TTS-Uninstall-%RANDOM%-%RANDOM%.ps1"

echo ==========================================
echo   F5-TTS Universal Uninstaller
echo ==========================================
echo.
echo This will remove the installed folder:
echo   %ROOT%
echo.
echo It will also remove desktop shortcuts that point to this installation.
echo.
echo It will NOT remove:
echo   C:\AI\F5-TTS
echo   Python
echo   Git
echo   NVIDIA Driver
echo.
echo IMPORTANT:
echo Please close F5-TTS and all related command windows first.
echo.
choice /C YN /N /M "Continue uninstall? [Y/N]: "
if errorlevel 2 exit /b 0

if not exist "%ROOT%" (
    echo.
    echo The install folder does not exist:
    echo   %ROOT%
    echo.
    pause
    exit /b 0
)

> "%TMPPS%" echo Start-Sleep -Seconds 2
>>"%TMPPS%" echo $root = 'C:\AI\F5-TTS-Universal'
>>"%TMPPS%" echo $desktop = [Environment]::GetFolderPath('Desktop')
>>"%TMPPS%" echo try {
>>"%TMPPS%" echo   $w = New-Object -ComObject WScript.Shell
>>"%TMPPS%" echo   Get-ChildItem -LiteralPath $desktop -Filter '*.lnk' -ErrorAction SilentlyContinue ^| ForEach-Object {
>>"%TMPPS%" echo     try {
>>"%TMPPS%" echo       $s = $w.CreateShortcut($_.FullName)
>>"%TMPPS%" echo       if ($s.TargetPath -like ($root + '*')) { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue }
>>"%TMPPS%" echo     } catch {}
>>"%TMPPS%" echo   }
>>"%TMPPS%" echo } catch {}
>>"%TMPPS%" echo try {
>>"%TMPPS%" echo   Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction Stop
>>"%TMPPS%" echo } catch {
>>"%TMPPS%" echo   Add-Type -AssemblyName PresentationFramework
>>"%TMPPS%" echo   [System.Windows.MessageBox]::Show(('Could not remove all files. Close F5-TTS and try again.' + [Environment]::NewLine + $_.Exception.Message),'F5-TTS Uninstaller') ^| Out-Null
>>"%TMPPS%" echo   exit 1
>>"%TMPPS%" echo }
>>"%TMPPS%" echo Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue

start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%TMPPS%"
exit /b 0
