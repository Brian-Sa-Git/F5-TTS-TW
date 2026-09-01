@echo off
chcp 65001 >nul
title F5-TTS 繁中通用一鍵安裝器
cd /d "%~dp0"

echo.
echo ==========================================
echo   F5-TTS 繁中通用一鍵安裝器 v1.0.0
echo ==========================================
echo.
echo 這個安裝器不綁定 RTX 型號。
echo 會自動偵測 NVIDIA 顯示卡與驅動能力。
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer.ps1"

echo.
echo 安裝器已結束。
pause
