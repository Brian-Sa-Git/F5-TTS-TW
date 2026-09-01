$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $PackageRoot "config.json"

# 優先從自己的 GitHub 取得最新設定。
# 如果沒有網路、GitHub 暫時無法連線或設定檔格式錯誤，會自動退回安裝包內的 config.json。
$RemoteConfigUrl = "https://raw.githubusercontent.com/Brian-Sa-Git/F5-TTS-TW/main/config.json"
$Config = $null

try {
    Write-Host "[資訊] 正在取得 GitHub 最新安裝設定..." -ForegroundColor Gray
    $RemoteConfigText = (Invoke-WebRequest -Uri $RemoteConfigUrl -UseBasicParsing -TimeoutSec 15).Content
    $Config = $RemoteConfigText | ConvertFrom-Json
    Write-Host "[完成] 已載入 GitHub 最新設定。" -ForegroundColor Green
}
catch {
    Write-Host "[注意] 無法取得 GitHub 最新設定，改用安裝包內建設定。" -ForegroundColor Yellow
    $Config = Get-Content -Raw -Encoding UTF8 $ConfigPath | ConvertFrom-Json
}

function Section([string]$Text) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Info([string]$Text) {
    Write-Host "[資訊] $Text" -ForegroundColor Gray
}

function Good([string]$Text) {
    Write-Host "[完成] $Text" -ForegroundColor Green
}

function Warn([string]$Text) {
    Write-Host "[注意] $Text" -ForegroundColor Yellow
}

function Fail([string]$Text) {
    Write-Host "[錯誤] $Text" -ForegroundColor Red
}

function Has-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Ensure-Winget {
    if (-not (Has-Command "winget")) {
        throw "找不到 winget。請先從 Microsoft Store 安裝/更新「App Installer」，再重新執行。"
    }
}

function Winget-Install([string]$Id) {
    Ensure-Winget
    Info "正在使用 winget 安裝：$Id"
    & winget install --id $Id -e --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Warn "winget 回傳非 0。若該軟體其實已安裝，可繼續；否則請查看上方訊息。"
    }
}

function Find-Python311 {
    if (Has-Command "py") {
        try {
            $p = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
        } catch {}
    }

    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python311\python.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }

    if (Has-Command "python") {
        try {
            $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($ver.Trim() -eq "3.11") {
                return (& python -c "import sys; print(sys.executable)").Trim()
            }
        } catch {}
    }

    return $null
}

function Find-Git {
    if (Has-Command "git") {
        return (Get-Command git).Source
    }
    $candidate = "$env:ProgramFiles\Git\cmd\git.exe"
    if (Test-Path $candidate) { return $candidate }
    return $null
}

function Find-FFmpegSharedBin {
    # 優先找 winget 的 Shared 版本，因為 TorchCodec 在 Windows 需要 shared DLL。
    $root = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path $root) {
        $dirs = Get-ChildItem -Path $root -Directory -Filter "Gyan.FFmpeg.Shared*" -ErrorAction SilentlyContinue
        foreach ($d in $dirs) {
            $dll = Get-ChildItem -Path $d.FullName -Recurse -Filter "avcodec-*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($dll) { return $dll.Directory.FullName }
        }
    }

    # 退而求其次：PATH 中 ffmpeg 所在資料夾若含 shared DLL 也可用。
    if (Has-Command "ffmpeg") {
        $ff = (Get-Command ffmpeg).Source
        $dir = Split-Path -Parent $ff
        if (Get-ChildItem -Path $dir -Filter "avcodec-*.dll" -ErrorAction SilentlyContinue) {
            return $dir
        }
    }
    return $null
}

function Get-NvidiaInfo {
    $result = [ordered]@{
        Present = $false
        Name = ""
        Driver = ""
        ReportedCuda = ""
        TorchIndex = "cpu"
        TorchVersion = "2.8.0"
        TorchCodecVersion = "0.7.0"
        Mode = "CPU"
    }

    if (-not (Has-Command "nvidia-smi")) {
        return [pscustomobject]$result
    }

    $result.Present = $true

    try {
        $gpuLine = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
        if ($gpuLine) {
            $parts = $gpuLine -split ","
            $result.Name = $parts[0].Trim()
            if ($parts.Count -gt 1) { $result.Driver = $parts[1].Trim() }
        }
    } catch {}

    try {
        $full = (& nvidia-smi 2>$null | Out-String)
        $m = [regex]::Match($full, "CUDA Version:\s*([0-9]+\.[0-9]+)")
        if ($m.Success) {
            $result.ReportedCuda = $m.Groups[1].Value
            $cuda = [version]$result.ReportedCuda

            # 不綁 RTX 型號；只依 Driver 可支援的 CUDA 上限選已知相容組合。
            if ($cuda -ge [version]"13.0") {
                $result.TorchIndex = "cu130"
                $result.TorchVersion = "2.10.0"
                $result.TorchCodecVersion = "0.10.0"
                $result.Mode = "NVIDIA CUDA"
            }
            elseif ($cuda -ge [version]"12.8") {
                $result.TorchIndex = "cu128"
                $result.TorchVersion = "2.9.1"
                $result.TorchCodecVersion = "0.9.0"
                $result.Mode = "NVIDIA CUDA"
            }
            elseif ($cuda -ge [version]"12.6") {
                $result.TorchIndex = "cu126"
                $result.TorchVersion = "2.8.0"
                $result.TorchCodecVersion = "0.7.0"
                $result.Mode = "NVIDIA CUDA"
            }
            else {
                $result.TorchIndex = "cpu"
                $result.TorchVersion = "2.8.0"
                $result.TorchCodecVersion = "0.7.0"
                $result.Mode = "CPU fallback"
            }
        }
    } catch {}

    return [pscustomobject]$result
}

function Copy-DirectoryContents([string]$Source, [string]$Destination) {
    if (-not (Test-Path $Source)) { return }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
}

Section "1/9  選擇安裝位置"

$defaultDir = $Config.default_install_dir
$answer = Read-Host "安裝位置（直接 Enter 使用 $defaultDir）"
if ([string]::IsNullOrWhiteSpace($answer)) {
    $InstallDir = $defaultDir
} else {
    $InstallDir = $answer.Trim().Trim('"')
}

$SourceDir = Join-Path $InstallDir "F5-TTS-src"
$VenvDir = Join-Path $InstallDir ".venv"
$ModelsDir = Join-Path $InstallDir "models\huggingface"
$RuntimeFFmpegDir = Join-Path $InstallDir "runtime\ffmpeg\bin"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

Info "安裝位置：$InstallDir"

Section "2/9  偵測電腦硬體"

$Gpu = Get-NvidiaInfo
if ($Gpu.Present) {
    Good "偵測到 NVIDIA：$($Gpu.Name)"
    Info "Driver：$($Gpu.Driver)"
    Info "nvidia-smi 顯示可支援 CUDA：$($Gpu.ReportedCuda)"
    Info "PyTorch 模式：$($Gpu.Mode) / $($Gpu.TorchIndex)"
    Info "相容版本：PyTorch $($Gpu.TorchVersion) / TorchCodec $($Gpu.TorchCodecVersion)"
    if ($Gpu.TorchIndex -eq "cpu") {
        Warn "NVIDIA Driver 顯示的 CUDA 能力低於目前安裝器的 GPU wheel 門檻，將先使用 CPU 模式。之後可更新顯卡驅動再重裝 GPU 版。"
    }
} else {
    Warn "未偵測到可用的 NVIDIA nvidia-smi，將安裝 CPU 版 PyTorch。"
    Info "這不會綁死硬體；之後換 NVIDIA 電腦重新執行安裝器即可自動選 GPU 版。"
}

Section "3/9  準備 Python / Git / Portable FFmpeg"

$PythonExe = Find-Python311
if (-not $PythonExe) {
    Winget-Install $Config.python_winget_id
    $PythonExe = Find-Python311
}
if (-not $PythonExe) {
    throw "Python 3.11 安裝後仍找不到，請重新開啟此安裝器再試一次。"
}
Good "Python 3.11：$PythonExe"

$GitExe = Find-Git
if (-not $GitExe) {
    Winget-Install $Config.git_winget_id
    $GitExe = Find-Git
}
if (-not $GitExe) {
    throw "Git 安裝後仍找不到，請重新開啟此安裝器再試一次。"
}
Good "Git：$GitExe"

# 使用安裝目錄內的 Portable FFmpeg 7.1.1，不安裝、不更新 Windows 全域 FFmpeg。
$FFmpegExe = Join-Path $RuntimeFFmpegDir "ffmpeg.exe"
$FFmpegDll = Get-ChildItem -Path $RuntimeFFmpegDir -Filter "avcodec-*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1

if (-not (Test-Path $FFmpegExe) -or -not $FFmpegDll) {
    Info "準備 Portable FFmpeg 7.1.1 Shared（不修改系統 FFmpeg）..."

    $FFmpegUrl = "https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/ffmpeg-7.1.1-full_build-shared.zip"
    $TempZip = Join-Path $env:TEMP "f5tts_ffmpeg_7.1.1_shared.zip"
    $TempDir = Join-Path $env:TEMP "f5tts_ffmpeg_7.1.1_shared"

    if (Test-Path $TempZip) { Remove-Item $TempZip -Force }
    if (Test-Path $TempDir) { Remove-Item $TempDir -Recurse -Force }

    Invoke-WebRequest -Uri $FFmpegUrl -OutFile $TempZip -UseBasicParsing
    Expand-Archive -Path $TempZip -DestinationPath $TempDir -Force

    $FoundFFmpeg = Get-ChildItem -Path $TempDir -Recurse -Filter "ffmpeg.exe" -ErrorAction Stop |
        Where-Object { $_.Directory.Name -eq "bin" } |
        Select-Object -First 1

    if (-not $FoundFFmpeg) {
        throw "Portable FFmpeg 解壓後找不到 ffmpeg.exe。"
    }

    $FFBin = $FoundFFmpeg.Directory.FullName
    New-Item -ItemType Directory -Force -Path $RuntimeFFmpegDir | Out-Null
    Copy-Item -Path (Join-Path $FFBin "*") -Destination $RuntimeFFmpegDir -Force

    Remove-Item $TempZip -Force -ErrorAction SilentlyContinue
    Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path $FFmpegExe)) {
    throw "Portable FFmpeg 建立失敗。"
}
Good "Portable FFmpeg 7.1.1 Shared：$RuntimeFFmpegDir"

Section "4/9  下載 F5-TTS 程式"

if (Test-Path (Join-Path $SourceDir ".git")) {
    Info "偵測到既有 F5-TTS 原始碼，更新指定版本。"
    & $GitExe -C $SourceDir fetch --tags --force
    & $GitExe -C $SourceDir checkout $Config.f5_ref
} else {
    if (Test-Path $SourceDir) {
        Warn "既有 $SourceDir 不是 Git repository，將改名備份。"
        $backup = "$SourceDir.backup.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Move-Item $SourceDir $backup
    }
    & $GitExe clone --branch $Config.f5_ref --depth 1 $Config.f5_repo $SourceDir
    if ($LASTEXITCODE -ne 0) { throw "F5-TTS Git clone 失敗。" }
}
Good "F5-TTS 原始碼已準備完成（$($Config.f5_ref)）"

Section "5/9  建立 Python 環境與相容 PyTorch / TorchCodec"

if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    & $PythonExe -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

& $VenvPython -m pip install --upgrade pip setuptools wheel

$torchIndex = $Gpu.TorchIndex
$torchVersion = $Gpu.TorchVersion
$torchCodecVersion = $Gpu.TorchCodecVersion

if ($torchIndex -eq "cpu") {
    Info "安裝 PyTorch $torchVersion CPU..."
    & $VenvPython -m pip install "torch==$torchVersion" "torchaudio==$torchVersion" --index-url "https://download.pytorch.org/whl/cpu"
} else {
    Info "安裝 PyTorch $torchVersion / $torchIndex..."
    & $VenvPython -m pip install "torch==$torchVersion" "torchaudio==$torchVersion" --index-url "https://download.pytorch.org/whl/$torchIndex"
}
if ($LASTEXITCODE -ne 0) { throw "PyTorch 安裝失敗。" }

Push-Location $SourceDir
try {
    & $VenvPython -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "F5-TTS Python 套件安裝失敗。" }
} finally {
    Pop-Location
}

# F5-TTS 目前對 torch / torchcodec 沒有鎖定配對版本；
# 安裝依賴後再次固定為相容組合，避免 pip 自動裝到不相容版本。
if ($torchIndex -eq "cpu") {
    & $VenvPython -m pip install --upgrade --force-reinstall "torch==$torchVersion" "torchaudio==$torchVersion" --index-url "https://download.pytorch.org/whl/cpu"
} else {
    & $VenvPython -m pip install --upgrade --force-reinstall "torch==$torchVersion" "torchaudio==$torchVersion" --index-url "https://download.pytorch.org/whl/$torchIndex"
}
if ($LASTEXITCODE -ne 0) { throw "固定 PyTorch 相容版本失敗。" }

# Windows 的 TTS 音訊解碼不需要 TorchCodec CUDA 解碼 wheel；
# 使用 PyPI 的相容 TorchCodec 版本即可，並沿用 Portable FFmpeg shared DLL。
& $VenvPython -m pip install --upgrade --force-reinstall "torchcodec==$torchCodecVersion"
if ($LASTEXITCODE -ne 0) { throw "TorchCodec $torchCodecVersion 安裝失敗。" }

Good "F5-TTS Python 環境完成：PyTorch $torchVersion / TorchCodec $torchCodecVersion"

Section "6/9  驗證 TorchCodec / Portable FFmpeg"

$env:PATH = "$RuntimeFFmpegDir;$env:PATH"

$site = (& $VenvPython -c "import site; print(site.getsitepackages()[0])").Trim()
$TorchCodecDir = Join-Path $site "torchcodec"

if (Test-Path $TorchCodecDir) {
    Copy-Item -Path (Join-Path $RuntimeFFmpegDir "*.dll") -Destination $TorchCodecDir -Force
    Info "已把 FFmpeg shared DLL 複製到 TorchCodec 目錄。"
}

$testOk = $true
try {
    & $VenvPython -c "import torch, torchcodec; print('Torch:', torch.__version__); print('TorchCodec:', torchcodec.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    if ($LASTEXITCODE -ne 0) { $testOk = $false }
} catch {
    $testOk = $false
}

if (-not $testOk) {
    Warn "TorchCodec 第一次測試失敗，嘗試安裝 Microsoft VC++ Runtime 後重試。"
    Winget-Install $Config.vcredist_winget_id
    if (Test-Path $TorchCodecDir) {
        Copy-Item -Path (Join-Path $RuntimeFFmpegDir "*.dll") -Destination $TorchCodecDir -Force
    }
    & $VenvPython -c "import torch, torchcodec; print('Torch:', torch.__version__); print('TorchCodec:', torchcodec.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "TorchCodec 仍無法載入。請保留這個視窗的錯誤訊息。"
    }
}
Good "TorchCodec / FFmpeg 測試通過"

Section "7/9  套用繁體中文客製版"

$CustomTarget = Join-Path $SourceDir "src\f5_tts\infer\infer_gradio.py"
$CustomBackup = "$CustomTarget.original"

if (-not (Test-Path $CustomBackup)) {
    Copy-Item $CustomTarget $CustomBackup -Force
}

if ($Config.custom_ui_url -and -not [string]::IsNullOrWhiteSpace($Config.custom_ui_url)) {
    Info "從 GitHub/網路下載客製 infer_gradio.py..."
    Invoke-WebRequest -Uri $Config.custom_ui_url -OutFile $CustomTarget -UseBasicParsing
} else {
    $BundledUI = Join-Path $PackageRoot "payload\infer_gradio.py"
    Copy-Item $BundledUI $CustomTarget -Force
}
Good "繁中介面、句尾停頓、年份安全辨識、常用 Seed 收藏已套用"

# 若上游舊版 finetune_gradio.py 仍存在 show_api 參數，移除它以相容新版 Gradio。
$FinetuneFile = Join-Path $SourceDir "src\f5_tts\train\finetune_gradio.py"
if (Test-Path $FinetuneFile) {
    $ft = Get-Content -Raw -Encoding UTF8 $FinetuneFile
    if ($ft.Contains(", show_api=api")) {
        $ft = $ft.Replace(", show_api=api", "")
        Set-Content -Path $FinetuneFile -Value $ft -Encoding UTF8
        Info "已自動套用 Finetune Gradio 相容修正。"
    }
}

Section "8/9  模型選擇"

# 若安裝包旁邊已有攜帶式 HF cache，先合併。
$PortableHF = Join-Path $PackageRoot "portable_models\huggingface"
if (Test-Path $PortableHF) {
    $hasFiles = Get-ChildItem -Path $PortableHF -Force -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hasFiles) {
        Info "偵測到隨身攜帶模型快取，正在合併..."
        Copy-DirectoryContents $PortableHF $ModelsDir
        Good "隨身模型已合併"
    }
}

$env:HF_HOME = $ModelsDir
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ModelsDir "hub"

Write-Host ""
Write-Host "F5-TTS v1 + Vocos 是基本使用所需，建議現在下載。" -ForegroundColor White
$dlBase = Read-Host "現在預先下載 F5-TTS v1 + Vocos？[Y/n]"
if ([string]::IsNullOrWhiteSpace($dlBase) -or $dlBase.Trim().ToLower() -ne "n") {
    $py = @"
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='SWivid/F5-TTS', filename='F5TTS_v1_Base/model_1250000.safetensors')
hf_hub_download(repo_id='SWivid/F5-TTS', filename='F5TTS_v1_Base/vocab.txt')
hf_hub_download(repo_id='charactr/vocos-mel-24khz', filename='config.yaml')
hf_hub_download(repo_id='charactr/vocos-mel-24khz', filename='pytorch_model.bin')
print('F5-TTS v1 + Vocos 完成')
"@
    & $VenvPython -c $py
}

$dlE2 = Read-Host "預先下載 E2-TTS？[y/N]"
if ($dlE2.Trim().ToLower() -eq "y") {
    & $VenvPython -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='SWivid/E2-TTS', filename='E2TTS_Base/model_1200000.safetensors'); print('E2-TTS 完成')"
}

$dlWhisper = Read-Host "預先下載 Whisper large-v3-turbo（自動辨識參考文字）？[Y/n]"
if ([string]::IsNullOrWhiteSpace($dlWhisper) -or $dlWhisper.Trim().ToLower() -ne "n") {
    & $VenvPython -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai/whisper-large-v3-turbo'); print('Whisper 完成')"
}

$dlQwen = Read-Host "預先下載 Qwen2.5-3B Voice Chat 模型（容量較大）？[y/N]"
if ($dlQwen.Trim().ToLower() -eq "y") {
    & $VenvPython -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-3B-Instruct'); print('Qwen 完成')"
}

Section "9/9  建立啟動器"

$LaunchBat = @"
@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "HF_HOME=%~dp0models\huggingface"
set "HUGGINGFACE_HUB_CACHE=%~dp0models\huggingface\hub"
set "PATH=%~dp0runtime\ffmpeg\bin;%PATH%"
call "%~dp0.venv\Scripts\activate.bat"
cd /d "%~dp0F5-TTS-src"
f5-tts_infer-gradio --inbrowser
pause
"@
[System.IO.File]::WriteAllText((Join-Path $InstallDir "開啟F5-TTS.bat"), $LaunchBat, [System.Text.Encoding]::ASCII)

$TrainBat = @"
@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "HF_HOME=%~dp0models\huggingface"
set "HUGGINGFACE_HUB_CACHE=%~dp0models\huggingface\hub"
set "PATH=%~dp0runtime\ffmpeg\bin;%PATH%"
call "%~dp0.venv\Scripts\activate.bat"
cd /d "%~dp0F5-TTS-src"
f5-tts_finetune-gradio --inbrowser
pause
"@
[System.IO.File]::WriteAllText((Join-Path $InstallDir "開啟F5-TTS微調.bat"), $TrainBat, [System.Text.Encoding]::ASCII)

$CheckBat = @"
@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PATH=%~dp0runtime\ffmpeg\bin;%PATH%"
"%~dp0.venv\Scripts\python.exe" -c "import torch, torchcodec; print('Torch:',torch.__version__); print('TorchCodec:',torchcodec.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
ffmpeg -version
pause
"@
[System.IO.File]::WriteAllText((Join-Path $InstallDir "檢查F5-TTS環境.bat"), $CheckBat, [System.Text.Encoding]::ASCII)

# 建立桌面捷徑（失敗也不影響安裝）
try {
    $Desktop = [Environment]::GetFolderPath("Desktop")
    $ws = New-Object -ComObject WScript.Shell

    $shortcut = $ws.CreateShortcut((Join-Path $Desktop "F5-TTS 繁中版.lnk"))
    $shortcut.TargetPath = Join-Path $InstallDir "開啟F5-TTS.bat"
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.Save()

    $shortcut2 = $ws.CreateShortcut((Join-Path $Desktop "F5-TTS 微調.lnk"))
    $shortcut2.TargetPath = Join-Path $InstallDir "開啟F5-TTS微調.bat"
    $shortcut2.WorkingDirectory = $InstallDir
    $shortcut2.Save()

    Good "桌面捷徑已建立"
} catch {
    Warn "桌面捷徑建立失敗，但安裝目錄內的 BAT 仍可直接使用。"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "安裝完成！" -ForegroundColor Green
Write-Host "位置：$InstallDir" -ForegroundColor Green
Write-Host "啟動：$(Join-Path $InstallDir '開啟F5-TTS.bat')" -ForegroundColor Green
Write-Host "微調：$(Join-Path $InstallDir '開啟F5-TTS微調.bat')" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "這個安裝器沒有綁定 RTX 型號。換電腦時重新執行，它會重新偵測硬體。" -ForegroundColor White
