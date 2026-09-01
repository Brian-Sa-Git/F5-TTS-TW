# F5-TTS-TW

**F5-TTS 繁體中文客製化介面與一鍵安裝工具**

本專案以 F5-TTS 為基礎，提供繁體中文介面、Windows 一鍵安裝器，以及針對英文年份、停頓與常用隨機種子等功能的客製化改善。

> 本專案為第三方繁體中文客製版本，並非 F5-TTS 官方專案。

---

## 📥 下載最新版

👉 **[前往 Releases 下載最新版](https://github.com/Brian-Sa-Git/F5-TTS-TW/releases/latest)**

下載 Release 頁面中提供的：

```text
F5-TTS_繁中通用一鍵安裝器_vX.X.X.zip
```

解壓縮後，雙擊：

```text
安裝F5-TTS.bat
```

即可開始安裝。

---

## ✨ 主要功能

### 繁體中文介面

將 F5-TTS WebUI 主要操作介面改為繁體中文，降低使用門檻。

包含：

- 基本 TTS
- 多角色語音
- Voice Chat
- 進階設定
- 模型選擇
- 微調介面
- 常用操作提示

---

### 句尾停頓設定

新增可調整的句尾停頓功能。

適合長篇旁白、教材、故事、說書等需要自然斷句的用途。

例如：

```text
第一句。
第二句。
第三句。
```

可以在句子之間加入固定停頓，避免每一句黏在一起。

---

### 英文年份安全辨識

針對英文年份與歷史年代進行文字前處理，例如：

```text
441 BCE
496–406 BCE
1945 CE
2024 AD
```

支援：

```text
BCE / bce / Bce
BC / bc
CE / ce
AD / ad
```

大小寫皆可自動辨識。

年份範圍也支援多種橫線：

```text
496-406 BCE
496–406 BCE
496—406 BCE
496−406 BCE
```

並避免誤處理一般英文複合字，例如：

```text
self-destruction
well-known
long-term
```

---

### 常用隨機種子收藏

如果某次生成的聲音特別自然，可以把 Seed 收藏起來。

支援：

- 新增常用 Seed
- 自訂備註名稱
- 快速套用
- 刪除收藏
- 關閉程式後仍保留

例如：

```text
12345 ｜ 自然美式
54321 ｜ 旁白穩定
777   ｜ 情緒自然
```

---

## 🤖 支援功能

目前安裝器可搭配：

- F5-TTS v1
- E2-TTS
- Whisper
- Multi-Speech
- Voice Chat
- F5-TTS Finetune / 微調

部分模型可於安裝時選擇是否預先下載。

---

## 🖥️ GPU 自動偵測

安裝器**不綁定特定 RTX 型號**。

不論目前使用：

```text
RTX 3050
RTX 4060
RTX 5060
RTX 5070
或其他 NVIDIA GPU
```

安裝時都會重新偵測電腦環境。

安裝器會檢查：

- NVIDIA GPU
- NVIDIA Driver
- CUDA 相容能力
- PyTorch 安裝模式

如果沒有可用的 NVIDIA GPU，則可退回 CPU 模式。

> GPU 加速仍需要電腦本身安裝相容的 NVIDIA 顯示卡驅動程式。

---

## 🌐 GitHub 自動取得最新設定

安裝器啟動時會優先取得本專案最新：

```text
config.json
```

並依照設定下載最新繁中客製介面：

```text
payload/infer_gradio.py
```

目前客製介面來源：

```text
https://raw.githubusercontent.com/Brian-Sa-Git/F5-TTS-TW/main/payload/infer_gradio.py
```

如果暫時無法連線 GitHub，安裝器會改用安裝包內建設定。

---

## 📦 模型可選擇一起攜帶

如果已經下載模型，也可以放入：

```text
portable_models/
```

安裝器會在安裝時嘗試合併已攜帶的 Hugging Face 模型快取。

這樣換電腦時，可以減少重新下載大型模型。

---

## 🚀 安裝方式

### 1. 下載安裝包

前往：

👉 **[Releases](https://github.com/Brian-Sa-Git/F5-TTS-TW/releases/latest)**

下載最新：

```text
F5-TTS_繁中通用一鍵安裝器_vX.X.X.zip
```

### 2. 解壓縮

請先完整解壓縮 ZIP。

不要直接在壓縮檔內執行 BAT。

### 3. 執行安裝

雙擊：

```text
安裝F5-TTS.bat
```

### 4. 選擇安裝位置

預設安裝位置：

```text
C:\AI\F5-TTS-Universal
```

也可以自行指定其他位置。

### 5. 選擇模型

安裝過程中可依需要選擇是否下載：

```text
F5-TTS v1
E2-TTS
Whisper
Qwen Voice Chat
```

### 6. 安裝完成

完成後會建立：

```text
開啟F5-TTS.bat
開啟F5-TTS微調.bat
檢查F5-TTS環境.bat
```

並嘗試建立桌面捷徑：

```text
F5-TTS 繁中版
F5-TTS 微調
```

---

## 🔧 安裝器會自動處理

安裝器會依環境處理：

- Python 3.11
- Git
- F5-TTS
- PyTorch
- TorchCodec
- FFmpeg Shared
- Microsoft VC++ Runtime（必要時）
- Hugging Face 模型
- F5-TTS 繁中客製介面
- Finetune Gradio 相容修正
- TorchCodec / FFmpeg DLL 相容處理

---

## 📁 Repository 結構

```text
F5-TTS-TW/
├─ README.md
├─ config.json
├─ installer.ps1
├─ 安裝F5-TTS.bat
├─ 使用說明.txt
├─ 第三方授權提醒.txt
│
├─ payload/
│  └─ infer_gradio.py
│
└─ portable_models/
   └─ README_模型攜帶方式.txt
```

---

## 🔄 更新方式

客製介面更新時，主要更新：

```text
payload/infer_gradio.py
```

安裝器設定則位於：

```text
config.json
```

新版正式發佈時，請建立新的 GitHub Release，例如：

```text
v1.0.2
v1.1.0
v2.0.0
```

---

## ⚠️ 注意事項

- 建議使用 Windows 10 / Windows 11 64-bit。
- 線上安裝模式需要網路連線。
- NVIDIA GPU 加速需要正常的 NVIDIA Driver。
- 不同 GPU、Driver、PyTorch 與 TorchCodec 版本可能存在相容性差異。
- 第一次執行模型時，部分模型可能需要額外下載。
- 大型模型不建議直接放入 GitHub Repository。
- 若要公開散布或商業使用，請確認 F5-TTS、模型及其他第三方套件各自的授權條款。

---

## 🔗 上游專案

F5-TTS 官方專案：

https://github.com/SWivid/F5-TTS

本專案主要提供繁體中文客製介面與 Windows 安裝流程。

---

## 📄 授權與第三方套件

本 Repository 中的客製程式、安裝腳本，以及第三方專案與模型可能具有不同授權條款。

使用前請分別確認：

- F5-TTS
- PyTorch
- TorchCodec
- FFmpeg
- Hugging Face 模型
- Whisper
- Qwen
- 其他相關套件

詳細提醒可參考：

```text
第三方授權提醒.txt
```
