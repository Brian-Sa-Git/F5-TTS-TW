# F5-TTS-TW

F5-TTS 繁體中文通用一鍵安裝器，提供繁體中文介面、Windows 一鍵安裝／更新、完整音訊工作台、長音檔自動分段、Whisper 批次轉文字、段落素材庫與 F5-TTS 語音合成功能。

目前正式版本：**v1.0.2**

---

## 主要特色

### 一鍵安裝／更新

解壓縮後只需要執行：

```text
安裝或更新F5-TTS.bat
```

程式會自動判斷：

- 尚未安裝 `F5-TTS-Universal` → 執行完整安裝
- 已經安裝 `F5-TTS-Universal` → 直接更新客製功能

更新模式不會重新安裝 Python、Git、PyTorch、模型或 NVIDIA 驅動。

---

## 繁體中文介面

包含：

- 基本 TTS
- 多角色語音
- 語音聊天
- 完整音訊工作台
- 常用隨機種子收藏
- 英文年份／年代安全辨識
- BCE / BC / CE / AD 大小寫辨識
- 多種年份範圍符號辨識
- 句尾停頓設定
- 自訂模型選項

---

# 完整音訊工作台

v1.0.2 新增完整音訊工作台，可直接處理長音檔。

流程：

```text
① 上傳原始音訊
↓
② 整個音檔全部自動分段
↓
③ 全部已裁切段落轉文字
↓
④ 段落素材庫
↓
⑤ 套用到基本 TTS
```

---

## 長音檔全部自動分段

不論原始音訊是：

- 38 秒
- 5 分鐘
- 1 小時
- 或更長

都會從頭到尾全部處理，不只挑一個片段。

提供：

```text
📦 全部自動分段（只裁切）
📦 全部自動分段＋優化
```

分段原則：

- 優先依自然停頓切分
- 一般每段盡量控制在 5～12 秒
- 單一句連續超過 12 秒時，會保底切分
- 所有段落都會自動輸出

---

## 音訊優化

「全部自動分段＋優化」預設包含：

- 整理頭尾長靜音
- 音量正規化
- 過濾低頻隆隆聲
- 轉為 Mono
- 24 kHz
- 16-bit WAV
- 輕微淡入淡出，降低切點喀聲

採用保守型處理，避免重度降噪破壞原始聲線特徵。

---

## Whisper 批次轉文字

可將全部已裁切段落一次轉成文字。

目前介面保留三個語言選項：

```text
自動辨識
中文
英文
```

音訊與文字會使用相同段落編號。

例如：

```text
林薇英文聲音英文_001_優化.wav
林薇英文聲音英文_001.txt
```

---

# 段落素材庫

工作台會顯示所有已產生的音訊與文字素材。

可以：

- 預覽任一音訊
- 查看任一文字
- 自由選擇音訊
- 自由選擇文字
- 音訊與文字可交叉配對
- 一鍵套用到「基本 TTS」

例如：

```text
音訊：003
文字：003
```

也可以自行選擇：

```text
音訊：003
文字：007
```

---

# 輸出資料夾

所有工作台輸出都會自動儲存，不需要另外按下載才會保留。

預設位置：

```text
C:\AI\F5-TTS-Universal\輸出檔案
```

每次上傳新的原始音訊，都會建立新的日期時間資料夾。

例如：

```text
輸出檔案\
└─ 2026-09-02_00-22-09_林薇英文聲音英文\
   ├─ 音訊\
   │  ├─ 林薇英文聲音英文_001_優化.wav
   │  ├─ 林薇英文聲音英文_002_優化.wav
   │  └─ 林薇英文聲音英文_003_優化.wav
   │
   ├─ 文字\
   │  ├─ 林薇英文聲音英文_001.txt
   │  ├─ 林薇英文聲音英文_002.txt
   │  └─ 林薇英文聲音英文_003.txt
   │
   └─ 合成\
      ├─ 林薇英文聲音英文_001_合成.wav
      └─ 林薇英文聲音英文_002_合成.wav
```

音訊、文字、合成分開存放。

---

# Basic TTS

Basic TTS 的參考來源有兩種：

### 方式 1：從音訊工作台套用

在段落素材庫選擇：

- 參考音訊
- 參考文字

再一鍵套用到 Basic TTS。

### 方式 2：自行選擇

也可以完全不使用工作台，直接在 Basic TTS：

- 自行上傳參考音訊
- 自行輸入參考文字

自行選擇的內容會直接取代先前由工作台套用的內容。

---

# 合成音訊自動儲存

Basic TTS 合成完成後：

- 頁面可直接下載 WAV
- WAV 會自動儲存到對應時間資料夾的 `合成` 子資料夾
- 可直接按「開啟合成資料夾」

若參考音訊來自段落素材庫，合成結果會回到原本同一個時間資料夾。

只有自行上傳新的外部參考音訊時，才會建立新的時間資料夾。

---

# GPU / PyTorch 相容處理

安裝器會偵測 NVIDIA GPU 與目前驅動環境，再選擇相容的 PyTorch / TorchCodec 組合。

目前安裝器包含的版本判斷：

```text
CUDA >= 13.0 → cu130
CUDA >= 12.8 → cu128
CUDA >= 12.6 → cu126
其他情況 → CPU fallback
```

安裝器不綁定特定 RTX 型號。

---

# Portable FFmpeg

本專案使用獨立 Portable FFmpeg。

預設安裝位置：

```text
C:\AI\F5-TTS-Universal\runtime\ffmpeg
```

不會修改 Windows 全域 FFmpeg。

另外包含 Windows TorchCodec DLL 搜尋修正，以提高 TorchCodec / FFmpeg 載入相容性。

---

# 模型

安裝流程可下載：

- F5-TTS v1
- Vocos
- Whisper large-v3-turbo
- E2-TTS（依安裝選項）
- Voice Chat 相關模型（依安裝選項）

模型檔案會保存在 F5-TTS-Universal 安裝目錄中。

---

# 桌面捷徑

安裝完成後會建立：

```text
F5-TTS 繁中版
F5-TTS 微調
```

並使用專用桌面圖示。

---

# 解除安裝

安裝目錄內提供：

```text
解除安裝F5-TTS.bat
```

解除安裝器會移除：

```text
C:\AI\F5-TTS-Universal
```

以及指向該安裝的桌面捷徑。

不會移除：

```text
C:\AI\F5-TTS
Python
Git
NVIDIA Driver
```

使用解除安裝器前，請先完全關閉 F5-TTS 與相關命令視窗。

---

# 安裝位置

預設：

```text
C:\AI\F5-TTS-Universal
```

主要結構：

```text
F5-TTS-Universal\
├─ .venv\
├─ F5-TTS-src\
├─ icons\
├─ models\
├─ runtime\
├─ 輸出檔案\
├─ 開啟F5-TTS.bat
├─ 開啟F5-TTS微調.bat
├─ 檢查F5-TTS環境.bat
└─ 解除安裝F5-TTS.bat
```

---

# GitHub 原始檔結構

```text
F5-TTS-TW\
├─ icons\
│  ├─ F5-TTS.ico
│  └─ F5-TTS-Finetune.ico
│
├─ payload\
│  └─ infer_gradio.py
│
├─ portable_models\
├─ README.md
├─ config.json
├─ installer.ps1
├─ 使用說明.txt
├─ 安裝或更新F5-TTS.bat
├─ 第三方授權提醒.txt
└─ 解除安裝F5-TTS.bat
```

v1.0.2 已淘汰舊的：

```text
安裝F5-TTS.bat
更新完整音訊工作台.bat
```

統一改為：

```text
安裝或更新F5-TTS.bat
```

---

# 下載

請到 GitHub Releases 下載最新版：

https://github.com/Brian-Sa-Git/F5-TTS-TW/releases/latest

Release 中請下載自行提供的 F5-TTS 安裝 ZIP。

GitHub 自動產生的：

```text
Source code (zip)
Source code (tar.gz)
```

是原始碼封裝，不是 Windows 一鍵安裝包。

---

# v1.0.2 更新內容

- 新增完整音訊工作台
- 新增長音檔全部自動分段
- 新增全部自動分段＋音訊優化
- 新增 Whisper 全部分段批次轉文字
- 新增段落素材庫
- 新增音訊／文字自由配對
- 新增一鍵套用到 Basic TTS
- 新增音訊／文字／合成分類資料夾
- 新增依日期時間建立每次處理工作資料夾
- 新增簡潔原始錄音名稱與段落編號
- 新增合成完成自動儲存 WAV
- 安裝與更新合併為單一入口
- 修正解除安裝功能
- 修正 Gradio 相容性
- 修正 TorchCodec / Portable FFmpeg Windows DLL 載入
- 修正合成檔案回存錯誤資料夾
- 修正輸出根目錄產生空白音訊／文字資料夾
- 修正完整音訊工作台啟動相容問題

---

# 第三方專案與授權

本專案基於 F5-TTS 及其相關開源套件進行 Windows 安裝流程、繁體中文介面與工作流程整合。

請同時遵守 F5-TTS、PyTorch、FFmpeg、Whisper、Transformers、Gradio、TorchCodec 及其他第三方套件各自的授權條款。

詳細提醒請參考：

```text
第三方授權提醒.txt
```
