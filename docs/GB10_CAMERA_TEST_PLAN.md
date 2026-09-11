# GB10（MSI EdgeXpert / Grace Blackwell）攝影機連接測試計畫

## 目的

回答老師交辦的問題：**FLIR 相機能否在 GB10 上被系統偵測到、並正常擷取畫面，且能否同時接上 3～5 台。**

範圍只到「用廠商的 SpinView 軟體看到每台相機的畫面」為止，不使用 GPIO 同步觸發、不寫任何程式、不包含把整套骨架分析 GUI 移植過去（那些都是後續、規模大得多的任務，等這一步確認可行再說）。GB10 是借測用機器，測試完成後需將 Spinnaker SDK 完整卸載、恢復原狀（見文末「測試後卸載」）。

## 目前已知的資訊（測試前先讀）

- GB10 的 CPU 是 ARM 架構（20 核 Arm，Cortex-X925 + Cortex-A725），作業系統是 **NVIDIA DGX OS**
- 廠商已提供 `spinnaker-4.4.0.246-noble-arm64-pkg.tar.gz`——確認支援 **Ubuntu 24.04/22.04 LTS + ARM64**，安裝方式是標準 `.deb` 套件
- **這份壓縮檔裡沒有 Python 綁定（PySpin）**，只有 C/C++ 函式庫與 GUI 工具（`SpinView_QT`）。已詢問廠商，等待回覆
- 我們的 Python 程式碼（`camera_objects/single_camera/flir_camera_system.py`）是透過 `import PySpin` 呼叫相機的，光有 C++ SDK 無法讓現有程式碼運作——但**可以**用來驗證「相機硬體本身能不能被系統看到、拿到畫面」，不需要等 Python 綁定到位

## 階段 0：確認 GB10 的作業系統版本

這一步決定廠商提供的 SDK 能不能直接用，是最優先要做的事。廠商給的 SDK 檔名是
`spinnaker-4.4.0.246-noble-arm64-pkg.tar.gz`——`noble` 是 Ubuntu 24.04 的代號，
`arm64` 是 CPU 架構，這份 SDK 就是專門為「Ubuntu 24.04 + ARM64」打造的。GB10 實際
是不是這個組合，必須先核對，避免裝到一半才發現版本不合、白忙一場。

```bash
cat /etc/os-release
uname -m
```

**這兩行指令在做什麼：**
- `cat /etc/os-release`：`cat` 是顯示檔案內容的指令，`/etc/os-release` 是 Linux 系統
  裡固定會有的檔案，記錄目前裝的是哪個發行版、哪個版本。輸出會類似：
  ```
  NAME="Ubuntu"
  VERSION="24.04 LTS (Noble Numbat)"
  VERSION_ID="24.04"
  ```
  要看的是 `VERSION_ID` 這一行。
- `uname -m`：顯示 CPU 架構的指令，會印出一個字，例如 `aarch64`（ARM 64 位元，
  GB10 預期會顯示這個）或 `x86_64`（一般 Intel/AMD 架構）。

**判斷標準：**
- `VERSION_ID` 顯示 `24.04` 或 `22.04`，且 `uname -m` 顯示 `aarch64` → 跟廠商 SDK 完全對應，可以直接進行階段 1
- 版本不同（例如 DGX OS 是基於其他版本、或是完全客製化的發行版）→ 先截圖版本資訊，這是需要回報給廠商確認的關鍵資訊，不要貿然安裝

## 階段 1：安裝 Spinnaker SDK（不含 Python 綁定）

```bash
# 把 spinnaker-4.4.0.246-noble-arm64-pkg.tar.gz 傳到 GB10 上後
tar -xzf spinnaker-4.4.0.246-noble-arm64-pkg.tar.gz
cd spinnaker-4.4.0.246-noble-arm64

# 先裝相依套件（README_ARM.md 第 1 節）
sudo apt-get install libusb-1.0-0 qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools

# 執行官方安裝腳本
sudo sh install_spinnaker_arm.sh
```

安裝過程會詢問是否要設定 udev 權限（讓一般使用者能存取 USB 裝置），**選是**。

**判斷標準：**
- 腳本跑完沒有紅字錯誤 → 進入階段 2
- 有錯誤 → 把完整錯誤訊息記錄下來（螢幕截圖或複製文字），這類套件相依性錯誤通常訊息本身就會指出缺少什麼

> 安裝完成後**必須重開機一次**——README 原文提到「udev 權限不會立即生效，可能需要重開機」，這是容易漏掉的一步。

## 階段 2：接上一台相機，用 SpinView 驗證

這一步對應我們在 Windows 上「先用 SpinView 確認硬體層級沒問題」的做法，原理相通。

```bash
spinview
```

（或完整路徑 `/opt/spinnaker/bin/SpinView_QT`）

**只接一台相機**（不用一開始就接兩台），確認：
1. 左側 Devices 清單裡有沒有出現這台相機、序號是否正確
2. 點開它，能不能看到即時畫面（就算是全黑或色偏都算「有畫面」，代表資料流通了）

**判斷標準：**
- 有偵測到、有畫面 → **這一步就已經回答了老師的核心問題：GB10 可以連上 FLIR 相機**，可以進入階段 3 做更完整的確認
- 偵測不到相機 → 先檢查：
  - USB 線是否確實插好（USB3 相機建議直接接主機的 USB3 埠，不要透過集線器）
  - 執行 `lsusb`，確認系統底層有沒有看到這個 USB 裝置（如果 `lsusb` 都看不到，是作業系統/驅動層問題，不是 Spinnaker 的問題）
  - 參考 README 第 3 節「USB CAMERA SETUP」，特別是 USB-FS buffer 限制（ARM 平台常見卡點）
- 有偵測到、但抓不到穩定畫面（類似我們在 Windows 上遇過的逾時錯誤）→ 檢查 README 第 3 節提到的 `coherent_pool`、`swiotlb` 這兩個 ARM 平台特有的核心參數調整

## 階段 3：接上 3～5 台相機，逐台確認畫面

**不接 GPIO 同步線，不需要觸發同步**，單純測試「一次接多台相機時，GB10 是否都能個別認出來、個別拿到畫面」。

1. 依序把相機一台一台接上（建議一台一台加，方便定位是哪一台或第幾台開始出問題，而不是一次全接再看）
2. 每接上一台，回到 SpinView 檢查 Devices 清單是否新增一筆、序號正確
3. 全部接上後，在清單裡逐一點開每一台，確認每台都能各自顯示畫面（不需要同時開著看，一台一台點開確認即可）

**判斷標準：**
- 全部台數都出現在清單、都能個別拿到畫面 → 核心問題確認可行，測試目的達成
- 接到第 N 台之後，後面的相機不再被偵測到，或畫面變得不穩定/卡頓 → 很可能是 USB 頻寬或 USB-FS buffer 不足（見 README 第 3 節），記錄「接到第幾台開始出問題」這個資訊，這對後續判斷（要不要分接到不同 USB 控制器、要不要用 USB Hub）很關鍵
- 部分相機完全沒反應 → 檢查是否接在同一個 USB 集線器/擴充卡上導致頻寬不足，換一個 USB 埠試試

## 待廠商回覆後才能進行的部分

- **Python 綁定（PySpin）的 ARM64 版本**：拿到後才能驗證「我們現有的 Python 程式碼」能不能在 GB10 上實際呼叫相機。在此之前，這份計畫只能證明「硬體相容、C++ 層可行」，還不能證明「我們的 GUI 系統能跑」

## 測試後卸載

GB10 是借測用機器，測試完成後需要把 Spinnaker SDK 完整移除，恢復到測試前的狀態。

```bash
cd spinnaker-4.4.0.246-noble-arm64
sudo sh remove_spinnaker_arm.sh
```

官方 README 說明：這個腳本會移除所有 Spinnaker 函式庫，並還原被安裝程序修改過的 udev 權限規則（也就是恢復 Ubuntu 預設的 USB 裝置權限設定）。

**卸載後建議額外確認：**
- 階段 1 額外裝的相依套件（`libusb-1.0-0 qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools`）不會被 `remove_spinnaker_arm.sh` 移除，這些是 Ubuntu 官方套件、非常通用（Qt 相關工具很多軟體都會用到），一般不需要特地移除；如果對方要求恢復到「完全乾淨、不多裝任何東西」的狀態，可另外執行 `sudo apt-get remove libusb-1.0-0 qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools`（但先確認這些套件沒有被系統其他既有程式依賴，避免移除後影響到原本就在跑的東西）
- 確認解壓縮出來的 `spinnaker-4.4.0.246-noble-arm64` 資料夾與原始的 `.tar.gz` 壓縮檔本身，測試完後一併從機器上刪除

## 記錄建議

測試時建議把每個階段的終端機輸出存下來（複製文字或截圖皆可），特別是：
- `cat /etc/os-release` 的完整輸出
- 每接上一台相機後，SpinView Devices 清單的畫面（截圖即可）
- 任何錯誤訊息的完整文字（不要只截圖看得到的部分，錯誤訊息常常很長，重要線索可能在後面）

測試完後把記錄丟回來，我可以幫忙判讀錯誤訊息、決定下一步。
