# Spinnaker SDK ARM 版 README（中文翻譯）

> 翻譯自廠商提供的 `spinnaker-4.4.0.246-noble-arm64-pkg.tar.gz` 內附 `README_ARM.md`（版權：FLIR Integrated Imaging Solutions, Inc.）。翻譯僅供內部參考，安裝與操作以原文件為準；原文件位於 `C:\Users\user\Downloads\spinnaker_arm_extract\spinnaker-4.4.0.246-noble-arm64\README_ARM.md`。

## 目錄

- [1. 相依套件](#1-相依套件)
  - [1.1 Ubuntu 24.04 與 22.04 相依套件](#11-ubuntu-2404-與-2204-相依套件)
  - [1.2 Ubuntu 20.04 相依套件](#12-ubuntu-2004-相依套件)
- [2. Spinnaker 安裝](#2-spinnaker-安裝)
- [3. USB 相機設定](#3-usb-相機設定)
- [4. GigE 相機設定](#4-gige-相機設定)
- [5. GenICam GenTL Producer 設定](#5-genicam-gentl-producer-設定)
- [6. Spinnaker 移除](#6-spinnaker-移除)
- [7. 執行內建工具](#7-執行內建工具)
- [8. Teledyne-MV Spinnaker 範例程式](#8-teledyne-mv-spinnaker-範例程式)
- [9. 常見問題](#9-常見問題)

---

## 1. 相依套件

在 Linux 上安裝 Spinnaker 前，需要先安裝幾個前置函式庫。

### 1.1 Ubuntu 24.04 與 22.04 相依套件

適用於 Ubuntu 24.04 與 22.04 LTS，Spinnaker 及其他元件相依的函式庫如下：

**1) 核心影像擷取函式庫**（Spinnaker 與 Spinnaker-C 核心）
- `libusb-1.0-0`（建議 1.0.17 以上版本）

**2) SpinView_QT**
- `qtbase5-dev`
- `qtchooser`
- `qt5-qmake`
- `qtbase5-dev-tools`

**強烈建議：**
- Ubuntu 24.04 LTS：Linux 核心版本 6.8 或以上
- Ubuntu 22.04 LTS：Linux 核心版本 5.15 或以上

#### 1.1.1 安裝方式

最簡單的方式是用系統內建的套件管理工具（Ubuntu 為 `apt`）：

```bash
sudo apt-get install libusb-1.0-0 qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
```

### 1.2 Ubuntu 20.04 相依套件

**1) 核心影像擷取函式庫**
- `libusb-1.0-0`（建議 1.0.17 以上版本）

**2) SpinView_QT**
- `qt5-default`

**強烈建議：**
- Ubuntu 20.04 LTS
- Linux 核心版本 5.4 以上

#### 1.2.1 安裝方式

```bash
sudo apt-get install libusb-1.0-0
```

---

## 2. Spinnaker 安裝

相依套件裝好後，用解壓縮出來的目錄裡附的 `install_spinnaker_arm.sh` 腳本安裝 Spinnaker 的 deb 檔：

```bash
sudo sh install_spinnaker_arm.sh
```

這個腳本會安裝所有 Spinnaker 函式庫、範例程式碼、範例應用程式與文件。

安裝腳本會詢問是否要設定 udev（讓一般使用者能使用 USB 裝置）。如果選擇設定，腳本會修改裝置節點的權限，讓該使用者有完整讀寫權限，**這會覆蓋 Ubuntu 預設的權限設定**。

> ⚠️ 重新啟動 udev 服務後，權限似乎不會立即套用到裝置節點上，**可能需要重新開機**，使用者才能正常存取裝置。

**Spinnaker 套件清單：**
- `libgentl_<版本>_<平台>.deb`
- `libspinnaker_<版本>_<平台>.deb`
- `libspinnaker-dev_<版本>_<平台>.deb`
- `libspinnaker-c_<版本>_<平台>.deb`
- `libspinnaker-c-dev_<版本>_<平台>.deb`
- `libspinvideo_<版本>_<平台>.deb`
- `libspinvideo-dev_<版本>_<平台>.deb`
- `libspinvideo-c_<版本>_<平台>.deb`
- `libspinvideo-c-dev_<版本>_<平台>.deb`
- `spinnaker_<版本>_<平台>.deb`
- `spinnaker-doc_<版本>_<平台>.deb`
- `spinupdate_<版本>_<平台>.deb`
- `spinupdate-dev_<版本>_<平台>.deb`
- `spinview-qt_<版本>_<平台>.deb`
- `spinview-qt-dev_<版本>_<平台>.deb`

開頭是 `lib` 的套件都是共用函式庫及其對應的開發套件。`spinnaker` 套件會安裝擷取應用程式，在終端機輸入 `spinview` 或從應用程式選單即可啟動。`spinnaker-doc` 套件內含 PDF 格式的官方文件（位於 `/opt/spinnaker/doc`）。

---

## 3. USB 相機設定

Linux 系統預設的 USB-FS 只允許所有 USB 裝置共用 16 MB 的緩衝記憶體，這在高解析度影像擷取或多相機架設時可能導致擷取異常，**這個上限必須調高才能發揮硬體的完整效能**。

> 本方法適用於 Ubuntu 標準安裝。若是執行其他發行版的 ARM 開發板，請參考官方文章：
> [Using Spinnaker on ARM and Embedded Systems](https://www.teledynevisionsolutions.com/support/support-center/application-note/iis/using-spinnaker-on-arm-and-embedded-systems/)

安裝程式會詢問是否自動設定適當的 USB-FS 記憶體限制，也可以隨時手動執行設定腳本：

```bash
sudo sh configure_usbfs.sh
```

**手動設定 USB-FS 記憶體限制：**

1. 若系統上不存在 `/etc/rc.local`，先建立並設定執行權限：
   ```bash
   sudo touch /etc/rc.local
   sudo chmod 744 /etc/rc.local
   ```

2. 開啟 `/etc/rc.local`：
   ```bash
   sudo nano /etc/rc.local
   ```
   在檔案末端加入：
   ```bash
   sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'
   ```

3. 存檔關閉。

4. 重新開機。

確認記憶體上限是否已更新：
```bash
cat /sys/module/usbcore/parameters/usbfs_memory_mb
```

若設定未生效，可以**暫時性**修改（下次重開機會失效）：
```bash
sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'
```

> 若使用多台 USB3 相機，USB-FS 記憶體上限可能需要超過 1000。

**部分 ARM 平台上，libusb 能提交的緩衝區數量還受以下兩個核心參數限制：**

**1) `coherent_pool`**

若 Spinnaker 無法開始擷取、且在 `dmesg` 看到以下類似訊息：
- `xhci_hcd 0000:01:00.0: Ring expansion failed`
- `usb 2-1: usbfs: usb_submit_urb returned -12`

可嘗試增加 coherent atomic allocation pool：
1. 用有權限的方式開啟 `/boot/firmware/cmdline.txt`
2. 在檔案末端加入（或修改既有項目）：
   ```
   coherent_pool=5M
   ```
3. 重新開機

**2) `swiotlb`**

若 Spinnaker 無法開始擷取、且在 `dmesg` 看到以下類似訊息：
- `xhci_hcd 0000:01:00.0: swiotlb buffer is full (sz: 16384 bytes), total 32768 (slots), used 32414 (slots)`
- `usb 2-1: usbfs: usb_submit_urb returned -11`

可嘗試增加 swiotlb 緩衝區：
1. 開啟 `/boot/firmware/cmdline.txt`
2. 加入（或修改既有項目）：
   ```
   swiotlb=32768
   ```
3. 重新開機

---

## 4. GigE 相機設定

> 本專案使用的是 USB3 相機（Grasshopper3），本節為 GigE（網路型）相機適用，僅供參考。

為避免網路介面封包遺失，可調整幾個參數。最重要的是 MTU（最大傳輸單元）大小與網卡驅動可用的接收緩衝區數量，這能降低需處理的封包數、減少 CPU 負擔與中斷次數。

SDK 提供的網路調校腳本可最大化 MTU（啟用 Jumbo frames）並最佳化部分網路設定，位於：
```
/opt/spinnaker/bin/
```

範例（調整 `eth0` 介面，需要管理員權限）：
```bash
sudo ./gev_nettweak eth0
```

（後續 4.1～4.6 小節涵蓋 RPF 停用、接收緩衝區調整、Jumbo Packet、5G 連線速度、Jetson 效能調校、網路最佳實務等細節，因本專案不使用 GigE 相機，此處從略；如未來改用 GigE 相機，請回頭參考原文完整內容。）

---

## 5. GenICam GenTL Producer 設定

若要讓支援 GenTL Producer 的應用程式使用 Spinnaker GenTL Producer，需要把 `Spinnaker_GenTL.cti` 的位置加入 `GENICAM_GENTL32_PATH`（32 位元）或 `GENICAM_GENTL64_PATH`（64 位元）環境變數。

安裝程式會詢問是否自動設定這些環境變數，也可以隨時手動執行：

```bash
# 32 位元函式庫版本
sudo sh configure_gentl_paths.sh 32

# 64 位元函式庫版本
sudo sh configure_gentl_paths.sh 64
```

### 5.1 GenTL 記錄檔

要啟用 Spinnaker GenTL 的記錄功能，把 `log4cpp.gentl.property`（位於 `/opt/spinnaker/lib/spinnaker-gentl`）複製到使用端應用程式的執行目錄下，可修改此檔調整記錄等級。

---

## 6. Spinnaker 移除

用附帶的解除安裝腳本移除 Spinnaker，此腳本也會移除 udev 規則，恢復 Ubuntu 原本的裝置節點權限：

```bash
sudo sh remove_spinnaker_arm.sh
```

---

## 7. 執行內建工具

除了預先編譯好的範例程式（如 Acquisition、ChunkData 等）與其原始碼外，Spinnaker SDK 也附帶幾個評估相機用的工具（如 SpinView）、更新韌體用的工具（如 SpinUpdateConsole）等。

SDK 附帶幾支命令列腳本，用來自動設定環境變數，讓這些工具可以在任何工作目錄下透過命令列啟動，或透過自訂捷徑啟動。

安裝程式會詢問是否自動設定這些環境變數，也可以隨時手動執行：

```bash
sudo sh configure_spinnaker_paths.sh
```

環境變數設定完成後，直接用對應的啟動指令即可執行工具，不需要切換到 `/opt/spinnaker/bin` 目錄或手動修改環境變數。

> 若要透過非登入 shell（例如用 `sudo`）執行範例程式，需要用 `-E` 參數傳遞使用者環境變數，例如：
> ```bash
> sudo -E /opt/spinnaker/bin/Enumeration
> ```

### 7.1 SpinView QT

圖形化應用程式，用於測試相機並檢視所有支援的 Teledyne 相機的即時影像串流。

- 位置：`/opt/spinnaker/bin/SpinView_QT`
- 簡稱：SpinView
- 啟動指令：`spinview`

### 7.2 SpinUpdateConsole

用於更新 Teledyne 相機韌體的主控台應用程式。

- 位置：`/opt/spinnaker/bin/SpinUpdateConsole`
- 簡稱：SpinUpdateConsole
- 啟動指令：`SpinUpdateConsole`

---

## 8. Teledyne-MV Spinnaker 範例程式

請至官方 GitHub 儲存庫查看可參考的範例程式：
<https://github.com/Teledyne-MV/Spinnaker-Examples>

---

## 9. 常見問題

### 9.1 顯示畫面閃爍

部分 ARM 開發板上，圖形檢視工具 SpinView_QT 會出現畫面閃爍的情況，這是圖形驅動程式造成的，可透過更換圖形驅動程式解決。例如在 Jetson Orin / XAVIER 開發板上，安裝 `xserver-xephyr` 套件並透過 Xephyr X server 執行 SpinView_QT 可解決此問題：

```bash
Xep7hyr -br -ac -noreset -screen 1920x1080 :1& DISPLAY=:1 SpinView_QT
```
