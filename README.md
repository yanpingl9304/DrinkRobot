# OpenCV & Qt (XCB) 衝突修復指南

當你在 Conda 環境中執行 OpenCV 或 PyQt5，遇到 Qt 插件初始化失敗或顯示器連接問題時，請參考此完整流程。

---

## 🛑 1. 錯誤現象 (Error)
如果程式噴出以下錯誤：
- `QObject::moveToThread: Current thread is not the object's thread...`
- `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"...`
- `This application failed to start because no Qt platform plugin could be initialized.`

---

## 🛠 2. 解決步驟 (Fix)

### 徹底移除 OpenCV 的 Qt 插件
這是為了解決 OpenCV 內建 Qt 與環境中 PyQt5 衝突的最直接方法。

> **⚠️ 注意：** 請將路徑中的 `/home/ncku_csie_rl/miniconda3/envs/Test_Env/` 替換為你電腦上的實際環境路徑。

```bash
# 刪除 OpenCV 自帶的插件資料夾
rm -rf /home/ncku_csie_rl/miniconda3/envs/Test_Env/lib/python3.10/site-packages/cv2/qt/plugins

# 1. 指向正確的 PyQt5 插件路徑
export QT_QPA_PLATFORM_PLUGIN_PATH=/home/ncku_csie_rl/miniconda3/envs/Test_Env/lib/python3.10/site-packages/PyQt5/Qt5/plugins

# 2. 設定顯示器編號 (預設為 :0)
export DISPLAY=:0

# 3. 授權 X11 服務存取
xhost +local:root > /dev/null 2>&1
```

### 如果報錯：could not connect to display :0

```bash
export DISPLAY=:1
```
---

## Vtube Studio教學

https://github.com/organics2016/pymouth

## 若裝完 pymouth 後 numpy報錯
```bash
pip install "numpy<2"
```
