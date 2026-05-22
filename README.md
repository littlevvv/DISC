# 椎間盤辨識系統 (Disc Recognition System)

![ITRI Logo](ico/ITRI128.ico)

一個基於深度學習的脊椎MRI影像椎間盤自動分割與等級分類系統，使用 Flask 提供網頁界面。

## 📋 目錄

- [功能特色](#功能特色)
- [技術架構](#技術架構)
- [系統需求](#系統需求)
- [安裝指南](#安裝指南)
- [使用方法](#使用方法)
- [API 端點](#api-端點)
- [模型說明](#模型說明)
- [文件結構](#文件結構)
- [相關論文](#相關論文)
- [授權協議](#授權協議)

## 🚀 功能特色

### 核心功能
- **語意分割**: 使用 UNET 模型進行椎間盤組織自動分割
- **等級分類**: 使用 DenseNet121 模型進行椎間盤退化等級自動分類  
- **網頁界面**: 直觀的 Flask 網頁界面，支援拖拽上傳
- **即時處理**: 上傳圖像後即時返回分析結果
- **結果可視化**: 提供原圖、分割結果、分類結果的對比顯示

### 輔助功能
- **論文下載**: 提供相關研究論文和專利文獻下載
- **結果保存**: 自動保存分析結果為圖像文件
- **錯誤處理**: 完善的錯誤處理和用戶反饋機制

## 🏗️ 技術架構

```
前端界面 (HTML/CSS/JavaScript)
        ↓
Flask Web 框架 (Python)
        ↓
圖像處理 (OpenCV + PIL)
        ↓
深度學習模型
├── UNET (語意分割)
└── DenseNet121 (等級分類)
        ↓
結果輸出 (Base64 編碼圖像)
```

### 技術棧
- **後端**: Flask, Python 3.6+
- **深度學習**: TensorFlow/Keras, OpenCV
- **前端**: HTML5, CSS3, JavaScript
- **圖像處理**: OpenCV, PIL, NumPy

## 💻 系統需求

### 硬體需求
- **CPU**: 建議 4 核心以上
- **記憶體**: 最少 8GB RAM (建議 16GB+)
- **儲存空間**: 至少 5GB 可用空間
- **GPU**: 可選，支援 CUDA 加速

### 軟體需求
- **作業系統**: Linux (Ubuntu 18.04+) / Windows 10+ / macOS
- **Python**: 3.6 - 3.8
- **Conda**: Miniconda 或 Anaconda

## 📦 安裝指南

### 1. 環境準備

```bash
# 創建 conda 環境
conda create -n disc python=3.6
conda activate disc

# 安裝必要套件
conda install tensorflow=2.4.3 keras=2.4.3 opencv=4.5.2
conda install flask scikit-learn matplotlib numpy pandas pillow
```

### 2. 下載模型文件

確保以下模型文件存在於正確位置：

```
/media/r300/1T/A30335/Disc/
├── ITRI_CMS_R100_unet_residual_vgg16.hdf5      # UNET 分割模型
└── past/grademodel/
    └── ITRI_CMSR100_DenseNet121_50_128_loss.hdf5  # DenseNet121 分類模型
```

### 3. 準備資源文件

```
/media/r300/1T/A30335/Disc/
├── ico/
│   └── ITRI128.ico                    # ITRI 圖標
└── paperReference/
    ├── 計量文章_脊椎MRI影像之椎間盤切割與分類技術_劉曉薇 賴程威_0610 1.pdf
    ├── TWI722264B.pdf                 # 台灣專利
    └── US12136485.pdf                 # 美國專利
```

### 4. 啟動應用

```bash
cd /media/r300/1T/A30335/Disc/
conda activate disc
python flask_disc_ui.py
```

系統將在 `http://localhost:6060` 啟動。

## 🔧 使用方法

### 基本使用流程

1. **啟動系統**: 執行 `python flask_disc_ui.py`
2. **開啟瀏覽器**: 訪問 `http://localhost:6060`
3. **上傳圖像**: 點擊上傳區域或拖拽圖像文件
4. **開始分析**: 點擊「開始分析」按鈕
5. **查看結果**: 系統將顯示分割和分類結果

### 支援的圖像格式
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

### 輸出結果
- **原始圖像**: 調整大小後的輸入圖像 (512×512)
- **UNET 語意切割**: 白色區域標示椎間盤組織位置
- **椎間盤等級分類**: 紅色邊界框標示位置並顯示等級 (Grade 1-5)

## 🔗 API 端點

### 主要端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/` | GET | 主頁面 |
| `/process` | POST | 圖像處理 API |
| `/favicon.ico` | GET | 網站圖標 |

### 論文下載端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/paper` | GET | 下載主要研究論文 |
| `/view_paper` | GET | 線上查看研究論文 |
| `/paper_twi` | GET | 下載台灣專利 TWI722264B |
| `/view_paper_twi` | GET | 線上查看台灣專利 |
| `/paper_us` | GET | 下載美國專利 US12136485 |
| `/view_paper_us` | GET | 線上查看美國專利 |

### API 使用範例

```bash
# 使用 curl 上傳圖像進行分析
curl -X POST -F "image=@your_image.jpg" http://localhost:6060/process
```

## 🧠 模型說明

### UNET 分割模型
- **架構**: UNET + ResNet + VGG16
- **輸入尺寸**: 512×512×3
- **輸出**: 3 類別分割結果
- **檔案**: `ITRI_CMS_R100_unet_residual_vgg16.hdf5`

### DenseNet121 分類模型
- **架構**: DenseNet121
- **輸入尺寸**: 128×128×3 (可自動解析)
- **輸出**: 椎間盤退化等級 (Grade 1-5)
- **檔案**: `ITRI_CMSR100_DenseNet121_50_128_loss.hdf5`

### 處理流程
1. 圖像預處理 (調整大小、正規化)
2. UNET 語意分割
3. 輪廓檢測與過濾
4. ROI 提取
5. DenseNet121 等級分類
6. 結果後處理與可視化

## 📁 文件結構

```
/media/r300/1T/A30335/Disc/
├── flask_disc_ui.py                   # 主程式
├── README.md                          # 說明文件
├── ITRI_CMS_R100_unet_residual_vgg16.hdf5  # UNET 模型
├── ico/
│   └── ITRI128.ico                    # 圖標文件
├── past/grademodel/
│   └── ITRI_CMSR100_DenseNet121_50_128_loss.hdf5  # 分類模型
├── paperReference/                    # 參考文獻
│   ├── 計量文章_脊椎MRI影像之椎間盤切割與分類技術_劉曉薇 賴程威_0610 1.pdf
│   ├── TWI722264B.pdf
│   └── US12136485.pdf
└── 輸出文件/
    ├── _result.jpg                    # 完整分析結果
    └── _result_disc.jpg               # 椎間盤標註結果
```

## 📚 相關論文

### 主要研究
**脊椎MRI影像之椎間盤切割與分類技術**
- 作者: 劉曉薇, 賴程威
- 機構: 工業技術研究院 (ITRI)
- 日期: 2021年6月

### 專利文獻
1. **TWI722264B** - 台灣專利
2. **US12136485** - 美國專利

所有文獻均可透過網頁界面下載或線上查看。

## 🔧 故障排除

### 常見問題

**1. 模型載入失敗**
```bash
# 檢查模型文件是否存在
ls -la ITRI_CMS_R100_unet_residual_vgg16.hdf5
ls -la past/grademodel/ITRI_CMSR100_DenseNet121_50_128_loss.hdf5
```

**2. 記憶體不足**
```bash
# 檢查系統記憶體
free -h
# 關閉其他程序釋放記憶體
```

**3. 端口被占用**
```bash
# 查找並終止占用端口的程序
sudo fuser -k 6060/tcp
```

**4. 圖像處理錯誤**
- 確保上傳的是有效的圖像文件
- 檢查圖像文件大小 (建議 < 10MB)
- 確認圖像格式為支援的類型

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request 來改進這個專案！

### 開發環境設置
1. Fork 這個專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權協議

本專案由工業技術研究院 (ITRI) 開發，相關權利請參考 ITRI 授權條款。

## 📧 聯絡資訊

- **開發單位**: 工業技術研究院 (ITRI)
- **技術支援**: [技術支援信箱]
- **專案維護**: [維護團隊信箱]

---

© 2021-2025 Industrial Technology Research Institute (ITRI). All rights reserved.
