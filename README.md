# DISC - 脊椎椎間盤 MRI 影像切割與分類系統

基於深度學習的腰薦椎 MRI 影像椎間盤（Intervertebral Disc）語意分割與退化等級分類系統。

## 功能

- **椎間盤語意分割**：使用 U-Net + Residual VGG16 模型對 MRI 影像進行椎間盤區域切割
- **退化等級分類**：依據 Pfirrmann 分級法自動判定椎間盤退化程度（Grade 1-5）
- **RAG 醫療問答助手**：整合 LangChain + Ollama + FAISS 向量資料庫，提供醫療器材相關知識問答
- **多種 UI 介面**：提供 Flask Web UI、Gradio UI 等多種互動方式

## 專案結構

```
DISC/
├── predict.py              # 核心推論模組 (MLpridict class)
├── disc_model.py           # 模型載入與單張測試
├── do_Test.py              # 批次測試腳本
├── flask_disc_ui2.py       # Flask Web UI（分割 + 分級）
├── flask_discchatbot_app.py # Flask + 聊天機器人整合 UI
├── flask_chat_app_v2.py    # Flask 聊天介面 v2
├── gradio_ui.py            # Gradio 完整 UI
├── gradio_ui_simple.py     # Gradio 簡易 UI
├── chatbot.py              # RAG 聊天機器人模組
├── chatbot_test.py         # 聊天機器人測試
├── db/                     # FAISS 向量資料庫（醫療器材知識）
├── templates/              # Flask HTML 模板
│   ├── index.html          # 主頁面
│   ├── integrated.html     # 整合介面
│   └── chat.html           # 聊天介面
├── static/                 # 前端靜態資源 (CSS/JS)
├── testresult/             # 測試結果輸出
├── paperReference/         # 相關論文與專利參考文獻
├── ico/                    # 應用程式圖標
└── disc_req.yaml           # Conda 環境依賴
```

## 模型架構

| 模型 | 用途 | 輸入尺寸 |
|------|------|----------|
| U-Net + Residual VGG16 | 椎間盤語意分割 | 512×512×3 |
| DenseNet121 | 退化等級分類 (Pfirrmann Grade) | 128×128 |

## 環境需求

- Python 3.8+
- TensorFlow / Keras
- OpenCV
- Flask / Gradio
- LangChain + FAISS（聊天功能）
- CUDA GPU（建議）

## 快速開始

### 1. 建立 Conda 環境

```bash
conda create -n disc python=3.8
conda activate disc
pip install tensorflow keras opencv-python flask gradio
pip install langchain langchain-community faiss-cpu sentence-transformers
```

### 2. 啟動 Web UI

```bash
# Flask 椎間盤分析 UI（含分割 + 分級）
python flask_disc_ui2.py

# Flask 整合版（分析 + 聊天機器人）
python flask_discchatbot_app.py

# Gradio UI
python gradio_ui.py
```

### 3. 批次測試

```bash
python do_Test.py
```

## 使用方式

1. 開啟 Web 介面（預設 http://localhost:5000）
2. 上傳腰薦椎 MRI 影像（支援 PNG/JPG）
3. 系統自動進行椎間盤分割，標示各椎間盤位置
4. 顯示每個椎間盤的退化等級（Pfirrmann Grade 1-5）
5. 可透過聊天介面詢問醫療器材相關問題

## 參考文獻

- Multitask Deep Learning for Segmentation and Lumbosacral Spine Inspection
- 脊椎 MRI 影像之椎間盤切割與分類技術（計量文章）
- 相關專利：TW202324441A, TWI722264B, US12136485

## 注意事項

- 模型檔案 (`.hdf5`) 因檔案過大未包含在 git 中，請另行取得
- `past/` 目錄包含歷史模型，亦未上傳至 git
