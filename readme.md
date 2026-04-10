# llm-tone-transfer

這個專案旨在實現大型語言模型（LLM）的語氣轉換與評分系統。

## 目錄結構

```text
llm-tone-transfer/
├── src/                # 原始程式碼
│   ├── prompt/         # 存放各種任務的 Prompt 範本
│   ├── data/           # 原始數據或處理後的數據集
│   └── util/           # 共用工具函式（如 API 調用、文本清理）
├── scorer/             # 模型評分模組（Scorer），用於評估語氣轉換效果
├── generator/          # 生成模組，負責執行語氣轉換任務
├── result/             # 存放實驗結果、生成的文本或評分報表
├── .env                # 環境變數設定（自行設定）
├── requirements.txt    # 專案依賴套件清單
└── README.md           # 專案說明文件
