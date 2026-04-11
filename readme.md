# llm-tone-transfer

這個專案旨在實現大型語言模型（LLM）的語氣轉換與評分系統。

## 主要目標

- 使用 generator 模組生成不同語氣等級的改寫句。
- 使用 scorer 模組評估生成結果是否符合目標語氣等級。

## 目錄結構

\`\`\`text
llm-tone-transfer/
├── src/                     # 原始程式碼與資料處理
│   └── data/                # 測試資料與 JSONL 資料集
├── scorer/                  # 評分模組，負責語氣等級判分
│   ├── model_clients.py     # 模型客戶端封裝
│   ├── README.md            # scorer 專屬說明文件
│   ├── requirements.txt     # scorer 專屬依賴清單
│   ├── scorer_prompt.py     # 評分 prompt 定義
│   └── scorer.py            # 評分主體邏輯
├── generator/               # 生成模組，負責語氣轉換任務
│   ├── main.py              # 生成流程主程式
│   ├── generator.py         # 各模型生成器定義
│   ├── change.py            # 同等級換句話說/同義改寫
│   ├── prompts.py           # prompt 建立器
│   ├── scorer.py            # local scorer wrapper
│   └── utils.py             # 共享工具與環境初始化
├── results/                 # 生成與評分結果檔案（輸出資料）
├── .env                     # 環境變數設定（自行設定）
├── requirements.txt        # 專案依賴套件清單
└── README.md                # 專案說明文件
\`\`\`

## 重要檔案說明

- `generator/`：生成模組，負責語氣轉換與句子生成流程。詳細內容請參考 [`generator/README.md`](generator/README.md)。
- `scorer/`：評分模組，負責語氣等級判分與模型封裝。詳細內容請參考 [`scorer/README.md`](scorer/README.md)。

## 執行方式

1. 先安裝依賴：

\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. 建立 `.env` 檔，並填入必要的 API key，例如：

\`\`\`text
GOOGLE_API_KEY=你的_google_api_key
HUGGINGFACE_TOKEN=你的_huggingface_token
GROQ_API_KEY=你的_qroq_api_key
OPENAI_API_KEY=你的_openai_api_key
\`\`\`



## generator結果比較

可參考 [`generator/README.md`](generator/README.md) 中 `Llama-3.1-8B-Instruct` 與 `Llama-Breeze2-8B-Instruct` 系列預測準確率比較表。