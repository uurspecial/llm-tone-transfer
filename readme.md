# 基於 LLM 的可控語氣轉換

這個專案旨在實現大型語言模型（LLM）的語氣轉換與評分系統。

## 主要目標

- 使用 generator 模組生成不同語氣等級的改寫句。
- 使用 scorer 模組評估生成結果是否符合目標語氣等級。

## 目錄結構

```text
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
```

## 等級定義
| 等級 | 名稱 | 英文定義 | 核心特徵與描述 | 範例關鍵字/風格 |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **溫和** | Polite / Warm | 語氣柔軟，以維護關係或展現禮貌為目的。 | 謝謝、麻煩你、喔、吧、不好意思 |
| **2** | **中性** | Neutral / Factual | 如機器人或新聞報導般陳述事實，不帶個人情緒。 | 僅傳遞資訊，無語助詞 |
| **3** | **不滿** | Direct Anger | 情緒直接外露，直球對決。表達憤怒、指責或命令。 | 閉嘴、你很煩、爛透了、抱怨 |
| **4** | **酸** | Sarcastic | 陰陽怪氣、高級反諷。用誇獎形式貶低，帶有幽默感。 | 你的智商真是人類奇蹟、挖苦 |

## 資料集
| 檔案名稱 | 核心格式 | 關鍵欄位 | 主要用途 | 作用說明 |
| :--- | :--- | :--- | :--- | :--- |
| **dataset.jsonl** | `{"text", "score"}` | `score: 1-4` | **原始總庫** | 所有資料的來源，用於等比例切分出其他子集。 |
| **dataset_scorer.jsonl** | `{"text", "score"}` | `score: 1-4` | **評分範例** | 提供給判分模型（Scorer）作為 Few-shot 範例，定義各等級語氣基準。 |
| **dataset_test.jsonl** | `{"text", "score"}` | `score: 1-4` | **原始測試基底** | 紀錄測試句的「原始語氣」，作為改寫任務的來源輸入。 |
| **dataset_test_generator.jsonl** | `{"text", "target"}` | `target: 1-4` | **生成任務指令** | 由 `dataset_test.jsonl` 擴展而來，指定原句需改寫成的目標語氣。 |
### 資料切分說明
```text
[dataset.jsonl] ───┬───> [dataset_scorer.jsonl] (提供評分標準範例)
                   └───> [dataset_test.jsonl]   (作為測試原句與 Ground Truth)

### 資料集範例  
| 等級 (Score) | 語氣分類 | 範例語句 | 語義解析 |
| :---: | :--- | :--- | :--- |
| **1** | **溫和 (Warm)** | 請多指教，很高興認識你。 | 展現高度禮貌與親和力，意在建立正向連結。 |
| **2** | **中性 (Neutral)** | 我們還是照原本的計畫進行吧。 | 純粹的事實陳述與決策告知，不帶任何情緒色彩。 |
| **3** | **不滿 (Direct)** | 你這樣做只會讓大家更討厭你而已。 | 直接的負面情緒與指責，語氣強硬且具備威脅感。 |
| **4** | **酸 (Sarcastic)** | 我小時候被狗咬過，所以現在看你有點害怕。 | **高級反諷**：透過誇飾比喻將對方比作狗，語帶幽默卻極具貶低意涵。 |

## 模型介紹
- `scorer`：語氣裁判系統 負責評分 高準確率。
詳細內容請參考 [`scorer/README.md`](scorer/README.md)。
- `generator`：負責生成改寫句子和變更語氣。詳細內容請參考 [`generator/README.md`](generator/README.md)。


## 執行方式

1. 先安裝依賴：

```bash
pip install -r requirements.txt
```

2. 建立 `.env` 檔，並填入必要的 API key，例如：

```text
GOOGLE_API_KEY=你的_google_api_key
HUGGINGFACE_TOKEN=你的_huggingface_token
GROQ_API_KEY=你的_qroq_api_key
OPENAI_API_KEY=你的_openai_api_key
```

## generator結果比較


| 系列 | zero-shot | 4-shot | 8-shot | 12-shot |
|---|---|---|---|---|
| `Llama-3.1-8B-Instruct` | 43.56% | 29.98% | 27.51% | 26.63% |
| `Llama-Breeze2-8B-Instruct` | 25.40% | 30.86% | 34.92% | 36.16% |