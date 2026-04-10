# scorer README

本資料夾包含語氣評分系統，主要用於對中文句子做 1-4 級語氣評分。
目前此專案在實驗中固定使用 `V1_SCALE_DEFINITION` 作為 prompt 版本。

> 注意：`results/` 目錄為本地輸出暫存資料，不會同步到 GitHub。README 中不包含 `results/` 的結構說明。

## 目錄與主要檔案

- `scorer.py`
  - 主程式入口。
  - 讀取資料集、建立 few-shot prompt、對測試集執行評分。
  - 支援參數：`-v/--version`、`-m/--model`、`-d/--dataset`、`--csv`。
- `scorer_prompt.py`
  - 等級定義與 prompt 組成。
  - 包含 `V1_SCALE_DEFINITION`、`V2_SCALE_DEFINITION`、`V3_SCALE_DEFINITION`、`V4_SCALE_DEFINITION`。
  - 目前固定使用 `V1_SCALE_DEFINITION`。
- `model_clients.py`
  - 模型呼叫封裝，提供統一 `generate(prompt)` 介面。
  - 支援 OpenAI、Gemini、Breeze、Llama、Groq 等模型。
- `requirements.txt`
  - Python 套件需求列表。

## 使用方法

### 安裝套件

```bash
python -m pip install -r scorer/requirements.txt
```

### 環境設定

在 `scorer/` 目錄建立 `.env`，加入以下必要環境變數：

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

### 執行評分

```bash
cd scorer
python scorer.py
```

指定模型：

```bash
python scorer.py -m gpt-4o-mini
python scorer.py -m gemini-1.0
python scorer.py -m breeze
python scorer.py -m llama
```

指定資料集：

```bash
python scorer.py -d dataset_scorer.jsonl
```

使用 CSV 輸入：

```bash
python scorer.py --csv /path/to/file.csv
```

## 固定 prompt 版本

本 README 與現行實驗流程均使用 `V1_SCALE_DEFINITION`：

- 代表目前專案固定使用版本 1 的等級定義。
- `scorer.py` 預設版本為 1，對應 `V1_SCALE_DEFINITION`。
- 其他版本仍可在程式中選擇，但本說明不鼓勵變更以維持實驗一致性。

## prompt 組成

`scorer_prompt.py` 的 prompt 主要包含：

1. `scale_definition`
   - 描述 1-4 級語氣標準。
   - 目前使用 `V1_SCALE_DEFINITION`。
2. `HARD_NEGATIVES`（可選）
   - 包含易混淆的邊界範例。
   - 目前預設不啟用。
3. `Few-shot 範例`
   - 使用內建 `FEW_SHOT_DATASET`。
   - 每個 example 顯示 1-4 級對應示例，協助模型理解評分標準。

### prompt 流程

- 先輸出 `scale_definition`
- 再補上少量 `Few-shot 範例`
- 最後詢問模型：
  > 請對下面句子進行分析與打分

## 支援模型比較

| 模型名稱 | 封裝類型 | 設備類型 | 建議情境 |
|----------|----------|----------|----------|
| `gpt-4o-mini` | OpenAI | 線上 API | 預設推薦，用於高品質評分 |
| `gemini-1.0` | Gemini | 線上 API | 對比測試或多語言評分 |
| `breeze` | Breeze | 本地 GPU | 成本敏感或無網路情境 |
| `llama` | Llama | 本地 GPU | 本地推理與離線評分 |
| `groq` | Groq | 雲端硬體 | 低延遲商用硬體 |

## 現有實驗結果比較

以下數據來自 `scorer/results` 目錄中的已保存 CSV 檔案，均使用 `V1_SCALE_DEFINITION`。

| 模型 / 配置 | 結果檔案 | 測試筆數 | 準確率 |
|-------------|----------|--------|--------|
| `gpt-4o-mini` | `gpt-4o-mini/v1_scored_1774266926.csv` | 91 | 93.41% |
| `llama-3.1-8B-Instruct` zero-shot | `results_Llama-3.1-8B-Instruct_zero-shot/...` | 567 | 43.56% |
| `llama-3.1-8B-Instruct` 4-shot | `results_Llama-3.1-8B-Instruct_4-shot/...` | 567 | 29.98% |
| `llama-3.1-8B-Instruct` 8-shot | `results_Llama-3.1-8B-Instruct_8-shot/...` | 567 | 27.51% |
| `llama-3.1-8B-Instruct` 12-shot | `results_Llama-3.1-8B-Instruct_12-shot/...` | 567 | 26.63% |
| `llama-Breeze2-8B-Instruct` zero-shot | `results_Llama-Breeze2-8B-Instruct_zero-shot/...` | 567 | 25.40% |
| `llama-Breeze2-8B-Instruct` 4-shot | `results_Llama-Breeze2-8B-Instruct_4-shot/...` | 567 | 30.86% |
| `llama-Breeze2-8B-Instruct` 8-shot | `results_Llama-Breeze2-8B-Instruct_8-shot/...` | 567 | 34.92% |
| `llama-Breeze2-8B-Instruct` 12-shot | `results_Llama-Breeze2-8B-Instruct_12-shot/...` | 567 | 36.16% |


## 結果格式說明

`scorer.py` 產生的評分結果通常包含以下欄位：

| 欄位 | 說明 |
|------|------|
| `text` | 待評分的原始句子 |
| `score` | 基準分數或目標分數 |
| `predicted_score` | 模型輸出的分數 |
| `model_name` | 使用的模型名稱 |
| `prompt_type` | prompt 版本或類型 |

### CSV 輸入格式

若使用 `--csv` 參數，輸入檔案需包含：
- `text`
- `output`
- `target`

程式會把 `output` 當作待評分文本，並將 `target` 轉換為 `score` 進行比對。

## 重要注意事項

- `results/` 目錄僅限本地輸出使用，不包含在 GitHub 上傳內容中。
- 若要重現分數實驗，請先安裝相依套件並設定 `.env`。
- 目前實驗流程固定使用 `V1_SCALE_DEFINITION`，避免版本差異導致結果不一致。

## 參考命令

```bash
cd scorer
python scorer.py -v 1 -m gpt-4o-mini -d dataset_scorer.jsonl
python scorer.py -v 1 -m gemini-1.0 -d dataset_scorer.jsonl
python scorer.py -v 1 -m breeze -d dataset_scorer.jsonl
```

## 版本摘要

| 版本 | 說明 |
|------|------|
| `V1_SCALE_DEFINITION` | 固定使用版本，最基本的 1-4 級定義 |
| `V2_SCALE_DEFINITION` | 排他性邊界版本，強調軟化詞語 |
| `V3_SCALE_DEFINITION` | 改進版本，加入警告條款 |
| `V4_SCALE_DEFINITION` | 重新定義 1/2 邊界 |

---

**最後更新**: 2026-04-10
