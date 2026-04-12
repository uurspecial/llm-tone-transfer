# Generator 測試說明

## 檔案結構

```
generator/
├── main.py        # 主程式，負責讀資料、呼叫生成器和 scorer、寫出結果
├── generator.py   # 不同模型的生成器類別，負責依照 target_level 產生改寫句
├── change.py      # 同等級換句話說功能
├── scorer.py      # 評分模組，回傳句子分數等級（假設永遠正確）
├── prompts.py     # prompt 建立器，負責組成 Basic 與 Data 兩種 prompt
├── utils.py       # 環境與共用工具函式
└── README.md      # 這份說明文件
```

## 各檔案說明

### `main.py`
- 主程式入口。
- 讀取資料集，逐句處理。
- 先評分原句，再依目標等級呼叫不同模型生成改寫句。
- 支援 `Groq-8B`、`Breeze`、`Llama` 三個生成模型。
- 每個模型會先以 Basic prompt 再以 Data prompt 生成兩種版本。
- 將結果寫成 CSV，並顯示 `is_correct` 正確率統計。

### `generator.py`
- 定義不同模型的生成器類別。
- 根據輸入 prompt 產生目標等級的改寫句。

### `change.py`
- 同等級換句話說功能。
- 保持語氣等級不變的情況下，產生句子變體。

### `scorer.py`
- 評分模組，回傳句子語氣等級分數。
- 本 README 假設 `scorer.py` 的評分結果永遠正確。
- `main.py` 會以 `predicted_score == target_level` 判斷生成是否正確。

### `prompts.py`
- 建立 prompt 的邏輯。
- `BasicPromptBuilder1`：僅提供語氣等級定義與輸入句子，zero-shot。
- `DataPromptBuilder1`：  
4-shot：包含 4 個範例（每個語氣類別各 1 個）。  
8-shot：包含 8 個範例（每個語氣類別各 2 個，為 4-shot 的超集）。  
12-shot：包含 12 個範例（每個語氣類別各 3 個，為 8-shot 的超集）。  
**目前每個等級只有 1 一句參考範例。**

### `utils.py`
- 環境設定與共用工具。
- 例如讀取 `.env`、取得 API key、初始化執行環境等。

## 資料集位置

`main.py` 會讀取：

- `src/data/dataset_test_generator.jsonl`

## 執行流程

1. `main.py` 讀取 `src/data/dataset_test_generator.jsonl`。
2. 取出每筆資料的 `original_text` 與 `target_level`。
3. 使用 `scorer.py` 評分原句，得到 `original_score`。
4. 對每個模型執行：
   - Basic prompt 生成改寫句。
   - Data prompt 生成改寫句。
5. 評分每個生成句，得到 `predicted_score`。
6. 比對 `predicted_score` 與 `target_level`，計算 `is_correct`。
7. 將結果寫成 CSV，並展示各模型與 prompt 的正確率。

## 假設條件

- `scorer.py` 的評分被視為絕對正確。
- `is_correct` 判斷條件為 `predicted_score == target_level`。
- 這代表輸出的正確率直接反映模型生成是否達到目標等級。

## 測試步驟

```bash
cd generator
python main.py
```

## 範例輸出

```text
============================================================
Processing S002: 你一定要在這種時候潑冷水嗎？
  → 目標生成等級: 4級
============================================================
  ✓ 原句評分結果: 3 級

    生成第 4 級版本...

    >>> 模型：Groq-8B
      [4級]: 你真的是來這裡找刺激的麼？ (Score: 4) ✓
      [4級]: 根據參考範例和嚴格執行規則，以下是改寫成第 4 級語氣的結果：

改寫句子："你還蠻有意思的，為什麼非要在這種時候潑冷水呢？"

在這個改寫中，我保留了原句的核心意思（你在這種時候潑冷水），但變更了語氣層級以達到第 (Score: 2) ✗

    >>> 模型：Breeze
      [4級]: 你真是個不折不扣的潑冷水鬼啊！ (Score: 4) ✓
      [4級]: 唉唷，別在重要時刻澆澆澆冷水嘛。 (Score: 3) ✗

    >>> 模型：Llama
      [4級]: 你居然想在這種時候潑冷水？ (Score: 4) ✓
      [4級]: * 「哼」: This particle is often used to express disdain, annoyance, or skepticism, which is in line (Score: 4) ✓

  ✓ S002 完成
```
## 實驗設計 (程式碼不在這)

每個模型各進行四種提示策略的實驗：

- **zero-shot**：僅提供語氣等級定義與輸入句子。
- **4-shot**：包含 4 個範例（每個語氣類別各 1 個）。
- **8-shot**：包含 8 個範例（每個語氣類別各 2 個，為 4-shot 的超集）。
- **12-shot**：包含 12 個範例（每個語氣類別各 3 個，為 8-shot 的超集）。
### 生成結果比較（預測準確率）

以下表格比較 `Llama-3.1-8B-Instruct` 系列與 `Llama-Breeze2-8B-Instruct` 系列的預測準確率：

| 系列 | zero-shot | 4-shot | 8-shot | 12-shot |
|---|---|---|---|---|
| `Llama-3.1-8B-Instruct` | 43.56% | 29.98% | 27.51% | 26.63% |
| `Llama-Breeze2-8B-Instruct` | 25.40% | 30.86% | 34.92% | 36.16% |

從比較結果可見：
- `Llama-Breeze2-8B-Instruct` 在 few-shot 設定（4-shot、8-shot、12-shot）上表現較佳。
- `Llama-3.1-8B-Instruct` 在 zero-shot 上具有較高準確率。



