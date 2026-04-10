# Dataset 說明

## 1. 4個檔案的用途

### dataset.jsonl
* 內容格式：每行 `{"text": "...", "score": "1"}`
* 範例：`{"text": "這個進度很理想，可以提早休息。", "score": "1"}`
* 作用：原始資料集。接下來等比例切分成dataset_scorer.jsonl和dataset_test.jsonl
  
### dataset_scorer.jsonl
* 內容格式：每行 `{"text": "...", "score": "1"}`
* 範例：`{"text": "這個進度很理想，可以提早休息。", "score": "1"}`
* 用途：
    * Scorer 訓練資料集
* 作用：提供「評分範例」給判分模型，讓模型知道 1～4 級語氣的範例句。

### dataset_test_generator.jsonl
* 內容格式：每行 `{"text": "...", "target": "2"}`、`{"text": "...", "target": "3"}`、`{"text": "...", "target": "4"}`。
* 範例：
    * `{"text": "原來躺平也能叫努力。", "target": "1"}`
    * `{"text": "原來躺平也能叫努力。", "target": "2"}`
    * `{"text": "原來躺平也能叫努力。", "target": "3"}`
* 用途：
    * `new/main.py` 的資料集，並以 `target` 當作想要生成的語氣等級。
* 作用：給生成器的輸入，告訴模型：
    * 原句是什麼
    * 要改寫成哪一個目標等級

### dataset_test.jsonl
* 內容格式：每行 `{"text": "...", "score": "1"}`、`{"text": "...", "score": "2"}` 等。
* 用途：
    * `new2/main.py` 會讀這個檔案，作為原始語句與其正確分數。
    * 有些 prompt builder 也會用它（或類似的資料）來做「原句 + 分數」的 prompt 設計。
* 作用：它是「原始測試資料集」，代表每句話本來的語氣級別。

## 2. dataset_test_generator.jsonl 的由來

從檔案內容和使用方式看，`dataset_test_generator.jsonl` 是從 `dataset_test.jsonl` 衍生出來的，做法大致是：
* 先拿 `dataset_test.jsonl` 中的每一個原句
* 再為該句建立「不同的 target 等級任務"

例如：
* `dataset_test.jsonl` 中：
    * `{"text": "這個進步很大，繼續保持。", "score": "1"}`
* 產生到 `dataset_test_generator.jsonl`：
    * `{"text": "這個進步很大，繼續保持。", "target": "2"}`
    * `{"text": "這個進步很大，繼續保持。", "target": "3"}`
    * `{"text": "這個進步很大，繼續保持。", "target": "4"}`

也就是：
* 原本是 1 級的句子，擴展成要改寫成 2、3、4 級的三個測試任務
* 如果原句是 4 級，則在 generator 測試集中可能會產生 `target:1,2,3` 這樣的任務
