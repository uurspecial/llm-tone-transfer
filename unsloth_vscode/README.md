# Unsloth Classifier Workflow (VSCode)

這份 README 只整理你目前分類任務會用到的 5 支檔案：

- `common.py`
- `train_classifier.py`
- `test_classifier.py`
- `stratified_5_fold.py`
- `train_5_fold.py`

任務標籤固定為：

- `1-溫和`
- `2-中性`
- `3-不滿`
- `4-酸`

## 1) 每個檔案在做什麼

### `common.py`

共用工具與常數：

- 路徑處理：`expand_path`
- 從 LoRA 讀 base model：`read_base_model_from_adapter`
- 從多種欄位組文字：`build_classification_text`
- 標籤正規化：`normalize_label`

重點是 `normalize_label`：

- 支援 `1..4`、`0..3`、`1-溫和`、`中性`、`sarcastic` 等多種寫法
- 先優先把資料集常見的 `1..4` 轉成內部 `0..3`，避免預測整體 +1 的錯位

---

### `train_classifier.py`

用 Unsloth 的 sequence classification + LoRA 進行訓練，支援 stratified K-fold。

主要功能：

- 可只用 `--train-file`，或合併 `--test-file` 後再做 CV
- 清理資料列（可容忍 label/score/output 欄位）
- `StratifiedKFold` 分層切分
- 每個 fold 輸出 LoRA 到 `--lora-output-dir/fold_k`
- 累積各 fold 指標到 `--metrics-file`，不覆蓋先前 fold
- 自動處理 4-bit Byte 初始化錯誤，必要時 fallback `load_in_4bit=False`

常用參數：

- `--train-file`
- `--test-file`（可省略）
- `--base-model`
- `--lora-output-dir`
- `--metrics-file`
- `--n-splits`（預設 5）
- `--fold-index`（只跑某一折）
- `--split-only`（只看分割，不訓練）

---

### `test_classifier.py`

用已訓練好的 LoRA classifier 做推論/評估。

主要功能：

- 支援單句模式：`--sentence`
- 支援整份資料集評估：`--test-file`
- 輸出：
  - `classifier_test_results.csv`
  - `classifier_metrics.json`
- CSV 欄位為：
  - `id,text,gold_label_id,pred_label_id,is_correct`
- 若 4-bit 載入遇到 Byte 初始化問題，會自動 fallback

---

### `stratified_5_fold.py`

把單一輸入檔（通常 `train.jsonl`）切成分層 K-fold 資料檔。

輸出：

- `output-dir/fold_0/train.jsonl`, `output-dir/fold_0/test.jsonl`, ...
- `output-dir/summary.json`

`summary.json` 會記錄：

- 每 fold 的 train/test 數量
- 每 fold 各類別比例
- train/test size 標準差
- 各類別比例標準差

---

### `train_5_fold.py`

自動依序呼叫 `train_classifier.py` 跑完整 K-fold，最後統整平均與標準差。

流程：

1. for fold in `0..n_splits-1` 呼叫 `train_classifier.py --fold-index fold`
2. 讀取 `--metrics-file` 中的 `fold_metrics`
3. 輸出 `--summary-file`（mean/std of acc, macro-F1, loss）

## 2) 資料格式

建議使用 `jsonl`，每行一筆。

建議欄位：

```json
{"id": 1, "text": "這句話內容", "label": 1}
```

也支援：

- 文字欄位：`sentence` / `text` / `input`（或 `instruction + input`）
- 標籤欄位：`label` / `labels` / `score` / `output`

## 3) 常用指令

### A. 先只檢查分層切分（不訓練）

```bash
python unsloth_vscode/train_classifier.py \
  --train-file unsloth_vscode/data/train.jsonl \
  --n-splits 5 \
  --split-only
```

### B. 建立固定的 stratified 5-fold 檔案

```bash
python unsloth_vscode/stratified_5_fold.py \
  --input-file unsloth_vscode/data/train.jsonl \
  --output-dir unsloth_vscode/data/stratified_5_fold \
  --n-splits 5 \
  --seed 3407
```

### C. 跑完整 5-fold 訓練（自動彙整 mean/std）

```bash
python unsloth_vscode/train_5_fold.py \
  --script-path unsloth_vscode/train_classifier.py \
  --train-file unsloth_vscode/data/train.jsonl \
  --base-model unsloth/Meta-Llama-3.1-8B \
  --lora-output-dir unsloth_vscode/classifier_lora_cv \
  --metrics-file unsloth_vscode/classifier_cv_metrics.json \
  --summary-file unsloth_vscode/classifier_cv_summary.json \
  --n-splits 5
```

### D. 評估某一折模型（例如 fold_0）

```bash
python unsloth_vscode/test_classifier.py \
  --test-file unsloth_vscode/data/test.jsonl \
  --lora-dir unsloth_vscode/classifier_lora_cv/fold_0 \
  --results-file unsloth_vscode/classifier_test_results.csv \
  --report-file unsloth_vscode/classifier_metrics.json
```

### E. 單句推論

```bash
python unsloth_vscode/test_classifier.py \
  --lora-dir unsloth_vscode/classifier_lora_cv/fold_0 \
  --sentence "你最近回訊息很慢耶"
```

## 4) 常見問題

### Q1. `normal_kernel_cuda not implemented for 'Byte'`

這通常是 4-bit sequence-classification 初始化問題。腳本已內建 fallback；若仍不穩定，可明確加：

```bash
--no-load-in-4bit
```

### Q2. `externally-managed-environment`

請使用專案/conda 環境，不要用系統 Python 的 pip。

### Q3. `device_map='auto' distributed mode`

請單進程執行（不要用 distributed launcher），或清掉 rank/world size 相關環境變數。

## 5) 輸出檔案速查

- LoRA 權重：`<lora-output-dir>/fold_k/`
- 訓練 CV 指標：`classifier_cv_metrics.json`
- 5-fold 統整：`classifier_cv_summary.json`
- 測試明細：`classifier_test_results.csv`
- 測試指標：`classifier_metrics.json`
