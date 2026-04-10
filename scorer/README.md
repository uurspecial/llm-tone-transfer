# scorer 在不同prompt和模型的表現

統一管理三個版本的 prompt，支持透過命令行參數選擇評分版本。

## 目錄結構

```
/Users/proflin/scorer/
├── scorer_prompt.py      # 三個版本的 prompt 定義
├── scorer.py             # 主程式（支持版本選擇）
├── requirements.txt      # 依賴套件（可選）
├── results/              # 輸出結果資料夾
└── README.md             # 本文件
```

## 三個 Prompt 版本

### 版本 1：基礎版本
- 來源：原始 `score_sentences.py`
- 特點：簡潔的語氣定義，沒有 hard_negatives 範例
- 使用場景：快速評分

### 版本 2：排他性邊界版本
- 來源：原始 `scorer_sentences2.py`
- 特點：【必須】包含軟化語氣的詞彙，包含 hard_negatives 範例
- 使用場景：嚴格標準評分
- 輸出格式：要求 CoT (Chain of Thought) 分析

### 版本 3：改進版本（推薦）
- 來源：原始 `scorer_sentences3.py`
- 特點：【通常】包含軟化語氣 + 防呆警告條款
- 使用場景：生產環境，防止過度腦補
- 輸出格式：要求 CoT 分析 + 警告提示

## 使用方法

### 基本用法

```bash
# 使用預設版本（版本 3）
python scorer.py

# 指定版本 1
python scorer.py -v 1

# 指定版本 2
python scorer.py --version 2

# 使用自訂資料集
python scorer.py -v 3 -d /path/to/dataset.jsonl
```

### 命令行參數

| 參數 | 簡寫 | 類型 | 預設值 | 說明 |
|------|------|------|--------|------|
| `--version` | `-v` | int | 3 | 選擇 prompt 版本 (1, 2, 3) |
| `--dataset` | `-d` | str | dataset1.jsonl | 資料集檔案路徑 |

## 輸出

程式執行完成後，結果儲存到 `results/` 資料夾：
- `scored_output_v1.csv` - 版本 1 的評分結果
- `scored_output_v2.csv` - 版本 2 的評分結果
- `scored_output_v3.csv` - 版本 3 的評分結果

每個檔案包含：
- 原始文本 (`text`)
- 真實分數 (`score`)
- 預測分數 (`predicted_score`)

## 程式輸出

評分完成後會顯示：
1. ✅ 資料集分割結果
2. 📊 詳細分類報告 (Precision, Recall, F1)
3. 📉 混淆矩陣
4. 💾 儲存位置確認

## 依賴套件

```
pandas
scikit-learn
openai
python-dotenv
```

## 環境設定

1. 建立 `.env` 檔案在 scorer 資料夾中
2. 新增 OpenAI API 金鑰：

```
OPENAI_API_KEY=your_api_key_here
```

## 概念說明

### 四級語氣標準

#### L1 - 溫和 (Polite / Warm)
- 有同理心
- 通常包含軟化語氣（不好意思、麻煩了、喔、吧、請）
- 或表達關心、幫忙的善意

#### L2 - 中性 (Neutral / Factual)
- 機器人般陳述事實
- 乾脆、沒有情緒字眼

#### L3 - 不滿 (Direct Anger / Complaint)
- 直接表達憤怒、指責、命令或抱怨
- 直球對決，沒有幽默感

#### L4 - 酸 (Sarcastic / Mocking)
- 高級反諷、陰陽怪氣
- 表面正面但實際貶低
- 使用誇飾的比喻

## 版本差異

| 特性 | V1 | V2 | V3 |
|------|----|----|-----|
| Hard Negatives | ❌ | ✅ | ✅ |
| CoT 輸出格式 | ❌ | ✅ | ✅ |
| L1 要求軟化詞 | 【常用】| 【必須】| 【通常】|
| 警告防呆條款 | ❌ | ❌ | ✅ |
| 推薦指數 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 常見問題

### Q: 為什麼版本 3 是推薦版本？
A: 版本 3 加入了警告條款「除非句子中有明顯的邏輯矛盾或誇飾，否則請勿將單純的禮貌與關心過度解讀為反諷 (L4)」，防止 LLM 過度腦補，提高準確度。

### Q: 如何切換版本進行對比？
A: 執行相同的命令但改變 `-v` 參數：
```bash
python scorer.py -v 1
python scorer.py -v 2
python scorer.py -v 3
```
結果會分別儲存在 `scored_output_v1.csv`、`scored_output_v2.csv`、`scored_output_v3.csv`。

### Q: 如何新增第四個版本？
A: 編輯 `scorer_prompt.py`，按照現有版本的格式新增 V4 定義，然後在 `scorer.py` 中更新 argparse 的 choices。

## 訊息說明

| 符號 | 含義 |
|------|------|
| 🚀 | 程式開始/進度 |
| ✅ | 成功 |
| ❌ | 失敗/錯誤 |
| ⚠️ | 警告 |
| 📊 | 統計資訊 |
| 📉 | 詳細分析 |
| 💾 | 檔案儲存 |

## 技術細節

### 評分邏輯
1. 載入 `dataset1.jsonl` 資料
2. 執行 80-20 分層抽樣（Few-Shot 範例 vs 測試集）
3. 為每句文本呼叫 OpenAI API (gpt-4o-mini)
4. 使用 Regex 解析回應中的分數
5. 計算精準度、Recall、F1 分數
6. 生成混淆矩陣

### Retry 機制
- 最多重試 5 次
- 指數退避：2s → 4s → 8s → 16s → 32s
- 處理速率限制 (429 error) 和額度不足

## 修改記錄

| 版本 | 日期 | 變更 |
|------|------|------|
| 1.0 | 2024-XX-XX | 初始版本，整合三個 prompt 版本 |

---

**最後更新**: 2026-03-10
