# prompts.py
import json
import random
import os

#
class BasicPromptBuilder1:
    """
    只使用態度與邏輯和 Few-Shot 
    """
    def __init__(self):
        self.level_defs = {
            1: "【Level 1：溫和】(重點在對方感受，語氣柔和)",
            2: "【Level 2：中性】(陳述事實，冷靜直接)",
            3: "【Level 3：不滿】(明顯不耐煩，帶情緒壓力)",
            4: "【Level 4：酸】(反語、假稱讚、陰陽怪氣)"
        }

    def build_prompt(self, original_text: str, target_level: int, original_level: int = None, max_chars: int = 60) -> str:
        """
        回傳語氣改寫 Prompt
        
        參數：
            original_text: 原始句子
            target_level: 要改寫為的目標等級 (1-4)
            original_level: 原句的原始等級 (1-4)，如果提供則說明是等級轉換
            max_chars: 最大字數限制
        """
        target_desc = self.level_defs.get(target_level, self.level_defs[2])
        examples_str = ""  # BasicPromptBuilder1 不使用 few-shot 範例

        if original_level is not None and original_level != target_level:
            # 等級轉換模式
            level_labels = {1: "一級", 2: "二級", 3: "三級", 4: "四級"}
            original_level_name = level_labels.get(original_level, f"第{original_level}級")
            target_level_name = level_labels.get(target_level, f"第{target_level}級")
            
            return f"""你是一位精通漢語語用學的「語氣改寫專家」。
原句已被評定為【第 {original_level} 級：{original_level_name}】語氣。
請將原句改寫為【第 {target_level} 級：{target_level_name}】的語氣。

### 目標語氣與態度：
{target_desc}

### 風格參考 (請學習其邏輯)：
{examples_str}

### 嚴格執行規則：
1. **保留原意**：改寫時必須保留原句的核心意思，只改變語氣層級。
2. **視角固定**：保持原句的人稱視角。
3. **多樣性**：請根據句子的情境發揮創意，**不要**死板地重複範例中的詞彙。
4. **長度限制**：{max_chars} 字以內。

### 開始改寫：
原始句子（第 {original_level} 級）："{original_text}"
改寫為第 {target_level} 級：
請只輸出改寫後句子，勿加說明。
"""
        else:
            # 普通改寫模式
            return f"""你是一位精通漢語語用學的「語氣改寫專家」。
請將使用者的「原始句子」改寫為符合「Level {target_level}」的語氣。

### 目標語氣與態度：
{target_desc}

### 風格參考 (請學習其邏輯)：
{examples_str}

### 嚴格執行規則：
1. **保留原意**：**絕對不要回答問題**。如果原句是問句，改寫後也要是問句（或反問）。
2. **視角固定**：保持原句的人稱視角。
3. **多樣性**：請根據句子的情境發揮創意，**不要**死板地重複範例中的詞彙。
4. **長度限制**：{max_chars} 字以內。

### 開始改寫：
原始句子："{original_text}"
Level {target_level} 改寫："""



class DataPromptBuilder1:
    """
    從 dataset_test.jsonl 動態載入每個等級前30句作為 Few-shot 範例的 Prompt Builder
    """
    def __init__(self, json_path=None, sample_size=30):
        """
        初始化 Prompt Builder，從指定的 JSONL 檔案載入範例
        
        參數：
            json_path: JSONL 檔案路徑。如果為 None，預設使用 src/data/dataset_test.jsonl
            sample_size: 每個等級要載入的句子數量，預設 30
        """
        self.level_defs = {
            1: "【Level 1：溫和】(重點在對方感受，語氣柔和)",
            2: "【Level 2：中性】(陳述事實，冷靜直接)",
            3: "【Level 3：不滿】(明顯不耐煩，帶情緒壓力)",
            4: "【Level 4：酸】(反語、假稱讚、陰陽怪氣)"
        }
        self.levels = [1, 2, 3, 4]
        
        # 從 dataset_test.jsonl 動態載入 few-shot 範例
        self.custom_examples = self._load_examples_from_dataset(json_path, sample_size)
    
    def _load_examples_from_dataset(self, json_path=None, sample_size=30):
        """
        從 JSONL 檔案中動態載入每個等級的範例
        """
        # 若未指定路徑，使用預設路徑 (相對於 generator 目錄的上一級)
        if json_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            json_path = os.path.join(parent_dir, "src", "data", "dataset_test.jsonl")
        else:
            # 如果指定了路徑，優先嘗試相對於 generator 目錄的上一級
            if not os.path.isabs(json_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                potential_path = os.path.join(parent_dir, json_path)
                if os.path.exists(potential_path):
                    json_path = potential_path
        
        examples = {1: [], 2: [], 3: [], 4: []}
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            score = int(data.get("score", 0))
                            if score in examples:
                                # 只保留每個等級前 sample_size 筆
                                if len(examples[score]) < sample_size:
                                    examples[score].append(data.get("text", ""))
            except Exception as e:
                print(f"⚠️ 警告：無法載入 {json_path}：{e}")
        else:
            print(f"⚠️ 警告：找不到檔案 {json_path}")
        
        return examples

    def build_prompt(self, original_text: str, target_level: int, original_level: int = None) -> str:
        """
        使用固定的自訂範例來構建 Prompt
        """
        if target_level not in self.levels:
            return "Error: Level must be 1, 2, 3, or 4."

        target_desc = self.level_defs.get(target_level)
        level_labels = {1: "一級", 2: "二級", 3: "三級", 4: "四級"}
        target_level_name = level_labels.get(target_level, f"第{target_level}級")
        
        # 抓取對應等級的 4 筆指定範例
        selected_examples = self.custom_examples.get(target_level, [])
        
        examples_str = ""
        if selected_examples:
            examples_str = f"以下是此語氣的 {len(selected_examples)} 筆【參考範例句】(請學習其用詞與風格)：\n"
            for ex in selected_examples:
                examples_str += f"- {ex}\n"
        else:
            examples_str = "(此等級目前無參考資料，請依定義發揮)\n"

        if original_level is not None and original_level != target_level:
            # 等級轉換模式
            original_level_name = level_labels.get(original_level, f"第{original_level}級")
            return f"""你是一位精通漢語語用學的「語氣改寫專家」。
原句已被評定為【第 {original_level} 級：{original_level_name}】語氣。
請將原句改寫為【第 {target_level} 級：{target_level_name}】的語氣。

### 目標語氣：
{target_desc}

### 參考範例（{target_level_name}風格）：
{examples_str}
### 嚴格執行規則：
1. **保留原意**：改寫時必須保留原句的核心意思，只改變語氣層級。
2. **視角固定**：保持原句的人稱視角。
3. **多樣性**：請根據句子的情境發揮創意，**不要**死板地重複範例中的詞彙。

### 開始改寫：
原始句子（第 {original_level} 級）："{original_text}"
改寫為第 {target_level} 級："""
        else:
            # 普通改寫模式
            level_label = level_labels.get(target_level, f"第{target_level}級改寫")
            return f"""你是一位語氣改寫專家。
請將「原句」改寫為「{level_label}」版本。

### 目標語氣：
{target_desc}

### 參考範例：
{examples_str}
### 你的任務：
原句："{original_text}"
{level_label}：
請只輸出改寫後句子，勿加說明。
"""
