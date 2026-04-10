# change.py
"""
用 Breeze 模型進行同等級的句義改寫（Paraphrase）
當 target_level == correct_level 時調用，生成相同語氣但不同表述的版本
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import DataPromptBuilder1


class BreezeParaphraser:
    """
    使用 Breeze 模型進行同等級句義改寫
    例如：輸入第 2 級的"好的，我知道了"，輸出另一個第 2 級的"沒問題，我明白了"
    """
    def __init__(self):
        self.model_name = "MediaTek-Research/Breeze-7B-Instruct-v1_0"
        print(f"🔄 正在載入 Breeze Paraphraser: {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        # 參考 dataset.jsonl 的 DataPromptBuilder
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(root_dir, "dataset.jsonl")
        self.data_prompter = DataPromptBuilder1(dataset_path)

        print(f"✅ Breeze Paraphraser 已載入")

    def build_prompt(self, original_text: str, level: int, max_chars: int = 60) -> str:
        """
        構建同等級句義改寫 Prompt（改为使用 DataPromptBuilder1）
        
        參數：
            original_text: 原始句子
            level: 句子的語氣等級 (1-4)
            max_chars: 最大字數限制
        """
        prompt = self.data_prompter.build_prompt(original_text, target_level=level, original_level=level)
        prompt += "\n\n請只輸出改寫後句子，勿加說明。"
        return prompt


    def paraphrase(self, text: str, level: int, temperature: float = 0.7, max_new_tokens: int = 100) -> str:
        """
        執行句義改寫
        
        參數：
            text: 原始句子
            level: 語氣等級 (1-4)
            temperature: 生成多樣性（0.5-1.0）
            max_new_tokens: 最大輸出令牌數
        
        返回：
            改寫後的句子
        """
        prompt = self.build_prompt(text, level)
        
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_ids = self.tokenizer([inputs], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        decoded = self.tokenizer.decode(outputs[0][inputs_ids.input_ids.shape[1]:], skip_special_tokens=True)
        result = (decoded or "").strip().split('\n')[0]
        return result if result else "Error"
