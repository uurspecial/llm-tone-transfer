"""
模型封裝模組：將線上與離線模型都包成統一介面。

只要遵守 BaseModelClient 介面，就能被 scorer.py 在執行階段插入。

線上模型示例：OpenAI API、gemini
離線模型示例：llama 或 qwen，可透過 huggingface/transformers、llama.cpp 等工具呼叫。

使用方式：
    client = model_clients.get_model_client(model_name)
    output = client.generate(prompt)
"""

from abc import ABC, abstractmethod
from typing import Optional
import os
import re
import time

try:
    import torch
except ImportError:
    torch = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    Groq = None
    HAS_GROQ = False


class BaseModelClient(ABC):
    """統一的模型介面"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """給定 prompt 回傳模型的原始文字輸出"""
        pass


from typing import Optional


class OpenAIModelClient(BaseModelClient):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name)
        if OpenAI is None:
            raise RuntimeError("OpenAI 套件未安裝，無法使用線上模型")
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 尚未設定")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """呼叫 OpenAI ChatCompletions API，保證回傳純字串。"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content
        if content is None:
            # 雖然理論上不該發生，但避免型別錯誤
            raise RuntimeError("OpenAI 回應內容為 None，無法解析成字串")
        assert isinstance(content, str)
        return content


class GeminiModelClient(BaseModelClient):
    """Gemini 模型介面。

    使用 google.generativeai（可選）或其他 Gemini SDK。
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name)
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("未安裝 google-generativeai，無法使用 Gemini 模型")

        if api_key is None:
            # 配合你目前的設定，讀取 GOOGLE_API_KEY
            api_key = os.getenv("GOOGLE_API_KEY") 
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 尚未設定")

        genai.configure(api_key=api_key) # type: ignore
        # 【修改重點 1】初始化模型物件
        self.model = genai.GenerativeModel(self.model_name) # type: ignore

    def generate(self, prompt: str) -> str:
        # 【修改重點 2】改用 generate_content
        response = self.model.generate_content(prompt) # type: ignore
        
        try:
            text = response.text
        except ValueError:
            raise RuntimeError(f"Gemini 回應無法解析 (可能觸發了安全過濾): {response.prompt_feedback}")
            
        if not text:
            raise RuntimeError("Gemini 回應內容為 None，無法解析成字串")
            
        assert isinstance(text, str)
        return text


class BreezeModelClient(BaseModelClient):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        if torch is None:
            raise RuntimeError("未安裝 torch，無法使用 Breeze 模型")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise RuntimeError("未安裝 transformers，無法使用 Breeze 模型")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def generate(self, prompt: str, temperature: float = 0.5, max_new_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return decoded.strip()


class LlamaModelClient(BaseModelClient):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        if torch is None:
            raise RuntimeError("未安裝 torch，無法使用 Llama 模型")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise RuntimeError("未安裝 transformers，無法使用 Llama 模型")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def generate(self, prompt: str, temperature: float = 0.5, max_new_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return decoded.strip()


class GroqModelClient(BaseModelClient):
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name)
        if not HAS_GROQ or Groq is None:
            raise RuntimeError("未安裝 groq，無法使用 Groq 模型")

        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY 尚未設定")

        self.client = Groq(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 100) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            text = response.choices[0].message.content
            if text is None:
                raise RuntimeError("Groq 回應內容為 None")
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Groq 生成失敗: {e}")


class LocalModelClient(BaseModelClient):
    """示意性的離線模型介面。

    實際的實作可以使用 transformers、llama.cpp、qwen 源碼等。
    這裡僅示範結構，使用者可在 __init__ 中載入權重、設定管道。
    """

    def __init__(self, model_name: str, model_path: Optional[str] = None):
        super().__init__(model_name)
        # 例如使用 transformers 的 pipeline
        try:
            from transformers import pipeline
        except ImportError:
            raise RuntimeError("未安裝 transformers，請安裝才能使用離線模型")

        # 根據 model_name 決定後端，這裡假設 model_path 是 HuggingFace 模型路徑
        self.pipe = pipeline("text-generation", model=model_path or model_name)

    def generate(self, prompt: str) -> str:
        # pipeline 會回傳 list[dict]，取第一個的 "generated_text"
        out = self.pipe(prompt, max_length=1024)[0]["generated_text"]
        return out


def get_model_client(model_name: str, **kwargs) -> BaseModelClient:
    """根據名稱回傳適當的客戶端。

    - "gpt", "gpt-4o-mini" 等被視為 OpenAI 線上模型
    - "gemini" 前綴/關鍵字將使用 GeminiModelClient
    - "breeze" 預設使用 MediaTek-Research/Breeze-7B-Instruct-v1_0
    - "llama" 預設使用 meta-llama/Meta-Llama-3-8B-Instruct
    - "groq" 預設使用 llama-3.3-70b-versatile
    - 其他本機模型可由 LocalModelClient 處理

    **kwargs 會傳給建構子，例如指定 model_path、api_key。
    """
    lower = model_name.lower()
    
    # 設定預設模型名稱
    if lower == "breeze":
        model_name = "MediaTek-Research/Breeze-7B-Instruct-v1_0"
    elif lower == "llama":
        model_name = "meta-llama/Llama-3.1-70B"
    elif lower == "groq":
        model_name = "llama-3.3-70b-versatile"
    
    if lower.startswith("gpt") or lower.startswith("oai"):
        return OpenAIModelClient(model_name, **kwargs)
    elif "gemini" in lower:
        return GeminiModelClient(model_name, **kwargs)
    elif "breeze" in lower:
        return BreezeModelClient(model_name, **kwargs)
    elif "llama" in lower and "groq" not in lower:
        return LlamaModelClient(model_name, **kwargs)
    elif "groq" in lower:
        return GroqModelClient(model_name, **kwargs)
    else:
        # 預設使用本機模型（例如可由 transformers pipeline 支援的模型）
        return LocalModelClient(model_name, **kwargs)
