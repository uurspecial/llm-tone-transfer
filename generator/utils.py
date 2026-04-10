# utils.py
import os
import sys
import io
from dotenv import load_dotenv

# 強制輸出編碼 (解決終端機中文亂碼)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def setup_env():
    """載入環境變數"""
    load_dotenv()
    # 這裡依照你的截圖，改抓 GOOGLE_API_KEY
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ 錯誤：找不到 GOOGLE_API_KEY，請檢查 .env 檔案")
        sys.exit(1)
    return api_key