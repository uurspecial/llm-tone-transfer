import gradio as gr
import os
import json
import datetime
from dotenv import load_dotenv

# 正確的匯入
from generator.generator import GroqLlama318B
from generator.scorer import Scorer

load_dotenv(dotenv_path="generator/.env")

# 建立 logs 目錄
os.makedirs("logs", exist_ok=True)

# 初始化
print("啟動系統中，正在載入模型...")
generator = GroqLlama318B()
scorer = Scorer(dataset_path="src/data/dataset_scorer.jsonl")
print("✅ 系統載入完成！")

# 後台記錄函數
def log_user_activity(input_text, target_level, output_text, predicted_score, success):
    """記錄使用者活動到 JSON 檔案"""
    timestamp = datetime.datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "input_text": input_text,
        "target_level": target_level,
        "output_text": output_text,
        "predicted_score": predicted_score,
        "success": success,
        "ip_address": "localhost"  # 在實際部署時可以取得真實 IP
    }
    
    # 每天一個日誌檔案
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = f"logs/user_activity_{date_str}.jsonl"
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"📝 已記錄使用者活動: {timestamp}")
    except Exception as e:
        print(f"❌ 記錄失敗: {e}")

# 核心邏輯
def process_tone(input_text, target_level):
    if not input_text.strip():
        return "請輸入文字！", ""
    
    level_int = int(target_level.split("-")[0])
    
    # 直接呼叫 generator.generate()
    output_text = generator.generate(
        text=input_text,
        level=level_int,
    )
    
    # 呼叫 scorer.score()
    predicted_score = scorer.score(output_text)
    
    success = (predicted_score == level_int)
    status = "✅ 轉換成功" if success else "❌ 轉換失敗"
    eval_result = f"裁判給分：Level {predicted_score}\n狀態：{status}"
    
    # 記錄使用者活動
    log_user_activity(input_text, target_level, output_text, predicted_score, success)
    
    return output_text, eval_result

# 後台管理函數
def get_recent_logs(days=1):
    """取得最近幾天的使用記錄"""
    logs = []
    for i in range(days):
        date = datetime.datetime.now() - datetime.timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        log_file = f"logs/user_activity_{date_str}.jsonl"
        
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line))
            except Exception as e:
                print(f"讀取日誌檔案錯誤: {e}")
    
    # 按時間排序，最新的在前
    logs.sort(key=lambda x: x["timestamp"], reverse=True)
    return logs[:50]  # 只顯示最近50筆

def format_logs_for_display(logs):
    """將日誌格式化為易讀的文字"""
    if not logs:
        return "📭 目前沒有使用記錄"
    
    result = f"📊 最近使用記錄 (共 {len(logs)} 筆)\n\n"
    
    for log in logs:
        timestamp = log["timestamp"][:19]  # 只顯示到秒
        success_icon = "✅" if log["success"] else "❌"
        
        result += f"{success_icon} {timestamp}\n"
        result += f"輸入: {log['input_text'][:30]}{'...' if len(log['input_text']) > 30 else ''}\n"
        result += f"目標: {log['target_level']} | 結果: Level {log['predicted_score']}\n"
        result += f"輸出: {log['output_text'][:50]}{'...' if len(log['output_text']) > 50 else ''}\n"
        result += "---\n"
    
    return result

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 LLM 可控語氣轉換系統")
    
    with gr.Tabs():
        # 主頁籤 - 語氣轉換
        with gr.TabItem("🎯 語氣轉換"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_text = gr.Textbox(label="💬 輸入原始句子", lines=3)
                    target_level = gr.Radio(
                        choices=["1-溫和", "2-中性", "3-不滿", "4-酸"],
                        value="4-酸",
                        label="🎯 選擇目標語氣"
                    )
                    submit_btn = gr.Button("🚀 開始轉換", variant="primary")
                
                with gr.Column(scale=1):
                    output_text = gr.Textbox(label="✨ 生成結果", lines=3, interactive=False)
                    eval_result = gr.Textbox(label="⚖️ 自動裁判評估", lines=2, interactive=False)
        
        # 後台管理籤
        with gr.TabItem("📊 使用記錄"):
            gr.Markdown("### 後台管理 - 使用者活動記錄")
            refresh_btn = gr.Button("🔄 重新整理記錄", variant="secondary")
            logs_display = gr.Textbox(
                label="最近使用記錄",
                lines=20,
                interactive=False,
                value=format_logs_for_display(get_recent_logs())
            )
    
    # 綁定按鈕事件
    submit_btn.click(
        fn=process_tone,
        inputs=[input_text, target_level],
        outputs=[output_text, eval_result]
    )
    
    refresh_btn.click(
        fn=lambda: format_logs_for_display(get_recent_logs()),
        inputs=[],
        outputs=[logs_display]
    )

if __name__ == "__main__":
    demo.launch(share=True)