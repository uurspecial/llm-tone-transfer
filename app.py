import gradio as gr
import os
from dotenv import load_dotenv

# 正確的匯入
from generator.generator import GroqLlama318B
from generator.scorer import Scorer

load_dotenv(dotenv_path="generator/.env")

# 初始化
print("啟動系統中，正在載入模型...")
generator = GroqLlama318B()
scorer = Scorer(dataset_path="src/data/dataset_scorer.jsonl")
print("✅ 系統載入完成！")

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
    
    status = "✅ 轉換成功" if predicted_score == level_int else "❌ 轉換失敗"
    eval_result = f"裁判給分：Level {predicted_score}\n狀態：{status}"
    
    return output_text, eval_result

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 LLM 可控語氣轉換系統")
    
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
    
    submit_btn.click(
        fn=process_tone,
        inputs=[input_text, target_level],
        outputs=[output_text, eval_result]
    )

if __name__ == "__main__":
    demo.launch(share=True)