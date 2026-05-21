from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
import torch
from tqdm import tqdm
import os

# ==========================================
# 1. 硬體與量化設定
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 啟動 4-bit 量化魔法
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "google/gemma-4-31B-it"
print(f"📦 正在載入 {model_name} (啟用 4-bit 量化)...")

# 載入模型 (加入 device_map="auto" 讓系統自己管記憶體，並套用量化)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",  
    quantization_config=quantization_config 
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def create_final_prompt(new_sentence):
    prompt = f"""User:你是一個語氣評分機器。你的唯一任務是閱讀最後的「目標句子」，並判斷它的語氣分數。
    警告：你「只能」輸出一個純數字（1、2、3 或 4），絕對不可以輸出任何解釋、標籤，也不准造句！

    【評分標準】
    1-溫和（Polite / Warm）：語氣柔軟，常使用「不好意思、謝謝、麻煩你、喔、吧」等語助詞，目的是維護關係或展現禮貌。
    2-中性（Neutral / Factual）：像機器人或新聞報導一樣陳述事實。不帶個人情緒，沒有明顯的語助詞，僅傳遞資訊。
    3-不滿（Direct Anger / Complaint）：情緒直接外露。直接表達憤怒、指責、命令或抱怨。特徵是「直球對決」，不拐彎抹角，沒有幽默感。（例如：閉嘴、你很煩、爛透了）。
    4-酸（Sarcastic / Mocking）：陰陽怪氣、高級反諷。使用「誇獎的形式來貶低」或「誇飾的比喻」。特徵是帶有幽默感、嘲諷、挖苦，比直接罵更刺耳。（例如：你的智商真是人類奇蹟）。

    【評分範例】
    目標句子：我注意到你最近氣色變好了。
    評分：1

    目標句子：你是不是變胖了。
    評分：2

    目標句子：你真的變胖了，尤其是你的下巴，不好看。
    評分：3

    目標句子：你最近看起來更有份量了，看得出生活過得很滋潤。
    評分：4

    【現在請執行任務】
    目標句子：{new_sentence}
    分數：
    Assistant: """
    return prompt

def score_sentence(new_sentence):
    prompt = create_final_prompt(new_sentence)
    text_input = tokenizer(prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(
        **text_input, 
        max_new_tokens=8,
        do_sample=False,
        repetition_penalty=1.2
    ) 
    
    input_length = text_input['input_ids'].shape[1]
    response_text = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
    
    return response_text

# ==========================================
# 3. 執行測試迴圈
# ==========================================
print("\n🚀 開始進行全面測試（目標 641 筆），使用 CUDA 加速推論中...")

correct_count = 0
total_count = 0

labels = ['1', '2', '3', '4', '解析失敗']
confusion_matrix = {true_score: {pred_score: 0 for pred_score in labels} for true_score in ['1', '2', '3', '4']}

# 確保 local 資料夾存在
os.makedirs('local', exist_ok=True)

with open('src/data/dataset.jsonl', 'r', encoding='utf-8') as f, \
     open('local/dataset_results4.jsonl', 'w', encoding='utf-8') as out_jsonl:
    
    pbar = tqdm(f, total=641, desc="評分進度", unit="筆", colour="green")
    
    for line in pbar:
        data = json.loads(line)
        target_text = data['text']
        ground_truth = str(data['score']) 
        
        raw_model_output = score_sentence(target_text)
        
        match = re.search(r'[1-4]', raw_model_output)
        clean_score = match.group(0) if match else "解析失敗"
        
        if ground_truth in confusion_matrix:
            confusion_matrix[ground_truth][clean_score] += 1
        
        is_correct = (clean_score == ground_truth)
        if is_correct:
            correct_count += 1
            
        total_count += 1
        
        current_acc = (correct_count / total_count) * 100
        pbar.set_postfix({
            "正確率": f"{current_acc:.1f}%", 
            "最新結果": f"{'✅' if is_correct else '❌'}"
        })
        
        data['model_score'] = clean_score
        data['is_correct'] = is_correct
        out_jsonl.write(json.dumps(data, ensure_ascii=False) + '\n')
        out_jsonl.flush()

# ==========================================
# 4. 結算成績與報告
# ==========================================
if total_count > 0:
    accuracy = (correct_count / total_count) * 100
    
    report = "\n" + "="*50 + "\n"
    report += "語氣評分模型 測試結果報告\n"
    report += "="*50 + "\n"
    report += f"總共測試：{total_count} 筆\n"
    report += f"答對題數：{correct_count} 筆\n"
    report += f"整體正確率 (Accuracy)：{accuracy:.2f}%\n\n"
    
    report += "混淆矩陣 (Confusion Matrix):\n"
    report += "          | 預測 1 | 預測 2 | 預測 3 | 預測 4 | 解析失敗 |\n"
    report += "-"*56 + "\n"
    
    for true_label in ['1', '2', '3', '4']:
        row = confusion_matrix[true_label]
        report += f" 真實 {true_label} 分 |"
        for pred_label in labels:
            report += f" {row[pred_label]:>6} |"
        report += "\n"
    
    print(report)
    
    with open('local/output4.txt', 'w', encoding='utf-8') as out_txt:
        out_txt.write(report)
        
    print("處理完畢！請查看你左側的檔案總管：")
    print(" 評估報告： local/output4.txt")
    print(" 詳細預測資料： local/dataset_results4.jsonl")