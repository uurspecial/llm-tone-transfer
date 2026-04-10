"""
統一的語氣評分程式
支持透過 argparse 選擇三個 prompt 版本進行評分

使用方式：
  python scorer.py -v 1          # 使用版本 1
  python scorer.py --version 2   # 使用版本 2
  python scorer.py -v 3          # 使用版本 3（預設）
  python scorer.py -v 4          # 使用版本 4（重新定義 1 ＆ 2 的邊界）
  python scorer.py -v 1 -m breeze # 使用 Breeze 模型
"""

import argparse
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
from dotenv import load_dotenv
import os
import time
from model_clients import get_model_client
from scorer_prompt import get_prompt_version, build_few_shot_prompt, AVAILABLE_VERSIONS

# === 初始化區塊 ===
load_dotenv()

# client 由 get_model_client 建立
client = None


# === 資料處理函式 ===
def load_dataset_from_jsonl(file_path):
    """從 JSON Lines 檔案中載入資料集並轉換為 DataFrame。"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        df = pd.DataFrame(data)
        df['score'] = df['score'].astype(int)
        return df
    except Exception as e:
        print(f"❌ 錯誤：載入或處理 {file_path} 時出錯: {e}")
        return None


def prepare_data(dataset_path="dataset_scorer.jsonl"):
    """準備訓練集和測試集"""
    df_all = load_dataset_from_jsonl(dataset_path)
    
    df_few_shot_examples = None
    df_test = None
    
    if df_all is not None:
        # 執行等比例 (Stratified) 訓練/測試分割
        if min(df_all['score'].value_counts()) < 2:
            print("⚠️ 警告：某些類別的樣本數過少，無法執行分層抽樣，將使用一般隨機抽樣。")
            stratify_param = None
        else:
            stratify_param = df_all['score']
        
        df_few_shot_examples, df_test = train_test_split(
            df_all,
            test_size=0.2,
            random_state=42,
            stratify=stratify_param
        )
        
        print("\n📊 資料集分割結果:")
        print(f"資料集大小: {len(df_few_shot_examples)}")
        print(df_few_shot_examples['score'].value_counts().sort_index())
        print("-" * 20)
        print(f"測試集大小: {len(df_test)}")
        print(df_test['score'].value_counts().sort_index())
    
    return df_few_shot_examples, df_test


# === 評分函式 ===
progress_counter = 0
total_sentences = 0


def score_sentence(sentence: str, few_shot_prompt: str) -> int:
    """呼叫 LLM，對一句話打分數，包含自動 Retry 機制與強化版解析邏輯。"""
    global client, progress_counter, total_sentences
    
    progress_counter += 1
    
    # 稍微調整指令，有些模型看到「絕對不要」反而會完全不輸出任何字
    prompt = few_shot_prompt + f"\n請對下面句子進行分析與打分：\n「{sentence}」\n【重要指令】：請直接回答一個數字（1, 2, 3 或 4），不要輸出任何解釋或標點符號。"
    
    max_retries = 5
    base_delay = 2
    
    if client is None:
        raise RuntimeError("模型客戶端尚未初始化，請先呼叫 get_model_client()")

    for attempt in range(max_retries):
        try:
            if attempt == 0:
                print(f"[{progress_counter}/{total_sentences}] 正在評分: 「{sentence[:15]}...」", end="")
            else:
                print(f" (重試 {attempt}/{max_retries})...", end="")

            raw_score_content = client.generate(prompt)
            
            # 💡 修正點：如果回應為空，主動拋出例外 (Exception)，強迫程式進入下方的 except 區塊進行等待與重試
            if not raw_score_content or str(raw_score_content).strip() == "":
                raise ValueError("API 回應為空 (Empty Response)")
            
            # 優先尋找標準格式 【最終分數：1-4】
            match = re.search(r'【最終分數：([1-4])】', raw_score_content)
            
            if match:
                score = int(match.group(1))
                print(f" -> ✅ 分數: {score}")
            else:
                # 備用方案：只抓取 1, 2, 3, 4 這四個合法數字
                valid_numbers = re.findall(r'[1-4]', raw_score_content)
                if valid_numbers:
                    # 抓取出現的第一個合法數字
                    score = int(valid_numbers[0])
                    snippet = raw_score_content.replace("\n", " ")[:25]
                    print(f" -> ⚠️ 備用解析分數: {score} (回應: {snippet}...)")
                else:
                    score = -1
                    print(f" -> ❌ 無法解析 (回應: {raw_score_content.strip()[:30]}...)")
            
            # 成功抓到分數後，稍微暫停避免打 API 太快
            time.sleep(1)
            return score
        
        except Exception as e:
            error_str = str(e).lower()
            # 將 Empty Response 也加入等待邏輯
            if "empty" in error_str or "rate_limit" in error_str or "429" in error_str or "quota" in error_str or "exhausted" in error_str:
                wait_time = base_delay * (2 ** attempt)
                print(f"\n⚠️ API 限制或異常 (錯誤: {e})，等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
            else:
                print(f"\n⚠️ API 未知錯誤: {e}")
                time.sleep(base_delay) # 就算未知錯誤也稍微等一下
    
    print(f"\n❌ 重試次數耗盡。")
    return -3


# === 主程式 ===
def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(
        description="語氣評分程式 - 支持三個 prompt 版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用的版本：
  1 - 基礎版本 (原始 score_sentences.py)
  2 - 排他性邊界版本（【必須】軟化詞彙）
  3 - 改進版本（【通常】軟化詞彙 + 警告條款）
  4 - 重新定義 1 ＆ 2 的邊界 (從版本1修改)

範例使用：
  python scorer.py -v 1
  python scorer.py --version 2
  python scorer.py -v 3
        """
    )
    parser.add_argument(
        "-v", "--version",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="選擇 prompt 版本 (預設: 1)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gpt-4o-mini",
        help="選擇模型（例如 gpt-4o-mini, gemini-1.0, llama, breeze）"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="dataset_scorer.jsonl",
        help="資料集檔案路徑 (預設: dataset_scorer.jsonl)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="CSV 檔案路徑 (格式: text,target,output,model_name,prompt_type)，評分 output 欄位"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f"🚀 開始評分程式")
    print(f"使用版本: {args.version}")
    print(f"版本描述: {AVAILABLE_VERSIONS[args.version]}")
    print(f"模型: {args.model}")
    if args.csv:
        print(f"CSV 檔案: {args.csv}")
    else:
        print(f"資料集: {args.dataset}")
    print(f"{'='*50}\n")
    
    global progress_counter, total_sentences
    progress_counter = 0
    
    # 準備資料
    if args.csv:
        # 讀取 CSV
        df_test = pd.read_csv(args.csv, encoding="utf-8")
        if 'output' not in df_test.columns or 'target' not in df_test.columns:
            print("❌ CSV 必須包含 'output' 和 'target' 欄位")
            return
        df_test['score'] = df_test['target'].astype(int)
        df_test['text'] = df_test['output']
        total_sentences = len(df_test)
        # 仍需從 dataset 建立 few_shot
        df_few_shot_examples, _ = prepare_data(args.dataset)
        if df_few_shot_examples is None:
            print("\n🚨 Few-shot 資料載入失敗，程式終止。")
            return
    else:
        df_few_shot_examples, df_test = prepare_data(args.dataset)
        if df_few_shot_examples is None or df_test is None:
            print("\n🚨 資料載入或分割失敗，程式終止。")
            return
        total_sentences = len(df_test)
    
    # 建立 Few-Shot 提示
    few_shot_prompt = build_few_shot_prompt(args.version, df_few_shot_examples)
    
    # 建立模型客戶端（會讀 OpenAI/GEMINI 密鑰或其他參數）
    global client
    client = get_model_client(args.model)

    # 開始評分
    total_sentences = len(df_test)
    print(f"\n🚀 開始對測試集 ({total_sentences} 句) 進行評分...")
    
    df_test["predicted_score"] = df_test["text"].apply(
        lambda x: score_sentence(x, few_shot_prompt)
    )
    
    print("\n✅ 評分完成。")
    
    # 設定輸出路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    if args.csv:
        base_name = os.path.splitext(os.path.basename(args.csv))[0]
        model_dir = os.path.join(results_dir, base_name)
    else:
        model_dir = os.path.join(results_dir, args.model.replace('/', '_'))
    os.makedirs(model_dir, exist_ok=True)
    
    # 檔案名稱本資訊包含版
    if args.csv:
        base_name = os.path.splitext(os.path.basename(args.csv))[0]
        output_filename = f"{base_name}_scored_{int(time.time())}.csv"
    else:
        output_filename = f"v{args.version}_scored_{int(time.time())}.csv"
    output_path = os.path.join(model_dir, output_filename)
    df_test.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 過濾掉 API 錯誤的資料來計算準確率
    valid_df = df_test[df_test["predicted_score"] > 0]
    
    if len(valid_df) == 0:
        print("\n❌ 沒有有效的預測結果。")
        return
    
    accuracy = (valid_df['score'] == valid_df['predicted_score']).mean() * 100
    
    # 同時寫入設定紀錄檔（同版本 prompt + model + key info + 準確度 + 混淆矩陣）
    if args.csv:
        config_filename = f"{base_name}_config.txt"
    else:
        config_filename = f"v{args.version}_config.txt"
    config_path = os.path.join(model_dir, config_filename)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("模型設定\n")
        f.write(f"模型: {args.model}\n")
        f.write(f"prompt 版本: {args.version}\n")
        f.write(f"版本描述: {AVAILABLE_VERSIONS[args.version]}\n")
        if args.csv:
            f.write(f"評分 CSV: {args.csv}\n")
        else:
            f.write(f"資料集: {args.dataset}\n")
        f.write("使用者輸入: 多模組化，來源由 model_clients.get_model_client()\n")
        f.write("API key 來源: OPENAI_API_KEY / GEMINI_API_KEY/GOOGLE_API_KEY (環境變數)\n")
        f.write("-----------------------------\n")
        f.write("few_shot_prompt 範例 (前 1000 字):\n")
        f.write(build_few_shot_prompt(args.version, df_few_shot_examples)[:1000])
        f.write("\n-----------------------------\n")
        f.write(f"預測準確度: {accuracy:.2f}%\n")
        f.write("混淆矩陣 (Confusion Matrix):\n")
        f.write("(列/Row: 真實分數, 欄/Col: 預測分數)\n")
        f.write(str(confusion_matrix(valid_df['score'], valid_df['predicted_score'])) + "\n")
    
    print("\n" + "=" * 50)
    print(f"📊 詳細分類報告 (版本 {args.version})")
    print("=" * 50)
    print(f"有效評分數: {len(valid_df)}/{len(df_test)}")
    print(f"預測準確度: {accuracy:.2f}%\n")
    
    print(classification_report(
        valid_df['score'],
        valid_df['predicted_score'],
        target_names=['L1 溫和', 'L2 中性', 'L3 不滿', 'L4 酸'],
        zero_division=0
    ))
    
    print("\n📉 混淆矩陣 (Confusion Matrix):")
    print("(列/Row: 真實分數, 欄/Col: 預測分數)")
    print(confusion_matrix(valid_df['score'], valid_df['predicted_score']))
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()