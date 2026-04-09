"""
統一管理四個版本的語氣評分 Prompt

版本 1：基礎版本 (原始 score_sentences.py)
版本 2：排他性邊界版本（【必須】軟化詞彙）
版本 3：改進版本（【通常】軟化詞彙 + 警告條款）
版本 4：重新定義 1 ＆ 2 的邊界 (從版本1修改)
"""

from typing import Optional,Any

FEW_SHOT_DATASET = [
    {
        "original": "你怎麼還沒來",
        1: "我有點擔心你，路上還順利嗎？慢慢來沒關係。",
        2: "路上塞車嗎，你已經遲到了喔。",
        3: "時間都過多久了，你卻還沒出現，大家都在等你你知道嗎?",
        4: "你的時間安排真的很自由，大家配合你就好。",
    },
    {
        "original": "為什麼不回我訊息",
        1: "剛剛沒看到你的回覆，我有點關心你是不是在忙。",
        2: "你還沒回我訊息，在幹嘛。",
        3: "訊息放著不回，我等很久了，你到底在做甚麼。",
        4: "原來回訊息也要看心情，長知識了，呵呵。",
    },
    {
        "original": "你報告做完了嗎",
        1: "想關心一下你那份報告進行得怎麼樣，需要我陪你一起看看嗎？",
        2: "那份報告你現在做到哪了?",
        3: "報告拖成這樣，你如果不想做可以直接說。",
        4: "你這進度控制得真穩定，幾乎沒有變化。",
    },
    {
        "original": "你今天心情不好嗎",
        1: "我感覺你今天情緒有點不好，如果想要聊天，我在這裡。",
        2: "你今天看起來心情很差欸。",
        3: "你在不爽甚麼，莫名其妙就要挨你一頓氣。",
        4: "你今天情緒表現得很完整，大家都感受到了。",
    },
    {
        "original": "你剛剛說甚麼",
        1: "剛剛那段我沒聽清楚，可以請你再說一次嗎？",
        2: "我剛沒聽清楚，你再說一次。",
        3: "你講話又快小聲我真的聽不到，可不可以好好再說一次阿。",
        4: "你剛剛那段說明很有深度，沒人聽得懂呢。",
    },
    {
        "original": "你衣服穿反了",
        1: "我剛剛好像有看到你衣服怪怪的，是不是穿反了？",
        2: "你衣服穿反了吧。",
        3: "你出門都沒檢查衣服有沒有穿反，不要出來丟人現眼。",
        4: "穿衣服都能穿反，令堂沒教過你嗎？",
    },
    {
        "original": "這個便當有點冷掉了",
        1: "這個便當摸起來有點涼，要不要我幫你熱一下？",
        2: "這個便當已經涼掉了，吃之前記得要加熱。",
        3: "這個便當都已經涼掉了還放在這裡是怎麼回事。",
        4: "這便當真不錯，直接走冷食路線。",
    },
    {
        "original": "你最近是不是胖了",
        1: "我注意到你最近氣色變好了。",
        2: "你是不是變胖了。",
        3: "你真的變胖了，尤其是你的下巴，不好看。",
        4: "你最近看起來更有份量了，看得出生活過得很滋潤。",
    },
    {
        "original": "你化的妝好難看",
        1: "你今天的妝好特別呀!感覺很用心喔。",
        2: "你今天的妝跟平常感覺不太一樣，是在嘗試新的風格嗎？",
        3: "你是在化妝還是補土，我看的真的很難受。",
        4: "你在COSPLAY鐘樓怪人嗎，可以原地出道了!。",
    },
    {
        "original": "你可以幫我個忙嗎",
        1: "哥哥現在有空嗎？我想找你幫個忙～～",
        2: "有空嗎?我需要你幫我處理一件事。",
        3: "你給我過來，現在去把這件事做好。",
        4: "哎唷，大忙人，不知道有沒有那個榮幸請您動動金手，幫我一個忙啊？",
    }
]

# ==========================================
# 版本 1: 基礎版本
# ==========================================
V1_SCALE_DEFINITION = """
請根據下面的語氣標準，給每一句話打分數（1～4）：

1-溫和（Polite / Warm）：語氣柔軟，常使用「不好意思、謝謝、麻煩你、喔、吧」等語助詞，目的是維護關係或展現禮貌。
2-中性（Neutral / Factual）：像機器人或新聞報導一樣陳述事實。不帶個人情緒，沒有明顯的語助詞，僅傳遞資訊。
3-不滿（Direct Anger / Complaint）：情緒直接外露。直接表達憤怒、指責、命令或抱怨。特徵是「直球對決」，不拐彎抹角，沒有幽默感。（例如：閉嘴、你很煩、爛透了）。
4-酸（Sarcastic / Mocking）：陰陽怪氣、高級反諷。使用「誇獎的形式來貶低」或「誇飾的比喻」。特徵是帶有幽默感、嘲諷、挖苦，比直接罵更刺耳。（例如：你的智商真是人類奇蹟）。
"""

# ==========================================
# 版本 2: 排他性邊界版本（【必須】軟化詞彙）
# ==========================================
V2_SCALE_DEFINITION = """
請根據以下語氣標準，對句子進行 1-4 評分。

【等級定義與排他邊界】
1-溫和（Polite / Warm）：有同理心，【必須】包含軟化語氣的詞彙（如：不好意思、麻煩了、喔、吧、請）。如果只是陳述事實但沒有軟化詞，請判為 2。
2-中性（Neutral / Factual）：如機器人般陳述事實。特徵是「乾脆、沒有情緒字眼、沒有明顯語助詞」。
3-不滿（Direct Anger / Complaint）：直接的負面情緒宣洩，直接指責、命令或抱怨，沒有幽默感。（例如：閉嘴、你很煩、爛透了）。
4-酸（Sarcastic / Mocking）：高級反諷、陰陽怪氣。特徵是「表面字義是正面的（如誇獎、關心），但實際用意是貶低或嘲笑」，或者使用了誇飾的比喻。如果是直接無腦罵人，請判為 3。

"""

# ==========================================
# 版本 3: 改進版本（【通常】軟化詞彙 + 【警告】條款）
# ==========================================
V3_SCALE_DEFINITION = """
請根據以下語氣標準，對句子進行 1-4 評分。

【等級定義與排他邊界】
1-溫和（Polite / Warm）：有同理心，【通常】包含軟化語氣的詞彙，或者表達出關心、幫忙的善意（如：不好意思、麻煩了、喔、吧、請）。如果只是陳述事實但沒有軟化詞，請判為 2。
2-中性（Neutral / Factual）：如機器人般陳述事實。特徵是「乾脆、沒有情緒字眼、沒有明顯語助詞」。
3-不滿（Direct Anger / Complaint）：直接的負面情緒宣洩，直接指責、命令或抱怨，沒有幽默感。（例如：閉嘴、你很煩、爛透了）。
4-酸（Sarcastic / Mocking）：高級反諷、陰陽怪氣。特徵是「表面字義是正面的（如誇獎、關心），但實際用意是貶低或嘲笑」，或者使用了誇飾的比喻。如果是直接無腦罵人，請判為 3。【警告】除非句子中有明顯的邏輯矛盾或誇飾，否則請勿將單純的禮貌與關心過度解讀為反諷 (L4)。

"""
# ==========================================
# 版本 4: 重新定義 1 ＆ 2 的邊界 (從版本1修改)
# ==========================================
V4_SCALE_DEFINITION = """
請根據下面的語氣標準，給每一句話打分數（1～4）：

1-溫和（Polite / Warm / Empathetic）：
語氣柔軟，目的是安撫對方或刻意展現善意。特徵是帶有**【主動的關懷、明顯的歉意、或積極的感謝】**。如果只是日常溝通的順口禮貌，請判為 2。（例如：沒關係你慢慢來、真的非常謝謝你）。

2-中性（Neutral / Factual / Routine）：
日常事務的陳述、提議、或行程告知。語氣平穩，專注於解決事情或傳遞資訊。【注意】句子中可能帶有日常對話的語氣詞（如：看看、稍微、要不要、先把），只要它沒有強烈的情感投入或刻意的關懷，都屬於 2。

3-不滿（Direct Anger / Complaint）：情緒直接外露。直接表達憤怒、指責、命令或抱怨。特徵是「直球對決」，不拐彎抹角，沒有幽默感。（例如：閉嘴、你很煩、爛透了）。

4-酸（Sarcastic / Mocking）：陰陽怪氣、高級反諷。使用「誇獎的形式來貶低」或「誇飾的比喻」。特徵是帶有幽默感、嘲諷、挖苦，比直接罵更刺耳。（例如：你的智商真是人類奇蹟）。
"""

HARD_NEGATIVES = """
【特別注意的易混淆範例】
「我剛睡醒，頭腦還沒完全清醒。」
分析：沒有語氣助詞，只是單純陳述生理狀態，沒有安撫對方。
【最終分數：2】

「我剛醒，腦袋有點亂，慢慢來沒關係啦。」
分析：有「沒關係啦」安撫對方，具備同理心與軟化詞。
【最終分數：1】

「你到底在吵什麼？還不快滾開！」
分析：直接的負面情緒與命令，沒有任何幽默或反諷包裝。
【最終分數：3】

「你這進度控制得真穩定，幾乎沒有變化。」
分析：表面誇獎「穩定」，實際在抱怨「沒進度」，屬於誇獎形式的貶低。
【最終分數：4】
"""

# ==========================================
# 提示詞組合函式
# ==========================================
def get_prompt_version(version: int = 1, scale_definition: Optional[str] = None) -> str:
    """
    根據版本號返回對應的 scale definition，或使用直接提供的 scale_definition

    Args:
        version: 版本號 (1, 2, 3，或 4)，預設 1
        scale_definition: 直接提供 scale definition 的內容

    Returns:
        scale_definition 字串

    Raises:
        ValueError: 非法版本號
    """
    if scale_definition is not None:
        return scale_definition

    if version == 1:
        return V1_SCALE_DEFINITION
    elif version == 2:
        return V2_SCALE_DEFINITION
    elif version == 3:
        return V3_SCALE_DEFINITION
    elif version == 4:
        return V4_SCALE_DEFINITION
    else:
        raise ValueError(f"❌ 不支援的版本號: {version}。請使用 1、2、3 或 4。")


def build_few_shot_prompt(
        version: int = 1,
        few_shot_dataset: Any = None,  # 加上 Any 型別標註
        include_hard_negatives: bool = False, 
        scale_definition: Optional[str] = None
    ) -> str:
    """
    建立完整的 Few-Shot 提示

    試用條件：
    - prompt 必須有 few_shot_examples（由 FEW_SHOT_DATASET 提供）
    - 可選擇 scale definition 版本
    - HARD_NEGATIVES 預設不加，若 include_hard_negatives=True 才加

    Args:
        version: 版本號 (1,2,3,4)
        few_shot_dataset: 自訂 few shot examples, 預設是 FEW_SHOT_DATASET
        include_hard_negatives: 是否額外包含 hard negatives

    Returns:
        完整的 prompt 字串
    """
    scale_def = get_prompt_version(version=version, scale_definition=scale_definition)

    if few_shot_dataset is None:
        few_shot_dataset = FEW_SHOT_DATASET

    if hasattr(few_shot_dataset, 'empty') and few_shot_dataset.empty:
        raise ValueError("few_shot_dataset 必須提供含至少一個項目的列表或 DataFrame")

    prompt = scale_def.strip() + "\n\n"

    if include_hard_negatives:
        prompt += HARD_NEGATIVES.strip() + "\n\n"

    prompt += "【Few-shot 範例】：\n"

    if hasattr(few_shot_dataset, 'iterrows'):
        # DataFrame input
        for _, row in few_shot_dataset.iterrows():
            text = row.get('text') or row.get('original') or row.get('sentence') or ""
            score = row.get('score', "")
            prompt += f"「{text}」 → {score}\n"
    elif isinstance(few_shot_dataset, list):
        # List of dictionary from FEW_SHOT_DATASET format
        for example in few_shot_dataset:
            prompt += f"原句：{example.get('original', '')}\n"
            for level in (1, 2, 3, 4):
                val = example.get(level, "")
                prompt += f"{level}: {val}\n"
            prompt += "\n"
    else:
        raise ValueError("few_shot_dataset 必須提供 list 或 DataFrame 格式")

    return prompt


# ==========================================
# 版本資訊
# ==========================================
AVAILABLE_VERSIONS = {
    1: "基礎版本 (原始 score_sentences.py)",
    2: "排他性邊界版本（【必須】軟化詞彙）",
    3: "改進版本（【通常】軟化詞彙 + 警告條款）",
    4: "重新定義 1 ＆ 2 的邊界 (從版本1修改)",
}
