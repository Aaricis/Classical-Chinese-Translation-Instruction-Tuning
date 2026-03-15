from transformers import BitsAndBytesConfig
import torch


# def get_prompt(instruction: str) -> str:
#     '''Format the instruction as a prompt for LLM.'''
#     return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"

# Zero-Shot
# def get_prompt(instruction: str) -> str:
#     return (f"你是一位精通中國古代文學與語言學的專家，擅長文言文與現代漢語之間的互譯。\
#             請遵循以下原則：\
#                 1. 準確理解用戶指令中的翻譯方向（現代轉文言，或文言轉現代）。\
#                 2. 譯文需簡練、準確，符合對應時代的語言習慣（文言文需古雅，現代文需通順）。\
#                 3. 直接輸出翻譯結果，不要添加「答案：」、「譯文：」等多餘前綴，也不要輸出解釋性文字。\
#                 4. 若原文涉及歷史人名、官職，請保留或根據常識還原，確保語義通順。\
#             {instruction}\
#             請直接輸出翻譯結果："
#             )

# def get_prompt(instruction: str) -> str:
#     return f"""
#     你是一位精通古代漢語與現代漢語的語言學專家。請準確翻譯下列文本。只輸出翻譯結果，不要添加任何解釋。
#     {instruction}
#     """


# def build_messages(instruction):
#     messages = [
#         {
#             "role": "system",
#             "content": "你是一位精通古代漢語與現代漢語的語言學專家。請準確翻譯下列文本。只輸出翻譯結果，不要添加任何解釋。"
#         },
#         {
#             "role": "user",
#             "content": instruction
#         }
#     ]
#
#     return messages

# def build_messages(instruction):
#     messages = [
#         {
#             "role": "system",
#             "content": "你是一位精通中國古代文學與語言學的專家，擅長文言文與現代漢語之間的互譯。\
#             請遵循以下原則：\
#             1. 準確理解用戶指令中的翻譯方向（現代轉文言，或文言轉現代）。\
#             2. 譯文需簡練、準確，符合對應時代的語言習慣（文言文需古雅，現代文需通順）。\
#             3. 直接輸出翻譯結果，不要添加「答案：」、「譯文：」等多餘前綴，也不要輸出解釋性文字。\
#             4. 若原文涉及歷史人名、官職，請保留或根據常識還原，確保語義通順。"
#         },
#         {
#             "role": "user",
#             "content": instruction
#         }
#     ]
#
#     return messages


# def get_prompt(instruction: str) -> str:
#     return f"""
#     你是古漢語與現代漢語翻譯專家，只輸出譯文。
#     {instruction}
#     """

# def build_messages(instruction):
#     messages = [
#         {
#             "role": "system",
#             "content": "你是古漢語與現代漢語翻譯專家，只輸出譯文。"
#         },
#         {
#             "role": "user",
#             "content": instruction
#         }
#     ]
#
#     return messages

def get_prompt(instruction: str) -> str:
    return f"你是一位精通古漢語與現代漢語的翻譯專家。請準確翻譯下列文本，只輸出譯文。\
            {instruction}"

def build_messages(instruction):
    return [
        {
            "role": "user",
            "content": (
                "你是一位精通古漢語與現代漢語的翻譯專家。"
                "請準確翻譯下列文本，只輸出譯文。\n"
                f"{instruction}"
            )
        }
    ]

def get_prompt_with_template(instruction: str, tokenizer) -> str:
    """
    Generates a prompt with chat template from tokenizer for fine-tuning (QLoRA) inference.
    """
    message = build_messages(instruction)

    # Use the tokenizer's chat template to create the prompt
    # 与训练时数据分布保持一致
    return tokenizer.apply_chat_template(message, tokenize=False)


# def get_prompt(instruction: str) -> str:
#     return f"""
#         你是一位文言文與現代漢語互譯專家。請嚴格遵守以下規則：
#         1. 【任務識別】若指令包含「翻譯成文言文」，則將後續內容譯為文言文；若指令包含「文言文翻譯」或「譯為現代漢語」，則將文言文譯為流暢的現代漢語。
#         2. 【輸出約束】直接輸出翻譯結果，嚴禁添加「答案：」、「譯文：」、「好的」等任何前綴或解釋性文字。
#         3. 【內容處理】保留歷史人名、官職、地名；文言文譯文需簡練古雅，現代文譯文需通順易懂。
#         4. 【格式規範】僅輸出純文本，不使用 Markdown、編號或額外換行。
#
#         {instruction}
#         請直接輸出翻譯結果：
#     """

# Few-Shot
# def get_prompt(instruction: str) -> str:
#     return (f"""你是一位精通中國古代文學與語言學的專家，擅長文言文與現代漢語之間的互譯。
#             請遵循以下原則：
#                 1. 準確理解用戶指令中的翻譯方向（現代轉文言，或文言轉現代）。
#                 2. 譯文需簡練、準確，符合對應時代的語言習慣（文言文需古雅，現代文需通順）。
#                 3. 直接輸出翻譯結果，不要添加「答案：」、「譯文：」等多餘前綴，也不要輸出解釋性文字。
#                 4. 若原文涉及歷史人名、官職，請保留或根據常識還原，確保語義通順。
#
#             請參考以下範例進行翻譯：
#
#             ### 範例 1
#             指令：翻譯成文言文：
#             於是，廢帝讓瀋慶之的堂侄、直將軍瀋攸之賜瀋慶之毒藥，命瀋慶之自殺。
#             輸出：帝乃使慶之從父兄子直閣將軍攸之賜慶之藥。
#
#             ### 範例 2
#             指令：文言文翻譯：
#             靈鑒忽臨，忻歡交集，乃迴燈拂席以延之。
#             輸出：靈仙忽然光臨，趙旭歡欣交集，於是他就把燈點亮，拂拭乾淨床席來延請仙女。
#
#             ### 範例 3
#             指令：翻譯成文言文：
#             硃全忠聽後哈哈大笑。
#             輸出: 全忠大笑。
#
#             ### 範例 4
#             指令： 文言文翻譯：
#             契丹主以陽城之戰為彥卿所敗，詰之。彥卿曰： 臣當時惟知為晉主竭力，今日死生惟命。
#             輸出: 契丹主因陽城之戰被符彥卿打敗，追問符彥卿，彥卿說： 臣當時隻知為晉主竭盡全力，今日死生聽你決定。
#
#             ### 範例 5
#             指令：翻譯成文言文：
#             希望您以後留意，不要再齣這樣的事，你的小女兒病就會好。
#             輸出: 以後幸長官留意，勿令如此。
#
#             ### 當前任務
#             指令：{instruction}
#             請直接輸出翻譯結果：
#         """
#         )

# def get_bnb_config() -> BitsAndBytesConfig:
#     '''Get the BitsAndBytesConfig.'''
#     pass

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return nf4_config
