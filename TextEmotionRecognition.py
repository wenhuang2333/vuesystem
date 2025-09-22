import asyncio
from pysentimiento import create_analyzer
from googletrans import Translator
from langdetect import detect, LangDetectException


async def analyze_and_print_emotion_fixed(text: str):
    """识别文本情绪，并打印结果"""
    if not isinstance(text, str) or not text.strip():
        print("错误：请输入有效的文本字符串。")
        return
    emotion_translation_map = {'joy': '喜悦😄', 'sadness': '悲伤😢', 'anger': '愤怒😠', 'fear': '恐惧😨',
                               'disgust': '厌恶🤢', 'surprise': '惊喜😮', 'others': '复杂😐'}
    try:
        translator = Translator()
        print("正在加载情绪分析模型...")
        emotion_analyzer = create_analyzer(task="emotion", lang="en")
        print("模型加载完毕！")
    except Exception as e:
        print(f"初始化翻译器或模型失败: {e}")
        return
    print("\n--- 情绪分析结果 ---")
    text_to_analyze = text
    translation_info = ""
    try:
        lang = detect(text)
        if lang.startswith('zh'):
            translated = await translator.translate(text, src='auto', dest='en')
            text_to_analyze = translated.text
            translation_info = f" (中文原文已翻译为: '{text_to_analyze}')"
    except LangDetectException:
        print(f"警告: 文本 '{text}' 语言检测失败，将按原文处理。")
    except Exception as e:
        print(f"错误: 文本 '{text}' 翻译失败: {e}，将按原文处理。")
    analysis_result = emotion_analyzer.predict(text_to_analyze)
    main_emotion_en = analysis_result.output
    main_emotion_zh = emotion_translation_map.get(main_emotion_en, main_emotion_en)
    probabilities = analysis_result.probas
    probabilities_zh = {emotion_translation_map.get(en_label, en_label): prob for en_label, prob in
                        probabilities.items()}
    print(f"文本: '{text}'{translation_info}")
    print(f"主要情绪: {main_emotion_zh}")
    print("--- 各情绪概率分布 ---")
    for emotion_zh, prob in probabilities_zh.items():
        print(f"- {emotion_zh:<5}: {prob:.2%}")


if __name__ == "__main__":
    asyncio.run(analyze_and_print_emotion_fixed("这部电影的结局太悲伤了，我忍不住哭了。"))
    asyncio.run(analyze_and_print_emotion_fixed("I am so happy and excited about my promotion!"))
    asyncio.run(analyze_and_print_emotion_fixed("那个客服太气人了，业务完全不熟练！"))
    asyncio.run(analyze_and_print_emotion_fixed("That customer service agent made me so angry with their incompetence!"))
    asyncio.run(analyze_and_print_emotion_fixed("走在漆黑的小巷里，我感到非常害怕。"))
    asyncio.run(analyze_and_print_emotion_fixed("Wow, I wasn't expecting that ending! What a surprise!"))
    asyncio.run(analyze_and_print_emotion_fixed("这只是一个关于今天天气怎么样的普通陈述。"))
    asyncio.run(analyze_and_print_emotion_fixed("This is just a regular statement about the weather."))
