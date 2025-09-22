import asyncio
from pysentimiento import create_analyzer
from googletrans import Translator
from langdetect import detect, LangDetectException


async def analyze_emotion_chinese_output(text_list):
    """
    使用pysentimiento库分析一系列文本的情绪概率。
    如果检测到文本为中文，则先翻译成英文再进行分析。
    """
    # 中英文情绪标签的映射字典
    emotion_translation_map1 = {
        'joy': '喜悦😄',
        'sadness': '悲伤😢',
        'anger': '愤怒😠',
        'fear': '恐惧😨',
        'disgust': '厌恶🤢',
        'surprise': '惊喜😮',
        'others': '其他😐'
    }
    emotion_translation_map = {
        'joy': 'joy',
        'sadness': 'sad',
        'anger': 'anger',
        'fear': 'fear',
        'disgust': 'disgust',
        'surprise': 'surprise',
        'others': 'others'
    }
    results = []  # 用于存储所有文本的分析结果

    # 初始化翻译器
    try:
        translator = Translator()
    except Exception as e:
        print(f"初始化翻译器失败: {e}")
        print("请检查您的网络连接或尝试更新库: pip install googletrans==4.0.0-rc1")
        return

    # 创建情绪分析器(第一次运行时会自动下载模型)
    try:
        print("正在加载情绪分析模型...")
        emotion_analyzer = create_analyzer(task="emotion", lang="en")
        print("模型加载完毕！")
    except Exception as e:
        print(f"初始化模型失败: {e}")
        print("请检查您的网络连接或尝试更新库: pip install -U pysentimiento transformers")
        return

    print("\n--- 多情绪概率分析  ---")

    for i, original_text in enumerate(text_list):
        text_to_analyze = original_text
        translation_info = ""
        re = {}  # 为每个文本创建一个新的字典

        try:
            lang = detect(original_text)
            if lang.startswith('zh'):
                translated = await translator.translate(original_text, src='auto', dest='en')
                text_to_analyze = translated.text
                translation_info = f" (中文原文已翻译为: '{text_to_analyze}')"
        except LangDetectException:
            print(f"\n警告: 文本 '{original_text}' 语言检测失败，将按原文处理。")
        except Exception as e:
            print(f"\n错误: 文本 '{original_text}' 翻译失败: {e}，将按原文处理。")

        analysis_result = emotion_analyzer.predict(text_to_analyze)

        main_emotion_en = analysis_result.output
        main_emotion_zh = emotion_translation_map.get(main_emotion_en, main_emotion_en)

        probabilities = analysis_result.probas
        probabilities_zh = {
            emotion_translation_map.get(en_label, en_label): prob
            for en_label, prob in probabilities.items()
        }

        print(f"\n文本 {i + 1}: '{original_text}'{translation_info}")
        print(f"主要情绪: {main_emotion_zh}")
        print("--- 各情绪概率分布 ---")

        for emotion_zh, prob in probabilities_zh.items():
            print(f"- {emotion_zh:<5}: {prob:.2%}")
            re[emotion_zh] = f"{prob:.2%}"

        re['main'] = main_emotion_zh
        results.append(re)  # 将每个文本的分析结果存储到列表中

    return results  # 返回所有文本的分析结果


def calculate_average_probabilities(results):
    """
    计算所有字典中相关属性的平均值
    """
    if not results:
        return {}

    # 初始化一个字典来存储总和
    total_probabilities = {key: 0.0 for key in results[0].keys() if key != 'main'}
    count = len(results)

    # 遍历所有结果，累加每个属性的值
    for result in results:
        for key, value in result.items():
            if key != 'main':
                # 去掉百分号并转换为浮点数
                total_probabilities[key] += float(value.strip('%'))

    # 计算平均值
    average_probabilities = {key: f"{value / count:.2f}%" for key, value in total_probabilities.items()}
    return average_probabilities

def text_test_before(sample_texts):
    results = asyncio.run(analyze_emotion_chinese_output(sample_texts))
    average_probabilities = calculate_average_probabilities(results)
    print("平均数",average_probabilities)
    return {key: float(value.strip('%')) for key, value in average_probabilities.items()}
def text_text(sample):
    re=[]
    sample_dic=text_test_before(sample)
    re.append(sample_dic['joy'])
    re.append(sample_dic['sad'])
    re.append(sample_dic['anger'])
    re.append(sample_dic['surprise'])
    re.append(sample_dic['disgust'])
    re.append(sample_dic['fear'])
    re.append(sample_dic['others'])
    return re


if __name__ == "__main__":
    sample_texts = [
        "I am so happy and excited about my promotion!",
        "这部电影的结局太悲伤了，我忍不住哭了。",
        "那个客服太气人了，业务完全不熟练！",
        "That customer service agent made me so angry with their incompetence!",
        "走在漆黑的小巷里，我感到非常害怕。",
        "Wow, I wasn't expecting that ending! What a surprise!",
        "这只是一个关于今天天气怎么样的普通陈述。",
        "This is just a regular statement about the weather."
    ]
    second_texts = [
        "I am so happy and excited about my promotion!",
        "I am excited!",
        "what a good day!I love it!",
        "Wow!I like the life now I have!"
    ]
    print(text_text(second_texts))