from pysentimiento import create_analyzer
import json
import re
def split_string(input_string):
    # 定义分割符号的正则表达式
    delimiters = r'[.!?:]'

    # 使用正则表达式分割字符串
    result = re.split(delimiters, input_string)

    # 去掉空字符串
    result = [s.strip() for s in result if s.strip()]

    return result
def give_it_true(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        content = file.read()
    # 去掉所有回车符
    content = content.replace('\n', '')
    return split_string(content)



def final(ans,length):
    others=0
    joy=0
    sadness=0
    anger=0
    surprise=0
    disgust=0
    fear=0
    for i in range(0,length):
        others+=float(ans[i]['others'][:-1])
        joy+=float(ans[i]['joy'][:-1])
        sadness+=float(ans[i]['sadness'][:-1])
        anger+=float(ans[i]['anger'][:-1])
        surprise+=float(ans[i]['surprise'][:-1])
        disgust+=float(ans[i]['disgust'][:-1])
        fear+=float(ans[i]['fear'][:-1])
    final_ans=dict()
    final_ans['others']=f"{others/length:.2f}"
    final_ans['joy']=f"{joy/length:.2f}"
    final_ans['sadness']=f"{sadness/length:.2f}"
    final_ans['anger']=f"{anger/length:.2f}"
    final_ans['surprise']=f"{surprise/length:.2f}"
    final_ans['disgust']=f"{disgust/length:.2f}"
    final_ans['fear']=f"{fear/length:.2f}"
    return final_ans

def text(filepath):
    """
    使用 pysentimiento分析一系列文本的情绪概率，
    """
    # 中英文情绪标签的映射字典
    emotion_translation_map = {
        'joy': 'joy',
        'sadness': 'sadness',
        'anger': 'anger',
        'fear': 'fear',
        'disgust': 'disgust',
        'surprise': 'surprise',
        'others': 'others'
    }

    # 创建情绪分析器(第一次运行时会自动下载模型)
    try:
        print("正在加载情绪分析模型...")
        emotion_analyzer = create_analyzer(task="emotion", lang="en")
        print("模型加载完毕！")
    except Exception as e:
        print(f"初始化模型失败: {e}")
        print("请检查您的网络连接或尝试更新库: pip install -U pysentimiento transformers")
        return

    results = []

    print("\n--- 多情绪概率分析  ---")
    text_list = give_it_true(filepath)
    length=len(text_list)
    ans = []
    for i, text in enumerate(text_list):
        re=dict()
        # 模型进行预测
        analysis_result = emotion_analyzer.predict(text)

        # 翻译主要情绪标签
        main_emotion_en = analysis_result.output
        main_emotion_zh = emotion_translation_map.get(main_emotion_en, main_emotion_en)

        # 翻译概率字典中的所有情绪标签
        probabilities = analysis_result.probas
        probabilities_zh = {
            emotion_translation_map.get(en_label, en_label): prob
            for en_label, prob in probabilities.items()
        }

        print(f"\n文本 {i + 1}: '{text}'")
        print(f"主要情绪: {main_emotion_zh}")
        print("--- 各情绪概率分布 ---")

        # 打印翻译后的结果
        for emotion_zh, prob in probabilities_zh.items():
            # 使用 f-string 的对齐功能让输出更整齐
            print(f"- {emotion_zh}: {prob:.2%}")
            re[emotion_zh]= f"{prob:.2%}"

        # 收集结果用于最后的表格展示
        row = {'文本': text, '主要情绪': main_emotion_zh}
        re['final_result']=main_emotion_zh
        # 将翻译后的概率字典也添加到结果中
        row.update({f"概率({k})": v for k, v in probabilities_zh.items()})
        results.append(row)
        ans.append(re)
    ok=final(ans,length)
    print(ok)

    return json.dumps(ok, ensure_ascii=False, indent=4)
