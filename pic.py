from deepface import DeepFace
import json
import numpy as np
# 用于翻译情绪标签的字典
emotion_translation = {'angry': '愤怒', 'disgust': '厌恶', 'fear': '恐惧', 'happy': '高兴', 'sad': '伤心', 'surprise': '惊讶', 'neutral': '复杂'}




# 使用DeepFace进行情绪分析
def picture(img_path):
    analysis_results = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
    if isinstance(analysis_results, list) and len(analysis_results) > 0:
        first_face_analysis = analysis_results[0]
        print("情绪分析结果 (所有概率):")
        print("-" * 30)
        emotion_scores = first_face_analysis['emotion']

        # 将 float32 转换为 float
        emotion_scores = {k: float(v) for k, v in emotion_scores.items()}

        for emotion_en, score in emotion_scores.items():
            emotion_cn = emotion_translation.get(emotion_en, emotion_en)
            print(f"- {emotion_cn:<5}: {score:.2f} %")
        print("-" * 30)
        dominant_emotion_en = first_face_analysis['dominant_emotion']
        dominant_emotion_cn = emotion_translation.get(dominant_emotion_en, dominant_emotion_en)
        print(f"\n检测到的主要情绪是: {dominant_emotion_cn}")
        emotion_scores['re'] = dominant_emotion_cn
        re = json.dumps(emotion_scores, indent=4, ensure_ascii=False)
        return re
    # except Exception as e:
    #     return(f"分析过程中发生错误: {e}")
# print(picture("C:\\Users\zhans\Desktop\\aaa\\aaa.jpg"))