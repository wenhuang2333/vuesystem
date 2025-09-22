"""
此程序用于测试预训练的SVM语音情感识别模型。
它会加载一个音频文件，提取其特征，然后使用预训练模型进行情感预测。
"""

# 通用模块导入
import os
import sys
import pickle
import numpy as np
import scipy
import librosa
from pydub import AudioSegment

# 报警模块导入
import warnings
warnings.filterwarnings('ignore')

# 机器学习模块导入
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer

# 音频预处理模块导入
from AudioLibrary.AudioSignal import AudioSignal
from AudioLibrary.AudioFeatures import AudioFeatures

# 模型和数据的文件夹路径
MODEL_FOLDER = './must2/Model/'
DATA_FOLDER = './must2/Datas/'

# 待测试的音频文件路径 (请将您的测试音频放在此路径下)
TEST_AUDIO_PATH = os.path.join(DATA_FOLDER, 'TestAudio', 'test_audio.wav')
path="./must2/Datas/TestAudio/悲伤.wav"
# 情感标签映射
# EMOTION_MAP = {'NEU': '平静😐', 'HAP': '高兴😄', 'SAD': '悲伤😢', 'ANG': '愤怒😠', 'FEA': '恐惧😨', 'DIS': '厌恶🤢', 'SUR': '惊讶😮'}
EMOTION_MAP = {'NEU': '平静', 'HAP': '高兴', 'SAD': '悲伤', 'ANG': '愤怒', 'FEA': '恐惧', 'DIS': '厌恶',
               'SUR': '惊讶'}

# 定义特征提取函数(与训练时保持一致)
def global_feature_statistics(y, win_size=0.025, win_step=0.01,
                              stats=['mean', 'std', 'med', 'kurt', 'skew', 'q1', 'q99', 'min', 'max', 'range'],
                              features_list=['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread',
                                             'spectral_entropy', 'spectral_flux', 'sprectral_rolloff', 'mfcc']):
    """从音频信号中提取全局特征统计数据"""
    # 训练时使用了12个MFCC系数
    nb_mfcc = 12
    # 使用AudioFeatures类
    audio_features = AudioFeatures(y, win_size, win_step)
    # 调用库中的特征提取方法
    features, _ = audio_features.global_feature_extraction(stats=stats, features_list=features_list, nb_mfcc=nb_mfcc)
    return features

# 主预测函数
def sound(audio_path):
    """
    对单个音频文件进行加载、预处理和情感预测。
    最终修正版：增加了强制数据净化步骤，确保进入PCA的数据绝对有效。
    """
    if not os.path.exists(audio_path):
        return f"错误：找不到音频文件，请检查路径：{audio_path}"

    print("--- 开始情感识别 ---")

    print("步骤 1/5: 加载预训练模型...")
    try:
        model = pickle.load(open(os.path.join(MODEL_FOLDER, 'MODEL_CLASSIFIER.p'), 'rb'))
        lb = pickle.load(open(os.path.join(MODEL_FOLDER, 'MODEL_ENCODER.p'), 'rb'))
        pca = pickle.load(open(os.path.join(MODEL_FOLDER, 'MODEL_PCA.p'), 'rb'))
        [MEAN, STD] = pickle.load(open(os.path.join(MODEL_FOLDER, 'MODEL_SCALER.p'), 'rb'))
        original_data_path = os.path.join(DATA_FOLDER, 'Pickle',
                                          '[RAVDESS][HAP-SAD-NEU-ANG-FEA-DIS-SUR][GLOBAL_STATS].p')
        [features_orig, labels_orig] = pickle.load(open(original_data_path, "rb"))
    except Exception as e:
        return f"错误：加载模型或数据文件失败。请检查路径和文件是否损坏。\n错误详情: {e}"

    print("步骤 2/5: 重新生成特征选择掩码...")
    X_train, _, y_train, _ = train_test_split(features_orig, labels_orig, test_size=0.2, random_state=123)
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    lb_train = LabelEncoder().fit(y_train)
    y_train_encoded = lb_train.transform(y_train)
    alpha = 0.01
    Kbest = SelectKBest(k="all")
    selected_features_obj = Kbest.fit(X_train_scaled, y_train_encoded)
    selection_mask = np.where(selected_features_obj.pvalues_ < alpha)[0]

    print(f"步骤 3/5: 从 '{os.path.basename(audio_path)}' 提取特征...")
    signal = AudioSignal(44100, filename=audio_path)
    stats_train = ['mean', 'std', 'med', 'kurt', 'skew', 'q1', 'q99', 'min', 'max', 'range']
    features_list_train = ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread',
                           'spectral_entropy', 'spectral_flux', 'sprectral_rolloff', 'mfcc']
    features_test = global_feature_statistics(signal, stats=stats_train, features_list=features_list_train)
    features_test = features_test.reshape(1, -1)

    print("步骤 4/5: 应用数据清洗、标准化、特征选择和PCA变换...")

    # 进行标准化缩放 (之前的步骤)
    STD_safe = np.where(STD == 0, 1.0, STD)
    features_test_scaled = (features_test - MEAN) / STD_safe

    # 使用掩码进行特征选择
    features_test_selected = features_test_scaled[:, selection_mask]

    # 在送入PCA之前，强制将任何剩余的 NaN/inf 值转换为0或大的有限数。这是确保数据清洁的最后一道防线。
    features_test_selected = np.nan_to_num(
        features_test_selected, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA变换
    features_test_pca = pca.transform(features_test_selected)

    print("步骤 5/5: 进行情感预测...")
    prediction_raw = model.predict(features_test_pca)
    prediction_label = lb.inverse_transform(prediction_raw)[0]

    emotion_code = prediction_label[2:]
    final_emotion = EMOTION_MAP.get(emotion_code, "未知情感")

    print("--- 识别完成 ---\n")
    print(f"{final_emotion}")
    return f"{final_emotion}"
# print(sound("C:\\Users\zhans\Desktop\\aaa\\aaa.wav"))
# 运行程序
# if __name__ == '__main__':
#     result = predict_emotion(path)
#     print(result)






