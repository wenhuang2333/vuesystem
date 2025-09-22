# predict.py (修改版)

import torch
import numpy as np
from scipy import io as scio
from scipy.stats import zscore
import os
import json
# 确保可以从其他文件导入模型定义
from must3.model import DGCNN

# --- 1. 配置加载 (这部分不变) ---
XDIM = [64, 62, 5]
K_ADJ = 40
NUM_OUT = 64
NCLASS = 3
MODEL_PATH = './must3/final_model.pth'
EMOTION_MAP = {
    0: "消极 ",  # 为了方便作为JSON的键，去掉了表情符号
    1: "中性 ",
    2: "积极 "
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_prediction_model(model_path, xdim, k_adj, num_out, nclass):
    """
    加载预训练好的DGCNN模型用于推理。(此函数不变)
    """
    print(f"正在从 '{model_path}' 加载模型...")
    model = DGCNN(xdim, k_adj, num_out, nclass=nclass)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}。")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("模型加载成功，并已设置为评估模式。")
    return model


def preprocess_input_mat(mat_file_path):
    """
    加载并预处理单个.mat文件。(此函数不变)
    """
    print(f"正在预处理文件: {mat_file_path}...")
    datasets = scio.loadmat(mat_file_path)
    if 'DE' not in datasets:
        raise ValueError("上传的 .mat 文件中必须包含名为 'DE' 的特征矩阵。")
    de_features = datasets['DE']
    data_all = np.transpose(de_features, [1, 0, 2])
    data_all = zscore(data_all, axis=0)
    data_tensor = torch.from_numpy(data_all).float()
    print(f"预处理完成，数据形状: {data_tensor.shape}")
    return data_tensor


# ================================================================= #
# ===================    核心修改在此函数中    ==================== #
# ================================================================= #
def predict_emotion_distribution(model, data_tensor):
    """
    使用加载好的模型进行预测，并返回各种情绪的占比。
    """
    print("开始进行情绪预测...")
    data_tensor = data_tensor.to(device)

    with torch.no_grad():
        outputs = model(data_tensor)
        _, predicted_indices = torch.max(outputs.data, 1)

    predicted_indices = predicted_indices.cpu().numpy()
    print(f"模型对文件中 {len(predicted_indices)} 个样本进行了预测。")

    # --- 结果聚合：计算占比 ---
    if len(predicted_indices) == 0:
        return {"error": "无有效预测"}

    # 获取总预测次数
    total_predictions = len(predicted_indices)

    # 初始化一个字典用于存放最终的占比结果
    emotion_distribution = {}

    # 遍历 EMOTION_MAP (0: "消极", 1: "中性", 2: "积极")
    for index, emotion_label in EMOTION_MAP.items():
        # 计算当前情绪标签 (index) 在所有预测中出现的次数
        count = np.sum(predicted_indices == index)
        # 计算占比 (次数 / 总数)
        proportion = count / total_predictions
        # 将结果存入字典，键为情绪标签，值为占比
        emotion_distribution[emotion_label] = proportion

    print(f"情绪占比计算完成: {emotion_distribution}")
    return emotion_distribution


# --- 主执行函数，用于直接运行此脚本进行测试 ---
# predict.py 的底部

def brain(test_file):
    ans=dict()
    if not os.path.exists(test_file):
        print(f"错误: 测试文件 '{test_file}' 不存在。")
    else:
        try:
            # 1. 加载模型 (不变)
            dgcnn_model = load_prediction_model(MODEL_PATH, XDIM, K_ADJ, NUM_OUT, NCLASS)

            # 2. 预处理整个输入文件 (不变)
            full_input_data = preprocess_input_mat(test_file)

            # =================== 【新的修改】 =================== #
            # 从完整的3394个样本中，只取前面的一部分进行分析，例如前300个样本
            # 你可以调整这个数字，模拟分析不同的数据片段
            num_samples_to_analyze = 300
            partial_input_data = full_input_data[:num_samples_to_analyze]

            print(f"\n注意：已从全部 {len(full_input_data)} 个样本中截取前 {len(partial_input_data)} 个进行分析...")
            # ===================================================== #

            # 3. 对截取后的部分数据执行预测
            distribution = predict_emotion_distribution(dgcnn_model, partial_input_data)

            # 4. 打印最终结果 (不变)
            print("\n=====================================")
            print(f"文件 '{os.path.basename(test_file)}' 的【部分数据】情绪占比如下:")
            for emotion, value in distribution.items():
                print(f"   >> {emotion}: {value:.2%}")
                ans[emotion]=f"{value:.2%}"
                ans[emotion]=ans[emotion][:-1]
                ans[emotion]=float(ans[emotion])
            print("=====================================")
            return json.dumps(ans, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"\n处理过程中发生错误: {e}")














# # predict.py
#
# import torch
# import numpy as np
# from scipy import io as scio
# from scipy.stats import zscore
# import os
#
# # 确保可以从其他文件导入模型定义
# from model import DGCNN
#
# # --- 1. 配置加载 ---
#
# # 定义你的模型在训练时使用的超参数。
# # !!! 这里的参数必须和你训练时使用的 main_DE_subject_independent_complete_version.py 中的参数完全一致 !!!
# XDIM = [64, 62, 5]  # [批量大小, 节点数(通道), 特征维度(频带)]
# K_ADJ = 40
# NUM_OUT = 64
# NCLASS = 3
#
# # 定义最终模型文件的路径
# MODEL_PATH = './final_model.pth'  # 这是你从第一步中复制并重命名的模型
#
# # 定义情绪标签映射
# # 模型的输出是 0, 1, 2，我们将其映射为人类可读的标签
# # 这个顺序需要和你训练时处理标签的顺序一致 (-1 -> 0, 0 -> 1, 1 -> 2)
# EMOTION_MAP = {
#     0: "消极 (Negative) 😢",
#     1: "中性 (Neutral) 😌",
#     2: "积极 (Positive) 😊"
# }
#
# # 自动选择设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def load_prediction_model(model_path, xdim, k_adj, num_out, nclass):
#     """
#     加载预训练好的DGCNN模型用于推理。
#     """
#     print(f"正在从 '{model_path}' 加载模型...")
#     # 1. 实例化模型结构，确保与训练时完全一致
#     model = DGCNN(xdim, k_adj, num_out, nclass=nclass)
#
#     # 2. 检查模型文件是否存在
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"模型文件未找到: {model_path}。请确保你已完成第一步，将训练好的模型放在正确位置。")
#
#     # 3. 加载模型权重 (state_dict)
#     # 你的训练脚本保存的是一个字典，所以需要先加载字典，再提取权重
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#
#     # 4. 将模型移动到指定设备 (CPU或GPU)
#     model.to(device)
#
#     # 5. !!! 关键步骤 !!! 将模型设置为评估模式 (evaluation mode)
#     # 这会关闭 Dropout 和 BatchNorm 的训练行为，对于正确的预测至关重要。
#     model.eval()
#
#     print("模型加载成功，并已设置为评估模式。")
#     return model
#
#
# def preprocess_input_mat(mat_file_path):
#     """
#     加载并预处理单个.mat文件，使其符合模型输入要求。
#     !!! 这里的预处理步骤必须和训练时的预处理完全一样 !!!
#     """
#     print(f"正在预处理文件: {mat_file_path}...")
#     # 1. 加载.mat文件
#     datasets = scio.loadmat(mat_file_path)
#     # 假设上传的.mat文件是经过 extract_DE.py 处理后的格式，包含'DE'键
#     if 'DE' not in datasets:
#         raise ValueError("上传的 .mat 文件中必须包含名为 'DE' 的特征矩阵。")
#     de_features = datasets['DE']
#
#     # 2. 转换数据维度以匹配模型输入
#     # (通道, 样本数, 频段) -> (样本数, 通道, 频段)
#     data_all = np.transpose(de_features, [1, 0, 2])
#
#     # 3. 对数据进行Z-Score标准化 (沿着样本维度)
#     # 这是你在训练脚本中对每个受试者都做的操作，所以这里也要做
#     data_all = zscore(data_all, axis=0)
#
#     # 4. 转换为PyTorch张量
#     data_tensor = torch.from_numpy(data_all).float()
#
#     print(f"预处理完成，数据形状: {data_tensor.shape}")
#     return data_tensor
#
#
# def predict_emotion(model, data_tensor):
#     """
#     使用加载好的模型对预处理后的数据进行预测。
#     """
#     print("开始进行情绪预测...")
#     # 将数据移动到模型所在的设备
#     data_tensor = data_tensor.to(device)
#
#     # 在 `torch.no_grad()` 上下文中进行预测，可以禁用梯度计算，节省内存和计算资源
#     with torch.no_grad():
#         # 将数据输入模型，得到原始输出 (logits)
#         outputs = model(data_tensor)
#         # 找到每个样本得分最高的类别索引作为预测结果
#         # outputs.data 的形状是 (样本数, 类别数)
#         _, predicted_indices = torch.max(outputs.data, 1)
#
#     # 将预测结果从GPU移回CPU，并转换为Numpy数组
#     predicted_indices = predicted_indices.cpu().numpy()
#     print(f"模型对文件中 {len(predicted_indices)} 个样本进行了预测。")
#
#     # --- 结果聚合：投票法 ---
#     # 一个.mat文件包含多个样本（时间点），模型会对每个样本都给出一个预测。
#     # 我们需要一个最终的、代表整个文件的预测结果。最简单有效的方法是“少数服从多数”。
#     if len(predicted_indices) == 0:
#         return "无有效预测"
#
#     # 统计每个类别出现的次数
#     counts = np.bincount(predicted_indices)
#     # 找到出现次数最多的类别索引
#     final_prediction_index = np.argmax(counts)
#
#     # 使用我们定义的映射，将索引转换为人类可读的标签
#     final_emotion = EMOTION_MAP[final_prediction_index]
#
#     print(f"投票聚合完成。")
#     return final_emotion
#
#
# # --- 主执行函数，用于直接运行此脚本进行测试 ---
# if __name__ == '__main__':
#     # 替换为你想要测试的.mat文件的路径
#     # 这个.mat文件应该是你的 extract_DE.py 脚本的输出之一
#     test_file = 'D:/Pycharm/python/pythonProject3/SEED_code/DE/session1/1_20131027.mat'
#
#     if not os.path.exists(test_file):
#         print(f"错误: 测试文件 '{test_file}' 不存在。请修改路径或运行预处理脚本。")
#     else:
#         try:
#             # 1. 加载模型
#             dgcnn_model = load_prediction_model(MODEL_PATH, XDIM, K_ADJ, NUM_OUT, NCLASS)
#             # 2. 预处理输入文件
#             input_data = preprocess_input_mat(test_file)
#             # 3. 执行预测
#             predicted_emotion = predict_emotion(dgcnn_model, input_data)
#             # 4. 打印最终结果
#             print("\n=====================================")
#             print(f"文件 '{os.path.basename(test_file)}' 的最终情绪预测结果是:")
#             print(f"   >> {predicted_emotion}")
#             print("=====================================")
#
#         except Exception as e:
#             print(f"\n处理过程中发生错误: {e}")