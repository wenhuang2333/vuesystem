"""
本程序用于加载已训练好的模型，在测试数据集上进行评估，并提供对单个文本样本进行预测的功能
"""

# 通用模块导入
import numpy as np
import pandas as pd
import json
import dill
import pickle
import os
from sklearn.model_selection import train_test_split as tts
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model

# 导入预处理器
from data_preprocessing import NLTKPreprocessor

# 定义标签常量
first_labels = ["Extraversion_probability", "Neuroticism_probability", "Agreeableness_probability", "Conscientiousness_probability", "Openness_probability"]
second_labels = ["Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness"]


class TestModel:
    """
    用于测试已保存模型的类
    """

    class KerasPipelineTransformer(BaseEstimator, TransformerMixin):
        """转换器，用于从已保存的文件中加载Keras模型并进行预测"""

        def __init__(self, model_name):
            self.model_name = model_name
            save_path = './must/textKeras/'

            # 检查文件是否存在
            model_path = save_path + self.model_name + '.h5'
            print(f"尝试加载模型路径: {model_path}")

            try:
                # 尝试使用 h5 格式加载
                self.classifier = load_model(model_path)
                print("Keras模型加载成功")
            except Exception as e:
                print(f"H5格式加载失败: {e}")
                # 回退到传统加载方式
                self.load_model_traditional(save_path, model_name)

        def load_model_traditional(self, save_path, model_name):
            """传统加载方式"""
            try:
                from tensorflow.keras.models import model_from_json
                from tensorflow.keras.layers import InputLayer

                # 创建自定义的 InputLayer 类，忽略 batch_shape 参数
                class CustomInputLayer(InputLayer):
                    def __init__(self, *args, **kwargs):
                        # 移除 batch_shape 参数
                        if 'batch_shape' in kwargs:
                            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                        super().__init__(*args, **kwargs)

                json_path = save_path + model_name + '.json'
                weights_path = save_path + model_name + '.weights.h5'

                print(f"尝试加载JSON: {json_path}")
                print(f"尝试加载权重: {weights_path}")

                with open(json_path, 'r', encoding='utf-8') as json_file:
                    model_json = json_file.read()

                # 使用自定义对象加载模型
                self.classifier = model_from_json(
                    model_json,
                    custom_objects={'InputLayer': CustomInputLayer}
                )

                self.classifier.load_weights(weights_path)
                self.classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                print("使用自定义 InputLayer 加载模型成功")
            except Exception as e:
                print(f"传统方式加载失败: {e}")
                # 尝试更深入的修复
                self.fix_model_config(save_path, model_name)

        def fix_model_config(self, save_path, model_name):
            """更深入地修复模型配置"""
            try:
                import json as json_lib
                json_path = save_path + model_name + '.json'
                weights_path = save_path + model_name + '.weights.h5'

                with open(json_path, 'r', encoding='utf-8') as json_file:
                    model_config = json_lib.load(json_file)

                # 递归修复所有层中的 batch_shape
                def fix_layer_config(layer):
                    if 'config' in layer and 'batch_shape' in layer['config']:
                        layer['config']['batch_input_shape'] = layer['config']['batch_shape']
                        del layer['config']['batch_shape']
                    return layer

                if 'config' in model_config:
                    if 'layers' in model_config['config']:
                        for layer in model_config['config']['layers']:
                            fix_layer_config(layer)

                # 保存修复后的配置
                fixed_json_path = save_path + model_name + '_fixed.json'
                with open(fixed_json_path, 'w', encoding='utf-8') as f:
                    json_lib.dump(model_config, f)

                # 使用修复后的配置加载模型
                from tensorflow.keras.models import model_from_json
                with open(fixed_json_path, 'r', encoding='utf-8') as f:
                    fixed_model_json = f.read()
                    self.classifier = model_from_json(fixed_model_json)

                self.classifier.load_weights(weights_path)
                self.classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                print("使用修复后的配置加载模型成功")

            except Exception as e:
                print(f"修复模型配置失败: {e}")
                raise e

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return self.classifier.predict(X)

    def multiclass_accuracy(self, predictions, target):
        """计算多标签分类的准确率"""
        score = []
        # 对每个标签类别单独计算准确率
        for j in range(target.shape[1]):
            count = 0
            for i in range(len(predictions)):
                if predictions[i][j] == target.iloc[i, j]:
                    count += 1
            score.append(count / len(predictions))
        return score

    def test_keras_model(self, X_test, y_test, model_name="Personality_traits_NN"):
        """测试Keras模型"""
        print(f"\n--- 正在测试Keras模型: {model_name} ---")

        # 构建一个用于测试的管道
        test_pipeline = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('classifier', self.KerasPipelineTransformer(model_name))
        ])

        # 进行预测
        y_pred_probs = test_pipeline.transform(X_test)
        # 将概率转换为类别
        y_pred_classes = [[0 if el < 0.5 else 1 for el in item] for item in y_pred_probs]

        # 计算准确率
        accuracy = self.multiclass_accuracy(y_pred_classes, y_test)
        labels = y_test.columns.tolist()
        print("各标签的准确率:")
        for label, acc in zip(labels, accuracy):
            print(f"{label}: {acc:.4f}")

        return y_pred_classes


class Predict:
    """
    使用已保存的模型对新文本进行预测
    """

    def __init__(self, keras_model_name="Personality_traits_NN", svm_model_name="Personality_traits_SVM"):
        self.labels = ['外向性 (Extraversion)', '神经质 (Neuroticism)', '随和性 (Agreeableness)',
                       '尽责性 (Conscientiousness)', '开放性 (Openness)']

        print("开始加载Keras模型...")
        # 加载Keras模型
        self.keras_pipeline = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('classifier', TestModel.KerasPipelineTransformer(keras_model_name))
        ])
        print("Keras模型加载完成")

        # 加载SVM模型
        save_path = "./must/textSVM/"
        svm_model_path = save_path + svm_model_name
        print(f"尝试加载SVM模型: {svm_model_path}")

        if os.path.exists(svm_model_path):
            try:
                with open(svm_model_path, 'rb') as f:
                    self.svm_model = dill.load(f)
                print("SVM模型加载成功")
            except Exception as e:
                print(f"SVM模型加载失败: {e}")
                self.svm_model = None
        else:
            print("SVM模型文件不存在")
            self.svm_model = None

    def predict_personality(self, text):
        """
        对输入的单条文本进行性格预测
        """
        print("\n--- 对新文本进行预测 ---")
        print(f"输入文本: '{text[:100]}...'")

        # Keras模型预测
        keras_pred_probs = self.keras_pipeline.transform([text])[0]
        keras_pred_classes = [1 if p >= 0.2 else 0 for p in keras_pred_probs]
        ans_K = dict()
        # ans_S=dict()
        ans = dict()
        print("\nKeras模型预测结果:")
        for label, prob, clazz, first, second in zip(self.labels, keras_pred_probs, keras_pred_classes, first_labels,
                                                     second_labels):
            print(f"- {label}: {'是 (High)' if clazz == 1 else '否 (Low)'} (概率: {prob:.5f})")
            ans_K[first] = '是 (High)' if clazz == 1 else '否 (Low)'
            ans_K[second] = f"{prob:.5f}"
        # SVM模型预测
        # svm_pred = self.svm_model.predict([text])[0]

        # print("\nSVM模型预测结果:")
        # for label, clazz in zip(self.labels, svm_pred):
        #     print(f"- {label}: {'是 (High)' if clazz == 1 else '否 (Low)'}")
        #     ans_S[label + '定性'] = '是 (High)' if clazz == 1 else '否 (Low)'
        ans['Keras'] = ans_K
        # ans['SVM']=ans_S
        return ans


def text(file_path):
    """处理文本预测的主函数"""
    try:
        print(f"开始处理文件: {file_path}")

        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        print(f"读取文本长度: {len(text_content)} 字符")

        # 直接进行预测
        print("开始初始化预测器...")
        predictor = Predict()
        print("预测器初始化完成，开始预测...")

        ans = predictor.predict_personality(text_content)
        json_ans = json.dumps(ans, indent=4, ensure_ascii=False)

        print("预测完成")
        return json_ans

    except Exception as e:
        error_msg = f"处理文本时出错: {str(e)}"
        print(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)

# print(text("C:\\Users\zhans\Desktop\\aaa.txt"))
