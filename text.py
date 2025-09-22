"""
本程序用于加载已训练好的模型，在测试数据集上进行评估，并提供对单个文本样本进行预测的功能
"""

# 通用模块导入
import numpy as np
import pandas as pd
import json
import dill
import pickle
from sklearn.model_selection import train_test_split as tts
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model

# 导入预处理器
from data_preprocessing import NLTKPreprocessor


class TestModel:
    """
    用于测试已保存模型的类
    """

    class KerasPipelineTransformer(BaseEstimator, TransformerMixin):
        """转换器，用于从已保存的文件中加载Keras模型并进行预测"""

        def __init__(self, model_name):
            self.model_name = model_name
            save_path = './must/textKeras/'
            # json_file = open(save_path + self.model_name + '.json', 'r')
            self.classifier = load_model(save_path + self.model_name + '.h5')


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

    # def test_svm_model(self, X_test, y_test, model_name="Personality_traits_SVM"):
    #     """测试SVM模型"""
    #     print(f"\n--- 正在测试SVM模型: {model_name} ---")
    #     save_path = "./must/textSVM/"
    #     with open(save_path + model_name, 'rb') as f:
    #         model = dill.load(f)
    #
    #     y_pred = model.predict(X_test)
    #
    #     accuracy = self.multiclass_accuracy(y_pred, y_test)
    #     labels = y_test.columns.tolist()
    #     print("各标签的准确率:")
    #     for label, acc in zip(labels, accuracy):
    #         print(f"{label}: {acc:.4f}")
    #
    #     return y_pred

first_labels=["Extraversion_probability","Neuroticism_probability","Agreeableness_probability","Conscientiousness_probability","Openness_probability"]
second_labels=["Extraversion","Neuroticism","Agreeableness","Conscientiousness","Openness"]
class Predict:
    """
    使用已保存的模型对新文本进行预测
    """

    def __init__(self, keras_model_name="Personality_traits_NN", svm_model_name="Personality_traits_SVM"):
        self.labels = ['外向性 (Extraversion)', '神经质 (Neuroticism)', '随和性 (Agreeableness)',
                       '尽责性 (Conscientiousness)', '开放性 (Openness)']
        # 加载Keras模型
        self.keras_pipeline = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('classifier', TestModel.KerasPipelineTransformer(keras_model_name))
        ])

        # 加载SVM模型
        save_path = "./must/textSVM/"
        print(save_path+ svm_model_name)
        with open(save_path + svm_model_name, 'rb') as f:
            self.svm_model = dill.load(f)

    def predict_personality(self, text):
        """
        对输入的单条文本进行性格预测
        """
        print("\n--- 对新文本进行预测 ---")
        print(f"输入文本: '{text[:100]}...'")

        # Keras模型预测
        keras_pred_probs = self.keras_pipeline.transform([text])[0]
        keras_pred_classes = [1 if p >= 0.2 else 0 for p in keras_pred_probs]
        ans_K=dict()
        # ans_S=dict()
        ans=dict()
        print("\nKeras模型预测结果:")
        for label, prob, clazz ,first,second in zip(self.labels, keras_pred_probs, keras_pred_classes,first_labels,second_labels):
            print(f"- {label}: {'是 (High)' if clazz == 1 else '否 (Low)'} (概率: {prob:.5f})")
            ans_K[ first]='是 (High)' if clazz == 1 else '否 (Low)'
            ans_K[second]=f"{prob:.5f}"
        # SVM模型预测
        # svm_pred = self.svm_model.predict([text])[0]

        # print("\nSVM模型预测结果:")
        # for label, clazz in zip(self.labels, svm_pred):
        #     print(f"- {label}: {'是 (High)' if clazz == 1 else '否 (Low)'}")
        #     ans_S[label + '定性'] = '是 (High)' if clazz == 1 else '否 (Low)'
        ans['Keras']=ans_K
        # ans['SVM']=ans_S
        return ans


def text(file_path):
    # 加载并准备数据
    print("正在加载和拆分数据...")
    data_essays = pd.read_csv('./source/essays.csv', encoding="ISO-8859-1")
    with open(file_path) as f:
        sorrow=f.read()
    for col in ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']:
        data_essays[col] = np.where(data_essays[col] == 'y', 1, 0)

    X_essays = data_essays['TEXT'].tolist()
    y_essays = data_essays[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]

    X_train, X_test, y_train, y_test = tts(X_essays, y_essays, test_size=0.1, random_state=42)

    # 在测试集上评估模型
    tester = TestModel()
    tester.test_keras_model(X_test, y_test)
    # tester.test_svm_model(X_test, y_test)

    # 对新样本进行预测（从测试集中取一个样本作为测试用例）
    predictor = Predict()
    sample_text = X_test[12]
    ans=predictor.predict_personality(sorrow)
    json_ans=json.dumps(ans,indent=4, ensure_ascii=False)
    return json_ans
# text("./source/love.txt")
print(text("C:\\Users\zhans\PycharmProjects\pythonProject1\情绪识别\服务器\source\\text.txt"))




