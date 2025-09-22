"""
本程序针对原始文本进行数据预处理，方便其他py文件从这里导入预处理器
使用nltk库对文本进行清理、分词、词性标注和词形还原，并最终将文本转换为模型可以处理的固定长度的数字序列
"""

# 通用模块导入
import re
import string
import numpy as np
import pickle
import tensorflow as tf
import nltk
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords as sw, wordnet as wn
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 确保已下载Nltk相关数据包
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("正在下载NLTK所需数据包...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    print("下载完成。")


class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    一个集成的NLTK预处理类，继承自Sklearn的BaseEstimator和TransformerMixin
    功能包括：文本清理、分词、词性标注、词形还原，并最终将文本向量化为固定长度的序列
    """

    def __init__(self, max_sentence_len=300, stopwords=None, punct=None, lower=True, strip=True):
        """
        初始化预处理器，以下为
        max_sentence_len: 句子最大长度，用于填充序列
        stopwords: 停用词列表，默认为NLTK的英文停用词
        punct: 标点符号集，默认为string.punctuation
        lower: 是否转换为小写
        strip: 是否移除首尾空白
        """
        self.lower = lower
        self.strip = strip
        self.stopwords = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.max_sentence_len = max_sentence_len

    def fit(self, X, y=None):
        """拟合函数，在Sklearn管道中需要，这里直接返回自身"""
        return self

    def inverse_transform(self, X):
        """逆转换函数，这里不需要，直接返回X"""
        return X

    def transform(self, X):
        """核心转换函数，对输入的文本列表X进行完整的预处理"""
        return np.array([self.tokenize(doc) for doc in X])

    def tokenize(self, document):
        """对单个文档进行处理，包括文本清理、分词、词形还原和向量化"""
        lemmatized_tokens = []

        # 文本清理：使用正则表达式替换特定缩写和无关字符
        document = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", document)
        document = re.sub(r"what's", "what is ", document)
        document = re.sub(r"\'s", " ", document)
        document = re.sub(r"\'ve", " have ", document)
        document = re.sub(r"can't", "cannot ", document)
        document = re.sub(r"n't", " not ", document)
        document = re.sub(r"i'm", "i am ", document)
        document = re.sub(r"\'re", " are ", document)
        document = re.sub(r"\'d", " would ", document)
        document = re.sub(r"\'ll", " will ", document)
        document = re.sub(r"(\d+)(k)", r"\g<1>000", document)

        # 将文档分割成句子
        for sent in sent_tokenize(document):
            # 对每个句子进行分词和词性标注
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # 应用预处理到每个词元
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # 如果是停用词或标点符号，则忽略
                if token in self.stopwords or all(char in self.punct for char in token):
                    continue

                # 词形还原
                lemma = self.lemmatize(token, tag)
                lemmatized_tokens.append(lemma)

        # 将处理后的词元列表连接成字符串，并进行向量化
        doc = ' '.join(lemmatized_tokens)
        tokenized_document = self.vectorize(np.array(doc)[np.newaxis])
        return tokenized_document

    def vectorize(self, doc):
        """使用预先保存的Keras Tokenizer将文本序列转换为数字序列，并进行填充"""
        # 加载在训练阶段保存的Tokenizer对象
        save_path = "./must/text/padding.pickle"
        with open(save_path, 'rb') as f:
            tokenizer = pickle.load(f)

        # 文本转序列
        doc_pad = tokenizer.texts_to_sequences(doc)
        # 序列填充
        doc_pad = pad_sequences(doc_pad, padding='pre', truncating='pre', maxlen=self.max_sentence_len)
        return np.squeeze(doc_pad)

    def lemmatize(self, token, tag):
        """将Penn Treebank词性标记转换为WordNet词性标记，以便进行有效的词形还原"""
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        return self.lemmatizer.lemmatize(token, tag)
