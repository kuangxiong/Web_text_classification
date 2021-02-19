# -*- coding:utf-8 -*-
"""
---------------------------------------------------------
 File Name: data_preload.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Wed Dec  9 19:38:01 2020
---------------------------------------------------------
"""

import tensorflow as tf
from tensorflow import keras
from word2vec.w2v_bilstm import ModelConfig
from w2v_config import GlobalConfig, BASE_PATH
from sklearn.model_selection import train_test_split
from data_util import data_load
import csv
import pickle
import re
import os
import numpy as np
import jieba


def get_vocab(word_dict_path):
    """
    获取词向量字典以及词对应的ID编号
    """
    word_dict_file = open(word_dict_path, "rb")
    word_dict = pickle.load(word_dict_file)
    return word_dict


def get_text_id(word_dict, train_text, max_length=ModelConfig.n_vocab):
    """
    将训练集上的文字字符转化为字|词编号

    Args:
        word_dict ([dict]): [字符对应的编号]
        train_text ([list]): [训练文本数据]
        max_length ([int], optional): [设置文本的最大长度]. Defaults to GlobalConfig.text_len.

    Returns:
        [type]: [description]
    """

    data_X, data_Y = [], []
    # train_label, test_label = [], []
    N = len(train_text)
    # N_train, N_test = len(train_text), len(test_text)
    for i in range(N):
        jieba_list = jieba.lcut(train_text[i][1])
        jieba_id = []
        for e in jieba_list:
            if e in word_dict:
                jieba_id.append(word_dict[e])
            else:
                jieba_id.append(word_dict["PAD"])

        data_X.append(jieba_id)
        data_Y.append(train_text[i][2])
    data_X_id = keras.preprocessing.sequence.pad_sequences(
        data_X,
        maxlen=max_length,
        dtype="int32",
        padding="post",
        truncating="post",
        value = 0.0,
    )

    #     for i in range(N_test):
    #         tmp = [word_dict[e] for e in test_text[i][1]]
    #         test_text_id.append(tmp)
    # #        test_id.append(test_text[i][0])
    #     test_text_id = keras.preprocessing.sequence.pad_sequences(test_text_id, maxlen=max_length, dtype='int32',
    #     padding='post', truncating='post', value=0.)
    # return train_text_id, train_id, test_text_id, test_id
    return data_X_id, data_Y


def load_model_dataset(GlobalConfig, max_length=ModelConfig.max_len):
    """
    生成训练数据集合

    Args:
        Global_config ([class]): [全局配置类型]
        max_length ([int], optional): [每段新闻截取的最长长度]. Defaults to ModelConfig.max_len.

    Returns:
        [type]: [训练集文件，测试集文件]
    """
    train_data, test_data = data_load(GlobalConfig)

    check_word_dict = os.path.join(BASE_PATH, GlobalConfig.vocab_dict)
    if os.path.isfile(check_word_dict):
        word_dict_file = open(check_word_dict, "rb")
        word_dict = pickle.load(word_dict_file)
    else:
        word_dict = get_word_dict(train_data, test_data, N=ModelConfig.word_dict_num)

    train_text, tmp_train_id = get_text_id(word_dict, train_data, max_length=max_length)
    tmp_train_id = list(map(int, tmp_train_id))
    # print(set(tmp_train_id))
    train_id = []
    for i in range(len(tmp_train_id)):
        # print(tmp_train_id[i])
        tmp = [0, 0, 0]
        tmp[tmp_train_id[i] + 1] = 1
        train_id.append(tmp)

    return train_text, train_id


if __name__ == "__main__":
    train_data, test_data = data_load(GlobalConfig)
    print(train_data[:5])
    print(len(train_data))
# word_dict = get_word_dict(train_data, test_data)
#    output = open("word_dict.pkl", "wb")
#    pickle.dump(word_dict, output)
    # train_text_id, _ ,test_text_id, _ = get_text_id(word_dict, train_data, test_data)
    train_text, train_id = load_model_dataset(
        GlobalConfig, max_length=ModelConfig.max_len
    )
    # train_data, train_label, dev_data, dev_label = get_X_and_Y_data(train_text, train_id)
    print(train_text[1], train_id[1])
    # print(train_data)
    # print(len(train_label), train_label)
    #
    #print(train_text_id[1])
