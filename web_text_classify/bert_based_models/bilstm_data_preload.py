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
from models.Bilstm import ModelConfig
from config import GlobalConfig, BASE_PATH
from sklearn.model_selection import train_test_split
import csv
import pickle
import re
import os


def get_word_dict(train_data, test_data, N=ModelConfig.n_vocab):
    """
    生成中文词字典

    Args:
        train_data ([list]): [训练数据集数据]
        test_data ([list]): [测试数据集数据]
    """
    tmpword_dict = {}
    # 获取训练集上的数据字典
    for i in range(len(train_data)):
        # print("test", train_data[i])
        tmp_data = train_data[i][1]
        for j in range(len(tmp_data)):
            if tmp_data[j] in tmpword_dict:
                tmpword_dict[tmp_data[j]] += 1
            else:
                tmpword_dict[tmp_data[j]] = 1

    # 获取测试集上的数据字典
    for i in range(len(test_data)):
        tmp_data = test_data[i][1]
        for j in range(len(tmp_data)):
            if tmp_data[j] in tmpword_dict:
                tmpword_dict[tmp_data[j]] += 1
            else:
                tmpword_dict[tmp_data[j]] = 1

    res_word_dict = sorted(tmpword_dict.items(), key=lambda x: x[1], reverse=True)
    word_dict = {}

    for i in range(min(N, len(res_word_dict))):
        # 编码从1开始，0用于padding的时候编码实用
        word_dict[res_word_dict[i][0]] = i + 1
    word_dict["UNK"] = len(word_dict) + 1
    return word_dict


def get_text_id(word_dict, train_text, max_length=ModelConfig.n_vocab):
    """
    将训练集上的文字字符转化为字编号

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
        tmp = [word_dict[e] for e in train_text[i][1]]
        data_X.append(tmp)
        data_Y.append(train_text[i][2])
    # print(123123, N, train_text, train_label)
    data_X_id = keras.preprocessing.sequence.pad_sequences(
        data_X,
        maxlen=max_length,
        dtype="int32",
        padding="post",
        truncating="post",
        value=0.0,
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

    check_word_dict = os.path.join(BASE_PATH, "data/word_dict.pkl")
    if os.path.isfile(check_word_dict):
        word_dict = pickle.loads(check_word_dict)
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


def get_X_and_Y_data(data_text, data_label):
    """
    [将数据分为训练集数据和验证集数据]

    Args:
        train_data ([list]): [列表中每个元素形如[id, text, label]的形式]

    Returns:
        [list]: [训练集数据、训练集标签、验证集数据、验证集标签]
    """
    train_X, dev_X, train_Y, dev_Y = train_test_split(
        data_text, data_label, test_size=0.3, random_state=5
    )
    return train_X, train_Y, dev_X, dev_Y


if __name__ == "__main__":
    train_data, test_data = data_load(GlobalConfig)
    print(train_data[:5])
    print(len(train_data))
    word_dict = get_word_dict(train_data, test_data)
    output = open("word_dict.pkl", "wb")
    pickle.dump(word_dict, output)
    # train_text_id, _ ,test_text_id, _ = get_text_id(word_dict, train_data, test_data)
    train_text, train_id = load_model_dataset(
        GlobalConfig, max_length=ModelConfig.max_len
    )
    # train_data, train_label, dev_data, dev_label = get_X_and_Y_data(train_text, train_id)
    print(train_text[1], train_id[1])
    # print(train_data)
    # print(len(train_label), train_label)
    # print(train_text_id[1])
