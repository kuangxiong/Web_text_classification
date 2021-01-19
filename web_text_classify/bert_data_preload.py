# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: data_preload.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Wed Dec  9 19:38:01 2020
---------------------------------------------------------
'''
import csv
import numpy as np
import os
import re
import pickle
import codecs

from data_util import data_load

from keras_bert import Tokenizer

from sklearn.model_selection import train_test_split

from config import GlobalConfig, BASE_PATH

import tensorflow as tf
from tensorflow import keras
# bert模型的配置文件
from models.bert_bilstm import BertBiLSTMConfig

import sys
sys.path.append("..")

def bert_get_text_id(BertBiLSTMConfig, train_text, training=True, max_length=BertBiLSTMConfig.max_len):
    """
    将训练集上的文字字符转化为字编号

    Args:
        word_dict ([dict]): [字符对应的编号]
        train_text ([list]): [训练文本数据]
        max_length ([int], optional): [设置文本的最大长度]. Defaults to GlobalConfig.text_len.

    Returns:
        [type]: [description]
    """

    token_dict = {}
    with codecs.open(GlobalConfig.bert_vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    tokenizer = Tokenizer(token_dict)
    data_X_ind, data_X_seg = [], []
    data_Y, data_id = [], []
    N = len(train_text)
    for i in range(N):
        tmp_text = train_text[i][1]
        indices, segments = tokenizer.encode(
            first=tmp_text, max_len=BertBiLSTMConfig.max_len)
        indices, segments = np.array(indices), np.array(segments)
        data_X_ind.append(indices)
        data_X_seg.append(segments)
        if training == True:
            data_Y.append(train_text[i][2])
        else:
            data_id.append(train_text[i][0])
    data_X_ind = keras.preprocessing.sequence.pad_sequences(data_X_ind, maxlen=max_length, dtype='int32',
                                                            padding='post', truncating='post', value=0.)

    data_X_seg = keras.preprocessing.sequence.pad_sequences(data_X_seg, maxlen=max_length, dtype='int32',
                                                            padding='post', truncating='post', value=0.)
    if training == True:
        return data_X_ind, data_X_seg, data_Y
    else:
        return data_X_ind, data_X_seg, data_id


def bert_load_data(GlobalConfig, training=True, max_length=BertBiLSTMConfig.max_len):
    """
    生成训练数据集合

    Args:
        Global_config ([class]): [全局配置类型]
        max_length ([int], optional): [每段新闻截取的最长长度]. Defaults to ModelConfig.max_len.

    Returns:
        [type]: [训练集文件，测试集文件]
    """
    train_data, test_data = data_load(GlobalConfig)

    if training == True:
        train_text, train_seg, train_label_id = bert_get_text_id(BertBiLSTMConfig,
                                                                 train_data, training,  max_length=max_length)
        train_id = list(map(int, train_label_id))
        train_label, N = [], len(train_label_id)
        for i in range(N):
            tmp = [0, 0, 0]
            tmp[train_id[i]+1] = 1
            train_label.append(tmp)
        return train_text, train_seg, train_label
    else:
        test_text, test_seg, test_id = bert_get_text_id(BertBiLSTMConfig,
                                                        test_data, training,  max_length=max_length)
        return test_text, test_seg, test_id


def bert_get_X_and_Y_data(data_text, data_label):
    """
    [将数据分为训练集数据和验证集数据]

    Args:
        train_data ([list]): [列表中每个元素形如[id, text, label]的形式]

    Returns:
        [list]: [训练集数据、训练集标签、验证集数据、验证集标签]
    """
    train_X, dev_X, train_Y, dev_Y = train_test_split(data_text, data_label,
                                                      test_size=0.3, random_state=5)
    return train_X, train_Y, dev_X, dev_Y


if __name__ == '__main__':
    print("helloworld")
    #    train_data, test_data = data_load(GlobalConfig)
    #    print(train_data[:5])
    #tmp1, tmp2 = bert_get_text_id(BertBiLSTMConfig, train_data)
    #    print(tmp1[5])
    #    print(tmp2[5])
    train_text, train_seg, train_label = bert_load_data(GlobalConfig)
#    print(len(train_text[5][0]))
#    print(train_label[5])
    # train_data, train_label, dev_data, dev_label = get_X_and_Y_data(train_text, train_id)
   # print(train_text[1], train_id[1])
    # print(train_data)
    # print(len(train_label), train_label)
    # print(train_text_id[1])
