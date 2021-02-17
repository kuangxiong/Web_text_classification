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
import tensorflow as tf
from tensorflow import keras

from data_util import data_load
from keras_bert import Tokenizer
# bert模型的配置文件

import sys
sys.path.append("..")


class BertDataPreload(object):
    """配置参数"""
     
    def __init__(self, GlobalConfig, ModelConfig):
        self.modelconfig = ModelConfig
        self.globalconfig = GlobalConfig

    def bert_get_text_id(self, train_text, training, max_length):
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
        with codecs.open(self.globalconfig.vocab_path, 'r', 'utf8') as reader:
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
                first=tmp_text, max_len= self.modelconfig.max_len)
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
    
    
    def bert_load_data(self, training=True):
        """
        生成训练数据集合
    
        Args:
            Global_config ([class]): [全局配置类型]
            max_length ([int], optional): [每段新闻截取的最长长度]. Defaults to ModelConfig.max_len.
    
        Returns:
            [type]: [训练集文件，测试集文件]
        """

        max_length = self.modelconfig.max_len

        train_data, test_data = data_load(self.globalconfig)
    
        if training == True:
            train_text, train_seg, train_label_id =  self.bert_get_text_id(train_data, training,  max_length)
            train_id = list(map(int, train_label_id))
            train_label, N = [], len(train_label_id)
            for i in range(N):
                tmp = [0, 0, 0]
                tmp[train_id[i]+1] = 1
                train_label.append(tmp)
            return train_text, train_seg, train_label
        else:
            test_text, test_seg, test_id = self.bert_get_text_id(test_data, training,  max_length)
            return test_text, test_seg, test_id

if __name__ == '__main__':
    print("helloworld")
#train_data, test_data = data_load(GlobalConfig)
#    print(train_data[:5])
    #tmp1, tmp2 = bert_get_text_id(BertBiLSTMConfig, train_data)
    #    print(tmp1[5])
    #    print(tmp2[5])
#train_text, train_seg, train_label = bert_load_data(GlobalConfig)
#    print(len(train_text[5][0]))
#    print(train_label[5])
    # train_data, train_label, dev_data, dev_label = get_X_and_Y_data(train_text, train_id)
   # print(train_text[1], train_id[1])
    # print(train_data)
    # print(len(train_label), train_label)
    # print(train_text_id[1])
