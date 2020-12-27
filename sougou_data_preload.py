# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: data_preload.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Wed Dec  9 19:38:01 2020
---------------------------------------------------------
''' 
import tensorflow as tf
from tensorflow import  keras
from models.SougouBilstm import SougouModelConfig
from config import GlobalConfig, BASE_PATH
from sklearn.model_selection import train_test_split
from data_preload import data_load
import codecs
import csv    
import pickle
import re
import os
import numpy as np


def sougou_get_text_emd(SougouModelConfig, train_text, max_length=SougouModelConfig.max_len):
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
    word2id_file = open(GlobalConfig.Sougou_word2id_path,"rb")
    wiki_embedding = open(GlobalConfig.Sougou_embedding_path, 'rb')

    word2id = pickle.load(word2id_file)
    wiki_embedding = pickle.load(wiki_embedding)
    N = len(train_text)
    for i in range(N):
        tmp_text = train_text[i][1]
        tmplist = []
        for e in tmp_text:
            if e in word2id:
                tmplist.append(word2id[e])
            else:
                tmplist.append(word2id["UNK"])
        data_X.append(tmplist)
        data_Y.append(train_text[i][2])
    data_X_pad = keras.preprocessing.sequence.pad_sequences(data_X, maxlen=max_length, dtype='int32',
    padding='post', truncating='post', value=0)
    data_X_embedding = []
    for i in range(len(data_X_pad)):
        tmp_embdding = [np.array(wiki_embedding[e]) for e in data_X_pad[i]]
        data_X_embedding.append(tmp_embdding)
    return data_X_embedding, data_Y 
    
def load_sougou_dataset(GlobalConfig, max_length = SougouModelConfig.max_len):
    """
    生成训练数据集合

    Args:
        Global_config ([class]): [全局配置类型]
        max_length ([int], optional): [每段新闻截取的最长长度]. Defaults to ModelConfig.max_len.

    Returns:
        [type]: [训练集文件，测试集文件]
    """

    train_data, test_data = data_load(GlobalConfig)

    train_text_emd, train_label_id = sougou_get_text_emd(SougouModelConfig, train_data, max_length=max_length)

    train_id = list(map(int, train_label_id))
    train_label, N = [], len(train_label_id)
    for i in range(N):
        tmp = [0, 0, 0]
        tmp[train_id[i]+1] = 1
        train_label.append(tmp)
    
    return train_text_emd, train_label

if __name__=='__main__':
#    train_data, test_data = data_load(GlobalConfig)
#    print(train_data[:5])
#tmp1, tmp2 = bert_get_text_id(BertModelConfig, train_data)
#    print(tmp1[5])
#    print(tmp2[5])
    train_text, train_label = load_sougou_dataset(GlobalConfig)
    train_text, train_label = np.array(train_text[:30]), np.array(train_label[:30])
    for i in range(len(train_text)):
        print("-*-"*10, i, train_text[i])
    
    tmp1 = tf.convert_to_tensor(train_text)
#    tmp2 = tf.convert_to_tensor(train_label) 
    
#    print(train_text)
#    print(len(train_text[1]))
#    print(train_label[5])
    # train_data, train_label, dev_data, dev_label = get_X_and_Y_data(train_text, train_id)
   # print(train_text[1], train_id[1])
    # print(train_data)
    # print(len(train_label), train_label)
    # print(train_text_id[1])
