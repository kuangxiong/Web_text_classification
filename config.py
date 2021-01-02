# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: config.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Wed Dec  9 18:00:43 2020
---------------------------------------------------------
'''
import os 
import sys 

#BASE_PATH = os.path.dirname(__file__)
BASE_PATH = "/home/featurize/data"

class GlobalConfig:
#    train_path = os.path.join(BASE_PATH, "nCoV_101k_train.labled.csv") 
    train_path = "../data/nCoV_100k_train.labled.csv"
#    train_label = os.path.join(BASE_PATH, 'data/Train_DataSet_label.csv')
#    test_path = os.path.join(BASE_PATH, "nCoV_10k_test.csv") 
    test_path = "../data/nCov_10k_test.csv" 
    vocab_dict=os.path.join(BASE_PATH,"word_dct.pkl")
#vocab_dict="data/word_dct.pkl"
#    save_path=os.path.join(BASE_PATH, "save_model")
    save_path="./save_model"
# Bert模型配置路径
    Bert_config_path = "../data/bert_source/Chinese_L-12_H-768_A-12/bert_config.json"
    Bert_checkpoint_path ="../data/bert_source/Chinese_L-12_H-768_A-12/bert_model.ckpt" 
    vocab_path = "../data/bert_source/Chinese_L-12_H-768_A-12/vocab.txt"

