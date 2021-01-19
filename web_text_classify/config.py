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
#    train_path = os.path.join(BASE_PATH, "data/data_source/nCoV_101k_train.labled.csv") 

    train_path = "data/data_source/nCoV_100k_train.labled.csv"
#    train_label = os.path.join(BASE_PATH, 'data/Train_DataSet_label.csv')
#    test_path = os.path.join(BASE_PATH, "data/data_source/nCoV_10k_test.csv") 

    test_path = "data/data_source/nCov_10k_test.csv" 
    vocab_dict=os.path.join(BASE_PATH,"word_dct.pkl")
#   vocab_dict="data/word_dct.pkl"
#   save_path=os.path.join(BASE_PATH, "save_model")
    save_path="../data/save_model"

# Bert模型路径

    bert_config_path = "./data/model_source/chinese_L-12_H-768_A-12/bert_config.json"
    bert_checkpoint_path ="./data/model_source/chinese_L-12_H-768_A-12/bert_model.ckpt" 
    bert_vocab_path = "./data/model_source/chinese_L-12_H-768_A-12/vocab.txt"
# BertWWm 模型路径
    bert_wwm_config_path = "./data/model_source/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json"
    bert_wwm_checkpoint_path ="./data/model_source/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt" 
    bert_wwm_vocab_path = "./data/model_source/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt"
# ernie 模型路径
    ernie_config_path = "../data/model_source/ernie_stable-1.0/ERNIE_tensor/bert_config.json"
    ernie_checkpoint_path ="../data/model_source/ernie_stable-1.0/ERNIE_tensor/bert_model.ckpt" 
    ernie_vocab_path = "../data/model_source/ernie_stable-1.0/ERNIE_tensor/vocab.txt"
# roberta-bert-wwm模型路径
    roberta_wwm_config_path = "../data/model_source/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json"
    roberta_wwm_checkpoint_path = "../data/model_source/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt" 
    roberta_wwm_vocab_path = "../data/model_source/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt"
    


