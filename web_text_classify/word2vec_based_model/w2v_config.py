# -*- coding:utf-8 -*-
"""
---------------------------------------------------------
 File Name: config.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Wed Dec  9 18:00:43 2020
---------------------------------------------------------
"""
import os
import sys

BASE_PATH = os.path.dirname(__file__)
#BASE_PATH = "/home/featurize/data"


class GlobalConfig:

    train_path = "data/data_source/nCoV_100k_train.labled.csv"
    test_path = "data/data_source/nCov_10k_test.csv"
    #vocab_dict = os.path.join(BASE_PATH, "word2vec/word2vec_model/word_dct.pkl")
    vocab_dict = os.path.join(BASE_PATH, "data/model_data/word2map.pkl")
    embedding_file = os.path.join(BASE_PATH, "data/model_data/w2v_matrix.npy")
    save_path = os.path.join(BASE_PATH, "data/save_model")
