# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: train.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Wed Dec  9 17:59:44 2020
---------------------------------------------------------
'''

import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import argparse
import os
import time

from Bilstm import BiLSTM, ModelConfig
from data_preload import load_model_dataset, get_X_and_Y_data

import sys
sys.path.append("..")
from config import GlobalConfig


parser = argparse.ArgumentParser(description="web-text—classifier")
parser.add_argument('--model', default="Bilstm", type=str)
args = parser.parse_args()

if __name__=='__main__':
    start_time = time.time()
    test_data, test_label = load_model_dataset(GlobalConfig, training=False, max_length = ModelConfig.max_len) 
    print(test_data[1], test_label[1])
    
    model_file = os.path.join(GlobalConfig.save_path, "model.h")
    model = keras.models.load_model(model_file)
    print(model.summary())
    print(test_data[1])
    result = model.predict(test_data)
    print(len(result), len(test_data))
    print(result)
    print(tf.argmax(result, 1)-1)

    # model.fit_generator(train)

    

