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
import csv

from BertWwmBilstm import bertwwm_bilstm, BertWwmModelConfig
from bertwwm_data_preload import load_bert_dataset
from keras_bert import get_custom_objects

import sys
sys.path.append("..")
from config import GlobalConfig


parser = argparse.ArgumentParser(description="web-text—classifier")
parser.add_argument('--model', default="Bilstm", type=str)
args = parser.parse_args()

if __name__=='__main__':
    start_time = time.time()
    test_text, test_seg, test_id = load_bert_dataset(GlobalConfig, training=False, max_length = BertWwmModelConfig.max_len) 
    
    #model_file = os.path.join(GlobalConfig.save_path, "model.h")
    model = keras.models.load_model("save_model/bert_model.h5", custom_objects =
            get_custom_objects())
    print(model.summary())
    result = model.predict([test_text, test_seg])
    print(result)
    print(tf.argmax(result, 1)-1)
    test_label = tf.argmax(result, 1)-1 
    result_file = open("result.csv", "w")
    result_writer = csv.writer(result_file)
    result_writer.writerow(["id","y"])
    for i in range(len(test_label)):
        result_writer.writerow([test_id[i][:-1], test_label[i].numpy()])
    result_file.close()



    

