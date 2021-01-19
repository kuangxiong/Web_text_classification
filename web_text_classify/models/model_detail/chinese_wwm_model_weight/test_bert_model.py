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
import sys
sys.path.append("..")
from config import GlobalConfig
import numpy as np
import argparse
import os
import time

from keras_bert import load_trained_model_from_checkpoint

from BertBilstm import bert_bilstm

from BertBilstm import BertModelConfig as ModelConfig

from bert_data_preload import load_bert_dataset

from config import GlobalConfig


parser = argparse.ArgumentParser(description="web-text-classifier")
parser.add_argument('--model', default="Bilstm", type=str)
args = parser.parse_args()

if __name__=='__main__':

    start_time = time.time()
#
#    bert_model = load_trained_model_from_checkpoint(ModelConfig.bert_config_path,\
#        ModelConfig.bert_checkpoint_path, seq_len=None)
#
#    for l in bert_model.layers:
#        l.trainable = True
#
#    text_id = tf.keras.layers.Input(shape=(ModelConfig.max_len, ), dtype=tf.int32, name='text_id')
#    segment_id = tf.keras.layers.Input(shape=(ModelConfig.max_len, ), dtype=tf.int32, name='segment_id')
#
#    bert_output = bert_model([text_id, segment_id])
#    bilstm_output = keras.layers.Bidirectional(keras.layers.LSTM(ModelConfig.hidden_size//2))(bert_output)
#    dropout = keras.layers.Dropout(ModelConfig.dropout)(bilstm_output, training=True)
#    output1 = keras.layers.Dense(64, activation='relu')(dropout)
#    output2 = keras.layers.Dense(3, activation='softmax')(output1)
#    
#    model = keras.Model(inputs=[text_id, segment_id], outputs=[output2])
    model =  bert_bilstm(ModelConfig)


    print(model.summary())


