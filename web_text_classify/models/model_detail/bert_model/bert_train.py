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

from models.BertBilstm import bert_bilstm, BertModelConfig
from data_preload import load_model_dataset, get_X_and_Y_data
from bert_data_preload import load_bert_dataset
from config import GlobalConfig


parser = argparse.ArgumentParser(description="web-text-classifier")
parser.add_argument('--model', default="Bilstm", type=str)
args = parser.parse_args()

if __name__=='__main__':
    start_time = time.time()
    train_data, train_seg, train_label = load_bert_dataset(GlobalConfig,
            training=True, max_length = BertModelConfig.max_len) 
#    print(train_data[1])

#    model_file = os.path.join(GlobalConfig.save_path, "bertmodel.h")
#    if os.path.isfile(model_file):
#        model = keras.models.load_model(model_file)
#    else:
    model =  bert_bilstm(BertModelConfig)
    adam = tf.keras.optimizers.Adam(BertModelConfig.learning_rate)
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=adam,
        metrics=['accuracy']
    )
    print(model.summary())

##    
##    # 设置回调, 只保留验证集 上最好的模型的
    checkpoint_cb = keras.callbacks.ModelCheckpoint("save_model/bert_model.h5", save_best_only = True)
#    # 设置早停, 当连续patience 个迭代都没有进展的时候，则不再训练
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True) 
    train_X_text, train_X_seg = np.asarray(train_data), np.asarray(train_seg)
    train_label = np.asarray(train_label)
    history = model.fit( 
        # 使用model.fit() 方法来执行训练过程，
        [train_X_text, train_X_seg], train_label, # 训练集的输入以及标签，
        epochs=50,  # 迭代次数 epochs为1
        batch_size=BertModelConfig.batch_size, # 每一批batch的大小
        validation_split=0.2, # 从训练中划分20%给验证集
        validation_freq=1, # 测试的间隔次数为2
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
##   model.save(model_file)
#    model.summary()
#
#
    # for echo in range(ModelConfig.num_epochs):
    #     step = 0
    #     for _, (text_batch, label_batch) in enumerate(train_dataset):
    #         step = step + 1 
    #   
    #     step = 0

    # model.fit_generator(train)
