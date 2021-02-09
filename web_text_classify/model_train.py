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
import numpy as np
import argparse
import os
import time

from config import GlobalConfig
from loss_func import focal_loss_fixed

import sys
sys.path.append("..")

#from models.bert_bilstm import bert_bilstm as mymodel 
#from models.bert_bilstm import BertBiLSTMConfig as ModelConfig

from models.roberta_wwm_bilstm import roberta_wwm_bilstm as mymodel
from models.roberta_wwm_bilstm import RobertaWwmBiLSTMConfig as ModelConfig
from bert_data_preload import BertDataPreload 

#from bert_data_preload import bert_load_data
from config import GlobalConfig

parser = argparse.ArgumentParser(description="web-text-classifier")
parser.add_argument('--model', default="Bilstm", type=str)
args = parser.parse_args()

if __name__=='__main__':
    start_time = time.time()
    model_name = "roberta_wwm_bilstm"
#    model_name = "bert_bilstm"
    globalconfig = GlobalConfig(model_name) 
    data_preload = BertDataPreload(globalconfig, ModelConfig)
    train_data, train_seg, train_label = data_preload.bert_load_data(training=True) 

#    print(train_data[1])

#    model_file = os.path.join(GlobalConfig.save_path, "bertmodel.h")
#    if os.path.isfile(model_file):
#        model = keras.models.load_model(model_file)
#    else:
    model =  mymodel(ModelConfig)
    adam = tf.keras.optimizers.Adam(ModelConfig.learning_rate)
    model.compile(
#        loss="categorical_crossentropy", 
#        loss=FocalLoss, 
        loss=focal_loss_fixed, 
        optimizer=adam,
        metrics=['accuracy']
    )
    print(model.summary())

##    
##    # 设置回调, 只保留验证集 上最好的模型的

    save_path  = globalconfig.save_model
    print(f"{save_path}/bert_model.h5")
    checkpoint_cb = keras.callbacks.ModelCheckpoint(f"{save_path}bert_model.h5", save_best_only = True)
#    # 设置早停, 当连续patience 个迭代都没有进展的时候，则不再训练
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True) 
    train_X_text, train_X_seg = np.asarray(train_data), np.asarray(train_seg)
    train_label = np.asarray(train_label)
    history = model.fit( 
        # 使用model.fit() 方法来执行训练过程，
        [train_X_text, train_X_seg], train_label, # 训练集的输入以及标签，
        epochs=50,  # 迭代次数 epochs为1
        batch_size=ModelConfig.batch_size, # 每一批batch的大小
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
