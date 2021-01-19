# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: loss_func.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Tue Jan 19 11:27:52 2021
---------------------------------------------------------
'''
import tensorflow as tf

def FocalLoss(gamma=0.2, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        
        y_true = tf.cast(y_true, tf.float32)
        loss = - y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)

        loss = tf.math.reduce_sum(loss, axis=-1)
        return loss 

    return focal_loss_fixed
