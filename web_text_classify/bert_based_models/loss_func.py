# -*- coding:utf-8 -*-
"""
---------------------------------------------------------
 File Name: loss_func.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Tue Jan 19 11:27:52 2021
---------------------------------------------------------
"""
import tensorflow as tf
from tensorflow.keras import losses

# def FocalLoss(gamma=0.2, alpha=0.25):
#    def focal_loss_fixed(y_true, y_pred):
#        y_pred = tf.nn.softmax(y_pred, axis=-1)
#        epsilon = tf.keras.backend.epsilon()
#        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
#
#        y_true = tf.cast(y_true, tf.float32)
#        loss = - y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
#
#        loss = tf.math.reduce_sum(loss, axis=-1)
#        return loss
#
#    return focal_loss_fixed


def focal_loss_fixed(y_true, y_pred):
    epsilon = 1.0e-9
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ce = tf.math.multiply(y_true, -tf.math.log(y_pred))
    #    weight = tf.math.subtract(1.0, y_pred)
    weight = tf.math.multiply(y_true, tf.math.pow(tf.math.subtract(1.0, y_pred), 0.2))

    loss = tf.math.multiply(0.25, tf.math.multiply(weight, ce))
    loss = tf.math.reduce_sum(loss, axis=-1)
    return loss


# def FocalLoss(gamma=0.2, alpha=0.25):
#    def focal_loss_fixed(y_true, y_pred):
#        epsilon = 1.0e-9
#        y_true = tf.convert_to_tensor(y_true)
#        print(y_true.shape)
#        ce = tf.multiply(y_true, -tf.log(y_pred))
#        weight = tf.multiply(y_true, tf.pow(tf.subtract(1.0, y_pred), gamma))
#
#        loss = tf.multiply(alpha, tf.multiply(weight, ce))
#        loss = tf.math.reduce_sum(loss, axis=-1)
#        return loss
#
#    return focal_loss_fixed

# class FocalLoss(losses.Loss):
#    def __init__(self, gamma=2.0, alpha=0.25):
#        self.gamma = gamma
#        self.alpha = alpha
#
#    def call(self, y_true, y_pred):
#        epsilon = 1.0e-9
#        y_true = tf.convert_to_tensor(y_true)
#        ce = tf.math.multiply(y_true, -tf.math.log(y_pred))
#        weight = tf.math.multiply(y_true, tf.math.pow(tf.subtract(1.0, y_pred),
#                    gamma))
#        loss = tf.math.multipy(alpha, tf.math.multiply(weight, ce))
#        loss = tf.math.reduce_sum(loss, axis=-1)
#        return loss
