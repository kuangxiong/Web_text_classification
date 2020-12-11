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



BASE_PATH = os.path.dirname(__file__)

class GlobalConfig:
    train_path = os.path.join(BASE_PATH, "data/Train_DataSet.csv") 
    train_label = os.path.join(BASE_PATH, 'data/Train_DataSet_label.csv')
    test_path = os.path.join(BASE_PATH, "data/Test_DataSet.csv") 

