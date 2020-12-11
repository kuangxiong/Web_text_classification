# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: data_preload.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Wed Dec  9 19:38:01 2020
---------------------------------------------------------
'''
from config import  GlobalConfig
import csv    


def data_load(GlobalConfig):
    """ 用于读取训练数据和测试数据

    Args:
        train_file ([str]): [训练数据]
        train_file ([str]): [训练数据对应的标签数据]
        test_file ([str]): [测试文件]

    """
    f_train_path = open(GlobalConfig.train_path)
    f_train_label = open(GlobalConfig.train_label)
    f_test_path = open(GlobalConfig.test_path)

    tmp_train_data = csv.reader(f_train_path)
    tmp_train_label = csv.reader(f_train_label)
    tmp_test_data = csv.reader(f_test_path)

    # 存放数据
    train_data, train_label, test_data = [], [], []

    for e in tmp_train_data:
        train_data.append(e)

    for e in tmp_train_label:
        train_label.append(e)

    for e in tmp_test_data:
        test_data.append(e) 
    
    f_train_path.close()
    f_train_label.close()
    f_test_path.close()

    return train_data, train_label, test_data   


if __name__=='__main__':
    train_data, train_label, test_data = data_load(GlobalConfig)
    print(train_data[1])
    print(train_label[1])
    print(test_data[1])

    
        

    






   





