# -*- coding:utf-8 -*-
"""
---------------------------------------------------------
 File Name: data_preload.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Wed Dec  9 19:38:01 2020
---------------------------------------------------------
"""
import tensorflow as tf
from tensorflow import keras

# from models.Bilstm import ModelConfig

from config import GlobalConfig, BASE_PATH
from sklearn.model_selection import train_test_split
import csv
import pickle
import re
import os


def data_load(GlobalConfig):
    """用于读取训练数据和测试数据

    Args:
        train_file ([str]): [训练数据]
        train_file ([str]): [训练数据对应的标签数据]
        test_file ([str]): [测试文件]

    """

    train_data, test_data = [], []
    with open(GlobalConfig.train_path, "rb") as f:
        #    with open("./data/nCoV_100k_train.labled.csv", 'rb') as f:
        for line in f.readlines():
            tmp = remove_redundant_inf(line)
            split_data = next(csv.reader(tmp.splitlines(), skipinitialspace=True))
            #            split_data = tmp.split(',')
            if split_data[6] in ["0", "1", "-1"] and split_data[3] != []:
                train_data.append([split_data[0], split_data[3], split_data[6]])

    with open(GlobalConfig.test_path, "rb") as f:
        #    with open("./data/nCov_10k_test.csv", 'rb') as f:
        for line in f.readlines():
            tmp = remove_redundant_inf(line)
            split_data = next(csv.reader(tmp.splitlines(), skipinitialspace=True))
            test_data.append([split_data[0], split_data[3]])

    return train_data[1:], test_data[1:]


def check_contain_chinese(check_str):
    """
    字符串中是否包含中文字符

    Args:
        check_str ([str]): [一个字符串，判断是否包含中文，进而判断ID的合法性]
    """
    for e in check_str:
        if u"\u4e00" <= e <= u"\u9fff":
            return True
    return False


def remove_html_punc(text):
    """
    去掉文本中的图片链接网页信息

    Args:
        text ([str]): [新闻网页文本]
    """
    # 去掉windown.open开头以 ;" 结尾的英文字段
    tmp = re.sub(r"window.open\(.+;\"", "", text)
    # 去掉 unload=开头，以；"结尾的字段
    tmp = re.sub(r"onload=\".+;\"", "", tmp)
    # 去掉 src=开头，以"结尾的字段
    tmp = re.sub(r"src=\".+\"", "", tmp)
    # 去掉style=开头, 以;"结尾的字段
    tmp = re.sub(r"style=\".+;\"", "", tmp)
    # 去掉>图片开头，jpg结尾的字段
    tmp = re.sub(r"/>图片:.+.jpg", "", tmp)
    # 去掉换行符号”\n“
    tmp = re.sub(r"\n", "", tmp)
    # 去掉空格符号
    tmp = re.sub(r" ", "", tmp)
    # 去掉"=750)"字样
    tmp = re.sub(r"=750\)", "", tmp)
    return tmp


def remove_redundant_inf(text):
    """[去掉冗余信息]

    Args:
        text ([str]): [数据集合上的某一列]

    Returns:
        [str]: [去掉冗余信息后的文本数据]
    """
    tmp = text.decode("gbk", "ignore")
    tmp = re.sub("\r\n", "", tmp)
    tmp = re.sub("//@\w+?:", "", tmp)
    tmp = re.sub("\?{3,}", "?", tmp)
    tmp = re.sub("(网页链接|展开全文)", "", tmp)
    tmp = re.sub("[a-z]{10,}", "", tmp)
    return tmp


if __name__ == "__main__":
    configname = "bert_bilstm"
    myGlobalConfig = GlobalConfig(configname)
    print(myGlobalConfig.train_path)
    train_data, test_data = data_load(myGlobalConfig)
    print(train_data[:5])
    print(len(train_data))
#    word_dict = get_word_dict(train_data, test_data)
#    output = open('word_dict.pkl','wb')
#    pickle.dump(word_dict, output)
#    # train_text_id, _ ,test_text_id, _ = get_text_id(word_dict, train_data, test_data)
#    train_text, train_id = load_model_dataset(GlobalConfig, max_length = ModelConfig.max_len)
#    # train_data, train_label, dev_data, dev_label = get_X_and_Y_data(train_text, train_id)
#    print(train_text[1], train_id[1])
#    # print(train_data)
#    # print(len(train_label), train_label)
#    # print(train_text_id[1])
