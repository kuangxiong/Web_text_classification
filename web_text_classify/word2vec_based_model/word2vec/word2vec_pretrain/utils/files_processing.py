# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : files_processing.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-11-08 19:57:42
"""
import random
import numpy as np
import pandas as pd
import os
import io
import re
import os
import math
from sklearn import preprocessing
from utils import segment
import pickle
import glob


def getFilePathList(file_dir):
    """
    获取file_dir目录下，所有文本路径，包括子目录文件
    :param file_dir:
    :return:
    """
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list


def get_files_list(file_dir, postfix="ALL"):
    """
    获得file_dir目录下，后缀名为postfix所有文件列表，包括子目录
    :param file_dir:
    :param postfix:
    :return:
    """
    postfix = postfix.split(".")[-1]
    file_list = []
    filePath_list = getFilePathList(file_dir)
    if postfix == "ALL":
        file_list = filePath_list
    else:
        for file in filePath_list:
            basename = os.path.basename(file)  # 获得路径下的文件名
            postfix_name = basename.split(".")[-1]
            if postfix_name == postfix:
                file_list.append(file)
    file_list.sort()
    return file_list

def delete_dir_file(dir_path):
    ls = os.listdir(dir_path)
    for i in ls:
        c_path = os.path.join(dir_path, i)
        if os.path.isdir(c_path):
            delete_dir_file(c_path)
        else:
            os.remove(c_path)