# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : segment.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2017-05-11 17:51:53
"""

##
import jieba
import os
import io
import math
import re
from utils import files_processing

"""
read() 每次读取整个文件，它通常将读取到底文件内容放到一个字符串变量中，也就是说 .read() 生成文件内容是一个字符串类型。
readline()每只读取文件的一行，通常也是读取到的一行内容放到一个字符串变量中，返回str类型。
readlines()每次按行读取整个文件内容，将读取到的内容放到一个列表中，返回list类型。
"""


def load_stopWords(path):
    """
    加载停用词
    :param path:
    :return:
    """
    stopwords = []
    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


def common_stopwords():
    """
    常用的停用词
    :return:
    """
    Stopwords = []
    # Stopwords=['\n','',' ','  ','\n\n']
    Stopwords = ["\n", "", " ", "\n\n"]
    return Stopwords

def batch_processing_files(files_list, segment_out_dir, batchSize, stopwords=[]):
    """
    批量分割文件字词，并将batchSize的文件合并一个文件
    :param files_list: 文件列表
    :param segment_out_dir: 字符分割文件输出的目录
    :param batchSize:
    :param stopwords: 停用词
    :return:
    """
    if not os.path.exists(segment_out_dir):
        os.makedirs(segment_out_dir)
    files_processing.delete_dir_file(segment_out_dir)

    sample_num = len(files_list)
    batchNum = int(math.ceil(1.0 * sample_num / batchSize))
    for i in range(batchNum):
        segment_out_name = os.path.join(segment_out_dir, "segment_{}.txt".format(i))
        start = i * batchSize
        end = min((i + 1) * batchSize, sample_num)
        batch_files = files_list[start:end]
        content_list = segment_files_list(batch_files, stopwords, segment_type="word")
        # content_list=padding_sentences(content_list, padding_token='<PAD>', padding_sentence_length=15)
        save_content_list(segment_out_name, content_list, mode="ab")
        print("segment files:{}".format(segment_out_name))

