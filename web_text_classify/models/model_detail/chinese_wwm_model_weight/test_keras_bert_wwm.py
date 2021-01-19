#-*- coding:utf-8 -*-
''' ---------------------------------------------------------
 File Name: test_keras_bert.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Sun Nov  1 00:12:02 2020
---------------------------------------------------------
'''
import os

# 设置预训练模型的路径
pretrained_path = 'chinese_wwm_ext_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

# 构建字典
# 也可以用 keras_bert 中的 load_vocabulary() 函数
# 传入 vocab_path 即可
# from keras_bert import load_vocabulary
# token_dict = load_vocabulary(vocab_path)
import numpy as np
import codecs
from keras_bert import Tokenizer
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

# 加载预训练模型
from keras_bert import load_trained_model_from_checkpoint
bert_layer = load_trained_model_from_checkpoint(config_path, checkpoint_path)

# 设置最后一层是可训练的
#bert_layer.layers[-1].trainable=True
#bert_layer.layers[-2].trainable=True
#bert_layer.layers[-3].trainable=True
#bert_layer.layers[-4].trainable=True
#bert_layer.layers[-5].trainable=True
#for layer in bert_layer.layers:
#    layer.trainable=True 
#print(bert_layer.layers)
bert_layer.summary()
# Tokenization

res = []
res_ind=[]
tokenizer = Tokenizer(token_dict)
text = '我爱读书'
tokens = tokenizer.tokenize(text)
indices, segments = tokenizer.encode(first=text, max_len=10)
print(indices, segments)
predicts = bert_layer([np.array([indices]),np.array(segments)])
print(predicts)
res.append(indices)
res_ind.append(segments)

text = '我爱学习'
tokens = tokenizer.tokenize(text)
indices, segments = tokenizer.encode(first=text, max_len=10)
predicts = bert_layer([np.array([indices]),np.array(segments)])
#print(predicts)
res.append(indices)
res_ind.append(segments)


#print(indices[:10])
#print(segments[:10])

# 提取特征

predicts = bert_layer([np.array(res),np.array(res_ind)])
#print(predicts)
# print(len(bert_layer.layers))

