# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : word2vec_gensim.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2017-05-11 17:04:35
"""

from gensim.models import word2vec
#from gensim import utils
import multiprocessing

def load_wordVectors(word2vec_path):
    w2vModel = word2vec.Word2Vec.load(word2vec_path)
    return w2vModel

def train_test():

    word2vec_path = "./models/word2Vec.model"
    model2 = load_wordVectors(word2vec_path)
    print(model2.wv.similarity("她", "他"))
    print(model2.wv["我爱你"])
#print(len(model2.wv["。"]))
#    print(model2.wv.vocab.items())
    print(len(model2.wv.vocab.items()))


if __name__ == "__main__":
    train_test()
    # segment_dir='data/THUCNews_segment'
    # out_word2vec_path='models/THUCNews_word2Vec/THUCNews_word2Vec_128.model'
    # # train_THUCNews(segment_dir, out_word2vec_path)
    #
    # w2vModel=load_wordVectors(out_word2vec_path)
    # word1='文化'
    # word2='紧急'
    # vecor1=w2vModel[word1]
    # vecor2=w2vModel[word2]
    # print(w2vModel.wv.similarity('你', '我'))
