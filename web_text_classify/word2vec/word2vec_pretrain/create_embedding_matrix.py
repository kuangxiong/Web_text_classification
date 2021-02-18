# -*-coding: utf-8 -*-

import numpy as np
from gensim.models import word2vec
import pickle

def load_wordVectors(word2vec_path):
    w2vModel = word2vec.Word2Vec.load(word2vec_path)
    return w2vModel

def create_embedding_matrix(word2vec_path):
    """
        构造嵌入矩阵和索引
    """
    model = load_wordVectors(word2vec_path)
    word2id ={"PAD":0}
    vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
    embedding_matrix = np.zeros((len(model.wv.vocab.items()) + 1,
            model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2id[word] = i + 1
        embedding_matrix[i + 1] = vocab_list[i][1]
    return word2id, embedding_matrix

def train_test():

    word2vec_path = "./models/word2Vec.model"
    model2 = load_wordVectors(word2vec_path)
    print(model2.wv.similarity("她", "他"))
    print(model2.wv["我爱你"])
    print(model2.wv.vocab.items())
    print(len(model2.wv.vocab.items()))


if __name__ == "__main__":
    word2id, matrix = create_embedding_matrix("./models/word2Vec.model")
    file = open("word2map.pkl", "wb")
    pickle.dump(word2id, file)
    np.save("w2v_matrix.npy", matrix)
    val = word2id["城市"]
    print(matrix[val])
    print(word2id["城市"])
    
#    train_test()
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
