import codecs
import numpy as np
import pickle

vocab = []
word2id={}
pretrained={}
flag = 0
for i, line in enumerate(codecs.open("wiki_100.utf8", 'r', encoding='utf-8')):
    if flag==0:
        flag+=1
        continue
    line = line.rstrip().split()
    word2id[line[0]] = flag
    pretrained[flag] = list(map(float, line[1:]))
    flag+=1
word2id['UNK'] = flag
np.random.seed(10)
pretrained[flag] = list(np.random.rand(100))  
# pretrained[flag] = pretrained[1]  


pretrained[0] = [0]*100
for i in range(1000):
    print(len(pretrained[i]))
emb_file = open("wiki_embedding.pkl",'wb')
word2id_file = open("word2id.pkl",'wb')
pickle.dump(word2id, word2id_file)
pickle.dump(pretrained, emb_file)
word2id_file.close()
emb_file.close()

emb_file = open("wiki_embedding.pkl",'rb')
#word2id_file = open("word2id.pkl","rb")
data = pickle.load(emb_file)
for i in range(100):
    print(len(data[i]))
#print(data)
