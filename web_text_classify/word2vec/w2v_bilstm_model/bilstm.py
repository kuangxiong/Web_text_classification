import tensorflow as tf
from tensorflow import keras

class W2VBiLSTMConfig:
    """

    Args:
        object ([type]): [description]
    """
    embedding_matrix = np.load("w2v_matrix.npy")
    word2id = pickle.load("word2map.pkl")
    word_dict_num = len(word2id)  # 字典的字数量
    embed_dim = len(embedding_matrix[0])
    model_name = "W2VBiLSTM"
    data_path="data"
    embsize=256
    dropout=0.5
    n_vocab= 10000
    num_epochs = 10
    batch_size =128 
    max_len = 100 
    learning_rate = 0.001
    hidden_size = 512
    n_classes = 3

def w2vbilstm(ModelConfig):
    """
    BiLSTM 模型构造

    Args:
        ModelConfig ([class]): [模型配置参数]

    Returns:
        [type]: [BiLSTM模型]
    """

    model = keras.models.Sequential()
    # model.add(keras.layers.Input(ModelConfig.max_len,))
    model.add(keras.layers.Embedding(
                ModelConfig.n_vocab, 
                ModelConfig.embsize
                weights = [ModelConfig.embeddings_matrix],
                trainable = True
                ))
    # model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(ModelConfig.hidden_size//2)))
    # model.add(keras.layers.Bidirectional(keras.layers.LSTM(ModelConfig.hidden_size//2, return_sequences=True)))
    model.add(keras.layers.Dropout(ModelConfig.dropout))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    return model

if __name__=="__main__":
    model = BiLSTM(BiLSTMConfig)
    print(model.summary())
   # print(BASE_PATH)
