import tensorflow as tf
from tensorflow import keras
import os
BASE_PATH = os.path.dirname(os.path.dirname(__file__))
print(BASE_PATH)


class ModelConfig:
    """

    Args:
        object ([type]): [description]
    """
    word_dict_num = 10000  # 字典的字数量
    embed_dim = 512
    model_name = "BiLSTM"
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

def BiLSTM(ModelConfig):

    model = keras.models.Sequential()
    # model.add(keras.layers.Input(ModelConfig.max_len,))
    model.add(keras.layers.Embedding(ModelConfig.n_vocab, ModelConfig.embsize))
    # model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(ModelConfig.hidden_size//2)))
    # model.add(keras.layers.Bidirectional(keras.layers.LSTM(ModelConfig.hidden_size//2, return_sequences=True)))
    model.add(keras.layers.Dropout(ModelConfig.dropout))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    return model
# model = BiLSTM(ModelConfig)
# print(model.summary())
# print(BASE_PATH)
