import tensorflow as tf
from tensorflow import keras
import os
from models import BertBilstm
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint

tmp_base_path = os.path.dirname(__file__)
BASE_PATH = os.path.dirname(tmp_base_path)

class SougouModelConfig:
    """ 
    Args:
        object ([type]): [description]
    """
    model_name = "BertBiLSTM"
    data_path="data"
    dropout=0.5
    n_vocab= 10000
    num_epochs = 10
    batch_size =128 
    max_len = 100   
    learning_rate = 0.001
    hidden_size = 512
    n_classes = 3
    embedding_size = 100 
    # Bert模型配置路径
    bert_config_path = os.path.join(BASE_PATH, "data/bert_source/Chinese_L-12_H-768_A-12/bert_config.json")
    bert_checkpoint_path = os.path.join(BASE_PATH, "data/bert_source/Chinese_L-12_H-768_A-12/bert_model.ckpt")
    bert_vocab_path = os.path.join(BASE_PATH, "data/bert_source/Chinese_L-12_H-768_A-12/vocab.txt")
    

def SougouBiLSTM(ModelConfig):
    
    input_txt = tf.keras.layers.Input(shape=(ModelConfig.max_len,
                ModelConfig.embedding_size), dtype=tf.float32, name='input_txt')

    bilstm_output = keras.layers.Bidirectional(keras.layers.LSTM(ModelConfig.hidden_size//2))(input_txt)
    dropout = keras.layers.Dropout(ModelConfig.dropout)(bilstm_output, training=True)

    #keras.layers.Dropout(ModelConfig.dropout)
    output1 = keras.layers.Dense(64, activation='relu')(dropout)
    output2 = keras.layers.Dense(3, activation='softmax')(output1)
    
    model = keras.Model(inputs=[input_txt], outputs=[output2])

    return model
