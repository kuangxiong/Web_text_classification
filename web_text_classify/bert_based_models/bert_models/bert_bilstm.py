import tensorflow as tf
from tensorflow import keras
import os
#from models import BertBilstm
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint

import sys
sys.path.append("..")
from config import GlobalConfig


class BertBiLSTMConfig:
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
    max_len = 150
    learning_rate = 0.001
    hidden_size = 512
    n_classes = 3
    embedding_size = 516
    # Bert模型配置路径
    global_config = GlobalConfig(model_name)
    bert_config_path = global_config.config_path
    bert_checkpoint_path = global_config.checkpoint_path
    bert_vocab_path = global_config.vocab_path
    

def bert_bilstm(ModelConfig):
    """
    bert-bilstm 模型构建

    Args:
        ModelConfig ([class]): [Bert-Bilstm的模型参数]
    """
    bert_model = load_trained_model_from_checkpoint(ModelConfig.bert_config_path,\
        ModelConfig.bert_checkpoint_path, seq_len=None)

#for l in bert_model.layers:
#        l.trainable = True

    text_id = tf.keras.layers.Input(shape=(ModelConfig.max_len, ), dtype=tf.int32, name='text_id')
    segment_id = tf.keras.layers.Input(shape=(ModelConfig.max_len, ), dtype=tf.int32, name='segment_id')

#    text_input = tf.keras.layers.Input(shape=(2, ModelConfig.max_len), dtype=tf.int32, name="input")
    bert_output = bert_model([text_id, segment_id])
# bert_output = bert_model(text_input)
    bilstm_output = keras.layers.Bidirectional(keras.layers.LSTM(ModelConfig.hidden_size//2))(bert_output)
    dropout = keras.layers.Dropout(ModelConfig.dropout)(bilstm_output, training=True)

    #keras.layers.Dropout(ModelConfig.dropout)
    output1 = keras.layers.Dense(64, activation='relu')(dropout)
    output2 = keras.layers.Dense(3, activation='softmax')(output1)
    
    model = keras.Model(inputs=[text_id, segment_id], outputs=[output2])

    return model
