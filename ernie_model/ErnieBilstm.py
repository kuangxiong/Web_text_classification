import tensorflow as tf
from tensorflow import keras
import os
from keras_bert import Tokenizer
from keras_ernie import load_trained_model_from_checkpoint

tmp_base_path = os.path.dirname(__file__)
BASE_PATH = os.path.dirname(tmp_base_path)
class ErnieModelConfig:
    """ 
    Args:
        object ([type]): [description]
    """
    model_name = "ErnieBiLSTM"
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

    ernie_config_path = os.path.join(BASE_PATH,
            "data/ERNIE_stable-1.0.1/ERNIE_tensor/bert_config.json")
    ernie_checkpoint_path = os.path.join(BASE_PATH, "data/ERNIE_stable-1.0.1/ERNIE_tensor/bert_model.ckpt")
    ernie_vocab_path = os.path.join(BASE_PATH,
            "data/ERNIE_stable-1.0.1/ERNIE_tensor/vocab.txt")
    

def ernie_bilstm(ModelConfig):
    
    bert_model = load_trained_model_from_checkpoint(ModelConfig.ernie_config_path,\
        ModelConfig.ernie_checkpoint_path, seq_len=None)

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
