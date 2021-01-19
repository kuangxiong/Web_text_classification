import tensorflow as tf
from tensorflow import keras
import os
#from models import BertBilstm
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint
#from attention import Attention

tmp_base_path = os.path.dirname(__file__)
BASE_PATH = os.path.dirname(tmp_base_path)

class BertWwmModelConfig:
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
    max_len = 200
    learning_rate = 0.001
    hidden_size = 512
    n_classes = 3
    embedding_size = 516
    # BertWwm模型配置路径

#    bert_config_path = os.path.join(BASE_PATH, "data/Chinese_Roberta_wwm_ext/bert_config.json")
    bert_config_path = "../data/chinese_roberta_wwm_ext/bert_config.json"
#    bert_checkpoint_path = os.path.join(BASE_PATH,
#    "data/Chinese_Roberta_wwm_ext/bert_model.ckpt")
    bert_checkpoint_path = "../data/chinese_roberta_wwm_ext/bert_model.ckpt"
# bert_vocab_path = os.path.join(BASE_PATH, "data/Chinese_wwm/vocab.txt")
    bert_vocab_path = "../data/chinese_roberta_wwm_ext/vocab.txt"
    

def bertwwm_bilstm(ModelConfig):
    
    bert_model = load_trained_model_from_checkpoint(ModelConfig.bert_config_path,\
        ModelConfig.bert_checkpoint_path, seq_len=ModelConfig.max_len)

#for l in bert_model.layers:
#        l.trainable = True

    text_id = tf.keras.layers.Input(shape=(ModelConfig.max_len, ), dtype=tf.int32, name='text_id')
    segment_id = tf.keras.layers.Input(shape=(ModelConfig.max_len, ), dtype=tf.int32, name='segment_id')

#    text_input = tf.keras.layers.Input(shape=(2, ModelConfig.max_len), dtype=tf.int32, name="input")
    bert_output = bert_model([text_id, segment_id])
# bert_output = bert_model(text_input)
    bilstm_output = keras.layers.Bidirectional(keras.layers.LSTM(ModelConfig.hidden_size//2, \
			return_sequences=True, dropout=0.2))(bert_output)
#    atten_output = Attention(name="attention_weight")(bilstm_output)
#	dropout = keras.layers.Dropout(ModelConfig.dropout)(atten_output, training=True)

    #keras.layers.Dropout(ModelConfig.dropout)
    output1 = keras.layers.Dense(64, activation='relu')(bilstm_output)
#output1 = keras.layers.Dense(64, activation='relu')(atten_output)
    output2 = keras.layers.Dense(3, activation='softmax')(output1)
    
    model = keras.Model(inputs=[text_id, segment_id], outputs=[output2])

    return model
