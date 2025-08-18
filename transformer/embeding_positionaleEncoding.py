from positionalEncoding import positional_encoding
import keras
from keras import layers
import tensorflow as tf


class preprocessing(layers.Layer):
    def __init__(self, vocab_size, embedding_dim, max_len=5000):
        super().__init__()
        self.embedding = keras.layers.Embedding(
            vocab_size, embedding_dim, name="embedding"
        )
        self.embedding_dim = embedding_dim
        self.pos_encoding = positional_encoding(max_len, embedding_dim)

    def call(self, input):
        seq_len = tf.shape(input)[1]
        input = self.embedding(input)
        # this will give me only first seq_len element in pos_encoding , like its shape is (1 , max_len ,embedding_dim) for exmaple  (1 , 5000 , 100)
        # if i have sentence the each have 4 token , i get first 4 pos_encoding , add them to embedding and give them to transformer_box
        positions = self.pos_encoding[:, :seq_len, :]
        return input + positions
