# I wrote this code with help from https://tinkerd.net/blog/machine-learning/bert-embeddings/
# It's a fantastic article that explains embeddings really well.
# ------------------------
#
from PositionalEncoding import positional_encoding
import keras
from keras import layers
import tensorflow as tf


class preprocessing(layers.Layer):
    def __init__(self, vocab_size, embedding_dim, max_len=5000):
        super().__init__()
        self.tok_embedding = keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            name="token_embedding",
        )
        self.seg_embedding = keras.layers.Embedding(
            2, embedding_dim, name="segment_embedding"
        )
        self.pos_encoding = positional_encoding(max_len, embedding_dim)
        self.embedding_dim = embedding_dim

    def call(self, input, scale_factor=10):
        token_ids, segment_ids = input
        seq_len = tf.shape(token_ids)[1]
        # i change this to orginal implemntaion in attention is all you need , which is sqrt of embedding_dim
        token_embedding = self.tok_embedding(token_ids)
        token_embedding *= tf.sqrt(tf.cast(self.embedding_dim, tf.float32))
        segment_embedding = self.seg_embedding(segment_ids)
        # dont ask me about this part
        positions = self.pos_encoding[:, :seq_len, :]
        position_embeddings = tf.tile(positions, [tf.shape(token_ids)[0], 1, 1])

        return token_embedding + segment_embedding + position_embeddings
