import tensorflow as tf
from keras import layers


# this is a simple embedding
class TokenEmbedding(layers.Layer):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True
        )
        self.scale = tf.math.sqrt(tf.cast(embedding_dim, tf.float32))

    def call(self, input):
        return self.embedding(input) * self.scale
