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
        # update 1 : after running the model , seems that positional encoding have much larger magitudes then my embedding (this find out by ai , i am too far from this vitions)
        # 391/391 ━━━━━━━━━━━━━━━━━━━━ 56s 140ms/step - accuracy: 0.5101 - loss: 0.6973 - val_accuracy: 0.5000 - val_loss: 0.6915
        # Epoch 2/2
        # 391/391 ━━━━━━━━━━━━━━━━━━━━ 56s 142ms/step - accuracy: 0.5502 - loss: 0.6872 - val_accuracy: 0.7248 - val_loss: 0.6703
        # so it get overwhelm the toekn embedding , i cahnge it from
        # input = self.embedding(input)
        # to
        input = self.embedding(input) * tf.math.sqrt(
            tf.cast(self.embedding_dim, tf.float32)
        )
        # basicly i multiply the embedding number in sqrt of embedding_dim , like sqrt(100) , which not right but it work
        # this will give me only first seq_len element in pos_encoding , like its shape is (1 , max_len ,embedding_dim) for exmaple  (1 , 5000 , 100)
        # if i have sentence the each have 4 token , i get first 4 pos_encoding , add them to embedding and give them to transformer_box
        positions = self.pos_encoding[:, :seq_len, :]
        return input + positions
