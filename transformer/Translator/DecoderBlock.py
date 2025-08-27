import tensorflow as tf
import keras
from keras import layers
import numpy as np


class TransformerBlock(layers.Layer):
    def __init__(self, embeding_dim, num_heads, feadForward_dim, rate=0.1):
        super().__init__()

        self.attention_layer = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embeding_dim
        )

        self.feedforward_network = keras.Sequential(
            [
                layers.Dense(feadForward_dim, activation="relu"),
                layers.Dense(embeding_dim),
            ]
        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attention_output = self.attention_layer(query=inputs, value=inputs)
        attention_output = self.dropout1(attention_output)

        out1 = self.layernorm1(inputs + attention_output)
        fnn_output = self.feedforward_network(out1)
        fnn_output = self.dropout2(fnn_output)
        return self.layernorm2(out1 + fnn_output)
