import tensorflow as tf
from keras import layers
from EncoderBlock import EncoderBlock


class EncoderStack(layers.Layer):
    def __init__(
        self, num_layers, embedding_dim, num_heads, feed_forward_dim, rate=0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.encoder_stack = [
            EncoderBlock(embedding_dim, num_heads, feed_forward_dim, rate)
            for _ in range(num_layers)
        ]

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=True, padding_mask=None):
        x = inputs
        for layer in self.encoder_stack:
            x = layer(x, training=training, padding_mask=padding_mask)
        return x
