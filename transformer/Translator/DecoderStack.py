import tensorflow as tf
from keras import layers
from DecoderBlock import DecoderBlock


class DecoderStack(layers.Layer):
    def __init__(
        self, num_layers, embedding_dim, num_heads, feed_forward_dim, rate=0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.decoder_stack = [
            DecoderBlock(embedding_dim, num_heads, feed_forward_dim, rate)
            for _ in range(num_layers)
        ]

    def call(self, inputs, enc_output, look_ahead_mask=None, padding_mask=None):
        x = inputs
        for layer in self.decoder_layers:
            x = layer(x, enc_output, look_ahead_mask, padding_mask)
        return x
