import tensorflow as tf
import keras
from keras import layers
import numpy as np


# this code was implemented based on this article : https://www.datacamp.com/tutorial/how-transformers-work
# if you check the decoder block image, you'll understand everything!
# attentiosn layers are designed like htis : https://jalammar.github.io/illustrated-transformer
class DecoderBlock(layers.Layer):
    def __init__(self, embeding_dim, num_heads, feadForward_dim, rate=0.1):
        super().__init__()
        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embeding_dim
        )
        self.encoder_decoder_attentions = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embeding_dim
        )

        self.feedforward_network_0 = keras.Sequential(
            [
                layers.Dense(feadForward_dim, activation="relu"),
                layers.Dense(embeding_dim),
            ]
        )
        self.layernorm_0 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_0 = layers.Dropout(rate)
        self.dropout_1 = layers.Dropout(rate)
        self.dropout_2 = layers.Dropout(rate)

    def call(self, inputs, enc_output, look_agead_mask=None, padding_mask=None):
        # i added this mask based on gpt's suggestion. it seems that when the attention layer learns its parameters,
        # it sometimes looks ahead into future inputs, which is like cheating. this mask prevents it from accessing positions like t+1.
        attention_output_0 = self.self_attention(
            query=inputs, value=inputs, key=inputs, attention_mask=look_agead_mask
        )
        attention_output_0 = self.dropout_0(attention_output_0)
        out_0 = self.layernorm_0(inputs + attention_output_0)

        attention_output_1 = self.encoder_decoder_attentions(
            query=out_0, value=enc_output, key=enc_output, attention_mask=padding_mask
        )
        attention_output_1 = self.dropout_1(attention_output_1)
        out_1 = self.layernorm_1(out_0 + attention_output_1)

        ffn_output = self.feedforward_network_0(out_1)
        ffn_output = self.dropout_2(ffn_output)
        return self.layernorm_2(out_1 + ffn_output)
