import keras as ks
import numpy as np
import pandas as pd
from TokenEmbedding import TokenEmbedding
from PositionalEncoding import positional_encoding
from EncoderStack import EncoderStack
from DecoderStack import DecoderStack


class Translator(ks.Model):
    def __init__(
        self,
        num_layers,
        embedding_dim,
        num_heads,
        feed_forward_dim,
        input_vocab_size,
        target_vocab_size,
        max_sequence_length,
        rate=0.1,
    ) -> None:
        super().__init__()
        self.encoder_embedding = TokenEmbedding(input_vocab_size, embedding_dim)
        self.decoder_embedding = TokenEmbedding(target_vocab_size, embedding_dim)
        self.positional_encoding = positional_encoding(
            max_sequence_length, embedding_dim
        )
        self.encoder = EncoderStack(
            num_layers, embedding_dim, num_heads, feed_forward_dim, rate
        )
        self.decoder = DecoderStack(
            num_layers, embedding_dim, num_heads, feed_forward_dim, rate
        )
        self.final_layer = ks.layers.Dense(target_vocab_size)

    def call(self, inputs, training=True):
        enc_inputs, dec_inputs = inputs  # Expect [encoder_input, decoder_input]

        # Create masks
        enc_padding_mask = self.create_padding_mask(enc_inputs)
        dec_padding_mask = self.create_padding_mask(dec_inputs)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(dec_inputs)[1])
        # Combine look-ahead and padding mask for decoder self-attention
        dec_self_mask = tf.maximum(dec_padding_mask, look_ahead_mask)

        # Encoder path: embed + positional encoding
        enc_emb = self.encoder_embedding(enc_inputs)
        enc_emb += self.positional_encoding[:, : tf.shape(enc_emb)[1], :]
        enc_output = self.encoder(enc_emb, enc_padding_mask)

        # Decoder path: embed + positional encoding
        dec_emb = self.decoder_embedding(dec_inputs)
        dec_emb += self.positional_encoding[:, : tf.shape(dec_emb)[1], :]
        dec_output = self.decoder(dec_emb, enc_output, dec_self_mask, enc_padding_mask)

        # Final output
        final_output = self.final_layer(dec_output)
        return final_output
