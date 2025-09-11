import tensorflow as tf
import keras as ks
from TokenEmbedding import TokenEmbedding
from EncoderStack import EncoderStack
from DecoderStack import DecoderStack
from PositionalEncoding import positional_encoding


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
    ):
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
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def build(self, input_shape):
        # input_shape: [(batch, enc_seq_len), (batch, dec_seq_len)]
        enc_input_shape, dec_input_shape = input_shape
        self.encoder_embedding.build(enc_input_shape)
        self.decoder_embedding.build(dec_input_shape)
        self.encoder.build((None, None, self.encoder_embedding.embedding_dim))
        self.decoder.build((None, None, self.decoder_embedding.embedding_dim))
        self.final_layer.build((None, None, self.decoder_embedding.embedding_dim))
        self.built = True

    def call(self, inputs, training=True):
        enc_inputs, dec_inputs = inputs
        enc_emb = self.encoder_embedding(enc_inputs)
        enc_emb += self.positional_encoding[:, : tf.shape(enc_emb)[1], :]
        enc_output = self.encoder(enc_emb)

        dec_emb = self.decoder_embedding(dec_inputs)
        dec_emb += self.positional_encoding[:, : tf.shape(dec_emb)[1], :]
        dec_output = self.decoder(dec_emb, enc_output)

        final_output = self.final_layer(dec_output)
        return final_output
