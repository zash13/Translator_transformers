import keras
from EncoderStack import EncoderStack
from DecoderStack import DecoderStack
from TokenEmbedding import TokenEmbedding
from PositionalEncoding import positional_encoding
import tensorflow as tf


class Translator(keras.Model):
    def __init__(
        self,
        num_layers,
        embedding_dim,
        num_heads,
        feed_forward_dim,
        input_vocab_size,
        target_vocab_size,
        max_sequence_length,
        pad_id,
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
        self.final_layer = keras.layers.Dense(target_vocab_size)
        self.pad_id = pad_id
        self.max_sequence_length = max_sequence_length

    def build(self, input_shape):
        # input_shape: [(batch, enc_seq_len), (batch, dec_seq_len)]
        # This model performs both training and prediction using batches, which is faster as you know.
        # That's why I accept the input in this format.
        enc_input_shape, dec_input_shape = input_shape
        self.encoder_embedding.build(enc_input_shape)
        self.decoder_embedding.build(dec_input_shape)
        self.encoder.build((None, None, self.encoder_embedding.embedding_dim))
        self.decoder.build((None, None, self.decoder_embedding.embedding_dim))
        self.final_layer.build((None, None, self.decoder_embedding.embedding_dim))
        self.built = True

    def call(self, inputs, training=True):
        enc_inputs, dec_inputs = inputs  # input = [encoder_input, decoder_input]

        enc_padding_mask = self.create_padding_mask(enc_inputs)
        dec_padding_mask = self.create_padding_mask(dec_inputs)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(dec_inputs)[1])
        dec_self_mask = tf.maximum(dec_padding_mask, look_ahead_mask)

        enc_emb = self.encoder_embedding(enc_inputs)
        enc_emb += self.positional_encoding[:, : tf.shape(enc_emb)[1], :]
        enc_output = self.encoder(
            enc_emb, training=training, padding_mask=enc_padding_mask
        )

        # very important: providing this variable helps the model understand its current position in the sequence.
        # it gives the model awareness of the past, how far it has progressed, and help it with next.
        dec_emb = self.decoder_embedding(dec_inputs)
        dec_emb += self.positional_encoding[:, : tf.shape(dec_emb)[1], :]
        dec_output = self.decoder(
            dec_emb,
            enc_output,
            training=training,
            look_ahead_mask=dec_self_mask,
            padding_mask=enc_padding_mask,
        )

        final_output = self.final_layer(dec_output)
        return final_output

    def create_padding_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq, self.pad_id), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def generate(self, inputs, max_sequence_length, start_token, end_token):
        batch_size = tf.shape(inputs)[0]

        # Encode once
        enc_padding_mask = self.create_padding_mask(inputs)
        enc_emb = self.encoder_embedding(inputs)
        enc_emb += self.positional_encoding[:, : tf.shape(enc_emb)[1], :]
        enc_output = self.encoder(
            enc_emb, training=False, padding_mask=enc_padding_mask
        )

        # Initialize with <BOS>
        decoded = tf.fill([batch_size, 1], start_token)
        finished = tf.zeros([batch_size], dtype=tf.bool)

        for _ in tf.range(max_sequence_length - 1):
            look_ahead_mask = self.create_look_ahead_mask(tf.shape(decoded)[1])
            dec_emb = self.decoder_embedding(decoded)
            dec_emb += self.positional_encoding[:, : tf.shape(dec_emb)[1], :]
            dec_output = self.decoder(
                dec_emb,
                enc_output,
                training=False,
                look_ahead_mask=look_ahead_mask,
                padding_mask=enc_padding_mask,
            )

            # Take last step logits
            last_token_output = dec_output[:, -1, :]
            logits = self.final_layer(last_token_output)
            predicted_ids = tf.argmax(logits, axis=-1, output_type=tf.int32)

            predicted_ids = tf.expand_dims(predicted_ids, 1)  # (batch, 1)
            decoded = tf.concat([decoded, predicted_ids], axis=1)

            # Mark finished sequences
            finished = finished | tf.equal(predicted_ids[:, 0], end_token)
            if tf.reduce_all(finished):
                break

        return decoded

    def generate2(self, inputs, max_sequence_length, start_token, end_token):
        """
        GREADY DECODING LOOP FOR PREDICTIONS
            args :
                inputs = encoder outputs
                max_sequence_length = maximum length of the generated sequence ( incude the start and end and stuff )
                start_token = integer id of the start of seq , like 1 for <sos>
                end_token = integer id of the end of seq , like 2 for <eos>

            Returns:
                Generated sequence tensor of shape (batch, max_len).
        """
        # here's what happens: first, we calculate all the encoder outputs.
        # then, we iterate over the sequence length to perform decoding.
        # for example, if we have 10 sentences, each with length 30, we start from position 0 to 29 (30 - 1 , which is start ) .
        # at each step, we decode the current token for all 10 sentences simultaneously.
        # it's like moving vertically through the batch instead of horizontally (sentence by sentence).
        #
        #
        # calculate encoder outputs
        batch_size = tf.shape(inputs)[0]
        enc_padding_mask = self.create_padding_mask(inputs)
        enc_emb = self.encoder_embedding(inputs)
        enc_emb += self.positional_encoding[:, : tf.shape(enc_emb)[1], :]
        enc_output = self.encoder(
            enc_emb, training=False, padding_mask=enc_padding_mask
        )
        # initialize decoded sequence with start token like bunch of tensors with start token [[1] , [1] , ...]
        decoded = tf.ones((batch_size, 1), dtype=tf.int32) * start_token
        finished = tf.zeros((batch_size,), dtype=tf.bool)

        for i in tf.range(max_sequence_length - 1):
            dec_padding_mask = self.create_padding_mask(decoded)
            look_ahead_mask = self.create_look_ahead_mask(tf.shape(decoded)[1])
            dec_self_mask = tf.maximum(dec_padding_mask, look_ahead_mask)
            dec_emb = self.decoder_embedding(decoded)
            dec_emb += self.positional_encoding[:, : tf.shape(dec_emb)[1], :]
            dec_output = self.decoder(
                dec_emb,
                enc_output,
                training=False,
                look_ahead_mask=dec_self_mask,
                padding_mask=enc_padding_mask,
            )
            # w
            last_token_output = dec_output[:, -1, :]
            logits = self.final_layer(last_token_output)
            # the use of argmax here is incorrect. there should be a parameter like π or α that represents
            # the probability or rate of choosing either the argmax or a lower-level alternative.
            # sometimes, there are other words similar to the argmax choice, but the model avoids selecting them.
            # the model can get stuck on a word due to poor training data or a bad input sequence,
            # or for other reasons.
            predicted_ids = tf.argmax(logits, axis=-1, output_type=tf.int32)
            predicted_ids = tf.expand_dims(predicted_ids, 1)  # (batch_size, 1)

            predicted_ids = tf.where(finished[:, tf.newaxis], 0, predicted_ids)
            decoded = tf.concat([decoded, predicted_ids], axis=1)

            finished = finished | (predicted_ids[:, 0] == end_token)
            if tf.reduce_all(finished):
                break
        return decoded
