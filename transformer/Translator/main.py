import numpy as np
import pandas as pd
import tensorflow as tf
from TranslatorModel import Translator


def test():
    num_layers = 2
    embedding_dim = 128
    num_heads = 4
    feed_forward_dim = 512
    input_vocab_size = 1000
    target_vocab_size = 1000
    max_sequence_length = 20
    batch_size = 64
    seq_len = 10

    enc_inputs = tf.random.uniform(
        (batch_size, seq_len), maxval=input_vocab_size, dtype=tf.int32
    )
    dec_inputs = tf.random.uniform(
        (batch_size, seq_len), maxval=target_vocab_size, dtype=tf.int32
    )

    model = Translator(
        num_layers,
        embedding_dim,
        num_heads,
        feed_forward_dim,
        input_vocab_size,
        target_vocab_size,
        max_sequence_length,
    )

    output = model([enc_inputs, dec_inputs])
    print("Input shapes:", enc_inputs.shape, dec_inputs.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test()
