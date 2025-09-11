import numpy as np
import tensorflow as tf
from TranslatorModel import Translator
from TokenEmbedding import TokenEmbedding
from EncoderStack import EncoderStack
from DecoderStack import DecoderStack


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

    print("Testing layers eagerly...")
    test_emb = TokenEmbedding(input_vocab_size, embedding_dim)
    test_enc = EncoderStack(num_layers, embedding_dim, num_heads, feed_forward_dim)
    test_dec = DecoderStack(num_layers, embedding_dim, num_heads, feed_forward_dim)

    # Test TokenEmbedding
    x = np.random.randint(0, input_vocab_size, (3, 4))
    emb_output = test_emb(x)
    print("TokenEmbedding output shape:", emb_output.shape)  # Should be (3, 4, 128)

    # Test EncoderStack
    enc_input = np.random.random((3, 4, embedding_dim))
    enc_output = test_enc(enc_input)
    print("EncoderStack output shape:", enc_output.shape)  # Should be (3, 4, 128)

    # Test DecoderStack
    dec_input = np.random.random((3, 4, embedding_dim))
    dec_output = test_dec(dec_input, enc_output)
    print("DecoderStack output shape:", dec_output.shape)  # Should be (3, 4, 128)

    # Test full model
    print("Testing full model...")
    model = Translator(
        num_layers,
        embedding_dim,
        num_heads,
        feed_forward_dim,
        input_vocab_size,
        target_vocab_size,
        max_sequence_length,
    )
    model.build([(None, seq_len), (None, seq_len)])
    output = model([enc_inputs, dec_inputs], training=True)
    print("Input shapes:", enc_inputs.shape, dec_inputs.shape)
    print("Output shape:", output.shape)  # Should be (64, 10, 1000)


if __name__ == "__main__":
    test()
