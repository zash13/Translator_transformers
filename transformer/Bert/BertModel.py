import keras
from keras import layers
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences

from keras.datasets import imdb

from transformerBlock import TransformerBlock
from BertEmbedding import preprocessing


class BertModel(keras.Model):
    def __init__(
        self,
        vocab_size,
        max_len,
        embedding_dim,
        num_heads,
        ff_dim,
        output_dim,
        num_layers,
        dropout_rate,
    ):
        super().__init__()

        self.preprocessing = preprocessing(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
        )

        self.transformerBlocks = [
            TransformerBlock(embedding_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]

        self.output_dense = layers.Dense(embedding_dim, activation="tanh")
        self.classifier = layers.Dense(output_dim, activation="softmax")

    def call(self, inputs, training=False):
        input_ids, token_type_ids = inputs

        x = self.preprocessing([input_ids, token_type_ids])

        for block in self.transformerBlocks:
            x = block(x)

        cls_output = x[:, 0, :]
        x = self.output_dense(cls_output)
        return self.classifier(x)
