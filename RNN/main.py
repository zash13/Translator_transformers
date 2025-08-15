import tensorflow as tf
from tensorflow import keras
import numpy as np
import re
from collections import defaultdict

SEQUENCE_LEN = 20
EMBEDDING_DIM = 100
RNN_LAYER_DIM = 100
EPOCHS = 100
BATCH_SIZE = 8


class RnnModel:
    def __init__(self, seq_len, vocab_size, h1_dim, embedding_dim) -> None:
        self.vocab_size = vocab_size
        self._model = self._build_model(seq_len, vocab_size, h1_dim, embedding_dim)

    def _build_model(self, seq_len, vocab_size, h1_dim, embedding_dim):
        input = keras.Input(shape=(seq_len,), dtype="int32")
        embedding = keras.layers.Embedding(vocab_size, embedding_dim)(input)
        rnn_layer = keras.layers.SimpleRNN(h1_dim)(embedding)
        output = keras.layers.Dense(vocab_size, activation="softmax")(rnn_layer)
        model = keras.Model(inputs=input, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def forward(self, input):
        return self._model.predict(input)

    def fit(self, input, target, batch_size=16, epochs=10):
        history = self._model.fit(
            input, target, batch_size=batch_size, epochs=epochs, verbose=1
        )
        return history.history

    def get_weights(self):
        return self._model.get_weights()

    def set_weights(self, weights):
        self._model.set_weights(weights)


data = """ The meeting of Diogenes of Sinope and Alexander the Great is one of the most discussed anecdotes from philosophical history. Many versions of it exist. The most popular relate it as evidence of Diogenes """
tokens = []
for sentence in data:
    for word in sentence.split():
        clean = re.sub(r"[,\!\?\(\):]", "", word.lower())
        if clean:
            tokens.append(clean)

special_tokens = {
    "<#START>": 0,
    "<#PAD>": 1,
    "<#UNKNOWN>": 2,
    "<#END>": 3,
}
vocab = sorted(set(tokens))
vocab_size = len(vocab) + len(special_tokens)
start_index = len(special_tokens)
word2idx = {word: idx + start_index for idx, word in enumerate(vocab)}
word2idx = {**special_tokens, **word2idx}
idx2word = {idx: word for word, idx in word2idx.items()}

token_ids = [word2idx.get(word, special_tokens["<#UNKNOWN>"]) for word in tokens]

X, y = [], []
for i in range(len(token_ids) - SEQUENCE_LEN):
    X.append(token_ids[i : i + SEQUENCE_LEN])
    y.append(token_ids[i + SEQUENCE_LEN])

X = np.array(X)
y = np.array(y)
rnn_model = RnnModel(SEQUENCE_LEN, vocab_size, RNN_LAYER_DIM, EMBEDDING_DIM)
rnn_model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
