from embeding_positionaleEncoding import preprocessing
from transformerBlock import TransformerBlock
import keras
from keras.preprocessing import sequence
from keras import layers


VOCAB_SIZE = 20000
MAX_LEN = 200
EMBEDDING_DIM = 64
NUM_HEADS = 2
FF_DIM = 64

(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_val = sequence.pad_sequences(x_val, maxlen=MAX_LEN)

print("x_train shape:", x_train.shape)
print("x_val shape:", x_val.shape)
inputs = keras.Input(shape=(MAX_LEN,))

x = preprocessing(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN)(
    inputs
)
x = TransformerBlock(EMBEDDING_DIM, NUM_HEADS, FF_DIM)(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()
history = model.fit(
    x_train, y_train, batch_size=64, epochs=2, validation_data=(x_val, y_val)
)
