from BertModel import BertModel
from BertPrepareData import bert_data
from sklearn.model_selection import train_test_split
import keras

VOCAB_SIZE = 30000
MAX_LEN = 512
EMBEDDING_DIM = 128
NUM_HEADS = 8
FF_DIM = 512
NUM_LAYERS = 4
DROPOUT_RATE = 0.1
OUTPUT_DIM = 2

x, segments, labels, VOCAB_SIZE = bert_data(max_len=MAX_LEN)

x_train, x_test, seg_train, seg_test, y_train, y_test = train_test_split(
    x, segments, labels, test_size=0.2, random_state=42
)

model = BertModel(
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    output_dim=OUTPUT_DIM,
    num_layers=NUM_LAYERS,
    dropout_rate=DROPOUT_RATE,
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.build([(None, MAX_LEN), (None, MAX_LEN)])
model.summary()

model.fit(
    [x_train, seg_train],
    y_train,
    validation_data=([x_test, seg_test], y_test),
    batch_size=32,
    epochs=3,
)
