# import tensorflow_hub as hub
# import bert.tokenization.FullTokenizer as tokenizer

# I'd like to use TensorFlow libraries here,
# but my internet speed is only around 0.3 Mbps,
# which means it takes over an hour to download both TensorFlow Hub and the tokenizer.
# So, for now, I'm skipping them.

from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
import keras

model_name = "bert-base-uncased"
bert_layer = TFAutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


class TextClassifier(keras.Model):
    def __init__(self, bert_layer, num_classes):
        super(TextClassifier, self).__init__()
        self.bert = bert_layer
        self.dropout = keras.layers.Dropout(0.3)
        self.dense = keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        outputs = self.bert(inputs)
        sequence_output = outputs.last_hidden_state
        cls_token = sequence_output[:, 0, :]
        x = self.dropout(cls_token)
        return self.dense(x)


model = TextClassifier(bert_layer, num_classes=5)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ----------------------------------
# reding data
import pandas as pd
from sklearn.model_selection import train_test_split
import os


df = pd.read_csv("/kaggle/input/bbc-news/bbc-text.csv")
print(df.info())
print(df.head())

label2id = {label: idx for idx, label in enumerate(sorted(df["category"].unique()))}
df["label_id"] = df["category"].map(label2id)

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label_id"], random_state=42
)


def preprocessing(texts, labels, tokenizer, max_length=128):
    enc = tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf",
    )
    inputs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc["token_type_ids"],
    }
    return inputs, tf.convert_to_tensor(labels.tolist())


train_enc, train_labels = preprocessing(
    train_df["text"], train_df["label_id"], tokenizer
)
val_enc, val_labels = preprocessing(val_df["text"], val_df["label_id"], tokenizer)

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_enc, train_labels)).shuffle(500).batch(16)
)
val_ds = tf.data.Dataset.from_tensor_slices((val_enc, val_labels)).batch(16)


model.fit(train_ds, validation_data=val_ds, epochs=10)
