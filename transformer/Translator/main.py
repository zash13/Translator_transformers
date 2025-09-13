import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import tensorflow as tf
from TranslatorModel import Translator

MAX_SEQUENCE_LENGTH = 30
INPUT_VOCAB_SIZE = 1000
TARGET_VOCAB_SIZE = 1000
BATCH_SIZE = 64
EPOCHES = 2


def read_data(fr_file, en_file):
    with open(fr_file, "r", encoding="utf-8") as f:
        fr_data = f.readlines()
    with open(en_file, "r", encoding="utf-8") as f:
        en_data = f.readlines()
    data = pd.DataFrame(
        {
            "fr": [line.strip() for line in fr_data],
            "en": [line.strip() for line in en_data],
        }
    )
    data = data[data["fr"].str.len() > 0]
    data = data[data["en"].str.len() > 0]

    data = data[data["fr"].str.split().str.len().between(3, MAX_SEQUENCE_LENGTH)]
    data = data[data["en"].str.split().str.len().between(3, MAX_SEQUENCE_LENGTH)]

    return data


def create_tokenizer(texts):
    vocab_size = INPUT_VOCAB_SIZE
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def preprocess_data(data, src_tokenizer, tgt_tokenizer, max_length=20):
    def encode(row):
        src = src_tokenizer.encode(row["fr"]).ids
        tgt = tgt_tokenizer.encode(row["en"]).ids
        src = (
            [src_tokenizer.token_to_id("[BOS]")]
            + src
            + [src_tokenizer.token_to_id("[EOS]")]
        )
        tgt = (
            [tgt_tokenizer.token_to_id("[BOS]")]
            + tgt
            + [tgt_tokenizer.token_to_id("[EOS]")]
        )
        src = src[:max_length] + [src_tokenizer.token_to_id("[PAD]")] * (
            max_length - len(src)
        )
        tgt = tgt[:max_length] + [tgt_tokenizer.token_to_id("[PAD]")] * (
            max_length - len(tgt)
        )
        # return encoder input, decoder input (tgt[:-1]), and target labels (tgt[1:])
        return src, tgt[:-1], tgt[1:]

    encoded = data.apply(encode, axis=1, result_type="expand")
    encoded.columns = ["src", "tgt_in", "tgt_out"]

    def gen():
        for _, row in encoded.iterrows():
            yield ((row["src"], row["tgt_in"]), row["tgt_out"])

    tf_dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (
                tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
                tf.TensorSpec(shape=(max_length - 1,), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(max_length - 1,), dtype=tf.int32),
        ),
    )

    return tf_dataset


def main():
    num_layers = 2
    embedding_dim = 128
    num_heads = 4
    feed_forward_dim = 512
    input_vocab_size = INPUT_VOCAB_SIZE
    target_vocab_size = TARGET_VOCAB_SIZE
    max_sequence_length = MAX_SEQUENCE_LENGTH
    batch_size = BATCH_SIZE
    max_train_samples = 100000
    max_val_samples = 10000

    fr_file = os.path.join(
        "..",
        "..",
        "datasets",
        "EuroparlDataset",
        "ParallelCorpus_French-English",
        "europarl-v7.fr-en.fr",
    )
    en_file = os.path.join(
        "..",
        "..",
        "datasets",
        "EuroparlDataset",
        "ParallelCorpus_French-English",
        "europarl-v7.fr-en.en",
    )
    print("loading europarl data...")
    full_data = read_data(fr_file, en_file)

    train_data = full_data.sample(n=max_train_samples, random_state=42)
    val_data = full_data.drop(train_data.index).sample(
        n=max_val_samples, random_state=42
    )

    print("training tokenizers...")
    src_tokenizer = create_tokenizer(train_data["fr"])
    tgt_tokenizer = create_tokenizer(train_data["en"])

    print("preprocessing data...")
    train_dataset = preprocess_data(
        train_data, src_tokenizer, tgt_tokenizer, max_sequence_length
    )
    val_dataset = preprocess_data(
        val_data, src_tokenizer, tgt_tokenizer, max_sequence_length
    )

    train_dataset = (
        train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("initializing model...")
    model = Translator(
        num_layers,
        embedding_dim,
        num_heads,
        feed_forward_dim,
        input_vocab_size,
        target_vocab_size,
        max_sequence_length,
    )
    model.build([(None, max_sequence_length), (None, max_sequence_length - 1)])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print("Training model...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHES)

    print("Testing inference...")
    sample = next(iter(val_dataset))
    enc_inputs, dec_inputs = (
        sample[0][0],
        sample[0][1],
    )
    output = model([enc_inputs, dec_inputs], training=False)
    predicted_ids = tf.argmax(output, axis=-1)

    sample_src = src_tokenizer.decode(enc_inputs[0].numpy(), skip_special_tokens=True)
    sample_tgt = tgt_tokenizer.decode(
        predicted_ids[0].numpy(), skip_special_tokens=True
    )
    sample_true = tgt_tokenizer.decode(sample[1][0].numpy(), skip_special_tokens=True)
    print("Sample source (French):", sample_src)
    print("Sample predicted target (English):", sample_tgt)
    print("Sample true target (English):", sample_true)


if __name__ == "__main__":
    main()
