import tensorflow as tf
import keras

import os
import pandas as pd
from tokenizers import Tokenizer
import tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from TranslatorModel import Translator
import sacrebleu
from tqdm import tqdm


gpus = tf.config.list_physical_devices("GPU")
print("Num GPUs available:", len(gpus))

MAX_SEQUENCE_LENGTH = 30
INPUT_VOCAB_SIZE = 100000
TARGET_VOCAB_SIZE = 60000
BATCH_SIZE = 64
EPOCHS = 1
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]


def read_data(src_file, tgt_file):
    with open(src_file, "r", encoding="utf-8") as f:
        src_data = f.readlines()
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_data = f.readlines()
    data = pd.DataFrame(
        {
            "src": [line.strip() for line in src_data],
            "tgt": [line.strip() for line in tgt_data],
        }
    )
    data = data[data["src"].str.len() > 0]
    data = data[data["tgt"].str.len() > 0]
    return data


def calculate_vocab_coverage(
    full_data, input_vocab_size, target_vocab_size, src_tokenizer, tgt_tokenizer
):
    src_vocab = set(src_tokenizer.get_vocab().keys())
    tgt_vocab = set(tgt_tokenizer.get_vocab().keys())
    src_coverage = min(input_vocab_size / len(src_vocab), 1.0) * 100
    tgt_coverage = min(target_vocab_size / len(tgt_vocab), 1.0) * 100
    print(f"Input vocab size in tokenizer: {len(src_vocab)}")
    print(f"Target vocab size in tokenizer: {len(tgt_vocab)}")
    print(f"Input tokenizer covers ~{src_coverage:.2f}% of vocab")
    print(f"Target tokenizer covers ~{tgt_coverage:.2f}% of vocab")
    return len(src_vocab), len(tgt_vocab), src_coverage, tgt_coverage


def create_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[PAD]"),
        pad_token="[PAD]",
        length=MAX_SEQUENCE_LENGTH,
    )
    return tokenizer


def preprocess_data(data, src_tokenizer, tgt_tokenizer, max_length=MAX_SEQUENCE_LENGTH):
    def encode(row):
        src_enc = src_tokenizer.encode(row["src"])
        tgt_enc = tgt_tokenizer.encode(row["tgt"])
        if len(src_enc.tokens) > max_length or len(tgt_enc.tokens) > max_length:
            print(
                f"Skipping: src_len={len(src_enc.tokens)}, tgt_len={len(tgt_enc.tokens)}"
            )
            return None
        src = src_enc.ids[:max_length]
        tgt = tgt_enc.ids[:max_length]
        src[0] = src_tokenizer.token_to_id("[BOS]")
        src[-1] = src_tokenizer.token_to_id("[EOS]")
        tgt[0] = tgt_tokenizer.token_to_id("[BOS]")
        tgt[-1] = tgt_tokenizer.token_to_id("[EOS]")
        if len(src) != max_length or len(tgt) != max_length:
            print(f"Invalid length: src={len(src)}, tgt={len(tgt)}")
            return None
        if src[0] != src_tokenizer.token_to_id("[BOS]") or src[
            -1
        ] != src_tokenizer.token_to_id("[EOS]"):
            print(f"Invalid src tokens: {src}")
            return None
        if tgt[0] != tgt_tokenizer.token_to_id("[BOS]") or tgt[
            -1
        ] != tgt_tokenizer.token_to_id("[EOS]"):
            print(f"Invalid tgt tokens: {tgt}")
            return None
        return src, tgt[:-1], tgt[1:]

    encoded_data = []
    for _, row in data.iterrows():
        result = encode(row)
        if result is not None:
            encoded_data.append(result)

    if not encoded_data:
        raise ValueError(
            "No valid samples after preprocessing. Increase max_length or check data."
        )
    encoded = pd.DataFrame(encoded_data, columns=["src", "tgt_in", "tgt_out"])
    print(f"Preprocessed {len(encoded)} samples")

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


def evaluate_bleu(model, dataset, src_tokenizer, tgt_tokenizer, max_length=30):
    refs, hyps = [], []
    for (src, _), tgt_out in dataset:
        pred = model.predict(
            src,
            max_length,
            start_token=tgt_tokenizer.token_to_id("[BOS]"),
            end_token=tgt_tokenizer.token_to_id("[EOS]"),
        )
        for p, t in zip(pred.numpy(), tgt_out.numpy()):
            p = [id for id in p if id != tgt_tokenizer.token_to_id("[PAD]")]
            t = [id for id in t if id != tgt_tokenizer.token_to_id("[PAD]")]
            hyp = tgt_tokenizer.decode(p[1:])  # Skip [BOS]
            ref = tgt_tokenizer.decode(t)
            hyps.append(hyp)
            refs.append([ref])  # sacrebleu expects list of refs
    bleu = sacrebleu.corpus_bleu(hyps, refs)
    return bleu.score


def main():
    num_layers = 2
    embedding_dim = 128
    num_heads = 4
    feed_forward_dim = 512
    input_vocab_size = INPUT_VOCAB_SIZE
    target_vocab_size = TARGET_VOCAB_SIZE
    max_sequence_length = MAX_SEQUENCE_LENGTH
    batch_size = BATCH_SIZE
    max_train_samples = 1000
    max_val_samples = 10

    src_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.en")
    tgt_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.fa")

    print("loading TEP data...")
    full_data = read_data(src_file, tgt_file)

    # proceed with train/val split on the reduced data
    train_data = full_data.sample(n=max_train_samples, random_state=42)
    val_data = full_data.drop(train_data.index).sample(
        n=max_val_samples, random_state=42
    )

    print("training tokenizers...")
    src_tokenizer = create_tokenizer(train_data["src"], INPUT_VOCAB_SIZE)  # persian
    tgt_tokenizer = create_tokenizer(train_data["tgt"], TARGET_VOCAB_SIZE)  # english

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
        pad_id=tgt_tokenizer.token_to_id("[PAD]"),
    )
    model.build([(None, max_sequence_length), (None, max_sequence_length - 1)])

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print("Training model...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
    print("Evaluating BLEU score...")
    bleu_score = evaluate_bleu(model, val_dataset, src_tokenizer, tgt_tokenizer)
    print(f"Validation BLEU: {bleu_score:.2f}")

    model.save_weights("translator_model_architecture.weights.h5")


if __name__ == "__main__":
    main()
