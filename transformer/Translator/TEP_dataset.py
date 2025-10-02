from numpy import full
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

MAX_SEQUENCE_LENGTH = 40
INPUT_VOCAB_SIZE = 14000
TARGET_VOCAB_SIZE = 16000
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
    full_data, input_vocab_size, target_vocab_size, lower=True
):
    src_vocab = set()
    for sentence in full_data["src"]:
        if lower:
            sentence = sentence.lower()
        src_vocab.update(sentence.split())  # Word-level
    tgt_vocab = set()
    for sentence in full_data["tgt"]:
        if lower:
            sentence = sentence.lower()
        tgt_vocab.update(sentence.split())
    src_coverage = min(input_vocab_size / len(src_vocab), 1.0) * 100
    tgt_coverage = min(target_vocab_size / len(tgt_vocab), 1.0) * 100
    print(f"Input (English) word vocab size: {len(src_vocab)}")
    print(f"Target (Persian) word vocab size: {len(tgt_vocab)}")
    print(f"Input tokenizer covers ~{src_coverage:.2f}% of words")
    print(f"Target tokenizer covers ~{tgt_coverage:.2f}% of words")
    # Real word-level size (for reference)
    return len(src_vocab), len(tgt_vocab), src_coverage, tgt_coverage


def create_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def preprocess_data(data, src_tokenizer, tgt_tokenizer, max_length=30):
    """
    Returns a tf.data.Dataset  ((src, tgt_in), tgt_out).
    src shape: (max_length,)           -> contains [BOS] ... [EOS] + PADs
    tgt_in shape: (max_length-1,)      -> sequence that the decoder consumes
    tgt_out shape: (max_length-1,)     -> expected decoder outputs (shifted)
    """
    pad_id_src = src_tokenizer.token_to_id("[PAD]")
    pad_id_tgt = tgt_tokenizer.token_to_id("[PAD]")
    bos_id_src = src_tokenizer.token_to_id("[BOS]")
    eos_id_src = src_tokenizer.token_to_id("[EOS]")
    bos_id_tgt = src_tokenizer.token_to_id("[BOS]")
    eos_id_tgt = src_tokenizer.token_to_id("[EOS]")
    encoded_data = []


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


MOCK_SENTENCES = [
    {"english": "Hello, how are you?", "persian": "سلام، حالت چطوره؟"},
    {"english": "I am fine, thank you.", "persian": "من خوبم، ممنون."},
    {"english": "What is your name?", "persian": "اسمت چیه؟"},
    {"english": "I live in Tehran.", "persian": "من در تهران زندگی می‌کنم."},
    {"english": "The weather is nice today.", "persian": "هوا امروز خوبه."},
    {"english": "I like reading books.", "persian": "من کتاب خوندن رو دوست دارم."},
    {"english": "See you later.", "persian": "بعداً می‌بینمت."},
    {"english": "Good morning!", "persian": "صبح بخیر!"},
    {"english": "I need help.", "persian": "من به کمک نیاز دارم."},
    {
        "english": "Goodbye, have a good day.",
        "persian": "خداحافظ، روز خوبی داشته باشی.",
    },
]


def test_mock_predictions(
    model, src_tokenizer, tgt_tokenizer, mock_sentences=MOCK_SENTENCES
):
    """
    Tests model predictions on a hardcoded list of mock English sentences.
    Prints a table: English Input | Expected Persian | Model Prediction
    """
    results = []
    bos_id = tgt_tokenizer.token_to_id("[BOS]")
    eos_id = tgt_tokenizer.token_to_id("[EOS]")
    pad_id = tgt_tokenizer.token_to_id("[PAD]")

    for sentence in mock_sentences:
        english_text = sentence["english"]
        expected_persian = sentence["persian"]

        # Tokenize English for encoder input (add [BOS]/[EOS], pad to max_length)
        enc_tokens = src_tokenizer.encode(english_text).ids
        enc_input = (
            [src_tokenizer.token_to_id("[BOS]")]
            + enc_tokens
            + [src_tokenizer.token_to_id("[EOS]")]
        )
        enc_input = enc_input[:MAX_SEQUENCE_LENGTH] + [
            src_tokenizer.token_to_id("[PAD]")
        ] * (MAX_SEQUENCE_LENGTH - len(enc_input))
        enc_input = tf.constant([enc_input])  # Batch of 1

        # Predict Persian tokens
        pred_tokens = model.predict(enc_input, MAX_SEQUENCE_LENGTH, bos_id, eos_id)

        # Decode prediction (remove special tokens)
        pred_ids = pred_tokens[0].numpy()  # First batch item
        pred_text = tgt_tokenizer.decode(
            [id for id in pred_ids if id not in [bos_id, eos_id, pad_id]]
        )

        results.append(
            {
                "English Input": english_text,
                "Expected Persian": expected_persian,
                "Model Prediction": pred_text,
            }
        )

    # Beautiful table print
    results_df = pd.DataFrame(results)
    print("\n=== Mock Sentence Predictions ===")
    print(
        results_df.to_string(index=False, max_colwidth=40)
    )  # Truncate for readability


def main():
    num_layers = 2
    embedding_dim = 128
    num_heads = 4
    feed_forward_dim = 512
    input_vocab_size = INPUT_VOCAB_SIZE
    target_vocab_size = TARGET_VOCAB_SIZE
    max_sequence_length = MAX_SEQUENCE_LENGTH
    batch_size = BATCH_SIZE
    # you can ether set this variable , or let it be maximum of waht we have in dataset
    max_train_samples = 10000
    val_samples_ratio = 0.1

    src_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.en")
    tgt_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.fa")

    print("loading TEP data...")
    full_data = read_data(src_file, tgt_file)
    total_sampels = len(full_data)
    max_train_samples = int(
        total_sampels * (1 - val_samples_ratio)
        if max_train_samples == 0
        else max_train_samples
    )
    max_val_samples = int(max_train_samples * val_samples_ratio)

    train_data = full_data.sample(n=max_train_samples, random_state=42)
    val_data = full_data.drop(train_data.index).sample(
        n=max_val_samples, random_state=42
    )
    print(
        f"total samples : {total_sampels} train samples :{max_train_samples} validation sampels :{max_val_samples} validation ratio: {val_samples_ratio} "
    )

    print("training tokenizers...")
    src_tokenizer = create_tokenizer(train_data["src"], INPUT_VOCAB_SIZE)  # persian
    tgt_tokenizer = create_tokenizer(train_data["tgt"], TARGET_VOCAB_SIZE)  # english

    print("information about vocab : ")
    calculate_vocab_coverage(full_data, input_vocab_size, target_vocab_size)
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
    test_mock_predictions(model, src_tokenizer, tgt_tokenizer)
    print("Evaluating BLEU score...")
    bleu_score = evaluate_bleu(model, val_dataset, src_tokenizer, tgt_tokenizer)
    print(f"Validation BLEU: {bleu_score:.2f}")

    model.save_weights("translator_model_architecture.weights.h5")


def interactive_translate(model, src_tokenizer, tgt_tokenizer):
    """
    Interactive mode: User inputs English sentence, model translates to Persian.
    Type 'quit' to exit.
    """
    bos_id = tgt_tokenizer.token_to_id("[BOS]")
    eos_id = tgt_tokenizer.token_to_id("[EOS]")
    pad_id = tgt_tokenizer.token_to_id("[PAD]")

    print("\n=== Interactive Translation Mode ===")
    print("Enter an English sentence (or 'quit' to exit):")

    while True:
        english_text = input("> ").strip()
        if english_text.lower() == "quit":
            print("Exiting interactive mode.")
            break

        if not english_text:
            print("Please enter a valid sentence.")
            continue

        # Tokenize English for encoder input (add [BOS]/[EOS], pad)
        enc_tokens = src_tokenizer.encode(english_text).ids
        enc_input = (
            [src_tokenizer.token_to_id("[BOS]")]
            + enc_tokens
            + [src_tokenizer.token_to_id("[EOS]")]
        )
        enc_input = enc_input[:MAX_SEQUENCE_LENGTH] + [
            src_tokenizer.token_to_id("[PAD]")
        ] * (MAX_SEQUENCE_LENGTH - len(enc_input))
        enc_input = tf.constant([enc_input])  # Batch of 1

        # Predict Persian tokens
        pred_tokens = model.predict(enc_input, MAX_SEQUENCE_LENGTH, bos_id, eos_id)

        # Decode prediction (remove special tokens)
        pred_ids = pred_tokens[0].numpy()  # First batch item
        pred_text = tgt_tokenizer.decode(
            [id for id in pred_ids if id not in [bos_id, eos_id, pad_id]]
        )

        print(f"Translation: {pred_text}\n")


if __name__ == "__main__":
    main()
