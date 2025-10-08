import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace


SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[MASK]"]


def create_persian_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,  # helps avoid noisy rare fragments
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def main2_persian_test():
    # Path to Persian file (only the .fa part of TEP)
    tgt_file = os.path.join("..", "..", "datasets", "TEP", "TEP.en-fa.fa")

    print("Loading Persian TEP data...")
    with open(tgt_file, "r", encoding="utf-8") as f:
        persian_sentences = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(persian_sentences)} Persian sentences.")

    # Limit for testing (you can increase later)
    sample_size = 200000  # adjust if needed
    persian_sentences = persian_sentences[:sample_size]

    vocab_size = 140_000
    print(f"Training tokenizer with vocab size = {vocab_size} ...")

    tokenizer = create_persian_tokenizer(persian_sentences, vocab_size)

    # Print first 100 tokens in vocab
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    first_100 = [token for token, idx in sorted_vocab[:1000]]
    print("\nFirst 100 tokens in vocabulary:")
    print(first_100)

    # Test on a real Persian sentence
    test_sentence = "من دوست دارم برنامه نویسی کنم"
    encoded = tokenizer.encode(test_sentence)
    print("\nTest sentence:", test_sentence)
    print("Tokens:", encoded.tokens)
    print("Decoded:", tokenizer.decode(encoded.ids))


if __name__ == "__main__":
    main2_persian_test()
